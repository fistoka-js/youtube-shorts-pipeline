"""Tests for failed-topic handling in run.py: the attempts count, queue/failed/,
malformed topic files, exit codes, and the picker.

All data is made up. Each test works in a throwaway folder under pytest's tmp_path:
a bare git repo called origin, and a clone called work (the working directory, in
CI mode so the claim path is real). The draft, produce and YouTube upload steps are
fakes, so nothing touches the network, YouTube or any paid service, and the real
queue/, drafts/, media/, logs/ and .verticals/ folders are never used.

Run with:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_queue_failures.py -v
"""

import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

import run
from run import ERROR_MAX_CHARS, MAX_ATTEMPTS
from verticals import upload as upload_module

PENDING = Path("queue/pending")
IN_PROGRESS = Path("queue/in-progress")
DONE = Path("queue/done")
FAILED = Path("queue/failed")
REVIEW = Path("queue/review")
TOPIC = "test1.json"
GOOD_TOPIC = {"topic": "Made-up topic", "niche": "science"}


def _git(cwd, *args):
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _configure(repo):
    _git(repo, "config", "user.name", "Made Up Tester")
    _git(repo, "config", "user.email", "tester@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """A work clone (the working directory) with empty queue folders, in CI mode on main."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", "-q", "--initial-branch=main", str(origin))
    _git(tmp_path, "init", "-q", "--initial-branch=main", str(seed))
    _configure(seed)
    for folder in ("queue/pending", "queue/done", "queue/in-progress", "queue/failed", "queue/review"):
        (seed / folder).mkdir(parents=True)
        (seed / folder / ".gitkeep").write_text("")
    (seed / "queue/variation_history.json").write_text("[]\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "Seed queue")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-q", "origin", "main")
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    _configure(work)
    monkeypatch.chdir(work)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    return work


class Pipeline:
    """Fake draft, produce and upload steps. draft_error is raised by the draft step (for
    every topic, or only for the names in fail_names). upload_error is raised by the upload
    step, after the claim in CI mode."""

    def __init__(self, draft_error=None, fail_names=None, upload_error=None):
        self.draft_error = draft_error
        self.fail_names = fail_names
        self.upload_error = upload_error
        self.cmd_run_calls = []
        self.upload_calls = 0

    def cmd_run(self, args):
        self.cmd_run_calls.append(args.review_name)
        if self.draft_error is not None and (
                self.fail_names is None or args.review_name in self.fail_names):
            raise self.draft_error
        upload_module.upload_to_youtube(Path("fake_video.mp4"), {}, None, "en", None)

    def upload(self, *args, **kwargs):
        self.upload_calls += 1
        if self.upload_error is not None:
            raise self.upload_error
        return "https://youtu.be/FAKEVIDEO01"


def run_main(pipeline):
    """Runs run.main() with the fakes in place. Returns (exit code, printed text)."""
    out = io.StringIO()
    with mock.patch("verticals.__main__.cmd_run", pipeline.cmd_run), \
            mock.patch.object(upload_module, "upload_to_youtube", pipeline.upload), \
            redirect_stdout(out):
        try:
            run.main()
            code = 0
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def topic_bytes(data):
    """The file format run.py writes: one line of JSON, UTF-8, LF, trailing newline."""
    return (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")


def write_topic(path: Path, data=None, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if raw is not None else topic_bytes(data))
    return path


def load(path: Path):
    return json.loads(path.read_bytes().decode("utf-8"))


# Failure before the claim: one attempt is recorded, the topic stays in pending/.

def test_first_failure_keeps_topic_in_pending_with_one_attempt(queue):
    original = {"topic": "Made-up topic é", "niche": "science", "platform": "shorts"}
    write_topic(queue / PENDING / TOPIC, original)

    code, out = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    raw = (queue / PENDING / TOPIC).read_bytes()
    assert load(queue / PENDING / TOPIC) == {**original, "attempts": 1}
    assert "error" not in load(queue / PENDING / TOPIC)
    assert raw.count(b"\n") == 1
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert "é".encode("utf-8") in raw  # written as UTF-8, not escaped
    assert not (queue / FAILED / TOPIC).exists()
    assert f"Attempt 1 of {MAX_ATTEMPTS} failed" in out
    assert "left in queue/pending/ (nothing was uploaded)" in out


# The last allowed failure: the topic moves to queue/failed/ with an error.

def test_last_failure_moves_topic_to_failed_with_short_error(queue):
    long_message = "First line of the made-up error. " + "x" * 600 + "\nSecond line that must not be saved."
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        code, out = run_main(Pipeline(draft_error=RuntimeError(long_message)))
        assert code == 1
        if attempt < MAX_ATTEMPTS:
            assert (queue / PENDING / TOPIC).exists()
            assert not (queue / FAILED / TOPIC).exists()

    assert not (queue / PENDING / TOPIC).exists()
    raw = (queue / FAILED / TOPIC).read_bytes()
    saved = load(queue / FAILED / TOPIC)
    assert saved["attempts"] == MAX_ATTEMPTS
    assert saved["topic"] == GOOD_TOPIC["topic"]
    assert saved["niche"] == GOOD_TOPIC["niche"]
    assert saved["error"] == long_message.splitlines()[0][:ERROR_MAX_CHARS]
    assert len(saved["error"]) == ERROR_MAX_CHARS
    assert "Second line" not in saved["error"]
    assert raw.count(b"\n") == 1
    assert raw.endswith(b"\n")
    assert f"Topic moved to queue/failed/ after {MAX_ATTEMPTS} failed attempts:" in out


@pytest.mark.parametrize("message", ["", "   \n\t\n"])
def test_empty_error_message_is_saved_as_the_exception_class_name(queue, message):
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)

    for _ in range(MAX_ATTEMPTS):
        run_main(Pipeline(draft_error=RuntimeError(message)))

    assert load(queue / FAILED / TOPIC)["error"] == "RuntimeError"


def test_existing_attempt_count_is_added_to(queue):
    write_topic(queue / PENDING / TOPIC, {**GOOD_TOPIC, "attempts": MAX_ATTEMPTS - 1})

    code, _ = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert load(queue / FAILED / TOPIC)["attempts"] == MAX_ATTEMPTS
    assert not (queue / PENDING / TOPIC).exists()


# The next topic is picked once the bad one has used up its attempts.

def test_bad_topic_does_not_block_the_next_one(queue):
    write_topic(queue / PENDING / "test2.json", {"topic": "Made-up topic 2"})
    write_topic(queue / PENDING / "test3.json", {"topic": "Made-up topic 3"})

    for _ in range(MAX_ATTEMPTS):
        code, _ = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure"),
                                    fail_names={"test2"}))
        assert code == 1
    assert (queue / FAILED / "test2.json").exists()

    fake = Pipeline(draft_error=RuntimeError("made-up draft failure"), fail_names={"test2"})
    code, _ = run_main(fake)

    assert code == 0
    assert fake.cmd_run_calls == ["test3"]
    assert (queue / DONE / "test3.json").exists()
    assert not (queue / PENDING / "test3.json").exists()
    assert not (queue / PENDING / "test2.json").exists()
    assert (queue / FAILED / "test2.json").exists()


# Success is unchanged: the file moves to done/ and gets no attempts field.

def test_success_moves_to_done_with_identical_bytes_and_no_attempts(queue):
    raw = topic_bytes(GOOD_TOPIC)
    (queue / PENDING / TOPIC).write_bytes(raw)

    code, _ = run_main(Pipeline())

    assert code == 0
    assert (queue / DONE / TOPIC).read_bytes() == raw
    assert "attempts" not in load(queue / DONE / TOPIC)


# A failure after the claim is unchanged: the topic stays in in-progress/, no attempts.

def test_failure_after_claim_is_unchanged(queue):
    raw = topic_bytes(GOOD_TOPIC)
    (queue / PENDING / TOPIC).write_bytes(raw)
    pipeline = Pipeline(upload_error=RuntimeError("made-up network error before insert"))

    code, out = run_main(pipeline)

    assert code == 1
    assert pipeline.upload_calls == 1
    assert (queue / IN_PROGRESS / TOPIC).read_bytes() == raw
    assert not (queue / PENDING / TOPIC).exists()
    assert not (queue / FAILED / TOPIC).exists()
    assert "left in queue/in-progress/" in out
    assert "attempt" not in out.lower()


# Malformed topic files go to queue/failed/ at once, and their text is kept (undecodable
# bytes become the replacement character, so the original bytes are not kept).

MALFORMED_CASES = [
    pytest.param(b'{"topic": "Made-up topic", "niche"', False, id="invalid-json"),
    pytest.param(b'["Made-up topic"]\n', False, id="list-not-object"),
    pytest.param(b'\xff\xfe\x00{"topic": "Made-up topic"}', False, id="undecodable-bytes"),
    pytest.param(b'{"niche": "science"}\n', True, id="missing-topic"),
    pytest.param(b'{"topic": "   ", "niche": "science"}\n', True, id="blank-topic"),
    pytest.param(b'{"topic": 42, "niche": "science"}\n', True, id="non-string-topic"),
]


@pytest.mark.parametrize("raw, is_object", MALFORMED_CASES)
def test_malformed_topic_goes_to_failed_at_once_without_crashing(queue, raw, is_object):
    write_topic(queue / PENDING / TOPIC, raw=raw)
    pipeline = Pipeline()

    code, out = run_main(pipeline)

    assert code == 1  # a clean SystemExit(1), not an uncaught exception
    assert pipeline.cmd_run_calls == []
    assert pipeline.upload_calls == 0
    assert not (queue / PENDING / TOPIC).exists()
    saved = load(queue / FAILED / TOPIC)
    assert isinstance(saved["error"], str) and saved["error"]
    assert len(saved["error"]) <= ERROR_MAX_CHARS
    assert "\n" not in saved["error"]
    assert "attempts" not in saved  # malformed files get no second chance and no count
    if is_object:
        # A JSON object keeps every field and gets the error added.
        assert {k: v for k, v in saved.items() if k != "error"} == json.loads(raw)
    else:
        # Anything else is wrapped so the file's text is kept.
        assert set(saved) == {"error", "original_text"}
        assert saved["original_text"] == raw.decode("utf-8", errors="replace")
    assert f"ERROR: {TOPIC} is malformed" in out
    assert "moved to queue/failed/" in out


# Exit codes: the run still ends non-zero, and a non-clean SystemExit keeps its code.

def test_generic_failure_exits_non_zero_on_every_attempt(queue):
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)

    for _ in range(MAX_ATTEMPTS):
        code, _ = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))
        assert code == 1


@pytest.mark.parametrize("exit_code, expected_error", [
    pytest.param(3, "the pipeline stopped with exit code 3", id="exit-code"),
    pytest.param("made-up message", "the pipeline stopped with message: made-up message",
                 id="message"),
])
def test_non_clean_exit_is_recorded_and_keeps_its_exit_code(queue, exit_code, expected_error):
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)

    for _ in range(MAX_ATTEMPTS):
        code, _ = run_main(Pipeline(draft_error=SystemExit(exit_code)))
        assert code == exit_code

    saved = load(queue / FAILED / TOPIC)
    assert saved["attempts"] == MAX_ATTEMPTS
    assert saved["error"] == expected_error[:ERROR_MAX_CHARS]


# The picker only looks at queue/pending/*.json.

def test_picker_ignores_failed_in_progress_review_and_gitkeep(queue):
    write_topic(queue / FAILED / "test1.json", {"topic": "Made-up failed"})
    write_topic(queue / IN_PROGRESS / "test2.json", {"topic": "Made-up in progress"})
    write_topic(queue / REVIEW / "test3.json", {"topic": "Made-up review"})
    write_topic(queue / DONE / "test4.json", {"topic": "Made-up done"})
    assert run.pick_next_topic_file() is None

    write_topic(queue / PENDING / "test5.json", {"topic": "Made-up pending"})
    assert run.pick_next_topic_file() == PENDING / "test5.json"


def test_run_processes_only_the_pending_topic(queue):
    failed_raw = topic_bytes({"topic": "Made-up failed"})
    in_progress_raw = topic_bytes({"topic": "Made-up in progress"})
    (queue / FAILED / "test1.json").write_bytes(failed_raw)
    (queue / IN_PROGRESS / "test2.json").write_bytes(in_progress_raw)
    write_topic(queue / PENDING / "test5.json", GOOD_TOPIC)

    fake = Pipeline()
    code, _ = run_main(fake)

    assert code == 0
    assert fake.cmd_run_calls == ["test5"]
    assert (queue / DONE / "test5.json").exists()
    assert (queue / FAILED / "test1.json").read_bytes() == failed_raw
    assert (queue / IN_PROGRESS / "test2.json").read_bytes() == in_progress_raw


# Errors that are not counted: ClaimError, ClaimUndoError and clean exits.

def test_claim_error_is_not_counted(queue):
    raw = topic_bytes(GOOD_TOPIC)
    (queue / PENDING / TOPIC).write_bytes(raw)
    pipeline = Pipeline()

    with mock.patch.object(run, "claim_topic", side_effect=run.ClaimError("made-up claim error")):
        code, out = run_main(pipeline)

    assert code == 1
    assert pipeline.upload_calls == 0
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert not (queue / FAILED / TOPIC).exists()
    assert "could not save the claim" in out


def test_claim_undo_error_is_not_counted(queue):
    raw = topic_bytes(GOOD_TOPIC)
    (queue / PENDING / TOPIC).write_bytes(raw)
    pipeline = Pipeline()

    with mock.patch.object(run, "claim_topic",
                           side_effect=run.ClaimUndoError("made-up undo error")):
        code, out = run_main(pipeline)

    assert code == 1
    assert pipeline.upload_calls == 0
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert not (queue / FAILED / TOPIC).exists()
    assert "could not undo the claim" in out


@pytest.mark.parametrize("exit_value", [0, None])
def test_clean_exit_is_not_recorded(queue, exit_value):
    raw = topic_bytes(GOOD_TOPIC)
    (queue / PENDING / TOPIC).write_bytes(raw)

    code, out = run_main(Pipeline(draft_error=SystemExit(exit_value)))

    assert code == exit_value
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert not (queue / FAILED / TOPIC).exists()
    assert "ERROR" not in out
    assert "left in queue" not in out


# Attempts values that are not whole numbers of 0 or more count as 0.

@pytest.mark.parametrize("value", ["1", -1, True, 1.0, None])
def test_bad_attempts_values_count_as_zero(queue, value):
    write_topic(queue / PENDING / TOPIC, {**GOOD_TOPIC, "attempts": value})

    run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert load(queue / PENDING / TOPIC)["attempts"] == 1


# If recording fails, the original error is still reported and the exit is non-zero.

@pytest.mark.parametrize("draft_error, expected_code, expected_text", [
    pytest.param(RuntimeError("made-up original error"), 1,
                 "made-up original error", id="generic-error"),
    pytest.param(SystemExit(3), 3, "the pipeline stopped with exit code 3", id="system-exit"),
])
def test_failed_recording_keeps_original_error_and_non_zero_exit(
        queue, monkeypatch, draft_error, expected_code, expected_text):
    raw = topic_bytes(GOOD_TOPIC)
    (queue / PENDING / TOPIC).write_bytes(raw)
    monkeypatch.setattr(run.os, "replace", mock.Mock(side_effect=OSError("made-up replace failure")))

    code, out = run_main(Pipeline(draft_error=draft_error))

    assert code == expected_code
    assert code != 0
    assert expected_text in out
    assert "WARNING: could not record the failed attempt" in out
    assert "made-up replace failure" in out
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert sorted(p.name for p in (queue / PENDING).iterdir()) == [".gitkeep", TOPIC]
    assert sorted(p.name for p in (queue / FAILED).iterdir()) == [".gitkeep"]


# A local run (no GITHUB_ACTIONS) where the upload started and then failed is counted.

def test_local_upload_failure_is_counted_with_the_local_message(queue, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS")
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)
    pipeline = Pipeline(upload_error=RuntimeError("made-up error after upload started"))

    code, out = run_main(pipeline)

    assert code == 1
    assert pipeline.upload_calls == 1
    assert load(queue / PENDING / TOPIC)["attempts"] == 1
    assert f"Topic left in queue/pending/. Attempt 1 of {MAX_ATTEMPTS} failed." in out
    assert "not certain whether a video went live" in out
    assert "nothing was uploaded" not in out
    assert not (queue / IN_PROGRESS / TOPIC).exists()


# Numbers, surrogates and unreadable files never crash the run.

def test_number_with_too_many_digits_is_malformed_not_a_crash(queue):
    raw = b'{"topic": "Made-up topic", "n": ' + b"9" * 5000 + b"}\n"
    write_topic(queue / PENDING / TOPIC, raw=raw)
    pipeline = Pipeline()

    code, out = run_main(pipeline)

    assert code == 1
    assert pipeline.cmd_run_calls == []
    assert not (queue / PENDING / TOPIC).exists()
    saved = load(queue / FAILED / TOPIC)
    assert set(saved) == {"error", "original_text"}
    assert saved["original_text"] == raw.decode("utf-8")
    assert saved["error"]
    assert "moved to queue/failed/" in out


def test_lone_surrogate_in_topic_file_is_still_counted(queue):
    # The file holds the escape text \ud800, which parses to a lone surrogate.
    write_topic(queue / PENDING / TOPIC,
                raw=b'{"topic": "Made-up topic", "note": "bad \\ud800 char"}\n')

    code, out = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert load(queue / PENDING / TOPIC) == {
        "topic": "Made-up topic", "note": "bad \ud800 char", "attempts": 1}
    assert f"Attempt 1 of {MAX_ATTEMPTS} failed" in out


def test_lone_surrogate_in_error_text_is_saved_escaped(queue):
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)
    for _ in range(MAX_ATTEMPTS):
        code, out = run_main(Pipeline(draft_error=RuntimeError("made-up \ud800 failure")))
        assert code == 1

    saved = load(queue / FAILED / TOPIC)
    assert saved["error"] == "made-up \\ud800 failure"
    assert "failed attempts: made-up" in out


def test_lone_surrogate_in_malformed_topic_is_moved_not_repeated(queue):
    write_topic(queue / PENDING / TOPIC, raw=b'{"niche": "bad \\ud800 char"}\n')

    code, out = run_main(Pipeline())

    assert code == 1
    assert not (queue / PENDING / TOPIC).exists()
    saved = load(queue / FAILED / TOPIC)
    assert saved["niche"] == "bad \ud800 char"
    assert saved["error"]


def test_unreadable_topic_file_is_moved_unchanged_without_a_crash(queue):
    raw = topic_bytes(GOOD_TOPIC)
    write_topic(queue / PENDING / TOPIC, raw=raw)
    pipeline = Pipeline()

    with mock.patch.object(run, "_read_topic_bytes",
                           side_effect=PermissionError("made-up permission error")):
        code, out = run_main(pipeline)

    assert code == 1
    assert pipeline.cmd_run_calls == []
    assert not (queue / PENDING / TOPIC).exists()
    assert (queue / FAILED / TOPIC).read_bytes() == raw
    assert "made-up permission error" in out
    assert "moved unchanged" in out


def test_unreadable_file_that_cannot_be_moved_exits_non_zero_and_stays(queue):
    raw = topic_bytes(GOOD_TOPIC)
    write_topic(queue / PENDING / TOPIC, raw=raw)

    with mock.patch.object(run, "_read_topic_bytes",
                           side_effect=PermissionError("made-up permission error")), \
            mock.patch.object(run.shutil, "move", side_effect=OSError("made-up move failure")):
        code, out = run_main(Pipeline())

    assert code == 1
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert "could not be moved" in out
    assert "made-up move failure" in out


def test_directory_named_like_a_topic_is_never_picked(queue):
    (queue / PENDING / "x.json").mkdir(parents=True)
    assert run.pick_next_topic_file() is None

    write_topic(queue / PENDING / "test5.json", GOOD_TOPIC)
    assert run.pick_next_topic_file() == PENDING / "test5.json"

    fake = Pipeline()
    code, _ = run_main(fake)
    assert code == 0
    assert fake.cmd_run_calls == ["test5"]
    assert (queue / PENDING / "x.json").is_dir()


# An existing error field is kept, and an existing failed file is never overwritten.

def test_existing_error_field_is_kept_on_a_first_failure(queue):
    write_topic(queue / PENDING / TOPIC, {**GOOD_TOPIC, "error": "made-up old note"})

    run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert load(queue / PENDING / TOPIC) == {
        **GOOD_TOPIC, "error": "made-up old note", "attempts": 1}


def test_existing_failed_file_is_never_overwritten(queue):
    old = topic_bytes({"topic": "Made-up old record", "error": "made-up old error"})
    (queue / FAILED / TOPIC).write_bytes(old)
    (queue / FAILED / "test1-2.json").write_bytes(topic_bytes({"topic": "Made-up second"}))
    write_topic(queue / PENDING / TOPIC, {**GOOD_TOPIC, "attempts": MAX_ATTEMPTS - 1})

    code, _ = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert (queue / FAILED / TOPIC).read_bytes() == old
    assert load(queue / FAILED / "test1-3.json")["attempts"] == MAX_ATTEMPTS
    assert not (queue / PENDING / TOPIC).exists()


def test_malformed_topic_does_not_overwrite_an_existing_failed_file(queue):
    old = topic_bytes({"topic": "Made-up old record"})
    (queue / FAILED / TOPIC).write_bytes(old)
    write_topic(queue / PENDING / TOPIC, raw=b'["Made-up topic"]\n')

    code, _ = run_main(Pipeline())

    assert code == 1
    assert (queue / FAILED / TOPIC).read_bytes() == old
    saved = load(queue / FAILED / "test1-2.json")
    assert set(saved) == {"error", "original_text"}


def test_moved_message_has_no_doubled_period(queue):
    write_topic(queue / PENDING / TOPIC, GOOD_TOPIC)
    for _ in range(MAX_ATTEMPTS):
        code, out = run_main(Pipeline(draft_error=RuntimeError("made-up failure.")))
        assert code == 1

    assert "failed attempts: made-up failure.\n" in out
    assert "failure.." not in out


# Long names: the temp file name never depends on the topic's name, and the names written
# to queue/failed/ are capped at MAX_FILE_NAME_BYTES (UTF-8 bytes, not characters).

def test_temp_file_name_does_not_depend_on_topic_name_length(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real_mkstemp = run.tempfile.mkstemp
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return real_mkstemp(**kwargs)

    long_name = "c" * 190 + ".json"
    with mock.patch.object(run.tempfile, "mkstemp", side_effect=spy):
        run._write_json_atomic(Path("queue/pending") / long_name, {"topic": "Made-up topic"})

    assert calls[0]["prefix"] == ".queue-"
    assert calls[0]["suffix"] == ".tmp"
    assert (Path("queue/pending") / long_name).exists()


def test_long_failed_name_is_shortened_and_keeps_counter_and_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failed = Path("queue/failed")
    failed.mkdir(parents=True)
    stem = "d" * 240
    for number in (0, 2, 3):  # the plain name, -2 and -3 are all taken
        (failed / run._failed_name(stem, ".json", number)).write_text("x")

    chosen = run._free_failed_path(stem + ".json")

    assert len(chosen.name.encode("utf-8")) <= run.MAX_FILE_NAME_BYTES
    assert chosen.name.endswith("-4.json")
    assert chosen.name.startswith("d" * 100)
    assert not chosen.exists()


def test_very_long_ascii_topic_name_is_recorded(queue):
    # 245 characters: longer than the cap. Skipped if this machine refuses to create the name.
    name = "e" * 240 + ".json"
    path = queue / PENDING / name
    try:
        write_topic(path, {**GOOD_TOPIC, "attempts": MAX_ATTEMPTS - 1})
    except OSError:
        pytest.skip("this machine cannot create a file name of this length")

    code, _ = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert not path.exists()
    moved = [p.name for p in (queue / FAILED).iterdir() if p.name != ".gitkeep"]
    assert len(moved) == 1
    assert len(moved[0].encode("utf-8")) <= run.MAX_FILE_NAME_BYTES
    assert moved[0].endswith(".json")
    assert load(queue / FAILED / moved[0])["error"] == "made-up draft failure"


def test_multibyte_topic_name_is_recorded_and_shortened(queue):
    # 70 Japanese characters (3 bytes each) plus .json: 215 bytes, longer than the cap.
    # Skipped if this machine refuses to create the name.
    name = "日" * 70 + ".json"
    path = queue / PENDING / name
    try:
        write_topic(path, {**GOOD_TOPIC, "attempts": MAX_ATTEMPTS - 1})
    except OSError:
        pytest.skip("this machine cannot create a file name of this length")

    code, _ = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert not path.exists()
    moved = [p.name for p in (queue / FAILED).iterdir() if p.name != ".gitkeep"]
    assert len(moved) == 1
    assert len(moved[0].encode("utf-8")) <= run.MAX_FILE_NAME_BYTES
    assert moved[0].endswith(".json")
    assert load(queue / FAILED / moved[0])["topic"] == GOOD_TOPIC["topic"]


@pytest.mark.parametrize("stem", [
    pytest.param("日" * 85, id="3-byte-cjk"),
    pytest.param("\U0001F389" * 100, id="4-byte-emoji"),
    pytest.param("é" * 90, id="combining-marks"),
    pytest.param("a.b.c." + "x" * 250, id="several-dots"),
    pytest.param("a" * 196, id="one-over-the-cap"),
    pytest.param("a" * 195, id="exactly-at-the-cap"),
])
def test_failed_names_fit_the_byte_cap_for_every_counter(stem):
    for number in (0, 2, 10, 100):
        name = run._failed_name(stem, ".json", number)
        assert len(name.encode("utf-8")) <= run.MAX_FILE_NAME_BYTES
        assert name.endswith(f"-{number}.json" if number else ".json")
        assert name[0] not in "-."


def test_multibyte_stems_are_cut_on_whole_characters():
    # 200 two-byte letters: the cut keeps whole letters only (97 letters = 194 bytes).
    assert run._failed_name("é" * 200, ".json") == "é" * 97 + ".json"
    # 100 four-byte emoji: 48 emoji = 192 bytes fit with the extension.
    assert run._failed_name("\U0001F389" * 100, ".json") == "\U0001F389" * 48 + ".json"
    # A stem exactly at the cap is not changed; one byte more is shortened.
    assert run._failed_name("a" * 195, ".json") == "a" * 195 + ".json"
    assert run._failed_name("a" * 196, ".json") == "a" * 195 + ".json"
    assert run._failed_name("a" * 195, ".json", 2) == "a" * 193 + "-2.json"


def test_empty_or_unencodable_stem_becomes_topic():
    assert run._failed_name("", ".json") == "topic.json"
    assert run._failed_name("\ud800x", ".json") == "topic.json"


def test_collision_chain_with_a_multibyte_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failed = Path("queue/failed")
    failed.mkdir(parents=True)
    stem = "日" * 85  # 255 bytes before the extension: longer than the cap
    for number in (0, 2, 3):  # the plain name, -2 and -3 are all taken
        (failed / run._failed_name(stem, ".json", number)).write_text("x", encoding="utf-8")

    chosen = run._free_failed_path(stem + ".json")

    assert len(chosen.name.encode("utf-8")) <= run.MAX_FILE_NAME_BYTES
    assert chosen.name.endswith("-4.json")
    assert not chosen.exists()


# A failed delete of the pending original: the copy is removed again (rollback). If the
# rollback fails too, the message says a copy exists and the original is still in pending.

def _unlink_fails_in(*folder_names):
    """Makes Path.unlink fail for files in the named folders. Other deletes behave normally."""
    real_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self.parent.name in folder_names:
            raise OSError("made-up unlink failure")
        return real_unlink(self, *args, **kwargs)

    return mock.patch.object(Path, "unlink", fake_unlink)


def test_failed_delete_is_rolled_back_and_topic_stays_pending(queue):
    raw = topic_bytes({**GOOD_TOPIC, "attempts": MAX_ATTEMPTS - 1})
    (queue / PENDING / TOPIC).write_bytes(raw)

    with _unlink_fails_in("pending"):
        code, out = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert sorted(p.name for p in (queue / FAILED).iterdir()) == [".gitkeep"]
    assert "is still in queue/pending/ and will be tried again" in out
    assert "was not recorded" in out
    assert "A copy of the topic" not in out


def test_failed_delete_and_rollback_both_fail_reports_the_copy(queue):
    raw = topic_bytes({**GOOD_TOPIC, "attempts": MAX_ATTEMPTS - 1})
    (queue / PENDING / TOPIC).write_bytes(raw)

    with _unlink_fails_in("pending", "failed"):
        code, out = run_main(Pipeline(draft_error=RuntimeError("made-up draft failure")))

    assert code == 1
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert (queue / FAILED / TOPIC).exists()
    assert "A copy of the topic is in queue/failed/test1.json" in out
    assert "Remove one of them by hand" in out


def test_malformed_failed_delete_rolled_back_stays_pending(queue):
    raw = b'["Made-up topic"]\n'
    (queue / PENDING / TOPIC).write_bytes(raw)

    with _unlink_fails_in("pending"):
        code, out = run_main(Pipeline())

    assert code == 1
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert sorted(p.name for p in (queue / FAILED).iterdir()) == [".gitkeep"]
    assert "The copy was removed again" in out
    assert "will be tried again" in out


def test_malformed_failed_delete_and_rollback_both_fail_reports_the_copy(queue):
    raw = b'["Made-up topic"]\n'
    (queue / PENDING / TOPIC).write_bytes(raw)

    with _unlink_fails_in("pending", "failed"):
        code, out = run_main(Pipeline())

    assert code == 1
    assert (queue / PENDING / TOPIC).read_bytes() == raw
    assert (queue / FAILED / TOPIC).exists()
    assert "A copy is in queue/failed/test1.json" in out
    assert "remove one of them by hand" in out
