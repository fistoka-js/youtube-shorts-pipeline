"""Queue orchestrator: pulls one topic file from queue/pending/, runs it
through the full pipeline (draft -> produce -> upload), and moves it to
queue/done/ when finished. Designed to be run unattended (e.g. by a
scheduled task or CI job) - one call processes exactly one topic.

Claiming (CI only): right before the YouTube upload starts, the topic file
is moved to queue/in-progress/, committed and pushed to the branch. That way
a second run cannot upload the same topic. If the claim cannot be pushed,
nothing is uploaded and the file goes back to queue/pending/. Local runs
(GITHUB_ACTIONS is not "true") never run git and never use in-progress/.

Failed topics: a failure before the claim counts as an attempt. The topic file gets an
"attempts" count and stays in queue/pending/ until MAX_ATTEMPTS is reached, then it moves
to queue/failed/ with an "error" field. A malformed topic file moves there at once.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

QUEUE_PENDING = Path("queue/pending")
QUEUE_IN_PROGRESS = Path("queue/in-progress")
QUEUE_DONE = Path("queue/done")
QUEUE_FAILED = Path("queue/failed")

# A topic that fails before the claim is tried this many times, then moved to queue/failed/.
MAX_ATTEMPTS = 2
# The saved error text is cut to this many characters.
ERROR_MAX_CHARS = 200
# The longest file name written to queue/failed/, in UTF-8 bytes (file systems count bytes,
# and most allow 255). A longer topic file name is shortened to fit.
MAX_FILE_NAME_BYTES = 200


class ClaimError(Exception):
    """The topic could not be claimed (saved to origin) before the upload."""


class ClaimUndoError(Exception):
    """A failed claim could not be fully undone locally. Not a ClaimError: the
    topic may not be back in queue/pending/, so the normal message must not be used."""


def _natural_key(name: str):
    """Sort key for file names: digit runs compare as numbers, text as lower-case text.
    Splitting on digits gives text at even positions and numbers at odd positions, so
    two keys always compare the same types at each position."""
    parts = re.split(r"(\d+)", name.lower())
    return [int(p) if i % 2 else p for i, p in enumerate(parts)]


def pick_next_topic_file() -> Path | None:
    """Return the next pending topic file in file-name order (test2 before test9 before
    test10), or None if the queue is empty. Modification time is ignored."""
    # Folders named like topic files are skipped (is_file), so they can never be picked.
    files = sorted((p for p in QUEUE_PENDING.glob("*.json") if p.is_file()),
                   key=lambda p: (_natural_key(p.name), p.name))
    return files[0] if files else None


def _git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _push(branch) -> bool:
    result = _git("push", "origin", f"HEAD:{branch}", check=False)
    if result.returncode != 0:
        print(f"git push failed: {result.stderr.strip()}")
    return result.returncode == 0


def _undo_claim(topic_file: Path, claimed: Path, committed: bool):
    """Remove the local claim so the workflow's later commit step cannot push it.
    The file goes back to pending/. Other queue changes (such as the variation
    history) stay in the working tree and are committed later by the workflow.
    Raises ClaimUndoError if any undo step fails."""
    problems = []
    _git("rebase", "--abort", check=False)  # only does something if a rebase is stuck
    if committed:
        reset = _git("reset", "--mixed", "HEAD~1", check=False)  # drop the claim commit, keep files
    else:
        reset = _git("reset", "--mixed", check=False)  # unstage whatever was staged
    if reset.returncode != 0:
        problems.append(reset.stderr.strip() or f"git reset exited with status {reset.returncode}")
    try:
        shutil.move(str(claimed), str(topic_file))
    except OSError as e:
        problems.append(f"could not move {claimed.name} back to queue/pending/: {e}")
    if problems:
        raise ClaimUndoError("\n".join(problems))


def claim_topic(topic_file: Path) -> Path:
    """Move the topic to queue/in-progress/, commit all of queue/, and push.
    If the push is rejected, pull --rebase and push once more. If that also
    fails, undo everything and raise ClaimError. If the undo itself fails,
    raise ClaimUndoError instead. Returns the new path."""
    QUEUE_IN_PROGRESS.mkdir(parents=True, exist_ok=True)
    claimed = QUEUE_IN_PROGRESS / topic_file.name
    moved = False
    committed = False
    try:
        branch = os.environ.get("GITHUB_REF_NAME") or _git(
            "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        shutil.move(str(topic_file), str(claimed))
        moved = True
        # Staging all of queue/ also saves the variation history, so the
        # working tree is clean for the pull --rebase below.
        _git("add", "queue/")
        _git("commit", "-m", f"Claim {topic_file.name} for upload")
        committed = True
        if _push(branch):
            return claimed
        print("Claim push was rejected - pulling and trying once more.")
        _git("pull", "--rebase", "origin", branch)
        if _push(branch):
            return claimed
        raise ClaimError("the claim push was rejected twice")
    except Exception as e:
        # For a failed git command, show git's own error text instead of the generic exit-status message.
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            detail = e.stderr.strip()
        else:
            detail = str(e)
        if moved:
            try:
                _undo_claim(topic_file, claimed, committed)
            except ClaimUndoError as undo_error:
                # Keep the original reason the claim failed, so the message shows both.
                raise ClaimUndoError(
                    f"the claim failed because: {detail}\n{undo_error}") from e
        raise ClaimError(detail) from e


def _printable(text) -> str:
    """Text that can always be printed or saved: characters that cannot be encoded as UTF-8
    (such as a lone surrogate) are written as escapes instead."""
    return str(text).encode("utf-8", errors="backslashreplace").decode("utf-8")


def _exception_text(e) -> str:
    """str(e), or the class name if str(e) itself fails."""
    try:
        return str(e)
    except Exception:
        return type(e).__name__


def _short_error(text, fallback: str) -> str:
    """The first non-empty line of text, cut to ERROR_MAX_CHARS characters. If the text has
    no such line, the fallback is used instead. The result is always printable and savable."""
    for line in str(text).splitlines():
        if line.strip():
            return _printable(line.strip())[:ERROR_MAX_CHARS]
    return _printable(fallback)[:ERROR_MAX_CHARS]


def _write_json_atomic(path: Path, data: dict):
    """Write data to path as one line of JSON (UTF-8, LF line ending, one trailing newline).
    The text goes to a temp file in the same folder first, then replaces path in one step,
    so a crash never leaves a half-written topic file. The temp file is removed on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate (from a \ud800-style escape) cannot be written as UTF-8, so this
        # file is written with escapes instead. Nothing is lost.
        payload = (json.dumps(data, ensure_ascii=True) + "\n").encode("utf-8")
    # The temp name is fixed and short, so it never depends on the topic's file name. The
    # leading dot keeps it out of the picker's *.json pattern.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".queue-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _parse_topic(raw: bytes):
    """Read the bytes of a topic file. Returns (data, problem). data is the parsed JSON
    object, or None if the file is not a readable JSON object. problem is None for a usable
    topic, otherwise a short reason. Never raises."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, f"the file is not readable UTF-8 text (byte {e.start})"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"the file is not valid JSON ({e.msg}, line {e.lineno})"
    except Exception as e:
        # Anything else the parser rejects, such as a number with too many digits or
        # nesting that is too deep.
        return None, (f"the file is not valid JSON "
                      f"({_short_error(_exception_text(e), type(e).__name__)})")
    if not isinstance(data, dict):
        return None, "the JSON is not an object (it is a list, number or text instead of { ... })"
    topic = data.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        return data, "the 'topic' field is missing, blank or not text"
    return data, None


def _read_topic_bytes(topic_file: Path) -> bytes:
    """Read a topic file's bytes. Kept as its own function so tests can simulate an unreadable file."""
    return topic_file.read_bytes()


def _attempts_so_far(data: dict) -> int:
    """The attempts count stored in a topic file. Anything that is not a whole number of
    0 or more (missing, text, negative, true/false, decimal, null) counts as 0."""
    value = data.get("attempts")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _failed_name(stem: str, suffix: str, number: int = 0) -> str:
    """A file name for queue/failed/ of at most MAX_FILE_NAME_BYTES bytes (UTF-8). A long stem
    is shortened by whole characters, so no character is cut in half. The -N counter (when
    number is given) and the extension are always kept. A stem that is empty or cannot be
    written as UTF-8 becomes "topic"."""
    tail = f"-{number}" if number else ""
    room = max(MAX_FILE_NAME_BYTES - len((tail + suffix).encode("utf-8")), 0)
    try:
        encoded = stem.encode("utf-8")
    except UnicodeEncodeError:
        encoded = b""
    # Cutting the bytes and decoding with errors="ignore" drops a partly cut character at the end.
    short = encoded[:room].decode("utf-8", errors="ignore")
    if not short:
        short = "topic"
    return f"{short}{tail}{suffix}"


def _free_failed_path(name: str) -> Path:
    """The first file name in queue/failed/ that is not taken: test1.json, then test1-2.json,
    test1-3.json, and so on. Long names are shortened to MAX_FILE_NAME_BYTES. An existing
    failed file is never overwritten."""
    base = Path(name)
    candidate = QUEUE_FAILED / _failed_name(base.stem, base.suffix)
    number = 2
    while candidate.exists():
        candidate = QUEUE_FAILED / _failed_name(base.stem, base.suffix, number)
        number += 1
    return candidate


class FailedMoveError(Exception):
    """The failed copy was written to queue/failed/, but the original could not be deleted from
    queue/pending/. copy_left is True when the copy could not be removed again, so both files
    now exist and a person has to remove one of them. copy_name is that copy's file name."""

    def __init__(self, message: str, copy_left: bool, copy_name: str = ""):
        super().__init__(message)
        self.copy_left = copy_left
        self.copy_name = copy_name


def _move_to_failed(topic_file: Path, data: dict) -> str:
    """Write the failed copy into queue/failed/, then delete the pending original. If the delete
    fails, the copy is removed again, so the topic stays in queue/pending/ and is tried again.
    Returns the file name in queue/failed/. Raises FailedMoveError if the delete fails."""
    target = _free_failed_path(topic_file.name)
    _write_json_atomic(target, data)
    try:
        topic_file.unlink()
    except Exception as unlink_error:
        detail = _exception_text(unlink_error)
        try:
            target.unlink()
        except Exception as rollback_error:
            raise FailedMoveError(
                f"the original could not be deleted ({detail}) and the copy could not be "
                f"removed again ({_exception_text(rollback_error)})",
                copy_left=True, copy_name=target.name) from unlink_error
        raise FailedMoveError(f"the original could not be deleted ({detail})",
                              copy_left=False) from unlink_error
    return target.name


def _count_failed_attempt(topic_file: Path, error_line: str):
    """Add one failed attempt to a topic that is still in queue/pending/. Returns
    ("kept", attempts) or ("moved", error_line). Raises if the file cannot be read or written."""
    data, problem = _parse_topic(topic_file.read_bytes())
    if data is None:
        raise ValueError(f"the topic file can no longer be read ({problem})")
    attempts = _attempts_so_far(data) + 1
    data["attempts"] = attempts
    if attempts < MAX_ATTEMPTS:
        _write_json_atomic(topic_file, data)
        return ("kept", attempts)
    data["error"] = error_line
    _move_to_failed(topic_file, data)  # raises FailedMoveError if the original cannot be deleted
    return ("moved", error_line)


def _record_failure(topic_file: Path, error_line: str):
    """Record a counted failure. Never raises: if recording fails, a WARNING is printed and
    ("unrecorded", None) is returned, so the original error is still reported."""
    try:
        return _count_failed_attempt(topic_file, error_line)
    except FailedMoveError as move_error:
        print(f"WARNING: the move of {topic_file.name} to queue/failed/ did not finish: "
              f"{_printable(_exception_text(move_error))}")
        if move_error.copy_left:
            return ("copy_left", move_error.copy_name)
        return ("stays", None)
    except Exception as record_error:
        print(f"WARNING: could not record the failed attempt for {topic_file.name}: "
              f"{_printable(_exception_text(record_error))}")
        return ("unrecorded", None)


def _move_malformed_topic(topic_file: Path, raw: bytes, data, problem: str):
    """A malformed topic can never succeed, so it goes to queue/failed/ at once, with no second
    try. A JSON object keeps all its fields and gets an error field. Anything else is saved as
    {"error": ..., "original_text": ...} that keeps the file's text. Bytes that are not valid
    UTF-8 are replaced with the replacement character, so the original bytes are not kept.
    Always exits with status 1."""
    error_line = _short_error(problem, "malformed topic file")
    if data is not None:
        record = data
        record["error"] = error_line
    else:
        record = {"error": error_line, "original_text": raw.decode("utf-8", errors="replace")}
    try:
        _move_to_failed(topic_file, record)
    except FailedMoveError as move_error:
        detail = _printable(_exception_text(move_error))
        if move_error.copy_left:
            print(f"ERROR: {topic_file.name} is malformed ({error_line}). The move to queue/failed/ "
                  f"did not finish ({detail}). A copy is in queue/failed/{move_error.copy_name} and "
                  "the original is still in queue/pending/ - remove one of them by hand.")
        else:
            print(f"ERROR: {topic_file.name} is malformed ({error_line}). The move to queue/failed/ "
                  f"did not finish ({detail}). The copy was removed again, so the file is still in "
                  "queue/pending/ and will be tried again.")
        sys.exit(1)
    except Exception as move_error:
        print(f"WARNING: could not move {topic_file.name} to queue/failed/: "
              f"{_printable(_exception_text(move_error))}")
        print(f"ERROR: {topic_file.name} is malformed ({error_line}). It could not be moved, "
              "so it is still in queue/pending/ and will be tried again.")
        sys.exit(1)
    print(f"ERROR: {topic_file.name} is malformed ({error_line}). It was moved to queue/failed/ "
          "right away and will not be retried.")
    sys.exit(1)


def _move_unreadable_topic(topic_file: Path, read_error):
    """A topic file that cannot be read is moved to queue/failed/ unchanged, so the queue
    does not get stuck on it. Always exits with status 1. If the move itself fails, the file
    stays in queue/pending/ and an ERROR says so (a limit: a file that cannot be moved needs a
    hand fix)."""
    reason = _short_error(_exception_text(read_error), type(read_error).__name__)
    try:
        shutil.move(str(topic_file), str(_free_failed_path(topic_file.name)))
    except Exception as move_error:
        print(f"ERROR: {topic_file.name} could not be read ({reason}) and could not be moved to "
              f"queue/failed/ ({_printable(_exception_text(move_error))}). It is still in "
              "queue/pending/ and will be tried again - check it by hand.")
        sys.exit(1)
    print(f"ERROR: {topic_file.name} could not be read ({reason}). It was moved unchanged to "
          "queue/failed/ right away and will not be retried.")
    sys.exit(1)


def _outcome_sentence(outcome, upload_started: bool) -> str:
    """The sentence that says where a counted failure left the topic."""
    kind, value = outcome
    if kind == "moved":
        end = "" if value.endswith(".") else "."
        return f"Topic moved to queue/failed/ after {MAX_ATTEMPTS} failed attempts: {value}{end}"
    if kind == "stays":
        return ("Topic is still in queue/pending/ and will be tried again. The failed attempt "
                "was not recorded (see the WARNING above).")
    if kind == "copy_left":
        return (f"A copy of the topic is in queue/failed/{_printable(value)}, and the original is "
                "still in queue/pending/. Remove one of them by hand (see the WARNING above).")
    if kind == "unrecorded":
        return ("Topic may still be in queue/pending/, but the failed attempt was not recorded "
                "(see the WARNING above).")
    if upload_started:
        return f"Topic left in queue/pending/. Attempt {value} of {MAX_ATTEMPTS} failed."
    return (f"Topic left in queue/pending/ (nothing was uploaded). "
            f"Attempt {value} of {MAX_ATTEMPTS} failed; it will be tried again.")


def _report_failure(topic_name: str, error_text: str, state: dict,
                    prefix: str = "ERROR processing", outcome=None):
    """Print the ERROR line (or a NOTE line for a clean exit) and the note on where the topic was left.
    outcome is set only for a counted failure (see _record_failure)."""
    print(f"{prefix} {topic_name}: {_printable(error_text)}")
    if state["claimed"]:
        print("Topic left in queue/in-progress/ - check YouTube Studio to see if a video went live, "
              "then see the README recovery note.")
    elif outcome is not None:
        print(_outcome_sentence(outcome, state["upload_started"]))
        if state["upload_started"]:
            print("The upload failed, and it is not certain whether a video went live - "
                  "check YouTube Studio before running this topic again.")
    elif state["upload_started"]:
        # Local run: no claim, so the file stays in pending/. The upload may have
        # gone live before the error, so do not say nothing was uploaded.
        print("Topic left in queue/pending/. The upload failed, and it is not certain whether "
              "a video went live - check YouTube Studio before running this topic again.")
    else:
        print("Topic left in queue/pending/ (nothing was uploaded).")


def main():
    QUEUE_PENDING.mkdir(parents=True, exist_ok=True)
    QUEUE_IN_PROGRESS.mkdir(parents=True, exist_ok=True)
    QUEUE_DONE.mkdir(parents=True, exist_ok=True)
    QUEUE_FAILED.mkdir(parents=True, exist_ok=True)

    topic_file = pick_next_topic_file()
    if not topic_file:
        print("Queue is empty - nothing to do.")
        return

    print(f"Processing: {topic_file.name}")
    try:
        raw = _read_topic_bytes(topic_file)
    except Exception as read_error:
        _move_unreadable_topic(topic_file, read_error)  # always exits with status 1
    data, problem = _parse_topic(raw)
    if problem:
        _move_malformed_topic(topic_file, raw, data, problem)  # always exits with status 1

    topic = data.get("topic")
    niche = data.get("niche", "general")
    platform = data.get("platform", "shorts")

    from verticals.__main__ import cmd_run
    import verticals.upload as upload_module

    class RunArgs:
        pass

    RunArgs.news = topic
    RunArgs.niche = niche
    RunArgs.platform = platform
    RunArgs.provider = None
    RunArgs.voice = None
    RunArgs.lang = "en"
    RunArgs.dry_run = False
    RunArgs.context = ""
    RunArgs.discover = False
    RunArgs.auto_pick = False
    RunArgs.review_name = topic_file.stem  # names queue/review/<stem>.md

    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    # Where the topic file is now: pending/ until claimed, then in-progress/.
    state = {"path": topic_file, "claimed": False, "upload_started": False, "inserted": False}
    real_upload = upload_module.upload_to_youtube

    def guarded_upload(*args, **kwargs):
        if in_ci:
            state["path"] = claim_topic(state["path"])  # raises ClaimError, no upload
            state["claimed"] = True
        state["upload_started"] = True
        url = real_upload(*args, **kwargs)
        state["inserted"] = True  # upload_to_youtube only returns after the video is live
        return url

    upload_module.upload_to_youtube = guarded_upload
    try:
        cmd_run(RunArgs())
    except ClaimUndoError as e:
        print(f"ERROR: could not undo the claim for {topic_file.name}. No video was uploaded.")
        print(f"Details: {e}")
        print("A leftover local claim may still be present. The workflow's final commit step may "
              "push it, so this topic could end up in queue/in-progress/.")
        print("See the README recovery note 'If a Topic Is Stuck in queue/in-progress/'.")
        sys.exit(1)
    except ClaimError as e:
        print(f"ERROR: could not save the claim for {topic_file.name} ({e}).")
        print("Topic left in queue/pending/, nothing was uploaded.")
        sys.exit(1)
    except Exception as e:
        if state["inserted"]:
            print(f"WARNING: {topic_file.name} was uploaded and is live, but a later step failed: {e}")
            print("The topic is still moved to queue/done/ and this run exits normally.")
        else:
            outcome = None
            if not state["claimed"]:
                # Still in pending/ (not claimed, not inserted): this counts as a failed attempt.
                outcome = _record_failure(topic_file,
                                          _short_error(_exception_text(e), type(e).__name__))
            _report_failure(topic_file.name, _exception_text(e), state, outcome=outcome)
            sys.exit(1)
    except SystemExit as e:
        # A step (such as cmd_upload's missing-video check) exited on purpose. The same exit
        # code is always passed on. A clean exit is silent unless the topic was already
        # claimed or the upload had started.
        code = e.code
        if code is None or code == 0:
            clean = True
            reason = "the pipeline stopped with a clean exit"
        elif isinstance(code, int):
            clean = False
            reason = f"the pipeline stopped with exit code {code}"
        else:
            clean = False
            reason = f"the pipeline stopped with message: {code}"
        touched = state["claimed"] or state["upload_started"] or state["inserted"]
        if state["inserted"]:
            print(f"WARNING: {topic_file.name} was uploaded and is live, but the run stopped early "
                  f"({reason}).")
            print("The topic was NOT moved to queue/done/ - check YouTube Studio.")
        elif not clean or touched:
            outcome = None
            if not clean and not state["claimed"]:
                # A non-clean exit while still in pending/ counts as a failed attempt.
                # A clean exit never does.
                outcome = _record_failure(topic_file, _short_error(reason, "unknown reason"))
            _report_failure(topic_file.name, reason, state,
                            prefix="ERROR processing" if not clean else "NOTE processing",
                            outcome=outcome)
        raise
    finally:
        upload_module.upload_to_youtube = real_upload

    done_path = QUEUE_DONE / topic_file.name
    shutil.move(str(state["path"]), str(done_path))
    print(f"Moved {topic_file.name} to queue/done/")


if __name__ == "__main__":
    main()
