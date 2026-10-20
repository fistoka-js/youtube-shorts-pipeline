"""Tests for the queue orchestrator (run.py): claiming a topic before the upload,
the one retry, and where the topic file ends up.

All data is made up. Each test builds throwaway git repos under pytest's tmp_path:
a bare repo called origin, a clone called work (the working directory), and,
when needed, a second clone called other (used to push ahead). The draft,
produce and YouTube upload steps are fakes, so nothing touches the network,
YouTube, or any paid service, and the real repo's git is never used.

Written with unittest classes, run by pytest:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_run_queue.py -v
"""

import io
import json
import os
import shutil
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import run
from verticals import upload as upload_module

TOPIC = "topic_a.json"
PENDING = "queue/pending/" + TOPIC
IN_PROGRESS = "queue/in-progress/" + TOPIC
DONE = "queue/done/" + TOPIC


@pytest.fixture(autouse=True)
def _tmp_path_for_unittest(request, tmp_path):
    """Hands each unittest test case its pytest tmp_path as self.tmp."""
    if request.instance is not None:
        request.instance.tmp = tmp_path
    yield


def _git(cwd, *args):
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _configure(repo):
    _git(repo, "config", "user.name", "Made Up Tester")
    _git(repo, "config", "user.email", "tester@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")


class FakePipeline:
    """Stands in for the real draft, produce and upload steps. cmd_run runs them
    in the same order as the real one. The upload step looks up
    verticals.upload.upload_to_youtube at call time, exactly as cmd_upload does."""

    def __init__(self, *, draft_error=None, upload_error=None,
                 after_upload_error=None, history_text=None):
        self.draft_error = draft_error
        self.upload_error = upload_error
        self.after_upload_error = after_upload_error
        self.history_text = history_text
        self.upload_calls = 0
        self.upload_after_run = None  # set by _run_main once run.main() has returned

    def cmd_run(self, args):
        if self.history_text is not None:
            # The draft step records its variation choice in the history file.
            Path("queue/variation_history.json").write_text(self.history_text)
        if self.draft_error:
            raise self.draft_error
        # Produce step: nothing to do in the fake.
        url = upload_module.upload_to_youtube(Path("fake_video.mp4"), {}, None, "en", None)
        if self.after_upload_error:
            raise self.after_upload_error
        return url

    def upload(self, *args, **kwargs):
        """Stands in for the real upload_to_youtube (no network)."""
        self.upload_calls += 1
        if self.upload_error:
            raise self.upload_error
        return "https://youtu.be/FAKEVIDEO01"


class _QueueTestCase(unittest.TestCase):
    def setUp(self):
        self.origin = self.tmp / "origin.git"
        seed = self.tmp / "seed"
        self.work = self.tmp / "work"
        _git(self.tmp, "init", "--bare", "-q", "--initial-branch=main", str(self.origin))
        _git(self.tmp, "init", "-q", "--initial-branch=main", str(seed))
        _configure(seed)
        for folder in ("queue/pending", "queue/done", "queue/in-progress"):
            (seed / folder).mkdir(parents=True)
            (seed / folder / ".gitkeep").write_text("")
        (seed / "queue/variation_history.json").write_text("[]\n")
        (seed / PENDING).write_text(json.dumps({"topic": "Made-up topic A", "niche": "science"}))
        _git(seed, "add", "-A")
        _git(seed, "commit", "-q", "-m", "Seed queue")
        _git(seed, "remote", "add", "origin", str(self.origin))
        _git(seed, "push", "-q", "origin", "main")
        _git(self.tmp, "clone", "-q", str(self.origin), str(self.work))
        _configure(self.work)

        # Run every case from the work clone, in CI mode on branch main.
        old_cwd = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, old_cwd)
        env_patch = mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true",
                                                 "GITHUB_REF_NAME": "main"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def _clone_other(self):
        other = self.tmp / "other"
        _git(self.tmp, "clone", "-q", str(self.origin), str(other))
        _configure(other)
        return other

    def _origin_files(self):
        return set(_git(self.origin, "ls-tree", "-r", "--name-only", "main").splitlines())

    def _origin_sha(self):
        return _git(self.origin, "rev-parse", "main")

    def _origin_subjects(self):
        return _git(self.origin, "log", "--format=%s", "main")

    def _local_head(self):
        return _git(self.work, "rev-parse", "HEAD")

    def _run_main(self, fake):
        """Runs run.main() with the fakes in place. Returns (exit code, printed text)."""
        out = io.StringIO()
        with mock.patch("verticals.__main__.cmd_run", fake.cmd_run), \
                mock.patch.object(upload_module, "upload_to_youtube", fake.upload), \
                redirect_stdout(out):
            try:
                run.main()
                code = 0
            except SystemExit as e:
                code = e.code
            # run.py should put the real upload function back before it returns.
            fake.upload_after_run = upload_module.upload_to_youtube
        return code, out.getvalue()


class TestClaimPushRetry(_QueueTestCase):
    def test_claim_push_rejected_once_then_retry_succeeds(self):
        # Another run pushes first, so our claim push is not a fast-forward and is rejected.
        other = self._clone_other()
        (other / "queue/done/from_other_run.json").write_text('{"topic": "Other run"}')
        _git(other, "add", "-A")
        _git(other, "commit", "-q", "-m", "Other run")
        _git(other, "push", "-q", "origin", "main")

        fake = FakePipeline()
        code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / DONE).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertIn(IN_PROGRESS, self._origin_files())
        self.assertNotIn(PENDING, self._origin_files())
        self.assertIn("Claim topic_a.json for upload", self._origin_subjects())
        self.assertIn("Other run", self._origin_subjects())
        self.assertIn("Claim push was rejected", out)


class TestClaimFailsTwice(_QueueTestCase):
    def test_claim_rejected_twice_never_uploads_and_undoes_claim(self):
        # A pre-receive hook that rejects every push, so both claim pushes fail.
        hook = self.origin / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho 'rejected by test hook' >&2\nexit 1\n", newline="\n")
        os.chmod(hook, 0o755)
        origin_before = self._origin_sha()

        fake = FakePipeline()
        code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 0)
        self.assertTrue((self.work / PENDING).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertEqual(self._origin_sha(), origin_before)
        self.assertIn(PENDING, self._origin_files())
        self.assertNotIn(IN_PROGRESS, self._origin_files())
        # No stray local claim commit is left behind.
        self.assertEqual(self._local_head(), origin_before)
        self.assertEqual(_git(self.work, "status", "--porcelain"), "")
        self.assertIn("left in queue/pending/", out)
        # git's own reason (from the hook) is shown, not just an exit status.
        self.assertIn("rejected by test hook", out)


class TestClaimErrorDetail(_QueueTestCase):
    def test_failed_pull_shows_git_error_text(self):
        real_git = run._git

        def fake_git(*args, check=True):
            if args[:1] == ("pull",):
                raise subprocess.CalledProcessError(
                    1, ["git", *args], stderr="fatal: made up pull error")
            return real_git(*args, check=check)

        with mock.patch.object(run, "_git", side_effect=fake_git), \
                mock.patch.object(run, "_push", return_value=False):
            with self.assertRaises(run.ClaimError) as cm:
                run.claim_topic(Path("queue/pending") / TOPIC)

        self.assertIn("fatal: made up pull error", str(cm.exception))
        self.assertNotIn("returned non-zero exit status", str(cm.exception))
        self.assertTrue((self.work / PENDING).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertEqual(_git(self.work, "status", "--porcelain"), "")


class TestLocalUploadFailure(_QueueTestCase):
    def test_failed_local_upload_says_it_is_uncertain_whether_video_went_live(self):
        os.environ.pop("GITHUB_ACTIONS", None)
        fake = FakePipeline(upload_error=RuntimeError("made-up error after upload started"))
        code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / PENDING).exists())
        self.assertFalse((self.work / DONE).exists())
        self.assertIn("not certain whether a video went live", out)
        self.assertNotIn("nothing was uploaded", out)


class TestFailureAfterClaim(_QueueTestCase):
    def test_upload_fails_before_insert_leaves_topic_in_progress(self):
        fake = FakePipeline(upload_error=RuntimeError("made-up network error before insert"))
        code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / IN_PROGRESS).exists())
        self.assertFalse((self.work / PENDING).exists())
        self.assertFalse((self.work / DONE).exists())
        self.assertIn(IN_PROGRESS, self._origin_files())
        self.assertIn("left in queue/in-progress/", out)

    def test_failure_after_insert_still_moves_topic_to_done(self):
        fake = FakePipeline(after_upload_error=RuntimeError("made-up state save error"))
        code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / DONE).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertIn(IN_PROGRESS, self._origin_files())
        self.assertIn("WARNING", out)
        # The real upload function is put back after the run.
        self.assertEqual(fake.upload_after_run, fake.upload)


class TestCiOnlyGuard(_QueueTestCase):
    def test_local_run_never_runs_git(self):
        os.environ.pop("GITHUB_ACTIONS", None)
        origin_before = self._origin_sha()
        fake = FakePipeline()

        with mock.patch.object(run.subprocess, "run", wraps=subprocess.run) as spy:
            code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        spy.assert_not_called()
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / DONE).exists())
        self.assertFalse((self.work / PENDING).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertEqual(self._origin_sha(), origin_before)


class TestFailureBeforeClaim(_QueueTestCase):
    def test_draft_failure_leaves_topic_in_pending_and_pushes_nothing(self):
        origin_before = self._origin_sha()
        fake = FakePipeline(draft_error=RuntimeError("made-up draft failure"))
        code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 0)
        self.assertTrue((self.work / PENDING).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertEqual(self._origin_sha(), origin_before)
        self.assertIn("left in queue/pending/ (nothing was uploaded)", out)


class TestDirtyVariationHistory(_QueueTestCase):
    def test_claim_succeeds_with_modified_variation_history(self):
        new_history = json.dumps([{"niche": "science", "intro_style": "made-up style"}]) + "\n"
        fake = FakePipeline(history_text=new_history)
        code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / DONE).exists())
        # The claim commit carried the modified history to origin.
        self.assertEqual(
            _git(self.origin, "show", "main:queue/variation_history.json"),
            new_history.strip(),
        )
        self.assertIn(IN_PROGRESS, self._origin_files())


class TestRealCmdUploadUsesPatchedUpload(unittest.TestCase):
    def test_real_cmd_upload_calls_patched_upload_to_youtube(self):
        from verticals import __main__ as main_module

        video = self.tmp / "fake_video.mp4"
        video.write_bytes(b"not a real video")
        draft_path = self.tmp / "draft.json"
        draft_path.write_text(json.dumps({
            "job_id": "made-up-job", "video_en": str(video), "srt_en": None, "niche": "science",
        }))
        fake_upload = mock.Mock(return_value="https://youtu.be/FAKEVIDEO02")
        args = SimpleNamespace(draft=str(draft_path), lang="en", force=False)

        with mock.patch.object(upload_module, "upload_to_youtube", fake_upload), \
                mock.patch("verticals.thumbnail.generate_thumbnail",
                           side_effect=RuntimeError("no thumbnail in test")), \
                mock.patch.object(main_module, "log"), \
                redirect_stdout(io.StringIO()):
            url = main_module.cmd_upload(args)

        self.assertEqual(url, "https://youtu.be/FAKEVIDEO02")
        fake_upload.assert_called_once()
        self.assertEqual(fake_upload.call_args[0][0], video)


class TestDirtyHistoryWithRejectedPush(_QueueTestCase):
    def test_modified_history_and_rejected_first_push_still_finishes(self):
        # Another run pushes a queue/ commit first, so our first claim push is rejected.
        other = self._clone_other()
        (other / "queue/done/from_other_run.json").write_text('{"topic": "Other run"}')
        _git(other, "add", "-A")
        _git(other, "commit", "-q", "-m", "Other run")
        _git(other, "push", "-q", "origin", "main")

        # The draft step changes the variation history before the claim happens.
        new_history = json.dumps([{"niche": "science", "intro_style": "made-up style"}]) + "\n"
        fake = FakePipeline(history_text=new_history)
        code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / DONE).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertIn("Claim push was rejected", out)
        # The claim and the modified history both reached origin.
        self.assertIn(IN_PROGRESS, self._origin_files())
        self.assertNotIn(PENDING, self._origin_files())
        self.assertEqual(
            _git(self.origin, "show", "main:queue/variation_history.json"),
            new_history.strip(),
        )
        self.assertIn("Claim topic_a.json for upload", self._origin_subjects())
        self.assertIn("Other run", self._origin_subjects())


class TestMissingProducedVideo(_QueueTestCase):
    def test_missing_video_exits_before_claim_and_upload(self):
        from verticals import __main__ as main_module

        real_cmd_upload = main_module.cmd_upload
        draft_path = self.tmp / "draft.json"
        draft_path.write_text(json.dumps({
            "job_id": "made-up-job",
            "video_en": str(self.tmp / "no_such_video.mp4"),
            "srt_en": None,
            "niche": "science",
        }))
        origin_before = self._origin_sha()
        fake = FakePipeline()

        def cmd_run_with_real_upload_step(args):
            # Draft and produce are skipped; the real cmd_upload runs against the draft.
            real_cmd_upload(SimpleNamespace(draft=str(draft_path), lang="en", force=False))

        fake.cmd_run = cmd_run_with_real_upload_step
        code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 0)
        self.assertEqual(self._origin_sha(), origin_before)
        self.assertEqual(self._local_head(), origin_before)
        self.assertTrue((self.work / PENDING).exists())
        self.assertFalse((self.work / IN_PROGRESS).exists())
        self.assertIn("pipeline stopped with exit code 1", out)
        self.assertIn("left in queue/pending/", out)
        self.assertIn("nothing was uploaded", out)
        self.assertNotIn("left in queue/in-progress/", out)
        self.assertEqual(fake.upload_after_run, fake.upload)


class TestUndoFailsLoudly(_QueueTestCase):
    def test_failed_reset_during_undo_exits_loudly_and_never_uploads(self):
        # Every push is rejected, so the claim is undone. Git's reset then fails.
        hook = self.origin / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho 'rejected by test hook' >&2\nexit 1\n", newline="\n")
        os.chmod(hook, 0o755)
        origin_before = self._origin_sha()
        real_git = run._git

        def fake_git(*args, check=True):
            if args[:1] == ("reset",):
                return subprocess.CompletedProcess(
                    ["git", *args], 1, stdout="", stderr="fatal: made up reset failure")
            return real_git(*args, check=check)

        fake = FakePipeline()
        with mock.patch.object(run, "_git", side_effect=fake_git):
            code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 0)
        self.assertEqual(self._origin_sha(), origin_before)
        self.assertIn("could not undo the claim", out)
        self.assertIn("No video was uploaded", out)
        self.assertIn("made up reset failure", out)
        self.assertIn("README recovery note", out)
        self.assertIn("the claim push was rejected twice", out)
        self.assertNotIn("Topic left in queue/pending/", out)
        self.assertEqual(fake.upload_after_run, fake.upload)

    def test_failed_plain_reset_when_commit_failed_exits_loudly(self):
        # The claim commit itself fails, so the undo uses the plain "reset --mixed" form,
        # and that reset fails too.
        real_git = run._git

        def fake_git(*args, check=True):
            if args[:1] == ("commit",):
                raise subprocess.CalledProcessError(
                    1, ["git", *args], stderr="fatal: made up commit failure")
            if args[:1] == ("reset",):
                return subprocess.CompletedProcess(
                    ["git", *args], 1, stdout="", stderr="fatal: made up plain reset failure")
            return real_git(*args, check=check)

        origin_before = self._origin_sha()
        fake = FakePipeline()
        with mock.patch.object(run, "_git", side_effect=fake_git):
            code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 0)
        self.assertEqual(self._origin_sha(), origin_before)
        self.assertIn("could not undo the claim", out)
        self.assertIn("No video was uploaded", out)
        self.assertIn("made up plain reset failure", out)
        self.assertIn("made up commit failure", out)
        self.assertNotIn("Topic left in queue/pending/", out)
        self.assertEqual(fake.upload_after_run, fake.upload)

    def test_failed_move_back_to_pending_exits_loudly(self):
        # Every push is rejected, so the claim is undone. Git's reset works, but moving
        # the file back to queue/pending/ fails.
        hook = self.origin / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho 'rejected by test hook' >&2\nexit 1\n", newline="\n")
        os.chmod(hook, 0o755)
        origin_before = self._origin_sha()
        real_move = shutil.move

        def fake_move(src, dst, *args, **kwargs):
            if "pending" in str(dst):
                raise OSError("made up move failure")
            return real_move(src, dst, *args, **kwargs)

        fake = FakePipeline()
        with mock.patch.object(run.shutil, "move", side_effect=fake_move):
            code, out = self._run_main(fake)

        self.assertEqual(code, 1)
        self.assertEqual(fake.upload_calls, 0)
        self.assertEqual(self._origin_sha(), origin_before)
        self.assertIn("could not undo the claim", out)
        self.assertIn("No video was uploaded", out)
        self.assertIn("made up move failure", out)
        self.assertIn("the claim push was rejected twice", out)
        self.assertNotIn("Topic left in queue/pending/", out)
        self.assertEqual(fake.upload_after_run, fake.upload)


class TestSystemExitBranches(_QueueTestCase):
    def test_exit_after_upload_is_live_keeps_exit_code_and_warns(self):
        def cmd_run_uploads_then_exits(args):
            upload_module.upload_to_youtube(Path("fake_video.mp4"), {}, None, "en", None)
            raise SystemExit(3)

        fake = FakePipeline()
        fake.cmd_run = cmd_run_uploads_then_exits
        code, out = self._run_main(fake)

        self.assertEqual(code, 3)
        self.assertEqual(fake.upload_calls, 1)
        self.assertIn("uploaded and is live", out)
        self.assertIn("NOT moved to queue/done/", out)
        self.assertTrue((self.work / IN_PROGRESS).exists())
        self.assertFalse((self.work / DONE).exists())
        self.assertEqual(fake.upload_after_run, fake.upload)

    def test_clean_exit_before_claim_passes_through_silently(self):
        for exit_value in (0, None):
            with self.subTest(exit_value=exit_value):
                # Each case starts with the topic still in pending/, because a clean exit
                # before the claim never moves it.
                fake = FakePipeline()

                def cmd_run_exits_cleanly(args, value=exit_value):
                    raise SystemExit(value)

                fake.cmd_run = cmd_run_exits_cleanly
                origin_before = self._origin_sha()
                code, out = self._run_main(fake)

                self.assertEqual(code, exit_value)
                self.assertEqual(fake.upload_calls, 0)
                self.assertEqual(self._origin_sha(), origin_before)
                self.assertTrue((self.work / PENDING).exists())
                self.assertNotIn("ERROR", out)
                self.assertNotIn("left in queue", out)
                self.assertEqual(fake.upload_after_run, fake.upload)

    def test_clean_exit_after_claim_keeps_topic_in_progress_with_note(self):
        # The claim is saved, then the upload step exits cleanly (exit code 0).
        fake = FakePipeline(upload_error=SystemExit(0))
        code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / IN_PROGRESS).exists())
        self.assertFalse((self.work / DONE).exists())
        self.assertIn(IN_PROGRESS, self._origin_files())
        self.assertIn("NOTE processing", out)
        self.assertIn("Topic left in queue/in-progress/", out)
        self.assertEqual(fake.upload_after_run, fake.upload)

    def test_clean_exit_after_local_upload_started_says_pending_and_uncertain(self):
        os.environ.pop("GITHUB_ACTIONS", None)
        fake = FakePipeline(upload_error=SystemExit(0))
        code, out = self._run_main(fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.upload_calls, 1)
        self.assertTrue((self.work / PENDING).exists())
        self.assertFalse((self.work / DONE).exists())
        self.assertIn("Topic left in queue/pending/", out)
        self.assertIn("not certain whether a video went live", out)
        self.assertEqual(fake.upload_after_run, fake.upload)

    def test_string_exit_value_is_shown_once_as_message_not_exit_code(self):
        fake = FakePipeline()

        def cmd_run_exits_with_message(args):
            raise SystemExit("made up message")

        fake.cmd_run = cmd_run_exits_with_message
        code, out = self._run_main(fake)

        self.assertEqual(code, "made up message")
        self.assertEqual(fake.upload_calls, 0)
        self.assertTrue((self.work / PENDING).exists())
        self.assertIn("stopped with message: made up message", out)
        self.assertEqual(out.count("made up message"), 1)
        self.assertNotIn("exit code", out)
        self.assertIn("left in queue/pending/ (nothing was uploaded)", out)
