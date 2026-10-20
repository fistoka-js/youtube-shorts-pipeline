"""Tests for three small hardening fixes:

1. A Pexels video file with a missing, None or empty width/height no longer
   crashes the b-roll stage (verticals/broll.py).
2. A download-and-crop that fails no longer leaves its .raw.mp4 temp file
   behind (verticals/broll.py).
3. run_cmd's error message is never empty: it names the command and the exit
   code, and includes stderr or stdout when there is any (verticals/config.py).

Written with unittest, like tests/test_variation_history.py. Run with:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_hygiene -v

Everything is made up. Nothing calls Pexels, ffmpeg or any paid service.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from verticals import broll as B
from verticals import config as C


class TestBadPexelsSizes(unittest.TestCase):
    def _video(self, width, height):
        return {"id": 1, "video_files": [{"width": width, "height": height,
                                          "link": "https://example.invalid/clip.mp4"}]}

    def test_none_width_does_not_crash_portrait_check(self):
        self.assertFalse(B._is_portrait(self._video(None, 1920)))

    def test_empty_string_height_does_not_crash_portrait_check(self):
        self.assertFalse(B._is_portrait(self._video(1080, "")))

    def test_none_and_empty_values_do_not_crash_file_pick(self):
        video = {"id": 2, "video_files": [
            {"width": None, "height": 1920, "link": "https://example.invalid/a.mp4"},
            {"width": 1080, "height": "", "link": "https://example.invalid/b.mp4"},
        ]}
        self.assertEqual(B._best_file_link(video), "https://example.invalid/a.mp4")

    def test_zero_size_still_handled(self):
        self.assertFalse(B._is_portrait(self._video(0, 1920)))

    def test_infinite_width_does_not_crash(self):
        # Python's json parser accepts Infinity, and int(inf) raises OverflowError.
        self.assertFalse(B._is_portrait(self._video(float("inf"), 1920)))
        self.assertEqual(B._pexels_size(float("inf")), 0)

    def test_valid_portrait_file_still_detected(self):
        self.assertTrue(B._is_portrait(self._video(1080, 1920)))

    def test_valid_landscape_file_still_not_portrait(self):
        self.assertFalse(B._is_portrait(self._video(1920, 1080)))

    def test_pexels_size_reads_numbers_and_treats_bad_values_as_zero(self):
        self.assertEqual(B._pexels_size(1080), 1080)
        self.assertEqual(B._pexels_size("1080"), 1080)
        self.assertEqual(B._pexels_size(None), 0)
        self.assertEqual(B._pexels_size(""), 0)
        self.assertEqual(B._pexels_size("n/a"), 0)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self._body


class TestRawFileCleanup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _run_failing_crop(self, final_error):
        out = self.tmp / "broll_1_0.mp4"
        raw = out.with_suffix(".raw.mp4")

        def fake_run_cmd(cmd, **kwargs):
            # Only the final ffmpeg crop step (the one with -vf) fails.
            if "-vf" in cmd:
                raise final_error
            return None

        with mock.patch.object(B.requests, "get", return_value=_FakeResponse(b"fake video")), \
                mock.patch.object(B, "run_cmd", side_effect=fake_run_cmd):
            with self.assertRaises(type(final_error)):
                # duration given and no crop prompt, so no probing or frame grabs happen.
                B._download_and_crop_video("https://example.invalid/x.mp4", out, 4.0)
        return out, raw

    def test_raw_file_removed_when_final_crop_fails(self):
        out, raw = self._run_failing_crop(RuntimeError("crop exploded"))
        self.assertFalse(raw.exists())
        self.assertFalse(out.exists())

    def test_raw_file_removed_on_success(self):
        out = self.tmp / "ok.mp4"
        raw = out.with_suffix(".raw.mp4")
        with mock.patch.object(B.requests, "get", return_value=_FakeResponse(b"fake video")), \
                mock.patch.object(B, "run_cmd", return_value=None):
            result = B._download_and_crop_video("https://example.invalid/x.mp4", out, 4.0)
        self.assertEqual(result, out)
        self.assertFalse(raw.exists())


class TestRunCmdErrorMessage(unittest.TestCase):
    def test_empty_output_failure_names_command_and_exit_code(self):
        cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]
        with self.assertRaises(RuntimeError) as cm:
            C.run_cmd(cmd, capture=True)
        message = str(cm.exception)
        self.assertIn("exit code 3", message)
        self.assertIn(cmd[1], message)
        self.assertTrue(message.strip())

    def test_stderr_is_included_when_present(self):
        cmd = [sys.executable, "-c", "import sys; sys.stderr.write('bad input file'); sys.exit(2)"]
        with self.assertRaises(RuntimeError) as cm:
            C.run_cmd(cmd, capture=True)
        message = str(cm.exception)
        self.assertIn("exit code 2", message)
        self.assertIn("stderr: bad input file", message)

    def test_stdout_is_included_when_stderr_is_empty(self):
        cmd = [sys.executable, "-c", "print('only on stdout'); import sys; sys.exit(4)"]
        with self.assertRaises(RuntimeError) as cm:
            C.run_cmd(cmd, capture=True)
        self.assertIn("stdout: only on stdout", str(cm.exception))

    def test_success_still_returns_result(self):
        cmd = [sys.executable, "-c", "print('hello')"]
        result = C.run_cmd(cmd, capture=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "hello")


if __name__ == "__main__":
    unittest.main()
