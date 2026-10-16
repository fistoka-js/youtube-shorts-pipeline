"""Tests for the Shorts portrait preference (broll._best_pool_match) and the
subject-aware crop (broll._download_and_crop_video / _decide_crop), plus the
new build-report crop fields.

Written with unittest so they run in the project venv without pytest:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_shorts_crop -v

All data is made up. CLIP, Pexels HTTP, ffmpeg and ffprobe are replaced with
fakes, so nothing touches the network or the real media folders.
"""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image, ImageDraw

from verticals import broll as B
from verticals import build_report as R


# ── Made-up helpers ──────────────────────────────────────────────────────────

class _ScoreModel:
    """Fake CLIP. The text vector is (1, 0), so a clip's score is the first
    value of its embedding. Image batches are returned as given (already vectors)."""

    def encode_text(self, tokens):
        return torch.tensor([[1.0, 0.0] for _ in tokens], dtype=torch.float32)

    def encode_image(self, batch):
        return batch


def _tokenizer(prompts):
    return list(prompts)


def _preprocess(img):
    """Fake CLIP preprocess: the window's mean red value r becomes the vector
    (r, 255 - r). After normalizing, its score against (1, 0) is
    r / sqrt(r^2 + (255 - r)^2), see _expected_score."""
    from PIL import ImageStat
    r = ImageStat.Stat(img).mean[0]
    return torch.tensor([r, 255.0 - r], dtype=torch.float32)


def _expected_score(r):
    return r / math.sqrt(r * r + (255.0 - r) ** 2)


def _score_for_score(s):
    """Embedding whose dot product with the text vector (1, 0) equals s."""
    return torch.tensor([s, math.sqrt(1.0 - s * s)], dtype=torch.float32)


def _video(vid, duration, files, width=None, height=None):
    return {"id": vid, "duration": duration, "width": width, "height": height,
            "video_files": files}


V_LAND_A = _video(101, 9.0, [{"width": 1920, "height": 1080, "link": "https://example.invalid/a.mp4"}],
                  width=1920, height=1080)
V_LAND_B = _video(102, 9.0, [{"width": 1920, "height": 1080, "link": "https://example.invalid/b.mp4"}],
                  width=1920, height=1080)
V_PORT = _video(201, 9.0, [{"width": 1080, "height": 1920, "link": "https://example.invalid/p.mp4"}],
                width=1080, height=1920)


def _entry(video, score):
    return {"video": video, "image_embedding": _score_for_score(score)}


SCALED_W_1920x1080 = 3413   # 1920x1080 scaled to 1920 tall, measured with real ffmpeg
OUT_W, OUT_H = 1080, 1920


def _scaled_frame(r_left, r_center, r_right, scaled_w=SCALED_W_1920x1080):
    """A made-up frame already scaled to the output height (1920 tall). Its three
    crop windows (left, centre, right; 1080 wide each) each have a flat red value."""
    lefts = B._scaled_window_lefts(scaled_w, OUT_W)
    img = Image.new("RGB", (scaled_w, OUT_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for left, red in ((lefts["left"], r_left), (lefts["center"], r_center),
                      (lefts["right"], r_right)):
        draw.rectangle([left, 0, left + OUT_W - 1, OUT_H - 1], fill=(red, 0, 0))
    return img


class _Quiet(unittest.TestCase):
    """Silences the build log and gives each test a temp folder that is removed."""

    def setUp(self):
        for target in ("verticals.build_report.log", "verticals.broll.log"):
            patcher = mock.patch(target)
            self.addCleanup(patcher.stop)
            patcher.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)


# ── 1. Portrait preference ───────────────────────────────────────────────────

class TestPortraitPreference(_Quiet):
    def _pick(self, pool, prefer, min_duration=0.0, score_out=None):
        with mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), None, _tokenizer)):
            return B._best_pool_match("reef", pool, set(), min_duration=min_duration,
                                      score_out=score_out, prefer_portrait=prefer)

    def test_portrait_within_margin_wins_when_on(self):
        # Landscape best 0.30, portrait 0.29: gap 0.01 is inside the 0.02 margin.
        pool = [_entry(V_LAND_A, 0.30), _entry(V_PORT, 0.29)]
        scored = {}
        picked = self._pick(pool, True, score_out=scored)
        self.assertEqual(picked["video"]["id"], 201)
        # The score reported is the real CLIP score, not the bonus-adjusted one.
        self.assertAlmostEqual(scored["score"], 0.29, places=5)

    def test_portrait_outside_margin_loses_when_on(self):
        # Gap 0.03 is outside the 0.02 margin, so the landscape clip still wins.
        pool = [_entry(V_LAND_A, 0.30), _entry(V_PORT, 0.27)]
        picked = self._pick(pool, True)
        self.assertEqual(picked["video"]["id"], 101)

    def test_same_pool_picks_landscape_when_off(self):
        # Feature OFF (the default): the plain best score wins, as before.
        pool = [_entry(V_LAND_A, 0.30), _entry(V_PORT, 0.29)]
        self.assertEqual(self._pick(pool, False)["video"]["id"], 101)

    def test_off_and_default_give_identical_picks_on_several_pools(self):
        rng = random.Random(3)
        videos = [V_LAND_A, V_LAND_B, V_PORT]
        for _ in range(40):
            pool = [_entry(v, rng.uniform(0.1, 0.4)) for v in videos]
            with mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), None, _tokenizer)):
                default = B._best_pool_match("reef", pool, set())
                off = B._best_pool_match("reef", pool, set(), prefer_portrait=False)
            self.assertIs(default, off)

    def test_landscape_fallback_when_no_portrait_in_pool(self):
        pool = [_entry(V_LAND_A, 0.20), _entry(V_LAND_B, 0.25)]
        self.assertEqual(self._pick(pool, True)["video"]["id"], 102)

    def test_landscape_fallback_when_portrait_is_too_weak(self):
        pool = [_entry(V_LAND_A, 0.40), _entry(V_PORT, 0.10)]
        self.assertEqual(self._pick(pool, True)["video"]["id"], 101)

    def test_long_form_duration_path_ignores_portrait_bonus_by_default(self):
        # Long-form calls with min_duration and no prefer_portrait: unchanged.
        pool = [_entry(V_LAND_A, 0.30), _entry(V_PORT, 0.29)]
        self.assertEqual(self._pick(pool, False, min_duration=5.0)["video"]["id"], 101)

    def test_portrait_detection_uses_the_downloaded_file_entry(self):
        # A landscape video whose only file entry is portrait counts as portrait,
        # because the download uses that file.
        odd = _video(301, 9.0, [{"width": 1080, "height": 1920, "link": "x"}])
        self.assertTrue(B._is_portrait(odd))
        self.assertFalse(B._is_portrait(V_LAND_A))
        self.assertTrue(B._is_portrait(V_PORT))

    def test_generate_broll_turns_it_on_and_long_form_does_not(self):
        pool = [_entry(V_LAND_A, 0.30), _entry(V_PORT, 0.29)]
        calls = []
        real = B._best_pool_match

        def spy(*args, **kwargs):
            calls.append(kwargs.get("prefer_portrait", False))
            return real(*args, **kwargs)

        with mock.patch.object(B, "get_pexels_key", return_value="k"), \
             mock.patch.object(B, "_build_subject_pool", return_value=pool), \
             mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), None, _tokenizer)), \
             mock.patch.object(B, "_best_pool_match", side_effect=spy), \
             mock.patch.object(B, "_download_and_crop_video", side_effect=lambda *a, **k: Path(a[1])):
            B.generate_broll(["reef. a"], self.tmp_path, broll_subject="reef")
            B.generate_broll_long_form(
                [{"id": "s1", "broll_subject": "reef", "broll_prompts": ["reef. b"]}],
                self.tmp_path, words=None)
        self.assertEqual(calls, [True, False])


# ── 2. Crop geometry and math ────────────────────────────────────────────────

class TestCropGeometry(unittest.TestCase):
    def test_windows_for_1920x1080_scaled_to_1080x1920(self):
        # Scaled frame is 3413 wide: right window is 2333 along, centre is 1166.
        self.assertEqual(B._scaled_window_lefts(3413, 1080),
                         {"left": 0, "center": 1166, "right": 2333})

    def test_windows_for_4_3_scaled(self):
        # 1440x1080 scales to 2560 wide (measured with real ffmpeg).
        self.assertEqual(B._scaled_window_lefts(2560, 1080),
                         {"left": 0, "center": 740, "right": 1480})

    def test_windows_for_ultra_wide_scaled(self):
        # 2560x1080 scales to 4551 wide (measured with real ffmpeg).
        self.assertEqual(B._scaled_window_lefts(4551, 1080),
                         {"left": 0, "center": 1735, "right": 3471})

    def test_right_window_is_flush_with_the_right_edge_for_measured_sizes(self):
        # Widths ffmpeg actually produced for real sources (measured, not computed):
        # the right window must end exactly at the frame's right edge.
        measured = {"1000x563": 3410, "1921x1081": 3412, "2048x858": 4583,
                    "1366x768": 3415, "1920x1080": 3413, "2560x1080": 4551}
        for label, scaled_w in measured.items():
            lefts = B._scaled_window_lefts(scaled_w, 1080)
            self.assertEqual(lefts["right"] + 1080, scaled_w, label)
            self.assertEqual(B._side_crop_x(lefts["right"]) % 2, 0, label)

    def test_slack_only_for_sources_wider_than_output_shape(self):
        self.assertTrue(B._horizontal_slack(1920, 1080, 1080, 1920))
        self.assertTrue(B._horizontal_slack(1440, 1080, 1080, 1920))   # 4:3 has slack
        self.assertFalse(B._horizontal_slack(1080, 1920, 1080, 1920))  # 9:16 has none

    def test_side_crop_x_is_even_and_never_negative(self):
        self.assertEqual(B._side_crop_x(2333), 2332)
        self.assertEqual(B._side_crop_x(0), 0)
        for left in range(0, 2400, 37):
            x = B._side_crop_x(left)
            self.assertEqual(x % 2, 0)
            self.assertTrue(0 <= x <= left)


class TestAveragingOverFrames(unittest.TestCase):
    def test_each_window_is_averaged_over_three_frames(self):
        frames = [_scaled_frame(50, 60, 250), _scaled_frame(50, 40, 200), _scaled_frame(50, 50, 150)]
        lefts = B._scaled_window_lefts(SCALED_W_1920x1080, OUT_W)
        with mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)):
            scores = B._score_crop_windows(frames, lefts, OUT_W, OUT_H, "reef")
        self.assertAlmostEqual(scores["left"], _expected_score(50), places=5)
        self.assertAlmostEqual(scores["center"],
                               sum(_expected_score(r) for r in (60, 40, 50)) / 3, places=5)
        self.assertAlmostEqual(scores["right"],
                               sum(_expected_score(r) for r in (250, 200, 150)) / 3, places=5)


class TestChooseCrop(unittest.TestCase):
    """Uses fixed window scores (the scoring itself is tested above)."""

    def _choose(self, left, center, right):
        scores = {"left": left, "center": center, "right": right}
        frame = Image.new("RGB", (SCALED_W_1920x1080, OUT_H))
        with mock.patch.object(B, "_score_crop_windows", return_value=scores):
            return B._choose_crop([frame], OUT_W, OUT_H, "reef")

    def test_frame_of_wrong_height_is_refused(self):
        with self.assertRaises(ValueError):
            B._choose_crop([Image.new("RGB", (3413, 1080))], OUT_W, OUT_H, "reef")

    def test_frame_without_slack_is_refused(self):
        with self.assertRaises(ValueError):
            B._choose_crop([Image.new("RGB", (1080, 1920))], OUT_W, OUT_H, "reef")

    def test_centre_kept_when_gain_is_under_minimum(self):
        choice = self._choose(0.10, 0.200, 0.205)  # gain 0.005 < 0.01
        self.assertEqual(choice["side"], "center")
        self.assertIsNone(choice["x"])
        self.assertEqual(choice["crop_offset"], 0.5)
        self.assertEqual(choice["chosen_score"], choice["center_score"])

    def test_right_window_used_when_gain_is_over_minimum(self):
        choice = self._choose(0.10, 0.200, 0.215)  # gain 0.015 >= 0.01
        self.assertEqual(choice["side"], "right")
        self.assertEqual(choice["x"], 2332)
        self.assertEqual(choice["crop_offset"], 1.0)
        self.assertAlmostEqual(choice["center_score"], 0.200)
        self.assertAlmostEqual(choice["chosen_score"], 0.215)

    def test_left_window_used_when_it_is_best_by_enough(self):
        choice = self._choose(0.250, 0.200, 0.10)
        self.assertEqual(choice["side"], "left")
        self.assertEqual(choice["x"], 0)
        self.assertEqual(choice["crop_offset"], 0.0)

    def test_centre_kept_when_centre_is_best(self):
        choice = self._choose(0.10, 0.300, 0.20)
        self.assertEqual(choice["side"], "center")
        self.assertIsNone(choice["x"])


# ── 3. The ffmpeg commands and the download step ─────────────────────────────

class _FakeMedia:
    """Fake run_cmd / requests. Records every ffmpeg and ffprobe command.
    ffprobe reports a made-up source size and length. Frame extraction (-ss)
    writes a frame already scaled to 1920 tall (the size the scale filter gives)
    with the given window colours. Crops write a dummy file."""

    def __init__(self, src_w=480, src_h=270, native=10.0, frame_colors=None,
                 fail_ss=False, fail_probe=False, check_raw=None):
        self.src_w, self.src_h, self.native = src_w, src_h, native
        self.frame_colors = frame_colors or [(50, 50, 250)] * 3
        self.fail_ss = fail_ss
        self.fail_probe = fail_probe
        self.check_raw = check_raw
        self.commands = []
        self.ss_times = []
        self.raw_seen_during_extract = []
        self.frame_index = 0

    def run(self, cmd, check=True, capture=False, **kwargs):
        self.commands.append(list(cmd))
        if cmd[0] == "ffprobe":
            if self.fail_probe:
                raise RuntimeError("ffprobe broke")
            if "csv=p=0" in cmd:  # plain length query (clip length probing)
                return SimpleNamespace(stdout=f"{self.native:.6f}\n", returncode=0)
            payload = {"streams": [{"width": self.src_w, "height": self.src_h}],
                       "format": {"duration": f"{self.native:.6f}"}}
            return SimpleNamespace(stdout=json.dumps(payload), returncode=0)
        if "-ss" in cmd:
            self.ss_times.append(cmd[cmd.index("-ss") + 1])
            self.raw_seen_during_extract.append(Path(cmd[cmd.index("-i") + 1]).exists())
            if self.fail_ss:
                raise RuntimeError("frame grab broke")
            rl, rc, rr = self.frame_colors[self.frame_index % len(self.frame_colors)]
            self.frame_index += 1
            _scaled_frame(rl, rc, rr).save(cmd[-1])
            return None
        # The crop/scale command writes the output clip (the file named before -y).
        Path(cmd[cmd.index("-y") - 1]).write_bytes(b"fake mp4")
        return None

    def crop_vf(self):
        cmds = [c for c in self.commands if c[0] == "ffmpeg" and "-t" in c]
        return cmds[-1][cmds[-1].index("-vf") + 1]

    def crop_command(self):
        return [c for c in self.commands if c[0] == "ffmpeg" and "-t" in c][-1]


def _download_with(media, out_path, duration, crop_prompt=None, crop_out=None,
                   width=1080, height=1920, log_mock=None):
    """Run _download_and_crop_video with the fake media and a made-up download."""
    response = mock.Mock()
    response.iter_content.return_value = [b"raw clip bytes"]
    with mock.patch.object(B, "run_cmd", side_effect=media.run), \
         mock.patch.object(B.requests, "get", return_value=response), \
         mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)):
        kwargs = {}
        if crop_prompt is not None:
            kwargs["crop_prompt"] = crop_prompt
        if crop_out is not None:
            kwargs["crop_out"] = crop_out
        return B._download_and_crop_video("https://example.invalid/clip.mp4",
                                          out_path, duration, width, height, **kwargs)


CENTER_VF = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"


class TestDownloadAndCropCommands(_Quiet):
    def _old_command(self):
        """The command the code built before this change (no crop arguments)."""
        media_old = _FakeMedia()
        old_dir = self.tmp_path / "old"
        old_dir.mkdir()
        _download_with(media_old, old_dir / "a.mp4", 4.0)
        return media_old.crop_command()

    def test_centre_kept_gives_the_old_command_exactly(self):
        media = _FakeMedia(frame_colors=[(50, 50, 50)] * 3)  # centre best, no gain
        crop_out = {}
        new_dir = self.tmp_path / "new"
        new_dir.mkdir()
        _download_with(media, new_dir / "a.mp4", 4.0, crop_prompt="reef", crop_out=crop_out)
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertEqual(crop_out["crop_side"], "center")
        # Same command as before, once the folder names (the only difference) are unified.
        new_cmd = [str(a).replace(str(new_dir), "DIR") for a in media.crop_command()]
        old_cmd = [str(a).replace(str(self.tmp_path / "old"), "DIR") for a in self._old_command()]
        self.assertEqual(new_cmd, old_cmd)

    def test_non_centre_offset_changes_only_the_crop_x(self):
        media = _FakeMedia(frame_colors=[(50, 50, 250)] * 3)  # right window wins
        crop_out = {}
        new_dir = self.tmp_path / "new"
        new_dir.mkdir()
        _download_with(media, new_dir / "a.mp4", 4.0, crop_prompt="reef", crop_out=crop_out)
        self.assertEqual(media.crop_vf(),
                         "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:2332:0")
        new_cmd = [str(a).replace(str(new_dir), "DIR") for a in media.crop_command()]
        old_cmd = [str(a).replace(str(self.tmp_path / "old"), "DIR") for a in self._old_command()]
        self.assertEqual(len(new_cmd), len(old_cmd))
        diffs = [(a, b) for a, b in zip(new_cmd, old_cmd) if a != b]
        self.assertEqual(len(diffs), 1)  # only the -vf value differs
        self.assertEqual(diffs[0][0].replace(":2332:0", ""), diffs[0][1])
        self.assertEqual(crop_out["crop_side"], "right")
        self.assertAlmostEqual(crop_out["center_score"], _expected_score(50), places=5)
        self.assertAlmostEqual(crop_out["chosen_score"], _expected_score(250), places=5)

    def test_frames_come_from_25_50_75_percent_of_kept_length(self):
        media = _FakeMedia(native=10.0)
        _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef")
        self.assertEqual(media.ss_times, ["1.000", "2.000", "3.000"])

    def test_kept_length_is_the_native_length_when_clip_is_shorter(self):
        media = _FakeMedia(native=3.0)
        _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef")
        self.assertEqual(media.ss_times, ["0.750", "1.500", "2.250"])

    def test_frames_are_read_from_the_raw_file_before_it_is_deleted(self):
        media = _FakeMedia()
        out = self.tmp_path / "a.mp4"
        _download_with(media, out, 4.0, crop_prompt="reef")
        self.assertEqual(media.raw_seen_during_extract, [True, True, True])
        self.assertFalse(out.with_suffix(".raw.mp4").exists())

    def test_temp_frames_are_cleaned_up(self):
        media = _FakeMedia()
        _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef")
        self.assertEqual(list(self.tmp_path.glob("crop_frames_*")), [])
        self.assertEqual(sorted(p.name for p in self.tmp_path.iterdir()), ["a.mp4"])

    def test_portrait_source_skips_the_step_entirely(self):
        media = _FakeMedia(src_w=1080, src_h=1920)
        crop_out = {}
        _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef",
                       crop_out=crop_out)
        self.assertEqual(media.ss_times, [])
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertEqual(crop_out, {})

    def test_long_form_style_call_without_prompt_never_probes(self):
        media = _FakeMedia()
        _download_with(media, self.tmp_path / "a.mp4", 4.0)
        self.assertFalse(any(c[0] == "ffprobe" for c in media.commands))
        self.assertEqual(media.ss_times, [])
        self.assertEqual(media.crop_vf(), CENTER_VF)

    def test_frame_failure_falls_back_to_centre_and_warns(self):
        media = _FakeMedia(fail_ss=True)
        crop_out = {}
        with mock.patch.object(B, "log") as log_mock:
            out = _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef",
                                 crop_out=crop_out)
        self.assertEqual(out, self.tmp_path / "a.mp4")
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertEqual(crop_out, {})
        self.assertTrue(any("WARNING" in str(c.args[0]) for c in log_mock.call_args_list))
        self.assertEqual(list(self.tmp_path.glob("crop_frames_*")), [])

    def test_probe_failure_falls_back_to_centre(self):
        media = _FakeMedia(fail_probe=True)
        crop_out = {}
        _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef", crop_out=crop_out)
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertEqual(crop_out, {})

    def test_blank_error_warning_still_names_the_error(self):
        # run_cmd(..., capture=True) with -v quiet can raise an error with no text.
        with mock.patch.object(B, "run_cmd", side_effect=RuntimeError("")), \
             mock.patch.object(B, "log") as log_mock:
            result = B._decide_crop(self.tmp_path / "x.raw.mp4", 4.0, OUT_W, OUT_H, "reef")
        self.assertIsNone(result)
        messages = [str(c.args[0]) for c in log_mock.call_args_list]
        warning = [m for m in messages if m.startswith("WARNING")]
        self.assertEqual(len(warning), 1)
        self.assertIn("RuntimeError: no details given", warning[0])
        self.assertTrue(warning[0].endswith("using the centre crop"))

    def test_error_with_text_keeps_its_text_in_the_warning(self):
        with mock.patch.object(B, "run_cmd", side_effect=RuntimeError("probe exploded")), \
             mock.patch.object(B, "log") as log_mock:
            B._decide_crop(self.tmp_path / "x.raw.mp4", 4.0, OUT_W, OUT_H, "reef")
        messages = [str(c.args[0]) for c in log_mock.call_args_list]
        self.assertTrue(any("RuntimeError: probe exploded" in m for m in messages))

    def test_clip_load_failure_falls_back_to_centre(self):
        media = _FakeMedia()
        response = mock.Mock()
        response.iter_content.return_value = [b"raw"]
        crop_out = {}
        with mock.patch.object(B, "run_cmd", side_effect=media.run), \
             mock.patch.object(B.requests, "get", return_value=response), \
             mock.patch.object(B, "_load_clip", side_effect=OSError("no model file")):
            B._download_and_crop_video("https://example.invalid/x.mp4", self.tmp_path / "a.mp4",
                                       4.0, 1080, 1920, crop_prompt="reef", crop_out=crop_out)
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertEqual(crop_out, {})

    def test_scoring_error_falls_back_to_centre(self):
        media = _FakeMedia()
        crop_out = {}
        with mock.patch.object(B, "_score_crop_windows", side_effect=ValueError("bad math")):
            _download_with(media, self.tmp_path / "a.mp4", 4.0, crop_prompt="reef",
                           crop_out=crop_out)
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertEqual(crop_out, {})


# ── 4. Build report fields ───────────────────────────────────────────────────

class TestReportCropFields(_Quiet):
    def test_empty_row_has_the_new_fields_as_null(self):
        row = R.empty_row(1, None, "video", "pool")
        for key in ("crop_offset", "crop_side", "center_score", "chosen_score"):
            self.assertIn(key, row)
            self.assertIsNone(row[key])

    def test_old_fields_are_still_there(self):
        row = R.make_clip_row(slot=1, section=None, kind="video", source="pool",
                              out_w=1080, out_h=1920, video=V_LAND_A,
                              file_entry=V_LAND_A["video_files"][0], clip_duration=4.0,
                              clip_score=0.3,
                              crop={"crop_offset": 1.0, "crop_side": "right",
                                    "center_score": 0.2, "chosen_score": 0.9})
        for key in ("slot", "section", "type", "pexels_id", "source_width", "source_height",
                    "native_duration", "orientation", "kept_fraction", "clip_score",
                    "source", "clip_duration", "slot_duration", "loop"):
            self.assertIn(key, row)
        self.assertEqual(row["clip_score"], 0.3)
        self.assertEqual(row["crop_offset"], 1.0)
        self.assertEqual(row["crop_side"], "right")
        self.assertEqual(row["center_score"], 0.2)
        self.assertEqual(row["chosen_score"], 0.9)

    def test_no_crop_decision_gives_null_fields(self):
        row = R.make_clip_row(slot=1, section=None, kind="video", source="pool",
                              out_w=1080, out_h=1920, video=V_PORT,
                              file_entry=V_PORT["video_files"][0], clip_duration=4.0)
        self.assertIsNone(row["crop_offset"])
        self.assertIsNone(row["center_score"])
        self.assertIsNone(row["chosen_score"])

    def test_image_rows_have_null_crop_fields(self):
        row = R.make_clip_row(slot=2, section=None, kind="image", source="image_fallback",
                              out_w=1080, out_h=1920,
                              crop={"crop_offset": 1.0, "center_score": 0.1, "chosen_score": 0.2})
        self.assertIsNone(row["crop_offset"])
        self.assertIsNone(row["center_score"])

    def test_table_columns_are_as_expected_with_crop_after_score(self):
        names = [name for name, _ in R._COLUMNS]
        self.assertEqual(names, ["Slot", "Section", "Pexels", "Src WxH", "Orient",
                                 "NatDur", "Kept", "Score", "Crop", "Source", "SlotDur", "Loop"])

    def test_generate_broll_records_crop_fields_for_a_pool_clip(self):
        pool = [_entry(V_LAND_A, 0.90)]
        rows = []
        response = mock.Mock()
        response.iter_content.return_value = [b"raw"]
        media = _FakeMedia(frame_colors=[(50, 50, 250)] * 3)
        with mock.patch.object(B, "get_pexels_key", return_value="k"), \
             mock.patch.object(B, "_build_subject_pool", return_value=pool), \
             mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)), \
             mock.patch.object(B, "run_cmd", side_effect=media.run), \
             mock.patch.object(B.requests, "get", return_value=response):
            frames = B.generate_broll(["reef. a"], self.tmp_path, clip_duration=4.0,
                                      broll_subject="reef", report_rows=rows)
        self.assertEqual(len(frames), 1)
        row = rows[0]
        self.assertEqual(row["source"], "pool")
        self.assertEqual(row["crop_side"], "right")
        self.assertEqual(row["crop_offset"], 1.0)
        self.assertAlmostEqual(row["center_score"], _expected_score(50), places=5)
        self.assertAlmostEqual(row["chosen_score"], _expected_score(250), places=5)
        self.assertIn(":2332:0", media.crop_vf())

    def test_long_form_rows_have_null_crop_fields(self):
        pool = [_entry(V_LAND_A, 0.90)]
        rows = []
        with mock.patch.object(B, "get_pexels_key", return_value="k"), \
             mock.patch.object(B, "_build_subject_pool", return_value=pool), \
             mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)), \
             mock.patch.object(B, "_download_and_crop_video",
                               side_effect=lambda *a, **k: Path(a[1])) as dl:
            B.generate_broll_long_form(
                [{"id": "s1", "broll_subject": "reef", "broll_prompts": ["reef. a"]}],
                self.tmp_path, width=1920, height=1080, words=None, report_rows=rows)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["crop_offset"])
        self.assertIsNone(rows[0]["center_score"])
        self.assertNotIn("crop_prompt", dl.call_args.kwargs)


class TestSquareGetsNoPortraitBonus(_Quiet):
    def test_square_file_is_not_portrait(self):
        square = _video(401, 9.0, [{"width": 1080, "height": 1080, "link": "x"}])
        self.assertFalse(B._is_portrait(square))

    def test_square_loses_to_landscape_within_margin(self):
        # Square 0.29 vs landscape 0.30: a square gets no bonus, so landscape wins.
        square = _video(401, 9.0, [{"width": 1080, "height": 1080, "link": "x"}])
        pool = [_entry(V_LAND_A, 0.30), _entry(square, 0.29)]
        with mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), None, _tokenizer)):
            picked = B._best_pool_match("reef", pool, set(), prefer_portrait=True)
        self.assertEqual(picked["video"]["id"], 101)


class TestLastResortSearchCrop(_Quiet):
    """The Shorts last-resort search path (no pool) gets the same subject-aware crop."""

    def test_last_resort_landscape_clip_is_cropped_and_recorded(self):
        found = _video(501, 9.0, [{"width": 1920, "height": 1080, "link": "https://example.invalid/l.mp4"}],
                       width=1920, height=1080)
        response = mock.Mock()
        response.iter_content.return_value = [b"raw"]
        media = _FakeMedia(frame_colors=[(50, 50, 250)] * 3)
        rows = []
        with mock.patch.object(B, "get_pexels_key", return_value="k"), \
             mock.patch.object(B, "_build_subject_pool", return_value=[]), \
             mock.patch.object(B, "_fetch_pexels_candidates", return_value=[found]), \
             mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)), \
             mock.patch.object(B, "run_cmd", side_effect=media.run), \
             mock.patch.object(B.requests, "get", return_value=response):
            frames = B.generate_broll(["reef. a"], self.tmp_path, clip_duration=4.0,
                                      broll_subject="reef", report_rows=rows)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], "last_resort_search")
        self.assertEqual(row["pexels_id"], 501)
        self.assertEqual(row["crop_side"], "right")
        self.assertEqual(row["crop_offset"], 1.0)
        self.assertAlmostEqual(row["center_score"], _expected_score(50), places=5)
        self.assertAlmostEqual(row["chosen_score"], _expected_score(250), places=5)
        self.assertIn(":2332:0", media.crop_vf())

    def test_last_resort_portrait_clip_skips_crop(self):
        portrait = _video(502, 9.0, [{"width": 1080, "height": 1920, "link": "https://example.invalid/p.mp4"}],
                          width=1080, height=1920)
        response = mock.Mock()
        response.iter_content.return_value = [b"raw"]
        media = _FakeMedia(src_w=1080, src_h=1920)
        rows = []
        with mock.patch.object(B, "get_pexels_key", return_value="k"), \
             mock.patch.object(B, "_build_subject_pool", return_value=[]), \
             mock.patch.object(B, "_fetch_pexels_candidates", return_value=[portrait]), \
             mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)), \
             mock.patch.object(B, "run_cmd", side_effect=media.run), \
             mock.patch.object(B.requests, "get", return_value=response):
            B.generate_broll(["reef. a"], self.tmp_path, clip_duration=4.0,
                             broll_subject="reef", report_rows=rows)
        self.assertEqual(media.ss_times, [])
        self.assertEqual(media.crop_vf(), CENTER_VF)
        self.assertIsNone(rows[0]["crop_offset"])
        self.assertIsNone(rows[0]["center_score"])


if __name__ == "__main__":
    unittest.main()
