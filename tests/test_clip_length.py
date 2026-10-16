"""Tests for the Shorts clip length (keep each clip's own length, capped), the
duration-aware pool selection, the slot-length estimate, and the report fields
(saved clip length, loop flag, est_slot_seconds).

Written with unittest so they run in the project venv without pytest:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_clip_length -v

All data is made up. CLIP, Pexels HTTP, ffmpeg and ffprobe are replaced with
fakes, so nothing touches the network or the real media folders.
"""

import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import json
import torch
from PIL import Image, ImageDraw, ImageStat

from verticals import broll as B
from verticals import build_report as R


OUT_W, OUT_H = 1080, 1920
SCALED_W = 3413  # 1920x1080 scaled to 1920 tall (real ffmpeg measurement)


# ── Made-up helpers ──────────────────────────────────────────────────────────

class _ScoreModel:
    """Fake CLIP: text vector is (1, 0), so a clip's score is its embedding's first value."""

    def encode_text(self, tokens):
        return torch.tensor([[1.0, 0.0] for _ in tokens], dtype=torch.float32)

    def encode_image(self, batch):
        return batch


def _tokenizer(prompts):
    return list(prompts)


def _preprocess(img):
    """Fake preprocess: a window's mean red value r becomes (r, 255 - r)."""
    r = ImageStat.Stat(img).mean[0]
    return torch.tensor([r, 255.0 - r], dtype=torch.float32)


def _scaled_frame(r_left, r_center, r_right):
    lefts = B._scaled_window_lefts(SCALED_W, OUT_W)
    img = Image.new("RGB", (SCALED_W, OUT_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for left, red in ((lefts["left"], r_left), (lefts["center"], r_center),
                      (lefts["right"], r_right)):
        draw.rectangle([left, 0, left + OUT_W - 1, OUT_H - 1], fill=(red, 0, 0))
    return img


def _vid(vid, duration, portrait=False):
    w, h = (1080, 1920) if portrait else (1920, 1080)
    return {"id": vid, "duration": duration, "width": w, "height": h,
            "video_files": [{"width": w, "height": h,
                             "link": f"https://example.invalid/{vid}.mp4"}]}


def _entry(video, score):
    return {"video": video, "image_embedding": torch.tensor(
        [score, math.sqrt(1.0 - score * score)], dtype=torch.float32)}


class _Media:
    """Fake ffmpeg/ffprobe. The raw download has raw_len seconds. A crop writes an
    output whose length is its -t value (as ffmpeg would). Frame grabs write a
    scaled frame with flat window colours."""

    def __init__(self, raw_len=9.0, width=1920, height=1080, raw_probe_ok=True,
                 out_probe_ok=True, crop_colors=(50, 50, 50)):
        self.raw_len = raw_len
        self.width, self.height = width, height
        self.raw_probe_ok = raw_probe_ok
        self.out_probe_ok = out_probe_ok
        self.crop_colors = crop_colors
        self.commands = []
        self.ss_times = []
        self.t_values = []
        self.out_len = None
        self.raw_probe_calls = 0

    def run(self, cmd, check=True, capture=False, **kwargs):
        self.commands.append(list(cmd))
        if cmd[0] == "ffprobe":
            target = str(cmd[-1])
            if "json" in cmd:
                payload = {"streams": [{"width": self.width, "height": self.height}],
                           "format": {"duration": f"{self.raw_len:.6f}"}}
                return SimpleNamespace(stdout=json.dumps(payload), returncode=0)
            if target.endswith(".raw.mp4"):
                self.raw_probe_calls += 1
                if not self.raw_probe_ok or self.raw_len is None:
                    raise RuntimeError("")
                return SimpleNamespace(stdout=f"{self.raw_len:.6f}\n", returncode=0)
            if not self.out_probe_ok or self.out_len is None:
                raise RuntimeError("")
            return SimpleNamespace(stdout=f"{self.out_len:.6f}\n", returncode=0)
        if "-ss" in cmd:
            self.ss_times.append(float(cmd[cmd.index("-ss") + 1]))
            _scaled_frame(*self.crop_colors).save(cmd[-1])
            return None
        self.t_values.append(cmd[cmd.index("-t") + 1])
        self.out_len = float(cmd[cmd.index("-t") + 1])
        Path(cmd[cmd.index("-y") - 1]).write_bytes(b"fake mp4")
        return None

    def crop_vf(self):
        crops = [c for c in self.commands if c[0] == "ffmpeg" and "-t" in c]
        return crops[-1][crops[-1].index("-vf") + 1]


def _response():
    resp = mock.Mock()
    resp.iter_content.return_value = [b"raw"]
    return resp


class _Quiet(unittest.TestCase):
    def setUp(self):
        for target in ("verticals.build_report.log", "verticals.broll.log"):
            patcher = mock.patch(target)
            self.addCleanup(patcher.stop)
            patcher.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)


def _download(media, out_path, duration, fallback=None, clip_out=None, crop_prompt=None,
              crop_out=None):
    with mock.patch.object(B, "run_cmd", side_effect=media.run), \
         mock.patch.object(B.requests, "get", return_value=_response()), \
         mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)):
        kwargs = {"fallback_duration": fallback, "clip_out": clip_out}
        if crop_prompt is not None:
            kwargs.update(crop_prompt=crop_prompt, crop_out=crop_out)
        return B._download_and_crop_video("https://example.invalid/c.mp4", out_path,
                                          duration, OUT_W, OUT_H, **kwargs)


def _generate(media, pool=None, search=None, est=None, clip_duration=None,
              report_rows=None, out_dir=None):
    """Run generate_broll with every outside call faked (Pexels, CLIP, ffmpeg, HTTP)."""
    with mock.patch.object(B, "get_pexels_key", return_value="k"), \
         mock.patch.object(B, "_build_subject_pool", return_value=pool or []), \
         mock.patch.object(B, "_fetch_pexels_candidates", return_value=search or []), \
         mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), _preprocess, _tokenizer)), \
         mock.patch.object(B, "run_cmd", side_effect=media.run), \
         mock.patch.object(B.requests, "get", return_value=_response()), \
         mock.patch("time.sleep", lambda s: None):
        kwargs = {"broll_subject": "reef", "est_slot_seconds": est,
                  "report_rows": report_rows}
        if clip_duration is not None:
            kwargs["clip_duration"] = clip_duration
        return B.generate_broll(["reef. a"], out_dir, **kwargs)


# ── 1. Constants and the word-count estimate ────────────────────────────────

class TestConstantsAndEstimate(unittest.TestCase):
    def test_named_constants_have_the_chosen_values(self):
        self.assertEqual(B.MAX_CLIP_SECONDS, 10.0)
        self.assertEqual(B.FIXED_CLIP_SECONDS, 4.0)
        self.assertEqual(B.DURATION_PREFERENCE_MARGIN, 0.02)
        self.assertEqual(B.SPEECH_WORDS_PER_MINUTE, 180)

    def test_168_words_over_18_clips(self):
        # 168 words at 180 words a minute = 56 s; split over 18 clips = 3.111 s each.
        est = B.estimate_slot_seconds(" ".join(["word"] * 168), 18)
        self.assertAlmostEqual(est, 56.0 / 18, places=6)

    def test_estimate_uses_the_speech_rate_constant(self):
        est = B.estimate_slot_seconds(" ".join(["word"] * 180), 1)
        self.assertAlmostEqual(est, 60.0, places=6)  # 180 words at 180 wpm = 60 s

    def test_no_script_or_no_clips_gives_none(self):
        self.assertIsNone(B.estimate_slot_seconds("", 5))
        self.assertIsNone(B.estimate_slot_seconds(None, 5))
        self.assertIsNone(B.estimate_slot_seconds("a b c", 0))


# ── 2. Download: keep the clip's own length, capped ─────────────────────────

class TestTrimToNativeLength(_Quiet):
    def test_case_a_three_second_clip_is_kept_at_three_seconds(self):
        media = _Media(raw_len=3.0)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", None, clip_out=clip)
        self.assertEqual(media.t_values, ["3.0"])
        self.assertEqual(clip["saved_clip_seconds"], 3.0)

    def test_case_b_nine_second_clip_is_kept_at_nine_seconds(self):
        media = _Media(raw_len=9.0)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", None, clip_out=clip)
        self.assertEqual(media.t_values, ["9.0"])
        self.assertEqual(clip["saved_clip_seconds"], 9.0)

    def test_fourteen_second_clip_is_capped_at_ten(self):
        media = _Media(raw_len=14.0)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", None, clip_out=clip)
        self.assertEqual(media.t_values, ["10.0"])
        self.assertEqual(clip["saved_clip_seconds"], 10.0)

    def test_native_probe_failure_uses_pexels_duration(self):
        media = _Media(raw_len=9.0, raw_probe_ok=False)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", None, fallback=6.0, clip_out=clip)
        self.assertEqual(media.t_values, ["6.0"])
        self.assertEqual(clip["saved_clip_seconds"], 6.0)

    def test_pexels_duration_is_capped_too(self):
        media = _Media(raw_len=9.0, raw_probe_ok=False)
        _download(media, self.tmp_path / "a.mp4", None, fallback=30.0)
        self.assertEqual(media.t_values, ["10.0"])

    def test_no_usable_length_falls_back_to_fixed_four_seconds(self):
        media = _Media(raw_len=None, raw_probe_ok=False)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", None, fallback=0, clip_out=clip)
        self.assertEqual(media.t_values, ["4.0"])
        self.assertEqual(clip["saved_clip_seconds"], 4.0)

    def test_saved_length_probe_failure_uses_the_kept_length(self):
        media = _Media(raw_len=9.0, out_probe_ok=False)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", None, clip_out=clip)
        self.assertEqual(clip["saved_clip_seconds"], 9.0)

    def test_explicit_duration_keeps_the_old_fixed_trim_and_does_not_probe_the_raw_file(self):
        media = _Media(raw_len=9.0)
        clip = {}
        _download(media, self.tmp_path / "a.mp4", 4.0, clip_out=clip)
        self.assertEqual(media.t_values, ["4.0"])
        self.assertEqual(media.raw_probe_calls, 0)

    def test_long_form_style_duration_command_is_unchanged(self):
        media = _Media(raw_len=9.0)
        _download(media, self.tmp_path / "a.mp4", 8.0)
        self.assertEqual(media.t_values, ["8.0"])
        self.assertEqual(media.raw_probe_calls, 0)

    def test_probe_of_a_missing_or_bad_file_returns_none(self):
        self.assertIsNone(B._probe_duration(self.tmp_path / "missing.mp4"))
        for value in ("nan", "inf", "-2.0", "abc", ""):
            media = _Media()
            with mock.patch.object(B, "run_cmd",
                                   return_value=SimpleNamespace(stdout=value, returncode=0)):
                self.assertIsNone(B._probe_duration(Path("x.mp4")), value)

    def test_crop_frames_follow_the_longer_kept_length(self):
        # A 10 s kept length (14 s source, capped): frames at 25/50/75% of 10 s.
        media = _Media(raw_len=14.0)
        crop_out = {}
        _download(media, self.tmp_path / "a.mp4", None, crop_prompt="reef", crop_out=crop_out)
        self.assertEqual(media.ss_times, [2.5, 5.0, 7.5])
        self.assertEqual(media.t_values, ["10.0"])
        self.assertIn("crop=1080:1920", media.crop_vf())  # centre kept (flat colours)

    def test_crop_still_moves_to_a_side_window_on_a_ten_second_clip(self):
        media = _Media(raw_len=14.0, crop_colors=(50, 50, 250))
        crop_out = {}
        _download(media, self.tmp_path / "a.mp4", None, crop_prompt="reef", crop_out=crop_out)
        self.assertEqual(crop_out["crop_side"], "right")
        self.assertIn(":2332:0", media.crop_vf())
        self.assertEqual(media.t_values, ["10.0"])


# ── 3. Pool selection: duration preference ──────────────────────────────────

class TestDurationPreference(_Quiet):
    def _pick(self, pool, est=None, prefer=False, min_duration=0.0, score_out=None):
        with mock.patch.object(B, "_load_clip", return_value=(_ScoreModel(), None, _tokenizer)):
            return B._best_pool_match("reef", pool, set(), min_duration=min_duration,
                                      score_out=score_out, prefer_portrait=prefer,
                                      est_slot_seconds=est)

    def test_no_estimate_keeps_the_old_best_score_pick(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.30)]
        self.assertEqual(self._pick(pool, est=None)["video"]["id"], 1)

    def test_longer_clip_within_margin_is_preferred(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.30)]  # gap 0.01 < 0.02
        scored = {}
        picked = self._pick(pool, est=7.0, score_out=scored)
        self.assertEqual(picked["video"]["id"], 2)
        self.assertAlmostEqual(scored["score"], 0.30, places=5)  # the real CLIP score

    def test_short_clip_kept_when_the_longer_one_is_outside_the_margin(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.27)]  # gap 0.04
        self.assertEqual(self._pick(pool, est=7.0)["video"]["id"], 1)

    def test_best_kept_when_no_clip_is_long_enough(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 4.0), 0.30)]
        self.assertEqual(self._pick(pool, est=7.0)["video"]["id"], 1)

    def test_clip_exactly_as_long_as_the_slot_qualifies(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 7.0), 0.30)]
        self.assertEqual(self._pick(pool, est=7.0)["video"]["id"], 2)

    def test_portrait_bonus_and_duration_rule_combine(self):
        # Portrait long P (0.30) gets +0.02 = 0.32, which beats landscape short L (0.31).
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0, portrait=True), 0.30)]
        self.assertEqual(self._pick(pool, est=7.0, prefer=True)["video"]["id"], 2)

    def test_portrait_bonus_ranking_alone_is_unchanged_without_estimate(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0, portrait=True), 0.30)]
        self.assertEqual(self._pick(pool, est=None, prefer=True)["video"]["id"], 2)

    def test_margin_is_measured_on_effective_scores(self):
        # Short best is portrait 0.30 (effective 0.32). Long landscape 0.29 (effective 0.29)
        # is 0.03 behind, so it is outside the margin and the short portrait stays.
        pool = [_entry(_vid(1, 3.0, portrait=True), 0.30), _entry(_vid(2, 9.0), 0.29)]
        self.assertEqual(self._pick(pool, est=7.0, prefer=True)["video"]["id"], 1)

    def test_long_form_path_ignores_the_estimate(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.30),
                _entry(_vid(3, 5.0), 0.20)]
        plain = self._pick(pool, est=None, min_duration=5.0)
        with_est = self._pick(pool, est=7.0, min_duration=5.0)
        self.assertIs(plain, with_est)

    def test_duration_rule_uses_pexels_metadata_and_downloads_nothing(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.30)]
        with mock.patch.object(B.requests, "get") as get:
            self._pick(pool, est=7.0)
        get.assert_not_called()

    def test_estimate_above_the_cap_is_compared_with_the_cap(self):
        # Estimate 10.5 s, but a saved clip is never longer than 10 s, so a 10 s clip
        # covers the slot as well as a 40 s clip does. The best 10 s clip must win.
        pool = [_entry(_vid(1, 10.0), 0.31), _entry(_vid(2, 40.0), 0.30)]
        self.assertEqual(self._pick(pool, est=10.5)["video"]["id"], 1)

    def test_estimate_above_the_cap_still_rejects_clips_shorter_than_the_cap(self):
        # 9.9 s is shorter than the 10 s cap, so it does not cover a 10.5 s estimate.
        pool = [_entry(_vid(1, 9.9), 0.31), _entry(_vid(2, 40.0), 0.30)]
        self.assertEqual(self._pick(pool, est=10.5)["video"]["id"], 2)

    def test_duration_margin_constant_is_used(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.31 - B.DURATION_PREFERENCE_MARGIN - 0.001)]
        self.assertEqual(self._pick(pool, est=7.0)["video"]["id"], 1)


# ── 4. generate_broll: selection, trimming, report ───────────────────────────

class TestGenerateBrollPaths(_Quiet):
    def test_pool_path_picks_the_long_clip_and_keeps_its_length(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.30)]
        media = _Media(raw_len=9.0)
        rows = []
        _generate(media, pool=pool, est=7.0, report_rows=rows, out_dir=self.tmp_path)
        self.assertEqual(rows[0]["pexels_id"], 2)
        self.assertEqual(media.t_values, ["9.0"])
        self.assertEqual(rows[0]["saved_clip_seconds"], 9.0)
        self.assertEqual(rows[0]["est_slot_seconds"], 7.0)

    def test_report_native_duration_is_pexels_and_saved_length_is_the_file(self):
        pool = [_entry(_vid(1, 9.0), 0.30)]
        media = _Media(raw_len=6.5)  # the real file is 6.5 s; Pexels says 9
        rows = []
        _generate(media, pool=pool, est=None, report_rows=rows, out_dir=self.tmp_path)
        self.assertEqual(rows[0]["native_duration"], 9.0)
        self.assertEqual(rows[0]["saved_clip_seconds"], 6.5)
        self.assertEqual(media.t_values, ["6.5"])

    def test_last_resort_path_gets_the_new_trimming(self):
        media = _Media(raw_len=3.0)
        rows = []
        _generate(media, pool=[], search=[_vid(501, 3.0)], est=None,
                  report_rows=rows, out_dir=self.tmp_path)
        self.assertEqual(rows[0]["source"], "last_resort_search")
        self.assertEqual(media.t_values, ["3.0"])
        self.assertEqual(rows[0]["saved_clip_seconds"], 3.0)
        self.assertIsNone(rows[0]["est_slot_seconds"])

    def test_last_resort_long_clip_is_capped(self):
        media = _Media(raw_len=14.0)
        _generate(media, pool=[], search=[_vid(502, 14.0)], out_dir=self.tmp_path)
        self.assertEqual(media.t_values, ["10.0"])

    def test_explicit_clip_duration_keeps_the_old_fixed_trim(self):
        pool = [_entry(_vid(1, 9.0), 0.30)]
        media = _Media(raw_len=9.0)
        rows = []
        _generate(media, pool=pool, clip_duration=4.0, report_rows=rows, out_dir=self.tmp_path)
        self.assertEqual(media.t_values, ["4.0"])
        self.assertEqual(rows[0]["clip_duration"], 4.0)

    def test_default_selection_is_unchanged_when_no_estimate(self):
        pool = [_entry(_vid(1, 3.0), 0.31), _entry(_vid(2, 9.0), 0.30)]
        rows = []
        _generate(_Media(), pool=pool, est=None, report_rows=rows, out_dir=self.tmp_path)
        self.assertEqual(rows[0]["pexels_id"], 1)
        self.assertIsNone(rows[0]["est_slot_seconds"])


# ── 5. Report: loop flag and fields ──────────────────────────────────────────

def _row(slot_native=9.0, saved=None, clip_duration=4.0):
    video = _vid(7, slot_native)
    return R.make_clip_row(slot=1, section=None, kind="video", source="pool", out_w=OUT_W,
                           out_h=OUT_H, video=video, file_entry=video["video_files"][0],
                           clip_duration=clip_duration, saved_clip_seconds=saved)


class TestLoopFlagUsesSavedLength(_Quiet):
    def test_case_a_seven_second_slot_with_three_second_clip_loops(self):
        rows = [_row(saved=3.0, clip_duration=3.0)]
        R.apply_slot_durations(rows, [7.0])
        self.assertEqual(rows[0]["saved_clip_seconds"], 3.0)
        self.assertIs(rows[0]["loop"], True)

    def test_case_b_seven_second_slot_with_nine_second_clip_does_not_loop(self):
        rows = [_row(saved=9.0, clip_duration=9.0)]
        R.apply_slot_durations(rows, [7.0])
        self.assertEqual(rows[0]["saved_clip_seconds"], 9.0)
        self.assertIs(rows[0]["loop"], False)

    def test_saved_length_wins_over_the_old_clip_duration(self):
        # Old rule would say "4 s clip, 7 s slot: loops". The real saved file is 9 s.
        rows = [_row(saved=9.0, clip_duration=4.0)]
        R.apply_slot_durations(rows, [7.0])
        self.assertIs(rows[0]["loop"], False)

    def test_within_tolerance_does_not_loop(self):
        rows = [_row(saved=7.0)]
        R.apply_slot_durations(rows, [7.04])
        self.assertIs(rows[0]["loop"], False)

    def test_rows_without_a_saved_length_keep_the_old_rule(self):
        rows = [_row(saved=None, clip_duration=4.0, slot_native=9.0)]
        R.apply_slot_durations(rows, [7.0])
        self.assertIs(rows[0]["loop"], True)  # old: min(4, 9) = 4 < 7

    def test_bad_saved_values_fall_back_to_the_old_rule(self):
        for bad in (True, "x", float("nan"), 0, -1.0):
            rows = [_row(saved=None, clip_duration=4.0)]
            rows[0]["saved_clip_seconds"] = bad
            R.apply_slot_durations(rows, [7.0])
            self.assertIs(rows[0]["loop"], True, bad)

    def test_new_fields_are_in_the_json_row_and_table_is_unchanged(self):
        row = R.empty_row(1, None, "video", "pool")
        self.assertIn("saved_clip_seconds", row)
        self.assertIn("est_slot_seconds", row)
        names = [name for name, _ in R._COLUMNS]
        self.assertEqual(names, ["Slot", "Section", "Pexels", "Src WxH", "Orient", "NatDur",
                                 "Kept", "Score", "Crop", "Source", "SlotDur", "Loop"])

    def test_est_slot_is_recorded_in_json_only(self):
        video = _vid(8, 9.0)
        row = R.make_clip_row(slot=1, section=None, kind="video", source="pool", out_w=OUT_W,
                              out_h=OUT_H, video=video, file_entry=video["video_files"][0],
                              clip_duration=4.0, est_slot_seconds=3.11)
        self.assertEqual(row["est_slot_seconds"], 3.11)
        self.assertNotIn("3.11", R.format_table([row])[2])


if __name__ == "__main__":
    unittest.main()
