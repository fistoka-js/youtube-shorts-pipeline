"""Tests for the b-roll build report (verticals/build_report.py) and the
report hooks in broll.py / assemble.py.

Written with unittest so they run in the project venv without pytest:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_build_report -v

All data is made up. CLIP, Pexels HTTP, ffmpeg and Pollinations are replaced
with fakes, so nothing touches the network or the real media folders.
"""

import json
import os
import random
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import torch

from verticals import assemble as A
from verticals import broll as B
from verticals import build_report as R


# ── Made-up helpers ──────────────────────────────────────────────────────────

def _text_vec(s: str, dim: int = 8) -> list:
    r = random.Random(zlib.crc32(s.encode("utf-8")))
    return [r.uniform(-1, 1) for _ in range(dim)]


class _FakeClip:
    """Stands in for the CLIP model. Text gets a fixed made-up vector per string."""

    def encode_text(self, prompts):
        return torch.tensor([_text_vec(p) for p in prompts], dtype=torch.float32)


def _fake_tokenizer(prompts):
    return list(prompts)


def _unit(vals):
    t = torch.tensor(vals, dtype=torch.float32)
    return t / t.norm()


def _video(vid, duration, files, width=None, height=None):
    return {
        "id": vid,
        "duration": duration,
        "width": width,
        "height": height,
        "video_files": files,
    }


# Made-up Pexels-style videos (links are example.invalid, never fetched).
V_LANDSCAPE_HD = _video(11, 9.0, [
    {"width": 1280, "height": 720, "link": "https://example.invalid/11_720.mp4"},
    {"width": 1920, "height": 1080, "link": "https://example.invalid/11_1080.mp4"},
], width=1920, height=1080)
V_PORTRAIT = _video(22, 6.0, [
    {"width": 1080, "height": 1920, "link": "https://example.invalid/22.mp4"},
], width=1080, height=1920)
V_4_3 = _video(33, 12.0, [
    {"width": 800, "height": 600, "link": "https://example.invalid/33.mp4"},
], width=800, height=600)


def _pool():
    """Three verified pool entries with made-up image embeddings."""
    rng = random.Random(7)
    entries = []
    for v in (V_LANDSCAPE_HD, V_PORTRAIT, V_4_3):
        emb = _unit([rng.uniform(-1, 1) for _ in range(8)])
        entries.append({"video": v, "image_embedding": emb})
    return entries


class _Quiet(unittest.TestCase):
    """Base class: silences the build log and makes temp folders that clean up."""

    def setUp(self):
        for target in ("verticals.build_report.log", "verticals.broll.log",
                       "verticals.assemble.log"):
            patcher = mock.patch(target)
            self.addCleanup(patcher.stop)
            patcher.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)


# ── Orientation and kept-width math ──────────────────────────────────────────

class TestOrientationAndKeptFraction(unittest.TestCase):
    def test_landscape_1920x1080_into_shorts_frame_keeps_0_316_of_width(self):
        frac = R.kept_fraction(1920, 1080, 1080, 1920)
        self.assertAlmostEqual(frac, 0.316, places=3)

    def test_portrait_1080x1920_into_shorts_frame_keeps_full_width(self):
        self.assertEqual(R.kept_fraction(1080, 1920, 1080, 1920), 1.0)

    def test_4_3_clip_into_shorts_frame(self):
        # 800x600 is 4:3 (1.333). Output is 9:16 (0.5625). 0.5625 / 1.3333 = 0.421875.
        self.assertAlmostEqual(R.kept_fraction(800, 600, 1080, 1920), 0.421875, places=4)

    def test_narrower_source_into_landscape_frame_keeps_full_width(self):
        # Source aspect is below the output aspect, so height is matched and
        # the top/bottom is cut. Full width is kept.
        self.assertEqual(R.kept_fraction(800, 600, 1920, 1080), 1.0)

    def test_unknown_or_bad_size_gives_none(self):
        self.assertIsNone(R.kept_fraction(None, 1080, 1080, 1920))
        self.assertIsNone(R.kept_fraction(0, 1080, 1080, 1920))
        self.assertIsNone(R.kept_fraction("wide", 1080, 1080, 1920))

    def test_orientation_landscape_portrait_and_square(self):
        self.assertEqual(R.orientation_of(1920, 1080), "landscape")
        self.assertEqual(R.orientation_of(1080, 1920), "portrait")
        self.assertEqual(R.orientation_of(1000, 1000), "portrait")  # square counts as portrait
        self.assertIsNone(R.orientation_of(None, 1080))


# ── Loop flag ────────────────────────────────────────────────────────────────

class TestLoopFlag(_Quiet):
    def test_loop_threshold_has_0_05s_tolerance(self):
        self.assertFalse(R.loops(4.04, 4.0))
        self.assertFalse(R.loops(4.0, 4.0))
        self.assertTrue(R.loops(4.06, 4.0))

    def test_unknown_lengths_give_none(self):
        self.assertIsNone(R.loops(None, 4.0))
        self.assertIsNone(R.loops(4.0, None))

    def test_saved_clip_is_the_shorter_of_clip_duration_and_native_duration(self):
        self.assertEqual(R.clip_length_on_disk(4.0, 3), 3.0)
        self.assertEqual(R.clip_length_on_disk(4.0, 10), 4.0)
        self.assertEqual(R.clip_length_on_disk(4.0, None), 4.0)

    def test_apply_slot_durations_sets_loop_per_slot(self):
        rows = [
            R.make_clip_row(slot=1, section=None, kind="video", source="pool",
                            out_w=1080, out_h=1920, video=V_PORTRAIT,
                            file_entry=V_PORTRAIT["video_files"][0], clip_duration=4.0),
            R.make_clip_row(slot=2, section=None, kind="video", source="pool",
                            out_w=1080, out_h=1920, video=V_PORTRAIT,
                            file_entry=V_PORTRAIT["video_files"][0], clip_duration=4.0),
            R.make_clip_row(slot=3, section=None, kind="image", source="image_fallback",
                            out_w=1080, out_h=1920),
        ]
        # Slot 1 fits its 4 s clip. Slot 2 is 6 s, so it loops. Slot 3 is an image.
        R.apply_slot_durations(rows, [4.02, 6.0, 5.0])
        self.assertIs(rows[0]["loop"], False)
        self.assertIs(rows[1]["loop"], True)
        self.assertIs(rows[2]["loop"], False)
        self.assertEqual(rows[1]["slot_duration"], 6.0)

    def test_short_pexels_clip_counts_as_looping_at_its_own_length(self):
        # The clip is 3 s native, asked for 4 s, so it is saved at 3 s.
        # A 3.2 s slot therefore has to loop.
        row = R.make_clip_row(slot=1, section=None, kind="video", source="pool",
                              out_w=1080, out_h=1920,
                              video={"id": 5, "duration": 3, "video_files": []},
                              file_entry={"width": 1080, "height": 1920},
                              clip_duration=4.0)
        R.apply_slot_durations([row], [3.2])
        self.assertIs(row["loop"], True)

    def test_slot_outside_assembly_list_is_left_alone(self):
        row = R.empty_row(3, None, "video", "pool")
        R.apply_slot_durations([row], [4.0])
        self.assertIsNone(row["slot_duration"])
        self.assertIsNone(row["loop"])


# ── Row contents ─────────────────────────────────────────────────────────────

class TestMakeClipRow(unittest.TestCase):
    def test_source_size_comes_from_the_downloaded_file_entry(self):
        row = R.make_clip_row(slot=1, section=None, kind="video", source="pool",
                              out_w=1080, out_h=1920, video=V_LANDSCAPE_HD,
                              file_entry=V_LANDSCAPE_HD["video_files"][1], clip_duration=4.0)
        self.assertEqual((row["source_width"], row["source_height"]), (1920, 1080))
        self.assertEqual(row["native_duration"], 9.0)  # top-level Pexels duration
        self.assertEqual(row["pexels_id"], 11)
        self.assertEqual(row["orientation"], "landscape")
        self.assertAlmostEqual(row["kept_fraction"], 0.316, places=3)

    def test_falls_back_to_top_level_size_when_file_entry_lacks_it(self):
        row = R.make_clip_row(slot=1, section=None, kind="video", source="pool",
                              out_w=1080, out_h=1920,
                              video={"id": 7, "duration": 5, "width": 1080, "height": 1920},
                              file_entry={"link": "https://example.invalid/x.mp4"})
        self.assertEqual((row["source_width"], row["source_height"]), (1080, 1920))
        self.assertEqual(row["kept_fraction"], 1.0)

    def test_image_row_has_no_pexels_fields(self):
        row = R.make_clip_row(slot=2, section="s1", kind="image", source="solid_color",
                              out_w=1080, out_h=1920)
        self.assertIsNone(row["pexels_id"])
        self.assertIsNone(row["kept_fraction"])
        self.assertIsNone(row["clip_score"])
        self.assertEqual(row["source"], "solid_color")


# ── Summary ──────────────────────────────────────────────────────────────────

class TestSummarize(unittest.TestCase):
    def _rows(self):
        return [
            R.make_clip_row(slot=1, section=None, kind="video", source="pool", out_w=1080, out_h=1920,
                            video=V_LANDSCAPE_HD, file_entry=V_LANDSCAPE_HD["video_files"][1], clip_duration=4.0),
            R.make_clip_row(slot=2, section=None, kind="video", source="pool", out_w=1080, out_h=1920,
                            video=V_PORTRAIT, file_entry=V_PORTRAIT["video_files"][0], clip_duration=4.0),
            R.make_clip_row(slot=3, section=None, kind="video", source="last_resort_search", out_w=1080,
                            out_h=1920, video=V_LANDSCAPE_HD, file_entry=V_LANDSCAPE_HD["video_files"][1],
                            clip_duration=4.0),
            R.make_clip_row(slot=4, section=None, kind="image", source="image_fallback", out_w=1080, out_h=1920),
        ]

    def test_summary_counts_landscape_average_and_duplicates(self):
        s = R.summarize(self._rows())
        self.assertEqual(s["landscape_clips"], 2)
        self.assertEqual(s["video_clips"], 3)
        self.assertEqual(s["clips_with_known_size"], 3)
        self.assertAlmostEqual(s["avg_width_kept"], (0.316 + 1.0 + 0.316) / 3, places=3)
        self.assertEqual(s["duplicate_pexels_ids"], [11])
        self.assertIn("2 landscape clip(s) of 3 video clip(s)", s["text"])
        self.assertIn("duplicate Pexels ids: 11", s["text"])
        self.assertIn("slots that loop: unknown", s["text"])  # assembly not run yet

    def test_summary_line_after_loops_known(self):
        rows = self._rows()
        R.apply_slot_durations(rows, [4.0, 4.0, 4.5, 4.0])  # slot 3 loops (4.5 > 4.05)
        s = R.summarize(rows)
        self.assertEqual(s["slots_that_loop"], 1)
        self.assertIn("slots that loop: 1", s["text"])

    def test_no_duplicates_says_none(self):
        s = R.summarize(self._rows()[:2])
        self.assertEqual(s["duplicate_pexels_ids"], [])
        self.assertIn("duplicate Pexels ids: none", s["text"])

    def test_empty_or_missing_fields_do_not_crash(self):
        s = R.summarize([{}, {"type": "video"}, "junk"])
        self.assertEqual(s["landscape_clips"], 0)
        self.assertIsNone(s["avg_width_kept"])
        self.assertIn("average width kept n/a", s["text"])


# ── Printed table ────────────────────────────────────────────────────────────

class TestFormatTable(_Quiet):
    def test_one_line_per_slot_with_header_and_separator(self):
        rows = [
            R.make_clip_row(slot=1, section=None, kind="video", source="pool", out_w=1080, out_h=1920,
                            video=V_LANDSCAPE_HD, file_entry=V_LANDSCAPE_HD["video_files"][1], clip_duration=4.0,
                            clip_score=0.25),
            R.make_clip_row(slot=2, section="s1", kind="image", source="image_fallback", out_w=1080, out_h=1920),
        ]
        R.apply_slot_durations(rows, [4.0, 5.0])
        lines = R.format_table(rows)
        self.assertEqual(len(lines), 2 + len(rows))  # header + separator + one per slot
        self.assertTrue(lines[0].startswith("Slot"))
        self.assertIn("Pexels", lines[0])
        first = lines[2]  # line 0 is the header, line 1 the separator
        self.assertIn("1920x1080", first)
        self.assertIn("landscape", first)
        self.assertIn("31.6%", first)
        self.assertIn("0.250", first)
        self.assertIn("pool", first)
        # Slot 1 has a 4.0 s slot for a 4.0 s clip (no loop); the image slot never loops.
        self.assertTrue(first.rstrip().endswith("no"))
        self.assertTrue(lines[3].rstrip().endswith("no"))

    def test_missing_and_bad_fields_do_not_crash(self):
        rows = [
            {},
            {"slot": 2, "type": "video", "kept_fraction": "bad", "native_duration": object()},
            "not a dict",
            R.empty_row(3, None, "image", "solid_color"),
        ]
        lines = R.format_table(rows)
        self.assertEqual(len(lines), 2 + len(rows))
        self.assertIn("?", lines[3])  # the unformattable value shows as "?"
        self.assertIn("solid_color", lines[-1])

    def test_print_table_logs_and_never_raises(self):
        with mock.patch.object(R, "format_table", side_effect=RuntimeError("boom")), \
             mock.patch.object(R, "log") as fake_log:
            R.print_table([{"slot": 1}], "title")  # must not raise
        messages = [c.args[0] for c in fake_log.call_args_list]
        self.assertTrue(any("could not print" in m for m in messages))

    def test_print_table_logs_title_then_each_line(self):
        with mock.patch.object(R, "log") as fake_log:
            R.print_table([R.empty_row(1, None, "image", "solid_color")], "My title")
        messages = [c.args[0] for c in fake_log.call_args_list]
        self.assertEqual(messages[0], "My title")
        self.assertEqual(len(messages), 1 + 2 + 1)


# ── Saving ───────────────────────────────────────────────────────────────────

class TestSaving(_Quiet):
    def test_report_path_sits_next_to_draft_and_is_per_job_and_language(self):
        p = R.report_path_for(self.tmp_path / "drafts" / "123.json", "123", "en")
        self.assertEqual(p, self.tmp_path / "drafts" / "build_report_123_en.json")
        self.assertNotEqual(p, R.report_path_for(self.tmp_path / "drafts" / "123.json", "123", "hi"))

    def test_write_report_writes_utf8_json_and_no_temp_file(self):
        target = self.tmp_path / "build_report_1_en.json"
        ok = R.write_report(target, {"note": "café", "clips": []})
        self.assertTrue(ok)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["note"], "café")
        self.assertFalse((self.tmp_path / "build_report_1_en.json.tmp").exists())

    def test_write_failure_returns_false_and_does_not_raise(self):
        target = self.tmp_path / "build_report_1_en.json"
        with mock.patch("verticals.build_report.os.replace", side_effect=OSError("disk full")):
            self.assertFalse(R.write_report(target, {"clips": []}))

    def test_unwritable_folder_returns_false_and_does_not_raise(self):
        blocker = self.tmp_path / "blocker"
        blocker.write_text("this is a file, not a folder")
        self.assertFalse(R.write_report(blocker / "build_report_1_en.json", {"clips": []}))

    def test_save_broll_stage_writes_report_and_survives_write_failure(self):
        rows = [R.empty_row(1, None, "image", "solid_color")]
        target = self.tmp_path / "build_report_9_en.json"
        R.save_broll_stage(target, "9", "en", "shorts", 1080, 1920, rows)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["stage"], "broll")
        self.assertEqual(data["output_size"], [1080, 1920])
        self.assertEqual(len(data["clips"]), 1)
        self.assertIn("summary", data)
        with mock.patch("verticals.build_report.os.replace", side_effect=OSError("locked")):
            R.save_broll_stage(target, "9", "en", "shorts", 1080, 1920, rows)  # must not raise

    def test_record_assembly_fills_in_slot_durations_and_summary(self):
        rows = [
            R.make_clip_row(slot=1, section=None, kind="video", source="pool", out_w=1080, out_h=1920,
                            video=V_PORTRAIT, file_entry=V_PORTRAIT["video_files"][0], clip_duration=4.0),
            R.make_clip_row(slot=2, section=None, kind="image", source="image_fallback", out_w=1080, out_h=1920),
        ]
        target = self.tmp_path / "build_report_9_en.json"
        R.save_broll_stage(target, "9", "en", "shorts", 1080, 1920, rows)
        R.record_assembly(target, [5.0, 3.0])
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["stage"], "assembled")
        self.assertIs(data["clips"][0]["loop"], True)
        self.assertEqual(data["clips"][0]["slot_duration"], 5.0)
        self.assertIs(data["clips"][1]["loop"], False)
        self.assertEqual(data["summary"]["slots_that_loop"], 1)

    def test_record_assembly_with_no_report_file_does_nothing(self):
        target = self.tmp_path / "missing.json"
        R.record_assembly(target, [4.0])
        self.assertFalse(target.exists())

    def test_record_assembly_with_no_report_file_logs_a_line(self):
        target = self.tmp_path / "missing.json"
        with mock.patch.object(R, "log") as fake_log:
            R.record_assembly(target, [4.0])
        messages = [c.args[0] for c in fake_log.call_args_list]
        self.assertTrue(any("No b-roll build report found" in m for m in messages))

    def test_save_broll_stage_logs_summary_line_with_loops_unknown(self):
        rows = [R.make_clip_row(slot=1, section=None, kind="video", source="pool", out_w=1080,
                                out_h=1920, video=V_PORTRAIT, file_entry=V_PORTRAIT["video_files"][0],
                                clip_duration=4.0)]
        target = self.tmp_path / "build_report_9_en.json"
        with mock.patch.object(R, "log") as fake_log:
            R.save_broll_stage(target, "9", "en", "shorts", 1080, 1920, rows)
        messages = [c.args[0] for c in fake_log.call_args_list]
        self.assertIn("B-roll summary: 0 landscape clip(s) of 1 video clip(s); average width kept "
                      "100.0% (over 1 clip(s) with known size); duplicate Pexels ids: none; "
                      "slots that loop: unknown", messages)

    def test_record_assembly_with_corrupt_report_does_not_raise(self):
        target = self.tmp_path / "corrupt.json"
        target.write_text("{not json", encoding="utf-8")
        R.record_assembly(target, [4.0])  # must not raise
        self.assertEqual(target.read_text(encoding="utf-8"), "{not json")


# ── Selection hooks in broll.py ──────────────────────────────────────────────

class TestBestPoolMatchScore(_Quiet):
    def _expected_score(self, prompt, entry):
        text = _unit(_text_vec(prompt))
        return float(torch.dot(text, entry["image_embedding"]))

    def test_choice_is_identical_with_and_without_score_out(self):
        pool = _pool()
        with mock.patch.object(B, "_load_clip", return_value=(_FakeClip(), None, _fake_tokenizer)):
            plain = B._best_pool_match("octopus arms", pool, set())
            scored = {}
            with_score = B._best_pool_match("octopus arms", pool, set(), score_out=scored)
        self.assertIs(plain, with_score)
        self.assertAlmostEqual(scored["score"], self._expected_score("octopus arms", plain), places=5)

    def test_duration_mode_choice_and_score_are_consistent(self):
        pool = _pool()
        with mock.patch.object(B, "_load_clip", return_value=(_FakeClip(), None, _fake_tokenizer)):
            plain = B._best_pool_match("reef", pool, set(), min_duration=5.0)
            scored = {}
            with_score = B._best_pool_match("reef", pool, set(), min_duration=5.0, score_out=scored)
        self.assertIs(plain, with_score)
        self.assertAlmostEqual(scored["score"], self._expected_score("reef", plain), places=5)

    def test_used_clips_are_still_excluded(self):
        pool = _pool()
        with mock.patch.object(B, "_load_clip", return_value=(_FakeClip(), None, _fake_tokenizer)):
            first = B._best_pool_match("reef", pool, set())
            second = B._best_pool_match("reef", pool, {first["video"]["id"]})
        self.assertNotEqual(second["video"]["id"], first["video"]["id"])

    def test_no_score_when_nothing_is_picked(self):
        scored = {}
        with mock.patch.object(B, "_load_clip", return_value=(_FakeClip(), None, _fake_tokenizer)):
            result = B._best_pool_match("reef", _pool(), {11, 22, 33}, score_out=scored)
        self.assertIsNone(result)
        self.assertEqual(scored, {})


class TestBestFileEntryMatchesLink(unittest.TestCase):
    def test_entry_and_link_agree_for_each_video(self):
        for v in (V_LANDSCAPE_HD, V_PORTRAIT, V_4_3):
            entry = B._best_file_entry(v)
            self.assertEqual(B._best_file_link(v), entry["link"])

    def test_no_files_gives_none(self):
        self.assertIsNone(B._best_file_entry({"video_files": []}))
        self.assertIsNone(B._best_file_link({"video_files": []}))


class TestGenerateBrollRecording(_Quiet):
    """Runs generate_broll and generate_broll_long_form with every outside call faked."""

    def _patches(self, downloads, pool=None, search_videos=None, pollinations_ok=False):
        def fake_download(url, out_path, duration, width=B.VIDEO_WIDTH, height=B.VIDEO_HEIGHT):
            downloads.append((url, Path(out_path).name, duration, width, height))
            return Path(out_path)

        def fake_pollinations(prompt, out_path):
            if not pollinations_ok:
                raise RuntimeError("offline in tests")
            Path(out_path).write_bytes(b"fake png")
            return out_path

        pool_entries = _pool() if pool is None else pool
        patches = [
            # Stub the key lookup so nothing reads config.json or the environment.
            mock.patch.object(B, "get_pexels_key", return_value="test-key"),
            mock.patch.object(B, "_load_clip", return_value=(_FakeClip(), None, _fake_tokenizer)),
            mock.patch.object(B, "_build_subject_pool", return_value=pool_entries),
            mock.patch.object(B, "_download_and_crop_video", side_effect=fake_download),
            mock.patch.object(B, "_fetch_pexels_candidates", return_value=search_videos or []),
            mock.patch.object(B, "_generate_pollinations_image", side_effect=fake_pollinations),
            mock.patch("time.sleep", lambda s: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_shorts_pool_slots_are_recorded_and_choice_matches_pool_match(self):
        downloads = []
        self._patches(downloads)
        pool = _pool()
        prompts = ["octopus arms. a", "reef. b"]
        expected_first = B._best_pool_match(prompts[0], pool, set())["video"]["id"]
        expected_second = B._best_pool_match(prompts[1], pool, {expected_first})["video"]["id"]

        rows = []
        frames = B.generate_broll(prompts, self.tmp_path, clip_duration=4.0,
                                  broll_subject="octopus", report_rows=rows)

        self.assertEqual(len(frames), 2)
        self.assertEqual([r["slot"] for r in rows], [1, 2])
        self.assertEqual([r["pexels_id"] for r in rows], [expected_first, expected_second])
        self.assertEqual({r["source"] for r in rows}, {"pool"})
        self.assertTrue(all(isinstance(r["clip_score"], float) for r in rows))
        self.assertTrue(all(r["section"] is None for r in rows))
        self.assertEqual(rows[0]["clip_duration"], 4.0)
        self.assertEqual(downloads[0][3:], (B.VIDEO_WIDTH, B.VIDEO_HEIGHT))

    def test_frames_and_downloads_are_identical_with_and_without_report(self):
        downloads = []
        self._patches(downloads)
        prompts = ["octopus arms. a", "reef. b"]
        frames_plain = B.generate_broll(prompts, self.tmp_path / "a", broll_subject="octopus")
        downloads_plain = list(downloads)
        downloads.clear()
        frames_report = B.generate_broll(prompts, self.tmp_path / "b", broll_subject="octopus",
                                         report_rows=[])
        downloads_report = list(downloads)
        self.assertEqual([(f["path"].name, f["type"]) for f in frames_plain],
                         [(f["path"].name, f["type"]) for f in frames_report])
        self.assertEqual(downloads_plain, downloads_report)

    def test_solid_colour_fallback_is_recorded_with_nulls(self):
        downloads = []
        self._patches(downloads, pool=[])  # no pool, no Pexels search results
        with mock.patch.object(B, "get_pexels_key", return_value=""):
            rows = []
            frames = B.generate_broll(["ocean. a"], self.tmp_path, report_rows=rows)
        self.assertEqual(frames[0]["type"], "image")
        self.assertEqual(rows[0]["source"], "solid_color")
        self.assertIsNone(rows[0]["pexels_id"])
        self.assertIsNone(rows[0]["clip_score"])
        self.assertIsNone(rows[0]["kept_fraction"])

    def test_pollinations_image_is_recorded_as_image_fallback(self):
        self._patches([], pool=[], pollinations_ok=True)
        with mock.patch.object(B, "get_pexels_key", return_value=""):
            rows = []
            B.generate_broll(["ocean. a"], self.tmp_path, report_rows=rows)
        self.assertEqual(rows[0]["source"], "image_fallback")
        self.assertEqual(rows[0]["type"], "image")
        self.assertIsNone(rows[0]["pexels_id"])

    def test_last_resort_search_clip_is_recorded_with_its_pexels_id(self):
        downloads = []
        self._patches(downloads, pool=[], search_videos=[V_4_3])
        rows = []
        B.generate_broll(["ocean. a"], self.tmp_path, report_rows=rows)
        self.assertEqual(rows[0]["source"], "last_resort_search")
        self.assertEqual(rows[0]["pexels_id"], 33)
        self.assertIsNone(rows[0]["clip_score"])
        self.assertEqual(rows[0]["orientation"], "landscape")

    def test_recording_failure_does_not_change_the_build(self):
        downloads = []
        self._patches(downloads)
        with mock.patch.object(R, "make_clip_row", side_effect=RuntimeError("broken report code")):
            rows = []
            frames = B.generate_broll(["octopus arms. a", "reef. b"], self.tmp_path,
                                      broll_subject="octopus", report_rows=rows)
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(rows), 2)  # still one row per slot
        self.assertEqual([r["slot"] for r in rows], [1, 2])

    def test_long_form_rows_carry_section_and_global_slot_number(self):
        downloads = []
        self._patches(downloads)
        sections = [
            {"id": "s1", "broll_subject": "octopus", "broll_prompts": ["octopus arms. a", "reef. b"]},
            {"id": "s2", "broll_subject": "octopus", "broll_prompts": ["ink cloud. c"]},
        ]
        rows = []
        frames = B.generate_broll_long_form(sections, self.tmp_path, width=1920, height=1080,
                                            words=None, report_rows=rows)
        self.assertEqual(len(frames), 3)
        self.assertEqual([r["slot"] for r in rows], [1, 2, 3])
        self.assertEqual([r["section"] for r in rows], ["s1", "s1", "s2"])
        self.assertTrue(all(r["source"] == "pool" for r in rows))
        # Long-form output is 1920x1080 (landscape). For this output, the clip is
        # cropped top/bottom, not at the sides, for all three made-up clips, so
        # each keeps its full width (1.0).
        self.assertEqual({r["pexels_id"]: r["kept_fraction"] for r in rows},
                         {11: 1.0, 22: 1.0, 33: 1.0})


# ── Assembly side-channel ────────────────────────────────────────────────────

class TestAssembleSlotSideChannel(_Quiet):
    """assemble_video is run with ffmpeg, the audio length and segment building
    faked. Checks the slot lengths reported via slot_out, and that passing
    slot_out changes nothing about the segments that get built."""

    def _run(self, frames, slot_out=None, **kwargs):
        out_dir = self.tmp_path / "work"
        out_dir.mkdir(exist_ok=True)
        with mock.patch.object(A, "get_audio_duration", return_value=10.0), \
             mock.patch.object(A, "_prepare_segment") as prep, \
             mock.patch.object(A, "run_cmd"):
            if slot_out is None:
                A.assemble_video(frames, self.tmp_path / "vo.mp3", out_dir, "job1", **kwargs)
            else:
                A.assemble_video(frames, self.tmp_path / "vo.mp3", out_dir, "job1",
                                 slot_out=slot_out, **kwargs)
        return prep.call_args_list

    def test_shorts_slot_lengths_match_the_segment_length_passed_to_ffmpeg(self):
        frames = [{"path": self.tmp_path / f"c{i}.mp4", "type": "video"} for i in range(3)]
        slot_out = []
        calls = self._run(frames, slot_out=slot_out)
        self.assertEqual(len(slot_out), 3)
        for i, call in enumerate(calls):
            self.assertAlmostEqual(call.args[2], slot_out[i], places=9)

    def test_passing_slot_out_does_not_change_the_segments_built(self):
        frames = [{"path": self.tmp_path / f"c{i}.mp4", "type": "video"} for i in range(3)]
        without = self._run(frames)
        with_out = self._run(frames, slot_out=[])
        self.assertEqual(
            [(c.args, c.kwargs) for c in without],
            [(c.args, c.kwargs) for c in with_out],
        )

    def test_long_form_slot_lengths_are_exact_frame_counts_over_30(self):
        frames = [{"path": self.tmp_path / f"c{i}.mp4", "type": "video", "section_id": "s1"} for i in range(3)]
        slot_out = []
        with mock.patch.object(A, "long_form_frame_counts", return_value=[30, 60, 45]):
            calls = self._run(frames, slot_out=slot_out, width=1920, height=1080,
                              sections=[{"id": "s1", "broll_prompts": ["a", "b", "c"]}])
        self.assertEqual(slot_out, [1.0, 2.0, 1.5])
        self.assertEqual([c.kwargs.get("frame_count") for c in calls], [30, 60, 45])


if __name__ == "__main__":
    unittest.main()
