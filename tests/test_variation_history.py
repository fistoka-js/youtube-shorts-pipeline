"""Tests for the variation history file (verticals/variation.py) and the cloud
workflow settings that point it at queue/.

Written with unittest so they run in the project venv without pytest:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_variation_history -v

All data is made up. Every history file lives in a temp directory, the real
.verticals\\variation_history.json is never read or written, and the logger is
replaced with a fake so nothing is written to the logs folder.
"""

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from verticals import variation as V


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "produce.yml"
ENV_NAME = "VARIATION_HISTORY_PATH"


def _entry(niche, i):
    """A made-up history entry using the first options of each list."""
    return {
        "niche": niche,
        "intro_style": V.INTRO_STYLES[i % len(V.INTRO_STYLES)],
        "narrative_structure": V.NARRATIVE_STRUCTURES[i % len(V.NARRATIVE_STRUCTURES)],
        "outro_treatment": V.OUTRO_TREATMENTS[i % len(V.OUTRO_TREATMENTS)],
        "visual_treatment": V.VISUAL_TREATMENTS[i % len(V.VISUAL_TREATMENTS)],
    }


class _HistoryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        # Silence and isolate the logger (it would otherwise write to LOGS_DIR).
        logger_patch = mock.patch.object(V, "get_logger")
        self.fake_get_logger = logger_patch.start()
        self.addCleanup(logger_patch.stop)
        self.fake_logger = self.fake_get_logger.return_value
        # Default path points at a temp file, so the real history is never touched.
        self.default_path = self.tmp / "default_dir" / "variation_history.json"
        default_patch = mock.patch.object(V, "HISTORY_PATH", self.default_path)
        default_patch.start()
        self.addCleanup(default_patch.stop)
        # Start every test with the env var unset.
        env_patch = mock.patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop(ENV_NAME, None)

    def set_env_path(self, path):
        os.environ[ENV_NAME] = str(path)


class TestPathSelection(_HistoryTestCase):
    def test_unset_uses_default_path(self):
        self.assertEqual(V._history_path(), self.default_path)

    def test_blank_env_var_uses_default_path(self):
        os.environ[ENV_NAME] = "   "
        self.assertEqual(V._history_path(), self.default_path)

    def test_env_var_overrides_default(self):
        target = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(target)
        self.assertEqual(V._history_path(), target)

    def test_env_var_write_does_not_touch_default_file(self):
        target = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(target)
        V.record_variation("science", _entry("science", 0))
        self.assertTrue(target.exists())
        self.assertFalse(self.default_path.exists())

    def test_default_path_still_used_for_read_and_write_when_unset(self):
        self.default_path.parent.mkdir(parents=True)
        self.default_path.write_text(json.dumps([_entry("science", 1)]))
        V.record_variation("science", _entry("science", 2))
        saved = json.loads(self.default_path.read_text())
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0], _entry("science", 1))


class TestEmptyAndMissing(_HistoryTestCase):
    def test_missing_file_loads_as_empty(self):
        self.set_env_path(self.tmp / "nope.json")
        self.assertEqual(V._load_history(), [])

    def test_missing_file_choose_returns_all_four_keys(self):
        self.set_env_path(self.tmp / "nope.json")
        random.seed(1)
        choice = V.choose_variation("science")
        self.assertEqual(
            set(choice),
            {"intro_style", "narrative_structure", "outro_treatment", "visual_treatment"},
        )
        self.assertIn(choice["intro_style"], V.INTRO_STYLES)
        self.assertIn(choice["narrative_structure"], V.NARRATIVE_STRUCTURES)
        self.assertIn(choice["outro_treatment"], V.OUTRO_TREATMENTS)
        self.assertIn(choice["visual_treatment"], V.VISUAL_TREATMENTS)

    def test_zero_byte_file_loads_as_empty_and_warns(self):
        path = self.tmp / "empty.json"
        path.write_bytes(b"")
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [])
        self.assertTrue(self.fake_logger.warning.called)

    def test_zero_byte_file_is_replaced_by_valid_write(self):
        path = self.tmp / "empty.json"
        path.write_bytes(b"")
        self.set_env_path(path)
        V.record_variation("science", _entry("science", 0))
        saved = json.loads(path.read_text())
        self.assertEqual(saved, [_entry("science", 0)])


class TestCorruptFiles(_HistoryTestCase):
    def test_invalid_json_loads_as_empty(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json at all", encoding="utf-8")
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [])
        self.assertTrue(self.fake_logger.warning.called)

    def test_invalid_json_is_replaced_cleanly_on_write(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json at all", encoding="utf-8")
        self.set_env_path(path)
        V.record_variation("science", _entry("science", 0))
        self.assertEqual(json.loads(path.read_text()), [_entry("science", 0)])

    def test_wrong_type_dict_loads_as_empty(self):
        path = self.tmp / "dict.json"
        path.write_text(json.dumps({"niche": "science"}))
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [])
        self.assertTrue(self.fake_logger.warning.called)

    def test_wrong_type_string_loads_as_empty(self):
        path = self.tmp / "str.json"
        path.write_text(json.dumps("hello"))
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [])

    def test_wrong_type_number_loads_as_empty(self):
        path = self.tmp / "num.json"
        path.write_text("42")
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [])

    def test_wrong_type_string_does_not_crash_choose_or_record(self):
        path = self.tmp / "str.json"
        path.write_text(json.dumps("hello"))
        self.set_env_path(path)
        random.seed(2)
        self.assertIn("intro_style", V.choose_variation("science"))
        V.record_variation("science", _entry("science", 0))
        self.assertEqual(json.loads(path.read_text()), [_entry("science", 0)])

    def test_list_with_non_dict_items_keeps_only_dicts(self):
        path = self.tmp / "mixed.json"
        good = _entry("science", 3)
        path.write_text(json.dumps([1, "x", None, good]))
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [good])


class TestParentFolderAndRoundTrip(_HistoryTestCase):
    def test_parent_folder_created_on_write(self):
        path = self.tmp / "queue" / "nested" / "variation_history.json"
        self.assertFalse(path.parent.exists())
        self.set_env_path(path)
        V.record_variation("science", _entry("science", 0))
        self.assertTrue(path.exists())

    def test_write_then_read_round_trips(self):
        path = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(path)
        V.record_variation("science", _entry("science", 0))
        V.record_variation("science", _entry("science", 1))
        self.assertEqual(V._load_history(), [_entry("science", 0), _entry("science", 1)])

    def test_file_is_valid_json_list_after_write(self):
        path = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(path)
        V.record_variation("science", _entry("science", 0))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(loaded, list)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(set(loaded[0]), {
            "niche", "intro_style", "narrative_structure", "outro_treatment", "visual_treatment",
        })

    def test_history_capped_at_50_entries(self):
        path = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(path)
        for i in range(55):
            V.record_variation("science", _entry("science", i))
        loaded = json.loads(path.read_text())
        self.assertEqual(len(loaded), 50)
        self.assertEqual(loaded[-1], _entry("science", 54))


class TestAvoidsRecentChoices(_HistoryTestCase):
    def test_last_three_videos_are_avoided(self):
        """With three recent videos in the same niche, the next choice must
        not reuse the intro/structure/outro options those three videos used,
        since each list has spare options. VISUAL_TREATMENTS has only three
        options, so all of them are recent and the current fallback (pick from
        the full list) applies. Checked over many seeds."""
        recent = [
            {"niche": "science", "intro_style": V.INTRO_STYLES[0],
             "narrative_structure": V.NARRATIVE_STRUCTURES[0],
             "outro_treatment": V.OUTRO_TREATMENTS[0],
             "visual_treatment": V.VISUAL_TREATMENTS[0]},
            {"niche": "science", "intro_style": V.INTRO_STYLES[1],
             "narrative_structure": V.NARRATIVE_STRUCTURES[1],
             "outro_treatment": V.OUTRO_TREATMENTS[1],
             "visual_treatment": V.VISUAL_TREATMENTS[1]},
            {"niche": "science", "intro_style": V.INTRO_STYLES[2],
             "narrative_structure": V.NARRATIVE_STRUCTURES[2],
             "outro_treatment": V.OUTRO_TREATMENTS[2],
             "visual_treatment": V.VISUAL_TREATMENTS[2]},
        ]
        path = self.tmp / "queue" / "variation_history.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(recent))
        self.set_env_path(path)
        for seed in range(200):
            random.seed(seed)
            choice = V.choose_variation("science", lookback=3)
            self.assertNotIn(choice["intro_style"], {h["intro_style"] for h in recent})
            self.assertNotIn(choice["narrative_structure"], {h["narrative_structure"] for h in recent})
            self.assertNotIn(choice["outro_treatment"], {h["outro_treatment"] for h in recent})
            # All three visual options are recent, so any visual option may be picked.
            self.assertIn(choice["visual_treatment"], V.VISUAL_TREATMENTS)

    def test_other_niches_do_not_count(self):
        """Entries for a different niche must not block options for this niche."""
        other = [dict(_entry("cooking", 0), niche="cooking")]
        path = self.tmp / "variation_history.json"
        path.write_text(json.dumps(other))
        self.set_env_path(path)
        random.seed(5)
        choice = V.choose_variation("science", lookback=3)
        # Nothing for "science" is recent, so any option is allowed.
        self.assertIn(choice["intro_style"], V.INTRO_STYLES)

    def test_falls_back_to_full_list_when_all_recently_used(self):
        """If every option was used recently, the full list is used instead of crashing."""
        recent = [
            {"niche": "science", "intro_style": s, "narrative_structure": n,
             "outro_treatment": o, "visual_treatment": v}
            for s, n, o, v in zip(V.INTRO_STYLES, V.NARRATIVE_STRUCTURES,
                                  V.OUTRO_TREATMENTS, V.VISUAL_TREATMENTS * 2)
        ]
        path = self.tmp / "variation_history.json"
        path.write_text(json.dumps(recent))
        self.set_env_path(path)
        random.seed(7)
        choice = V.choose_variation("science", lookback=len(recent))
        self.assertIn(choice["intro_style"], V.INTRO_STYLES)


class TestAtomicWrite(_HistoryTestCase):
    def test_no_temp_file_left_after_good_write(self):
        path = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(path)
        V.record_variation("science", _entry("science", 0))
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["variation_history.json"])

    def test_failed_replace_keeps_existing_valid_history(self):
        path = self.tmp / "queue" / "variation_history.json"
        path.parent.mkdir(parents=True)
        original = json.dumps([_entry("science", 0)], indent=2)
        path.write_text(original)
        self.set_env_path(path)
        with mock.patch.object(V.os, "replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                V.record_variation("science", _entry("science", 1))
        self.assertEqual(path.read_text(), original)
        self.assertEqual(V._load_history(), [_entry("science", 0)])
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["variation_history.json"])

    def test_failed_serialise_keeps_existing_valid_history(self):
        path = self.tmp / "variation_history.json"
        original = json.dumps([_entry("science", 0)], indent=2)
        path.write_text(original)
        self.set_env_path(path)
        with mock.patch.object(V.json, "dumps", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                V.record_variation("science", _entry("science", 1))
        self.assertEqual(path.read_text(), original)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["variation_history.json"])


class TestDefaultPathUnchanged(_HistoryTestCase):
    """With VARIATION_HISTORY_PATH unset, loading returns exactly what HEAD
    returned: the parsed JSON as-is, [] for a missing or unparseable file, and
    no logging. These expected values are HEAD's behaviour, checked line by line
    against HEAD's _load_history."""

    def _write_default(self, content: bytes):
        self.default_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_path.write_bytes(content)

    def test_valid_default_history_loads_and_is_silent(self):
        self._write_default(json.dumps([_entry("science", 0), _entry("science", 1)]).encode())
        self.assertEqual(V._load_history(), [_entry("science", 0), _entry("science", 1)])
        self.assertFalse(self.fake_get_logger.called)

    def test_missing_default_file_returns_empty(self):
        self.assertEqual(V._load_history(), [])
        self.assertFalse(self.fake_get_logger.called)

    def test_corrupt_default_file_returns_empty_without_logging(self):
        self._write_default(b"{not json")
        self.assertEqual(V._load_history(), [])
        self.assertFalse(self.fake_get_logger.called)

    def test_empty_default_file_returns_empty_without_logging(self):
        self._write_default(b"")
        self.assertEqual(V._load_history(), [])
        self.assertFalse(self.fake_get_logger.called)

    def test_wrong_type_dict_default_file_returned_as_is_like_head(self):
        self._write_default(json.dumps({"a": 1}).encode())
        self.assertEqual(V._load_history(), {"a": 1})
        self.assertFalse(self.fake_get_logger.called)

    def test_wrong_type_string_default_file_returned_as_is_like_head(self):
        self._write_default(json.dumps("hello").encode())
        self.assertEqual(V._load_history(), "hello")
        self.assertFalse(self.fake_get_logger.called)

    def test_wrong_type_number_default_file_returned_as_is_like_head(self):
        self._write_default(b"42")
        self.assertEqual(V._load_history(), 42)
        self.assertFalse(self.fake_get_logger.called)

    def test_mixed_list_default_file_returned_unfiltered_like_head(self):
        mixed = [1, "x", None, _entry("science", 3)]
        self._write_default(json.dumps(mixed).encode())
        self.assertEqual(V._load_history(), mixed)
        self.assertFalse(self.fake_get_logger.called)

    def test_env_var_path_still_warns_on_corrupt_file(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json")
        self.set_env_path(path)
        self.assertEqual(V._load_history(), [])
        self.assertTrue(self.fake_logger.warning.called)


class TestDefaultWriteIsPlain(_HistoryTestCase):
    """With VARIATION_HISTORY_PATH unset, writes are the plain write from HEAD:
    no temp file, no os.replace."""

    def test_default_write_is_plain_write_to_default_path(self):
        with mock.patch.object(V.os, "replace") as replace, \
                mock.patch.object(V.Path, "write_text", autospec=True,
                                  side_effect=V.Path.write_text) as write_text:
            V.record_variation("science", _entry("science", 0))
        replace.assert_not_called()
        self.assertEqual(write_text.call_count, 1)
        self.assertEqual(write_text.call_args[0][0], self.default_path)

    def test_default_write_leaves_no_temp_file(self):
        V.record_variation("science", _entry("science", 0))
        V.record_variation("science", _entry("science", 1))
        self.assertEqual(sorted(p.name for p in self.default_path.parent.iterdir()),
                         ["variation_history.json"])
        self.assertEqual(json.loads(self.default_path.read_text()),
                         [_entry("science", 0), _entry("science", 1)])

    def test_default_write_creates_parent_folder(self):
        self.assertFalse(self.default_path.parent.exists())
        V.record_variation("science", _entry("science", 0))
        self.assertTrue(self.default_path.exists())

    def test_blank_env_var_write_is_plain_write(self):
        os.environ[ENV_NAME] = "  "
        with mock.patch.object(V.os, "replace") as replace:
            V.record_variation("science", _entry("science", 0))
        replace.assert_not_called()
        self.assertTrue(self.default_path.exists())


class TestEnvVarWriteIsAtomic(_HistoryTestCase):
    def test_env_var_write_uses_temp_then_replace(self):
        path = self.tmp / "queue" / "variation_history.json"
        self.set_env_path(path)
        with mock.patch.object(V.os, "replace", wraps=V.os.replace) as replace:
            V.record_variation("science", _entry("science", 0))
        replace.assert_called_once()
        self.assertEqual(replace.call_args[0][1], path)
        self.assertEqual(replace.call_args[0][0], path.with_name("variation_history.json.tmp"))

    def test_failed_write_on_env_path_keeps_old_file_and_no_temp(self):
        path = self.tmp / "queue" / "variation_history.json"
        path.parent.mkdir(parents=True)
        original = json.dumps([_entry("science", 0)], indent=2)
        path.write_text(original)
        self.set_env_path(path)
        with mock.patch.object(V.os, "replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                V.record_variation("science", _entry("science", 1))
        self.assertEqual(path.read_text(), original)
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["variation_history.json"])


class TestWorkflowFile(unittest.TestCase):
    """Checks the GitHub Actions workflow still parses and has the intended settings."""

    @classmethod
    def setUpClass(cls):
        with WORKFLOW.open("r", encoding="utf-8") as f:
            cls.data = yaml.safe_load(f)

    def test_parses_as_mapping(self):
        self.assertIsInstance(self.data, dict)

    def test_on_block_still_present(self):
        # PyYAML reads a bare `on:` key as the boolean True, not the string "on".
        self.assertIn(True, self.data)
        self.assertEqual(self.data[True], {"workflow_dispatch": {}})

    def test_concurrency_group_and_cancel_flag(self):
        concurrency = self.data["concurrency"]
        self.assertEqual(concurrency["group"], "produce-queue")
        self.assertIs(concurrency["cancel-in-progress"], False)

    def test_concurrency_is_workflow_level_not_job_level(self):
        self.assertNotIn("concurrency", self.data["jobs"]["produce"])

    def test_variation_env_var_set_in_job_env(self):
        env = self.data["jobs"]["produce"]["env"]
        self.assertEqual(
            env["VARIATION_HISTORY_PATH"],
            "${{ github.workspace }}/queue/variation_history.json",
        )
        self.assertEqual(
            env["VERTICALS_SKILL_DIR"],
            "${{ github.workspace }}/.verticals",
        )

    def test_step_names_unchanged(self):
        names = [s.get("name", s.get("uses")) for s in self.data["jobs"]["produce"]["steps"]]
        self.assertEqual(names, [
            "Check out repo",
            "Set up Python",
            "Install ffmpeg",
            "Install Python dependencies",
            "Restore Google client secret",
            "Restore YouTube OAuth token",
            "Write API keys to config",
            "Run queue orchestrator",
            "Commit queue state changes",
        ])

    def test_commit_step_still_stages_queue(self):
        steps = {s.get("name"): s for s in self.data["jobs"]["produce"]["steps"]}
        commit = steps["Commit queue state changes"]
        self.assertEqual(commit["if"], "always()")
        self.assertIn("git add queue/", commit["run"])
        self.assertIn("git push", commit["run"])


if __name__ == "__main__":
    unittest.main()
