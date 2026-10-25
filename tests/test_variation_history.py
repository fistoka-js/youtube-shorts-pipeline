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
import re
import shlex
import shutil
import subprocess
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
        options, so all of them are recent and the fallback applies: the visual
        choice must not equal the most recent video's visual treatment.
        Checked over many seeds."""
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
            # All three visual options are recent, so the fallback applies and
            # the choice must differ from the most recent video's visual treatment.
            self.assertNotEqual(choice["visual_treatment"], recent[-1]["visual_treatment"])

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

    def test_falls_back_to_all_but_most_recent_when_all_recently_used(self):
        """If every option was used recently, each field falls back to everything
        except the most recent video's value, and nothing crashes. Checked over many seeds."""
        recent = [
            {"niche": "science", "intro_style": s, "narrative_structure": n,
             "outro_treatment": o, "visual_treatment": v}
            for s, n, o, v in zip(V.INTRO_STYLES, V.NARRATIVE_STRUCTURES,
                                  V.OUTRO_TREATMENTS, V.VISUAL_TREATMENTS * 2)
        ]
        path = self.tmp / "variation_history.json"
        path.write_text(json.dumps(recent))
        self.set_env_path(path)
        for seed in range(200):
            random.seed(seed)
            choice = V.choose_variation("science", lookback=len(recent))
            for field in ("intro_style", "narrative_structure", "outro_treatment", "visual_treatment"):
                self.assertNotEqual(choice[field], recent[-1][field])

    def test_visual_never_repeats_most_recent_when_all_three_recent(self):
        """With all three visual options in the recent history, the visual choice
        never equals the most recent video's, and both other options do come up."""
        recent = [
            {"niche": "science", "intro_style": V.INTRO_STYLES[0],
             "narrative_structure": V.NARRATIVE_STRUCTURES[0],
             "outro_treatment": V.OUTRO_TREATMENTS[0],
             "visual_treatment": v}
            for v in (V.VISUAL_TREATMENTS[1], V.VISUAL_TREATMENTS[2], V.VISUAL_TREATMENTS[0])
        ]
        path = self.tmp / "variation_history.json"
        path.write_text(json.dumps(recent))
        self.set_env_path(path)
        seen = set()
        for seed in range(200):
            random.seed(seed)
            choice = V.choose_variation("science", lookback=3)
            self.assertNotEqual(choice["visual_treatment"], V.VISUAL_TREATMENTS[0])
            seen.add(choice["visual_treatment"])
        self.assertEqual(seen, {V.VISUAL_TREATMENTS[1], V.VISUAL_TREATMENTS[2]})

    def test_one_option_pool_uses_that_option(self):
        """If a pool has only one option and it was used recently, that option is
        used instead of crashing (excluding it would leave nothing)."""
        only = "only visual option"
        recent = [{"niche": "science", "intro_style": V.INTRO_STYLES[0],
                   "narrative_structure": V.NARRATIVE_STRUCTURES[0],
                   "outro_treatment": V.OUTRO_TREATMENTS[0],
                   "visual_treatment": only}]
        path = self.tmp / "variation_history.json"
        path.write_text(json.dumps(recent))
        self.set_env_path(path)
        random.seed(11)
        with mock.patch.object(V, "VISUAL_TREATMENTS", [only]):
            choice = V.choose_variation("science", lookback=3)
        self.assertEqual(choice["visual_treatment"], only)


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

    def test_top_level_permissions_allow_contents_and_issues_write(self):
        # contents: write pushes the queue commit; issues: write opens or closes the low-queue issue.
        self.assertEqual(self.data["permissions"], {"contents": "write", "issues": "write"})

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
            "Warn if the queue is low",
        ])

    def test_commit_step_still_stages_queue(self):
        steps = {s.get("name"): s for s in self.data["jobs"]["produce"]["steps"]}
        commit = steps["Commit queue state changes"]
        self.assertEqual(commit["if"], "always()")
        self.assertIn("git add queue/", commit["run"])
        self.assertIn("git push", commit["run"])

    def test_checkout_uses_branch_ref(self):
        steps = {s.get("name"): s for s in self.data["jobs"]["produce"]["steps"]}
        checkout = steps["Check out repo"]
        self.assertEqual(checkout["with"]["ref"], "${{ github.ref_name }}")

    def test_orchestrator_sets_git_identity_before_run_py(self):
        # run.py commits a queue claim before uploading, so the identity must be set first.
        steps = {s.get("name"): s for s in self.data["jobs"]["produce"]["steps"]}
        lines = [line.strip() for line in steps["Run queue orchestrator"]["run"].splitlines()]
        name_line = 'git config user.name "github-actions[bot]"'
        email_line = 'git config user.email "github-actions[bot]@users.noreply.github.com"'
        self.assertIn(name_line, lines)
        self.assertIn(email_line, lines)
        self.assertIn("python run.py", lines)
        self.assertLess(lines.index(name_line), lines.index("python run.py"))
        self.assertLess(lines.index(email_line), lines.index("python run.py"))

    def test_commit_step_pulls_rebase_between_commit_and_push(self):
        steps = {s.get("name"): s for s in self.data["jobs"]["produce"]["steps"]}
        lines = [line.strip() for line in steps["Commit queue state changes"]["run"].splitlines()]
        pull = "git pull --rebase origin ${{ github.ref_name }}"
        self.assertIn(pull, lines)
        commit_index = next(i for i, line in enumerate(lines) if "git commit" in line)
        self.assertLess(commit_index, lines.index(pull))
        self.assertLess(lines.index(pull), lines.index("git push"))

    def _ffmpeg_run(self):
        steps = {s.get("name"): s for s in self.data["jobs"]["produce"]["steps"]}
        return steps["Install ffmpeg"]["run"]

    def test_ffmpeg_step_retry_loop_timeout_and_exits(self):
        run = self._ffmpeg_run()
        self.assertRegex(run, r"for attempt in (1 2 3|\$\(seq 1 3\)|\{1\.\.3\})")
        self.assertIn("sudo timeout --kill-after=15 240 bash -c 'apt-get update && apt-get install -y ffmpeg'", run)
        self.assertIn('echo "ffmpeg install: succeeded on attempt $attempt"', run)
        self.assertIn("::error::ffmpeg install failed after 3 attempts", run)

    def test_ffmpeg_step_mirror_switch_text_targets_and_flags(self):
        # What the sed really does is checked by TestMirrorSwitchSed; where it runs by TestFfmpegStepBehaviour.
        run = self._ffmpeg_run()
        sed = re.search(r"sudo sed -i [^\n]*\|\| true", run)
        self.assertIsNotNone(sed)
        line = sed.group(0)
        self.assertIn(r"mirror+file:/etc/apt/apt-mirrors\.txt", line)
        self.assertIn(r"azure\.archive\.ubuntu\.com", line)
        self.assertIn("https://archive.ubuntu.com/ubuntu/", line)
        self.assertRegex(line, r" /etc/apt/sources\.list /etc/apt/sources\.list\.d/\*\.sources \|\| true$")
        self.assertEqual(len(re.findall(r"s\|[^|']*\|[^|']*\|g", line)), 2)

    def test_ffmpeg_step_messages_and_guarded_repairs(self):
        run = self._ffmpeg_run()
        self.assertIn("ffmpeg install: attempt 1 of 3", run)
        self.assertIn("failed or timed out", run)
        self.assertRegex(run, r"sudo timeout --kill-after=15 60 dpkg --configure -a \|\| true")

    def test_ffmpeg_step_does_not_skip_recommends(self):
        self.assertNotIn("--no-install-recommends", self._ffmpeg_run())

    def test_produce_job_timeout_is_45_minutes(self):
        self.assertEqual(self.data["jobs"]["produce"]["timeout-minutes"], 45)

    def test_warn_step_is_last_after_push_and_never_fails_run(self):
        steps = self.data["jobs"]["produce"]["steps"]
        self.assertEqual(steps[-2]["name"], "Commit queue state changes")
        warn = steps[-1]
        self.assertEqual(warn["name"], "Warn if the queue is low")
        self.assertEqual(warn["if"], "always()")
        self.assertIs(warn["continue-on-error"], True)
        self.assertEqual(warn["env"]["GH_TOKEN"], "${{ github.token }}")
        self.assertEqual(warn["env"]["GH_REPO"], "${{ github.repository }}")
        self.assertEqual(warn["env"]["OWNER"], "${{ github.repository_owner }}")
        # The script body takes repository values only through env:, never pasted in.
        self.assertNotIn("${{", warn["run"])


def _find_bash():
    """Return a Git Bash (or Linux bash) that can run the step, or None. Never the WSL launcher."""
    if os.name != "nt":
        candidates = [shutil.which("bash")]
    else:
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
        if os.environ.get("LOCALAPPDATA"):
            candidates.append(os.path.join(os.environ["LOCALAPPDATA"], "Programs", "Git", "bin", "bash.exe"))
        git = shutil.which("git")  # e.g. ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
        if git:
            candidates.append(os.path.join(os.path.dirname(os.path.dirname(git)), "bin", "bash.exe"))
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate) or "system32" in candidate.lower():
            continue
        try:
            result = subprocess.run([candidate, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and "ok" in result.stdout:
            return candidate
    return None


_BASH_FOUND = []


def _bash():
    """Find bash the first time a test needs it, not at import time."""
    if not _BASH_FOUND:
        _BASH_FOUND.append(_find_bash())
    return _BASH_FOUND[0]


# Fake commands. Each one prints a tagged line so the test can see what the step really did.
# Fake sudo just runs its arguments, so the fake sed/dpkg/timeout/sleep are what actually run.
# Env switches: APT_FAIL_CALLS (apt update fails on the first N calls), SED_FAIL and DPKG_FAIL
# (those fail when set), TIMEOUT_RC with TIMEOUT_RC_CALLS (the first N apt timeouts return that code).
FAKE_COMMANDS = {
    # The fake sudo marks the command it runs as root.
    "sudo": 'FAKE_SUDO=1 exec "$@"\n',
    # The timeout fake reports its kill-after and duration, and marks commands it wraps as bounded.
    "timeout": (
        'KA=""\n'
        'while [ "${1#--}" != "$1" ]; do\n'
        '  case "$1" in --kill-after=*) KA="${1#--kill-after=}";; esac\n'
        "  shift\n"
        "done\n"
        'SECS="$1"; shift\n'
        'echo "[fake timeout] kill=$KA secs=$SECS cmd=$1"\n'
        'case "$*" in *apt-get*) if [ -n "$TIMEOUT_RC" ]; then\n'
        '  n=$(cat "$FAKE_DIR/timeout_count" 2>/dev/null || echo 0)\n'
        '  n=$((n+1)); echo "$n" > "$FAKE_DIR/timeout_count"\n'
        '  if [ "$n" -le "${TIMEOUT_RC_CALLS:-0}" ]; then echo "[fake timeout] exit $TIMEOUT_RC"; exit "$TIMEOUT_RC"; fi\n'
        "fi;; esac\n"
        'FAKE_BOUNDED=1 exec "$@"\n'
    ),
    # apt-get refuses to run unless started through the fake sudo, and it accepts only the approved
    # `install -y ffmpeg`. Every call reports whether it ran under the fake timeout (bounded=yes or no).
    "apt-get": (
        'if [ -z "$FAKE_SUDO" ]; then echo "[fake apt-get] $1 not root"; exit 1; fi\n'
        'echo "[fake apt-get] $1 bounded=${FAKE_BOUNDED:-no}"\n'
        'if [ "$1" = update ]; then\n'
        '  n=$(cat "$FAKE_DIR/apt_count" 2>/dev/null || echo 0)\n'
        '  n=$((n+1)); echo "$n" > "$FAKE_DIR/apt_count"\n'
        '  if [ "$n" -le "$APT_FAIL_CALLS" ]; then echo "[fake apt-get] update failed"; exit 1; fi\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$*" != "install -y ffmpeg" ]; then echo "[fake apt-get] install args rejected: $*"; exit 1; fi\n'
        "exit 0\n"
    ),
    "dpkg": 'echo "[fake dpkg] $*"\nif [ -n "$DPKG_FAIL" ]; then exit 1; fi\nexit 0\n',
    "sed": 'echo "[fake sed] $*"\nif [ -n "$SED_FAIL" ]; then exit 1; fi\nexit 0\n',
    "sleep": 'echo "[fake sleep] $*"\nexit 0\n',
}


def _posix_path(path):
    """Turn a Windows path into the /c/... form Git Bash uses inside scripts."""
    text = str(path)
    if os.name != "nt":
        return text
    return "/" + text[0].lower() + text[2:].replace("\\", "/")


class TestFfmpegStepBehaviour(unittest.TestCase):
    """Runs the real Install ffmpeg step under bash -e with fake commands and checks what happened."""

    @classmethod
    def setUpClass(cls):
        if _bash() is None:
            raise unittest.SkipTest("no usable bash (Git Bash on Windows) found; behaviour tests skipped")
        with WORKFLOW.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        steps = {s.get("name"): s for s in data["jobs"]["produce"]["steps"]}
        cls.step_text = steps["Install ffmpeg"]["run"]

    def _run_step(self, apt_fail_calls=0, **extra_env):
        bash = _bash()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dir = root / "bin"
            fake_dir.mkdir()
            for name, body in FAKE_COMMANDS.items():
                with (fake_dir / name).open("w", encoding="utf-8", newline="\n") as f:
                    f.write("#!/bin/bash\n" + body)
            fake_posix = _posix_path(fake_dir)
            harness = (
                f'export FAKE_DIR="{fake_posix}"\n'
                f'export PATH="{fake_posix}:$PATH"\n'
                f'chmod +x "{fake_posix}"/*\n'
                + self.step_text
            )
            harness_file = root / "step.sh"
            with harness_file.open("w", encoding="utf-8", newline="\n") as f:
                f.write(harness)
            env = dict(os.environ, APT_FAIL_CALLS=str(apt_fail_calls))
            for key in ("SED_FAIL", "DPKG_FAIL", "TIMEOUT_RC", "TIMEOUT_RC_CALLS", "FAKE_SUDO", "FAKE_BOUNDED"):
                env.pop(key, None)
            env.update({key: str(value) for key, value in extra_env.items()})
            return subprocess.run(
                [bash, "-e", _posix_path(harness_file)],
                capture_output=True, text=True, timeout=120, env=env,
            )

    @staticmethod
    def _events(output):
        """The order of the step's messages and fake commands, reduced to short tags."""
        events = []
        for line in output.splitlines():
            if re.search(r"ffmpeg install: attempt \d of 3", line):
                events.append("start" + re.search(r"attempt (\d) of 3", line).group(1))
            elif re.search(r"ffmpeg install: attempt \d failed", line):
                events.append("failed" + re.search(r"attempt (\d) failed", line).group(1))
            elif re.search(r"ffmpeg install: succeeded on attempt \d", line):
                events.append("succeeded" + re.search(r"attempt (\d)", line).group(1))
            elif "::error::ffmpeg install failed after 3 attempts" in line:
                events.append("error")
            elif "[fake apt-get]" in line and "not root" in line:
                events.append("NOT_ROOT")
            elif "[fake apt-get]" in line and "bounded=no" in line:
                events.append("UNBOUNDED")
            elif "[fake sed]" in line:
                events.append("sed")
            elif "[fake dpkg]" in line:
                events.append("dpkg")
            elif "[fake sleep]" in line:
                events.append("sleep")
        return events

    @staticmethod
    def _tagged(output, tag):
        """The text after each '[fake <tag>] ' marker, in order (e.g. the sleep durations)."""
        marker = f"[fake {tag}] "
        return [line.split(marker, 1)[1] for line in output.splitlines() if marker in line]

    def test_first_try_success_never_switches_mirror_or_sleeps(self):
        result = self._run_step(apt_fail_calls=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._events(result.stdout), ["start1", "succeeded1"])
        self.assertEqual(self._tagged(result.stdout, "timeout"), ["kill=15 secs=240 cmd=bash"])
        self.assertEqual(self._tagged(result.stdout, "sleep"), [])

    def test_two_failures_then_success_switches_on_attempts_two_and_three_only(self):
        result = self._run_step(apt_fail_calls=2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._events(result.stdout), [
            "start1", "failed1", "sleep",
            "start2", "dpkg", "sed", "failed2", "sleep",
            "start3", "dpkg", "sed", "succeeded3",
        ])
        self.assertEqual(self._tagged(result.stdout, "timeout"), [
            "kill=15 secs=240 cmd=bash",
            "kill=15 secs=60 cmd=dpkg", "kill=15 secs=240 cmd=bash",
            "kill=15 secs=60 cmd=dpkg", "kill=15 secs=240 cmd=bash",
        ])
        self.assertEqual(self._tagged(result.stdout, "sleep"), ["10", "10"])

    def test_every_attempt_failing_exits_1_and_does_not_sleep_after_the_last(self):
        result = self._run_step(apt_fail_calls=99)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self._events(result.stdout), [
            "start1", "failed1", "sleep",
            "start2", "dpkg", "sed", "failed2", "sleep",
            "start3", "dpkg", "sed", "failed3", "error",
        ])
        self.assertEqual(self._tagged(result.stdout, "sleep"), ["10", "10"])

    def test_failing_sed_and_dpkg_are_ignored_under_bash_e(self):
        result = self._run_step(apt_fail_calls=2, SED_FAIL=1, DPKG_FAIL=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._events(result.stdout), [
            "start1", "failed1", "sleep",
            "start2", "dpkg", "sed", "failed2", "sleep",
            "start3", "dpkg", "sed", "succeeded3",
        ])

    def test_timeout_exit_codes_124_and_137_count_as_failed_attempts(self):
        for code in (124, 137):
            with self.subTest(code=code):
                result = self._run_step(TIMEOUT_RC=code, TIMEOUT_RC_CALLS=1)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self._events(result.stdout), [
                    "start1", "failed1", "sleep",
                    "start2", "dpkg", "sed", "succeeded2",
                ])
                result = self._run_step(TIMEOUT_RC=code, TIMEOUT_RC_CALLS=99)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(self._events(result.stdout)[-1], "error")


class TestMirrorSwitchSed(unittest.TestCase):
    """Runs the real mirror-switch sed from the step on sample apt source files (no fakes)."""

    DEB822_BEFORE = (
        "# mirrors: mirror+file:/etc/apt/apt-mirrors.txt mirror+file:/etc/apt/apt-mirrors.txt\n"
        "Types: deb\nURIs: mirror+file:/etc/apt/apt-mirrors.txt\nSuites: noble noble-updates\nComponents: main\n\n"
        "Types: deb\nURIs: http://azure.archive.ubuntu.com/ubuntu/\nSuites: noble-backports\n\n"
        "Types: deb\nURIs: https://security.ubuntu.com/ubuntu/\nSuites: noble-security\n"
    )
    DEB822_AFTER = (
        "# mirrors: https://archive.ubuntu.com/ubuntu/ https://archive.ubuntu.com/ubuntu/\n"
        "Types: deb\nURIs: https://archive.ubuntu.com/ubuntu/\nSuites: noble noble-updates\nComponents: main\n\n"
        "Types: deb\nURIs: https://archive.ubuntu.com/ubuntu/\nSuites: noble-backports\n\n"
        "Types: deb\nURIs: https://security.ubuntu.com/ubuntu/\nSuites: noble-security\n"
    )
    LIST_BEFORE = (
        "# mirrors: http://azure.archive.ubuntu.com/ubuntu/ http://azure.archive.ubuntu.com/ubuntu/\n"
        "deb http://azure.archive.ubuntu.com/ubuntu/ noble main\n"
    )
    LIST_AFTER = (
        "# mirrors: https://archive.ubuntu.com/ubuntu/ https://archive.ubuntu.com/ubuntu/\n"
        "deb https://archive.ubuntu.com/ubuntu/ noble main\n"
    )

    @classmethod
    def setUpClass(cls):
        if _bash() is None:
            raise unittest.SkipTest("no usable bash (Git Bash on Windows) found; sed tests skipped")
        with WORKFLOW.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        steps = {s.get("name"): s for s in data["jobs"]["produce"]["steps"]}
        sed = re.search(r"sudo sed -i [^\n]*\|\| true", steps["Install ffmpeg"]["run"])
        assert sed is not None, "mirror-switch sed line not found in the Install ffmpeg step"
        cls.sed_line = sed.group(0)

    def test_real_sed_rewrites_both_layouts_and_nothing_else_and_is_repeatable(self):
        # The folder name has a space on purpose, so the test also covers paths like "John Smith".
        with tempfile.TemporaryDirectory(prefix="mirror sed ") as tmp:
            root = Path(tmp)
            (root / "ubuntu.sources").write_text(self.DEB822_BEFORE, encoding="utf-8", newline="\n")
            (root / "sources.list").write_text(self.LIST_BEFORE, encoding="utf-8", newline="\n")
            # Same command as the step, with the real apt paths pointed at the sample files.
            cmd = self.sed_line[len("sudo "):].replace(" || true", "")
            posix_root = shlex.quote(_posix_path(root))
            cmd = cmd.replace("/etc/apt/sources.list.d/*.sources", f"{posix_root}/*.sources")
            cmd = cmd.replace(" /etc/apt/sources.list ", f" {posix_root}/sources.list ")
            self.assertNotIn("/etc/apt/sources", cmd)
            for run_number in (1, 2):
                with self.subTest(run=run_number):
                    result = subprocess.run([_bash(), "-c", cmd], capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual((root / "ubuntu.sources").read_text(encoding="utf-8"), self.DEB822_AFTER)
                    self.assertEqual((root / "sources.list").read_text(encoding="utf-8"), self.LIST_AFTER)


# Fake gh: logs every call to gh_calls.log. It prints the open-issue number that the step's --jq
# filter would print (FAKE_GH_LIST), because real gh applies --jq itself and there is no jq here.
# Fake gh. Every call is logged to gh_calls.log. "issue list" with --label topic-suggestions returns
# FAKE_GH_PAST (the past suggestion bodies); any other list returns FAKE_GH_LIST (open issue numbers).
# Switches: FAKE_GH_LIST_FAIL, FAKE_GH_PAST_FAIL, FAKE_GH_CREATE_FAIL_ASSIGNEE, FAKE_GH_CREATE_FAIL_LABEL,
# FAKE_GH_LABEL_FAIL. When "issue create" gets --body-file, a copy of that file is kept in last_body.md.
FAKE_GH = r'''#!/bin/bash
echo "$*" >> "$FAKE_DIR/gh_calls.log"
case "$1 $2" in
  "issue list")
    if [ -n "$FAKE_GH_LIST_FAIL" ]; then echo "fake gh: list failed" >&2; exit 1; fi
    case "$*" in
      *"--label topic-suggestions"*)
        if [ -n "$FAKE_GH_PAST_FAIL" ]; then echo "fake gh: past list failed" >&2; exit 1; fi
        printf '%s' "$FAKE_GH_PAST"
        ;;
      *)
        printf '%s' "$FAKE_GH_LIST"
        ;;
    esac
    ;;
  "issue create")
    if [ -n "$FAKE_GH_CREATE_FAIL_ASSIGNEE" ] && [[ "$*" == *--assignee* ]]; then
      echo "fake gh: assignee rejected" >&2; exit 1
    fi
    if [ -n "$FAKE_GH_CREATE_FAIL_LABEL" ] && [[ "$*" == *--label* ]]; then
      echo "fake gh: label rejected" >&2; exit 1
    fi
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--body-file" ]; then cp "$a" "$FAKE_DIR/last_body.md"; fi
      prev="$a"
    done
    echo "https://github.com/jacob/youtube-shorts-pipeline/issues/99"
    ;;
  "issue close")
    echo "fake gh: closed $3"
    ;;
  "label create")
    if [ -n "$FAKE_GH_LABEL_FAIL" ]; then echo "fake gh: label already exists" >&2; exit 1; fi
    echo "fake gh: label created"
    ;;
  *)
    echo "fake gh: unexpected call: $*" >&2; exit 1
    ;;
esac
'''

# Fake python for the suggest step. Only "python -m verticals suggest ..." is accepted; anything else
# fails loudly, so a real python (and a real LLM call) can never run from these tests. It logs its
# arguments, keeps a copy of the --exclude-file as exclude_seen.txt, and writes FAKE_PY_CONTENT to the
# --out file only when FAKE_PY_SUGGEST is "ok". Otherwise it fails and writes nothing.
FAKE_PYTHON = r'''#!/bin/bash
echo "$*" >> "$FAKE_DIR/python_calls.log"
if [ "$1" != "-m" ] || [ "$2" != "verticals" ] || [ "$3" != "suggest" ]; then
  echo "fake python: unexpected call: $*" >&2; exit 1
fi
shift 3
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--exclude-file" ]; then cp "$a" "$FAKE_DIR/exclude_seen.txt"; fi
  if [ "$prev" = "--out" ]; then out="$a"; fi
  prev="$a"
done
if [ "$FAKE_PY_SUGGEST" != "ok" ]; then echo "fake python: suggest failed" >&2; exit 1; fi
printf '%s\n' "$FAKE_PY_CONTENT" > "$out"
exit 0
'''


class TestQueueWarnStepBehaviour(unittest.TestCase):
    """Runs the real 'Warn if the queue is low' script under bash -e with a fake gh command."""

    @classmethod
    def setUpClass(cls):
        if _bash() is None:
            raise unittest.SkipTest("no usable bash (Git Bash on Windows) found; queue warning tests skipped")
        with WORKFLOW.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        steps = {s.get("name"): s for s in data["jobs"]["produce"]["steps"]}
        cls.step_text = steps["Warn if the queue is low"]["run"]

    # Environment switches the fakes read. Each run starts with all of them unset.
    FAKE_SWITCHES = (
        "FAKE_GH_LIST_FAIL", "FAKE_GH_CREATE_FAIL_ASSIGNEE", "FAKE_GH_PAST", "FAKE_GH_PAST_FAIL",
        "FAKE_GH_CREATE_FAIL_LABEL", "FAKE_GH_LABEL_FAIL", "FAKE_PY_SUGGEST", "FAKE_PY_CONTENT",
    )

    def _run_step(self, files=(), open_issue="", **extra_env):
        """Returns (result, gh call lines). Files are created in queue/pending/ of a temp folder.
        Also sets self.artifacts: the body file gh was given, the exclusion file python was given,
        and the python call log (each None when it was not written)."""
        bash = _bash()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dir = root / "bin"
            fake_dir.mkdir()
            (fake_dir / "gh").write_text(FAKE_GH, encoding="utf-8", newline="\n")
            (fake_dir / "python").write_text(FAKE_PYTHON, encoding="utf-8", newline="\n")
            work = root / "work"
            pending = work / "queue" / "pending"
            pending.mkdir(parents=True)
            for name in files:
                (pending / name).write_text('{"topic": "made up"}\n', encoding="utf-8", newline="\n")
            fake_posix = _posix_path(fake_dir)
            harness = (
                f'export FAKE_DIR="{fake_posix}"\n'
                f'export PATH="{fake_posix}:$PATH"\n'
                f'chmod +x "{fake_posix}/gh" "{fake_posix}/python"\n'
                f'cd "{_posix_path(work)}"\n'
                + self.step_text
            )
            harness_file = root / "step.sh"
            with harness_file.open("w", encoding="utf-8", newline="\n") as f:
                f.write(harness)
            env = dict(os.environ)
            for key in self.FAKE_SWITCHES:
                env.pop(key, None)
            env.update({
                "OWNER": "jacob-test",
                "GH_REPO": "jacob-test/youtube-shorts-pipeline",
                "GH_TOKEN": "fake-token-for-tests",
                "FAKE_GH_LIST": open_issue,
            })
            env.update({key: str(value) for key, value in extra_env.items()})
            result = subprocess.run(
                [bash, "-e", _posix_path(harness_file)],
                capture_output=True, text=True, timeout=120, env=env,
            )
            log = fake_dir / "gh_calls.log"
            calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []

            def _read(name):
                path = fake_dir / name
                return path.read_text(encoding="utf-8") if path.exists() else None

            self.artifacts = {
                "body": _read("last_body.md"),
                "exclude": _read("exclude_seen.txt"),
                "python_calls": _read("python_calls.log"),
            }
            return result, calls

    @staticmethod
    def _creates(calls):
        return [c for c in calls if c.startswith("issue create")]

    @staticmethod
    def _closes(calls):
        return [c for c in calls if c.startswith("issue close")]

    def test_two_files_and_no_open_issue_creates_issue_assigned_to_owner(self):
        result, calls = self._run_step(files=["001-a.json", "002-b.json"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Queue check: 2 topic(s) pending", result.stdout)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 1, calls)
        self.assertIn("--title Queue low: add topics", creates[0])
        self.assertIn("--assignee jacob-test", creates[0])
        self.assertIn("README.md#adding-topics-and-the-low-queue-issue", creates[0])
        self.assertEqual(self._closes(calls), [])

    def test_list_call_matches_title_exactly_and_uses_limit_100(self):
        # A new issue is about to be created, so the step also lists past topic-suggestion issues.
        _, calls = self._run_step(files=["001-a.json"])
        lists = [c for c in calls if c.startswith("issue list")]
        self.assertEqual(len(lists), 2, calls)
        self.assertIn("--state open", lists[0])
        self.assertIn("--limit 100", lists[0])
        self.assertIn('select(.title == "Queue low: add topics")', lists[0])
        self.assertIn("--label topic-suggestions", lists[1])
        self.assertIn("--state all", lists[1])
        self.assertIn("--limit 100", lists[1])
        self.assertIn("--json body", lists[1])

    def test_three_files_and_open_issue_closes_it(self):
        result, calls = self._run_step(files=["001-a.json", "002-b.json", "003-c.json"], open_issue="42")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._creates(calls), [])
        closes = self._closes(calls)
        self.assertEqual(len(closes), 1, calls)
        self.assertTrue(closes[0].startswith("issue close 42 --comment"), closes[0])
        self.assertIn("Closed issue #42", result.stdout)

    def test_two_files_and_open_issue_does_nothing(self):
        result, calls = self._run_step(files=["001-a.json", "002-b.json"], open_issue="42")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._creates(calls), [])
        self.assertEqual(self._closes(calls), [])
        self.assertIn("Left alone: issue #42", result.stdout)

    def test_gitkeep_is_not_counted(self):
        # Two topics plus .gitkeep: counting .gitkeep would make 3 and close nothing, but the
        # step must count 2 and open the issue.
        result, calls = self._run_step(files=["001-a.json", "002-b.json", ".gitkeep"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Queue check: 2 topic(s) pending", result.stdout)
        self.assertEqual(len(self._creates(calls)), 1, calls)

    def test_failed_create_with_assignee_retries_without_assignee(self):
        result, calls = self._run_step(files=["001-a.json"], FAKE_GH_CREATE_FAIL_ASSIGNEE="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 2, calls)
        self.assertIn("--assignee jacob-test", creates[0])
        self.assertNotIn("--assignee", creates[1])
        self.assertIn("--title Queue low: add topics", creates[1])
        self.assertIn("Created issue", result.stdout)

    def test_failed_issue_list_creates_and_closes_nothing(self):
        result, calls = self._run_step(files=["001-a.json"], FAKE_GH_LIST_FAIL="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._creates(calls), [])
        self.assertEqual(self._closes(calls), [])
        self.assertIn("Could not list open issues", result.stdout)

    # ── Topic ideas in the new "Queue low" issue (the suggest step) ──────────────

    PAST_BODIES = (
        "Topic ideas from a past run.\n\n"
        "- [x] old topic | how things work | footage: good\n"
        "- [ ] Another old one? | what happens when scenarios | footage: thin\n"
        "not a checklist line\n"
    )
    IDEAS_CONTENT = (
        "Tick the ideas you want by changing [ ] to [x].\n\n"
        "- [ ] why do cats purr | animal and human behavior curiosities | footage: good\n"
        "- [ ] how do bridges sway | how things work | footage: thin\n"
    )

    def test_new_issue_with_ideas_uses_body_file_label_and_assignee(self):
        result, calls = self._run_step(
            files=["001-a.json"], FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
            FAKE_GH_PAST=self.PAST_BODIES,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 1, calls)
        self.assertIn("--body-file", creates[0])
        self.assertIn("--label topic-suggestions", creates[0])
        self.assertIn("--assignee jacob-test", creates[0])
        self.assertIn("--title Queue low: add topics", creates[0])
        self.assertNotIn(" --body ", creates[0])
        body = self.artifacts["body"]
        self.assertIsNotNone(body, "gh issue create was not given a body file")
        self.assertIn("Only 1 topic(s) are waiting in queue/pending/", body)
        self.assertIn("README.md#adding-topics-and-the-low-queue-issue", body)
        self.assertIn("- [ ] why do cats purr | animal and human behavior curiosities | footage: good", body)
        self.assertIn("- [ ] how do bridges sway | how things work | footage: thin", body)
        self.assertIn("Created issue", result.stdout)

    def test_past_suggestion_topics_are_extracted_for_the_exclusion_file(self):
        result, calls = self._run_step(
            files=["001-a.json"], FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
            FAKE_GH_PAST=self.PAST_BODIES,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exclude = self.artifacts["exclude"]
        self.assertIsNotNone(exclude, "suggest was not given an exclusion file")
        # Ticked and unticked checklist lines give their topic; other lines are ignored.
        self.assertEqual([line.strip() for line in exclude.splitlines()], ["old topic", "Another old one?"])
        python_calls = self.artifacts["python_calls"]
        self.assertIn("-m verticals suggest --exclude-file", python_calls)
        self.assertIn("--out suggestions.md", python_calls)
        lists = [c for c in calls if c.startswith("issue list")]
        self.assertIn("--label topic-suggestions", lists[-1])

    def test_suggest_failure_creates_the_plain_warning_issue(self):
        result, calls = self._run_step(files=["001-a.json"])  # FAKE_PY_SUGGEST unset: suggest fails
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("did not produce ideas", result.stdout)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 1, calls)
        self.assertIn("--assignee jacob-test", creates[0])
        self.assertIn(" --body ", creates[0])
        self.assertIn("Only 1 topic(s) are waiting in queue/pending/", creates[0])
        self.assertNotIn("--label", creates[0])
        self.assertNotIn("--body-file", creates[0])
        self.assertFalse(any(c.startswith("label create") for c in calls), calls)

    def test_open_issue_does_not_run_suggest_or_create(self):
        result, calls = self._run_step(
            files=["001-a.json"], open_issue="42", FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIsNone(self.artifacts["python_calls"], "suggest must not run when an issue is open")
        self.assertEqual(self._creates(calls), [])
        self.assertIn("Left alone: issue #42", result.stdout)

    def test_label_failure_still_creates_the_issue_with_ideas(self):
        result, calls = self._run_step(
            files=["001-a.json"], FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
            FAKE_GH_LABEL_FAIL="1",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("was not created", result.stdout)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 1, calls)
        self.assertIn("--label topic-suggestions", creates[0])
        self.assertIn("why do cats purr", self.artifacts["body"])

    def test_failed_create_with_assignee_retries_with_ideas_without_assignee(self):
        result, calls = self._run_step(
            files=["001-a.json"], FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
            FAKE_GH_CREATE_FAIL_ASSIGNEE="1",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 2, calls)
        self.assertIn("--assignee jacob-test", creates[0])
        self.assertIn("--body-file", creates[0])
        self.assertNotIn("--assignee", creates[1])
        self.assertIn("--body-file", creates[1])
        self.assertIn("--label topic-suggestions", creates[1])
        self.assertIn("without an assignee", result.stdout)

    def test_label_rejected_falls_back_to_the_plain_warning_issue(self):
        result, calls = self._run_step(
            files=["001-a.json"], FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
            FAKE_GH_CREATE_FAIL_LABEL="1",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        creates = self._creates(calls)
        # Two attempts with the label (with and without assignee) both fail; then the plain issue.
        self.assertEqual(len(creates), 3, calls)
        self.assertIn("--label topic-suggestions", creates[0])
        self.assertIn("--label topic-suggestions", creates[1])
        self.assertNotIn("--label", creates[2])
        self.assertIn("--assignee jacob-test", creates[2])
        self.assertIn(" --body ", creates[2])
        self.assertIn("Creating the issue with topic ideas failed", result.stdout)

    def test_failed_past_list_suggests_with_an_empty_exclusion_file(self):
        result, calls = self._run_step(
            files=["001-a.json"], FAKE_PY_SUGGEST="ok", FAKE_PY_CONTENT=self.IDEAS_CONTENT,
            FAKE_GH_PAST_FAIL="1",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.artifacts["exclude"], "")
        self.assertIn("Could not list past topic-suggestion issues", result.stdout)
        creates = self._creates(calls)
        self.assertEqual(len(creates), 1, calls)
        self.assertIn("--label topic-suggestions", creates[0])


if __name__ == "__main__":
    unittest.main()
