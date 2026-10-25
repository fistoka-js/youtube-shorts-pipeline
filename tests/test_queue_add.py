"""Tests for the add-ticked-topics step: verticals/queue_add.py, its `queue-add` command, and the
workflow .github/workflows/add-topics.yml, plus the label lines added to produce.yml and
suggest-topics.yml.

Written with unittest, like the other test files. Run with:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_queue_add -v

Everything is offline. Every file lives in a temp folder. The workflow tests run the real YAML step
scripts under Git Bash with a fake gh, a fake git and a fake python (the fake python runs the real
`python -m verticals queue-add` in the temp folder), so no GitHub, git push or network call can happen.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from verticals import queue_add as Q
from verticals import suggest as S
from tests.test_variation_history import _bash, _posix_path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADD_TOPICS = REPO_ROOT / ".github" / "workflows" / "add-topics.yml"
PRODUCE = REPO_ROOT / ".github" / "workflows" / "produce.yml"
SUGGEST_WF = REPO_ROOT / ".github" / "workflows" / "suggest-topics.yml"
FOLDERS = ("pending", "in-progress", "done", "failed")


def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _tree(root: Path) -> dict:
    """Every file under root as {relative posix path: bytes}."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _topic_json(topic: str, name_suffix: str = "") -> bytes:
    return json.dumps({"topic": topic, "niche": "science", "platform": "shorts"}).encode("utf-8") + b"\n"


class _TempQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.queue = self.root / "queue"
        for folder in FOLDERS:
            (self.queue / folder).mkdir(parents=True)
        self.manifest = self.root / "manifest.txt"
        self.comment = self.root / "comment.md"

    def tearDown(self):
        self._tmp.cleanup()

    def add(self, body: str):
        return Q.add_topics(body, self.queue, self.manifest, self.comment)

    def pending_names(self):
        return sorted(p.name for p in (self.queue / "pending").iterdir())


# ─────────────────────────────────────────────────────
# Parsing the ticked lines
# ─────────────────────────────────────────────────────
class TestTickedLines(unittest.TestCase):
    def test_x_and_capital_x_are_kept_and_unticked_lines_are_ignored(self):
        body = ("Intro text\n- [ ] Unticked idea | how things work | footage: good\n"
                "- [x] Ticked one | how things work\n- [X] Ticked two | cats\nplain line\n")
        self.assertEqual(Q.ticked_topics(body), ["Ticked one", "Ticked two"])

    def test_line_without_a_pipe_uses_the_whole_rest_of_the_line(self):
        self.assertEqual(Q.ticked_topics("- [x] Why do cats purr?"), ["Why do cats purr?"])

    def test_text_before_the_first_spaced_pipe_is_the_topic(self):
        self.assertEqual(Q.ticked_topics("- [x] a | b | c"), ["a"])

    def test_pipe_without_spaces_is_part_of_the_topic(self):
        self.assertEqual(Q.ticked_topics("- [x] a|b"), ["a|b"])

    def test_backticks_and_shell_looking_text_stay_plain_text(self):
        body = "- [x] `rm -rf /` $(whoami) ${HOME} && echo hi | cat"
        self.assertEqual(Q.ticked_topics(body), ["`rm -rf /` $(whoami) ${HOME} && echo hi"])

    def test_emoji_is_kept(self):
        self.assertEqual(Q.ticked_topics("- [x] Why do cats purr 🐱 | cat"), ["Why do cats purr 🐱"])

    def test_very_long_topic_is_capped_at_120_characters(self):
        topics = Q.ticked_topics("- [x] " + "word " * 80 + "| cat")
        self.assertEqual(len(topics), 1)
        self.assertLessEqual(len(topics[0]), Q.MAX_TOPIC_CHARS)
        self.assertFalse(topics[0].endswith(" "))

    def test_markdown_link_is_kept_as_plain_text(self):
        body = "- [x] [Why do bees dance](https://example.com/bees) | cat"
        self.assertEqual(Q.ticked_topics(body), ["[Why do bees dance](https://example.com/bees)"])

    def test_crlf_body_gives_the_same_topics(self):
        body = "- [x] Topic one | c\r\n- [ ] not this\r\n- [x] Topic two | c\r\n"
        self.assertEqual(Q.ticked_topics(body), ["Topic one", "Topic two"])

    def test_leading_and_trailing_whitespace_on_a_line_is_tolerated(self):
        self.assertEqual(Q.ticked_topics("   - [x]   Spaced topic   | cat   \t"), ["Spaced topic"])

    def test_empty_ticked_topics_are_dropped(self):
        self.assertEqual(Q.ticked_topics("- [x] | cat\n- [x]\n- [x]    \n- [ ] x\n"), [])

    def test_control_characters_become_spaces_and_whitespace_collapses(self):
        self.assertEqual(Q.ticked_topics("- [x] Tab\there\x00zero\x1bescape   end | c"),
                         ["Tab here zero escape end"])

    def test_nothing_ticked_gives_an_empty_list(self):
        self.assertEqual(Q.ticked_topics("hello\n- [ ] nope | c\n"), [])
        self.assertEqual(Q.ticked_topics(""), [])


# ─────────────────────────────────────────────────────
# Slugs and the comparison key
# ─────────────────────────────────────────────────────
class TestSlugs(unittest.TestCase):
    def test_non_ascii_is_folded_to_ascii_words(self):
        self.assertEqual(Q.slugify("Café naïve façade"), "cafe-naive-facade")

    def test_punctuation_and_case_are_removed(self):
        self.assertEqual(Q.slugify("Why do cats purr?!"), "why-do-cats-purr")

    def test_long_slug_is_cut_at_a_hyphen_within_40_characters(self):
        slug = Q.slugify("a" * 30 + " " + "b" * 30)
        self.assertLessEqual(len(slug), Q.MAX_SLUG_CHARS)
        self.assertEqual(slug, "a" * 30)

    def test_long_word_with_no_hyphen_is_cut_at_40(self):
        self.assertEqual(Q.slugify("x" * 50), "x" * 40)

    def test_exactly_40_characters_at_a_hyphen_is_kept_whole(self):
        slug = Q.slugify("a" * 40 + " b")
        self.assertEqual(slug, "a" * 40)

    def test_cut_never_ends_in_a_hyphen(self):
        for topic in ("a" * 38 + " bb cc", "a" * 37 + " bb cc dd", "one two three four five six seven eight nine ten eleven"):
            slug = Q.slugify(topic)
            self.assertLessEqual(len(slug), Q.MAX_SLUG_CHARS)
            self.assertFalse(slug.endswith("-"), slug)
            self.assertFalse(slug.startswith("-"), slug)

    def test_empty_slug_becomes_topic(self):
        self.assertEqual(Q.slugify("???"), "topic")
        self.assertEqual(Q.slugify("🐱🐱"), "topic")
        self.assertEqual(Q.slugify(""), "topic")

    def test_key_matches_the_suggester_comparison_key(self):
        samples = ["Why do cats purr?", "WHY do cats   purr!!!", "Café — naïve", "a|b @c #d", "", "🐱 cat"]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(Q._key(text), S._key(text))


# ─────────────────────────────────────────────────────
# Duplicates
# ─────────────────────────────────────────────────────
class TestDuplicates(_TempQueue):
    def test_a_topic_in_each_queue_folder_is_a_duplicate(self):
        _write(self.queue / "pending" / "001-a.json", _topic_json("Topic A"))
        _write(self.queue / "in-progress" / "002-b.json", _topic_json("Topic B"))
        _write(self.queue / "done" / "003-c.json", _topic_json("Topic C"))
        _write(self.queue / "failed" / "004-d.json", _topic_json("Topic D"))
        body = "- [x] Topic A | c\n- [x] Topic B | c\n- [x] Topic C | c\n- [x] Topic D | c\n- [x] Topic E | c\n"
        result = self.add(body)
        self.assertEqual(result["skipped"], ["Topic A", "Topic B", "Topic C", "Topic D"])
        self.assertEqual([t for t, _ in result["queued"]], ["Topic E"])

    def test_case_and_punctuation_are_ignored(self):
        _write(self.queue / "pending" / "001-cats.json", _topic_json("Why do cats purr"))
        result = self.add("- [x] WHY do cats purr!!! | c\n")
        self.assertEqual(result["skipped"], ["WHY do cats purr!!!"])
        self.assertEqual(result["queued"], [])

    def test_failed_wrapper_file_without_topic_is_not_a_duplicate(self):
        _write(self.queue / "failed" / "016-x.json",
               json.dumps({"error": "bad", "original_text": "Some text"}).encode("utf-8"))
        result = self.add("- [x] Some text | c\n")
        self.assertEqual(result["skipped"], [])
        self.assertEqual([t for t, _ in result["queued"]], ["Some text"])

    def test_repeat_within_the_same_issue_is_skipped_once(self):
        result = self.add("- [x] Same topic | c\n- [X] same TOPIC! | c\n")
        self.assertEqual([t for t, _ in result["queued"]], ["Same topic"])
        self.assertEqual(result["skipped"], ["same TOPIC!"])
        self.assertEqual(len(self.pending_names()), 1)

    def test_queue_topics_in_suggest_and_here_agree(self):
        _write(self.queue / "pending" / "001-a.json", _topic_json("Alpha topic"))
        _write(self.queue / "done" / "002-b.json", _topic_json("Beta topic?"))
        _write(self.queue / "failed" / "003-c.json", b"not json")
        expected = {S._key(t) for t in S.queue_topics(self.queue)}
        self.assertEqual(Q.existing_keys(self.queue), expected)


# ─────────────────────────────────────────────────────
# Numbering and file names
# ─────────────────────────────────────────────────────
class TestNumbering(_TempQueue):
    def test_empty_queue_starts_at_001(self):
        self.add("- [x] First | c\n")
        self.assertEqual(self.pending_names(), ["001-first.json"])

    def test_mixed_names_continue_after_the_highest_first_integer(self):
        _write(self.queue / "pending" / "test15.json", _topic_json("Chameleon"))
        _write(self.queue / "failed" / "016-foo.json", _topic_json("Foo"))
        _write(self.queue / "done" / "test2.json", _topic_json("Two"))
        _write(self.queue / "in-progress" / ".gitkeep", b"")
        self.assertEqual(Q.next_number(self.queue), 17)
        self.add("- [x] New one | c\n")
        self.assertIn("017-new-one.json", self.pending_names())

    def test_gaps_are_not_filled(self):
        _write(self.queue / "pending" / "003-a.json", _topic_json("A"))
        _write(self.queue / "done" / "010-b.json", _topic_json("B"))
        self.add("- [x] Gap | c\n")
        self.assertEqual(self.pending_names(), ["003-a.json", "011-gap.json"])

    def test_numbers_are_zero_padded_and_1000_stays_1000(self):
        _write(self.queue / "pending" / "999-a.json", _topic_json("A"))
        self.add("- [x] Next | c\n")
        self.assertIn("1000-next.json", self.pending_names())

    def test_one_number_per_new_file(self):
        self.add("- [x] One | c\n- [x] Two | c\n")
        self.assertEqual(self.pending_names(), ["001-one.json", "002-two.json"])

    def test_names_with_no_integer_count_as_nothing(self):
        _write(self.queue / "pending" / "notes.json", _topic_json("Notes"))
        self.assertEqual(Q.next_number(self.queue), 1)

    def test_a_date_named_file_moves_the_next_number_to_its_year(self):
        # Documented behaviour of the rule "highest first integer + 1": 2026-10-21-... counts as 2026.
        _write(self.queue / "pending" / "2026-10-21-yawn.json", _topic_json("Yawn"))
        self.assertEqual(Q.next_number(self.queue), 2027)

    def test_an_existing_name_is_never_overwritten(self):
        pending = self.queue / "pending"
        _write(pending / "005-x.json", b"OLD CONTENT\n")
        number, name = Q._create_file(pending, 5, "x", b"NEW\n")
        self.assertEqual((number, name), (6, "006-x.json"))
        self.assertEqual((pending / "005-x.json").read_bytes(), b"OLD CONTENT\n")
        self.assertEqual((pending / "006-x.json").read_bytes(), b"NEW\n")
        self.assertEqual(sorted(p.name for p in pending.iterdir()), ["005-x.json", "006-x.json"])

    def test_no_temp_file_is_left_behind(self):
        self.add("- [x] Clean | c\n- [x] Clean two | c\n")
        self.assertEqual(sorted(p.name for p in (self.queue / "pending").iterdir()),
                         ["001-clean.json", "002-clean-two.json"])


# ─────────────────────────────────────────────────────
# What gets written
# ─────────────────────────────────────────────────────
class TestWriting(_TempQueue):
    def test_file_content_is_exact_json_and_one_newline_without_bom(self):
        topic = 'Café "quoted" 🐱 \\ back'
        self.add(f"- [x] {topic} | c\n")
        raw = (self.queue / "pending" / "001-cafe-quoted-back.json").read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(raw, json.dumps({"topic": topic, "niche": "science", "platform": "shorts"}).encode("utf-8") + b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(json.loads(raw.decode("utf-8")), {"topic": topic, "niche": "science", "platform": "shorts"})

    def test_nothing_ticked_creates_no_file_and_an_empty_manifest(self):
        result = self.add("no ticks here\n- [ ] nope | c\n")
        self.assertEqual(result["queued"], [])
        self.assertEqual(self.pending_names(), [])
        self.assertEqual(self.manifest.read_bytes(), b"")
        text = self.comment.read_text(encoding="utf-8")
        self.assertIn("No topics were ticked", text)

    def test_everything_duplicate_says_so_in_the_comment(self):
        _write(self.queue / "pending" / "001-a.json", _topic_json("Topic A"))
        self.add("- [x] Topic A | c\n")
        self.assertEqual(self.pending_names(), ["001-a.json"])
        self.assertEqual(self.manifest.read_bytes(), b"")
        self.assertIn("Every ticked topic is already in the queue", self.comment.read_text(encoding="utf-8"))

    def test_manifest_lists_each_new_file_as_a_relative_path(self):
        _write(self.queue / "pending" / "test15.json", _topic_json("Chameleon"))
        self.add("- [x] One | c\n- [x] Two | c\n")
        # The in-process call uses a temp queue folder, so only the end of each path is fixed here;
        # the command-line test checks the exact relative form the workflow uses.
        lines = self.manifest.read_text(encoding="utf-8").splitlines()
        self.assertEqual([line.rsplit("/pending/", 1)[1] for line in lines], ["016-one.json", "017-two.json"])
        self.assertTrue(all(line.endswith(".json") and "/pending/" in line for line in lines))

    def test_files_are_only_ever_created_inside_queue_pending(self):
        _write(self.queue / "pending" / "001-a.json", _topic_json("A"))
        _write(self.queue / "done" / "002-b.json", _topic_json("B"))
        before = _tree(self.queue)
        self.add("- [x] A | c\n- [x] C one | c\n- [x] D two | c\n")
        after = _tree(self.queue)
        for path, data in before.items():
            self.assertEqual(after[path], data, path)
        new_paths = set(after) - set(before)
        # The highest number in the queue is 002 (done/002-b.json), so the new files are 003 and 004.
        self.assertEqual(new_paths, {"pending/003-c-one.json", "pending/004-d-two.json"})

    def test_comment_has_no_mentions_or_issue_references(self):
        self.add("- [x] Ask @octocat why #5 closes #5 | c\n")
        text = self.comment.read_text(encoding="utf-8")
        self.assertNotIn("@", text)
        self.assertNotIn("#", text)
        self.assertNotIn("closes #", text.lower())
        self.assertIn("octocat", text)

    def test_comment_escapes_markdown_characters(self):
        self.add("- [x] What [is] *this*? | c\n")
        text = self.comment.read_text(encoding="utf-8")
        self.assertIn(r"What \[is\] \*this\*?", text)

    def test_comment_lists_queued_and_skipped_with_file_names(self):
        _write(self.queue / "pending" / "001-a.json", _topic_json("Topic A"))
        self.add("- [x] Topic A | c\n- [x] Topic B | c\n")
        text = self.comment.read_text(encoding="utf-8")
        self.assertIn("Added 1 topic(s) to the queue", text)
        self.assertIn("`002-topic-b.json`", text)
        self.assertIn("Skipped 1 duplicate(s)", text)
        self.assertIn("- Topic A", text)



# ─────────────────────────────────────────────────────
# The command
# ─────────────────────────────────────────────────────
class TestCommand(_TempQueue):
    def _run_cli(self, args, env_extra=None, body_name="ISSUE_BODY", body=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["VERTICALS_SKILL_DIR"] = str(self.root / "skill")
        env.pop(body_name, None)
        if body is not None:
            env[body_name] = body
        env.update(env_extra or {})
        return subprocess.run([sys.executable, "-m", "verticals", "queue-add", *args], cwd=self.root,
                              env=env, capture_output=True, text=True, timeout=120)

    def test_cli_reads_the_body_from_the_named_environment_variable(self):
        result = self._run_cli(["--body-env", "MY_BODY", "--manifest", "m.txt", "--comment-out", "c.md"],
                               body_name="MY_BODY", body="- [x] From the env | cat\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 topic(s) queued", result.stdout)
        self.assertEqual((self.root / "m.txt").read_text(encoding="utf-8").splitlines(),
                         ["queue/pending/001-from-the-env.json"])
        self.assertEqual(json.loads((self.root / "queue" / "pending" / "001-from-the-env.json").read_text(encoding="utf-8")),
                         {"topic": "From the env", "niche": "science", "platform": "shorts"})

    def test_cli_nothing_ticked_exits_0_with_empty_manifest(self):
        result = self._run_cli(["--body-env", "ISSUE_BODY", "--manifest", "m.txt", "--comment-out", "c.md"],
                               body="nothing ticked\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "m.txt").read_bytes(), b"")
        self.assertIn("0 topic(s) queued", result.stdout)

    def test_cli_missing_environment_variable_exits_1(self):
        result = self._run_cli(["--body-env", "NOT_SET_ANYWHERE", "--manifest", "m.txt", "--comment-out", "c.md"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("NOT_SET_ANYWHERE is not set", result.stderr)

    def test_cli_refuses_a_body_on_the_command_line(self):
        result = self._run_cli(["--body", "- [x] from argv | c", "--manifest", "m.txt", "--comment-out", "c.md"])
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "queue" / "pending" / "001-from-argv.json").exists())

    def test_cli_runs_with_pyyaml_and_requests_unavailable(self):
        # The add-topics workflow installs no packages, so the command must not need PyYAML or requests.
        code = ("import sys, runpy\n"
                "sys.modules['yaml'] = None\nsys.modules['requests'] = None\n"
                "sys.argv = ['verticals', 'queue-add', '--body-env', 'ISSUE_BODY', "
                "'--manifest', 'm.txt', '--comment-out', 'c.md']\n"
                "runpy.run_module('verticals', run_name='__main__')\n")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["VERTICALS_SKILL_DIR"] = str(self.root / "skill")
        env["ISSUE_BODY"] = "- [x] Works without packages | c\n"
        result = subprocess.run([sys.executable, "-c", code], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "m.txt").read_text(encoding="utf-8").splitlines(),
                         ["queue/pending/001-works-without-packages.json"])

    def test_unwritable_queue_folder_is_a_real_error_with_exit_1(self):
        (self.queue / "pending").rmdir()
        (self.queue / "pending").write_text("a file, not a folder", encoding="utf-8")
        with mock.patch.dict(os.environ, {"ISSUE_BODY": "- [x] Blocked | c\n"}):
            code = Q.main(["--body-env", "ISSUE_BODY", "--manifest", str(self.manifest),
                           "--comment-out", str(self.comment), "--queue-dir", str(self.queue)])
        self.assertEqual(code, 1)

    def test_main_entry_point_dispatches_queue_add_before_the_setup_wizard(self):
        from verticals import __main__ as M
        argv = ["verticals", "queue-add", "--body-env", "ISSUE_BODY",
                "--manifest", str(self.manifest), "--comment-out", str(self.comment),
                "--queue-dir", str(self.queue)]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"ISSUE_BODY": "- [x] Dispatched | c\n"}), \
                mock.patch.object(M, "maybe_run_setup", side_effect=AssertionError("setup must not run")):
            M.main()  # returns normally on success
        self.assertEqual(self.pending_names(), ["001-dispatched.json"])

    def test_main_entry_point_exits_1_on_a_real_error(self):
        from verticals import __main__ as M
        (self.queue / "pending").rmdir()
        (self.queue / "pending").write_text("blocker", encoding="utf-8")
        argv = ["verticals", "queue-add", "--body-env", "ISSUE_BODY",
                "--manifest", str(self.manifest), "--comment-out", str(self.comment),
                "--queue-dir", str(self.queue)]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {"ISSUE_BODY": "- [x] X | c\n"}):
            with self.assertRaises(SystemExit) as cm:
                M.main()
        self.assertEqual(cm.exception.code, 1)


# ─────────────────────────────────────────────────────
# Workflow text
# ─────────────────────────────────────────────────────
def _load(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestAddTopicsWorkflowText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = ADD_TOPICS.read_text(encoding="utf-8")
        cls.data = _load(ADD_TOPICS)
        cls.job = cls.data["jobs"]["add-topics"]
        cls.steps = cls.job["steps"]
        cls.by_name = {s["name"]: s for s in cls.steps}

    def test_name_and_trigger_is_issues_labeled_only(self):
        self.assertEqual(self.data["name"], "Add ticked topics to the queue")
        self.assertEqual(self.data[True], {"issues": {"types": ["labeled"]}})

    def test_job_if_has_all_five_guards(self):
        cond = self.job["if"]
        self.assertIn("github.event.label.name == 'add-to-queue'", cond)
        self.assertIn("contains(github.event.issue.labels.*.name, 'topic-suggestions')", cond)
        self.assertIn("github.event.issue.user.login == 'github-actions[bot]'", cond)
        self.assertIn("github.actor == github.repository_owner", cond)
        self.assertIn("github.event.issue.state == 'open'", cond)
        self.assertEqual(cond.count("&&"), 4)

    def test_permissions_are_exactly_contents_and_issues_write(self):
        self.assertEqual(self.data["permissions"], {"contents": "write", "issues": "write"})

    def test_same_concurrency_group_as_produce(self):
        produce = _load(PRODUCE)
        self.assertEqual(self.data["concurrency"], produce["concurrency"])
        self.assertEqual(self.data["concurrency"], {"group": "produce-queue", "cancel-in-progress": False})

    def test_timeout_is_set(self):
        self.assertEqual(self.job["timeout-minutes"], 10)

    def test_pinned_action_versions_equal_produce(self):
        produce = _load(PRODUCE)
        produce_uses = sorted(s["uses"] for s in produce["jobs"]["produce"]["steps"] if "uses" in s)
        here_uses = sorted(s["uses"] for s in self.steps if "uses" in s)
        self.assertEqual(here_uses, produce_uses)
        self.assertEqual(here_uses, ["actions/checkout@v7", "actions/setup-python@v7"])

    def test_checkout_uses_main_and_python_is_312(self):
        self.assertEqual(self.by_name["Check out repo"]["with"]["ref"], "main")
        self.assertEqual(self.by_name["Set up Python"]["with"]["python-version"], "3.12")

    def test_no_pip_install_and_no_third_party_action(self):
        self.assertNotRegex(self.text, r"pip\s+install")
        for step in self.steps:
            if "uses" in step:
                self.assertTrue(step["uses"].startswith("actions/"), step["uses"])

    def test_body_comes_only_through_env_and_no_expression_is_in_any_script(self):
        self.assertEqual(self.by_name["Add the ticked topics to the queue"]["env"],
                         {"ISSUE_BODY": "${{ github.event.issue.body }}"})
        self.assertEqual(self.job["env"], {"ISSUE_NUMBER": "${{ github.event.issue.number }}"})
        for step in self.steps:
            if "run" in step:
                self.assertNotIn("${{", step["run"], step["name"])
        for step in self.steps:
            for key, value in (step.get("env") or {}).items():
                if "${{" in str(value):
                    self.assertIn(value, ("${{ github.event.issue.body }}", "${{ github.event.issue.number }}",
                                          "${{ github.token }}", "${{ github.repository }}"), key)

    def test_gh_steps_get_token_and_repo_from_env(self):
        for name in ("Comment on the issue and close it", "Comment on the issue and remove the label when nothing was queued"):
            env = self.by_name[name]["env"]
            self.assertEqual(env["GH_TOKEN"], "${{ github.token }}")
            self.assertEqual(env["GH_REPO"], "${{ github.repository }}")

    def test_python_step_runs_queue_add_with_env_name_and_file_outputs(self):
        run = self.by_name["Add the ticked topics to the queue"]["run"]
        self.assertIn("python -m verticals queue-add --body-env ISSUE_BODY", run)
        self.assertIn('--manifest "$RUNNER_TEMP/add-topics-manifest.txt"', run)
        self.assertIn('--comment-out "$RUNNER_TEMP/add-topics-comment.md"', run)

    def test_git_adds_only_named_files_and_never_everything(self):
        run = self.by_name["Commit and push the new queue files"]["run"]
        self.assertNotRegex(run, r"git add\s+(-A|--all|\.|-u)(\s|$)")
        self.assertIsNone(re.search(r"^\s*git add\s*$", run, re.MULTILINE))
        self.assertIn('git add -- "$file"', run)
        self.assertIn("while IFS= read -r file", run)

    def test_commit_message_and_push_order(self):
        run = self.by_name["Commit and push the new queue files"]["run"]
        self.assertIn('git commit -m "Queue: add $count topics from suggestion issue #$ISSUE_NUMBER"', run)
        self.assertIn("git pull --rebase origin main", run)
        self.assertLess(run.index("git commit"), run.index("git pull --rebase origin main"))
        self.assertLess(run.index("git pull --rebase origin main"), run.index("git push"))
        self.assertEqual(run.count("push_queue()"), 1)

    def test_comment_uses_body_file_and_close_follows_it(self):
        run = self.by_name["Comment on the issue and close it"]["run"]
        self.assertIn('gh issue comment "$ISSUE_NUMBER" --body-file "$RUNNER_TEMP/add-topics-comment.md"', run)
        self.assertIn('gh issue close "$ISSUE_NUMBER"', run)
        self.assertLess(run.index("gh issue comment"), run.index("gh issue close"))
        for line in self.text.splitlines():
            if "gh issue comment" in line:
                self.assertIn("--body-file", line)

    def test_nothing_queued_path_removes_the_label(self):
        run = self.by_name["Comment on the issue and remove the label when nothing was queued"]["run"]
        self.assertIn('gh issue edit "$ISSUE_NUMBER" --remove-label add-to-queue', run)
        self.assertNotIn("gh issue close", run)
        self.assertEqual(self.by_name["Comment on the issue and remove the label when nothing was queued"]["if"],
                         "steps.push.outputs.queued == 'false'")
        self.assertEqual(self.by_name["Comment on the issue and close it"]["if"],
                         "steps.push.outputs.queued == 'true'")

    def test_every_run_script_passes_bash_syntax_check(self):
        bash = _bash()
        if bash is None:
            self.skipTest("no usable bash (Git Bash on Windows) found")
        for step in self.steps:
            if "run" not in step:
                continue
            with self.subTest(step=step["name"]), tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "step.sh"
                script.write_text(step["run"], encoding="utf-8", newline="\n")
                result = subprocess.run([bash, "-n", _posix_path(script)], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)


class TestLabelLinesInOtherWorkflows(unittest.TestCase):
    def test_produce_warn_step_creates_add_to_queue_next_to_topic_suggestions(self):
        step = _load(PRODUCE)["jobs"]["produce"]["steps"]
        warn = next(s for s in step if s.get("name") == "Warn if the queue is low")
        run = warn["run"]
        self.assertIn("gh label create add-to-queue --description \"Add the ticked topics to the queue\" --color 1D76DB > /dev/null 2>&1",
                      run)
        self.assertLess(run.index("gh label create topic-suggestions"), run.index("gh label create add-to-queue"))
        self.assertLess(run.index("gh label create add-to-queue"), run.index("gh issue create --title \"$TITLE\" --body-file"))
        self.assertEqual(warn["if"], "always()")
        self.assertIs(warn["continue-on-error"], True)

    def test_suggest_topics_creates_add_to_queue_label_and_never_fails(self):
        steps = _load(SUGGEST_WF)["jobs"]["suggest"]["steps"]
        label = next(s for s in steps if s.get("name") == "Ensure the topic-suggestions label exists")
        self.assertIs(label["continue-on-error"], True)
        self.assertIn("gh label create add-to-queue --description \"Add the ticked topics to the queue\" --color 1D76DB > /dev/null 2>&1",
                      label["run"])
        self.assertIn("Created label add-to-queue.", label["run"])


# ─────────────────────────────────────────────────────
# Workflow behaviour: the real YAML step scripts under Git Bash
# ─────────────────────────────────────────────────────
_EXPR = re.compile(r"\$\{\{\s*([^}]+?)\s*\}\}")
_CONDITIONS = {
    "steps.push.outputs.queued == 'true'": lambda outputs: outputs.get("queued") == "true",
    "steps.push.outputs.queued == 'false'": lambda outputs: outputs.get("queued") == "false",
}

FAKE_GIT = r'''#!/bin/bash
echo "git $*" >> "$FAKE_DIR/calls.log"
if [ "$1" = "push" ]; then
  n=$(cat "$FAKE_DIR/push_count" 2>/dev/null || echo 0)
  n=$((n + 1)); echo "$n" > "$FAKE_DIR/push_count"
  if [ "$n" -le "${FAKE_GIT_PUSH_FAILS:-0}" ]; then echo "fake git: push rejected" >&2; exit 1; fi
fi
exit 0
'''

FAKE_GH = r'''#!/bin/bash
echo "gh $*" >> "$FAKE_DIR/calls.log"
case "$1 $2" in
  "issue comment")
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--body-file" ]; then cp "$a" "$FAKE_DIR/last_comment.md"; fi
      prev="$a"
    done
    if [ -n "$FAKE_GH_COMMENT_FAIL" ]; then echo "fake gh: comment failed" >&2; exit 1; fi
    echo "https://github.com/jacob-test/youtube-shorts-pipeline/issues/7#issuecomment-1"
    ;;
  "issue close")
    if [ -n "$FAKE_GH_CLOSE_FAIL" ]; then echo "fake gh: close failed" >&2; exit 1; fi
    echo "fake gh: closed $3"
    ;;
  "issue edit")
    if [ -n "$FAKE_GH_EDIT_FAIL" ]; then echo "fake gh: edit failed" >&2; exit 1; fi
    echo "fake gh: edited $3"
    ;;
  *)
    echo "fake gh: unexpected call: $*" >&2; exit 1
    ;;
esac
'''

# The fake python logs the call, then runs the REAL interpreter (with the project on PYTHONPATH).
FAKE_PYTHON_TEMPLATE = '''#!/bin/bash
echo "python $*" >> "$FAKE_DIR/calls.log"
exec "{real}" "$@"
'''


class TestAddTopicsStepBehaviour(unittest.TestCase):
    """Runs the real step scripts from the YAML, in order, the way the runner does (bash -e, pipefail),
    with fake gh, fake git and a fake python that runs the real queue-add command."""

    @classmethod
    def setUpClass(cls):
        if _bash() is None:
            raise unittest.SkipTest("no usable bash (Git Bash on Windows) found; workflow behaviour tests skipped")
        cls.data = _load(ADD_TOPICS)
        cls.steps = cls.data["jobs"]["add-topics"]["steps"]
        cls.job_env = cls.data["jobs"]["add-topics"]["env"]

    def _run_job(self, body, push_fails=0, gh_fail=None, pending=None):
        """Run the job's steps. Returns a dict with the step names that ran, the exit code of the
        first failing step (0 when none failed), the outputs, the gh/git/python calls and the comment."""
        gh_fail = gh_fail or {}
        values = {
            "github.event.issue.body": body,
            "github.event.issue.number": "7",
            "github.token": "fake-token-for-tests",
            "github.repository": "jacob-test/youtube-shorts-pipeline",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "bin"
            fake.mkdir()
            work = root / "work"
            for folder in FOLDERS:
                (work / "queue" / folder).mkdir(parents=True)
            for folder, names in (pending or {}).items():
                for name, topic in names:
                    _write(work / "queue" / folder / name, _topic_json(topic))
            (work / "runner_tmp").mkdir()
            (work / "github_output.txt").write_bytes(b"")
            real_python = _posix_path(sys.executable)
            files = {"git": FAKE_GIT, "gh": FAKE_GH, "python": FAKE_PYTHON_TEMPLATE.format(real=real_python)}
            for name, text in files.items():
                path = fake / name
                path.write_text(text, encoding="utf-8", newline="\n")
                path.chmod(0o755)
            fake_posix = _posix_path(fake)
            env_base = dict(os.environ)
            env_base.update({
                "PYTHONPATH": str(REPO_ROOT),
                "VERTICALS_SKILL_DIR": str(root / "skill"),
                "RUNNER_TEMP": "runner_tmp",
                "GITHUB_OUTPUT": "github_output.txt",
                "FAKE_DIR": fake_posix,
                "FAKE_GIT_PUSH_FAILS": str(push_fails),
            })
            for key in ("FAKE_GH_COMMENT_FAIL", "FAKE_GH_CLOSE_FAIL", "FAKE_GH_EDIT_FAIL"):
                env_base.pop(key, None)
            for key in gh_fail:
                env_base[key] = "1"
            for key in ("ISSUE_BODY", "ISSUE_NUMBER", "GH_TOKEN", "GH_REPO"):
                env_base.pop(key, None)

            outputs = {}
            ran, skipped = [], []
            failed_rc = 0
            for index, step in enumerate(self.steps):
                name = step["name"]
                if "uses" in step:
                    ran.append(name)  # checkout and setup-python are not run here
                    continue
                cond = step.get("if")
                if cond is not None:
                    if cond not in _CONDITIONS:
                        raise AssertionError(f"the harness does not know the condition {cond!r}")
                    if not _CONDITIONS[cond](outputs):
                        skipped.append(name)
                        continue
                env = dict(env_base)
                for source in (self.job_env, step.get("env") or {}):
                    for key, value in source.items():
                        env[key] = _EXPR.sub(lambda m: values[m.group(1)], str(value))
                script = root / f"step_{index}.sh"
                harness = (f'export FAKE_DIR="{fake_posix}"\n'
                           f'export PATH="{fake_posix}:$PATH"\n'
                           f'cd "{_posix_path(work)}"\n' + step["run"])
                script.write_text(harness, encoding="utf-8", newline="\n")
                result = subprocess.run([_bash(), "--noprofile", "--norc", "-eo", "pipefail", _posix_path(script)],
                                        cwd=str(work), env=env, capture_output=True, text=True, timeout=300)
                ran.append(name)
                output_file = work / "github_output.txt"
                for line in output_file.read_text(encoding="utf-8").splitlines():
                    if "=" in line:
                        key, _, value = line.partition("=")
                        outputs[key] = value
                if result.returncode != 0 and not step.get("continue-on-error"):
                    failed_rc = result.returncode
                    failed_rc = failed_rc or 1
                    last_output = result.stdout + result.stderr
                    break
            else:
                last_output = ""
            calls = (root / "bin" / "calls.log")
            call_lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
            comment_file = fake / "last_comment.md"
            comment = comment_file.read_text(encoding="utf-8") if comment_file.exists() else None
            manifest = (work / "runner_tmp" / "add-topics-manifest.txt").read_text(encoding="utf-8") \
                if (work / "runner_tmp" / "add-topics-manifest.txt").exists() else None
            return {
                "ran": ran, "skipped": skipped, "rc": failed_rc, "outputs": outputs,
                "calls": call_lines, "comment": comment, "manifest": manifest, "log": last_output,
                "queue_pending": sorted(p.name for p in (work / "queue" / "pending").iterdir()),
            }

    BODY_TWO = ("Some preface\n- [x] Why do bees dance | how things work | footage: good\n"
                "- [ ] Not this one | cat\n- [X] Why is the sky blue? | cat | footage: thin\n")

    def test_a_ticked_topic_is_queued_pushed_commented_and_closed_in_order(self):
        result = self._run_job(self.BODY_TWO, pending={"pending": [("test15.json", "Chameleon colours")],
                                                       "done": [("016-foo.json", "Something else")]})
        self.assertEqual(result["rc"], 0, result["log"])
        self.assertEqual(result["queue_pending"], ["017-why-do-bees-dance.json", "018-why-is-the-sky-blue.json", "test15.json"])
        self.assertEqual(result["manifest"].splitlines(),
                         ["queue/pending/017-why-do-bees-dance.json", "queue/pending/018-why-is-the-sky-blue.json"])
        interesting = [c for c in result["calls"] if c.startswith(("git add", "git commit", "git pull", "git push",
                                                                      "gh issue", "python -m verticals queue-add"))]
        self.assertEqual(interesting, [
            "python -m verticals queue-add --body-env ISSUE_BODY --manifest runner_tmp/add-topics-manifest.txt "
            "--comment-out runner_tmp/add-topics-comment.md",
            "git add -- queue/pending/017-why-do-bees-dance.json",
            "git add -- queue/pending/018-why-is-the-sky-blue.json",
            "git commit -m Queue: add 2 topics from suggestion issue #7",
            "git pull --rebase origin main",
            "git push",
            "gh issue comment 7 --body-file runner_tmp/add-topics-comment.md",
            "gh issue close 7",
        ])
        self.assertIn("Why do bees dance", result["comment"])
        self.assertIn("`017-why-do-bees-dance.json`", result["comment"])
        self.assertEqual(result["outputs"].get("queued"), "true")

    def test_nothing_ticked_removes_the_label_and_never_commits_or_closes(self):
        result = self._run_job("No ticks at all.\n- [ ] still not | cat\n")
        self.assertEqual(result["rc"], 0, result["log"])
        self.assertEqual(result["queue_pending"], [])
        self.assertEqual(result["manifest"], "")
        self.assertEqual(result["outputs"].get("queued"), "false")
        self.assertFalse(any(c.startswith(("git commit", "git push", "git pull", "git add")) for c in result["calls"]),
                         result["calls"])
        self.assertFalse(any(c.startswith("gh issue close") for c in result["calls"]))
        self.assertIn("gh issue comment 7 --body-file runner_tmp/add-topics-comment.md", result["calls"])
        self.assertIn("gh issue edit 7 --remove-label add-to-queue", result["calls"])
        self.assertIn("No topics were ticked", result["comment"])
        self.assertIn("Comment on the issue and close it", result["skipped"])
        self.assertIn("Comment on the issue and remove the label when nothing was queued", result["ran"])

    def test_everything_duplicate_removes_the_label_and_never_commits(self):
        result = self._run_job("- [x] Topic A | c\n", pending={"pending": [("001-a.json", "Topic A")]})
        self.assertEqual(result["rc"], 0, result["log"])
        self.assertEqual(result["queue_pending"], ["001-a.json"])
        self.assertFalse(any(c.startswith(("git commit", "git push")) for c in result["calls"]))
        self.assertIn("gh issue edit 7 --remove-label add-to-queue", result["calls"])
        self.assertIn("Every ticked topic is already in the queue", result["comment"])

    def test_push_that_fails_twice_goes_red_with_no_comment_and_no_close(self):
        result = self._run_job(self.BODY_TWO, push_fails=2)
        self.assertNotEqual(result["rc"], 0)
        pushes = [c for c in result["calls"] if c == "git push"]
        pulls = [c for c in result["calls"] if c == "git pull --rebase origin main"]
        self.assertEqual(len(pushes), 2)
        self.assertEqual(len(pulls), 2)
        self.assertFalse(any(c.startswith("gh ") for c in result["calls"]), result["calls"])
        self.assertIn("Could not push the new queue files after two tries", result["log"])
        self.assertNotIn("Comment on the issue and close it", result["ran"])
        self.assertNotIn("Comment on the issue and remove the label when nothing was queued", result["ran"])

    def test_push_that_fails_once_is_retried_and_then_reported(self):
        result = self._run_job(self.BODY_TWO, push_fails=1)
        self.assertEqual(result["rc"], 0, result["log"])
        self.assertEqual(sum(1 for c in result["calls"] if c == "git push"), 2)
        self.assertEqual(sum(1 for c in result["calls"] if c == "git pull --rebase origin main"), 2)
        self.assertIn("gh issue close 7", result["calls"])

    def test_failed_comment_after_a_good_push_turns_the_run_red_and_keeps_the_issue_open(self):
        result = self._run_job(self.BODY_TWO, gh_fail={"FAKE_GH_COMMENT_FAIL": 1})
        self.assertNotEqual(result["rc"], 0)
        self.assertIn("git push", result["calls"])
        self.assertFalse(any(c.startswith("gh issue close") for c in result["calls"]))

    def test_failed_close_after_a_good_push_turns_the_run_red(self):
        result = self._run_job(self.BODY_TWO, gh_fail={"FAKE_GH_CLOSE_FAIL": 1})
        self.assertNotEqual(result["rc"], 0)
        self.assertIn("gh issue comment 7 --body-file runner_tmp/add-topics-comment.md", result["calls"])
        self.assertIn("gh issue close 7", result["calls"])

    def test_nothing_ticked_with_a_failing_label_edit_turns_the_run_red(self):
        result = self._run_job("nothing", gh_fail={"FAKE_GH_EDIT_FAIL": 1})
        self.assertNotEqual(result["rc"], 0)
        self.assertIn("gh issue comment 7 --body-file runner_tmp/add-topics-comment.md", result["calls"])


if __name__ == "__main__":
    unittest.main()
