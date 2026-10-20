"""Tests for the cloud install pins (constraints-ci.txt), the workflow pip step,
and the pinned GitHub Actions versions.

Written with unittest, like tests/test_variation_history.py. Run with:
    .\\venv\\Scripts\\python.exe -m unittest tests.test_ci_pins -v

Reads only project files. Nothing is installed, downloaded or run in the cloud.
"""

import re
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
CONSTRAINTS = REPO_ROOT / "constraints-ci.txt"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "produce.yml"

PIN_LINE = re.compile(r"^[A-Za-z0-9._-]+==[A-Za-z0-9._+-]+$")


def _normalize(name):
    """PEP 503 name normalization: lowercase, runs of - _ . become a single -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names():
    names = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        names.append(re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0])
    return names


def _constraint_lines():
    return CONSTRAINTS.read_text(encoding="utf-8").splitlines()


class TestConstraintsCoverRequirements(unittest.TestCase):
    def test_every_requirement_has_a_pin(self):
        pinned = {_normalize(line.split("==")[0]) for line in _constraint_lines() if line}
        missing = [n for n in _requirement_names() if _normalize(n) not in pinned]
        self.assertEqual(missing, [], "requirements.txt packages with no pin in constraints-ci.txt")

    def test_requirements_list_is_not_empty(self):
        # Guards the coverage test above from passing on an empty list.
        self.assertGreater(len(_requirement_names()), 10)


class TestConstraintsWellFormed(unittest.TestCase):
    # 119 is the number of packages installed in the green cloud run of Oct 20.
    # Update this number on purpose when pins are added or removed.
    EXPECTED_ENTRIES = 119

    def test_exact_entry_count(self):
        entries = [line for line in _constraint_lines() if line]
        self.assertEqual(len(entries), self.EXPECTED_ENTRIES)

    def test_every_non_empty_line_is_name_equals_version(self):
        bad = [line for line in _constraint_lines() if line and not PIN_LINE.match(line)]
        self.assertEqual(bad, [])

    def test_no_blank_lines_inside_the_file(self):
        lines = CONSTRAINTS.read_text(encoding="utf-8").split("\n")
        # The file ends with one newline, so the last split item is empty and is the only allowed one.
        self.assertEqual(lines[-1], "")
        self.assertNotIn("", lines[:-1])

    def test_no_duplicate_names_after_normalizing(self):
        names = [_normalize(line.split("==")[0]) for line in _constraint_lines() if line]
        self.assertEqual(len(names), len(set(names)))

    def test_sorted_case_insensitively_by_name(self):
        names = [line.split("==")[0] for line in _constraint_lines() if line]
        self.assertEqual(names, sorted(names, key=str.lower))


class TestWorkflowPipStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with WORKFLOW.open("r", encoding="utf-8") as f:
            cls.data = yaml.safe_load(f)
        cls.steps = {s.get("name"): s for s in cls.data["jobs"]["produce"]["steps"]}

    def test_install_step_uses_constraints_file(self):
        step = self.steps["Install Python dependencies"]
        self.assertEqual(step["run"], "pip install -r requirements.txt -c constraints-ci.txt")


class TestPinnedActionVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with WORKFLOW.open("r", encoding="utf-8") as f:
            cls.data = yaml.safe_load(f)
        cls.uses = [s["uses"] for s in cls.data["jobs"]["produce"]["steps"] if "uses" in s]

    def test_checkout_is_v7(self):
        self.assertIn("actions/checkout@v7", self.uses)

    def test_setup_python_is_v7(self):
        self.assertIn("actions/setup-python@v7", self.uses)

    def test_no_old_action_versions_left(self):
        self.assertNotIn("actions/checkout@v4", self.uses)
        self.assertNotIn("actions/setup-python@v5", self.uses)

    def test_runner_is_still_ubuntu_24_04(self):
        self.assertEqual(self.data["jobs"]["produce"]["runs-on"], "ubuntu-24.04")


if __name__ == "__main__":
    unittest.main()
