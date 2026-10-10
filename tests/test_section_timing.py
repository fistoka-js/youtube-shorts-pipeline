"""Tests for verticals/section_timing.py - where each section starts in the
Whisper word list.

Runs with pytest, or without it:  python tests\\test_section_timing.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verticals import section_timing  # noqa: E402
from verticals.section_timing import compute_section_timings  # noqa: E402


def _words(text):
    return [{"word": w, "start": float(i), "end": float(i) + 0.5}
            for i, w in enumerate(text.split())]


def test_six_word_match_is_found():
    words = _words("alpha beta gamma delta epsilon zeta eta theta")
    secs = [{"id": "s1", "narration": "alpha beta gamma delta epsilon zeta"}]
    t = compute_section_timings(secs, words)[0]
    assert t["method"] == "match6"
    assert t["start_idx"] == 0


def test_three_word_match_is_accepted():
    words = _words("x y z alpha beta gamma delta")
    secs = [{"id": "s1", "narration": "alpha beta gamma and more words"}]
    t = compute_section_timings(secs, words)[0]
    assert t["method"] == "match3"
    assert t["start_idx"] == 3


def test_two_word_common_phrase_falls_back():
    # s1 matches at word 0 and has 8 script words, so the fallback estimate for
    # s2 is word 8. Whisper contains "the dog" (2 words) at word 9. s2's opening
    # words "the dog ran far away" must NOT match on just "the dog": it should
    # fall back to the estimate (8), not jump to 9.
    words = _words("alpha beta gamma delta epsilon zeta eta theta iota the dog sat there and then more")
    secs = [
        {"id": "s1", "narration": "alpha beta gamma delta epsilon zeta eta theta"},
        {"id": "s2", "narration": "the dog ran far away"},
    ]
    with patch.object(section_timing, "log") as mock_log:
        timings = compute_section_timings(secs, words)
    s2 = timings[1]
    assert s2["method"] == "proportional", s2
    assert s2["start_idx"] == 8, s2
    assert mock_log.call_count == 1
    assert "Section s2" in mock_log.call_args[0][0]


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
