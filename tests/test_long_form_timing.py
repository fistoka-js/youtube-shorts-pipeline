"""Tests for verticals/assemble.py long_form_frame_counts - long-form clips switch
at each section's real start, and every clip is a whole number of 1/30 s frames.

Made-up data only. Runs with pytest, or without it:
    python tests\\test_long_form_timing.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verticals import assemble  # noqa: E402
from verticals.assemble import FPS, long_form_frame_counts  # noqa: E402
from verticals.section_timing import compute_section_timings  # noqa: E402

# 18 Whisper words, one every 0.8 s, last one ends at 15.0 s total.
WORDS = [{"word": f"{s}{k}", "start": round(i * 0.8, 3), "end": round(i * 0.8 + 0.5, 3)}
         for i, (s, k) in enumerate((s, k) for s in "abc" for k in range(1, 7))]
TOTAL = 15.0
EXPECTED_FRAMES = round(TOTAL * FPS)  # 450

SECTIONS = [
    {"id": "s1", "narration": "a1 a2 a3 a4 a5 a6 extra", "broll_prompts": ["p", "q"]},
    {"id": "s2", "narration": "b1 b2 b3 b4 b5 b6 extra", "broll_prompts": ["p", "q", "r"]},
    {"id": "s3", "narration": "c1 c2 c3 c4 c5 c6 extra", "broll_prompts": ["p"]},
]


def _frames(counts_by_section):
    out = []
    for sid, n in counts_by_section:
        out += [{"path": f"{sid}_{i}.mp4", "type": "video", "section_id": sid} for i in range(n)]
    return out


def _clip_starts(counts):
    starts, t = [], 0
    for c in counts:
        starts.append(t / FPS)
        t += c
    return starts


def test_clips_start_at_section_starts():
    frames = _frames([("s1", 2), ("s2", 3), ("s3", 1)])
    counts = long_form_frame_counts(frames, SECTIONS, WORDS, TOTAL)
    starts = _clip_starts(counts)
    timings = {t["id"]: t["start"] for t in compute_section_timings(SECTIONS, WORDS)}
    assert timings == {"s1": 0.0, "s2": 4.8, "s3": 9.6}
    # first clip of each section begins at the section start (to the 1/30 s grid)
    assert abs(starts[0] - timings["s1"]) < 1 / 60
    assert abs(starts[2] - timings["s2"]) < 1 / 60
    assert abs(starts[5] - timings["s3"]) < 1 / 60
    # clips inside a section split its span into equal shares
    assert [round(x, 2) for x in starts] == [0.0, 2.4, 4.8, 6.4, 8.0, 9.6]


def test_durations_sum_to_audio_length_exactly():
    frames = _frames([("s1", 2), ("s2", 3), ("s3", 1)])
    counts = long_form_frame_counts(frames, SECTIONS, WORDS, TOTAL)
    assert sum(counts) == EXPECTED_FRAMES
    assert all(c >= 1 for c in counts)
    assert abs(sum(counts) / FPS - TOTAL) < 1 / 60


def test_mismatch_warns_and_falls_back_without_crashing():
    # s2 has 2 clips but 3 broll prompts
    frames = _frames([("s1", 2), ("s2", 2), ("s3", 1)])
    with patch.object(assemble, "log") as mock_log:
        counts = long_form_frame_counts(frames, SECTIONS, WORDS, TOTAL)
    messages = [c[0][0] for c in mock_log.call_args_list]
    assert any(m.startswith("WARNING") and "section s2" in m for m in messages), messages
    assert len(counts) == len(frames)
    assert sum(counts) == EXPECTED_FRAMES


def test_one_clip_section_covers_its_whole_span():
    sections = [
        {"id": "s1", "narration": "a1 a2 a3 a4 a5 a6 extra", "broll_prompts": ["p"]},
        {"id": "s2", "narration": "b1 b2 b3 b4 b5 b6 extra", "broll_prompts": ["p"]},
        {"id": "s3", "narration": "c1 c2 c3 c4 c5 c6 extra", "broll_prompts": ["p"]},
    ]
    frames = _frames([("s1", 1), ("s2", 1), ("s3", 1)])
    counts = long_form_frame_counts(frames, sections, WORDS, TOTAL)
    assert _clip_starts(counts) == [0.0, 4.8, 9.6]
    assert counts == [144, 144, 162]
    assert sum(counts) == EXPECTED_FRAMES


def test_zero_clip_section_gives_its_time_to_the_section_before():
    sections = [
        {"id": "s1", "narration": "a1 a2 a3 a4 a5 a6 extra", "broll_prompts": ["p", "q"]},
        {"id": "s2", "narration": "b1 b2 b3 b4 b5 b6 extra", "broll_prompts": []},
        {"id": "s3", "narration": "c1 c2 c3 c4 c5 c6 extra", "broll_prompts": ["p"]},
    ]
    frames = _frames([("s1", 2), ("s3", 1)])
    with patch.object(assemble, "log") as mock_log:
        counts = long_form_frame_counts(frames, sections, WORDS, TOTAL)
    assert mock_log.call_count == 0, mock_log.call_args_list
    # s1 runs from 0 to s3's start (9.6 s), so its two clips are 0.0 and 4.8
    assert _clip_starts(counts) == [0.0, 4.8, 9.6]
    assert sum(counts) == EXPECTED_FRAMES


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
