"""Edge TTS retry behaviour and the macOS 'say' fallback rule.

Everything external is mocked: the Edge call, sleeps, the 'say' command,
the provider keys and the loggers. Nothing sleeps and nothing real is written.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from verticals.tts import _generate_edge_tts, generate_voiceover

GOOD_AUDIO = b"x" * 5000
TINY_AUDIO = b"x" * 10
CLEAN_TEXT = "This is a short test voiceover about how sleep helps the brain."


class FakeEdge:
    """Stands in for _edge_tts_generate. Each call uses the next outcome:
    an Exception is raised, bytes are written to the output file."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self, text, voice, output_path, rate="+12%"):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        output_path.write_bytes(outcome)


@pytest.fixture(autouse=True)
def mocked_world():
    """Mock logs, sleeps, keys and the say/run_cmd commands for every test."""
    with patch("verticals.tts.log") as tts_log, \
         patch("verticals.script_clean.log"), \
         patch("verticals.tts.time.sleep") as sleep_mock, \
         patch("verticals.tts.get_minimax_key", return_value=""), \
         patch("verticals.tts.get_elevenlabs_key", return_value=""), \
         patch("verticals.tts.get_60db_key", return_value=""), \
         patch("verticals.tts._generate_say") as say_mock, \
         patch("verticals.tts.run_cmd") as run_mock:
        yield SimpleNamespace(
            log=tts_log, sleep=sleep_mock, say=say_mock, run=run_mock
        )


# (a) two failures (an exception, then a tiny file) then a success
def test_two_failures_then_success_produces_audio(tmp_path, mocked_world):
    edge = FakeEdge([
        Exception("No audio was received. Please verify that your parameters are correct."),
        TINY_AUDIO,
        GOOD_AUDIO,
    ])
    with patch("verticals.tts._edge_tts_generate", new=edge):
        result = generate_voiceover(CLEAN_TEXT, tmp_path, "en", provider="edge")

    assert result == tmp_path / "voiceover_en.mp3"
    assert result.read_bytes() == GOOD_AUDIO
    assert edge.calls == 3
    mocked_world.say.assert_not_called()
    mocked_world.run.assert_not_called()


# (b) three failures on Linux: clear error, and no say / run_cmd / other voice
def test_three_failures_on_linux_raise_clear_error(tmp_path, mocked_world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    edge = FakeEdge([
        Exception("first problem"),
        Exception("second problem"),
        Exception("network reset"),
    ])
    with patch("verticals.tts._edge_tts_generate", new=edge):
        with pytest.raises(RuntimeError) as exc:
            generate_voiceover(CLEAN_TEXT, tmp_path, "en", provider="edge")

    assert str(exc.value) == (
        "No working voice: Edge TTS failed after 3 attempts (last error: network reset)"
    )
    assert edge.calls == 3
    mocked_world.say.assert_not_called()
    mocked_world.run.assert_not_called()


# (c) a tiny audio file counts as a failed attempt
def test_tiny_audio_file_counts_as_failure(tmp_path, mocked_world):
    edge = FakeEdge([TINY_AUDIO, GOOD_AUDIO])
    with patch("verticals.tts._edge_tts_generate", new=edge):
        result = generate_voiceover(CLEAN_TEXT, tmp_path, "en", provider="edge")

    assert edge.calls == 2
    assert result.read_bytes() == GOOD_AUDIO
    logged = [c.args[0] for c in mocked_world.log.call_args_list]
    assert "Edge TTS attempt 1/3 failed: audio file too small (10 bytes)" in logged


# (d) on macOS, after three Edge failures, 'say' is still the last fallback
def test_macos_still_uses_say_after_edge_failures(tmp_path, mocked_world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    say_file = tmp_path / "voiceover_say.mp3"
    mocked_world.say.return_value = say_file
    edge = FakeEdge([Exception("a"), Exception("b"), Exception("c")])
    with patch("verticals.tts._edge_tts_generate", new=edge):
        result = generate_voiceover(CLEAN_TEXT, tmp_path, "en", provider="edge")

    assert result == say_file
    mocked_world.say.assert_called_once()
    assert mocked_world.say.call_args.args[1] == tmp_path


# (e) the waits are exactly [2, 5], and the final log line has the length and excerpt
def test_waits_and_final_log_line(tmp_path, mocked_world):
    script = "Line one of the voiceover is here.\nLine two carries on for a good while longer."
    edge = FakeEdge([Exception("x"), Exception("y"), Exception("z")])
    with patch("verticals.tts._edge_tts_generate", new=edge):
        with pytest.raises(RuntimeError):
            _generate_edge_tts(script, tmp_path, "en")

    assert [c.args[0] for c in mocked_world.sleep.call_args_list] == [2, 5]

    logged = [c.args[0] for c in mocked_world.log.call_args_list]
    excerpt = script[:60].replace("\n", " ")
    final_lines = [m for m in logged if str(len(script)) in m and excerpt in m]
    assert len(final_lines) == 1
    assert "\n" not in final_lines[0]
