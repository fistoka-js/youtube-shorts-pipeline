"""Tests for the final loudness step in verticals/assemble.py.

The step measures the finished audio with loudnorm (pass 1) and then applies one
linear gain with loudnorm (pass 2). The goal is as loud as the true-peak limit
(TRUE_PEAK_LIMIT_DBTP) allows, with LOUDNESS_MAX_LUFS (-14) only as an upper cap.
Real voices are normally stopped by the peak limit (about -19 to -20 LUFS). Only
quiet-peaked material can reach the cap.

Most tests mock run_cmd. The tests at the end render tiny synthetic clips with real
ffmpeg and are skipped if ffmpeg is not on PATH.
"""

import re
import shutil
import subprocess
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from verticals import assemble as A
from verticals import music
from verticals.assemble import build_audio_filter

HAS_FFMPEG = shutil.which("ffmpeg") is not None
RATE = 24000
# loudnorm and ebur128 print values to 0.01 dB. Allow a little for that.
MEASURE_DB = 0.05


def _measure_stderr(i: str, tp: str, lra: str = "2.10", thresh: str = "-36.88") -> str:
    """The text loudnorm prints in pass 1, as ffmpeg writes it to stderr."""
    return ("[Parsed_loudnorm_0 @ 0000] \n{\n"
            f'\t"input_i" : "{i}",\n\t"input_tp" : "{tp}",\n\t"input_lra" : "{lra}",\n'
            f'\t"input_thresh" : "{thresh}",\n\t"normalization_type" : "dynamic"\n}}\n')


def _filter_values(f: str) -> dict:
    """The key=value options of a loudnorm filter string, e.g. {'I': '-19.53', ...}."""
    body = f.split("=", 1)[1]
    return dict(part.split("=", 1) for part in body.split(":") if "=" in part)


def _audio_maps(cmd: list) -> list:
    """The stream names given after each -map (the video map 0:v is left out)."""
    return [cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] == "-map" and cmd[i + 1] != "0:v"]


class TestConstants:
    def test_cap_and_peak_limit_have_the_chosen_values(self):
        assert A.LOUDNESS_MAX_LUFS == -14.0
        assert A.TRUE_PEAK_LIMIT_DBTP == -1.5

    def test_other_choices_exist_and_are_sensible(self):
        assert A.LOUDNESS_LRA_TARGET == 11.0
        assert 0.0 < A.LOUDNESS_PEAK_MARGIN_DB < 1.0
        assert A.LOUDNESS_MEASURE_TIMEOUT_SECONDS > 0

    def test_floor_and_maximum_boost_have_the_chosen_values(self):
        assert A.LOUDNESS_MIN_INPUT_LUFS == -50.0
        assert A.LOUDNESS_MAX_BOOST_DB == 20.0

    def test_voice_floor_has_the_chosen_value(self):
        assert A.LOUDNESS_MIN_VOICE_LUFS == -45.0

    def test_pass1_filter_measures_with_the_cap_and_prints_json(self):
        f = A._loudnorm_measure_filter()
        assert f.startswith("loudnorm=I=-14:TP=-1.5:LRA=11")
        assert "print_format=json" in f


class TestPass2Filter:
    """Pure arithmetic on pass 1's numbers (no ffmpeg)."""

    def test_cap_is_reached_when_the_peak_allows_it(self):
        pass2, target, _, _ = A._loudness_filter_from_measurement(_measure_stderr("-24.00", "-14.00"))
        assert target == -14.0
        assert _filter_values(pass2)["offset"] == "10.00"

    def test_gain_stops_at_the_peak_limit_when_the_cap_is_not_reachable(self):
        # Measured on a real voiceover: -26.51 LUFS, true peak -8.58 dBTP.
        pass2, target, i, tp = A._loudness_filter_from_measurement(_measure_stderr("-26.51", "-8.58"))
        assert target == pytest.approx(-19.53, abs=0.005)
        assert _filter_values(pass2)["offset"] == "6.98"
        # The peak after the gain is the limit minus the margin, so it is below the limit
        assert tp + 6.98 <= A.TRUE_PEAK_LIMIT_DBTP

    def test_quiet_gain_is_linear_and_never_dynamic(self):
        pass2, _, _, _ = A._loudness_filter_from_measurement(_measure_stderr("-26.51", "-8.58"))
        assert "linear=true" in pass2
        assert "dynamic" not in pass2

    def test_a_loud_mix_is_turned_down_to_the_cap(self):
        pass2, target, _, _ = A._loudness_filter_from_measurement(_measure_stderr("-10.00", "-1.00"))
        assert target == -14.0
        assert _filter_values(pass2)["offset"] == "-4.00"

    def test_measured_values_are_passed_on_for_pass2(self):
        pass2, _, _, _ = A._loudness_filter_from_measurement(_measure_stderr("-26.51", "-8.58", "2.10", "-36.88"))
        v = _filter_values(pass2)
        assert v["measured_I"] == "-26.51"
        assert v["measured_TP"] == "-8.58"
        assert v["measured_LRA"] == "2.10"
        assert v["measured_thresh"] == "-36.88"

    def test_near_silent_input_below_the_floor_raises(self):
        # Measured on a real 3 s near-silent file: -61.2 LUFS. It must not be boosted by 46 dB.
        with pytest.raises(A.LoudnessOutOfRange, match="below the floor"):
            A._loudness_filter_from_measurement(_measure_stderr("-61.20", "-40.00"))

    def test_a_gain_above_the_maximum_boost_raises(self):
        # -40 LUFS with a true peak of -30 dBTP would need about +26 dB
        with pytest.raises(A.LoudnessOutOfRange, match="maximum boost"):
            A._loudness_filter_from_measurement(_measure_stderr("-40.00", "-30.00"))

    def test_a_normal_gain_is_not_refused(self):
        pass2, _, _, _ = A._loudness_filter_from_measurement(_measure_stderr("-26.51", "-8.58"))
        assert _filter_values(pass2)["offset"] == "6.98"

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            A._loudness_filter_from_measurement("ffmpeg printed nothing useful")

    def test_minus_infinity_from_silence_raises(self):
        with pytest.raises(ValueError, match="silent"):
            A._loudness_filter_from_measurement(_measure_stderr("-inf", "-inf"))


class TestMeasuringPassFallback:
    """If the measuring pass fails, the step returns None, logs a warning and never raises."""

    VOICE = Path("voice.mp3")

    def _run_step(self, fake_run, music=None):
        with mock.patch.object(A, "run_cmd", side_effect=fake_run), mock.patch.object(A, "log") as log:
            result = A._loudness_step(self.VOICE, music, 10.0, None, 0.0, 0.0)
        return result, log

    def test_ffmpeg_error_logs_a_warning_and_returns_none(self):
        def boom(cmd, **kwargs):
            raise RuntimeError("ffmpeg exited with 1")
        result, log = self._run_step(boom)
        assert result is None
        assert log.call_count == 1
        assert "WARNING" in log.call_args.args[0]
        assert "loudness step skipped" in log.call_args.args[0]

    def test_timeout_logs_a_readable_warning_with_no_command_line(self):
        def slow(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        result, log = self._run_step(slow, music=PULSAR)
        assert result is None
        message = log.call_args.args[0]
        assert "WARNING" in message
        assert "the measuring pass timed out after 300 seconds" in message
        assert "ffmpeg" not in message
        assert "-loglevel" not in message
        assert "pulsar" not in message and "\\" not in message and "/" not in message

    def test_other_unexpected_error_logs_its_type_only(self):
        def odd(cmd, **kwargs):
            raise OSError("cannot start the process")
        result, log = self._run_step(odd)
        assert result is None
        message = log.call_args.args[0]
        assert "the measuring pass failed (OSError)" in message
        assert "cannot start the process" not in message
        assert "ffmpeg" not in message

    def test_unparsable_output_returns_none(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr="no measurement here"))
        assert result is None
        assert "WARNING" in log.call_args.args[0]

    def test_silent_audio_returns_none(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-inf", "-inf")))
        assert result is None
        assert "WARNING" in log.call_args.args[0]

    def test_a_mocked_runner_with_no_stderr_returns_none(self):
        # run_cmd replaced by a bare MagicMock (as some older tests do) must not raise
        result, _ = self._run_step(lambda cmd, **kw: mock.MagicMock())
        assert result is None

    def test_success_returns_the_pass2_filter_and_logs_the_level(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-26.51", "-8.58")))
        assert result is not None and result.startswith("loudnorm=I=-19.53")
        assert "stopped by the true-peak limit" in log.call_args.args[0]

    def test_near_silent_audio_below_the_floor_warns_and_returns_none(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-61.20", "-40.00")))
        assert result is None
        message = log.call_args.args[0]
        assert "WARNING" in message
        assert "below the voice floor" in message  # the voice floor (-45) is checked before the mix floor (-50)
        assert "kept at its current level" in message

    def test_a_gain_above_the_maximum_boost_warns_and_returns_none(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-40.00", "-30.00")))
        assert result is None
        assert "maximum boost" in log.call_args.args[0]

    def test_a_level_above_the_peak_limit_is_lowered_and_the_log_says_so(self):
        # Input is above the peak limit: the gain is negative, so the level comes down
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-19.70", "-0.90")))
        assert result is not None
        message = log.call_args.args[0]
        assert "lowered" in message
        assert "true peak is already at or above the limit" in message
        assert "stopped by the true-peak limit" not in message

    def test_a_level_above_the_cap_is_lowered_to_the_cap_and_says_so(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-10.00", "-5.00")))
        assert result is not None
        message = log.call_args.args[0]
        assert "lowered to the -14 LUFS cap" in message
        assert "stopped by the true-peak limit" not in message

    def test_a_quiet_voiceover_is_refused_before_the_mix_is_measured(self):
        # The voice alone is -60 LUFS: the step must stop there, never measuring the music mix
        calls = []

        def fake(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(stderr=_measure_stderr("-60.00", "-40.00"))
        result, log = self._run_step(fake, music=Path("pulsar.mp3"))
        assert result is None
        assert len(calls) == 1
        assert "below the voice floor" in log.call_args.args[0]
        assert "no music is boosted" in log.call_args.args[0]

    def test_the_log_says_mix_when_music_is_added(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-26.51", "-8.58")),
                                     music=Path("pulsar.mp3"))
        assert result is not None
        message = log.call_args.args[0]
        assert message.startswith("Loudness: mix measures -26.5 LUFS")
        assert "voiceover measures" not in message

    def test_the_log_says_voiceover_when_there_is_no_music(self):
        result, log = self._run_step(lambda cmd, **kw: mock.Mock(stderr=_measure_stderr("-20.00", "-2.00")))
        assert result is not None
        message = log.call_args.args[0]
        assert message.startswith("Loudness: voiceover measures -20.0 LUFS")
        assert "mix measures" not in message

    def test_timeout_is_passed_to_the_runner(self):
        seen = {}

        def record(cmd, **kwargs):
            seen.update(kwargs)
            return mock.Mock(stderr=_measure_stderr("-20.00", "-10.00"))
        self._run_step(record)
        assert seen["timeout"] == A.LOUDNESS_MEASURE_TIMEOUT_SECONDS
        assert seen["capture"] is True

    def test_measure_has_the_same_inputs_as_the_render_and_no_input_trim(self):
        # A -t on the input changes the format ffmpeg gives the mix, which changes the
        # true peak (found by testing: 3 dB difference), so the measuring pass must not use it.
        seen = {}

        def record(cmd, **kwargs):
            seen["cmd"] = cmd
            return mock.Mock(stderr=_measure_stderr("-20.00", "-10.00"))
        self._run_step(record, music=Path("pulsar.mp3"))
        cmd = seen["cmd"]
        assert "-t" not in cmd
        assert cmd[cmd.index("-i") + 2] == "-i"  # voice, then music, as in the render

    def test_music_branch_measures_the_same_graph_the_render_uses(self):
        seen = {}

        def record(cmd, **kwargs):
            seen["cmd"] = cmd
            return mock.Mock(stderr=_measure_stderr("-20.00", "-10.00"))
        self._run_step(record, music=Path("pulsar.mp3"))
        cmd = seen["cmd"]
        fc = cmd[cmd.index("-filter_complex") + 1]
        expected = build_audio_filter(10.0, None, 0.0, 0.0, voice_index=0, music_index=1)
        assert fc.startswith(expected)
        assert "[aout]loudnorm=" in fc
        assert cmd[-3:] == ["-f", "null", "-"]


class TestAssembleVideoWithTheStep:
    """assemble_video with ffmpeg mocked: the final command gets the step or not."""

    @pytest.fixture
    def env(self, tmp_path):
        out_dir = tmp_path / "work"
        out_dir.mkdir()
        frames = [{"path": tmp_path / "c0.mp4", "type": "video"}]
        recorded = []
        return out_dir, frames, recorded, tmp_path

    def _assemble(self, env, measure_stderr=None, measure_error=None, music=None):
        out_dir, frames, recorded, tmp_path = env
        if music:
            # assemble_video only uses the music when the file exists
            (tmp_path / music).write_bytes(b"")
            music = str(tmp_path / music)

        def fake_run(cmd, check=True, capture=False, **kwargs):
            if capture:
                if measure_error is not None:
                    raise measure_error
                return mock.Mock(stderr=measure_stderr)
            recorded.append(cmd)
            return None

        with mock.patch.object(A, "get_audio_duration", return_value=10.0), \
             mock.patch.object(A, "_prepare_segment"), \
             mock.patch.object(A, "run_cmd", side_effect=fake_run), \
             mock.patch.object(A, "MEDIA_DIR", tmp_path), \
             mock.patch.object(A, "log") as log:
            out = A.assemble_video(frames, tmp_path / "vo.mp3", out_dir, "job1",
                                   music_path=music, start_at=16.6, gain_db=-5.3,
                                   duck_filter=None)
        return out, recorded[-1], log

    def test_music_branch_failed_measure_keeps_the_old_final_command(self, env):
        _, cmd, log = self._assemble(env, measure_error=RuntimeError("boom"), music="music.mp3")
        assert "loudnorm" not in " ".join(cmd)
        assert _audio_maps(cmd) == ["[aout]"]
        assert any("loudness step skipped" in c.args[0] for c in log.call_args_list)

    def test_no_music_branch_failed_measure_keeps_the_old_final_command(self, env):
        _, cmd, log = self._assemble(env, measure_error=RuntimeError("boom"), music=None)
        assert "loudnorm" not in " ".join(cmd)
        assert "-af" not in cmd
        assert any("loudness step skipped" in c.args[0] for c in log.call_args_list)

    def test_music_branch_adds_the_step_after_the_mix(self, env):
        _, cmd, _ = self._assemble(env, measure_stderr=_measure_stderr("-26.51", "-8.58"), music="music.mp3")
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "[aout]loudnorm=I=-19.53" in fc
        assert "[afinal]" in fc
        assert _audio_maps(cmd) == ["[afinal]"]

    def test_no_music_branch_adds_the_step_as_an_audio_filter(self, env):
        _, cmd, _ = self._assemble(env, measure_stderr=_measure_stderr("-26.51", "-8.58"), music=None)
        assert cmd[cmd.index("-af") + 1].startswith("loudnorm=I=-19.53")
        # the video is not re-encoded just for the loudness step
        assert cmd[cmd.index("-c:v") + 1] == "copy"


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def _tone(seconds: float, freq: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return amp * np.sin(2 * np.pi * freq * t)


def _ebur(path: Path) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) of a file, measured by ffmpeg ebur128.

    The LUFS comes from the per-frame metadata, which has 3 decimals. The summary
    printed by ebur128 has only 1 decimal, which is too coarse for the 0.1 dB gap check.
    """
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                        "-af", "ebur128=metadata=1:peak=true,ametadata=mode=print:key=lavfi.r128.I:file=-",
                        "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    lufs = float(re.findall(r"lavfi\.r128\.I=(-?[\d.]+)", r.stdout)[-1])
    tp = re.search(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+) dBFS", r.stderr)
    return lufs, float(tp.group(1))


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
class TestRealFfmpegLoudnessStep:
    """Real two-pass run on small synthetic clips. The voice is a 220 Hz tone and the
    music a 330 Hz tone, so both are steady and easy to measure.

    Synthetic tones have a low peak-to-loudness ratio (quiet-peaked material), so for these
    the -14 LUFS CAP IS reachable. The tests that check the level within 1 dB of the cap are
    the synthetic case where the cap is reachable. Real voices are different: their peaks
    sit far above their average, so the true-peak limit wins and the level stays under the
    cap. That is checked in TestRealVoiceover."""

    def test_clip_already_above_the_peak_limit_is_lowered_and_logged_as_lowered(self, tmp_path):
        # A quiet tone with one click that reaches -0.9 dBFS: the input's peak is already over the limit
        voice = _tone(6.0, 220, 0.03)
        voice[int(3.0 * RATE):int(3.01 * RATE)] = 0.9
        _write_wav(tmp_path / "hot.wav", voice)
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "hot.wav", None, 6.0, None, 0.0, 0.0)
        assert loud is not None
        message = log.call_args.args[0]
        assert "lowered" in message and "stopped by the true-peak limit" not in message
        out = tmp_path / "hot_after.wav"
        _ffmpeg(["-i", str(tmp_path / "hot.wav"), "-af", loud, str(out)])
        lufs, tp = _ebur(out)
        in_lufs, _ = _ebur(tmp_path / "hot.wav")
        assert lufs < in_lufs  # the level really does come down
        assert tp <= A.TRUE_PEAK_LIMIT_DBTP + MEASURE_DB

    @pytest.fixture
    def clips(self, tmp_path):
        # Levels chosen so the gain needed is about +15 dB, inside the 20 dB maximum boost,
        # and the peak after the gain stays below the limit (so the target is reachable)
        _write_wav(tmp_path / "voice.wav", _tone(6.0, 220, 0.1))
        _write_wav(tmp_path / "music.wav", _tone(6.0, 330, 0.05))
        return tmp_path

    def test_quiet_peaked_no_music_clip_reaches_the_cap_and_stays_below_the_peak_limit(self, clips):
        """The synthetic case where the cap IS reachable: within 1 dB of the -14 LUFS cap."""
        loud = A._loudness_step(clips / "voice.wav", None, 6.0, None, 0.0, 0.0)
        assert loud is not None
        out = clips / "no_music_after.wav"
        _ffmpeg(["-i", str(clips / "voice.wav"), "-af", loud, str(out)])
        lufs, tp = _ebur(out)
        assert lufs == pytest.approx(A.LOUDNESS_MAX_LUFS, abs=1.0)
        assert tp <= A.TRUE_PEAK_LIMIT_DBTP

    def test_quiet_peaked_music_clip_reaches_the_cap_and_stays_below_the_peak_limit(self, clips):
        """The synthetic case where the cap IS reachable (with music): within 1 dB of the cap."""
        # Same graph shape as assemble_video: the step is appended after [aout]
        graph = build_audio_filter(6.0, None, 0.0, 0.0, voice_index=0, music_index=1)
        loud = A._loudness_step(clips / "voice.wav", str(clips / "music.wav"), 6.0, None, 0.0, 0.0)
        assert loud is not None
        out = clips / "music_after.wav"
        _ffmpeg(["-i", str(clips / "voice.wav"), "-i", str(clips / "music.wav"),
                 "-filter_complex", f"{graph};[aout]{loud}[afinal]", "-map", "[afinal]", str(out)])
        lufs, tp = _ebur(out)
        assert lufs == pytest.approx(A.LOUDNESS_MAX_LUFS, abs=1.0)
        assert tp <= A.TRUE_PEAK_LIMIT_DBTP

    def test_aac_encoded_output_stays_below_the_peak_limit(self, clips):
        # The renders are AAC in an mp4, so the limit must hold after encoding, not only before
        graph = build_audio_filter(6.0, None, 0.0, 0.0, voice_index=0, music_index=1)
        loud = A._loudness_step(clips / "voice.wav", str(clips / "music.wav"), 6.0, None, 0.0, 0.0)
        assert loud is not None
        out = clips / "music_after.m4a"
        _ffmpeg(["-i", str(clips / "voice.wav"), "-i", str(clips / "music.wav"),
                 "-filter_complex", f"{graph};[aout]{loud}[afinal]", "-map", "[afinal]",
                 "-c:a", "aac", str(out)])
        _, tp = _ebur(out)
        assert tp <= A.TRUE_PEAK_LIMIT_DBTP + MEASURE_DB

    def test_high_peak_clip_is_stopped_by_the_peak_limit_before_the_cap(self, tmp_path):
        # A quiet tone with one loud click: the click sets the peak, so the target is not reachable
        voice = _tone(6.0, 220, 0.03)
        voice[int(3.0 * RATE):int(3.01 * RATE)] = 0.5
        _write_wav(tmp_path / "clicky.wav", voice)
        loud = A._loudness_step(tmp_path / "clicky.wav", None, 6.0, None, 0.0, 0.0)
        assert loud is not None
        out = tmp_path / "clicky_after.wav"
        _ffmpeg(["-i", str(tmp_path / "clicky.wav"), "-af", loud, str(out)])
        in_lufs, _ = _ebur(tmp_path / "clicky.wav")
        lufs, tp = _ebur(out)
        target = float(_filter_values(loud)["I"])
        offset = float(_filter_values(loud)["offset"])
        assert target < A.LOUDNESS_MAX_LUFS  # the peak stopped the gain
        # The gain applied is exactly the offset the step worked out
        assert (lufs - in_lufs) == pytest.approx(offset, abs=MEASURE_DB)
        # loudnorm measures the input with its own integration, which for this short clip
        # with a click differs from ebur128 by about 0.4 dB, so the level is checked to 0.5 dB
        assert lufs == pytest.approx(target, abs=0.5)
        assert tp <= A.TRUE_PEAK_LIMIT_DBTP + MEASURE_DB
        assert tp == pytest.approx(A.TRUE_PEAK_LIMIT_DBTP - A.LOUDNESS_PEAK_MARGIN_DB, abs=0.3)

    def test_voice_to_music_gap_is_unchanged_by_the_step(self, clips):
        # Each stem is measured alone, before and after the same final gain
        graph = build_audio_filter(6.0, None, 0.0, 0.0, voice_index=0, music_index=1)
        music_chain = graph.split(";", 1)[0]
        loud = A._loudness_step(clips / "voice.wav", str(clips / "music.wav"), 6.0, None, 0.0, 0.0)
        assert loud is not None

        voice_raw, music_raw = clips / "voice_raw.wav", clips / "music_raw.wav"
        _ffmpeg(["-i", str(clips / "voice.wav"), "-c:a", "pcm_s16le", str(voice_raw)])
        _ffmpeg(["-i", str(clips / "voice.wav"), "-i", str(clips / "music.wav"),
                 "-filter_complex", music_chain, "-map", "[music]", "-c:a", "pcm_s16le", str(music_raw)])
        voice_after, music_after = clips / "voice_after.wav", clips / "music_after.wav"
        _ffmpeg(["-i", str(voice_raw), "-af", loud, str(voice_after)])
        _ffmpeg(["-i", str(music_raw), "-af", loud, str(music_after)])

        gap_before = _ebur(voice_raw)[0] - _ebur(music_raw)[0]
        gap_after = _ebur(voice_after)[0] - _ebur(music_after)[0]
        assert gap_after == pytest.approx(gap_before, abs=0.1)

    def test_silent_clip_falls_back_without_raising(self, tmp_path):
        _write_wav(tmp_path / "silent.wav", np.zeros(RATE * 3))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "silent.wav", None, 3.0, None, 0.0, 0.0)
        assert loud is None
        assert "WARNING" in log.call_args.args[0]

    def test_near_silent_clip_below_the_floor_warns_and_gets_no_boost(self, tmp_path):
        # A tone at 0.002 amplitude: about -57 LUFS, above loudnorm's -70 LUFS gate (so it
        # is measured, not -inf) but below the -45 LUFS voice floor. (A -80 dBFS tone is gated
        # out by loudnorm and reports -inf, which is the separate silent-audio path.)
        _write_wav(tmp_path / "faint.wav", _tone(3.0, 220, 0.002))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "faint.wav", None, 3.0, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "WARNING" in message and "below the voice floor" in message

    def test_missing_voiceover_falls_back_without_raising(self, tmp_path):
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "nope.mp3", None, 3.0, None, 0.0, 0.0)
        assert loud is None
        assert "WARNING" in log.call_args.args[0]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
class TestReadableWarnings:
    """The warning text must say what happened, in plain words."""

    def test_a_very_short_audible_clip_says_too_short_not_silent(self, tmp_path):
        # 0.05 s of a 440 Hz tone: audible, but shorter than the 400 ms gating block,
        # so loudnorm reports -inf. It must not be called silent.
        _write_wav(tmp_path / "short.wav", _tone(0.05, 440, 0.3))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "short.wav", None, 0.05, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "too short to measure" in message
        assert "silent" not in message

    def test_a_truly_silent_clip_still_says_silent(self, tmp_path):
        _write_wav(tmp_path / "silent.wav", np.zeros(RATE * 3))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "silent.wav", None, 3.0, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "silent or too quiet to measure" in message
        assert "too short" not in message

    def test_an_empty_mp3_says_empty_or_cannot_be_read(self, tmp_path):
        (tmp_path / "empty.mp3").write_bytes(b"")
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "empty.mp3", None, 1.0, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "the audio file is empty or cannot be read" in message
        assert "exit" not in message.lower()
        assert not re.search(r"\d{7,}", message)  # no raw exit code

    def test_a_corrupt_mp3_says_empty_or_cannot_be_read_and_never_raises(self, tmp_path):
        (tmp_path / "garbage.mp3").write_bytes(b"this is not audio at all" * 50)
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "garbage.mp3", None, 1.0, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "the audio file is empty or cannot be read" in message
        assert not re.search(r"\d{7,}", message)

    def test_a_runner_error_on_the_voiceover_gives_the_readable_message(self):
        def boom(cmd, **kwargs):
            raise RuntimeError("ffmpeg exited with 3199971767")
        with mock.patch.object(A, "run_cmd", side_effect=boom), mock.patch.object(A, "log") as log:
            loud = A._loudness_step(Path("nowhere.mp3"), None, 1.0, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "the audio file is empty or cannot be read" in message
        assert "3199971767" not in message


REPO_ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR_FOR_TESTS = REPO_ROOT / ".verticals" / "media"
PULSAR = REPO_ROOT / "music" / "pulsar.mp3"
# Only the tests that really use pulsar carry this skip. The others run without it.
NEEDS_PULSAR = pytest.mark.skipif(not PULSAR.exists(), reason="music/pulsar.mp3 is not in the repo")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
class TestVoiceoverGuardWithMusic:
    """Real ffmpeg with the real pulsar track. With music, a silent or very quiet voiceover
    must not be replaced by a boost of the music alone."""

    def _voice(self, tmp_path, name, samples):
        _write_wav(tmp_path / name, samples)
        return tmp_path / name

    @NEEDS_PULSAR
    def test_silent_voiceover_with_music_returns_no_step_and_warns(self, tmp_path):
        voice = self._voice(tmp_path, "silent.wav", np.zeros(RATE * 3))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(voice, str(PULSAR), 3.0, None, 16.6, -5.3)
        assert loud is None
        message = log.call_args.args[0]
        assert "WARNING" in message and "silent" in message
        assert "kept at its current level" in message

    @NEEDS_PULSAR
    def test_very_quiet_voiceover_with_music_returns_no_step_and_warns(self, tmp_path):
        # 0.002 amplitude is about -57 LUFS: measurable, but below the -45 LUFS voice floor
        voice = self._voice(tmp_path, "quiet.wav", _tone(3.0, 220, 0.002))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(voice, str(PULSAR), 3.0, None, 16.6, -5.3)
        assert loud is None
        message = log.call_args.args[0]
        assert "WARNING" in message and "below the voice floor" in message

    def test_quiet_audible_tone_at_minus_60_db_is_too_quiet_not_silent(self, tmp_path):
        # A real 3 s 440 Hz tone at -60 dBFS: audible. loudnorm measures it (about -64 LUFS,
        # a finite value), so it goes through the voice floor. The warning must say it is
        # too quiet, and must not call it silent. Nothing is raised, and no step is applied.
        _write_wav(tmp_path / "tone60.wav", _tone(3.0, 440, 0.001))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "tone60.wav", None, 3.0, None, 0.0, 0.0)
        assert loud is None
        message = log.call_args.args[0]
        assert "WARNING" in message
        assert "too quiet to use" in message
        assert "silent" not in message

    @NEEDS_PULSAR
    def test_quiet_audible_tone_at_minus_60_db_with_music_applies_no_step(self, tmp_path):
        _write_wav(tmp_path / "tone60m.wav", _tone(3.0, 440, 0.001))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "tone60m.wav", str(PULSAR), 3.0, None, 16.6, -5.3)
        assert loud is None
        message = log.call_args.args[0]
        assert "too quiet to use" in message
        assert "silent" not in message

    def test_a_tone_below_the_gate_reports_silent_or_too_quiet_to_measure(self, tmp_path):
        # A real 3 s 440 Hz tone at -80 dBFS: below loudnorm's -70 LUFS gate, so it reports -inf.
        _write_wav(tmp_path / "tone80.wav", _tone(3.0, 440, 0.0001))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(tmp_path / "tone80.wav", None, 3.0, None, 0.0, 0.0)
        assert loud is None
        assert "silent or too quiet to measure" in log.call_args.args[0]

    @NEEDS_PULSAR
    def test_normal_voiceover_with_music_still_gets_its_step(self, tmp_path):
        voice = self._voice(tmp_path, "normal.wav", _tone(3.0, 220, 0.1))
        loud = A._loudness_step(voice, str(PULSAR), 3.0, None, 16.6, -5.3)
        assert loud is not None
        assert loud.startswith("loudnorm=I=")

    def test_silent_voiceover_without_music_still_falls_back_with_a_warning(self, tmp_path):
        voice = self._voice(tmp_path, "silent2.wav", np.zeros(RATE * 3))
        with mock.patch.object(A, "log") as log:
            loud = A._loudness_step(voice, None, 3.0, None, 0.0, 0.0)
        assert loud is None
        assert "silent" in log.call_args.args[0]


def _real_voiceover() -> Path | None:
    """A real ~60 s voiceover from the local media folder, or None. The folder is not in
    git, so this test skips when it is missing."""
    for vo in sorted(MEDIA_DIR_FOR_TESTS.glob("work_*/voiceover_en.mp3")):
        try:
            dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                                        "-of", "csv=p=0", str(vo)],
                                       capture_output=True, text=True, check=True).stdout)
        except Exception:
            continue
        if 40.0 <= dur <= 90.0:
            return vo
    return None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
class TestRealVoiceover:
    """REAL voices, not synthetic tones. Their peaks sit far above their average loudness,
    so the true-peak limit wins and the level stays under the -14 LUFS cap (normally about
    -19 to -20 LUFS). These tests check that honest result: the peak lands at the limit, the
    level is not above the cap, and it does not drop unless the input was already over the
    limit. Duck settings are production-style (duck_levels)
    with one full-length speech region, which is what Whisper gave for these files."""

    @pytest.fixture
    def voiceover(self):
        vo = _real_voiceover()
        if vo is None:
            pytest.skip("no real ~60 s voiceover in .verticals/media")
        return vo

    @staticmethod
    def _duration(path: Path) -> float:
        return float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", str(path)],
                                    capture_output=True, text=True, check=True).stdout)

    def _check_peak_limit_wins(self, tmp_path, vo, music_path, duck, start_at, gain_db):
        dur = self._duration(vo)
        if music_path:
            graph = build_audio_filter(dur, duck, start_at, gain_db, voice_index=0, music_index=1)
            inputs = ["-i", str(vo), "-i", str(music_path)]
        else:
            graph = None
            inputs = ["-i", str(vo)]
        before = tmp_path / "before.m4a"
        after = tmp_path / "after.m4a"
        if graph:
            _ffmpeg([*inputs, "-filter_complex", graph, "-map", "[aout]", "-c:a", "aac", str(before)])
        else:
            _ffmpeg([*inputs, "-c:a", "aac", str(before)])
        loud = A._loudness_step(vo, str(music_path) if music_path else None, dur, duck, start_at, gain_db)
        assert loud is not None, "the step must run on a real voiceover"
        if graph:
            _ffmpeg([*inputs, "-filter_complex", f"{graph};[aout]{loud}[afinal]", "-map", "[afinal]",
                     "-c:a", "aac", str(after)])
        else:
            _ffmpeg([*inputs, "-af", loud, "-c:a", "aac", str(after)])
        before_lufs, before_tp = _ebur(before)
        after_lufs, after_tp = _ebur(after)
        # The peak lands at the limit (minus the margin, and the AAC encode may move it a little)
        assert after_tp <= A.TRUE_PEAK_LIMIT_DBTP + MEASURE_DB
        assert after_tp >= A.TRUE_PEAK_LIMIT_DBTP - 0.4
        # The peak limit wins on a real voice: the level stays at or under the cap
        assert after_lufs <= A.LOUDNESS_MAX_LUFS
        # Not quieter than before, unless the input was already over the peak limit
        if before_tp <= A.TRUE_PEAK_LIMIT_DBTP:
            assert after_lufs > before_lufs
        return before_lufs, before_tp, after_lufs, after_tp

    @NEEDS_PULSAR
    def test_real_voiceover_with_pulsar_peak_limit_wins_and_stays_under_the_cap(self, tmp_path, voiceover):
        dur = self._duration(voiceover)
        duck = music.build_duck_filter([(0.0, dur)])
        entry = music._load_track_entries()["pulsar.mp3"]
        self._check_peak_limit_wins(tmp_path, voiceover, PULSAR, duck,
                                    entry["start_at"], entry["gain_db"])

    def test_real_voiceover_without_music_peak_limit_wins_and_stays_under_the_cap(self, tmp_path, voiceover):
        self._check_peak_limit_wins(tmp_path, voiceover, None, None, 0.0, 0.0)
