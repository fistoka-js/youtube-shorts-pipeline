"""ffmpeg video assembly — frames + voiceover + music + captions."""

import json
import math
import re
import subprocess
from pathlib import Path

from .broll import animate_frame
from .config import MEDIA_DIR, VIDEO_WIDTH, VIDEO_HEIGHT, run_cmd
from .log import log
from .section_timing import compute_section_timings

# Frame rate of every segment (see _prepare_segment: -r 30 / animate_frame fps=30).
FPS = 30

# Final loudness step, applied to the finished audio with and without music.
# The goal is: as loud as the true-peak limit below allows. With one linear gain
# (which keeps the voice-to-music balance), real voiceovers measured so far land at about
# -19 to -20 LUFS. Reasoning: real speech peaks sit far above its average level, so the
# peak limit stops the gain before any loudness target is reached.
# LOUDNESS_MAX_LUFS (-14 LUFS, the YouTube reference) is only an UPPER CAP: the gain never
# takes the level above it. It can be reached only by quiet-peaked material. Going closer
# to -14 on real voices would need a limiter or dynamic loudnorm, which would change the
# voice-to-music balance.
# The log line from _loudness_step shows the level actually chosen for each video.
# The -14 cap and the -1.5 dBTP limit are choices, not measured.
LOUDNESS_MAX_LUFS = -14.0
# The true-peak limit (-1.5 dBTP is the headroom, a choice).
TRUE_PEAK_LIMIT_DBTP = -1.5
# LRA is the EBU R128 streaming value. It only matters if loudnorm uses dynamic mode,
# which this step never does.
LOUDNESS_LRA_TARGET = 11.0
# Headroom kept below the peak limit (a choice). The gain is sized to peak at
# TRUE_PEAK_LIMIT_DBTP minus this. Loudnorm switches to dynamic mode when the gain
# reaches the limit exactly, so a small margin keeps it linear. Encoded AAC and mp3
# output in the measurements so far stayed at or below the limit, so the margin is
# enough for this material.
LOUDNESS_PEAK_MARGIN_DB = 0.1
# Time limit for EACH measuring pass (a choice). It is audio only, so it is quick. A render
# with music runs two passes (the voiceover alone, then the mix), so the worst case is twice
# this value. Real runs take about 10 s in total on a 10 minute voiceover.
LOUDNESS_MEASURE_TIMEOUT_SECONDS = 300
# Guards on the gain (choices). A near-silent input, for example a failed voice render,
# must never be boosted by an absurd amount. The step is skipped, with a warning, if the
# measured loudness is below LOUDNESS_MIN_INPUT_LUFS or the gain needed is above
# LOUDNESS_MAX_BOOST_DB. Real voiceovers measured so far need a gain of roughly 0 to +8 dB
# (a voice already over the peak limit needs a little less than 0), and the level of
# real mixes is about -19 to -27 LUFS. So the 20 dB cap and the -50 LUFS floor are far
# outside normal use.
LOUDNESS_MIN_INPUT_LUFS = -50.0
LOUDNESS_MAX_BOOST_DB = 20.0
# Floor for the VOICEOVER on its own (a choice). Checked before anything else, in both
# branches. If the voiceover is silent or too quiet to measure, or below this, the step is
# skipped, so with music the music is never boosted on its own. Real voiceovers measured so
# far land at about -19 to -21 LUFS, so -45 is far below normal use. The voiceover-only pass
# is quick: about 5 s on a 10 minute voiceover in the timing so far.
LOUDNESS_MIN_VOICE_LUFS = -45.0
# Loudness is measured over 400 ms blocks (ITU-R BS.1770), so a shorter clip has no
# measurable level. This is a fact of the standard, not a choice.
LOUDNESS_MIN_MEASURABLE_SECONDS = 0.4


def _ffmpeg_has_libass() -> bool:
    """Check whether this ffmpeg build ships the `ass` filter (libass)."""
    try:
        r = run_cmd(["ffmpeg", "-hide_banner", "-filters"], capture=True)
        return any(line.split()[1:2] == ["ass"] for line in r.stdout.splitlines())
    except Exception:
        return False


def get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds."""
    r = run_cmd(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture=True,
    )
    return float(r.stdout.strip())


def _prepare_segment(frame: dict, out_path: Path, duration: float, effect: str,
                     width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT,
                     frame_count: int | None = None) -> Path:
    """Turn one b-roll item (video clip or still image) into a fixed-duration
    segment at the target dimensions, ready for concatenation.

    Video clips are downloaded at their own length, capped at broll.py's
    MAX_CLIP_SECONDS (Shorts), or at a fixed length for long-form, but the
    actual duration needed here is computed from
    real speech timing and can be much longer (especially for long-form,
    where ~50 clips share ~10 minutes of narration). -stream_loop makes the
    source clip repeat as many times as needed to fill the requested
    duration, instead of silently truncating to whatever's on disk.
    """
    if frame.get("type") == "video":
        src = frame["path"]
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        # frame_count (long-form) gives an exact number of frames; otherwise
        # the length is set by -t as before (Shorts path).
        length_args = ["-frames:v", str(frame_count)] if frame_count is not None else ["-t", str(duration)]
        run_cmd([
            "ffmpeg", "-stream_loop", "-1", "-i", str(src), *length_args,
            "-vf", vf, "-an", "-r", "30", "-pix_fmt", "yuv420p",
            str(out_path), "-y", "-loglevel", "quiet",
        ])
    else:
        animate_frame(frame["path"], out_path, duration, effect, width, height,
                      frames=frame_count)
    return out_path

def _compute_frame_durations(frame_count: int, words: list[dict] | None, total_duration: float) -> list[float]:
    """Compute a duration per b-roll clip based on actual speech timing.

    Splits the transcript's words into frame_count contiguous chunks by word
    count, then uses each chunk's real start time as that clip's boundary -
    so a clip lasts as long as the words it's meant to accompany actually
    took to speak, instead of an even average across the whole video.

    Falls back to a flat average if no word timestamps are available.
    """
    if not words or frame_count <= 0:
        per_frame = total_duration / max(frame_count, 1)
        return [per_frame] * frame_count

    word_count = len(words)
    chunk_size = word_count / frame_count
    boundaries = []
    for i in range(frame_count):
        idx = min(round(i * chunk_size), word_count - 1)
        boundaries.append(words[idx]["start"])
    boundaries.append(total_duration)

    durations = []
    for i in range(frame_count):
        d = boundaries[i + 1] - boundaries[i]
        durations.append(max(d, 0.5))  # floor so no clip is near-zero length
    return durations


def _frame_counts_from_bounds(bounds: list[float]) -> list[int]:
    """Turn clip boundary times (n+1 values, in seconds, nondecreasing) into a
    whole number of 1/30 s frames per clip.

    Each boundary is snapped to the frame grid using the running total, so the
    rounding error never accumulates: the frame counts add up to exactly
    round(last boundary * 30). Every clip gets at least one frame.
    """
    grid = [round(b * FPS) for b in bounds]
    return [max(grid[i + 1] - grid[i], 1) for i in range(len(bounds) - 1)]


def _equal_chunk_frame_counts(frame_count: int, words: list[dict] | None,
                              total_duration: float) -> list[int]:
    """Frame counts from the old equal word-chunk split (no padding)."""
    durs = _compute_frame_durations(frame_count, words, total_duration)
    bounds = [0.0]
    for d in durs:
        bounds.append(bounds[-1] + d)
    bounds[-1] = total_duration
    return _frame_counts_from_bounds(bounds)


def long_form_frame_counts(frames: list[dict], sections: list[dict],
                           words: list[dict] | None, total_duration: float) -> list[int]:
    """Frames per clip for a long-form video, so footage switches at section starts.

    Each section's real start comes from the shared section_timing helper. A
    section's time runs from its start to the next section that has clips (the
    last one runs to the audio end), and its clips split that time into equal
    shares, so its first clip starts at the section start. Sections with no
    clips give their time to the section before them.

    The clips must match the draft: each section's clips (grouped by section_id,
    in order) must number the same as its broll_prompts. If not, a warning is
    logged and the old equal word-chunk split is used instead, so nothing crashes.
    """
    n = len(frames)
    if n == 0:
        return []
    if not sections or not words:
        log("WARNING: long-form clip timing needs sections and word timestamps - "
            "using the old equal word-chunk split")
        return _equal_chunk_frame_counts(n, words, total_duration)

    def _fallback(reason: str) -> list[int]:
        log(f"WARNING: long-form clip timing fell back to the old equal word-chunk split: {reason}")
        return _equal_chunk_frame_counts(n, words, total_duration)

    section_ids = {sec.get("id", "") for sec in sections}
    per_section: dict[str, list[int]] = {}
    last_sid = None
    finished = set()
    for i, f in enumerate(frames):
        sid = f.get("section_id", "")
        if sid not in section_ids:
            return _fallback(f"clip {i + 1} has unknown section_id {sid!r}")
        if sid != last_sid:
            if sid in finished:
                return _fallback(f"clips for section {sid} are not all together")
            if last_sid is not None:
                finished.add(last_sid)
            last_sid = sid
        per_section.setdefault(sid, []).append(i)

    for sec in sections:
        sid = sec.get("id", "")
        got = len(per_section.get(sid, []))
        expected = len(sec.get("broll_prompts", []) or [])
        if got != expected:
            return _fallback(f"section {sid} has {got} clips but {expected} broll prompts")

    timings = compute_section_timings(sections, words)
    start_of = {t["id"]: t["start"] for t in timings}
    clip_sections = [sec.get("id", "") for sec in sections if per_section.get(sec.get("id", ""))]

    clip_start = [0.0] * n  # start time of each clip
    for pos, sid in enumerate(clip_sections):
        idxs = per_section[sid]
        start = start_of[sid]
        if pos + 1 < len(clip_sections):
            end = start_of[clip_sections[pos + 1]]
        else:
            end = total_duration
        end = max(end, start)
        share = (end - start) / len(idxs)
        for c, i in enumerate(idxs):
            clip_start[i] = start + c * share
    clip_start[0] = 0.0  # the first clip always begins at the very start

    if any(clip_start[i] < clip_start[i - 1] for i in range(1, n)):
        return _fallback("section starts are out of order in the clip list")

    counts = _frame_counts_from_bounds(clip_start + [total_duration])
    expected_total = round(total_duration * FPS)
    if sum(counts) != expected_total:
        log(f"WARNING: long-form clip frames add up to {sum(counts)}, expected {expected_total} "
            f"(audio {total_duration:.3f}s at {FPS} fps)")
    return counts


# Music fades: 1 s in at the start, 2 s out at the end of the video
MUSIC_FADE_IN_SECONDS = 1.0
MUSIC_FADE_OUT_SECONDS = 2.0


def build_audio_filter(duration: float, duck_filter: str | None = None,
                       start_at: float = 0.0, gain_db: float = 0.0,
                       voice_index: int = 1, music_index: int = 2) -> str:
    """Filter graph that mixes the music under the voice. Used by assemble_video
    and by scripts/audition_music_levels.py, so both render the same mix.

    voice_index and music_index are the ffmpeg input numbers. assemble_video has
    the video as input 0, the voiceover as 1 and the music as 2. The music starts
    at start_at seconds into the track and loops from there (not from 0 s). It
    gets gain_db, then the duck volume, then a fade in and out. amix gives both
    inputs the same weight, so the voice-to-music difference is kept.
    """
    fade_out_start = max(duration - MUSIC_FADE_OUT_SECONDS, 0.0)
    parts = [
        f"[{music_index}:a]atrim=start={start_at:.3f},asetpts=PTS-STARTPTS",
        "aloop=loop=-1:size=2e+09",
        f"atrim=0:{duration:.3f}",
        f"volume={gain_db:.2f}dB",
    ]
    if duck_filter:
        parts.append(duck_filter)
    parts.append(f"afade=t=in:st=0:d={MUSIC_FADE_IN_SECONDS:g}")
    parts.append(f"afade=t=out:st={fade_out_start:.3f}:d={MUSIC_FADE_OUT_SECONDS:g}")
    music_filter = ",".join(parts) + "[music]"
    return (f"{music_filter};[{voice_index}:a][music]"
            f"amix=inputs=2:duration=first:dropout_transition=2[aout]")


class LoudnessOutOfRange(ValueError):
    """The measured audio is outside the floor or the maximum boost. The step is skipped."""


def _loudnorm_measure_filter() -> str:
    """Pass 1 of the loudness step. loudnorm only measures here; its output is thrown away."""
    return (f"loudnorm=I={LOUDNESS_MAX_LUFS:g}:TP={TRUE_PEAK_LIMIT_DBTP:g}"
            f":LRA={LOUDNESS_LRA_TARGET:g}:print_format=json")


class SilentMeasurement(ValueError):
    """loudnorm measured -inf: the audio has no level to measure (digital silence)."""


def _measured_stats(stderr: str) -> dict:
    """Read loudnorm's pass 1 JSON into {input_i, input_tp, input_lra, input_thresh}.

    Raises SilentMeasurement if a value is -inf (silence), and ValueError if the JSON is
    missing or a value is not a number."""
    stats = None
    for block in re.findall(r"\{[^{}]*\}", stderr):
        try:
            parsed = json.loads(block)
        except ValueError:
            continue
        if isinstance(parsed, dict) and "input_i" in parsed:
            stats = parsed
    if stats is None:
        raise ValueError("no loudnorm measurement in the ffmpeg output")
    measured = {}
    for key in ("input_i", "input_tp", "input_lra", "input_thresh"):
        value = float(stats[key])
        if not math.isfinite(value):
            raise SilentMeasurement(f"{key} is {stats[key]} (silent audio?)")
        measured[key] = value
    return measured


def _loudness_filter_from_measurement(stderr: str) -> tuple[str, float, float, float]:
    """Turn loudnorm's pass 1 JSON into the pass 2 filter.

    Returns (filter, target LUFS, measured LUFS, measured true peak). Raises ValueError
    if the JSON is missing or a value is not finite (for example -inf from silence).

    The goal is as loud as the true-peak limit allows. The gain is one number for the
    whole mix (linear mode), so the voice-to-music balance is kept. It is the gain that
    brings the true peak to TRUE_PEAK_LIMIT_DBTP (less LOUDNESS_PEAK_MARGIN_DB). It is
    never more than the cap LOUDNESS_MAX_LUFS, so it stops at the cap if the cap comes
    first (quiet-peaked material only). loudnorm is never asked for dynamic mode. If the
    input is already above the peak limit, the gain is negative: the level is LOWERED
    (the peak limit wins). Raises LoudnessOutOfRange if the input is below
    LOUDNESS_MIN_INPUT_LUFS or the gain needed is above LOUDNESS_MAX_BOOST_DB.
    """
    measured = _measured_stats(stderr)
    if measured["input_i"] < LOUDNESS_MIN_INPUT_LUFS:
        raise LoudnessOutOfRange(
            f"the audio measures {measured['input_i']:.1f} LUFS, below the floor of "
            f"{LOUDNESS_MIN_INPUT_LUFS:g} LUFS (near-silent?)")
    headroom_db = TRUE_PEAK_LIMIT_DBTP - LOUDNESS_PEAK_MARGIN_DB - measured["input_tp"]
    target = min(LOUDNESS_MAX_LUFS, measured["input_i"] + headroom_db)  # the level reached
    offset = target - measured["input_i"]
    if offset > LOUDNESS_MAX_BOOST_DB:
        raise LoudnessOutOfRange(
            f"the gain needed is {offset:+.1f} dB, above the maximum boost of "
            f"{LOUDNESS_MAX_BOOST_DB:g} dB")
    # loudnorm falls back to dynamic mode when the measured LRA is exactly 0 (a perfectly
    # steady tone; found by testing). A floor of 0.1 LU keeps it linear. Real voiceovers
    # measure about 2 LU, so the floor does not change them.
    measured_lra = max(measured["input_lra"], 0.1)
    pass2 = (f"loudnorm=I={target:.2f}:TP={TRUE_PEAK_LIMIT_DBTP:g}:LRA={LOUDNESS_LRA_TARGET:g}"
             f":measured_I={measured['input_i']:.2f}:measured_LRA={measured_lra:.2f}"
             f":measured_TP={measured['input_tp']:.2f}:measured_thresh={measured['input_thresh']:.2f}"
             f":offset={offset:.2f}:linear=true")
    return pass2, target, measured["input_i"], measured["input_tp"]


def _silence_reason(voiceover: Path) -> str:
    """Say why loudnorm measured -inf. A clip shorter than the 400 ms gating block cannot be
    measured (it may be audible, just too short). A longer clip with -inf is either digital
    silence or a level below loudnorm's absolute gate (-70 LUFS). The two cannot be told
    apart from this measurement, so the wording covers both."""
    try:
        seconds = get_audio_duration(voiceover)
    except Exception:
        seconds = None
    if seconds is not None and seconds < LOUDNESS_MIN_MEASURABLE_SECONDS:
        return (f"the voiceover is too short to measure ({seconds:.2f} s; loudness needs "
                f"at least {LOUDNESS_MIN_MEASURABLE_SECONDS:g} s)")
    return "the voiceover is silent or too quiet to measure"


def _failure_text(error: Exception) -> str:
    """A short, readable sentence for the log. It never includes the command line, a file
    path, or ffmpeg's own message (those are long and hard to read)."""
    if isinstance(error, subprocess.TimeoutExpired):
        seconds = getattr(error, "timeout", None) or LOUDNESS_MEASURE_TIMEOUT_SECONDS
        return f"the measuring pass timed out after {seconds:g} seconds"
    if isinstance(error, RuntimeError):
        return "the measuring pass failed (ffmpeg could not measure the audio)"
    return f"the measuring pass failed ({type(error).__name__})"


def _loudness_step(voiceover: Path, music_path: str | None, duration: float,
                   duck_filter: str | None, start_at: float, gain_db: float) -> str | None:
    """Pass 1 of the final loudness step. Measures the finished audio (the voiceover,
    or the voiceover mixed with the music, built by build_audio_filter exactly as the
    final render builds it), audio only, to a null output. Returns the loudnorm filter
    for the final render, or None if the measurement failed. A warning is logged in
    that case and the audio is kept as it is. Never raises."""
    def _measure(inputs: list[str]) -> str:
        r = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", *inputs,
                     "-vn", "-f", "null", "-"],
                    capture=True, timeout=LOUDNESS_MEASURE_TIMEOUT_SECONDS,
                    encoding="utf-8", errors="replace")
        return r.stderr

    try:
        measure = _loudnorm_measure_filter()
        # An empty file is reported in words, not as an ffmpeg error.
        try:
            empty = Path(voiceover).stat().st_size == 0
        except OSError:
            empty = False
        if empty:
            raise LoudnessOutOfRange("the audio file is empty or cannot be read")
        # Guard first, on the voiceover alone. With music, a silent voiceover would leave only
        # the music, and the step would boost it. So a silent or near-silent voiceover skips the step.
        try:
            voice_stderr = _measure(["-i", str(voiceover), "-af", measure])
        except RuntimeError as e:
            # ffmpeg could not decode the file (corrupt, unreadable or missing). Its exit
            # code is not useful to the reader, so only the readable reason is logged.
            raise LoudnessOutOfRange("the audio file is empty or cannot be read") from e
        try:
            voice_lufs = _measured_stats(voice_stderr)["input_i"]
        except SilentMeasurement as e:
            raise LoudnessOutOfRange(_silence_reason(voiceover)) from e
        if voice_lufs < LOUDNESS_MIN_VOICE_LUFS:
            raise LoudnessOutOfRange(
                f"the voiceover alone measures {voice_lufs:.1f} LUFS, below the voice floor of "
                f"{LOUDNESS_MIN_VOICE_LUFS:g} LUFS (too quiet to use, so no music is boosted)")
        if music_path:
            # The same inputs as the final render. Do not add -t to the input: it changes the
            # format ffmpeg gives the mix (sample rate and channels), which changes the peaks.
            graph = build_audio_filter(duration, duck_filter, start_at, gain_db,
                                       voice_index=0, music_index=1)
            stderr = _measure(["-i", str(voiceover), "-i", str(music_path),
                               "-filter_complex", f"{graph};[aout]{measure}[lnout]",
                               "-map", "[lnout]"])
        else:
            stderr = voice_stderr  # with no music, the voiceover is the whole mix
        pass2, target, input_i, input_tp = _loudness_filter_from_measurement(stderr)
    except LoudnessOutOfRange as e:
        log(f"WARNING: loudness step skipped, {str(e)[:300]} - the audio is kept at its current level")
        return None
    except Exception as e:
        log(f"WARNING: loudness step skipped, {_failure_text(e)} "
            "- the audio is kept at its current level")
        return None
    offset = target - input_i
    if offset < 0:
        # The input is louder than the level reached, or its true peak is already above the
        # limit. The level comes DOWN. This can make a video quieter than before, which is
        # the intended result of the peak limit.
        if input_tp > TRUE_PEAK_LIMIT_DBTP - LOUDNESS_PEAK_MARGIN_DB:
            why = " (lowered: the true peak is already at or above the limit)"
        else:
            why = f" (lowered to the {LOUDNESS_MAX_LUFS:g} LUFS cap)"
    elif target < LOUDNESS_MAX_LUFS - 1e-9:
        why = " (stopped by the true-peak limit, so the cap is not reached)"
    else:
        why = f" (reached the {LOUDNESS_MAX_LUFS:g} LUFS cap)"
    what_was_measured = "mix" if music_path else "voiceover"
    log(f"Loudness: {what_was_measured} measures {input_i:.1f} LUFS, true peak {input_tp:.1f} dBTP; "
        f"setting it to {target:.1f} LUFS{why}")
    return pass2


def assemble_video(
    frames: list[dict],
    voiceover: Path,
    out_dir: Path,
    job_id: str,
    lang: str = "en",
    ass_path: str | None = None,
    music_path: str | None = None,
    duck_filter: str | None = None,
    start_at: float = 0.0,
    gain_db: float = 0.0,
    title: str | None = None,
    words: list[dict] | None = None,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    sections: list[dict] | None = None,
    slot_out: list | None = None,
) -> Path:
    """Assemble final video from b-roll (video clips + image fallbacks),
voiceover, captions, and music. Defaults to portrait (Shorts); pass
width/height=1920/1080 for landscape long-form output.

sections: long-form only. When given, clips switch at each section's real
start (see long_form_frame_counts) and each clip gets an exact frame count.
When omitted (Shorts), the original equal word-chunk timing is used.

slot_out: optional list. When given, the length in seconds of each slot's
segment (one per frame, in order) is appended to it, for the build report."""
    log("Assembling video...")
    duration = get_audio_duration(voiceover)
    effects = ["zoom_in", "pan_right", "zoom_out"]

    frame_counts = None
    frame_durations = None
    if sections is None:
        frame_durations = _compute_frame_durations(len(frames), words, duration)
    else:
        frame_counts = long_form_frame_counts(frames, sections, words, duration)

    # Prepare each b-roll item as a segment sized to match the actual speech
    # it accompanies (falls back to an even split if no word timing is available)
    segments = []
    slot_seconds = []  # length of each slot's segment, for the build report
    for i, frame in enumerate(frames):
        seg = out_dir / f"seg_{i}.mp4"
        if frame_counts is None:
            slot_seconds.append(frame_durations[i] + 0.1)
            _prepare_segment(frame, seg, frame_durations[i] + 0.1, effects[i % len(effects)], width, height)
        else:
            slot_seconds.append(frame_counts[i] / FPS)
            _prepare_segment(frame, seg, frame_counts[i] / FPS, effects[i % len(effects)], width, height,
                             frame_count=frame_counts[i])
        segments.append(seg)
    if slot_out is not None:
        slot_out.extend(slot_seconds)

    # Concat segments (escape single quotes for ffmpeg concat demuxer)
    concat_file = out_dir / "concat.txt"
    def _esc(p):
        return str(p).replace("'", "'\\''" )
    concat_file.write_text("\n".join(f"file '{_esc(p)}'" for p in segments))

    merged_video = out_dir / "merged_video.mp4"
    run_cmd([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        str(merged_video), "-y", "-loglevel", "quiet",
    ])

    # Build the final ffmpeg command with optional captions + music
    import re
    safe_title = re.sub(r'[^\w\s-]', '', title or job_id).strip()
    safe_title = re.sub(r'[\s_]+', '_', safe_title)
    if len(safe_title) > 50:
        safe_title = safe_title[:50].rsplit('_', 1)[0]
    safe_title = safe_title or job_id
    out_path = MEDIA_DIR / f"{safe_title}_{lang}.mp4"

    # Determine video filter (captions via ASS)
    vf_parts = []
    if ass_path and Path(ass_path).exists():
        if _ffmpeg_has_libass():
            escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            vf_parts.append(f"ass='{escaped_ass}'")
        else:
            log(
                "WARNING: this ffmpeg build has no libass — captions will NOT "
                "be burned in. The SRT is still uploaded to YouTube. Install "
                "an ffmpeg with libass (brew/apt builds include it) for "
                "burned-in captions."
            )
    vf = ",".join(vf_parts) if vf_parts else None

    has_music = bool(music_path and Path(music_path).exists())
    # Final loudness step: measured on the same audio that is rendered below (None = skipped)
    loudness = _loudness_step(voiceover, music_path if has_music else None, duration,
                              duck_filter, start_at, gain_db)

    if has_music:
        cmd = ["ffmpeg", "-i", str(merged_video), "-i", str(voiceover)]

        audio_filter = build_audio_filter(duration, duck_filter, start_at, gain_db)
        audio_out = "[aout]"
        if loudness:
            audio_filter += f";[aout]{loudness}[afinal]"
            audio_out = "[afinal]"

        cmd += [
            "-i", str(music_path),
            "-filter_complex", audio_filter,
        ]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-map", "0:v", "-map", audio_out,
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]
    else:
        cmd = ["ffmpeg", "-i", str(merged_video), "-i", str(voiceover)]

        if vf:
            cmd += ["-vf", vf]
        if loudness:
            cmd += ["-af", loudness]

        cmd += [
            "-c:v", "libx264" if vf else "copy",
            "-c:a", "aac", "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]

    run_cmd(cmd)
    log(f"Video assembled: {out_path}")
    return out_path