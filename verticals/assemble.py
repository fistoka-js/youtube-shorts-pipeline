"""ffmpeg video assembly — frames + voiceover + music + captions."""

from pathlib import Path

from .broll import animate_frame
from .config import MEDIA_DIR, VIDEO_WIDTH, VIDEO_HEIGHT, run_cmd
from .log import log
from .section_timing import compute_section_timings

# Frame rate of every segment (see _prepare_segment: -r 30 / animate_frame fps=30).
FPS = 30


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

    Video clips are downloaded at a fixed short duration (see broll.py's
    clip_duration), but the actual duration needed here is computed from
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


def assemble_video(
    frames: list[dict],
    voiceover: Path,
    out_dir: Path,
    job_id: str,
    lang: str = "en",
    ass_path: str | None = None,
    music_path: str | None = None,
    duck_filter: str | None = None,
    title: str | None = None,
    words: list[dict] | None = None,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    sections: list[dict] | None = None,
) -> Path:
    """Assemble final video from b-roll (video clips + image fallbacks),
voiceover, captions, and music. Defaults to portrait (Shorts); pass
width/height=1920/1080 for landscape long-form output.

sections: long-form only. When given, clips switch at each section's real
start (see long_form_frame_counts) and each clip gets an exact frame count.
When omitted (Shorts), the original equal word-chunk timing is used."""
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
    for i, frame in enumerate(frames):
        seg = out_dir / f"seg_{i}.mp4"
        if frame_counts is None:
            _prepare_segment(frame, seg, frame_durations[i] + 0.1, effects[i % len(effects)], width, height)
        else:
            _prepare_segment(frame, seg, frame_counts[i] / FPS, effects[i % len(effects)], width, height,
                             frame_count=frame_counts[i])
        segments.append(seg)

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

    if music_path and Path(music_path).exists():
        cmd = ["ffmpeg", "-i", str(merged_video), "-i", str(voiceover)]

        music_filter = f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{duration}"
        if duck_filter:
            music_filter += f",{duck_filter}"
        music_filter += "[music]"

        audio_filter = f"{music_filter};[1:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"

        cmd += [
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", audio_filter,
        ]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]
    else:
        cmd = ["ffmpeg", "-i", str(merged_video), "-i", str(voiceover)]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-c:v", "libx264" if vf else "copy",
            "-c:a", "aac", "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]

    run_cmd(cmd)
    log(f"Video assembled: {out_path}")
    return out_path