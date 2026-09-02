"""ffmpeg video assembly — frames + voiceover + music + captions."""

from pathlib import Path

from .broll import animate_frame
from .config import MEDIA_DIR, VIDEO_WIDTH, VIDEO_HEIGHT, run_cmd
from .log import log


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


def _prepare_segment(frame: dict, out_path: Path, duration: float, effect: str) -> Path:
    """Turn one b-roll item (video clip or still image) into a fixed-duration
    portrait segment, ready for concatenation."""
    if frame.get("type") == "video":
        src = frame["path"]
        vf = (
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        )
        run_cmd([
            "ffmpeg", "-i", str(src), "-t", str(duration),
            "-vf", vf, "-an", "-r", "30", "-pix_fmt", "yuv420p",
            str(out_path), "-y", "-loglevel", "quiet",
        ])
    else:
        animate_frame(frame["path"], out_path, duration, effect)
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
) -> Path:
    """Assemble final video from b-roll (video clips + image fallbacks),
voiceover, captions, and music."""
    log("Assembling video...")
    duration = get_audio_duration(voiceover)
    frame_durations = _compute_frame_durations(len(frames), words, duration)
    effects = ["zoom_in", "pan_right", "zoom_out"]

    # Prepare each b-roll item as a segment sized to match the actual speech
    # it accompanies (falls back to an even split if no word timing is available)
    segments = []
    for i, frame in enumerate(frames):
        seg = out_dir / f"seg_{i}.mp4"
        _prepare_segment(frame, seg, frame_durations[i] + 0.1, effects[i % len(effects)])
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