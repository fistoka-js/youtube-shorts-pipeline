"""Extract companion Shorts from a produced long-form video, using the
shorts_cutpoints Claude identified at draft time and real Whisper word
timing to find each cutpoint's exact start/end in the final video."""

from pathlib import Path

from .config import run_cmd
from .log import log


def _compute_section_spans(sections: list, words: list[dict], total_duration: float) -> dict:
    """Real start/end time for every section (not just start, like chapters
    needs) - walks the same concatenated-narration word-count order used
    everywhere else in this pipeline."""
    spans = {}
    word_idx = 0
    for i, sec in enumerate(sections):
        sec_id = sec.get("id", "")
        sec_word_count = len(sec.get("narration", "").split())
        start_idx = min(word_idx, len(words) - 1)
        end_idx = min(word_idx + sec_word_count - 1, len(words) - 1)
        start_time = words[start_idx]["start"] if start_idx < len(words) else 0.0
        end_time = words[end_idx]["end"] if end_idx < len(words) else total_duration
        spans[sec_id] = (start_time, end_time)
        word_idx += sec_word_count
    return spans


def _trim_and_crop_to_portrait(video_path: Path, out_path: Path, start: float, end: float):
    """Trim the landscape long-form video to [start, end] and center-crop
    to portrait 1080x1920 for Shorts. Captions already burned into the
    source will end up roughly centered but not resized for portrait -
    a reasonable first pass, not pixel-perfect Shorts styling."""
    duration = max(end - start, 1.0)
    vf = "crop=ih*9/16:ih,scale=1080:1920"
    run_cmd([
        "ffmpeg", "-ss", str(start), "-i", str(video_path), "-t", str(duration),
        "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out_path), "-y", "-loglevel", "quiet",
    ])


def extract_shorts_from_cutpoints(draft: dict, video_path: Path, out_dir: Path) -> list[dict]:
    """Extract one Short per shorts_cutpoint in the draft.

    Returns a list of dicts: {"path": Path, "section_ids": [...], "reason": str}
    """
    sections = draft.get("sections", [])
    cutpoints = draft.get("shorts_cutpoints", [])
    state = draft.get("_pipeline_state", {})
    words = state.get("captions", {}).get("artifacts", {}).get("words")

    if not words:
        log("No word timestamps available - cannot extract Shorts (re-run produce first)")
        return []
    if not cutpoints:
        log("No shorts_cutpoints in this draft - nothing to extract")
        return []

    from .assemble import get_audio_duration
    total_duration = get_audio_duration(video_path)

    spans = _compute_section_spans(sections, words, total_duration)

    results = []
    for i, cp in enumerate(cutpoints):
        section_ids = cp.get("section_ids", [])
        valid_spans = [spans[sid] for sid in section_ids if sid in spans]
        if not valid_spans:
            log(f"Cutpoint {i+1}: no matching sections found, skipping")
            continue

        start = min(s for s, e in valid_spans)
        end = max(e for s, e in valid_spans)
        clip_duration = end - start

        if clip_duration < 15 or clip_duration > 90:
            log(f"Cutpoint {i+1}: duration {clip_duration:.1f}s outside 15-90s Shorts range, skipping")
            continue

        out_path = out_dir / f"short_{i+1}_{'_'.join(section_ids)}.mp4"
        log(f"Extracting Short {i+1}: {section_ids} ({clip_duration:.1f}s, {start:.1f}s-{end:.1f}s)")
        _trim_and_crop_to_portrait(video_path, out_path, start, end)

        results.append({
            "path": out_path,
            "section_ids": section_ids,
            "reason": cp.get("reason", ""),
        })

    return results
