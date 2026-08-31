"""Whisper word-level timestamps + ASS subtitle generation + Pillow fallback."""

import difflib
import re
from pathlib import Path

from .log import log


def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation, for comparison purposes only."""
    return re.sub(r"[^\w']", "", w).lower()


def _align_to_script(whisper_words: list[dict], script_text: str) -> list[dict]:
    """Correct Whisper's transcribed word TEXT using the real script as
    ground truth, while keeping Whisper's real timestamps.

    Whisper knows WHEN words were spoken (from the real audio) but can
    mishear common words (e.g. "purr" -> "per") or occasionally hallucinate
    phrases that were never said. Since we know the exact script that was
    fed to the TTS engine, we align Whisper's words against the script's
    words and:
      - where they line up 1:1 but disagree (a mishearing), keep Whisper's
        timestamp but use the script's word
      - where Whisper has an extra word the script doesn't (a
        hallucination), drop it
      - where the script has a word Whisper missed, insert it with a
        timestamp interpolated between its neighbors
      - where a mismatched stretch can't be cleanly matched word-for-word,
        leave Whisper's original words alone rather than risk a wrong guess
    """
    if not script_text or not whisper_words:
        return whisper_words

    script_words = script_text.split()
    if not script_words:
        return whisper_words

    whisper_norm = [_normalize_word(w["word"]) for w in whisper_words]
    script_norm = [_normalize_word(w) for w in script_words]

    matcher = difflib.SequenceMatcher(None, whisper_norm, script_norm, autojunk=False)
    corrected = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            corrected.extend(whisper_words[i1:i2])

        elif tag == "replace":
            w_slice = whisper_words[i1:i2]
            s_slice = script_words[j1:j2]
            if len(w_slice) == len(s_slice):
                for ww, sw in zip(w_slice, s_slice):
                    corrected.append({"word": sw, "start": ww["start"], "end": ww["end"]})
            else:
                corrected.extend(w_slice)

        elif tag == "delete":
            continue  # hallucinated word(s) - drop

        elif tag == "insert":
            prev_end = corrected[-1]["end"] if corrected else 0.0
            next_start = whisper_words[i2]["start"] if i2 < len(whisper_words) else prev_end + 1.0
            missing = script_words[j1:j2]
            span = max(next_start - prev_end, 0.05)
            step = span / max(len(missing), 1)
            for k, sw in enumerate(missing):
                corrected.append({
                    "word": sw,
                    "start": prev_end + step * k,
                    "end": prev_end + step * (k + 1),
                })

    return corrected


def _has_ass_filter() -> bool:
    """Check if ffmpeg has libass (for ASS subtitle burn-in)."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=5,
        )
        return "ass" in r.stdout
    except Exception:
        return False


def _whisper_word_timestamps(audio_path: Path, lang: str = "en", script_text: str = "") -> list[dict]:
    """Get word-level timestamps from Whisper.

    Returns list of {"word": str, "start": float, "end": float}.
    """
    try:
        import whisper
    except ImportError:
        log("Whisper not installed — skipping word timestamps")
        return []

    log("Running Whisper for word-level timestamps...")
    model = whisper.load_model("base")
    # NOTE: we deliberately do NOT pass initial_prompt. Whisper treats it as
    # speech that happened immediately before the audio and tries to
    # "continue" from it - feeding it script text causes Whisper to skip
    # re-transcribing matching audio and can hallucinate nearby text.
    # Instead we transcribe with no bias at all, then correct known-good
    # vocabulary afterward by aligning against the real script (see
    # _align_to_script below) - this can't reintroduce that bug since it
    # never touches Whisper's own decoding.
    result = model.transcribe(
        str(audio_path),
        language=lang[:2],
        word_timestamps=True,
    )

    raw_words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            raw_words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    # Merge tokens that Whisper split on a hyphen (e.g. "cry" + "-like" -> "cry-like")
    merged = []
    for w in raw_words:
        if merged and w["word"].startswith("-"):
            merged[-1]["word"] += w["word"]
            merged[-1]["end"] = w["end"]
        else:
            merged.append(w)

    # Drop any fully-empty tokens before alignment
    cleaned = [w for w in merged if w["word"].strip()]

    # Correct Whisper's text against the real script (ground truth), fixing
    # mishearings and dropping hallucinated words, while keeping Whisper's
    # own real timestamps.
    cleaned = _align_to_script(cleaned, script_text)

    # Strip stray em/en dashes AFTER alignment, since the script's own words
    # (pulled in by _align_to_script) can reintroduce them.
    for w in cleaned:
        w["word"] = w["word"].replace("—", "").replace("–", "").strip()
    cleaned = [w for w in cleaned if w["word"]]

    # Force strictly increasing, non-overlapping timestamps. Whisper occasionally
    # emits out-of-order or overlapping words on fast/mumbled speech; without this,
    # multiple captions can render simultaneously ("flashing").
    fixed = []
    prev_end = 0.0
    for w in cleaned:
        start = max(w["start"], prev_end)
        end = max(w["end"], start + 0.05)
        fixed.append({"word": w["word"], "start": start, "end": end})
        prev_end = end

    # Make each word's display window run continuously until the next word starts,
    # so there is never a blank gap on screen between words — except when the real
    # gap is long (a genuine pause), in which case we cap display time instead of
    # holding one word artificially long.
    # Estimate a realistic seconds-per-character rate from words whose original
    # Whisper timing looks trustworthy (not suspiciously compressed).
    reliable = [w for w in fixed if (w["end"] - w["start"]) >= 0.08]
    if reliable:
        total_chars = sum(len(w["word"]) for w in reliable)
        total_time = sum(w["end"] - w["start"] for w in reliable)
        sec_per_char = (total_time / total_chars) if total_chars else 0.05
    else:
        sec_per_char = 0.05
    sec_per_char = max(sec_per_char, 0.02)  # sane floor

    max_display = 1.2  # long genuine pauses don't leave one word "stuck" on screen
    words = []
    cursor = 0.0
    i = 0
    while i < len(fixed):
        w = fixed[i]
        orig_dur = w["end"] - w["start"]
        if orig_dur < 0.08:
            # Compressed cluster: gather consecutive suspect words and redistribute
            # time proportional to word length, using the estimated speaking rate.
            cluster = [w]
            j = i + 1
            while j < len(fixed) and (fixed[j]["end"] - fixed[j]["start"]) < 0.08:
                cluster.append(fixed[j])
                j += 1
            start = max(cluster[0]["start"], cursor)
            for cw in cluster:
                dur = max(len(cw["word"]) * sec_per_char, 0.12)
                end = start + dur
                words.append({"word": cw["word"], "start": start, "end": end})
                cursor = end
                start = end
            i = j
        else:
            start = max(w["start"], cursor)
            if i + 1 < len(fixed):
                natural_end = fixed[i + 1]["start"]
            else:
                natural_end = w["end"]
            end = max(natural_end, start + 0.15)
            end = min(end, start + max_display)
            if end <= start:
                end = start + 0.05
            words.append({"word": w["word"], "start": start, "end": end})
            cursor = end
            i += 1

    log(f"Got {len(words)} word timestamps.")
    return words

def _group_words(words: list[dict], group_size: int = 4) -> list[list[dict]]:
    groups = []
    current = []
    for w in words:
        current.append(w)
        text = w.get("word", "").strip()
        ends_sentence = text.endswith((".", "!", "?"))
        if len(current) >= group_size or ends_sentence:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS timestamp: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _generate_ass(
    words: list[dict],
    output_path: Path,
    video_width: int = 1080,
    video_height: int = 1920,
    highlight_color: str = "#FFFF00",
    group_size: int = 4,
    font_family: str = "Arial",
    font_size: int = 72,
):
    """Generate ASS subtitle file with word-by-word color highlighting.

    White text for inactive words, highlight color for current word.
    Semi-transparent background, positioned at lower third (~70% down).

    The font_family is taken from the niche profile (captions.font_family) so
    non-Latin scripts (Korean, Japanese, Chinese, Arabic, etc.) can render
    correctly. The default "Arial" preserves the original behavior for English.
    """
    # ASS header
    margin_v = int(video_height * 0.25)  # ~75% down from top = 25% from bottom
    header = f"""[Script Info]
Title: Pipeline Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,3,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Convert hex color to ASS BGR format (e.g. #00FF88 -> 88FF00).
    # Override tags use &HBBGGRR& without the alpha byte.
    hc = highlight_color.lstrip("#")
    if len(hc) == 6:
        ass_highlight = f"&H{hc[4:6]}{hc[2:4]}{hc[0:2]}&"
    else:
        ass_highlight = "&H00FFFF&"  # fallback yellow

    groups = _group_words(words, group_size=group_size)
    events = []

    for group in groups:
        if not group:
            continue

        group_start = group[0]["start"]
        group_end = group[-1]["end"]

        # For each word in the group being active, emit one dialogue line
        for active_idx, active_word in enumerate(group):
            start = active_word["start"]
            end = active_word["end"]

            # Build text with override tags: highlight color for active, white for rest
            parts = []
            for j, w in enumerate(group):
                if j == active_idx:
                    parts.append(f"{{\\c{ass_highlight}\\b1\\fs80}}{w['word']}{{\\r}}")
                else:
                    parts.append(w["word"])

            text = " ".join(parts)
            events.append(
                f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{text}"
            )

    output_path.write_text(header + "\n".join(events), encoding="utf-8")
    log(f"ASS captions saved: {output_path.name}")
    return output_path


def _generate_srt(words: list[dict], output_path: Path, group_size: int = 4) -> Path:
    """Generate standard SRT file from word timestamps."""
    groups = _group_words(words, group_size=group_size)
    lines = []

    for i, group in enumerate(groups, 1):
        if not group:
            continue
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"] for w in group)

        start_ts = _srt_time(start)
        end_ts = _srt_time(end)
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"SRT captions saved: {output_path.name}")
    return output_path


def _srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_captions(
    audio_path: Path,
    work_dir: Path,
    lang: str = "en",
    highlight_color: str = "#FFFF00",
    words_per_group: int = 4,
    font_family: str = "Arial",
    font_size: int = 72,
    script_text: str = "",
) -> dict:
    """Generate captions: ASS (for burn-in) + SRT (for YouTube upload).

    Args:
        font_family: ASS Style font name. Use a CJK-capable font (e.g.
            "Noto Sans CJK KR", "Noto Sans CJK JP") for non-Latin languages,
            otherwise glyphs render as boxes. Pulled from the niche profile's
            captions.font_family field.
        font_size: ASS Style font size. Pulled from the niche profile's
            captions.font_size field.

    Returns dict with keys: srt_path, ass_path, words (for music ducking).
    """
    words = _whisper_word_timestamps(audio_path, lang, script_text=script_text)

    result = {"words": words}

    if not words:
        log("No word timestamps — skipping caption generation")
        # Fallback: run whisper CLI for SRT only
        try:
            from .config import run_cmd
            run_cmd([
                "whisper", str(audio_path),
                "--model", "base",
                "--language", lang[:2],
                "--output_format", "srt",
                "--output_dir", str(work_dir),
            ], capture=True)
            candidates = list(work_dir.glob("*.srt"))
            if candidates:
                srt = candidates[0]
                final = audio_path.with_suffix(".srt")
                srt.rename(final)
                result["srt_path"] = str(final)
        except Exception as e:
            log(f"Whisper CLI fallback failed: {e}")
        return result

    # Generate SRT
    srt_path = work_dir / f"captions_{lang}.srt"
    _generate_srt(words, srt_path, group_size=words_per_group)
    result["srt_path"] = str(srt_path)

    # Generate ASS for burn-in (niche-aware highlight color)
    ass_path = work_dir / f"captions_{lang}.ass"
    _generate_ass(
        words, ass_path,
        highlight_color=highlight_color,
        group_size=words_per_group,
        font_family=font_family,
        font_size=font_size,
    )
    result["ass_path"] = str(ass_path)

    return result
