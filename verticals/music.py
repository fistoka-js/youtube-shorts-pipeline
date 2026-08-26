"""Background music — track selection + volume ducking."""

import random
from pathlib import Path

from .log import log

# Music directory ships with the package
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"


def _find_tracks() -> list[Path]:
    """Find all MP3 tracks in the music/ directory."""
    if not MUSIC_DIR.exists():
        return []
    return sorted(MUSIC_DIR.glob("*.mp3"))


def _get_speech_regions(audio_path: Path) -> list[tuple[float, float]]:
    """Extract speech regions from Whisper word timestamps (reuses captions data).

    Falls back to treating the entire audio as one speech region.
    """
    try:
        from .captions import _whisper_word_timestamps
        words = _whisper_word_timestamps(audio_path)
        if words:
            # Merge close words into speech regions (gap < 0.5s = same region)
            regions = []
            region_start = words[0]["start"]
            region_end = words[0]["end"]

            for w in words[1:]:
                if w["start"] - region_end < 0.5:
                    region_end = w["end"]
                else:
                    regions.append((region_start, region_end))
                    region_start = w["start"]
                    region_end = w["end"]
            regions.append((region_start, region_end))
            return regions
    except Exception:
        pass

    # Fallback: get total duration and treat as one speech region
    try:
        from .assemble import get_audio_duration
        dur = get_audio_duration(audio_path)
        return [(0.0, dur)]
    except Exception:
        return [(0.0, 60.0)]


def build_duck_filter(speech_regions: list[tuple[float, float]], buffer: float = 0.3, vol_speech: float = 0.12, vol_gap: float = 0.25) -> str:
    """Build ffmpeg volume filter expression for ducking during speech.

    During speech: volume = vol_speech (default 0.12)
    During gaps: volume = vol_gap (default 0.25)
    Transitions smoothed by ±buffer seconds.
    """
    if not speech_regions:
        return f"volume={vol_gap}"

    # Build between() conditions for speech regions
    conditions = []
    for start, end in speech_regions:
        s = max(0, start - buffer)
        e = end + buffer
        conditions.append(f"between(t,{s:.2f},{e:.2f})")

    condition_expr = "+".join(conditions)
    return f"volume='if({condition_expr}, {vol_speech}, {vol_gap})':eval=frame"


def _score_track(track: Path, keywords: list[str]) -> int:
    """Score a track filename against mood/tag keywords (case-insensitive substring match)."""
    name = track.stem.lower().replace("-", " ").replace("_", " ")
    score = 0
    for kw in keywords:
        kw = kw.lower().strip()
        if kw and kw in name:
            score += 1
    return score


def select_and_prepare_music(
    voiceover_path: Path,
    work_dir: Path,
    duck_speech: float = 0.12,
    duck_gap: float = 0.25,
    mood_keywords: list[str] | None = None,
) -> dict:
    """Select a track matching the niche's mood keywords (falls back to random
    among top scorers, or fully random if nothing matches), build duck filter.

    Returns dict with track_path and duck_filter for use by assemble.py.
    """
    tracks = _find_tracks()
    if not tracks:
        log("No music tracks found in music/ — skipping background music")
        return {}

    if mood_keywords:
        scored = [(t, _score_track(t, mood_keywords)) for t in tracks]
        best_score = max(s for _, s in scored)
        if best_score > 0:
            candidates = [t for t, s in scored if s == best_score]
            track = random.choice(candidates)
            log(f"Selected music track (mood match, score {best_score}): {track.name}")
        else:
            track = random.choice(tracks)
            log(f"No mood match found — random pick: {track.name}")
    else:
        track = random.choice(tracks)
        log(f"Selected music track: {track.name}")

    # Get speech regions for ducking
    speech_regions = _get_speech_regions(voiceover_path)
    duck_filter = build_duck_filter(speech_regions, vol_speech=duck_speech, vol_gap=duck_gap)
    log(f"Built duck filter with {len(speech_regions)} speech regions")

    return {
        "track_path": str(track),
        "duck_filter": duck_filter,
    }
