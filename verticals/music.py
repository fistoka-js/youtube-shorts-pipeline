"""Background music — track selection + volume ducking."""

import random
from pathlib import Path

import yaml

from .log import log

# Music directory ships with the package
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"

# Each track's mood tags live in music/tracks.yaml, keyed by file name
TRACK_TAGS_FILE_NAME = "tracks.yaml"

# Tag used for the default pick when no track matches the niche's mood
FALLBACK_TAG = "ambient"

# The moods a video can ask for (the Shorts draft's "music_mood" field)
VIDEO_MOODS = ("cosmic", "mysterious", "hopeful", "quirky")

# Points a track gets for having the video's mood as one of its tags
VIDEO_MOOD_BONUS = 10


def clean_video_mood(value) -> str | None:
    """Return the mood in lowercase if it is one of VIDEO_MOODS, else None."""
    if not isinstance(value, str):
        return None
    mood = value.strip().lower()
    return mood if mood in VIDEO_MOODS else None


def _find_tracks() -> list[Path]:
    """Find all MP3 tracks in the music/ directory."""
    if not MUSIC_DIR.exists():
        return []
    return sorted(MUSIC_DIR.glob("*.mp3"))


def _load_track_tags() -> dict[str, set[str]]:
    """Read music/tracks.yaml into {file name: set of lowercase tags}.

    Returns {} if the file is missing, unreadable or not in the expected shape.
    With no tags loaded, no track can be matched, so the caller plays no music
    rather than a random track.
    """
    tags_file = MUSIC_DIR / TRACK_TAGS_FILE_NAME
    try:
        with open(tags_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        log(f"music/{TRACK_TAGS_FILE_NAME} not found — no tracks can be matched by mood")
        return {}
    except Exception as e:
        log(f"Could not read music/{TRACK_TAGS_FILE_NAME} ({e}) — no tracks can be matched by mood")
        return {}

    if not isinstance(data, dict):
        log(f"music/{TRACK_TAGS_FILE_NAME} is not a list of file names with tags — no tracks can be matched by mood")
        return {}

    tags_by_name = {}
    for name, value in data.items():
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, list):
            value = []
        tags_by_name[str(name)] = {str(t).lower().strip() for t in value if str(t).strip()}
    return tags_by_name


def _get_speech_regions(audio_path: Path, words: list[dict] | None = None) -> list[tuple[float, float]]:
    """Extract speech regions from Whisper word timestamps.

    If words is provided (already computed by the captions step), reuses it
    instead of re-running Whisper transcription a second time on the same
    audio. Falls back to running Whisper itself (and then to treating the
    entire audio as one speech region) only if words wasn't provided.
    """
    try:
        if words is None:
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


def select_and_prepare_music(
    voiceover_path: Path,
    work_dir: Path,
    duck_speech: float = 0.12,
    duck_gap: float = 0.25,
    mood_keywords: list[str] | None = None,
    words: list[dict] | None = None,
    video_mood: str | None = None,
) -> dict:
    """Select a track whose tags overlap the niche's mood keywords, build duck filter.

    Each track's score is how many of its tags match a keyword (case-insensitive).
    If video_mood is one of VIDEO_MOODS and the track has that tag, the track
    gets VIDEO_MOOD_BONUS more points. A random pick is made among the top
    scorers. If nothing overlaps, or no keywords are given, a random pick is made
    among tracks tagged "ambient". If there is no ambient track either, no music
    is used and {} is returned. Tracks that are not listed in music/tracks.yaml
    have no tags, so they are never picked.

    Returns dict with track_path and duck_filter for use by assemble.py.
    """
    tracks = _find_tracks()
    if not tracks:
        log("No music tracks found in music/ — skipping background music")
        return {}

    tags_by_name = _load_track_tags()
    tagged = []
    untagged = []
    for t in tracks:
        if t.name in tags_by_name:
            tagged.append((t, tags_by_name[t.name]))
        else:
            untagged.append(t.name)
    if untagged:
        log(f"Not in music/{TRACK_TAGS_FILE_NAME}, so never picked: {', '.join(untagged)}")
    if not tagged:
        log("No tagged music tracks available — skipping background music")
        return {}

    keywords = {kw.lower().strip() for kw in (mood_keywords or []) if kw and kw.strip()}
    mood = clean_video_mood(video_mood)
    mood_note = ""
    if video_mood is not None:
        if mood:
            mood_note = f", video mood {mood}"
            log(f"Video mood: {mood}")
        else:
            log(f"Video mood '{video_mood}' is not one of {', '.join(VIDEO_MOODS)} — ignoring it")
    track = None
    if keywords or mood:
        scored = [
            (t, len(tags & keywords) + (VIDEO_MOOD_BONUS if mood and mood in tags else 0))
            for t, tags in tagged
        ]
        best_score = max(s for _, s in scored)
        if best_score > 0:
            candidates = [t for t, s in scored if s == best_score]
            track = random.choice(candidates)
            log(f"Selected music track (mood match, score {best_score}{mood_note}): {track.name}")

    if track is None:
        ambient = [t for t, tags in tagged if FALLBACK_TAG in tags]
        if not ambient:
            log(f"No mood match and no '{FALLBACK_TAG}' track in music/{TRACK_TAGS_FILE_NAME} — skipping background music")
            return {}
        track = random.choice(ambient)
        log(f"No mood match found — '{FALLBACK_TAG}' fallback pick: {track.name}")

    # Get speech regions for ducking
    speech_regions = _get_speech_regions(voiceover_path, words=words)
    duck_filter = build_duck_filter(speech_regions, vol_speech=duck_speech, vol_gap=duck_gap)
    log(f"Built duck filter with {len(speech_regions)} speech regions")

    return {
        "track_path": str(track),
        "duck_filter": duck_filter,
    }
