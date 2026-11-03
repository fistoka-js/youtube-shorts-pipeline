"""Background music — track selection + volume ducking."""

import math
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

# Which tracks a niche may use is set per niche, as "allowed_tracks" in the niche's
# music block (see niches/science.yaml). A niche without that key has no limit and
# picks by its tags, as before. There is no global list here.

# Music level. Every track is first gained to TARGET_LUFS (gain_db in tracks.yaml,
# from scripts/measure_music_tracks.py, which uses the same target). Then the
# volume is set relative to the voice. Keep TARGET_LUFS equal to the script's.
TARGET_LUFS = -16.0
# Loudness of the voice. Measured: the existing voiceovers are -19.5 to -20.6
# LUFS (mean about -20.3). Used only to turn "below the voice" into a level.
VOICE_LUFS = -20.3
# Music during speech sits this many dB below the voice (a choice, picked by ear after
# listening to the audition in scripts/audition_music_levels.py, which tested 12, 16 and 20).
MUSIC_BELOW_VOICE_DB = 19.0
# In the gaps between speech the music is raised by the same ratio the old
# ducking used (0.25 / 0.12, about 6.4 dB).
GAP_BOOST_DB = 20 * math.log10(0.25 / 0.12)
# Cap: in the gaps the music is never closer than this to the voice level
# (a choice). Stops a high MUSIC_BELOW_VOICE_DB from making the music louder than the voice.
GAP_MUSIC_MIN_BELOW_VOICE_DB = 6.0


# Note: the speech regions currently merge into one block, because Whisper's word times
# have no gaps (each word ends where the next begins). So in practice the duck is a
# constant speech level, and the gap level and its cap rarely matter.
def duck_levels(music_below_voice_db: float = MUSIC_BELOW_VOICE_DB) -> tuple[float, float]:
    """Return (speech volume, gap volume) as linear multipliers for a track already gained to TARGET_LUFS.

    speech: music is music_below_voice_db below the voice.
    gap:    music is GAP_BOOST_DB louder than that, but never closer than
            GAP_MUSIC_MIN_BELOW_VOICE_DB below the voice.
    Raises ValueError if music_below_voice_db is less than GAP_MUSIC_MIN_BELOW_VOICE_DB,
    because the gap would then be quieter than the speech level.
    """
    if music_below_voice_db < GAP_MUSIC_MIN_BELOW_VOICE_DB:
        # Below the cap the speech level itself would be louder than the gap cap,
        # so the gap would be quieter than speech (an inverted duck). Refuse it.
        raise ValueError(
            f"music_below_voice_db={music_below_voice_db} is less than "
            f"GAP_MUSIC_MIN_BELOW_VOICE_DB={GAP_MUSIC_MIN_BELOW_VOICE_DB}: the gap level would be "
            "quieter than the speech level")
    speech_rel = -music_below_voice_db
    gap_rel = min(speech_rel + GAP_BOOST_DB, -GAP_MUSIC_MIN_BELOW_VOICE_DB)

    def to_volume(rel_db: float) -> float:
        return 10 ** ((VOICE_LUFS + rel_db - TARGET_LUFS) / 20)

    return to_volume(speech_rel), to_volume(gap_rel)


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# Limits on the numbers read from tracks.yaml (choices). start_at is where playback
# starts; the measuring script only ever chooses starts up to 45 s. gain_db is the
# gain to reach the target; the measured gains run from about -9 to +7 dB.
# The 45 s limit is defined here and imported by scripts/measure_music_tracks.py, which
# uses it to decide whether a track is unusable. The thresholds the script uses to
# measure tracks (6 dB for the loud part, 3 dB for steady, the 90th percentile for the
# loud level) are choices in that script: LOUD_WINDOW_DB, STEADY_TOLERANCE_DB and
# LOUD_PERCENTILE. They are not used here.
MAX_START_AT_SECONDS = 45.0
MAX_ABS_GAIN_DB = 24.0


def _valid_start_at(value) -> bool:
    return _is_number(value) and math.isfinite(value) and 0.0 <= value <= MAX_START_AT_SECONDS


def _valid_gain_db(value) -> bool:
    return _is_number(value) and math.isfinite(value) and abs(value) <= MAX_ABS_GAIN_DB


def _parse_track_entry(name: str, value) -> dict:
    """Turn one tracks.yaml entry into {tags, start_at, gain_db, usable}.

    An entry is either a tag list (or a single tag string), which starts at
    0 s with no gain change, or a mapping with tags, start_at, gain_db and
    unusable. A track is unusable if it is marked unusable, or if start_at or
    gain_db is not a finite number inside the limits (see MAX_START_AT_SECONDS
    and MAX_ABS_GAIN_DB). This covers negative, inf and nan values.
    """
    if isinstance(value, dict):
        tags_value = value.get("tags")
        start_at = value.get("start_at", 0.0)
        gain_db = value.get("gain_db", 0.0)
        unusable = value.get("unusable", False) is True
    else:
        tags_value = value
        start_at, gain_db, unusable = 0.0, 0.0, False

    if isinstance(tags_value, str):
        tags_value = [tags_value]
    elif not isinstance(tags_value, list):
        tags_value = []
    tags = {str(t).lower().strip() for t in tags_value if str(t).strip()}

    if not unusable and not (_valid_start_at(start_at) and _valid_gain_db(gain_db)):
        log(f"music/{TRACK_TAGS_FILE_NAME}: {name} has start_at={start_at!r} or gain_db={gain_db!r}, "
            f"which is not a finite number in the allowed range (start_at 0 to {MAX_START_AT_SECONDS:g} s, "
            f"gain_db within ±{MAX_ABS_GAIN_DB:g} dB) — treating it as unusable")
        unusable = True

    if unusable:
        return {"tags": tags, "start_at": None, "gain_db": None, "usable": False}
    return {"tags": tags, "start_at": float(start_at), "gain_db": float(gain_db), "usable": True}


def _load_track_entries() -> dict[str, dict]:
    """Read music/tracks.yaml into {file name: parsed entry} (see _parse_track_entry).

    Returns {} if the file is missing, unreadable or not in the expected shape.
    With no entries loaded, no track can be matched, so the caller plays no music
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

    return {str(name): _parse_track_entry(str(name), value) for name, value in data.items()}


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


def build_duck_filter(speech_regions: list[tuple[float, float]], buffer: float = 0.3,
                      vol_speech: float | None = None, vol_gap: float | None = None) -> str:
    """Build ffmpeg volume filter expression for ducking during speech.

    During speech: volume = vol_speech
    During gaps: volume = vol_gap
    Both default to duck_levels() (set from MUSIC_BELOW_VOICE_DB and the cap).
    The "never closer than GAP_MUSIC_MIN_BELOW_VOICE_DB" cap lives in duck_levels,
    not here: explicit vol_speech and vol_gap are used as given.
    Transitions smoothed by ±buffer seconds.
    """
    default_speech, default_gap = duck_levels()
    if vol_speech is None:
        vol_speech = default_speech
    if vol_gap is None:
        vol_gap = default_gap
    if not speech_regions:
        return f"volume={vol_gap:.4f}"

    # Build between() conditions for speech regions
    conditions = []
    for start, end in speech_regions:
        s = max(0, start - buffer)
        e = end + buffer
        conditions.append(f"between(t,{s:.2f},{e:.2f})")

    condition_expr = "+".join(conditions)
    return f"volume='if({condition_expr}, {vol_speech:.4f}, {vol_gap:.4f})':eval=frame"


def select_and_prepare_music(
    voiceover_path: Path,
    work_dir: Path,
    mood_keywords: list[str] | None = None,
    words: list[dict] | None = None,
    video_mood: str | None = None,
    allowed_tracks: list[str] | None = None,
) -> dict:
    """Select a track whose tags overlap the niche's mood keywords, build duck filter.

    allowed_tracks is the niche's list of file names (niche music.allowed_tracks).
    If it is None the niche has no limit and every track is considered. If it is
    given, only those file names are considered. Either way a track must be usable
    in music/tracks.yaml. Each track's score is how many of its tags match a keyword
    (case-insensitive). If video_mood is one of VIDEO_MOODS and the track has that
    tag, the track gets VIDEO_MOOD_BONUS more points. A random pick is made among
    the top scorers. If nothing overlaps, or no keywords are given, a random pick
    is made among allowed tracks tagged "ambient". If none is left, no music is
    used and {} is returned. Tracks that are not listed in music/tracks.yaml
    have no tags, so they are never picked.

    Returns dict with track_path, duck_filter, start_at and gain_db for use by
    assemble.py.
    """
    tracks = _find_tracks()
    if not tracks:
        log("No music tracks found in music/ — skipping background music")
        return {}

    if allowed_tracks is not None:
        if not isinstance(allowed_tracks, (list, tuple)) or not all(isinstance(n, str) for n in allowed_tracks):
            log(f"allowed_tracks in the niche must be a list of file names such as [pulsar.mp3], "
                f"not {allowed_tracks!r} — skipping background music")
            return {}
        on_disk = {t.name for t in tracks}
        for name in allowed_tracks:
            if name not in on_disk:
                log(f"allowed_tracks lists {name} but there is no such file in music/")

    entries = _load_track_entries()
    tagged = []
    untagged = []
    excluded = []
    for t in tracks:
        entry = entries.get(t.name)
        if entry is None:
            untagged.append(t.name)
        elif (allowed_tracks is not None and t.name not in allowed_tracks) or not entry["usable"]:
            excluded.append(t.name)
        else:
            tagged.append((t, entry))
    if untagged:
        log(f"Not in music/{TRACK_TAGS_FILE_NAME}, so never picked: {', '.join(untagged)}")
    if excluded:
        log(f"Not allowed by the niche or unusable (see allowed_tracks and tracks.yaml), so never picked: {', '.join(excluded)}")
    if not tagged:
        log("No allowed, usable music tracks available — skipping background music")
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
            (t, len(entry["tags"] & keywords) + (VIDEO_MOOD_BONUS if mood and mood in entry["tags"] else 0))
            for t, entry in tagged
        ]
        best_score = max(s for _, s in scored)
        if best_score > 0:
            candidates = [t for t, s in scored if s == best_score]
            track = random.choice(candidates)
            log(f"Selected music track (mood match, score {best_score}{mood_note}): {track.name}")

    if track is None:
        ambient = [t for t, entry in tagged if FALLBACK_TAG in entry["tags"]]
        if not ambient:
            log(f"No mood match and no '{FALLBACK_TAG}' track in music/{TRACK_TAGS_FILE_NAME} — skipping background music")
            return {}
        track = random.choice(ambient)
        log(f"No mood match found — '{FALLBACK_TAG}' fallback pick: {track.name}")

    entry = entries[track.name]

    # Get speech regions for ducking
    speech_regions = _get_speech_regions(voiceover_path, words=words)
    duck_filter = build_duck_filter(speech_regions)
    log(f"Built duck filter with {len(speech_regions)} speech regions")
    log(f"Music {track.name}: start at {entry['start_at']:.1f} s, gain {entry['gain_db']:+.1f} dB")

    return {
        "track_path": str(track),
        "duck_filter": duck_filter,
        "start_at": entry["start_at"],
        "gain_db": entry["gain_db"],
    }
