"""Render 40 s audition mixes of one voiceover with music at 12, 16 and 20 dB below the voice.

Run from the project folder with the venv Python:
    .\\venv\\Scripts\\python.exe scripts\\audition_music_levels.py

Writes audition_<track>_<level>dB.mp3 into the output folder and prints measured
levels. Stems and measurement WAVs go to the stems folder. Both can be set with
options (the defaults are the ones used for the first audition):
    --voice VOICEOVER.mp3   default .verticals/media/work_1788205414_en/voiceover_en.mp3
    --out-dir FOLDER        default C:\\Users\\jpsve\\Downloads\\audition
    --stems-dir FOLDER      default a folder in the system temp directory

The mix is built by verticals/assemble.build_audio_filter and
verticals/music.build_duck_filter, the same code the pipeline uses.

The duck uses the SAME speech regions as production: verticals/music.py's
_get_speech_regions on Whisper word times. Production passes the caption words
(aligned to the script); this audition runs Whisper on the voiceover without the
script, so the words can differ slightly. In practice Whisper's word times have
no gaps (each word ends where the next begins), so the regions merge into one
and the duck is a constant speech level. The audition therefore shows the
production mix, NOT a pause-by-pause duck. Its "duck depth" column is expected
to be about 0 dB for that reason.

The measurements use ffmpeg silencedetect (silence of 0.3 s or more below
-35 dB) only to decide which parts of the sample are speech and which are
pauses. They do not change the mix. Level = how far the music sits below the
voice during speech.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verticals import music  # noqa: E402
from verticals.assemble import MUSIC_FADE_IN_SECONDS, build_audio_filter  # noqa: E402

VOICE = ROOT / ".verticals" / "media" / "work_1788205414_en" / "voiceover_en.mp3"
AUDITION_DIR = Path(r"C:\Users\jpsve\Downloads\audition")
STEMS_DIR = Path(tempfile.gettempdir()) / "music_audition_stems"
SAMPLE_SECONDS = 40.0
TRACKS = ["pulsar.mp3", "event-horizon.mp3"]
LEVELS_DB = [12, 16, 20]


def ffmpeg(args: list[str]) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", *args],
                       capture_output=True, text=True, check=True)
    return r.stderr


def speech_intervals(voice: Path, seconds: float) -> list[tuple[float, float]]:
    """Speech = everything that is not silence of 0.3 s or more below -35 dB."""
    err = ffmpeg(["-t", str(seconds), "-i", str(voice),
                  "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"])
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start: ([0-9.]+)", err)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end: ([0-9.]+)", err)]
    silences = list(zip(starts, ends + [seconds] * (len(starts) - len(ends))))
    speech, cursor = [], 0.0
    for s, e in silences:
        if s > cursor:
            speech.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < seconds:
        speech.append((cursor, seconds))
    return speech


def integrated_lufs(path: Path) -> float:
    err = ffmpeg(["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    return float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", err)[-1])


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        channels = w.getnchannels()
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64) / 32768.0


def mean_power_db(samples: np.ndarray, mask: np.ndarray) -> float:
    return 10 * np.log10(np.mean(samples[mask] ** 2))


def render(track: Path, entry: dict, duck: str, out: Path, codec: list[str]) -> None:
    audio_filter = build_audio_filter(SAMPLE_SECONDS, duck, entry["start_at"], entry["gain_db"],
                                      voice_index=0, music_index=1)
    ffmpeg(["-y", "-t", str(SAMPLE_SECONDS), "-i", str(VOICE), "-i", str(track),
            "-filter_complex", audio_filter, "-map", "[aout]", *codec, str(out)])


def music_chain(audio_filter: str) -> str:
    """The music branch alone: the part of the graph before amix."""
    return audio_filter.split(";", 1)[0]


def main() -> int:
    global VOICE, AUDITION_DIR, STEMS_DIR
    parser = argparse.ArgumentParser(description="Render the music-level audition files.")
    parser.add_argument("--voice", type=Path, default=VOICE)
    parser.add_argument("--out-dir", type=Path, default=AUDITION_DIR)
    parser.add_argument("--stems-dir", type=Path, default=STEMS_DIR)
    args = parser.parse_args()
    VOICE, AUDITION_DIR, STEMS_DIR = args.voice, args.out_dir, args.stems_dir
    AUDITION_DIR.mkdir(parents=True, exist_ok=True)
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    entries = music._load_track_entries()
    # Duck regions: production's method (Whisper words -> _get_speech_regions), clipped to the sample
    regions = [(max(0.0, s), min(e, SAMPLE_SECONDS)) for s, e in music._get_speech_regions(VOICE)
               if s < SAMPLE_SECONDS]
    print(f"Duck regions (production method, clipped to the sample): {[(round(a, 2), round(b, 2)) for a, b in regions]}")
    # Measurement intervals: silencedetect speech vs pauses (measurement only)
    intervals = speech_intervals(VOICE, SAMPLE_SECONDS)
    print(f"Voice sample: {VOICE.name}, first {SAMPLE_SECONDS:g} s; {len(intervals)} speech intervals (measurement)")

    voice_stem = STEMS_DIR / "voice.wav"
    ffmpeg(["-y", "-t", str(SAMPLE_SECONDS), "-i", str(VOICE), "-ac", "2", "-ar", "44100", str(voice_stem)])
    voice = read_wav(voice_stem)
    n = voice.shape[0]
    sr = 44100
    speech_mask = np.zeros(n, dtype=bool)
    for s, e in intervals:
        speech_mask[int(s * sr):int(e * sr)] = True
    # Leave out the first second (music fade-in) and the last 2 s (fade-out) from the measurements
    fade_in = int(MUSIC_FADE_IN_SECONDS * sr)
    keep = np.ones(n, dtype=bool)
    keep[:fade_in] = False
    keep[n - 2 * sr:] = False
    measure_speech = speech_mask & keep
    measure_gap = (~speech_mask) & keep
    voice_speech_db = mean_power_db(voice, measure_speech)
    voice_gap_db = mean_power_db(voice, measure_gap)
    print(f"Voice level: speech {voice_speech_db:.1f} dBFS (sample RMS), gap {voice_gap_db:.1f} dBFS")

    rows = []
    for track_name in TRACKS:
        track = music.MUSIC_DIR / track_name
        entry = entries[track_name]
        for level in LEVELS_DB:
            speech_vol, gap_vol = music.duck_levels(level)
            duck = music.build_duck_filter(regions, vol_speech=speech_vol, vol_gap=gap_vol)
            stem_base = f"{track.stem}_{level}dB"
            # The music branch alone, through the same chain (for measurement)
            music_stem = STEMS_DIR / f"music_{stem_base}.wav"
            full = build_audio_filter(SAMPLE_SECONDS, duck, entry["start_at"], entry["gain_db"],
                                      voice_index=0, music_index=1)
            ffmpeg(["-y", "-t", str(SAMPLE_SECONDS), "-i", str(VOICE), "-i", str(track),
                    "-filter_complex", music_chain(full), "-map", "[music]", "-ac", "2", "-ar", "44100",
                    str(music_stem)])
            music_audio = read_wav(music_stem)
            speech_db = mean_power_db(music_audio, measure_speech)
            gap_db = mean_power_db(music_audio, measure_gap)

            out = AUDITION_DIR / f"audition_{track.stem}_{level}dB.mp3"
            render(track, entry, duck, out, ["-c:a", "libmp3lame", "-b:a", "192k"])
            mix_wav = STEMS_DIR / f"mix_{stem_base}.wav"
            render(track, entry, duck, mix_wav, ["-ac", "2", "-ar", "44100"])
            mix = read_wav(mix_wav)
            peak = float(np.max(np.abs(mix)))
            rows.append({
                "file": out.name,
                "speech_diff": voice_speech_db - speech_db,
                "duck_depth": gap_db - speech_db,
                "mix_lufs": integrated_lufs(out),
                "peak": peak,
                "start_at": entry["start_at"],
                "gain_db": entry["gain_db"],
            })
            print(f"rendered {out}")

    print()
    print("duck depth = music in pauses minus music in speech (design: about +6 dB; 0 means the duck never opens)")
    print(f"{'file':<36}{'music below voice (speech)':>28}{'duck depth':>12}{'mix LUFS':>10}{'peak':>8}")
    for r in rows:
        print(f"{r['file']:<36}{r['speech_diff']:>26.1f} dB{r['duck_depth']:>10.1f} dB"
              f"{r['mix_lufs']:>10.1f}{r['peak']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
