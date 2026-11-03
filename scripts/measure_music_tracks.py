"""Measure each background music track and write the results into music/tracks.yaml.

Run from the project folder with the venv Python:
    .\\venv\\Scripts\\python.exe scripts\\measure_music_tracks.py            (writes tracks.yaml)
    .\\venv\\Scripts\\python.exe scripts\\measure_music_tracks.py --dry-run  (prints table only)

Needs ffmpeg on PATH (uses the ebur128 filter). Re-run it if a track file changes.

How the numbers are computed
----------------------------
ffmpeg's ebur128 filter gives a "short-term" loudness value S(t) every 0.1 s.
S(t) is the loudness of the 3 seconds that end at time t (ffmpeg's convention).
Silent frames come out as -120.7 LUFS and are ignored (anything below -70 LUFS).

1. loud level  = the 90th percentile of S(t) over the track (its loud sections).
2. loud part   = every frame whose S(t) is within 6 dB of the loud level, i.e.
                 S(t) >= loud level - 6.
3. steady_lufs = the loudness of the loud part, averaged by energy:
                 10 * log10( mean of 10^(S/10) over the loud part ).
4. start_at    = the first start a (0.1 s steps) such that the 3 s that BEGIN
                 at a is within 3 dB of steady_lufs. The short-term reading at
                 a + 3 is used to find the candidate (S(a+3) covers the 3 s from a
                 to a+3). The candidate is then confirmed on the heard audio: that
                 3 s is cut with ffmpeg (-ss a -t 3) and its integrated loudness
                 must also be within 3 dB of steady. start_lufs is that clip's
                 loudness. For a reading before 3 s, a is 0 (ffmpeg's first full
                 window is reported at about 2.9 s). If no start is found at or
                 before 45 s, the track is UNUSABLE: it gets no start_at and is
                 never picked (unusable: true).
5. gain_db     = TARGET_LUFS - steady_lufs. Adding gain_db brings the track's
                 steady loudness to TARGET_LUFS.

Choices (not measured; change here and re-run if you disagree):
    TARGET_LUFS = -16.0        single loudness every track is normalised to
                               (defined once, in verticals/music.py)
    LOUD_WINDOW_DB = 6.0       how close to the loud level counts as "loud"
    STEADY_TOLERANCE_DB = 3.0  how close to steady counts as "reached steady"
    MAX_START_AT_SECONDS = 45.0  longest wait allowed before a track is unusable
                               (defined once, in verticals/music.py)
    LOUD_PERCENTILE = 90       which level counts as the track's loud level

The yaml file keeps its header comments. Entries are rewritten as mappings with
tags, steady_lufs, start_lufs, start_at, gain_db (or tags, steady_lufs, unusable).
Tracks in music/ with no entry in tracks.yaml are reported and left alone.
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MUSIC_DIR = ROOT / "music"
TRACKS_FILE = MUSIC_DIR / "tracks.yaml"

# The one definition of the target, shared with the pipeline (verticals/music.py)
from verticals.music import MAX_START_AT_SECONDS, TARGET_LUFS  # noqa: E402

LOUD_WINDOW_DB = 6.0
STEADY_TOLERANCE_DB = 3.0
LOUD_PERCENTILE = 90
SILENCE_FLOOR_LUFS = -70.0
WINDOW_SECONDS = 3.0  # length of the short-term loudness window

_FRAME_RE = re.compile(r"pts_time:([0-9.]+)")
_VALUE_RE = re.compile(r"lavfi\.r128\.S=(-?[0-9.]+|-inf|inf)")


def short_term_loudness(path: Path) -> list[tuple[float, float]]:
    """Return [(time, S)] for every 0.1 s step, using ffmpeg's ebur128 filter."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(path),
        "-af", "ebur128=metadata=1:peak=none,ametadata=print:key=lavfi.r128.S:file=-",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    points = []
    time = None
    for line in r.stdout.splitlines():
        m = _FRAME_RE.search(line)
        if m:
            time = float(m.group(1))
            continue
        v = _VALUE_RE.search(line)
        if v and time is not None:
            value = float(v.group(1))  # "-inf" / "inf" parse as float too
            points.append((time, value))
            time = None
    return points


def clip_lufs(path: Path, start: float) -> float:
    """Integrated loudness of the 3 s clip that playback hears from `start`.

    Cut with ffmpeg (-ss start -t 3) and measured with ebur128. This is the
    check on the heard audio itself, independent of the full-track S(t) values.
    """
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{WINDOW_SECONDS:g}",
         "-i", str(path), "-af", "ebur128=peak=none", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    return float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", r.stderr)[-1])


def analyse(points: list[tuple[float, float]], heard=None) -> dict:
    """Apply the rules in the module docstring to one track's S(t) values.

    heard(start) returns the measured loudness of the clip from `start`. When
    given, a candidate start is accepted only if that clip is within 3 dB of
    steady too, and start_lufs is that measured value.
    """
    audible = [(t, s) for t, s in points if math.isfinite(s) and s > SILENCE_FLOOR_LUFS]
    if not audible:
        return {"usable": False, "reason": "no audible sound", "steady_lufs": None}

    values = sorted(s for _, s in audible)
    loud_level = values[min(len(values) - 1, round(LOUD_PERCENTILE / 100 * (len(values) - 1)))]
    loud_part = [s for _, s in audible if s >= loud_level - LOUD_WINDOW_DB]
    energy = sum(10 ** (s / 10) for s in loud_part) / len(loud_part)
    steady = 10 * math.log10(energy)

    # A reading at time T covers the 3 s that END at T. Playback that starts at
    # time a hears the 3 s from a to a+3, which is the reading at a+3. So the
    # candidate start for a reading at T is a = T - 3 (never below 0).
    start = None
    for t, s in audible:
        candidate = max(0.0, t - WINDOW_SECONDS)
        if candidate > MAX_START_AT_SECONDS:
            break
        if abs(s - steady) > STEADY_TOLERANCE_DB:
            continue
        clip = heard(candidate) if heard else s
        if abs(clip - steady) <= STEADY_TOLERANCE_DB:
            start = (candidate, clip)
            break
    if start is None:
        return {"usable": False, "reason": f"does not reach steady within {MAX_START_AT_SECONDS:g} s",
                "steady_lufs": round(steady, 1)}

    return {
        "usable": True,
        "steady_lufs": round(steady, 1),
        "start_lufs": round(start[1], 1),
        "start_at": round(start[0], 1),
        "gain_db": round(TARGET_LUFS - steady, 1),
    }


def _header_and_groups(text: str) -> tuple[list[str], list[tuple[str, bool]]]:
    """Split the file into its header comment lines and the (key, starts_group) list.

    starts_group is True when the key's line follows a blank line, so the
    rewritten file keeps the same blank-line grouping as before.
    """
    lines = text.splitlines()
    header = []
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        header.append(lines[i])
        i += 1
    # Keep the header up to its last comment line, plus one blank line before entries
    while header and not header[-1].strip():
        header.pop()
    header.append("")

    keys = []
    prev_blank = False
    for line in lines[i:]:
        if not line.strip():
            prev_blank = True
            continue
        if line.startswith("#"):
            continue
        m = re.match(r"^([^\s#][^:]*):", line)
        if m:
            keys.append((m.group(1).strip(), prev_blank))
        prev_blank = False
    return header, keys


def _flow_list(tags) -> str:
    return "[" + ", ".join(str(t) for t in tags) + "]"


def _entry_lines(name: str, tags, result: dict) -> list[str]:
    lines = [f"{name}:", f"  tags: {_flow_list(tags)}"]
    if result.get("steady_lufs") is not None:
        lines.append(f"  steady_lufs: {result['steady_lufs']}")
    if result["usable"]:
        lines.append(f"  start_lufs: {result['start_lufs']}")
        lines.append(f"  start_at: {result['start_at']}")
        lines.append(f"  gain_db: {result['gain_db']}")
    else:
        lines.append(f"  unusable: true  # never reaches steady loudness within {MAX_START_AT_SECONDS:g} s: never picked")
    return lines


def _tags_of(value) -> list:
    if isinstance(value, dict):
        value = value.get("tags")
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _raw_entries(text: str) -> dict[str, list[str]]:
    """Map each top-level key to its exact lines (the key line and its indented lines)."""
    raw: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^([^\s#][^:]*):", line)
        if m:
            current = m.group(1).strip()
            raw[current] = [line]
        elif current is not None and line.startswith((" ", "\t")):
            raw[current].append(line)
        else:
            current = None
    return raw


def build_file(text: str, results: dict[str, dict]) -> str:
    """Return the new tracks.yaml text, keeping the header and blank-line groups.

    An entry that was not measured (its mp3 is missing from music/) is copied
    back exactly as it was, so a rewrite never deletes data.
    """
    header, keys = _header_and_groups(text)
    data = yaml.safe_load(text) or {}
    raw = _raw_entries(text)
    out = list(header)
    first = True
    for name, starts_group in keys:
        if not first and starts_group:
            out.append("")
        first = False
        if name in results:
            out.extend(_entry_lines(name, _tags_of(data.get(name)), results[name]))
        else:
            out.extend(raw.get(name, []))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the table, do not write tracks.yaml")
    args = parser.parse_args(argv)

    text = TRACKS_FILE.read_text(encoding="utf-8")
    listed = list((yaml.safe_load(text) or {}).keys())
    on_disk = {p.name for p in MUSIC_DIR.glob("*.mp3")}
    for name in listed:
        if name not in on_disk:
            print(f"WARNING: {name} is in tracks.yaml but not in music/ - not measured, kept as it was")
    for name in sorted(on_disk - set(listed)):
        print(f"WARNING: {name} is in music/ but not in tracks.yaml - not measured, never picked")

    results = {}
    print(f"{'track':<42}{'steady':>8}{'start_lufs':>12}{'start_at':>10}{'gain_db':>9}  status")
    for name in listed:
        if name not in on_disk:
            continue
        path = MUSIC_DIR / name
        result = analyse(short_term_loudness(path), heard=lambda a, p=path: clip_lufs(p, a))
        results[name] = result
        if result["usable"]:
            print(f"{name:<42}{result['steady_lufs']:>8}{result['start_lufs']:>12}"
                  f"{result['start_at']:>10}{result['gain_db']:>9}  usable")
        else:
            print(f"{name:<42}{str(result.get('steady_lufs')):>8}{'-':>12}{'-':>10}{'-':>9}  UNUSABLE: {result['reason']}")

    if args.dry_run:
        print("\n--dry-run: tracks.yaml not written")
        return 0
    TRACKS_FILE.write_text(build_file(text, results), encoding="utf-8")
    print(f"\nWrote {TRACKS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
