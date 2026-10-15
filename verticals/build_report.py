"""Per-clip b-roll build report.

Records, for every b-roll slot, where the clip came from and how it was fitted
to the output frame. It is informational only: nothing in here picks a clip,
crops it, trims it or changes any timing. Every public function catches its own
errors and logs a warning, so a report problem can never fail or alter a build.

Saved file: <drafts folder>/build_report_<job_id>_<lang>.json
One file per job AND language: the English and Hindi runs of the same job each
source their own b-roll, so a shared file would overwrite the other language.
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from .log import log

# A slot this many seconds longer than its saved clip still counts as not
# looping (tolerance for float / frame-rounding noise).
LOOP_TOLERANCE_S = 0.05

# (column title, column width) for the printed table.
_COLUMNS = [
    ("Slot", 5), ("Section", 13), ("Pexels", 10), ("Src WxH", 12), ("Orient", 10),
    ("NatDur", 8), ("Kept", 8), ("Score", 7), ("Source", 19), ("SlotDur", 9), ("Loop", 5),
]


def orientation_of(width, height) -> str | None:
    """'landscape' when width > height, otherwise 'portrait'.

    So a square clip counts as portrait. Returns None when the size is unknown."""
    try:
        if not width or not height:
            return None
        return "landscape" if width > height else "portrait"
    except TypeError:
        return None


def kept_fraction(src_w, src_h, out_w, out_h) -> float | None:
    """Share of the source WIDTH that survives the crop to the output shape.

    Mirrors the ffmpeg chain used when clips are saved and placed
    (scale=W:H:force_original_aspect_ratio=increase, then crop=W:H). The source
    is scaled until it covers the output frame, and the overflow is cut away:
      - source wider than the output shape: the height is matched and the sides
        are cut, so kept = (out_w / out_h) / (src_w / src_h).
      - otherwise: the full width is kept (1.0) and the top/bottom is cut.
    Returns None when any size is missing or not positive."""
    try:
        if not (src_w and src_h and out_w and out_h):
            return None
        if min(src_w, src_h, out_w, out_h) <= 0:
            return None
        return min(1.0, (out_w / out_h) / (src_w / src_h))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def clip_length_on_disk(clip_duration, native_duration) -> float | None:
    """Length of the saved clip in seconds.

    The download is trimmed to clip_duration (ffmpeg -t). A Pexels clip that is
    shorter than clip_duration is saved at its own length, so the smaller of the
    two is used. Note: Pexels' top-level duration may be rounded to whole seconds,
    so this is approximate for very short clips."""
    try:
        if clip_duration is None:
            return None
        clip_duration = float(clip_duration)
        if native_duration:
            return min(clip_duration, float(native_duration))
        return clip_duration
    except (TypeError, ValueError):
        return None


def loops(slot_duration, clip_length) -> bool | None:
    """True when the slot is longer than the saved clip (plus the tolerance), so
    the clip has to repeat to fill it. None when either length is unknown."""
    try:
        if slot_duration is None or clip_length is None:
            return None
        return float(slot_duration) > clip_length + LOOP_TOLERANCE_S
    except (TypeError, ValueError):
        return None


def empty_row(slot, section, kind, source) -> dict:
    """A row with every measurement left empty. Used when a clip cannot be
    measured, so the table still has one row per slot."""
    return {
        "slot": slot,
        "section": section,
        "type": kind,
        "pexels_id": None,
        "source_width": None,
        "source_height": None,
        "native_duration": None,
        "orientation": None,
        "kept_fraction": None,
        "clip_score": None,
        "source": source,
        "clip_duration": None,
        "slot_duration": None,
        "loop": None,
    }


def make_clip_row(*, slot, section, kind, source, out_w, out_h,
                  video=None, file_entry=None, clip_duration=None,
                  clip_score=None) -> dict:
    """One build-report row for one slot.

    kind: "video" (a Pexels clip) or "image" (Pollinations still or solid colour).
    video: the Pexels video dict from the pool or search. Its top-level "id" is
        the Pexels id, and its top-level "duration" (seconds, from the Pexels API
        metadata) is the native duration. The downloaded file itself is never
        read, because it is deleted or re-encoded after download.
    file_entry: the video_files entry that was actually downloaded (the one that
        _best_file_link picks). Its "width" and "height" are the source size. If
        the entry lacks them, the video's top-level width/height are used.
    slot_duration and loop are filled in later by apply_slot_durations, after
    assembly, because only assembly knows how long each slot really is."""
    row = empty_row(slot, section, kind, source)
    row["clip_score"] = clip_score
    if kind != "video":
        return row
    video = video or {}
    file_entry = file_entry or {}
    width = file_entry.get("width") or video.get("width")
    height = file_entry.get("height") or video.get("height")
    row.update(
        pexels_id=video.get("id"),
        source_width=width,
        source_height=height,
        native_duration=video.get("duration") or None,
        orientation=orientation_of(width, height),
        kept_fraction=kept_fraction(width, height, out_w, out_h),
        clip_duration=clip_duration,
    )
    return row


def apply_slot_durations(rows: list, slot_durations: list) -> None:
    """Fill slot_duration and loop into the rows, from assembly's per-slot lengths.

    slot_durations[k] belongs to the row whose slot number is k + 1. Images never
    loop (Ken Burns runs for the full slot). Rows with bad data are skipped."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            k = int(row.get("slot")) - 1
            if not 0 <= k < len(slot_durations):
                continue
            slot_len = float(slot_durations[k])
            row["slot_duration"] = round(slot_len, 3)
            if row.get("type") == "video":
                row["loop"] = loops(
                    slot_len,
                    clip_length_on_disk(row.get("clip_duration"), row.get("native_duration")),
                )
            else:
                row["loop"] = False
        except (TypeError, ValueError):
            continue


def summarize(rows: list) -> dict:
    """Totals for the whole build: landscape count, average width kept, reused
    Pexels ids, and how many slots loop. Missing fields are treated as unknown."""
    rows = [r for r in rows if isinstance(r, dict)]
    video_rows = [r for r in rows if r.get("type") == "video"]
    landscape = sum(1 for r in video_rows if r.get("orientation") == "landscape")

    kepts = [float(r["kept_fraction"]) for r in rows
             if isinstance(r.get("kept_fraction"), (int, float))]
    avg_kept = sum(kepts) / len(kepts) if kepts else None

    id_counts = Counter(r.get("pexels_id") for r in video_rows
                        if r.get("pexels_id") is not None)
    dup_ids = sorted((pid for pid, n in id_counts.items() if n > 1), key=str)

    timed = [r for r in rows if r.get("slot_duration") is not None]
    looping = sum(1 for r in timed if r.get("loop") is True) if timed else None

    avg_text = f"{avg_kept * 100:.1f}%" if avg_kept is not None else "n/a"
    dup_text = ", ".join(str(pid) for pid in dup_ids) if dup_ids else "none"
    loop_text = str(looping) if looping is not None else "unknown"
    text = (
        f"B-roll summary: {landscape} landscape clip(s) of {len(video_rows)} video clip(s); "
        f"average width kept {avg_text} (over {len(kepts)} clip(s) with known size); "
        f"duplicate Pexels ids: {dup_text}; slots that loop: {loop_text}"
    )
    return {
        "landscape_clips": landscape,
        "video_clips": len(video_rows),
        "avg_width_kept": round(avg_kept, 4) if avg_kept is not None else None,
        "clips_with_known_size": len(kepts),
        "duplicate_pexels_ids": dup_ids,
        "slots_that_loop": looping,
        "text": text,
    }


# ── Printed table ────────────────────────────────────────────────────────────

def _cell(value, fmt):
    """Format one value; "-" when it is missing, "?" when it cannot be formatted."""
    if value is None:
        return "-"
    try:
        return fmt(value)
    except (TypeError, ValueError):
        return "?"


def _row_cells(row: dict) -> list:
    width, height = row.get("source_width"), row.get("source_height")
    size = "-" if width is None or height is None else f"{width}x{height}"
    loop = row.get("loop")
    return [
        _cell(row.get("slot"), str),
        _cell(row.get("section") or None, str),
        _cell(row.get("pexels_id"), str),
        size,
        _cell(row.get("orientation"), str),
        _cell(row.get("native_duration"), lambda d: f"{float(d):.1f}s"),
        _cell(row.get("kept_fraction"), lambda k: f"{float(k) * 100:.1f}%"),
        _cell(row.get("clip_score"), lambda s: f"{float(s):.3f}"),
        _cell(row.get("source"), str),
        _cell(row.get("slot_duration"), lambda d: f"{float(d):.2f}s"),
        "-" if loop is None else ("yes" if loop else "no"),
    ]


def format_table(rows: list) -> list[str]:
    """Fixed-width text lines: header, separator, one line per slot."""
    def line(cells):
        parts = []
        for cell, (_, w) in zip(cells, _COLUMNS):
            parts.append(str(cell)[:w - 1].ljust(w))
        return " ".join(parts).rstrip()

    out = [
        line([name for name, _ in _COLUMNS]),
        line(["-" * (w - 2) for _, w in _COLUMNS]),
    ]
    for row in rows:
        out.append(line(_row_cells(row if isinstance(row, dict) else {})))
    return out


def print_table(rows: list, title: str) -> None:
    """Write the table to the build log, one log line per table line."""
    try:
        log(title)
        for text in format_table(rows):
            log(text)
    except Exception as e:
        log(f"WARNING: could not print the b-roll build table: {e}")


# ── Saving ───────────────────────────────────────────────────────────────────

def report_path_for(draft_path, job_id, lang) -> Path:
    """The report sits in the same folder as the draft JSON."""
    return Path(draft_path).parent / f"build_report_{job_id}_{lang}.json"


def write_report(path, data: dict) -> bool:
    """Write the report as UTF-8 JSON: write a temp file, then swap it into place,
    so a crash mid-write never leaves a half-written report. Returns False (and
    logs a warning) on any failure instead of raising."""
    tmp = None
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as e:
        log(f"WARNING: could not save the b-roll build report: {e}")
        try:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _report_doc(job_id, lang, platform, out_w, out_h, stage, rows) -> dict:
    return {
        "job_id": job_id,
        "lang": lang,
        "platform": platform,
        "output_size": [out_w, out_h],
        "stage": stage,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "clips": rows,
        "summary": summarize(rows),
    }


def save_broll_stage(report_path, job_id, lang, platform, out_w, out_h, rows) -> None:
    """Called at the end of the b-roll stage. Prints the table and saves the report.
    Slot duration and loop are still blank here; they are filled in after assembly."""
    try:
        print_table(rows, f"B-roll build report ({len(rows)} slot(s); slot dur and loop "
                          f"are filled in after assembly):")
        # Summary now, with loop count "unknown" until assembly fills it in.
        log(summarize(rows)["text"])
        write_report(report_path, _report_doc(job_id, lang, platform, out_w, out_h,
                                              "broll", rows))
    except Exception as e:
        log(f"WARNING: b-roll build report skipped: {e}")


def record_assembly(report_path, slot_durations) -> None:
    """Called after assembly. Loads the saved report, fills in slot duration and
    loop, prints the table again, logs the summary line and saves the report.
    If no report exists (b-roll stage was skipped and never reported), logs a
    line saying so and does nothing else."""
    try:
        path = Path(report_path)
        if not path.exists():
            log(f"No b-roll build report found at {path.name}, so no table or summary "
                f"after assembly (the b-roll stage did not save one for this run)")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("clips") or []
        apply_slot_durations(rows, slot_durations)
        data["stage"] = "assembled"
        data["clips"] = rows
        data["summary"] = summarize(rows)
        data["written_at"] = datetime.now().isoformat(timespec="seconds")
        print_table(rows, "B-roll build report (after assembly):")
        log(data["summary"]["text"])
        write_report(path, data)
    except Exception as e:
        log(f"WARNING: b-roll build report not updated after assembly: {e}")
