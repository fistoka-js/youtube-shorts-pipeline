"""Review sheets: one Markdown file per queued video, for the person who reviews uploads.

The sheet lives at queue/review/<queue file name without .json>.md under the
current working directory. build_review_sheet() is pure (text in, text out), so
it can be tested without touching files. write_review_sheet() and add_video_line()
do the file work. The caller decides what to do when they fail.
"""

import os
import re
from contextlib import suppress
from pathlib import Path

REVIEW_DIR = Path("queue") / "review"
FLAG_LINE = "NEEDS FACT CHECK"
# The verdicts that flag a claim. "simplified" is deliberately not here. Shared with fact_check.
FLAGGING_VERDICTS = ("contradicted", "known_misconception", "unsupported")
SIMPLIFICATIONS_HEADING = "## Simplifications (for your eyes)"
VIDEO_PREFIX = "Video: "
NO_SOURCES_TEXT = "The research step does not keep source URLs, so none can be listed."
_MISSING_AUDIT = {"available": False, "ok": False, "claims": [], "error": "no fact check record was saved"}


def review_sheet_path(review_name: str) -> Path:
    return REVIEW_DIR / f"{review_name}.md"


def _fence(text) -> str:
    """Wrap text in a code fence longer than any backtick run inside it, so the text cannot break the layout."""
    text = "" if text is None else str(text)
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    mark = "`" * max(3, longest + 1)
    return f"{mark}text\n{text}\n{mark}"


def one_line(text) -> str:
    """Collapse all whitespace (including newlines) to single spaces. Shared with fact_check."""
    return " ".join(str(text).split())


def _result_word(audit: dict) -> str:
    if not audit.get("available"):
        return "unavailable"
    return "passed" if audit.get("ok") else "failed"


def _claim_bullets(claims: list[dict]) -> list[str]:
    bullets = []
    for c in claims:
        line = f"- {one_line(c.get('verdict', ''))}: {one_line(c.get('claim', ''))}"
        note = one_line(c.get("note", ""))
        if note:
            line += f" (Note: {note})"
        bullets.append(line)
    return bullets


def _marker_text(fact_check: dict) -> str:
    first = fact_check.get("markers_removed_first", 0)
    redraft = fact_check.get("markers_removed_redraft")
    if redraft is None:
        return f"Bracketed markers removed by the cleaner: {first}."
    return (f"Bracketed markers removed by the cleaner: {first} in the first script, "
            f"{redraft} in the redraft.")


def build_review_sheet(*, queue_name: str, topic: str, fact_check: dict | None,
                       final_script: str, research: str, video_url: str | None = None) -> str:
    """Return the sheet text. Sections are in the order the review needs them."""
    fc = fact_check or {}
    first = fc.get("first_audit") or _MISSING_AUDIT
    final = fc.get("final_audit") or first
    redrafted = bool(fc.get("redrafted"))
    redraft_error = fc.get("redraft_error")
    needs = bool(fc.get("needs_fact_check", True)) if fact_check else True

    out: list[str] = []
    if needs:
        out += [FLAG_LINE, ""]
    out += [f"# Review: {queue_name}", "", f"Topic: {one_line(topic)}", ""]

    out += ["## Audit result", f"- Result: {_result_word(final)}"]
    if redrafted:
        out.append("- Redraft: yes. The script was rewritten once after the first audit, and audited again.")
    elif redraft_error:
        out.append(f"- Redraft: failed, so the first script was kept. Error: {one_line(redraft_error)}")
    else:
        out.append("- Redraft: no")
    out.append(f"- Error: {one_line(final['error']) if final.get('error') else 'none'}")
    out.append("")

    out.append("## Claims checked (final audit)")
    if not final.get("available"):
        out.append("- No claims were checked, because the audit was unavailable.")
    elif not final.get("claims"):
        out.append("- No claims are listed. (The fact check never passes an empty claim list.)")
    else:
        out += _claim_bullets(final["claims"])
    out.append("")

    simplified = [c for c in final.get("claims", []) if c.get("verdict") == "simplified"]
    if simplified:
        out.append(SIMPLIFICATIONS_HEADING)
        out += _claim_bullets(simplified)
        out.append("")

    if redrafted:
        out.append("## First audit flagged claims (before the redraft)")
        if not first.get("available"):
            out.append("- No claims were checked, because the first audit was unavailable.")
        else:
            flagged = [c for c in first.get("claims", []) if c.get("verdict") in FLAGGING_VERDICTS]
            out += _claim_bullets(flagged) or ["- None."]
        out.append("")

    out += ["## Final script", _fence(final_script), ""]
    out += ["## Research sources", NO_SOURCES_TEXT, ""]
    out += ["## Research text the model saw", _fence(research), ""]

    out += ["## Raw script as the model first returned it (before cleaning)",
            _fence(fc.get("raw_script_first", "")), ""]
    if fc.get("raw_script_redraft") is not None:
        out += ["## Raw redraft script as the model returned it (before cleaning)",
                _fence(fc["raw_script_redraft"]), ""]
    out += [_marker_text(fc), ""]

    if video_url:
        out.append(f"{VIDEO_PREFIX}{video_url}")
    return "\n".join(out) + "\n"


def _write_bytes_atomic(path: Path, text: str) -> None:
    """Encode the whole text first, then write it to a temporary file and move it into place.

    Encoding first means a bad character cannot leave an empty or half-written sheet. A character
    that UTF-8 cannot hold (such as a lone surrogate) is written as a visible escape, for example
    \\ud800, so it is never silently dropped. Bytes are written exactly, so \\r\\n and \\n inside a
    raw script stay as the model wrote them. On any failure the temporary file is removed and the
    error is raised; the old file, if there was one, is left as it was."""
    data = text.encode("utf-8", errors="backslashreplace")
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            tmp.unlink()
        raise


def write_review_sheet(review_name: str, *, topic: str, fact_check: dict | None,
                       final_script: str, research: str) -> Path:
    """Write queue/review/<review_name>.md under the current working directory (UTF-8).

    The raw script is kept byte for byte, except that a character UTF-8 cannot represent (a lone
    surrogate) is shown as an escape such as \\ud800, because it cannot be stored as text."""
    path = review_sheet_path(review_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_review_sheet(queue_name=review_name, topic=topic, fact_check=fact_check,
                              final_script=final_script, research=research)
    _write_bytes_atomic(path, text)
    return path


def add_video_line(review_name: str, url: str) -> Path | None:
    """Put a 'Video: <url>' line at the end of the sheet, replacing an earlier one if there is one.

    The layout matches build_review_sheet(video_url=...): a blank line before the Video line,
    and the file ends with one newline. Only the last line is touched. Everything before it is
    kept byte for byte, so the raw script sections are never rewritten.

    Returns None without writing anything when there is no sheet for this queue file: the file is
    missing, or it is empty or whitespace only (for example a sheet that failed to write). A Video
    line alone never becomes a sheet. A failure on an existing sheet still raises."""
    path = review_sheet_path(review_name)
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    if not text.strip():
        return None
    stripped = text.rstrip("\r\n")
    start_of_last_line = max(stripped.rfind("\n"), stripped.rfind("\r")) + 1
    if stripped[start_of_last_line:].startswith(VIDEO_PREFIX):
        before = stripped[:start_of_last_line]  # keeps the blank line, drops the old Video line
    else:
        before = stripped + "\n\n"
    _write_bytes_atomic(path, before + f"{VIDEO_PREFIX}{url}\n")
    return path
