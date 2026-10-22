"""Remove bracketed visual notes from script text before it is read aloud.

The Shorts script can contain inline notes such as "[B-ROLL: person yawning]"
between sentences. Those notes are for the editor, not the viewer, so they
must never reach the voice, the captions, the duration estimate or a file
name. clean_script() removes them.

Rules, applied only when the text contains at least one square bracket
(text without any square bracket is returned exactly as given):
  1. Each outermost [...] segment is removed, including anything nested in
     it, and counts as one marker. Any other stray "[" or "]" is also removed
     but is not counted.
  2. Spacing is changed only where something was removed. The spaces on each
     side of the removal become one space, unless no space is needed: at a
     line start or end (LF, CR or CRLF), or when the removal sits directly
     against a token. Directly against means no space was written between the
     removal and the token. Such a token is then joined with no space when it
     is punctuation (. , ! ? ; : the ellipsis character, the Hindi danda and
     double danda), a symbol (% or an em or en dash), a closing ) ] } or a
     closing quote, or when it is an opening ( or quote directly before the
     removal. If the author wrote a space there, one space is kept, so a token
     such as "yawn [x] ." keeps "." as its own word. A straight quote after a
     removal closes only when a matching quote is still open before it, so
     'He said [x]"...no"' keeps its space. An apostrophe inside a word is never
     treated as a quote mark.
  3. A line that held only removed text is dropped. A run of blank lines that
     the removal leaves behind collapses to one blank line.
  4. Everything else is untouched: other spacing, existing blank lines, leading
     indentation, punctuation, parentheses, asterisks and "SFX:" labels.
"""

from .log import log

# Opening and closing curly quotes and guillemets are unambiguous. Straight
# quotes (" and ') are paired: a straight quote right after a removal is a
# closing quote only when an unmatched quote of the same kind is still open
# before the removal, and it is followed by a space, a line end, the end of the
# text or punctuation. Otherwise it is an opening quote.
_OPEN_QUOTES = frozenset("“‘«")
_CLOSE_QUOTES = frozenset("”’»")
_STRAIGHT_QUOTES = frozenset("\"'")
# Punctuation that joins the word before a removal when it is attached to it:
# . , ! ? ; : the ellipsis character, and the Hindi danda and double danda.
_PUNCT = frozenset(".,!?;:…।॥")
# Symbols that join when attached: percent sign, em dash, en dash.
_SYMBOLS = frozenset("%—–")
_CLOSERS = frozenset(")]}")
_ATTACHABLE = _PUNCT | _SYMBOLS | _CLOSERS
_CUT = object()  # marks where something was removed


def _outer_marker_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of the outermost matched [...] segments.

    End is exclusive. A "[" with no matching "]" is not a range.
    """
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch == "[":
            stack.append(i)
        elif ch == "]" and stack:
            pairs.append((stack.pop(), i))
    pairs.sort()

    outer: list[tuple[int, int]] = []
    for start, end in pairs:
        if outer and start < outer[-1][1]:
            continue  # nested inside the marker already recorded
        outer.append((start, end + 1))
    return outer


def _removed_spans(text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Every (start, end) span to remove: the marker ranges and stray brackets."""
    range_ends = dict(ranges)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        if i in range_ends:
            spans.append((i, range_ends[i]))
            i = range_ends[i]
            continue
        if text[i] in "[]":
            spans.append((i, i + 1))
        i += 1
    return spans


def _build_items(text: str, spans: list[tuple[int, int]]) -> list:
    """Alternate kept text (str) and _CUT markers. Adjacent removals become one
    cut, and spaces that sit only between two removals go with the removal."""
    raw: list = []
    pos = 0
    for start, end in spans:
        if start > pos:
            raw.append(text[pos:start])
        raw.append(_CUT)
        pos = end
    if pos < len(text):
        raw.append(text[pos:])

    items: list = []
    for k, piece in enumerate(raw):
        if piece is _CUT:
            if items and items[-1] is _CUT:
                continue
            items.append(_CUT)
            continue
        if (items and items[-1] is _CUT and piece.strip(" \t") == ""
                and k + 1 < len(raw) and raw[k + 1] is _CUT):
            continue
        items.append(piece)
    return items


def _break_len(s: str) -> int:
    """Length of the line break at the start of s (2 for CRLF, 1 for LF or CR)."""
    if s.startswith("\r\n"):
        return 2
    if s[:1] in ("\n", "\r"):
        return 1
    return 0


def _ends_blank_line(out: list[str]) -> bool:
    """True when the text so far ends on a blank line (or is empty)."""
    if not out:
        return True
    if out[-1] not in "\r\n":
        return False
    j = len(out) - 2
    if out[-1] == "\n" and j >= 0 and out[j] == "\r":
        j -= 1  # the CRLF pair counts as one break
    while j >= 0 and out[j] in " \t":
        j -= 1
    return j < 0 or out[j] in "\r\n"


def _is_opening_quote(out: list[str], pos: int) -> bool:
    ch = out[pos]
    if ch == "(":
        return True
    if ch in _OPEN_QUOTES:
        return True
    if ch in _STRAIGHT_QUOTES:
        before = out[pos - 1] if pos > 0 else None
        return before is None or before in " \t\r\n"
    return False


def _is_closing_quote(ch: str, after, dq_open: int, sq_open: int) -> bool:
    """dq_open and sq_open count the straight double and single quotes already
    written before the removal. An odd count means one is still open."""
    if ch in _CLOSE_QUOTES:
        return True
    if ch in _STRAIGHT_QUOTES:
        if not (after is None or after in " \t\r\n" or after in _ATTACHABLE):
            return False
        open_count = dq_open if ch == '"' else sq_open
        return open_count % 2 == 1
    return False


def strip_script_markers(text) -> tuple[str, int]:
    """Return (cleaned_text, number_of_markers_removed). Does not log.

    None becomes "". Non-string input is returned unchanged with a count of 0.
    """
    if text is None:
        return "", 0
    if not isinstance(text, str):
        return text, 0
    if "[" not in text and "]" not in text:
        return text, 0

    ranges = _outer_marker_ranges(text)
    items = _build_items(text, _removed_spans(text, ranges))

    out: list[str] = []
    dq_open = 0  # straight double quotes written so far
    sq_open = 0  # straight single quotes written so far
    for k, item in enumerate(items):
        if item is not _CUT:
            # Count straight quotes for the pairing rule. An apostrophe inside a
            # word (directly after a letter or digit) is not a quote mark.
            prev_char = out[-1] if out else ""
            for ch in item:
                if ch == '"':
                    dq_open += 1
                elif ch == "'" and not prev_char.isalnum():
                    sq_open += 1
                prev_char = ch
            out.extend(item)
            continue

        # The removal point: drop the spaces just before it, then decide what
        # joins the text on the left with the text on the right.
        space_before = bool(out) and out[-1] in " \t"
        while out and out[-1] in " \t":
            out.pop()
        raw_right = items[k + 1] if k + 1 < len(items) else ""
        space_after = raw_right[:1] in (" ", "\t")
        right = raw_right.lstrip(" \t")
        nxt = right[:1] or None
        nlen = _break_len(right)
        line_start = not out or out[-1] in "\r\n"

        if line_start and (nxt is None or nlen):
            # The removal emptied its line.
            if nxt is None:
                if out and out[-1] in "\r\n":
                    last = out.pop()
                    if last == "\n" and out and out[-1] == "\r":
                        out.pop()
                new_right = ""
            else:
                new_right = right[nlen:]
                if _ends_blank_line(out):
                    # The removal left a run of blank lines: keep one.
                    while True:
                        stripped = new_right.lstrip(" \t")
                        n = _break_len(stripped)
                        if not n:
                            break
                        new_right = stripped[n:]
        else:
            after = right[1] if len(right) > 1 else None
            # A space the author wrote before or after the removal is kept, so a
            # punctuation token or bracket that was a word of its own stays one.
            if nxt is None or nlen or line_start:
                join = ""
            elif nxt in _ATTACHABLE or _is_closing_quote(nxt, after, dq_open, sq_open):
                join = " " if space_after else ""
            elif _is_opening_quote(out, len(out) - 1):
                join = " " if space_before else ""
            else:
                join = " "
            if join:
                out.append(join)
            new_right = right

        if k + 1 < len(items):
            items[k + 1] = new_right

    return "".join(out), len(ranges)


def log_removed(count: int) -> None:
    """Write the one "Removed N ..." line, but only when N is above zero."""
    if count:
        log(f"Removed {count} bracketed marker(s) from the script.")


def clean_script(text) -> str:
    """Return the script with bracketed markers removed.

    Logs one line, "Removed N bracketed marker(s) from the script.", only when
    N is above zero.
    """
    cleaned, removed = strip_script_markers(text)
    log_removed(removed)
    return cleaned
