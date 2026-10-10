"""Shared section timing for long-form videos.

Chapters, Shorts cutpoints and per-section b-roll all need to know where each
script section starts in the real audio. Counting words in the script only
works if the script and Whisper have the same word count, which they often do
not, so boundaries drifted late in long videos. This module instead finds each
section's opening words inside the Whisper word list, so all three stages use
the same boundaries."""

import re

from .log import log

# How many opening words to try matching, longest first. A match is never
# accepted on fewer than MIN_MATCH_WORDS words: 1 or 2 words (e.g. "the dog")
# are too common and would match the wrong place in the transcript.
MAX_MATCH_WORDS = 6
MIN_MATCH_WORDS = 3


_DASHES = re.compile(r"[-‐‑‒–—―]")


def _normalize(token: str) -> str:
    """Lowercase and strip punctuation. May return an empty string."""
    return re.sub(r"[^\w]", "", token.lower())


def _tokens(text: str) -> list[str]:
    """Split text into normalized word tokens. Dashes separate words (so an
    unspaced "ten—or" becomes "ten" and "or", as Whisper would hear it),
    punctuation is stripped, and tokens that end up empty are dropped."""
    spaced = _DASHES.sub(" ", text)
    return [n for n in (_normalize(part) for part in spaced.split()) if n]


def compute_section_timings(sections: list, words: list[dict]) -> list[dict]:
    """Find where each section starts and ends in the Whisper word list.

    sections: list of section dicts, each with "id" and "narration".
    words: list of {"word", "start", "end"} dicts from Whisper, in order.

    Returns one dict per section, in the same order as `sections`:
        {"id": str, "start_idx": int, "start": float, "end": float, "method": str}
    where method is "match6".."match3" when the opening words were found in the
    audio (never fewer than 3 words), or "proportional" when they were not and
    the old word-count estimate was used instead.

    A section's end is the end of the word just before the next section starts
    (or the last word for the final section).
    """
    if not words or not sections:
        return []

    last = len(words) - 1

    # Flatten Whisper words into (word_index, normalized_token) pairs, skipping
    # tokens that are empty after normalizing.
    flat = []
    for idx, w in enumerate(words):
        for tok in _tokens(str(w.get("word", ""))):
            flat.append((idx, tok))

    # Pass 1: find each section's matched start in the audio. A section that
    # cannot be matched gets None here and its old-style word-count estimate.
    matched = []    # per section: Whisper word index if matched, else None
    methods = []    # per section: "match6".."match3" or "proportional"
    estimates = []  # per section: old word-count estimate (used only on fallback)
    cursor = 0      # position in `flat`; only ever moves forward on a match
    word_count_so_far = 0  # old-style raw word count

    for sec in sections:
        sec_id = sec.get("id", "")
        raw_words = sec.get("narration", "").split()
        sec_tokens = _tokens(sec.get("narration", ""))

        found = None  # (flat_position, k)
        for k in range(min(MAX_MATCH_WORDS, len(sec_tokens)), MIN_MATCH_WORDS - 1, -1):
            head = sec_tokens[:k]
            for p in range(cursor, len(flat) - k + 1):
                if all(flat[p + j][1] == head[j] for j in range(k)):
                    found = (p, k)
                    break
            if found:
                break

        if found:
            p, k = found
            matched.append(flat[p][0])
            methods.append(f"match{k}")
            cursor = p + 1
        else:
            matched.append(None)
            methods.append("proportional")
            if raw_words:
                log(f"Section {sec_id}: opening words not found in audio, using word-count estimate")

        estimates.append(min(word_count_so_far, last))
        word_count_so_far += len(raw_words)

    # Pass 2: resolve starts. Matched starts are kept exactly as found. A
    # fallback estimate is kept inside its neighbours - not before the previous
    # section's start, and not after the next matched section's start - so one
    # bad estimate can neither move a good match nor make spans invert.
    starts = []
    prev_start = 0
    for i in range(len(sections)):
        if matched[i] is not None:
            start_idx = matched[i]
        else:
            next_matched = next((m for m in matched[i + 1:] if m is not None), last)
            start_idx = min(max(estimates[i], prev_start), next_matched)
        starts.append(start_idx)
        prev_start = start_idx

    timings = []
    for i, sec in enumerate(sections):
        start_idx = starts[i]
        if i + 1 < len(starts):
            end_idx = max(start_idx, starts[i + 1] - 1)
        else:
            end_idx = max(start_idx, last)
        timings.append({
            "id": sec.get("id", ""),
            "start_idx": start_idx,
            "start": words[start_idx]["start"],
            "end": words[end_idx]["end"],
            "method": methods[i],
        })
    return timings
