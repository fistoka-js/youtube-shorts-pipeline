"""Queue add: turns the ticked checklist lines of a topic-suggestion issue into queue files.

`python -m verticals queue-add` is what the "Add ticked topics to the queue" workflow runs. It:
1. Reads the issue body from an environment variable (never from the command line).
2. Keeps the lines that start with "- [x] " or "- [X] ". The topic is the text before the first " | ".
3. Skips a topic that matches (ignoring case and punctuation) a topic already in queue/pending,
   queue/in-progress, queue/done or queue/failed, or one repeated earlier in the same issue.
4. Writes one queue/pending/NNN-<slug>.json file per remaining topic. Nothing is ever overwritten.
5. Writes a manifest (one new file path per line) and a comment body for the issue.

Standard library only: the add-topics workflow installs no packages. Every path is relative to the
current directory, so the workflow runs it from the repository root.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

QUEUE_FOLDERS = ("pending", "in-progress", "done", "failed")
MAX_TOPIC_CHARS = 120
MAX_SLUG_CHARS = 40
NICHE = "science"
PLATFORM = "shorts"

_TICKED = re.compile(r"^- \[[xX]\] (.*)$")
_TOPIC_END = re.compile(r"^\|\s|\s\|\s")  # " | " between the topic and the category, or a leading "| "
_HAS_WORD = re.compile(r"[^\W_]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MARKDOWN_SPECIAL = "\\`*_[]()<>~|!"


def _key(text: str) -> str:
    """Comparison form of a topic: lower case, punctuation removed, single spaces only.
    Same rule as verticals/suggest.py (_key), copied so this module does not import the LLM code."""
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    return " ".join(text.split())


def clean_topic(raw: str) -> str:
    """Plain one-line text: control characters become spaces, runs of whitespace collapse,
    and the result is cut to MAX_TOPIC_CHARS. Empty when nothing usable is left."""
    text = _CONTROL.sub(" ", raw)
    text = " ".join(text.split())
    return text[:MAX_TOPIC_CHARS].rstrip()


def ticked_topics(body: str) -> list[str]:
    """The cleaned topics of the ticked lines, in order. Unticked lines and every other line are
    ignored. Lines may end with CRLF and may have spaces around them. Repeats are kept here."""
    topics = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        match = _TICKED.match(line)
        if not match:
            continue
        # The topic is the text before the first " | " (or a pipe that opens the rest of the line).
        topic = clean_topic(_TOPIC_END.split(match.group(1), maxsplit=1)[0])
        if _HAS_WORD.search(topic):  # a topic with no letters or digits is empty for this purpose
            topics.append(topic)
    return topics


def slugify(topic: str) -> str:
    """ASCII words of the topic joined with hyphens, at most MAX_SLUG_CHARS long. The cut is made
    at a hyphen where possible and never ends in a hyphen. "topic" when nothing is left."""
    ascii_text = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode("ascii").lower()
    slug = "-".join(re.findall(r"[a-z0-9]+", ascii_text))
    if len(slug) > MAX_SLUG_CHARS:
        cut = slug[:MAX_SLUG_CHARS]
        if slug[MAX_SLUG_CHARS] != "-":
            dash = cut.rfind("-")
            if dash > 0:
                cut = cut[:dash]
        slug = cut.rstrip("-")
    return slug or "topic"


def _topic_from_file(path: Path) -> str | None:
    """The topic in one queue file, or None when the file is not usable. Same rule as
    verticals/suggest.py (_topic_from_file): a UTF-8 BOM is tolerated, and the file must be a JSON
    object with a non-blank text 'topic'."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8-sig"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    topic = data.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        return None
    return topic


def existing_keys(queue_dir: Path) -> set[str]:
    """The comparison keys of every usable topic in the four queue folders."""
    keys = set()
    for folder in QUEUE_FOLDERS:
        folder_path = Path(queue_dir) / folder
        if not folder_path.is_dir():
            continue
        for path in sorted(folder_path.glob("*.json")):
            if not path.is_file():
                continue
            topic = _topic_from_file(path)
            if topic is not None:
                keys.add(_key(topic))
    return keys


def next_number(queue_dir: Path) -> int:
    """The highest first integer in any file name across the four queue folders, plus 1.
    Names with no integer count as nothing. With no numbers at all, this is 1."""
    highest = 0
    for folder in QUEUE_FOLDERS:
        folder_path = Path(queue_dir) / folder
        if not folder_path.is_dir():
            continue
        for name in os.listdir(folder_path):
            match = re.search(r"[0-9]+", name)
            if match:
                highest = max(highest, int(match.group()))
    return highest + 1


def _create_file(pending: Path, number: int, slug: str, data: bytes) -> tuple[int, str]:
    """Write one new queue file and return (its number, its name). The data goes to a temporary
    file in the same folder first, then os.link gives it the final name. os.link fails when the
    name already exists, so an existing file is never overwritten; the next number is tried."""
    fd, tmp_name = tempfile.mkstemp(dir=pending, prefix=".adding-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        n = number
        while True:
            name = f"{n:03d}-{slug}.json"
            try:
                os.link(tmp_name, pending / name)
                return n, name
            except FileExistsError:
                n += 1
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def _comment_text(topic: str) -> str:
    """A topic as plain text for the issue comment: no @ mentions, no # references (so no issue
    auto-close keywords), and Markdown characters escaped."""
    text = topic.replace("@", "").replace("#", "")
    return "".join("\\" + ch if ch in _MARKDOWN_SPECIAL else ch for ch in text)


def build_comment(queued: list[tuple[str, str]], skipped: list[str], ticked_any: bool) -> str:
    """The comment body. Says plainly when nothing was ticked or everything was a duplicate."""
    lines = []
    if not ticked_any:
        lines.append("No topics were ticked on this issue, so nothing was added to the queue.")
        lines.append("")
        lines.append("To add topics, tick the ideas you want (change [ ] to [x]) and add the label "
                     "add-to-queue again. The label has been removed, so it can be added again.")
    elif not queued:
        lines.append("Every ticked topic is already in the queue (or was repeated on this issue), "
                     "so nothing was added.")
        lines.append("")
        lines.append("The label add-to-queue has been removed, so it can be added again after you "
                     "tick different topics.")
    else:
        lines.append(f"Added {len(queued)} topic(s) to the queue:")
        lines.append("")
        for topic, name in queued:
            lines.append(f"- {_comment_text(topic)} (`{name}`)")
    if skipped:
        lines.append("")
        lines.append(f"Skipped {len(skipped)} duplicate(s), already in the queue or repeated on this issue:")
        lines.append("")
        for topic in skipped:
            lines.append(f"- {_comment_text(topic)}")
    return "\n".join(lines) + "\n"


def add_topics(body: str, queue_dir: Path, manifest: Path, comment_out: Path) -> dict:
    """Queue the ticked topics of the body, then write the manifest and the comment body.
    Raises OSError when a file cannot be written. Returns a summary dict."""
    queue_dir = Path(queue_dir)
    topics = ticked_topics(body)
    seen = existing_keys(queue_dir)
    fresh, skipped = [], []
    for topic in topics:
        key = _key(topic)
        if key in seen:
            skipped.append(topic)
            continue
        seen.add(key)
        fresh.append(topic)

    pending = queue_dir / "pending"
    created = []  # (topic, file name)
    if fresh:
        pending.mkdir(parents=True, exist_ok=True)
        number = next_number(queue_dir)
        for topic in fresh:
            content = json.dumps({"topic": topic, "niche": NICHE, "platform": PLATFORM}) + "\n"
            number, name = _create_file(pending, number, slugify(topic), content.encode("utf-8"))
            number += 1
            created.append((topic, name))

    rel_folder = f"{queue_dir.as_posix()}/pending"
    manifest_text = "".join(f"{rel_folder}/{name}\n" for _, name in created)
    Path(manifest).write_bytes(manifest_text.encode("utf-8"))
    Path(comment_out).write_bytes(
        build_comment(created, skipped, ticked_any=bool(topics)).encode("utf-8")
    )
    return {"queued": created, "skipped": skipped, "ticked": len(topics)}


def main(argv=None) -> int:
    """The queue-add command. Returns the exit code: 0 when it ran (even when nothing was queued),
    1 for a real error such as an unwritable folder or a missing environment variable."""
    # allow_abbrev=False: an option such as --body is rejected, not read as --body-env.
    parser = argparse.ArgumentParser(prog="python -m verticals queue-add", allow_abbrev=False,
                                     description="Turn the ticked topics of a suggestion issue into queue files.")
    parser.add_argument("--body-env", required=True,
                        help="Name of the environment variable that holds the issue body")
    parser.add_argument("--manifest", required=True, help="File to write: one new file path per line")
    parser.add_argument("--comment-out", required=True, help="File to write: the comment body for the issue")
    parser.add_argument("--queue-dir", default="queue", help="Queue folder (default: queue)")
    args = parser.parse_args(argv)

    body = os.environ.get(args.body_env)
    if body is None:
        print(f"queue-add: the environment variable {args.body_env} is not set.", file=sys.stderr)
        return 1
    try:
        result = add_topics(body, Path(args.queue_dir), Path(args.manifest), Path(args.comment_out))
    except OSError as e:
        print(f"queue-add: could not write the queue files ({type(e).__name__}: {e}).", file=sys.stderr)
        return 1
    names = ", ".join(name for _, name in result["queued"])
    print(f"queue-add: {len(result['queued'])} topic(s) queued"
          f"{' (' + names + ')' if names else ''}; "
          f"{len(result['skipped'])} duplicate(s) skipped; {result['ticked']} ticked line(s) read.")
    return 0
