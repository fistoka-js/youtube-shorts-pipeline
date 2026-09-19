"""Queue orchestrator: pulls one topic file from queue/pending/, runs it
through the full pipeline (draft -> produce -> upload), and moves it to
queue/done/ when finished. Designed to be run unattended (e.g. by a
scheduled task or CI job) - one call processes exactly one topic.
"""

import json
import shutil
import sys
from pathlib import Path

QUEUE_PENDING = Path("queue/pending")
QUEUE_DONE = Path("queue/done")


def pick_next_topic_file() -> Path | None:
    """Return the oldest pending topic file, or None if the queue is empty."""
    files = sorted(QUEUE_PENDING.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return files[0] if files else None


def main():
    QUEUE_PENDING.mkdir(parents=True, exist_ok=True)
    QUEUE_DONE.mkdir(parents=True, exist_ok=True)

    topic_file = pick_next_topic_file()
    if not topic_file:
        print("Queue is empty - nothing to do.")
        return

    print(f"Processing: {topic_file.name}")
    data = json.loads(topic_file.read_text())

    topic = data.get("topic")
    niche = data.get("niche", "general")
    platform = data.get("platform", "shorts")

    if not topic:
        print(f"ERROR: {topic_file.name} has no 'topic' field - skipping, leaving in place")
        sys.exit(1)

    from verticals.__main__ import cmd_run

    class RunArgs:
        pass

    RunArgs.news = topic
    RunArgs.niche = niche
    RunArgs.platform = platform
    RunArgs.provider = None
    RunArgs.voice = None
    RunArgs.lang = "en"
    RunArgs.dry_run = False
    RunArgs.context = ""
    RunArgs.discover = False
    RunArgs.auto_pick = False

    try:
        cmd_run(RunArgs())
    except Exception as e:
        print(f"ERROR processing {topic_file.name}: {e}")
        print("Leaving file in queue/pending/ for retry.")
        sys.exit(1)

    done_path = QUEUE_DONE / topic_file.name
    shutil.move(str(topic_file), str(done_path))
    print(f"Moved {topic_file.name} to queue/done/")


if __name__ == "__main__":
    main()