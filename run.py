"""Queue orchestrator: pulls one topic file from queue/pending/, runs it
through the full pipeline (draft -> produce -> upload), and moves it to
queue/done/ when finished. Designed to be run unattended (e.g. by a
scheduled task or CI job) - one call processes exactly one topic.

Claiming (CI only): right before the YouTube upload starts, the topic file
is moved to queue/in-progress/, committed and pushed to the branch. That way
a second run cannot upload the same topic. If the claim cannot be pushed,
nothing is uploaded and the file goes back to queue/pending/. Local runs
(GITHUB_ACTIONS is not "true") never run git and never use in-progress/.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

QUEUE_PENDING = Path("queue/pending")
QUEUE_IN_PROGRESS = Path("queue/in-progress")
QUEUE_DONE = Path("queue/done")


class ClaimError(Exception):
    """The topic could not be claimed (saved to origin) before the upload."""


class ClaimUndoError(Exception):
    """A failed claim could not be fully undone locally. Not a ClaimError: the
    topic may not be back in queue/pending/, so the normal message must not be used."""


def _natural_key(name: str):
    """Sort key for file names: digit runs compare as numbers, text as lower-case text.
    Splitting on digits gives text at even positions and numbers at odd positions, so
    two keys always compare the same types at each position."""
    parts = re.split(r"(\d+)", name.lower())
    return [int(p) if i % 2 else p for i, p in enumerate(parts)]


def pick_next_topic_file() -> Path | None:
    """Return the next pending topic file in file-name order (test2 before test9 before
    test10), or None if the queue is empty. Modification time is ignored."""
    files = sorted(QUEUE_PENDING.glob("*.json"), key=lambda p: (_natural_key(p.name), p.name))
    return files[0] if files else None


def _git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _push(branch) -> bool:
    result = _git("push", "origin", f"HEAD:{branch}", check=False)
    if result.returncode != 0:
        print(f"git push failed: {result.stderr.strip()}")
    return result.returncode == 0


def _undo_claim(topic_file: Path, claimed: Path, committed: bool):
    """Remove the local claim so the workflow's later commit step cannot push it.
    The file goes back to pending/. Other queue changes (such as the variation
    history) stay in the working tree and are committed later by the workflow.
    Raises ClaimUndoError if any undo step fails."""
    problems = []
    _git("rebase", "--abort", check=False)  # only does something if a rebase is stuck
    if committed:
        reset = _git("reset", "--mixed", "HEAD~1", check=False)  # drop the claim commit, keep files
    else:
        reset = _git("reset", "--mixed", check=False)  # unstage whatever was staged
    if reset.returncode != 0:
        problems.append(reset.stderr.strip() or f"git reset exited with status {reset.returncode}")
    try:
        shutil.move(str(claimed), str(topic_file))
    except OSError as e:
        problems.append(f"could not move {claimed.name} back to queue/pending/: {e}")
    if problems:
        raise ClaimUndoError("\n".join(problems))


def claim_topic(topic_file: Path) -> Path:
    """Move the topic to queue/in-progress/, commit all of queue/, and push.
    If the push is rejected, pull --rebase and push once more. If that also
    fails, undo everything and raise ClaimError. If the undo itself fails,
    raise ClaimUndoError instead. Returns the new path."""
    QUEUE_IN_PROGRESS.mkdir(parents=True, exist_ok=True)
    claimed = QUEUE_IN_PROGRESS / topic_file.name
    moved = False
    committed = False
    try:
        branch = os.environ.get("GITHUB_REF_NAME") or _git(
            "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        shutil.move(str(topic_file), str(claimed))
        moved = True
        # Staging all of queue/ also saves the variation history, so the
        # working tree is clean for the pull --rebase below.
        _git("add", "queue/")
        _git("commit", "-m", f"Claim {topic_file.name} for upload")
        committed = True
        if _push(branch):
            return claimed
        print("Claim push was rejected - pulling and trying once more.")
        _git("pull", "--rebase", "origin", branch)
        if _push(branch):
            return claimed
        raise ClaimError("the claim push was rejected twice")
    except Exception as e:
        # For a failed git command, show git's own error text instead of the generic exit-status message.
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            detail = e.stderr.strip()
        else:
            detail = str(e)
        if moved:
            try:
                _undo_claim(topic_file, claimed, committed)
            except ClaimUndoError as undo_error:
                # Keep the original reason the claim failed, so the message shows both.
                raise ClaimUndoError(
                    f"the claim failed because: {detail}\n{undo_error}") from e
        raise ClaimError(detail) from e


def _report_failure(topic_name: str, error_text: str, state: dict,
                    prefix: str = "ERROR processing"):
    """Print the ERROR line (or a NOTE line for a clean exit) and the note on where the topic was left."""
    print(f"{prefix} {topic_name}: {error_text}")
    if state["claimed"]:
        print("Topic left in queue/in-progress/ - check YouTube Studio to see if a video went live, "
              "then see the README recovery note.")
    elif state["upload_started"]:
        # Local run: no claim, so the file stays in pending/. The upload may have
        # gone live before the error, so do not say nothing was uploaded.
        print("Topic left in queue/pending/. The upload failed, and it is not certain whether "
              "a video went live - check YouTube Studio before running this topic again.")
    else:
        print("Topic left in queue/pending/ (nothing was uploaded).")


def main():
    QUEUE_PENDING.mkdir(parents=True, exist_ok=True)
    QUEUE_IN_PROGRESS.mkdir(parents=True, exist_ok=True)
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
    import verticals.upload as upload_module

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

    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    # Where the topic file is now: pending/ until claimed, then in-progress/.
    state = {"path": topic_file, "claimed": False, "upload_started": False, "inserted": False}
    real_upload = upload_module.upload_to_youtube

    def guarded_upload(*args, **kwargs):
        if in_ci:
            state["path"] = claim_topic(state["path"])  # raises ClaimError, no upload
            state["claimed"] = True
        state["upload_started"] = True
        url = real_upload(*args, **kwargs)
        state["inserted"] = True  # upload_to_youtube only returns after the video is live
        return url

    upload_module.upload_to_youtube = guarded_upload
    try:
        cmd_run(RunArgs())
    except ClaimUndoError as e:
        print(f"ERROR: could not undo the claim for {topic_file.name}. No video was uploaded.")
        print(f"Details: {e}")
        print("A leftover local claim may still be present. The workflow's final commit step may "
              "push it, so this topic could end up in queue/in-progress/.")
        print("See the README recovery note 'If a Topic Is Stuck in queue/in-progress/'.")
        sys.exit(1)
    except ClaimError as e:
        print(f"ERROR: could not save the claim for {topic_file.name} ({e}).")
        print("Topic left in queue/pending/, nothing was uploaded.")
        sys.exit(1)
    except Exception as e:
        if state["inserted"]:
            print(f"WARNING: {topic_file.name} was uploaded and is live, but a later step failed: {e}")
            print("The topic is still moved to queue/done/ and this run exits normally.")
        else:
            _report_failure(topic_file.name, e, state)
            sys.exit(1)
    except SystemExit as e:
        # A step (such as cmd_upload's missing-video check) exited on purpose. The same exit
        # code is always passed on. A clean exit is silent unless the topic was already
        # claimed or the upload had started.
        code = e.code
        if code is None or code == 0:
            clean = True
            reason = "the pipeline stopped with a clean exit"
        elif isinstance(code, int):
            clean = False
            reason = f"the pipeline stopped with exit code {code}"
        else:
            clean = False
            reason = f"the pipeline stopped with message: {code}"
        touched = state["claimed"] or state["upload_started"] or state["inserted"]
        if state["inserted"]:
            print(f"WARNING: {topic_file.name} was uploaded and is live, but the run stopped early "
                  f"({reason}).")
            print("The topic was NOT moved to queue/done/ - check YouTube Studio.")
        elif not clean or touched:
            _report_failure(topic_file.name, reason, state,
                            prefix="ERROR processing" if not clean else "NOTE processing")
        raise
    finally:
        upload_module.upload_to_youtube = real_upload

    done_path = QUEUE_DONE / topic_file.name
    shutil.move(str(state["path"]), str(done_path))
    print(f"Moved {topic_file.name} to queue/done/")


if __name__ == "__main__":
    main()
