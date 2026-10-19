"""Anti-templating variation layer.

Tracks recent structural choices (intro style, narrative structure, outro
treatment) across videos and picks ones that weren't used recently, so
consecutive videos don't share the same skeleton - this is the mitigation
YouTube's "inauthentic content" policy specifically targets (channels that
upload the same template with superficial word swaps).
"""

import json
import os
import random
from pathlib import Path

from .config import SKILL_DIR
from .log import get_logger

HISTORY_PATH = SKILL_DIR / "variation_history.json"

INTRO_STYLES = [
    "cold-open curiosity gap: state the most surprising fact immediately, no setup",
    "direct question: open by asking the exact question the video answers",
    "myth-busting: open by stating a common belief, then say it's wrong",
    "scene-setting: open by describing a vivid, specific moment or scenario",
    "stakes-first: open by explaining why this fact matters before revealing it",
]

NARRATIVE_STRUCTURES = [
    "chronological: walk through how/why something developed over time, in order",
    "question-led: structure the whole video as a chain of questions, each answered before raising the next",
    "myth-busting: structure around correcting a series of common misconceptions",
    "what-if: explore a hypothetical framing throughout (what if this trait didn't exist, etc.)",
    "process-walkthrough: explain a mechanism or system step by step, in the order it actually operates",
]

OUTRO_TREATMENTS = [
    "direct question to the audience inviting a comment",
    "forward-looking teaser about a related unanswered question",
    "reflective wrap-up connecting the specific fact to a bigger idea",
    "call-back to the video's opening hook, resolving it",
    "plain subscribe CTA with no extra framing",
]

VISUAL_TREATMENTS = [
    "stock-footage-led: maximize real subject footage, keep abstract/diagram prompts to the bare minimum needed",
    "still-image-with-motion: allow more still-photo-style prompts (landscapes, close-ups, portraits) that work well with Ken Burns pans/zooms, alongside video clips",
    "diagram-and-data: lean into the upper end of the allowed abstract/diagram prompt ratio - more anatomical diagrams, data visualizations, and explanatory graphics than usual",
]


def _override_path() -> Path | None:
    """The path from VARIATION_HISTORY_PATH, or None when it is unset or blank."""
    override = os.environ.get("VARIATION_HISTORY_PATH", "").strip()
    return Path(override) if override else None


def _history_path() -> Path:
    """Where the history file lives. VARIATION_HISTORY_PATH overrides the
    default (the cloud workflow points it at queue/ so the history is saved
    back to the repo). Unset or blank means the default HISTORY_PATH."""
    override = _override_path()
    return override if override is not None else HISTORY_PATH


def _load_history() -> list[dict]:
    path = _history_path()
    if not path.exists():
        return []
    if _override_path() is None:
        # Default path: exactly as before - the parsed file as-is, and an
        # empty history only if the file cannot be read or parsed.
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    # Env var path (cloud): validate, so an empty or damaged file cannot crash the run.
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        get_logger().warning(f"Variation history {path} is unreadable ({e}); starting with an empty history.")
        return []
    if not isinstance(data, list):
        get_logger().warning(f"Variation history {path} is not a list; starting with an empty history.")
        return []
    return [h for h in data if isinstance(h, dict)]


def _save_history(history: list[dict]):
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if _override_path() is None:
        # Default path: the same plain write as before.
        path.write_text(json.dumps(history, indent=2))
        return
    # Env var path (cloud): write to a temp file and swap it in, so a crash or
    # error mid-write cannot leave a half-written history in place of the valid one.
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(history, indent=2))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def choose_variation(niche: str, lookback: int = 3) -> dict:
    """Pick an intro style, narrative structure, outro treatment, and visual
    treatment that weren't used in the last `lookback` videos for this
    niche. If every option was used recently (e.g. the pool is smaller than
    lookback), pick from everything except the most recent video's choice for
    that field, so consecutive videos never share it. If the pool has only one
    option, that option is used. With no recent video, pick from the full list."""
    history = _load_history()
    recent = [h for h in history if h.get("niche") == niche][-lookback:]
    last = recent[-1] if recent else {}

    recent_intros = {h.get("intro_style") for h in recent}
    recent_structures = {h.get("narrative_structure") for h in recent}
    recent_outros = {h.get("outro_treatment") for h in recent}
    recent_visuals = {h.get("visual_treatment") for h in recent}

    def pick(options: list[str], recently_used: set[str], field: str) -> str:
        available = [o for o in options if o not in recently_used]
        if available:
            return random.choice(available)
        # Everything was used recently: avoid only the most recent video's value.
        not_last = [o for o in options if o != last.get(field)]
        return random.choice(not_last or options)

    return {
        "intro_style": pick(INTRO_STYLES, recent_intros, "intro_style"),
        "narrative_structure": pick(NARRATIVE_STRUCTURES, recent_structures, "narrative_structure"),
        "outro_treatment": pick(OUTRO_TREATMENTS, recent_outros, "outro_treatment"),
        "visual_treatment": pick(VISUAL_TREATMENTS, recent_visuals, "visual_treatment"),
    }


def record_variation(niche: str, variation: dict):
    """Append this video's chosen variation to history after a successful draft."""
    history = _load_history()
    history.append({
        "niche": niche,
        "intro_style": variation.get("intro_style", ""),
        "narrative_structure": variation.get("narrative_structure", ""),
        "outro_treatment": variation.get("outro_treatment", ""),
        "visual_treatment": variation.get("visual_treatment", ""),
    })
    # Keep history from growing unbounded - last 50 entries is plenty
    _save_history(history[-50:])


def format_variation_guidance(variation: dict) -> str:
    """Format a variation choice as prompt instructions for draft.py."""
    return (
        f"\nSTRUCTURAL VARIATION (mandatory for this video, differs from recent videos):\n"
        f"- Intro style: {variation['intro_style']}\n"
        f"- Narrative structure: {variation['narrative_structure']}\n"
        f"- Outro treatment: {variation['outro_treatment']}\n"
        f"- Visual treatment for b-roll prompts: {variation['visual_treatment']}\n"
    )