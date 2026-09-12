"""Anti-templating variation layer.

Tracks recent structural choices (intro style, narrative structure, outro
treatment) across videos and picks ones that weren't used recently, so
consecutive videos don't share the same skeleton - this is the mitigation
YouTube's "inauthentic content" policy specifically targets (channels that
upload the same template with superficial word swaps).
"""

import json
import random
from pathlib import Path

from .config import SKILL_DIR

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


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except Exception:
        return []


def _save_history(history: list[dict]):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def choose_variation(niche: str, lookback: int = 3) -> dict:
    """Pick an intro style, narrative structure, and outro treatment that
    weren't used in the last `lookback` videos for this niche - falls back
    to picking from the full list if everything's been used recently
    (e.g. lookback exceeds the number of available options)."""
    history = _load_history()
    recent = [h for h in history if h.get("niche") == niche][-lookback:]

    recent_intros = {h.get("intro_style") for h in recent}
    recent_structures = {h.get("narrative_structure") for h in recent}
    recent_outros = {h.get("outro_treatment") for h in recent}

    def pick(options: list[str], recently_used: set[str]) -> str:
        available = [o for o in options if o not in recently_used]
        return random.choice(available or options)

    return {
        "intro_style": pick(INTRO_STYLES, recent_intros),
        "narrative_structure": pick(NARRATIVE_STRUCTURES, recent_structures),
        "outro_treatment": pick(OUTRO_TREATMENTS, recent_outros),
    }


def record_variation(niche: str, variation: dict):
    """Append this video's chosen variation to history after a successful draft."""
    history = _load_history()
    history.append({
        "niche": niche,
        "intro_style": variation.get("intro_style", ""),
        "narrative_structure": variation.get("narrative_structure", ""),
        "outro_treatment": variation.get("outro_treatment", ""),
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
    )