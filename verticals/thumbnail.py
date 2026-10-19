"""Thumbnail generation — Pexels landscape photo (16:9, 1280x720) + Pillow text overlay.

The photo is chosen by CLIP: about 15 landscape Pexels candidates are scored
against the topic, and only a photo that clears THUMB_MIN_CLIP_SCORE is used.
"""

import os
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from . import broll as _broll
from .log import log

THUMB_WIDTH = 1280
THUMB_HEIGHT = 720

# Thumbnail: how many Pexels landscape photos are fetched and scored per query.
THUMB_CANDIDATES = 15

# Thumbnail: the lowest CLIP score (photo against "a photo of {subject}") a photo
# must reach to be used. Below it the photo is judged off-topic: the next query is
# tried, then the plain colour frame. Measured with CLIP ViT-B-32 on real Pexels
# results, by a one-off scratch script kept outside the repo (live network calls,
# so not a unit test): on-topic best photos scored 0.267 to 0.338 (onion, bat, ice cube), and a
# wrong-topic photo (chickpeas scored against "onion") topped out at 0.235. 0.25 sits
# between them. Only three subjects were measured, so this is a judgement call: if
# real drafts fall back to the colour frame too often, or pick off-topic photos,
# adjust it from the "CLIP score" values in the build log.
THUMB_MIN_CLIP_SCORE = 0.25


def get_pexels_key() -> str:
    """Load PEXELS_API_KEY from env or config.json."""
    import json
    if os.environ.get("PEXELS_API_KEY"):
        return os.environ["PEXELS_API_KEY"]
    try:
        from .config import SKILL_DIR
        config_path = SKILL_DIR / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            return data.get("PEXELS_API_KEY", "")
    except Exception:
        pass
    return ""


def _candidate_count_text(n: int) -> str:
    """"0 candidates", "1 candidate", "15 candidates" (correct singular and plural)."""
    return f"{n} candidate" if n == 1 else f"{n} candidates"


def _clean_prompt_query(prompt: str) -> str:
    """Strip cinematography jargon from the thumbnail prompt, keeping the subject.
    Returns "" when nothing is left (the caller decides the last resort)."""
    query = prompt.split(".")[0].split(",")[0].strip()
    jargon = [
        "bold white text reading", "text reading", "over a", "dramatic close-up of",
        "close-up of", "close up of", "cinematic and striking", "shallow depth of field",
    ]
    for j in jargon:
        query = re.sub(j, "", query, flags=re.IGNORECASE).strip()
    return query.strip("'\" ")


def _search_pexels_photos(query: str, api_key: str) -> list[dict]:
    """Search Pexels for landscape photos. Returns the raw photo dicts (may be empty)."""
    r = requests.get(
        "https://api.pexels.com/v1/search",
        params={"query": query, "orientation": "landscape", "per_page": THUMB_CANDIDATES},
        headers={"Authorization": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        # Logged so a rate limit or bad key is not mistaken for "no results".
        log(f"Pexels photo search for \"{query}\" failed: HTTP {r.status_code}")
        return []
    return r.json().get("photos", [])


def _clip_text_feature(text: str, clip):
    """CLIP text vector for one text, unit length. Computed once per scoring run.
    clip is the (model, preprocess, tokenizer) tuple from broll._load_clip."""
    import torch
    model, _, tokenizer = clip
    with torch.no_grad():
        feat = model.encode_text(tokenizer([text]))
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat


def _clip_score_image(img: Image.Image, text_feat, clip) -> float:
    """CLIP similarity of one image to a text vector from _clip_text_feature,
    both unit length (same method as broll._best_pool_match)."""
    import torch
    model, preprocess, _ = clip
    with torch.no_grad():
        img_feat = model.encode_image(preprocess(img).unsqueeze(0))
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        return float((img_feat @ text_feat.T).item())


def _score_photos(photos: list[dict], subject_text: str, clip) -> list[tuple[float, dict]]:
    """Score each photo's small preview against "a photo of {subject_text}".
    A photo that cannot be downloaded or scored is skipped. Best score first."""
    text_feat = _clip_text_feature(f"a photo of {subject_text}", clip)
    scored = []
    for photo in photos:
        url = (photo.get("src") or {}).get("medium")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            scored.append((_clip_score_image(img, text_feat, clip), photo))
        except Exception as e:
            log(f"  Skipped Pexels photo {photo.get('id')}: {type(e).__name__}: {e}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _download_full_photo(photo: dict, output_path: Path) -> bool:
    """Download the full-size image of a photo to output_path. False on failure,
    including a reply that is not a readable image (e.g. an HTML error page)."""
    src = photo.get("src") or {}
    url = src.get("large2x") or src.get("original")
    try:
        if not url:
            raise ValueError("no full-size image URL")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        img.load()  # raises if the bytes are not a complete image
        img.convert("RGB").save(output_path)
        return True
    except Exception as e:
        log(f"  Could not download Pexels photo {photo.get('id')}: {e}")
        return False


def _generate_thumb_image(subject: str, prompt: str, output_path: Path, details: dict):
    """Get a real photo from Pexels that CLIP judges to match the topic, or a
    solid colour frame when none does. Fills details with what was done."""
    api_key = get_pexels_key()
    subject = (subject or "").strip()
    cleaned = _clean_prompt_query(prompt or "")
    # The text CLIP scores photos against: the subject, or the cleaned prompt when
    # there is no subject. It is the same text for both queries, so the second query
    # is judged by the same topic as the first.
    scoring_text = subject or cleaned or "cat"

    # Query 1: the subject (or the cleaned prompt when there is no subject).
    # Query 2: the cleaned prompt, only when it adds something new.
    queries = [("subject" if subject else "cleaned_prompt", subject or cleaned or "cat")]
    if cleaned and cleaned.lower() != queries[0][1].lower():
        queries.append(("cleaned_prompt", cleaned))

    planned = ", ".join(q for _, q in queries)
    details.update(search_words=planned, candidates=0, photo_id=None,
                   clip_score=None, query_won=None, fallback="colour_frame")

    chosen = None  # (score, photo, query_label, query, candidate_count, query_index)
    searched = []  # (query, candidate_count) for every query actually sent to Pexels
    if not api_key:
        log("No Pexels key — using solid color fallback")
    else:
        try:
            clip = _broll._load_clip()
        except Exception as e:
            # Never pick an unscored photo: without CLIP there is no relevance check.
            log(f"CLIP model unavailable ({type(e).__name__}: {e}) — "
                f"using solid color fallback instead of an unchecked photo")
            clip = None

        if clip is not None:
            for idx, (label, query) in enumerate(queries):
                try:
                    photos = _search_pexels_photos(query, api_key)
                except Exception as e:
                    log(f"Pexels photo search failed: {e}")
                    photos = []
                searched.append((query, len(photos)))
                try:
                    scored = _score_photos(photos, scoring_text, clip)
                except Exception as e:
                    log(f"CLIP scoring failed for \"{query}\": {type(e).__name__}: {e}")
                    scored = []

                # Best first. Take the best photo that clears the threshold and
                # downloads; if its full-size download fails, try the next one.
                for score, photo in scored:
                    if score < THUMB_MIN_CLIP_SCORE:
                        break
                    if _download_full_photo(photo, output_path):
                        chosen = (score, photo, label, query, len(photos), idx)
                        break
                if chosen is not None:
                    break

                if not scored:
                    why = "no scorable photos came back"
                elif scored[0][0] < THUMB_MIN_CLIP_SCORE:
                    why = f"best CLIP score {scored[0][0]:.3f} is below {THUMB_MIN_CLIP_SCORE}"
                else:
                    why = "no photo above the threshold could be downloaded"
                if idx + 1 < len(queries):
                    log(f"No good photo for \"{query}\" ({why}) — "
                        f"trying second query \"{queries[idx + 1][1]}\"")
                else:
                    log(f"No good photo for \"{query}\" ({why})")

    if chosen is not None:
        score, photo, label, query, count, idx = chosen
        photo_id = photo.get("id")
        fallback = "second_query" if idx > 0 else "none"
        details.update(search_words=query, candidates=count, photo_id=photo_id,
                       clip_score=round(score, 4), query_won=label, fallback=fallback)
        log(f"Thumbnail photo: search words \"{query}\", {_candidate_count_text(count)}, "
            f"chosen Pexels id {photo_id}, CLIP score {score:.3f}"
            + (" (second query used)" if idx > 0 else ""))
    else:
        # Fallback: solid color frame if no Pexels key, CLIP unavailable, or no photo passed
        searched_text = ", ".join(f"\"{q}\" ({_candidate_count_text(n)})"
                                  for q, n in searched) or planned
        total = sum(n for _, n in searched)
        details.update(search_words=", ".join(q for q, _ in searched) or planned,
                       candidates=total)
        log(f"No Pexels photo passed the CLIP check (minimum {THUMB_MIN_CLIP_SCORE}) "
            f"after searching {searched_text} — using solid color fallback")
        img = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), (20, 20, 60))
        img.save(output_path)

    img = Image.open(output_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Crop to exact 16:9 rather than stretch, to avoid distortion
    target_ratio = THUMB_WIDTH / THUMB_HEIGHT
    w, h = img.size
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
    img.save(output_path)


def _overlay_title(image_path: Path, title: str, output_path: Path):
    """Overlay bold title text with drop shadow on the thumbnail."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    font_size = 96
    font = None
    for font_name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    max_width = THUMB_WIDTH - 100
    lines = _wrap_text(draw, title, font, max_width)
    text_block = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), text_block, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (THUMB_WIDTH - text_w) // 2
    y = THUMB_HEIGHT - text_h - 140  # keep clear of Shorts UI overlay near bottom

    shadow_offset = 4
    draw.multiline_text(
        (x + shadow_offset, y + shadow_offset),
        text_block, fill=(0, 0, 0), font=font, align="center",
    )
    draw.multiline_text(
        (x, y), text_block, fill=(255, 255, 255), font=font, align="center",
    )

    img.save(output_path)


def _wrap_text(draw: ImageDraw.Draw, text: str, font, max_width: int) -> list[str]:
    """Simple word-wrap for Pillow text rendering."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_thumbnail(draft: dict, out_dir: Path, details: dict | None = None) -> Path:
    """Generate a YouTube thumbnail (1280x720, 16:9) from a CLIP-checked Pexels photo + text overlay.

    Searches Pexels for the draft's broll_subject (falling back to the cleaned
    thumbnail_prompt), scores the candidates with CLIP, and overlays a short title.
    details: optional dict, filled with search_words, candidates, photo_id,
    clip_score, query_won and fallback (for the draft's thumbnail stage record).
    Returns path to the final thumbnail PNG.
    """
    subject = draft.get("broll_subject") or ""
    prompt = draft.get("thumbnail_prompt") or ""
    title = draft.get("youtube_title", draft.get("news", ""))
    job_id = draft.get("job_id", "unknown")

    raw_path = out_dir / f"thumb_raw_{job_id}.png"
    final_path = out_dir / f"thumb_{job_id}.png"

    log("Sourcing thumbnail photo via Pexels...")
    thumb_details = {}
    _generate_thumb_image(subject, prompt, raw_path, thumb_details)
    if details is not None:
        details.update(thumb_details)

    log("Adding title overlay...")
    _overlay_title(raw_path, title, final_path)

    log(f"Thumbnail saved: {final_path.name}")
    return final_path
