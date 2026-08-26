"""Thumbnail generation — Pexels photo (9:16, matching Shorts) + Pillow text overlay."""

from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .log import log

THUMB_WIDTH = 1280
THUMB_HEIGHT = 720


def get_pexels_key() -> str:
    """Load PEXELS_API_KEY from env or config.json."""
    import os
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


def _search_pexels_photo(query: str, api_key: str) -> str | None:
    """Search Pexels for a portrait-friendly photo. Returns a direct image URL or None."""
    r = requests.get(
        "https://api.pexels.com/v1/search",
        params={"query": query, "orientation": "portrait", "per_page": 3},
        headers={"Authorization": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    photos = data.get("photos", [])
    if not photos:
        return None
    photo = photos[0]
    src = photo.get("src", {})
    return src.get("portrait") or src.get("large2x") or src.get("original")


def _generate_thumb_image(prompt: str, output_path: Path):
    """Get a real photo from Pexels matching the thumbnail prompt's subject."""
    api_key = get_pexels_key()
    photo_url = None

    if api_key:
        # Strip cinematography jargon, keep just the subject (same approach as broll.py)
        import re
        query = prompt.split(".")[0].split(",")[0].strip()
        jargon = [
            "bold white text reading", "text reading", "over a", "dramatic close-up of",
            "close-up of", "close up of", "cinematic and striking", "shallow depth of field",
        ]
        for j in jargon:
            query = re.sub(j, "", query, flags=re.IGNORECASE).strip()
        query = query.strip("'\" ") or "cat"

        try:
            photo_url = _search_pexels_photo(query, api_key)
        except Exception as e:
            log(f"Pexels photo search failed: {e}")

    if photo_url:
        resp = requests.get(photo_url, timeout=60)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
    else:
        # Fallback: solid color frame if no Pexels key or no match
        log("No Pexels photo match — using solid color fallback")
        img = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), (20, 20, 60))
        img.save(output_path)

    img = Image.open(output_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Crop to exact 9:16 rather than stretch, to avoid distortion
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


def generate_thumbnail(draft: dict, out_dir: Path) -> Path:
    """Generate a YouTube Shorts thumbnail (9:16) from a real Pexels photo + text overlay.

    Uses the thumbnail_prompt from the draft as a search query, overlays a short title.
    Returns path to the final thumbnail PNG.
    """
    prompt = draft.get("thumbnail_prompt", "cat")
    title = draft.get("youtube_title", draft.get("news", ""))
    job_id = draft.get("job_id", "unknown")

    raw_path = out_dir / f"thumb_raw_{job_id}.png"
    final_path = out_dir / f"thumb_{job_id}.png"

    log("Sourcing thumbnail photo via Pexels...")
    _generate_thumb_image(prompt, raw_path)

    log("Adding title overlay...")
    _overlay_title(raw_path, title, final_path)

    log(f"Thumbnail saved: {final_path.name}")
    return final_path