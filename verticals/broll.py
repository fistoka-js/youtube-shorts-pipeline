"""Pexels stock video b-roll + Pollinations image fallback + Ken Burns animation."""

import base64
from pathlib import Path

import requests
from PIL import Image

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, get_gemini_key, run_cmd
from .log import log
import re

from .log import log
from .retry import with_retry

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


def _search_pexels_video(query: str, api_key: str) -> str | None:
    """Search Pexels for a portrait-friendly video clip. Returns a direct MP4 URL or None."""
    r = requests.get(
        "https://api.pexels.com/videos/search",
        params={"query": query, "orientation": "portrait", "per_page": 3, "size": "medium"},
        headers={"Authorization": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    videos = data.get("videos", [])
    if not videos:
        return None

    # Pexels doesn't give us a relevance score, but each video's own "url"
    # field is a human-readable slug generated from its title (e.g.
    # ".../video/cat-playing-with-tape-1358988/"). Check the top few
    # candidates for one whose slug actually contains a query keyword,
    # instead of blindly trusting result #1 - which can occasionally be a
    # poor match (e.g. searching "octopus eye" returning an unrelated
    # close-up of a human forehead).
    query_words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3]
    video = videos[0]
    matched = False
    for candidate in videos[:3]:
        slug = candidate.get("url", "").lower()
        if any(w in slug for w in query_words):
            video = candidate
            matched = True
            break
    if not matched:
        log(f"  No confident keyword match for \"{query}\" in top results - using best guess")
    files = video.get("video_files", [])
    portrait_files = [f for f in files if f.get("height", 0) > f.get("width", 0)]
    candidates = portrait_files or files
    if not candidates:
        return None
    best = max(candidates, key=lambda f: f.get("height", 0))
    return best.get("link")


def _download_and_crop_video(url: str, out_path: Path, duration: float) -> Path:
    """Download a Pexels clip and crop/scale/trim it to exact portrait dimensions."""
    tmp_path = out_path.with_suffix(".raw.mp4")
    r = requests.get(url, timeout=90, stream=True)
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)

    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
    )
    run_cmd([
        "ffmpeg", "-i", str(tmp_path), "-t", str(duration),
        "-vf", vf, "-an", "-r", "30", "-pix_fmt", "yuv420p",
        str(out_path), "-y", "-loglevel", "quiet",
    ])
    tmp_path.unlink(missing_ok=True)
    return out_path


@with_retry(max_retries=3, base_delay=2.0)
def _generate_image_gemini(prompt: str, output_path: Path, api_key: str):
    """Unused now (kept for reference); Pollinations is the image fallback."""
    raise RuntimeError("Gemini image path disabled — using Pollinations fallback")


def _fallback_frame(i: int, out_dir: Path) -> Path:
    """Solid colour fallback frame if all else fails."""
    colors = [(20, 20, 60), (40, 10, 40), (10, 30, 50)]
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), colors[i % len(colors)])
    path = out_dir / f"broll_{i}.png"
    img.save(path)
    return path


def _generate_pollinations_image(prompt: str, out_path: Path) -> Path:
    """Generate a still image via Pollinations as fallback when no video match exists."""
    encoded_prompt = requests.utils.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={VIDEO_WIDTH}&height={VIDEO_HEIGHT}&nologo=true"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    img = Image.open(out_path)
    if img.size != (VIDEO_WIDTH, VIDEO_HEIGHT):
        img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
        img.save(out_path)
    return out_path


def generate_broll(prompts: list, out_dir: Path, clip_duration: float = 4.0) -> list[dict]:
    """Generate b-roll: real Pexels video clips where a good match exists,
    Pollinations-generated stills (for Ken Burns) as fallback otherwise.

    Returns a list of dicts: {"path": Path, "type": "video" | "image"}.
    """
    import time

    pexels_key = get_pexels_key()
    frames = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        log(f"Sourcing b-roll {i+1}/{total}...")

        # Derive a short search query from the prompt: strip technical/cinematography
        # jargon (Pexels indexes real footage, not photographer-brief phrasing) and
        # keep just the subject.
        import re
        raw_query = prompt.split(".")[0].split(",")[0].strip()
        jargon = [
            "extreme close-up of", "extreme close up of", "close-up of", "close up of",
            "cinematic establishing shot of", "cinematic shot of", "wide cinematic shot of",
            "wide shot of", "dramatic close-up of", "dramatic low-angle shot of",
            "slow-motion close-up of", "slow motion close-up of", "overhead shot of",
            "aerial cinematic shot of", "atmospheric shot of", "a shot of",
        ]
        query = raw_query
        for j in jargon:
            query = re.sub(j, "", query, flags=re.IGNORECASE).strip()
        query = query or raw_query  # never end up with an empty query

        got_video = False
        if pexels_key:
            try:
                video_url = _search_pexels_video(query, pexels_key)
                if video_url:
                    out_path = out_dir / f"broll_{i}.mp4"
                    _download_and_crop_video(video_url, out_path, clip_duration)
                    frames.append({"path": out_path, "type": "video"})
                    got_video = True
                    log(f"  Found Pexels footage: \"{query}\"")
            except Exception as e:
                log(f"  Pexels search/download failed: {e}")

        if not got_video:
            try:
                out_path = out_dir / f"broll_{i}.png"
                _generate_pollinations_image(prompt, out_path)
                frames.append({"path": out_path, "type": "image"})
                log(f"  No stock match — generated fallback image")
                time.sleep(1.0)
            except Exception as e:
                log(f"  Fallback image failed: {e} — using solid color")
                frames.append({"path": _fallback_frame(i, out_dir), "type": "image"})

    return frames


def animate_frame(img_path: Path, out_path: Path, duration: float, effect: str = "zoom_in"):
    """Ken Burns animation on a single still frame (used only for image fallbacks)."""
    fps = 30
    frames = int(duration * fps)
    w, h = VIDEO_WIDTH, VIDEO_HEIGHT

    if effect == "zoom_in":
        vf = (
            f"scale={int(w * 1.12)}:{int(h * 1.12)},"
            f"zoompan=z='1.12-0.12*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    elif effect == "pan_right":
        vf = (
            f"scale={int(w * 1.15)}:{int(h * 1.15)},"
            f"zoompan=z=1.15:x='0.15*iw*on/{frames}':y='ih*0.075'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    else:  # zoom_out
        vf = (
            f"scale={int(w * 1.12)}:{int(h * 1.12)},"
            f"zoompan=z='1.0+0.12*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )

    run_cmd([
        "ffmpeg", "-loop", "1", "-i", str(img_path),
        "-vf", vf, "-t", str(duration), "-r", str(fps),
        "-pix_fmt", "yuv420p", str(out_path), "-y", "-loglevel", "quiet",
    ])