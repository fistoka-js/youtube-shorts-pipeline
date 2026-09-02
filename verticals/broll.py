"""Pexels stock video b-roll (pool-first) + Pollinations image fallback + Ken Burns."""

from pathlib import Path
import re

import requests
from PIL import Image

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, run_cmd
from .log import log
from .retry import with_retry


def get_pexels_key() -> str:
    """Load PEXELS_API_KEY from env or config.json."""
    import os, json
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


# ── CLIP (lazy-loaded, used for pool verification + prompt matching) ─────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip():
    """Lazy-load CLIP once per process."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        log("  Loading CLIP model (first call only)...")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _embed_texts_ensembled(model, tokenizer, class_prompt_lists):
    """Encode each class as the mean of several prompt-template embeddings,
    re-normalized after averaging. Reduces sensitivity to exact phrasing."""
    import torch
    class_embeds = []
    for prompts in class_prompt_lists:
        text_input = tokenizer(prompts)
        with torch.no_grad():
            feats = model.encode_text(text_input)
            feats /= feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            mean /= mean.norm()
        class_embeds.append(mean)
    return torch.stack(class_embeds)


# ── Pool: search once, verify once, match per-prompt ─────────────────────

def _build_subject_pool(subject: str, api_key: str, topic_context: str = "") -> list:
    """Search Pexels ONCE for the main subject, fetch 50 candidates,
    CLIP-verify each against food/toy/illustration negatives, and return
    confirmed-correct ones with pre-computed image embeddings."""
    import torch
    from io import BytesIO

    if not subject or not api_key:
        return []

    log(f"Building subject pool: searching Pexels for \"{subject}\" (50 candidates)...")
    r = requests.get(
        "https://api.pexels.com/videos/search",
        params={"query": subject, "per_page": 50, "size": "medium"},
        headers={"Authorization": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        log(f"  Pool search failed: HTTP {r.status_code}")
        return []

    candidates = r.json().get("videos", [])
    log(f"  Got {len(candidates)} candidates from Pexels")
    if not candidates:
        return []

    model, preprocess, tokenizer = _load_clip()

    positive_templates = [
        f"a real photograph of a {subject}",
        f"a photo of an actual {subject}",
        f"wildlife or nature footage of a {subject}",
        f"a {subject} in its natural environment",
    ]
    # Fixed negative categories, each skipped when topic overlaps.
    neg_conflicts = {
        "food": ["food", "cook", "recipe", "meal", "eat", "dish", "kitchen"],
        "toy":  ["toy", "costume", "replica", "figurine", "model", "prop"],
        "illu": ["diagram", "illustration", "cartoon", "animat", "draw"],
    }
    neg_templates = {
        "food": [
            "a cooked meal or prepared food dish",
            "a plate of food on a table",
            "a culinary or restaurant food photo",
        ],
        "toy": [
            "a toy, costume, or artistic replica",
            "a plastic figurine or model",
            "a person wearing a costume",
        ],
        "illu": [
            "an illustration, diagram, drawing, or cartoon",
            "a hand-drawn or digital graphic",
            "an animated or cartoon rendering",
        ],
    }
    topic_lower = topic_context.lower()
    active = [k for k, words in neg_conflicts.items()
              if not any(w in topic_lower for w in words)]
    class_lists = [positive_templates] + [neg_templates[k] for k in active]
    text_features = _embed_texts_ensembled(model, tokenizer, class_lists)

    verified = []
    for v in candidates:
        thumb_url = v.get("image", "")
        if not thumb_url:
            continue
        try:
            resp = requests.get(thumb_url, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img_input = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                img_feat = model.encode_image(img_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ text_features.T).squeeze(0)
            scores = sim.tolist()
            if scores[0] - max(scores[1:]) < -0.02:
                continue
            verified.append({"video": v, "image_embedding": img_feat.squeeze(0)})
        except Exception:
            continue

    log(f"  Pool verified: {len(verified)} of {len(candidates)} passed")
    return verified


def _best_pool_match(prompt_text: str, pool: list, used_ids: set) -> dict | None:
    """Pick the best-matching clip from the verified pool for a prompt,
    using CLIP text-to-image similarity. Excludes already-used clips."""
    import torch
    available = [p for p in pool if p["video"].get("id") not in used_ids]
    if not available:
        return None
    model, _, tokenizer = _load_clip()
    text_input = tokenizer([prompt_text])
    with torch.no_grad():
        text_feat = model.encode_text(text_input)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    best_score, best_entry = -1.0, None
    for entry in available:
        score = (text_feat @ entry["image_embedding"].unsqueeze(1)).item()
        if score > best_score:
            best_score, best_entry = score, entry
    return best_entry


# ── Individual search (last resort when pool is empty/exhausted) ─────────

def _fetch_pexels_candidates(query: str, api_key: str) -> list[dict]:
    """Raw Pexels video search."""
    r = requests.get(
        "https://api.pexels.com/videos/search",
        params={"query": query, "per_page": 8, "size": "medium"},
        headers={"Authorization": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        return []
    return r.json().get("videos", [])


def _best_file_link(video: dict) -> str | None:
    """Pick the highest-res file link from a Pexels video, preferring portrait."""
    files = video.get("video_files", [])
    portrait = [f for f in files if f.get("height", 0) > f.get("width", 0)]
    candidates = portrait or files
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.get("height", 0)).get("link")


def _search_pexels_video(
    query: str, api_key: str,
    exclude_ids: set | None = None,
    fallback_query: str = "",
    topic_context: str = "",
) -> tuple[str | None, int | None]:
    """Individual Pexels search with fallback — used only when the pool is
    empty or exhausted. Returns (MP4 URL, video id) or (None, None)."""
    exclude_ids = exclude_ids or set()
    videos = _fetch_pexels_candidates(query, api_key)
    available = [v for v in videos if v.get("id") not in exclude_ids]

    for candidate in available[:5]:
        link = _best_file_link(candidate)
        if link:
            return link, candidate.get("id")

    if fallback_query and fallback_query.lower() != query.lower():
        log(f"  No match for \"{query}\" — trying fallback \"{fallback_query}\"")
        fb = _fetch_pexels_candidates(fallback_query, api_key)
        for candidate in [v for v in fb if v.get("id") not in exclude_ids][:5]:
            link = _best_file_link(candidate)
            if link:
                return link, candidate.get("id")

    return None, None


# ── Download / fallback helpers ──────────────────────────────────────────

def _download_and_crop_video(url: str, out_path: Path, duration: float) -> Path:
    """Download a Pexels clip and crop/scale/trim to exact portrait dimensions."""
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


def _fallback_frame(i: int, out_dir: Path) -> Path:
    """Solid-color fallback frame if all else fails."""
    colors = [(20, 20, 60), (40, 10, 40), (10, 30, 50)]
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), colors[i % len(colors)])
    path = out_dir / f"broll_{i}.png"
    img.save(path)
    return path


def _generate_pollinations_image(prompt: str, out_path: Path) -> Path:
    """Generate a still image via Pollinations as fallback."""
    encoded = requests.utils.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={VIDEO_WIDTH}&height={VIDEO_HEIGHT}&nologo=true"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    img = Image.open(out_path)
    if img.size != (VIDEO_WIDTH, VIDEO_HEIGHT):
        img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
        img.save(out_path)
    return out_path


# ── Main entry point ─────────────────────────────────────────────────────

def generate_broll(prompts: list, out_dir: Path, clip_duration: float = 4.0,
                   topic_context: str = "", broll_subject: str = "") -> list[dict]:
    """Pool-first b-roll: search Pexels ONCE for the main subject, verify in
    batch, then match each prompt to the best clip from that pool. Falls back
    to individual search only when the pool is empty or exhausted.

    Returns a list of dicts: {"path": Path, "type": "video" | "image"}.
    """
    import time

    # Normalize prompts: accept both {"prompt": ..., "negative": ...} dicts
    # and plain strings (old drafts / graceful degrade).
    normalized = []
    for p in prompts:
        if isinstance(p, dict):
            normalized.append(str(p.get("prompt", "")).strip())
        else:
            normalized.append(str(p).strip())

    pexels_key = get_pexels_key()
    frames = []
    total = len(normalized)
    used_video_ids = set()
    subject = broll_subject.strip() if broll_subject else ""

    # Build verified pool up front.
    pool = []
    if pexels_key and subject:
        pool = _build_subject_pool(subject, pexels_key, topic_context)

    for i, prompt in enumerate(normalized):
        log(f"Sourcing b-roll {i+1}/{total}...")
        got_video = False

        # 1. Try pool (always, regardless of whether prompt mentions subject).
        if pool and not got_video:
            match = _best_pool_match(prompt, pool, used_video_ids)
            if match:
                video = match["video"]
                link = _best_file_link(video)
                if link:
                    vid_id = video.get("id")
                    if vid_id is not None:
                        used_video_ids.add(vid_id)
                    out_path = out_dir / f"broll_{i}.mp4"
                    _download_and_crop_video(link, out_path, clip_duration)
                    frames.append({"path": out_path, "type": "video"})
                    got_video = True
                    log(f"  Matched from pool: \"{prompt[:60]}\"")

        # 2. Individual Pexels search (pool empty or exhausted).
        if not got_video and pexels_key:
            try:
                url, vid_id = _search_pexels_video(
                    prompt.split(".")[0].strip(), pexels_key,
                    exclude_ids=used_video_ids, fallback_query=subject,
                    topic_context=topic_context,
                )
                if url:
                    if vid_id is not None:
                        used_video_ids.add(vid_id)
                    out_path = out_dir / f"broll_{i}.mp4"
                    _download_and_crop_video(url, out_path, clip_duration)
                    frames.append({"path": out_path, "type": "video"})
                    got_video = True
                    log(f"  Found via individual search: \"{prompt[:60]}\"")
            except Exception as e:
                log(f"  Individual search failed: {e}")

        # 3. Pollinations AI-generated still.
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
    """Ken Burns animation on a still frame (used for image fallbacks)."""
    fps = 30
    n = int(duration * fps)
    w, h = VIDEO_WIDTH, VIDEO_HEIGHT
    if effect == "zoom_in":
        vf = (f"scale={int(w*1.12)}:{int(h*1.12)},"
              f"zoompan=z='1.12-0.12*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
              f":d={n}:s={w}x{h}:fps={fps}")
    elif effect == "pan_right":
        vf = (f"scale={int(w*1.15)}:{int(h*1.15)},"
              f"zoompan=z=1.15:x='0.15*iw*on/{n}':y='ih*0.075'"
              f":d={n}:s={w}x{h}:fps={fps}")
    else:
        vf = (f"scale={int(w*1.12)}:{int(h*1.12)},"
              f"zoompan=z='1.0+0.12*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
              f":d={n}:s={w}x{h}:fps={fps}")
    run_cmd([
        "ffmpeg", "-loop", "1", "-i", str(img_path),
        "-vf", vf, "-t", str(duration), "-r", str(fps),
        "-pix_fmt", "yuv420p", str(out_path), "-y", "-loglevel", "quiet",
    ])