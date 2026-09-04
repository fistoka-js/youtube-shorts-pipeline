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

def _build_subject_pool(subject: str, api_key: str, topic_context: str = "",
                        orientation: str | None = None) -> list:
    """Search Pexels ONCE for the main subject, fetch 50 candidates,
    CLIP-verify each against food/toy/illustration negatives, and return
    confirmed-correct ones with pre-computed image embeddings."""
    import torch
    from io import BytesIO

    if not subject or not api_key:
        return []

    log(f"Building subject pool: searching Pexels for \"{subject}\" (50 candidates)...")
    params = {"query": subject, "per_page": 50, "size": "medium"}
    if orientation:
        params["orientation"] = orientation
    r = requests.get(
        "https://api.pexels.com/videos/search",
        params=params,
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


def _best_pool_match(prompt_text: str, pool: list, used_ids: set,
                     min_duration: float = 0.0) -> dict | None:
    """Pick the best-matching clip from the verified pool for a prompt,
    using CLIP text-to-image similarity. Excludes already-used clips.

    When min_duration is given, prefers clips that are actually long enough
    to cover the segment without looping - among candidates within the top
    similarity tier, picks the longest rather than picking on pure
    similarity alone (which frequently chose a great-match-but-too-short
    clip over a slightly-less-perfect-but-long-enough one).
    """
    import torch
    available = [p for p in pool if p["video"].get("id") not in used_ids]
    if not available:
        return None
    model, _, tokenizer = _load_clip()
    text_input = tokenizer([prompt_text])
    with torch.no_grad():
        text_feat = model.encode_text(text_input)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    scored = []
    for entry in available:
        score = (text_feat @ entry["image_embedding"].unsqueeze(1)).item()
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)

    if min_duration > 0:
        # Consider the top 40% of candidates by similarity (a reasonable
        # relevance floor), then among those, prefer the longest clip -
        # trimming down a longer clip looks far better than looping a
        # short one, and Pexels rarely returns an exact-length match.
        top_n = max(1, int(len(scored) * 0.4))
        top_candidates = scored[:top_n]
        best_entry = max(
            top_candidates,
            key=lambda x: _pexels_video_duration(x[1]["video"]),
        )[1]
        return best_entry

    return scored[0][1] if scored else None


def _pexels_video_duration(video: dict) -> float:
    """Pexels video metadata includes a top-level duration in seconds."""
    return float(video.get("duration", 0) or 0)

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

def _download_and_crop_video(url: str, out_path: Path, duration: float,
                              width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT) -> Path:
    """Download a Pexels clip and crop/scale/trim to the target dimensions
    (portrait for Shorts by default, landscape for long-form when width/
    height are passed explicitly)."""
    tmp_path = out_path.with_suffix(".raw.mp4")
    r = requests.get(url, timeout=90, stream=True)
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
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

def generate_broll_long_form(sections: list, out_dir: Path,
                             topic_context: str = "",
                             width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT,
                             orientation: str | None = "landscape",
                             words: list[dict] | None = None) -> list[dict]:
    """Pool-first b-roll for long-form videos with multiple sections, each
    potentially having a different visual subject (e.g. "octopus" for most
    sections, "medical research" for a science-context section).

    Builds ONE verified pool per unique subject across the whole video
    (not per section) to avoid redundant Pexels searches when consecutive
    sections share a subject, then matches each section's own prompts
    against its own subject's pool.

    Returns a list of dicts: {"path": Path, "type": "video" | "image",
    "section_id": str} — section_id lets assembly know which narration
    chunk each clip belongs to.
    """
    import time

    pexels_key = get_pexels_key()

    # Collect unique subjects across all sections, build one pool each.
    unique_subjects = []
    seen = set()
    for sec in sections:
        subj = (sec.get("broll_subject") or "").strip()
        if subj and subj.lower() not in seen:
            seen.add(subj.lower())
            unique_subjects.append(subj)

    pools_by_subject = {}
    if pexels_key:
        for subj in unique_subjects:
            pools_by_subject[subj.lower()] = _build_subject_pool(subj, pexels_key, topic_context, orientation)

    frames = []
    used_video_ids = set()  # global across the whole video, not per-pool

    # Map each section to its real speech span using Whisper word timestamps:
    # walk through the concatenated narration in the same order sections
    # were joined (cmd_produce_long_form joins them with "\n\n"), matching
    # word counts to timestamp indices. This gives each section's ACTUAL
    # start/end time in the final audio, not an estimate from word count.
    section_spans = {}  # sec_id -> (start_time, end_time)
    if words:
        word_idx = 0
        for sec in sections:
            sec_word_count = len(sec.get("narration", "").split())
            start_idx = min(word_idx, len(words) - 1)
            end_idx = min(word_idx + sec_word_count - 1, len(words) - 1)
            if sec_word_count > 0 and start_idx < len(words):
                start_time = words[start_idx]["start"]
                end_time = words[end_idx]["end"] if end_idx < len(words) else words[-1]["end"]
                section_spans[sec.get("id", "")] = (start_time, end_time)
            word_idx += sec_word_count

    for sec in sections:
        sec_id = sec.get("id", "")
        subject = (sec.get("broll_subject") or "").strip()
        prompts = sec.get("broll_prompts", [])
        pool = pools_by_subject.get(subject.lower(), [])

        # Real per-clip duration: this section's ACTUAL speech duration
        # (from Whisper timestamps) divided across its own prompt count.
        # Falls back to a flat estimate only if word timestamps weren't
        # available (e.g. captions stage was skipped for some reason).
        if sec_id in section_spans and prompts:
            span_start, span_end = section_spans[sec_id]
            sec_seconds = max(span_end - span_start, 1.0)
            clip_duration = max(4.0, min(25.0, sec_seconds / len(prompts)))
        else:
            clip_duration = 8.0  # sane fallback if word timing is missing

        for i, prompt in enumerate(prompts):
            prompt = str(prompt).strip()
            log(f"Sourcing b-roll for {sec_id} ({i+1}/{len(prompts)})...")
            got_video = False

            if pool and not got_video:
                match = _best_pool_match(prompt, pool, used_video_ids, min_duration=clip_duration)
                if match:
                    video = match["video"]
                    link = _best_file_link(video)
                    if link:
                        vid_id = video.get("id")
                        if vid_id is not None:
                            used_video_ids.add(vid_id)
                        out_path = out_dir / f"broll_{sec_id}_{i}.mp4"
                        _download_and_crop_video(link, out_path, clip_duration, width, height)
                        frames.append({"path": out_path, "type": "video", "section_id": sec_id})
                        got_video = True
                        log(f"  Matched from pool: \"{prompt[:60]}\"")

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
                        out_path = out_dir / f"broll_{sec_id}_{i}.mp4"
                        _download_and_crop_video(url, out_path, clip_duration, width, height)
                        frames.append({"path": out_path, "type": "video", "section_id": sec_id})
                        got_video = True
                        log(f"  Found via individual search: \"{prompt[:60]}\"")
                except Exception as e:
                    log(f"  Individual search failed: {e}")

            if not got_video:
                try:
                    out_path = out_dir / f"broll_{sec_id}_{i}.png"
                    _generate_pollinations_image(prompt, out_path)
                    frames.append({"path": out_path, "type": "image", "section_id": sec_id})
                    log(f"  No stock match \u2014 generated fallback image")
                    time.sleep(1.0)
                except Exception as e:
                    log(f"  Fallback image failed: {e} \u2014 using solid color")
                    frames.append({"path": _fallback_frame(len(frames), out_dir), "type": "image", "section_id": sec_id})

    return frames

def animate_frame(img_path: Path, out_path: Path, duration: float, effect: str = "zoom_in",
                  width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT):
    """Ken Burns animation on a still frame (used for image fallbacks)."""
    fps = 30
    n = int(duration * fps)
    w, h = width, height
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