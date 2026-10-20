"""Pexels stock video b-roll (pool-first) + Pollinations image fallback + Ken Burns."""

from pathlib import Path
import json
import math
import re
import tempfile

import requests
from PIL import Image

from . import build_report
from .config import VIDEO_WIDTH, VIDEO_HEIGHT, run_cmd
from .log import log
from .retry import with_retry
from .section_timing import compute_section_timings

# Shorts only: a portrait clip wins over the best landscape clip when its CLIP
# match score is within this margin of it (portrait clips lose less to the 9:16
# crop). 0.02 is a small guess, not measured on real Pexels clips: large enough
# to decide near-ties, small enough that a clearly weaker portrait clip still loses.
PORTRAIT_PREFERENCE_MARGIN = 0.02

# Shorts only: when cropping a landscape clip to 9:16, a left or right window
# replaces the centre window only if its averaged CLIP score beats the centre
# by at least this much. 0.01 is a small guess: smaller gaps are within the
# noise of scoring only 3 frames, so the centre (the old behaviour) is kept.
CROP_MIN_GAIN = 0.01

# Shorts: a downloaded clip keeps its own length, but never more than this many
# seconds. The cap stops very long Pexels clips from being cut in ways that look odd.
MAX_CLIP_SECONDS = 10.0

# Shorts: the clip length used only when no usable length is known at all (the
# old fixed trim, which every Shorts clip used to get).
FIXED_CLIP_SECONDS = 4.0

# Shorts: a longer clip is preferred over the best-scoring clip when its score
# (with the portrait bonus) is within this margin of the best. 0.02 is the same
# scale as PORTRAIT_PREFERENCE_MARGIN, so a clearly better short clip still wins.
DURATION_PREFERENCE_MARGIN = 0.02

# Shorts: speaking rate used to estimate how long each b-roll slot will be, before
# the voiceover exists (b-roll runs first in cmd_produce). Measured on 16 existing
# Edge TTS voiceovers at the +12% rate: pooled 181.6 words per minute, median 179.8,
# spread 151-221. The old guess of 168 did not match, so 180 is used. It only
# steers which clip is picked, so a rough value is enough.
SPEECH_WORDS_PER_MINUTE = 180


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


def _is_portrait(video: dict) -> bool:
    """True when the Pexels video's downloaded file is strictly taller than it
    is wide. Uses the same file entry as the download, and the same strict
    height > width rule that _best_file_entry uses to prefer portrait files.
    Square files do NOT count: a square clip loses width in the 9:16 crop, so
    it gets no portrait bonus."""
    entry = _best_file_entry(video) or {}
    width = _pexels_size(entry.get("width")) or _pexels_size(video.get("width"))
    height = _pexels_size(entry.get("height")) or _pexels_size(video.get("height"))
    if not width or not height:
        return False
    return height > width


def _duration_aware_pick(scored: list, prefer_portrait: bool,
                        est_slot_seconds: float) -> tuple:
    """Shorts only. scored is (raw CLIP score, pool entry), best first by the
    effective score (raw score plus the portrait bonus, if on). Among the
    candidates within DURATION_PREFERENCE_MARGIN of the best effective score,
    returns the best-scoring one whose Pexels duration covers est_slot_seconds,
    capped at MAX_CLIP_SECONDS (a saved clip is never longer than that).
    If none does, returns the original best pick. Nothing is downloaded: the
    duration is Pexels' own metadata."""
    def effective(item):
        bonus = (PORTRAIT_PREFERENCE_MARGIN
                 if prefer_portrait and _is_portrait(item[1]["video"]) else 0.0)
        return item[0] + bonus

    # A saved clip is never longer than MAX_CLIP_SECONDS, so an estimate above
    # the cap can only ever be met by a clip of the cap length.
    needed = min(est_slot_seconds, MAX_CLIP_SECONDS)
    best = effective(scored[0])
    for item in scored:
        if effective(item) < best - DURATION_PREFERENCE_MARGIN:
            break
        if _pexels_video_duration(item[1]["video"]) >= needed:
            return item
    return scored[0]


def _best_pool_match(prompt_text: str, pool: list, used_ids: set,
                     min_duration: float = 0.0,
                     score_out: dict | None = None,
                     prefer_portrait: bool = False,
                     est_slot_seconds: float | None = None) -> dict | None:
    """Pick the best-matching clip from the verified pool for a prompt,
    using CLIP text-to-image similarity. Excludes already-used clips.

    When min_duration is given, prefers clips that are actually long enough
    to cover the segment without looping - among candidates within the top
    similarity tier, picks the longest rather than picking on pure
    similarity alone (which frequently chose a great-match-but-too-short
    clip over a slightly-less-perfect-but-long-enough one).

    score_out: optional dict. When given and a clip is picked, the CLIP score
    of that clip is stored in it under "score" (for the build report only).
    The choice itself is identical with or without it.

    prefer_portrait: optional, default off (long-form does not use it). When
    on (Shorts), portrait clips get a bonus of PORTRAIT_PREFERENCE_MARGIN when
    ranking, so a portrait clip wins over the best landscape clip if it is
    within that margin. If no portrait clip is that close, the best overall
    clip still wins. score_out still holds the true CLIP score, not the bonus.

    est_slot_seconds: optional, default None (old behaviour). Shorts only: the
    estimated slot length. Among clips within DURATION_PREFERENCE_MARGIN of the
    best effective score, a clip whose Pexels duration covers the slot is
    preferred (see _duration_aware_pick). Ignored when min_duration is given,
    so long-form is never affected.
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

    if prefer_portrait:
        scored.sort(
            key=lambda x: x[0] + (PORTRAIT_PREFERENCE_MARGIN
                                  if _is_portrait(x[1]["video"]) else 0.0),
            reverse=True,
        )

    if min_duration > 0:
        # Consider the top 40% of candidates by similarity (a reasonable
        # relevance floor), then among those, prefer the longest clip -
        # trimming down a longer clip looks far better than looping a
        # short one, and Pexels rarely returns an exact-length match.
        top_n = max(1, int(len(scored) * 0.4))
        top_candidates = scored[:top_n]
        best_score, best_entry = max(
            top_candidates,
            key=lambda x: _pexels_video_duration(x[1]["video"]),
        )
        if score_out is not None:
            score_out["score"] = float(best_score)
        return best_entry

    if not scored:
        return None
    if est_slot_seconds is not None:
        best_score, best_entry = _duration_aware_pick(scored, prefer_portrait,
                                                      est_slot_seconds)
    else:
        best_score, best_entry = scored[0]
    if score_out is not None:
        score_out["score"] = float(best_score)
    return best_entry


def _pexels_video_duration(video: dict) -> float:
    """Pexels video metadata includes a top-level duration in seconds."""
    return float(video.get("duration", 0) or 0)


def _row_clip_duration(clip_duration, clip_info: dict):
    """The clip_duration to show in the report row: the explicit trim when one
    was given, otherwise the real saved length (None when unknown)."""
    if clip_duration is not None:
        return clip_duration
    return clip_info.get("saved_clip_seconds")


def estimate_slot_seconds(script: str, n_clips: int) -> float | None:
    """Estimated length of each b-roll slot, from the script's word count.

    The voiceover is not made yet when b-roll runs, so the length is estimated:
    total seconds = words / (SPEECH_WORDS_PER_MINUTE / 60), split evenly over
    the clips (the same flat average assembly uses without word timing).
    Returns None when there is no script text or no clips."""
    try:
        words = len(str(script or "").split())
    except Exception:
        return None
    if words == 0 or n_clips <= 0:
        return None
    total_seconds = words / (SPEECH_WORDS_PER_MINUTE / 60.0)
    return total_seconds / n_clips

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


def _pexels_size(value) -> int:
    """A Pexels width or height as an int. Missing, None, empty, non-numeric or
    infinite values count as 0 (unknown), so one bad entry cannot crash the comparison."""
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _best_file_entry(video: dict) -> dict | None:
    """The video_files entry that _best_file_link picks (highest height,
    portrait files preferred). Used by the build report to read the size of
    the file that is actually downloaded."""
    files = video.get("video_files", [])
    portrait = [f for f in files
                if _pexels_size(f.get("height")) > _pexels_size(f.get("width"))]
    candidates = portrait or files
    if not candidates:
        return None
    return max(candidates, key=lambda f: _pexels_size(f.get("height")))


def _best_file_link(video: dict) -> str | None:
    """Pick the highest-res file link from a Pexels video, preferring portrait."""
    entry = _best_file_entry(video)
    if entry is None:
        return None
    return entry.get("link")


def _search_pexels_video(
    query: str, api_key: str,
    exclude_ids: set | None = None,
    fallback_query: str = "",
    topic_context: str = "",
    picked_out: dict | None = None,
) -> tuple[str | None, int | None]:
    """Individual Pexels search with fallback — used only when the pool is
    empty or exhausted. Returns (MP4 URL, video id) or (None, None).

    picked_out: optional dict. When given and a clip is chosen, the Pexels video
    dict is stored in it under "video" (for the build report only)."""
    exclude_ids = exclude_ids or set()
    videos = _fetch_pexels_candidates(query, api_key)
    available = [v for v in videos if v.get("id") not in exclude_ids]

    for candidate in available[:5]:
        link = _best_file_link(candidate)
        if link:
            if picked_out is not None:
                picked_out["video"] = candidate
            return link, candidate.get("id")

    if fallback_query and fallback_query.lower() != query.lower():
        log(f"  No match for \"{query}\" — trying fallback \"{fallback_query}\"")
        fb = _fetch_pexels_candidates(fallback_query, api_key)
        for candidate in [v for v in fb if v.get("id") not in exclude_ids][:5]:
            link = _best_file_link(candidate)
            if link:
                if picked_out is not None:
                    picked_out["video"] = candidate
                return link, candidate.get("id")

    return None, None


# ── Download / fallback helpers ──────────────────────────────────────────

def _horizontal_slack(src_w, src_h, out_w, out_h) -> bool:
    """True when fitting the source to the output height cuts away width, i.e.
    the source is wider than the output shape. Sources without this (e.g. 9:16
    portrait) lose nothing sideways, so no crop choice is made for them."""
    return src_w * out_h > src_h * out_w


def _scaled_window_lefts(scaled_w: int, out_w: int) -> dict:
    """Left edge of the three candidate windows (left, center, right) on the frame
    as ffmpeg scales it to the output height (scaled_w pixels wide). Each window
    is out_w wide: the same region that scale-then-crop keeps. The right window
    is exactly flush with the right edge."""
    right = scaled_w - out_w
    return {"left": 0, "center": right // 2, "right": right}


def _side_crop_x(left: int) -> int:
    """Crop x for a side window, made even (keeps chroma aligned for yuv420p).
    Left is never past the valid range, because it comes from _scaled_window_lefts."""
    return left - (left % 2)


def _score_crop_windows(frames: list, lefts: dict, out_w: int, out_h: int,
                        prompt: str) -> dict:
    """CLIP score of each window against the prompt, averaged over the frames.
    The frames are already scaled to the output height, so each window is
    out_w wide and the full frame tall. Same normalization as _best_pool_match
    (text and image vectors unit length)."""
    import torch
    model, preprocess, tokenizer = _load_clip()
    with torch.no_grad():
        text_feat = model.encode_text(tokenizer([prompt]))
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    sides = list(lefts)
    totals = {s: 0.0 for s in sides}
    for frame in frames:
        batch = torch.stack([
            preprocess(frame.crop((lefts[s], 0, lefts[s] + out_w, out_h)))
            for s in sides
        ])
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
            sims = (feats @ text_feat.T).squeeze(1).tolist()
        for s, sim in zip(sides, sims):
            totals[s] += float(sim)
    return {s: totals[s] / len(frames) for s in sides}


def _choose_crop(frames: list, out_w: int, out_h: int, prompt: str) -> dict:
    """Pick the crop window. The centre is kept unless the best side window beats
    it by at least CROP_MIN_GAIN. frames are scaled to the output height (see
    _decide_crop). Returns the side, the crop x (None = centre, so the old
    command is used), the offset fraction and both scores."""
    scaled_w, scaled_h = frames[0].size
    if scaled_h != out_h or scaled_w <= out_w:
        raise ValueError(f"unexpected scaled frame size {scaled_w}x{scaled_h}")
    lefts = _scaled_window_lefts(scaled_w, out_w)
    scores = _score_crop_windows(frames, lefts, out_w, out_h, prompt)
    center_score = scores["center"]
    best = max(scores, key=lambda s: scores[s])
    side = "center"
    if best != "center" and scores[best] - center_score >= CROP_MIN_GAIN:
        side = best
    x = None if side == "center" else _side_crop_x(lefts[side])
    return {
        "side": side,
        "x": x,
        "crop_offset": {"left": 0.0, "center": 0.5, "right": 1.0}[side],
        "center_score": center_score,
        "chosen_score": scores[side],
    }


def _decide_crop(raw_path: Path, duration: float, out_w: int, out_h: int,
                 prompt: str) -> dict | None:
    """Full crop decision for one raw download. Returns the _choose_crop result,
    or None when no decision is made (no horizontal slack, or any failure).
    Failures are logged as warnings and never raised: the caller then uses the
    plain centre crop. Temp frames live in a temp folder that is always removed."""
    try:
        probe = json.loads(run_cmd([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(raw_path),
        ], capture=True).stdout)
        src_w = int(probe["streams"][0]["width"])
        src_h = int(probe["streams"][0]["height"])
        native = float(probe["format"]["duration"])
        if not _horizontal_slack(src_w, src_h, out_w, out_h):
            return None
        # Kept length = the part the -t duration keeps, or the whole clip if shorter.
        kept = min(float(duration), native)
        # The frames use the same scale filter as the final command, so their width
        # is exactly the width ffmpeg will scale the clip to.
        scale_vf = f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
        with tempfile.TemporaryDirectory(prefix="crop_frames_", dir=raw_path.parent) as work:
            frames = []
            for k, frac in enumerate((0.25, 0.5, 0.75)):
                frame_path = Path(work) / f"frame_{k}.png"
                run_cmd([
                    "ffmpeg", "-ss", f"{kept * frac:.3f}", "-i", str(raw_path),
                    "-vf", scale_vf, "-frames:v", "1", "-y", "-loglevel", "quiet",
                    str(frame_path),
                ])
                with Image.open(frame_path) as im:
                    frames.append(im.convert("RGB"))
            return _choose_crop(frames, out_w, out_h, prompt)
    except Exception as e:
        # The fallback is a safety net for any error that carries no text, so name the type too.
        reason = str(e) or "no details given"
        log(f"WARNING: crop check skipped ({type(e).__name__}: {reason}), using the centre crop")
        return None


def _probe_duration(path: Path) -> float | None:
    """Length of a media file in seconds, read locally with ffprobe (no network).
    None when it cannot be read or is not a usable positive number."""
    try:
        r = run_cmd(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(path)], capture=True)
        value = float(r.stdout.strip())
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _download_and_crop_video(url: str, out_path: Path, duration: float | None,
                              width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT,
                              crop_prompt: str | None = None,
                              crop_out: dict | None = None,
                              fallback_duration: float | None = None,
                              clip_out: dict | None = None) -> Path:
    """Download a Pexels clip and crop/scale/trim to the target dimensions
    (portrait for Shorts by default, landscape for long-form when width/
    height are passed explicitly).

    duration: the length to keep in seconds (old fixed trim, used by long-form
    and by explicit clip_duration calls). None (Shorts default) keeps the clip's
    own length, capped at MAX_CLIP_SECONDS. The native length is probed from the
    raw download; if that fails, fallback_duration (Pexels' metadata) is used;
    if neither is usable, FIXED_CLIP_SECONDS is used, as before.

    crop_prompt: optional, default off (long-form does not use it). When given
    (Shorts), a landscape source is cropped at the left, centre or right window
    that best matches crop_prompt (see _decide_crop). Otherwise, or when the
    centre wins, the command is the plain centre crop as before.
    crop_out: optional dict. When a crop decision is made, it is filled with
    crop_offset, crop_side, center_score and chosen_score (for the build report).
    fallback_duration: Pexels' own duration, used only when probing fails.
    clip_out: optional dict, filled with saved_clip_seconds (the length of the
    file that was written, probed locally; the kept length if probing fails)."""
    tmp_path = out_path.with_suffix(".raw.mp4")
    try:
        r = requests.get(url, timeout=90, stream=True)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        if duration is None:
            native = _probe_duration(tmp_path)
            if native is None:
                native = _usable_or_none(fallback_duration)
                log(f"  Could not read the clip length from the file; "
                    f"using {'Pexels' if native is not None else 'the fixed'} length")
            kept = (min(native, MAX_CLIP_SECONDS) if native is not None
                    else FIXED_CLIP_SECONDS)
            t_arg = str(kept)
        else:
            kept = float(duration)
            t_arg = str(duration)
        crop_x = None
        if crop_prompt is not None:
            decision = _decide_crop(tmp_path, kept, width, height, crop_prompt)
            if decision is not None:
                crop_x = decision["x"]
                if crop_out is not None:
                    crop_out.update(
                        crop_offset=decision["crop_offset"],
                        crop_side=decision["side"],
                        center_score=decision["center_score"],
                        chosen_score=decision["chosen_score"],
                    )
        crop_expr = (f"crop={width}:{height}" if crop_x is None
                     else f"crop={width}:{height}:{crop_x}:0")
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"{crop_expr}"
        )
        run_cmd([
            "ffmpeg", "-i", str(tmp_path), "-t", t_arg,
            "-vf", vf, "-an", "-r", "30", "-pix_fmt", "yuv420p",
            str(out_path), "-y", "-loglevel", "quiet",
        ])
        if clip_out is not None:
            saved = _probe_duration(out_path)
            clip_out["saved_clip_seconds"] = saved if saved is not None else kept
    finally:
        # The raw download is only a temp file. Remove it whether the crop worked or failed.
        tmp_path.unlink(missing_ok=True)
    return out_path


def _usable_or_none(value) -> float | None:
    """value as a float when it is a finite positive number, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


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


def _record_clip(report_rows, *, slot, section, kind, source, out_w, out_h,
                 video=None, clip_duration=None, clip_score=None,
                 crop=None, saved_clip_seconds=None,
                 est_slot_seconds=None) -> None:
    """Append one build-report row (see build_report.py). Does nothing when
    report_rows is None, and never raises: the report must not change a build.
    crop: optional dict from _download_and_crop_video's crop_out (crop fields).
    saved_clip_seconds: the real length of the saved clip file, when known.
    est_slot_seconds: the slot length estimate used for selection, when given."""
    if report_rows is None:
        return
    try:
        entry = _best_file_entry(video) if video is not None else None
        row = build_report.make_clip_row(
            slot=slot, section=section, kind=kind, source=source,
            out_w=out_w, out_h=out_h, video=video, file_entry=entry,
            clip_duration=clip_duration, clip_score=clip_score,
            crop=crop, saved_clip_seconds=saved_clip_seconds,
            est_slot_seconds=est_slot_seconds,
        )
    except Exception as e:
        log(f"  Build report could not measure slot {slot}: {e}")
        row = build_report.empty_row(slot, section, kind, source)
    report_rows.append(row)


# ── Main entry point ─────────────────────────────────────────────────────

def generate_broll(prompts: list, out_dir: Path, clip_duration: float | None = None,
                   topic_context: str = "", broll_subject: str = "",
                   report_rows: list | None = None,
                   est_slot_seconds: float | None = None) -> list[dict]:
    """Pool-first b-roll: search Pexels ONCE for the main subject, verify in
    batch, then match each prompt to the best clip from that pool. Falls back
    to individual search only when the pool is empty or exhausted.

    Returns a list of dicts: {"path": Path, "type": "video" | "image"}.

    clip_duration: None (default) keeps each clip's own length, capped at
    MAX_CLIP_SECONDS. An explicit number keeps the old fixed trim to that length.

    est_slot_seconds: optional, default None (old selection). The estimated
    slot length, used to prefer pool clips long enough for their slot.

    report_rows: optional list. When given, one build-report row per slot is
    appended to it (in slot order). It does not change the clips chosen.
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
            score_out = {}
            match = _best_pool_match(prompt, pool, used_video_ids, score_out=score_out,
                                     prefer_portrait=True,
                                     est_slot_seconds=est_slot_seconds)
            if match:
                video = match["video"]
                link = _best_file_link(video)
                if link:
                    vid_id = video.get("id")
                    if vid_id is not None:
                        used_video_ids.add(vid_id)
                    out_path = out_dir / f"broll_{i}.mp4"
                    crop_info = {}
                    clip_info = {}
                    _download_and_crop_video(link, out_path, clip_duration,
                                             crop_prompt=prompt, crop_out=crop_info,
                                             fallback_duration=_pexels_video_duration(video),
                                             clip_out=clip_info)
                    frames.append({"path": out_path, "type": "video"})
                    _record_clip(report_rows, slot=i + 1, section=None, kind="video",
                                 source="pool", out_w=VIDEO_WIDTH, out_h=VIDEO_HEIGHT,
                                 video=video,
                                 clip_duration=_row_clip_duration(clip_duration, clip_info),
                                 clip_score=score_out.get("score"), crop=crop_info,
                                 saved_clip_seconds=clip_info.get("saved_clip_seconds"),
                                 est_slot_seconds=est_slot_seconds)
                    got_video = True
                    log(f"  Matched from pool: \"{prompt[:60]}\"")

        # 2. Individual Pexels search (pool empty or exhausted).
        if not got_video and pexels_key:
            try:
                picked = {}
                url, vid_id = _search_pexels_video(
                    prompt.split(".")[0].strip(), pexels_key,
                    exclude_ids=used_video_ids, fallback_query=subject,
                    topic_context=topic_context, picked_out=picked,
                )
                if url:
                    if vid_id is not None:
                        used_video_ids.add(vid_id)
                    out_path = out_dir / f"broll_{i}.mp4"
                    crop_info = {}
                    clip_info = {}
                    picked_video = picked.get("video")
                    _download_and_crop_video(
                        url, out_path, clip_duration,
                        crop_prompt=prompt, crop_out=crop_info,
                        fallback_duration=(_pexels_video_duration(picked_video)
                                           if picked_video else None),
                        clip_out=clip_info)
                    frames.append({"path": out_path, "type": "video"})
                    _record_clip(report_rows, slot=i + 1, section=None, kind="video",
                                 source="last_resort_search", out_w=VIDEO_WIDTH,
                                 out_h=VIDEO_HEIGHT, video=picked_video,
                                 clip_duration=_row_clip_duration(clip_duration, clip_info),
                                 crop=crop_info,
                                 saved_clip_seconds=clip_info.get("saved_clip_seconds"))
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
                _record_clip(report_rows, slot=i + 1, section=None, kind="image",
                             source="image_fallback", out_w=VIDEO_WIDTH, out_h=VIDEO_HEIGHT)
                log(f"  No stock match — generated fallback image")
                time.sleep(1.0)
            except Exception as e:
                log(f"  Fallback image failed: {e} — using solid color")
                frames.append({"path": _fallback_frame(i, out_dir), "type": "image"})
                _record_clip(report_rows, slot=i + 1, section=None, kind="image",
                             source="solid_color", out_w=VIDEO_WIDTH, out_h=VIDEO_HEIGHT)

    return frames

def generate_broll_long_form(sections: list, out_dir: Path,
                             topic_context: str = "",
                             width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT,
                             orientation: str | None = "landscape",
                             words: list[dict] | None = None,
                             report_rows: list | None = None) -> list[dict]:
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

    report_rows: optional list. When given, one build-report row per slot is
    appended to it (slot = position in the returned list, 1-based). It does
    not change the clips chosen or their timing.
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

    # Map each section to its real speech span using Whisper word timestamps
    # (shared helper: finds each section's opening words in the audio, so the
    # span is the ACTUAL start/end time in the final audio, not a word-count
    # estimate).
    section_spans = {}  # sec_id -> (start_time, end_time)
    if words:
        timings = compute_section_timings(sections, words)
        for sec, t in zip(sections, timings):
            if len(sec.get("narration", "").split()) > 0:
                section_spans[sec.get("id", "")] = (t["start"], t["end"])

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
                score_out = {}
                match = _best_pool_match(prompt, pool, used_video_ids,
                                         min_duration=clip_duration, score_out=score_out)
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
                        _record_clip(report_rows, slot=len(frames), section=sec_id, kind="video",
                                     source="pool", out_w=width, out_h=height, video=video,
                                     clip_duration=clip_duration,
                                     clip_score=score_out.get("score"))
                        got_video = True
                        log(f"  Matched from pool: \"{prompt[:60]}\"")

            if not got_video and pexels_key:
                try:
                    picked = {}
                    url, vid_id = _search_pexels_video(
                        prompt.split(".")[0].strip(), pexels_key,
                        exclude_ids=used_video_ids, fallback_query=subject,
                        topic_context=topic_context, picked_out=picked,
                    )
                    if url:
                        if vid_id is not None:
                            used_video_ids.add(vid_id)
                        out_path = out_dir / f"broll_{sec_id}_{i}.mp4"
                        _download_and_crop_video(url, out_path, clip_duration, width, height)
                        frames.append({"path": out_path, "type": "video", "section_id": sec_id})
                        _record_clip(report_rows, slot=len(frames), section=sec_id, kind="video",
                                     source="last_resort_search", out_w=width, out_h=height,
                                     video=picked.get("video"), clip_duration=clip_duration)
                        got_video = True
                        log(f"  Found via individual search: \"{prompt[:60]}\"")
                except Exception as e:
                    log(f"  Individual search failed: {e}")

            if not got_video:
                try:
                    out_path = out_dir / f"broll_{sec_id}_{i}.png"
                    _generate_pollinations_image(prompt, out_path)
                    frames.append({"path": out_path, "type": "image", "section_id": sec_id})
                    _record_clip(report_rows, slot=len(frames), section=sec_id, kind="image",
                                 source="image_fallback", out_w=width, out_h=height)
                    log(f"  No stock match \u2014 generated fallback image")
                    time.sleep(1.0)
                except Exception as e:
                    log(f"  Fallback image failed: {e} \u2014 using solid color")
                    frames.append({"path": _fallback_frame(len(frames), out_dir), "type": "image", "section_id": sec_id})
                    _record_clip(report_rows, slot=len(frames), section=sec_id, kind="image",
                                 source="solid_color", out_w=width, out_h=height)

    return frames

def animate_frame(img_path: Path, out_path: Path, duration: float, effect: str = "zoom_in",
                  width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT,
                  frames: int | None = None):
    """Ken Burns animation on a still frame (used for image fallbacks).

    frames: optional exact output frame count. When given (long-form), the
    clip is exactly that many frames at 30 fps; when omitted, the length is
    duration * 30 frames as before (Shorts path)."""
    fps = 30
    n = frames if frames is not None else int(duration * fps)
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
    length_args = ["-frames:v", str(frames)] if frames is not None else ["-t", str(duration)]
    run_cmd([
        "ffmpeg", "-loop", "1", "-i", str(img_path),
        "-vf", vf, *length_args, "-r", str(fps),
        "-pix_fmt", "yuv420p", str(out_path), "-y", "-loglevel", "quiet",
    ])