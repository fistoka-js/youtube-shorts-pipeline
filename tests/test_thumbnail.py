"""Tests for the thumbnail photo choice (verticals/thumbnail.py).

All data is made up. Pexels HTTP (requests.get) and the CLIP scorer are faked,
so nothing touches the network or the real CLIP model. Run with:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_thumbnail.py -v
"""

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from verticals import thumbnail as T

SEARCH_URL = "https://api.pexels.com/v1/search"


def _png(red: int, size=(320, 180)) -> bytes:
    """A made-up image whose top-left red value is `red`. The fake scorer reads
    that value back, so it tells the test which photo it is looking at."""
    buf = BytesIO()
    Image.new("RGB", size, (red, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakePexels:
    """Stands in for requests.get. photos_by_query maps a search query to its
    made-up photo list. scores maps photo id -> the CLIP score the fake gives it.
    broken_ids are photos whose small preview fails to download."""

    def __init__(self, photos_by_query, scores, broken_ids=(), broken_full_ids=(),
                 html_full_ids=()):
        self.photos_by_query = photos_by_query
        self.scores = scores
        self.broken_ids = set(broken_ids)
        self.broken_full_ids = set(broken_full_ids)
        self.html_full_ids = set(html_full_ids)
        self.search_calls = []
        self.downloads = []

    def __call__(self, url, params=None, headers=None, timeout=None, **kwargs):
        if url == SEARCH_URL:
            self.search_calls.append(dict(params))
            photos = self.photos_by_query.get(params["query"], [])
            return _Resp(200, {"photos": photos})
        self.downloads.append(url)
        if "/medium/" in url:
            pid = int(url.rsplit("/", 1)[1].split(".")[0])
            if pid in self.broken_ids:
                return _Resp(500)
            return _Resp(200, content=_png(pid))
        if "/full/" in url:
            pid = int(url.rsplit("/", 1)[1].split(".")[0])
            if pid in self.broken_full_ids:
                return _Resp(500)
            if pid in self.html_full_ids:
                # A 200 reply that is an error page, not an image.
                return _Resp(200, content=b"<html><body>Rate limited</body></html>")
            return _Resp(200, content=_png(pid, size=(640, 360)))
        return _Resp(404)


def _photo(pid: int) -> dict:
    return {
        "id": pid,
        "url": f"https://www.pexels.com/photo/made-up-photo-{pid}/",
        "src": {
            "medium": f"https://img.test/medium/{pid}.jpg",
            "large2x": f"https://img.test/full/{pid}.png",
        },
    }


def _photos(ids):
    return [_photo(i) for i in ids]


@pytest.fixture
def logs(monkeypatch):
    lines = []
    monkeypatch.setattr(T, "log", lambda msg: lines.append(msg))
    return lines


@pytest.fixture
def fake_clip(monkeypatch):
    """CLIP is 'loaded' (a dummy tuple) and the per-image score is looked up
    from the scores given to _install, by the photo id stored in the image.
    Every text the code scores against is recorded in state.texts."""
    monkeypatch.setenv("PEXELS_API_KEY", "made-up-key")
    monkeypatch.setattr(T._broll, "_load_clip", lambda: ("m", "p", "t"))
    state = SimpleNamespace(scores={}, texts=[])

    # The "text feature" is the text itself, so tests can see what was scored against.
    monkeypatch.setattr(T, "_clip_text_feature", lambda text, clip: text)

    def score(img, text_feat, clip):
        state.texts.append(text_feat)
        return state.scores[img.getpixel((0, 0))[0]]

    monkeypatch.setattr(T, "_clip_score_image", score)
    return state


def _install(monkeypatch, fake_clip, photos_by_query, scores, broken_ids=(),
             broken_full_ids=(), html_full_ids=()):
    fake = FakePexels(photos_by_query, scores, broken_ids, broken_full_ids, html_full_ids)
    fake_clip.scores = scores
    monkeypatch.setattr(T.requests, "get", fake)
    return fake


def _run(draft, tmp_path):
    details = {}
    path = T.generate_thumbnail(draft, tmp_path, details=details)
    return path, details


def _is_colour_frame(path):
    """True only when the output is the plain colour frame (20, 20, 60) at 1280x720."""
    img = Image.open(path).convert("RGB")
    return img.size == (T.THUMB_WIDTH, T.THUMB_HEIGHT) and img.getpixel((640, 360)) == (20, 20, 60)


# ── Search words ─────────────────────────────────────────────────────────────

def test_subject_is_used_when_present(monkeypatch, fake_clip, logs, tmp_path):
    fake = _install(monkeypatch, fake_clip,
                    {"onion": _photos([1])}, {1: 0.35})
    draft = {"job_id": "j1", "broll_subject": "onion",
             "thumbnail_prompt": "A dramatic close-up of chickpeas on a plate"}
    _run(draft, tmp_path)
    assert fake.search_calls[0]["query"] == "onion"
    assert fake_clip.texts[0] == "a photo of onion"


@pytest.mark.parametrize("subject", [None, "", "   "])
def test_falls_back_to_cleaned_prompt_when_no_subject(monkeypatch, fake_clip, logs,
                                                      tmp_path, subject):
    fake = _install(monkeypatch, fake_clip,
                    {"chickpeas on a plate": _photos([2])}, {2: 0.35})
    draft = {"job_id": "j2", "broll_subject": subject,
             "thumbnail_prompt": "chickpeas on a plate. Warm light"}
    _run(draft, tmp_path)
    assert fake.search_calls[0]["query"] == "chickpeas on a plate"
    assert fake_clip.texts[0] == "a photo of chickpeas on a plate"
    # Same query, so there is no second search.
    assert len(fake.search_calls) == 1


def test_cat_is_the_last_resort(monkeypatch, fake_clip, logs, tmp_path):
    fake = _install(monkeypatch, fake_clip, {"cat": _photos([3])}, {3: 0.35})
    draft = {"job_id": "j3", "broll_subject": "", "thumbnail_prompt": "."}
    _run(draft, tmp_path)
    assert fake.search_calls[0]["query"] == "cat"


def test_cat_used_when_subject_and_prompt_are_missing(monkeypatch, fake_clip, logs, tmp_path):
    fake = _install(monkeypatch, fake_clip, {"cat": _photos([3])}, {3: 0.35})
    _run({"job_id": "j3b"}, tmp_path)
    assert fake.search_calls[0]["query"] == "cat"


# ── Pexels request ───────────────────────────────────────────────────────────

def test_requests_landscape_photos_with_15_candidates(monkeypatch, fake_clip, logs, tmp_path):
    fake = _install(monkeypatch, fake_clip, {"onion": _photos([1])}, {1: 0.35})
    _run({"job_id": "j4", "broll_subject": "onion"}, tmp_path)
    call = fake.search_calls[0]
    assert call["orientation"] == "landscape"
    assert call["per_page"] == 15


# ── Choice ───────────────────────────────────────────────────────────────────

def test_best_scoring_candidate_is_chosen_not_first(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.20, 2: 0.31, 3: 0.28}
    fake = _install(monkeypatch, fake_clip, {"onion": _photos([1, 2, 3])}, scores)
    path, details = _run({"job_id": "j5", "broll_subject": "onion"}, tmp_path)
    assert details["photo_id"] == 2
    assert "https://img.test/full/2.png" in fake.downloads
    assert "https://img.test/full/1.png" not in fake.downloads
    # Photo 2 is a solid image with red = 2, so the output must be that photo.
    assert Image.open(path).convert("RGB").getpixel((640, 360))[0] == 2
    assert not _is_colour_frame(path)


def test_failed_candidate_is_skipped_not_fatal(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.40, 2: 0.30}
    _install(monkeypatch, fake_clip, {"onion": _photos([1, 2])}, scores, broken_ids={1})
    _, details = _run({"job_id": "j5b", "broll_subject": "onion"}, tmp_path)
    assert details["photo_id"] == 2


# ── Second query and colour frame ────────────────────────────────────────────

def test_weak_subject_results_try_cleaned_prompt_second(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.10, 2: 0.12, 9: 0.40}
    fake = _install(monkeypatch, fake_clip,
                    {"onion": _photos([1, 2]), "chickpeas": _photos([9])}, scores)
    _, details = _run({"job_id": "j6", "broll_subject": "onion",
                       "thumbnail_prompt": "chickpeas"}, tmp_path)
    assert [c["query"] for c in fake.search_calls] == ["onion", "chickpeas"]
    assert details["photo_id"] == 9
    assert details["fallback"] == "second_query"
    # The second query is still scored against the subject text.
    assert fake_clip.texts[-1] == "a photo of onion"
    assert any("trying second query \"chickpeas\"" in line for line in logs)


def test_failed_full_download_tries_next_best_photo(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.40, 2: 0.30}
    fake = _install(monkeypatch, fake_clip, {"onion": _photos([1, 2])}, scores,
                    broken_full_ids={1})
    path, details = _run({"job_id": "j6b", "broll_subject": "onion"}, tmp_path)
    assert details["photo_id"] == 2
    assert details["fallback"] == "none"
    assert "https://img.test/full/1.png" in fake.downloads
    assert "https://img.test/full/2.png" in fake.downloads
    # Photo 2 is a solid image with red = 2; the colour frame is (20, 20, 60).
    assert Image.open(path).convert("RGB").getpixel((640, 360))[0] == 2


def test_colour_frame_record_and_log_show_all_searches(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.10, 2: 0.11, 9: 0.12}
    _install(monkeypatch, fake_clip,
             {"onion": _photos([1, 2]), "chickpeas": _photos([9])}, scores)
    _, details = _run({"job_id": "j6c", "broll_subject": "onion",
                       "thumbnail_prompt": "chickpeas"}, tmp_path)
    assert details["search_words"] == "onion, chickpeas"
    assert details["candidates"] == 3
    assert details["photo_id"] is None
    failure = [line for line in logs if "passed the CLIP check" in line]
    assert len(failure) == 1
    assert "\"onion\" (2 candidates)" in failure[0]
    assert "\"chickpeas\" (1 candidate)" in failure[0]


def test_both_weak_gives_colour_frame(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.10, 9: 0.12}
    fake = _install(monkeypatch, fake_clip,
                    {"onion": _photos([1]), "chickpeas": _photos([9])}, scores)
    path, details = _run({"job_id": "j7", "broll_subject": "onion",
                          "thumbnail_prompt": "chickpeas"}, tmp_path)
    assert len(fake.search_calls) == 2
    assert details["photo_id"] is None
    assert details["fallback"] == "colour_frame"
    assert not any("/full/" in u for u in fake.downloads)
    assert _is_colour_frame(path)
    assert any("solid color fallback" in line for line in logs)


def test_clip_unavailable_gives_colour_frame_not_first_photo(monkeypatch, fake_clip, logs, tmp_path):
    fake = _install(monkeypatch, fake_clip, {"onion": _photos([1])}, {1: 0.9})

    def broken_load():
        raise ImportError("open_clip is not installed")

    monkeypatch.setattr(T._broll, "_load_clip", broken_load)
    path, details = _run({"job_id": "j8", "broll_subject": "onion"}, tmp_path)
    assert fake.search_calls == []
    assert not any("/full/" in u for u in fake.downloads)
    assert details["photo_id"] is None
    assert any("CLIP model unavailable" in line and "open_clip is not installed" in line
               for line in logs)
    assert _is_colour_frame(path)


def test_no_pexels_key_gives_colour_frame(monkeypatch, fake_clip, logs, tmp_path):
    monkeypatch.setenv("PEXELS_API_KEY", "")
    monkeypatch.setattr(T, "get_pexels_key", lambda: "")
    fake = _install(monkeypatch, fake_clip, {"onion": _photos([1])}, {1: 0.9})
    _, details = _run({"job_id": "j8b", "broll_subject": "onion"}, tmp_path)
    assert fake.search_calls == []
    assert details["photo_id"] is None


# ── Log line and output file ─────────────────────────────────────────────────

def test_log_line_has_query_count_photo_id_and_score(monkeypatch, fake_clip, logs, tmp_path):
    scores = {i: 0.20 for i in range(1, 16)}
    scores[7] = 0.33
    _install(monkeypatch, fake_clip, {"onion": _photos(range(1, 16))}, scores)
    _run({"job_id": "j9", "broll_subject": "onion"}, tmp_path)
    matches = [line for line in logs if line.startswith("Thumbnail photo:")]
    assert len(matches) == 1
    line = matches[0]
    assert "search words \"onion\"" in line
    assert "15 candidates" in line
    assert "chosen Pexels id 7" in line
    assert "CLIP score 0.330" in line


def test_output_files_and_size(monkeypatch, fake_clip, logs, tmp_path):
    _install(monkeypatch, fake_clip, {"onion": _photos([4])}, {4: 0.35})
    path, _ = _run({"job_id": "j10", "broll_subject": "onion",
                    "youtube_title": "Why onions make you cry"}, tmp_path)
    assert path == tmp_path / "thumb_j10.png"
    assert (tmp_path / "thumb_raw_j10.png").exists()
    img = Image.open(path)
    assert img.size == (1280, 720)


def test_generate_thumbnail_works_without_details_argument(monkeypatch, fake_clip, logs, tmp_path):
    _install(monkeypatch, fake_clip, {"onion": _photos([4])}, {4: 0.35})
    path = T.generate_thumbnail({"job_id": "j11", "broll_subject": "onion",
                                 "youtube_title": "t"}, tmp_path)
    assert path == tmp_path / "thumb_j11.png"


def test_candidate_wording_is_right_for_0_1_and_many(monkeypatch, fake_clip, logs, tmp_path):
    # 1 candidate: the chosen-photo line says "1 candidate", not "1 candidates".
    _install(monkeypatch, fake_clip, {"onion": _photos([1])}, {1: 0.40})
    _run({"job_id": "j20a", "broll_subject": "onion"}, tmp_path)
    one = [line for line in logs if line.startswith("Thumbnail photo:")]
    assert len(one) == 1
    assert "search words \"onion\", 1 candidate, " in one[0]
    assert "candidates" not in one[0]

    # 0 candidates everywhere: the failure line says "(0 candidates)" and
    # "(1 candidate)" for the single-result query in the same line.
    logs.clear()
    _install(monkeypatch, fake_clip, {"onion": [], "chickpeas": _photos([9])}, {9: 0.10})
    _run({"job_id": "j20b", "broll_subject": "onion",
          "thumbnail_prompt": "chickpeas"}, tmp_path)
    failure = [line for line in logs if "passed the CLIP check" in line]
    assert len(failure) == 1
    assert "\"onion\" (0 candidates)" in failure[0]
    assert "\"chickpeas\" (1 candidate)" in failure[0]

    # Many: the chosen-photo line keeps the plural.
    logs.clear()
    scores = {i: 0.20 for i in range(1, 16)}
    scores[3] = 0.35
    _install(monkeypatch, fake_clip, {"onion": _photos(range(1, 16))}, scores)
    _run({"job_id": "j20c", "broll_subject": "onion"}, tmp_path)
    many = [line for line in logs if line.startswith("Thumbnail photo:")]
    assert "15 candidates, " in many[0]


def test_non_200_search_reply_is_logged_not_silent(monkeypatch, fake_clip, logs, tmp_path):
    def rate_limited(url, params=None, headers=None, timeout=None, **kwargs):
        return _Resp(429)

    monkeypatch.setattr(T.requests, "get", rate_limited)
    path, details = _run({"job_id": "j12", "broll_subject": "onion"}, tmp_path)
    assert any("Pexels photo search for \"onion\" failed: HTTP 429" in line for line in logs)
    assert details["photo_id"] is None
    assert _is_colour_frame(path)


def test_search_exception_is_logged_and_colour_frame_used(monkeypatch, fake_clip, logs, tmp_path):
    def network_down(url, params=None, headers=None, timeout=None, **kwargs):
        raise ConnectionError("made-up network error")

    monkeypatch.setattr(T.requests, "get", network_down)
    path, details = _run({"job_id": "j13", "broll_subject": "onion"}, tmp_path)
    assert any("Pexels photo search failed" in line and "made-up network error" in line
               for line in logs)
    assert details["photo_id"] is None
    assert _is_colour_frame(path)


def test_clip_scoring_exception_is_logged_and_colour_frame_used(monkeypatch, fake_clip, logs, tmp_path):
    _install(monkeypatch, fake_clip, {"onion": _photos([1])}, {1: 0.9})

    def broken_scoring(photos, subject_text, clip):
        raise RuntimeError("made-up CLIP failure")

    monkeypatch.setattr(T, "_score_photos", broken_scoring)
    path, details = _run({"job_id": "j14", "broll_subject": "onion"}, tmp_path)
    assert any("CLIP scoring failed for \"onion\"" in line and "made-up CLIP failure" in line
               for line in logs)
    assert details["photo_id"] is None
    assert _is_colour_frame(path)


def test_html_reply_with_200_is_not_chosen(monkeypatch, fake_clip, logs, tmp_path):
    scores = {1: 0.40, 2: 0.30}
    fake = _install(monkeypatch, fake_clip, {"onion": _photos([1, 2])}, scores,
                    html_full_ids={1})
    path, details = _run({"job_id": "j15", "broll_subject": "onion"}, tmp_path)
    assert details["photo_id"] == 2
    assert any("Could not download Pexels photo 1" in line for line in logs)
    assert not any("Thumbnail photo: search words" in line and "id 1," in line for line in logs)
    assert Image.open(path).convert("RGB").getpixel((640, 360))[0] == 2


def test_upload_stage_record_keeps_thumbnail_details(monkeypatch, tmp_path):
    """cmd_upload must store the thumbnail details in the thumbnail stage record.
    The thumbnail generator, the uploader and the media folder are faked."""
    import json
    from verticals import __main__ as M

    video = tmp_path / "video_en.mp4"
    video.write_bytes(b"made-up video")
    draft = {"job_id": "j16", "video_en": str(video), "youtube_title": "t",
             "broll_subject": "onion"}
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft))
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(M, "MEDIA_DIR", media)

    def fake_generate(draft, out_dir, details=None):
        details.update(search_words="onion", candidates=15, photo_id=6316541,
                       clip_score=0.3359, query_won="subject", fallback="none")
        path = out_dir / "thumb_j16.png"
        path.write_bytes(b"made-up png")
        return path

    monkeypatch.setattr(T, "generate_thumbnail", fake_generate)
    import verticals.upload as U
    monkeypatch.setattr(U, "upload_to_youtube", lambda *a, **k: "https://youtu.be/made-up")

    # The b-roll stage writes this report during produce; the upload adds to it.
    from verticals import build_report
    report_path = build_report.report_path_for(draft_path, "j16", "en")
    report_path.write_text(json.dumps({"job_id": "j16", "clips": []}))

    M.cmd_upload(SimpleNamespace(draft=str(draft_path), lang="en", force=True))

    saved = json.loads(draft_path.read_text())
    record = saved["_pipeline_state"]["thumbnail"]
    assert record["status"] == "done"
    assert record["artifacts"]["path"] == str(media / "thumb_j16.png")
    assert record["artifacts"]["photo_id"] == 6316541
    assert record["artifacts"]["clip_score"] == 0.3359
    assert record["artifacts"]["search_words"] == "onion"
    assert record["artifacts"]["fallback"] == "none"

    report = json.loads(report_path.read_text())
    assert report["thumbnail"]["photo_id"] == 6316541
    assert report["thumbnail"]["search_words"] == "onion"
    assert report["clips"] == []


def test_upload_without_build_report_still_records_thumbnail(monkeypatch, tmp_path):
    """No build report on disk (produce never wrote one): the upload must still
    finish, keep the stage record, and not create a report file."""
    import json
    from verticals import __main__ as M
    from verticals import build_report

    video = tmp_path / "video_en.mp4"
    video.write_bytes(b"made-up video")
    draft = {"job_id": "j17", "video_en": str(video), "youtube_title": "t"}
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft))
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(M, "MEDIA_DIR", media)

    def fake_generate(draft, out_dir, details=None):
        details.update(search_words="cat", candidates=0, photo_id=None,
                       clip_score=None, query_won=None, fallback="colour_frame")
        path = out_dir / "thumb_j17.png"
        path.write_bytes(b"made-up png")
        return path

    monkeypatch.setattr(T, "generate_thumbnail", fake_generate)
    import verticals.upload as U
    monkeypatch.setattr(U, "upload_to_youtube", lambda *a, **k: "https://youtu.be/made-up")

    M.cmd_upload(SimpleNamespace(draft=str(draft_path), lang="en", force=True))

    saved = json.loads(draft_path.read_text())
    assert saved["_pipeline_state"]["thumbnail"]["artifacts"]["fallback"] == "colour_frame"
    assert not build_report.report_path_for(draft_path, "j17", "en").exists()


def test_real_clip_helpers_compute_normalised_dot_product():
    """Runs the real _clip_text_feature and _clip_score_image with a tiny made-up
    model, so the normalisation and the matrix product are really exercised."""
    import torch

    seen_tokens = []

    class TinyModel:
        def encode_text(self, tokens):
            seen_tokens.append(tokens)
            return torch.tensor([[3.0, 4.0]])  # normalises to [0.6, 0.8]

        def encode_image(self, batch):
            assert batch.shape[0] == 1
            return torch.tensor([[0.0, 5.0]])  # normalises to [0, 1]

    def tokenizer(texts):
        return list(texts)

    def preprocess(img):
        return torch.zeros(3, 2, 2)

    clip = (TinyModel(), preprocess, tokenizer)
    text_feat = T._clip_text_feature("a photo of onion", clip)
    assert seen_tokens == [["a photo of onion"]]
    assert torch.allclose(text_feat, torch.tensor([[0.6, 0.8]]), atol=1e-6)

    score = T._clip_score_image(Image.new("RGB", (4, 4)), text_feat, clip)
    assert abs(score - 0.8) < 1e-6  # dot product of [0, 1] and [0.6, 0.8]
