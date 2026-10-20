"""Tests for verticals/upload.py retry behaviour.

Nothing here touches the network, YouTube, a real token file, or a real delay.
The token path, the credentials loader, the Google client, and time.sleep are all
replaced with fakes.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from verticals import upload as upload_module
from verticals.upload import UploadResponseError, upload_to_youtube

FAKE_ID = "FAKEVIDEO01"
DRAFT = {"niche": "science", "youtube_title": "Made-up title", "news": "made-up news"}


class _BrokenPath:
    """Stands in for a thumbnail path whose exists() check fails unexpectedly."""

    def exists(self):
        raise OSError("made-up disk error")


@pytest.fixture
def fake_upload(tmp_path):
    """Replaces the token, credentials, Google client, sleep and logging. Yields a namespace
    whose .service is the fake YouTube service and .sleep is the fake time.sleep."""
    creds = MagicMock(expired=False)
    service = MagicMock()
    env = SimpleNamespace(service=service, sleep=None, tmp=tmp_path)
    with patch.object(upload_module, "get_youtube_token_path", return_value=tmp_path / "token.json"), \
            patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
                  return_value=creds), \
            patch("googleapiclient.discovery.build", return_value=service), \
            patch("googleapiclient.http.MediaFileUpload", MagicMock()), \
            patch("verticals.niche.load_niche", return_value={}), \
            patch.object(upload_module, "log"), \
            patch("verticals.retry.time.sleep") as sleep:
        env.sleep = sleep
        yield env


def _set_chunk_results(env, results):
    """Makes the resumable upload's next_chunk() return or raise these results in order.
    Returns the fake insert request so tests can count how many were made."""
    req = MagicMock()
    req.next_chunk.side_effect = results
    env.service.videos.return_value.insert.return_value = req
    return req


def test_error_after_insert_returns_url_and_does_not_insert_again(fake_upload):
    req = _set_chunk_results(fake_upload, [(None, {"id": FAKE_ID})])

    url = upload_to_youtube(Path("fake_video.mp4"), DRAFT, None, "en", _BrokenPath())

    assert url == f"https://youtu.be/{FAKE_ID}"
    assert fake_upload.service.videos.return_value.insert.call_count == 1
    assert req.next_chunk.call_count == 1
    fake_upload.sleep.assert_not_called()


@pytest.mark.parametrize("response", [{}, {"id": ""}, {"id": "   "}, {"title": "no id"}, "not a dict"])
def test_response_without_id_raises_and_is_not_retried(fake_upload, response):
    _set_chunk_results(fake_upload, [(None, response)])

    with pytest.raises(UploadResponseError) as caught:
        upload_to_youtube(Path("fake_video.mp4"), DRAFT, None, "en", None)

    assert "YouTube Studio" in str(caught.value)
    assert fake_upload.service.videos.return_value.insert.call_count == 1
    fake_upload.sleep.assert_not_called()


def test_network_error_before_insert_completes_is_retried(fake_upload):
    _set_chunk_results(fake_upload, [ConnectionError("made-up network drop"),
                                     (None, {"id": FAKE_ID})])

    url = upload_to_youtube(Path("fake_video.mp4"), DRAFT, None, "en", None)

    assert url == f"https://youtu.be/{FAKE_ID}"
    assert fake_upload.service.videos.return_value.insert.call_count == 2
    assert fake_upload.sleep.call_count == 1
