"""YouTube API upload + thumbnail + captions."""

from pathlib import Path

from .config import get_youtube_token_path, write_secret_file
from .log import log
from .retry import with_retry


class UploadResponseError(Exception):
    """YouTube's reply to the upload had no video id. The video may already exist in
    YouTube Studio, so this is never retried: a retry could upload it a second time."""


def _video_id_from_response(response) -> str:
    video_id = response.get("id") if isinstance(response, dict) else None
    if not isinstance(video_id, str) or not video_id.strip():
        raise UploadResponseError(
            "YouTube's reply to the upload had no video id. The video may already exist "
            "in YouTube Studio. Check Studio (including private videos) before running "
            "this topic again."
        )
    return video_id


def _log_quietly(message: str) -> None:
    """Logs a message. A logging failure must never fail an upload that already happened."""
    try:
        log(message)
    except Exception:
        pass


def _finish_after_insert(youtube, video_id: str, srt_path: Path, lang: str, thumbnail_path: Path) -> None:
    """Captions and thumbnail, run after the video exists. Nothing here may raise,
    because an error would make with_retry upload the video again."""
    from googleapiclient.http import MediaFileUpload

    try:
        if srt_path and srt_path.exists():
            try:
                youtube.captions().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "videoId": video_id,
                            "language": lang,
                            "name": lang.upper(),
                            "isDraft": False,
                        }
                    },
                    media_body=MediaFileUpload(str(srt_path), mimetype="application/octet-stream"),
                ).execute()
                log("Captions uploaded.")
            except Exception as e:
                log(f"Caption upload failed: {e}")

        if thumbnail_path and thumbnail_path.exists():
            try:
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
                ).execute()
                log("Thumbnail uploaded.")
            except Exception as e:
                log(f"Thumbnail upload failed: {e}")
    except Exception as e:
        _log_quietly(f"Post-upload step failed, but the video is uploaded: {e}")


@with_retry(max_retries=2, base_delay=5.0, no_retry=(UploadResponseError,))
def upload_to_youtube(
    video_path: Path,
    draft: dict,
    srt_path: Path = None,
    lang: str = "en",
    thumbnail_path: Path = None,
) -> str:
    """Upload video to YouTube with metadata, captions, and optional thumbnail."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from .niche import load_niche

    niche_name = draft.get("niche", "general")
    try:
        profile = load_niche(niche_name)
        category_id = str(profile.get("youtube_category_id", "27"))
    except Exception:
        category_id = "27"  # Education, safe fallback if niche fails to load

    token_path = get_youtube_token_path()
    creds = Credentials.from_authorized_user_file(str(token_path))
    if creds.expired:
        if creds.refresh_token:
            creds.refresh(Request())
            write_secret_file(token_path, creds.to_json())
        else:
            raise RuntimeError(
                "YouTube OAuth token is expired and has no refresh token.\n"
                "Re-run: python3 scripts/setup_youtube_oauth.py"
            )

    youtube = build("youtube", "v3", credentials=creds)
    log(f"Uploading {video_path.name}...")

    description = draft.get("youtube_description", "")

    # Append YouTube chapter markers if present (long-form only) - YouTube
    # auto-detects these from timestamp lines in the description as long
    # as the first is 0:00 and there are 3+, each >=10s apart (already
    # enforced when chapters were computed).
    chapters = draft.get("chapters") or []
    if chapters:
        chapter_lines = []
        for ch in chapters:
            total_seconds = int(ch["start_seconds"])
            h, rem = divmod(total_seconds, 3600)
            m, s = divmod(rem, 60)
            timestamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            chapter_lines.append(f"{timestamp} {ch['heading']}")
        description = description.rstrip() + "\n\n" + "\n".join(chapter_lines)

    body = {
        "snippet": {
            "title": draft.get("youtube_title", draft["news"])[:100],
            "description": description,
            "tags": draft.get("youtube_tags", "").split(","),
            "categoryId": category_id,
            "defaultLanguage": lang,
            "defaultAudioLanguage": lang,
        },
        "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False},
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            log(f"Upload progress: {int(status.progress() * 100)}%")

    # The video now exists on YouTube. Nothing below may start another insert.
    video_id = _video_id_from_response(response)
    url = f"https://youtu.be/{video_id}"
    _log_quietly(f"Uploaded: {url}")

    _finish_after_insert(youtube, video_id, srt_path, lang, thumbnail_path)
    return url
