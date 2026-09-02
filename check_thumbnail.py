import json
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

token_path = Path(r"E:\Youtube Project\youtube-shorts-pipeline\.verticals\youtube_token.json")
data = json.loads(token_path.read_text())
creds = Credentials(
    token=data["token"],
    refresh_token=data["refresh_token"],
    token_uri=data["token_uri"],
    client_id=data["client_id"],
    client_secret=data["client_secret"],
    scopes=data["scopes"],
)

youtube = build("youtube", "v3", credentials=creds)
r = youtube.videos().list(part="snippet", id="wL4HpWOZbc4").execute()
print(r["items"][0]["snippet"]["thumbnails"])
