"""DuckDuckGo research — anti-hallucination gate."""

import requests
from html.parser import HTMLParser

from .config import extract_keywords
from .log import log
from .retry import with_retry


@with_retry(max_retries=2, base_delay=2.0)
def _fetch_ddg(keywords: str) -> str:
    """Fetch search snippets from DuckDuckGo HTML endpoint."""
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    r = requests.post(url, data={"q": keywords}, headers=headers, timeout=10)
    r.raise_for_status()
    return r.text


def _parse_snippets(html: str) -> list[str]:
    """Extract result snippets from a DuckDuckGo HTML results page."""
    snippets = []

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self._in = False
            self._text = []

        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "a" and "result__snippet" in d.get("class", ""):
                self._in = True
                self._text = []

        def handle_endtag(self, tag):
            if self._in and tag == "a":
                snippets.append("".join(self._text).strip())
                self._in = False

        def handle_data(self, data):
            if self._in:
                self._text.append(data)

    p = Parser()
    p.feed(html)
    return snippets


def research_topic(news: str) -> str:
    """Multi-angle DuckDuckGo search -> extract facts for anti-hallucination gate.

    Runs a few varied queries (base topic, "how it works", "review") to widen
    coverage versus a single search, which matters most for niche topics with
    thin web presence. Falls back gracefully if any individual query fails.
    """
    log("Researching topic via DuckDuckGo...")
    keywords = extract_keywords(news)

    query_variants = [
        keywords,
        f"{keywords} explained",
        f"{keywords} review",
    ]

    all_snippets = []
    seen = set()
    queries_succeeded = 0

    for query in query_variants:
        try:
            html = _fetch_ddg(query)
            snippets = _parse_snippets(html)
            for s in snippets:
                s = s[:300]  # sanitize: truncate each to limit prompt injection surface
                if s and s not in seen:
                    seen.add(s)
                    all_snippets.append(s)
            if snippets:
                queries_succeeded += 1
        except Exception as e:
            log(f"  Query '{query}' failed: {e}")
            continue

    if all_snippets:
        # Cap total volume so the research prompt doesn't grow unbounded
        research = "\n".join(all_snippets[:15])
        log(f"Found {len(all_snippets)} unique snippets across {queries_succeeded}/{len(query_variants)} queries.")
        return research

    log("No research results from any query — proceeding without.")
    return f"Topic: {news}\n(No live research available — script must stay general.)"