"""Tavily research — anti-hallucination gate."""
import requests

from .config import extract_keywords, get_tavily_key
from .log import log
from .retry import with_retry


def _truncate_at_sentence(text: str, max_chars: int = 500) -> str:
    """Truncate text to at most max_chars, cutting at the last sentence-ending
    punctuation before the limit so snippets don't chop off mid-word/mid-sentence.
    Falls back to a word boundary if no good sentence break is found."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for punct in [". ", "! ", "? "]:
        idx = truncated.rfind(punct)
        if idx != -1 and idx > max_chars * 0.4:
            return truncated[: idx + 1].strip()
    idx = truncated.rfind(" ")
    if idx != -1:
        return truncated[:idx].strip() + "..."
    return truncated.strip() + "..."


@with_retry(max_retries=2, base_delay=2.0)
def _fetch_tavily(query: str) -> dict:
    """Fetch search results from the Tavily API."""
    api_key = get_tavily_key()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set — add it to config.json")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def research_topic(news: str) -> str:
    """Multi-angle Tavily search -> extract facts for anti-hallucination gate.

    Runs a couple of varied queries (base topic, "explained") to widen
    coverage versus a single search. Falls back gracefully if any
    individual query fails.
    """
    log("Researching topic via Tavily...")
    keywords = extract_keywords(news)
    query_variants = [
        keywords,
        f"{keywords} explained",
    ]

    all_snippets = []
    seen = set()
    queries_succeeded = 0

    for query in query_variants:
        try:
            data = _fetch_tavily(query)
            results = data.get("results", [])
            for item in results:
                content = item.get("content", "")
                content = _truncate_at_sentence(content, 500)  # sanitize: limit prompt injection surface
                if content and content not in seen:
                    seen.add(content)
                    all_snippets.append(content)
            if results:
                queries_succeeded += 1
        except Exception as e:
            log(f"  Query '{query}' failed: {e}")
            continue

    if all_snippets:
        research = "\n".join(all_snippets[:15])
        log(f"Found {len(all_snippets)} unique snippets across {queries_succeeded}/{len(query_variants)} queries.")
        return research

    log("No research results from any query — proceeding without.")
    return f"Topic: {news}\n(No live research available — script must stay general.)"