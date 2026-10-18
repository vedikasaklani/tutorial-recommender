import math
import re
import requests
import os

from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ponytail: naive fixed thresholds, not a learned/relevance-aware score.
# Tighten or expose as config if these measurably cut the wrong videos.
MIN_DURATION_SECONDS = 240  # YouTube's own "short" bucket is <4min; stay above it to skip Shorts
MIN_VIEW_COUNT = 1000

_ISO8601_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _duration_seconds(iso_duration):
    match = _ISO8601_DURATION_RE.fullmatch(iso_duration)
    if not match:
        return 0
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _is_english(snippet):
    lang = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
    return lang is None or lang.startswith("en")  # benefit of the doubt when unset


def search_youtube(prompt, api_key, max_results=5):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": prompt,
        "type": "video",
        "maxResults": max_results,
        "relevanceLanguage": "en",
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    results = []
    for item in resp.json().get("items", []):
        vid = item["id"]["videoId"]
        snippet = item["snippet"]
        results.append({
            "video_id": vid,
            "title": snippet["title"],
            "description": snippet["description"],
            "link": f"https://www.youtube.com/watch?v={vid}",
        })
    return results


def fetch_video_details(video_ids, api_key):
    """Look up duration/views/language for up to 50 video IDs per call."""
    details = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = requests.get("https://www.googleapis.com/youtube/v3/videos", params={
            "part": "contentDetails,statistics,snippet",
            "id": ",".join(batch),
            "key": api_key,
        }, timeout=15)
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            details[item["id"]] = item
    return details


def build_video_pool(queries, api_key, max_results=5, trace=None, topic=None):
    """Search all queries, dedupe, filter out Shorts/low-view/non-English, rank by views.

    trace: optional dict that receives the raw candidates/details (used to record eval fixtures)."""
    candidates = {}
    for query in queries:
        for rank, c in enumerate(search_youtube(query, api_key, max_results=10)):  # over-fetch, filtering will cut some
            candidates.setdefault(c["video_id"], {**c, "ranks": []})["ranks"].append(rank)  # YouTube's own order per query

    if not candidates:
        return []
    details = fetch_video_details(list(candidates.keys()), api_key)
    if trace is not None:
        trace.update(candidates=candidates, details=details)
    return filter_and_rank(candidates, details, max_results, topic)


_STOP = {"with", "from", "that", "this", "using", "into", "about", "their", "system", "systems",
         "implementation", "design", "study", "report", "project"}


def _topic_prefixes(topic):
    """5-letter prefixes of the topic's words, so "balancer" also matches "balancing"."""
    words = re.findall(r"[a-z]{4,}", (topic or "").lower())
    return {w[:5] for w in words if w not in _STOP} | ({"http"} if "http" in (topic or "").lower() else set())


def _relevance(video, prefixes):
    tokens = re.findall(r"[a-z]+", f"{video['title']} {video['description']}".lower())
    return sum(any(t.startswith(p) for t in tokens) for p in prefixes)


# ponytail: hand-picked weights; tune with `python -m eval.run --sweep-rank` once fixtures have gold videos.
W_SEARCH, W_TOPIC, W_VIEWS = 1.0, 1.0, 1.5


def _score(video, prefixes, w_search, w_topic, w_views):
    """Fuse relevance and popularity: YouTube's rank for each query that found the video
    (1, 1/2, 1/3, ...; summed so multi-query hits win), topic-word overlap (0-1), and views
    (log10 scaled to ~0-1 so 10M views is not 1000x better than 10k)."""
    search = sum(1 / (r + 1) for r in video.get("ranks", []))
    topic = _relevance(video, prefixes) / len(prefixes) if prefixes else 0
    return w_search * search + w_topic * topic + w_views * math.log10(max(video["views"], 1)) / 7


def filter_and_rank(candidates, details, max_results=5, topic=None, w_search=W_SEARCH, w_topic=W_TOPIC, w_views=W_VIEWS):
    """Pure part of build_video_pool: no network, so it can be replayed from recorded data."""
    pool = []
    for video_id, c in candidates.items():
        item = details.get(video_id)
        if not item:
            continue
        duration = _duration_seconds(item["contentDetails"].get("duration", ""))  # missing for premieres/live -> 0 -> filtered
        views = int(item["statistics"].get("viewCount", 0))
        if duration < MIN_DURATION_SECONDS or views < MIN_VIEW_COUNT or not _is_english(item["snippet"]):
            continue
        pool.append({**c, "views": views})  # keeps "ranks" for scoring; stripped below

    prefixes = _topic_prefixes(topic)
    pool.sort(key=lambda c: (_score(c, prefixes, w_search, w_topic, w_views), c["views"]), reverse=True)
    return [{k: v for k, v in c.items() if k != "ranks"} for c in pool[:max_results]]


def _demo():
    assert _duration_seconds("PT10M30S") == 630
    assert _duration_seconds("PT1H2M3S") == 3723
    assert _duration_seconds("PT45S") == 45
    assert _duration_seconds("P0D") == 0  # live/unknown duration -> filtered out, not crashed
    assert _is_english({}) is True
    assert _is_english({"defaultAudioLanguage": "en-US"}) is True
    assert _is_english({"defaultAudioLanguage": "hi"}) is False
    print("youtube_api self-check passed")


if __name__ == "__main__":
    _demo()
