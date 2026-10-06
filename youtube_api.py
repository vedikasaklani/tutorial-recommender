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


def build_video_pool(queries, api_key, max_results=5):
    """Search all queries, dedupe, filter out Shorts/low-view/non-English, rank by views."""
    candidates = {}
    for query in queries:
        for c in search_youtube(query, api_key, max_results=10):  # over-fetch, filtering will cut some
            candidates.setdefault(c["video_id"], c)

    if not candidates:
        return []
    details = fetch_video_details(list(candidates.keys()), api_key)

    pool = []
    for video_id, c in candidates.items():
        item = details.get(video_id)
        if not item:
            continue
        duration = _duration_seconds(item["contentDetails"]["duration"])
        views = int(item["statistics"].get("viewCount", 0))
        if duration < MIN_DURATION_SECONDS or views < MIN_VIEW_COUNT or not _is_english(item["snippet"]):
            continue
        pool.append({**c, "views": views})

    pool.sort(key=lambda c: c["views"], reverse=True)
    return pool[:max_results]


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
