import requests
import os

from dotenv import load_dotenv
from hugging_face import queries

load_dotenv() 

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") 


def search_youtube(prompt, api_key, max_results=5):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": prompt,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
    }
    resp = requests.get(url, params=params)
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


def build_video_pool(queries, api_key, max_results=5):
    """Search all keywords dedupe candidates by video_id."""
    pool = {}
    for query in queries:
        for c in search_youtube(query, api_key, max_results):
            pool.setdefault(c["video_id"], c)
    return pool



links=build_video_pool(queries, YOUTUBE_API_KEY, 3)