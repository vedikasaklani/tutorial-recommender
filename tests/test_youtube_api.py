import youtube_api as yt


def test_duration_seconds_parses_hours_minutes_seconds():
    assert yt._duration_seconds("PT1H2M3S") == 3723


def test_duration_seconds_parses_minutes_only():
    assert yt._duration_seconds("PT10M30S") == 630


def test_duration_seconds_parses_seconds_only():
    assert yt._duration_seconds("PT45S") == 45


def test_duration_seconds_unparseable_returns_zero():
    assert yt._duration_seconds("P0D") == 0


def test_is_english_defaults_true_when_unset():
    assert yt._is_english({}) is True


def test_is_english_true_for_en_variants():
    assert yt._is_english({"defaultAudioLanguage": "en-US"}) is True


def test_is_english_false_for_other_languages():
    assert yt._is_english({"defaultAudioLanguage": "hi"}) is False


def _video(video_id, title="t", duration="PT10M0S", views=5000, lang="en"):
    return {
        "video_id": video_id,
        "title": title,
        "description": "d",
        "link": f"https://www.youtube.com/watch?v={video_id}",
    }, {
        "id": video_id,
        "contentDetails": {"duration": duration},
        "statistics": {"viewCount": str(views)},
        "snippet": {"defaultAudioLanguage": lang} if lang else {},
    }


def test_build_video_pool_filters_shorts_low_views_and_non_english(monkeypatch):
    candidates = {
        "ok": _video("ok", duration="PT10M0S", views=5000, lang="en"),
        "too_short": _video("too_short", duration="PT0M30S", views=5000, lang="en"),
        "too_few_views": _video("too_few_views", duration="PT10M0S", views=10, lang="en"),
        "non_english": _video("non_english", duration="PT10M0S", views=5000, lang="hi"),
    }

    def fake_search(prompt, api_key, max_results=10):
        return [c for c, _ in candidates.values()]

    def fake_details(video_ids, api_key):
        return {vid: details for vid, (_, details) in candidates.items()}

    monkeypatch.setattr(yt, "search_youtube", fake_search)
    monkeypatch.setattr(yt, "fetch_video_details", fake_details)

    result = yt.build_video_pool(["query"], api_key="fake", max_results=5)

    assert [v["video_id"] for v in result] == ["ok"]


def test_build_video_pool_ties_on_search_rank_fall_back_to_views_and_cap(monkeypatch):
    candidates = {
        "low": _video("low", views=1000),
        "high": _video("high", views=9000),
        "mid": _video("mid", views=5000),
    }
    # one query per video, each ranked #1 for its own query -> equal search score, views decide
    monkeypatch.setattr(yt, "search_youtube", lambda prompt, api_key, max_results=10: [candidates[prompt][0]])
    monkeypatch.setattr(yt, "fetch_video_details", lambda video_ids, api_key: {vid: d for vid, (_, d) in candidates.items()})

    result = yt.build_video_pool(["low", "high", "mid"], api_key="fake", max_results=2)

    assert [v["video_id"] for v in result] == ["high", "mid"]


def test_youtube_rank_and_multi_query_hits_beat_raw_views(monkeypatch):
    popular, popular_d = _video("popular", views=900_000)
    relevant, relevant_d = _video("relevant", views=20_000)
    # "relevant" is #1 for both queries, "popular" only #5 for one of them
    results = {"q1": [relevant, popular], "q2": [relevant]}
    monkeypatch.setattr(yt, "search_youtube", lambda prompt, api_key, max_results=10: results[prompt])
    monkeypatch.setattr(yt, "fetch_video_details", lambda ids, api_key: {"popular": popular_d, "relevant": relevant_d})

    result = yt.build_video_pool(["q1", "q2"], api_key="fake")

    assert [v["video_id"] for v in result] == ["relevant", "popular"]
    assert "ranks" not in result[0]  # internal field is not leaked to the extension


def test_build_video_pool_returns_empty_when_no_candidates(monkeypatch):
    monkeypatch.setattr(yt, "search_youtube", lambda prompt, api_key, max_results=10: [])
    assert yt.build_video_pool(["query"], api_key="fake") == []


def test_topic_outranks_raw_views():
    generic, generic_d = _video("generic", views=200_000)
    on_topic, on_topic_d = _video("on_topic", views=2000)
    generic["title"] = "10 Design Patterns Explained"
    on_topic["title"] = "Load Balancing Algorithms Explained"
    on_topic["description"] = "round robin vs least connections in a load balancer"
    cands = {"generic": generic, "on_topic": on_topic}
    details = {"generic": generic_d, "on_topic": on_topic_d}

    assert [v["video_id"] for v in yt.filter_and_rank(cands, details, topic="multithreaded HTTP load balancer")] == ["on_topic", "generic"]
    assert [v["video_id"] for v in yt.filter_and_rank(cands, details)] == ["generic", "on_topic"]  # no topic: views only
