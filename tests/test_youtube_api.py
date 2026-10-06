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


def test_build_video_pool_ranks_by_views_and_caps_at_max_results(monkeypatch):
    candidates = {
        "low": _video("low", views=1000),
        "high": _video("high", views=9000),
        "mid": _video("mid", views=5000),
    }

    monkeypatch.setattr(yt, "search_youtube", lambda prompt, api_key, max_results=10: [c for c, _ in candidates.values()])
    monkeypatch.setattr(yt, "fetch_video_details", lambda video_ids, api_key: {vid: d for vid, (_, d) in candidates.items()})

    result = yt.build_video_pool(["query"], api_key="fake", max_results=2)

    assert [v["video_id"] for v in result] == ["high", "mid"]


def test_build_video_pool_returns_empty_when_no_candidates(monkeypatch):
    monkeypatch.setattr(yt, "search_youtube", lambda prompt, api_key, max_results=10: [])
    assert yt.build_video_pool(["query"], api_key="fake") == []
