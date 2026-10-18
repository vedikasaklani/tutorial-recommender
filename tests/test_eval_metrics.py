from eval import metrics as m


def test_pair_key_is_stable_and_depends_on_both_parts():
    assert m.pair_key("chunk", "query") == m.pair_key("chunk", "query")
    assert m.pair_key("chunk", "query") != m.pair_key("chunk", "other query")


def test_precision_at_k():
    assert m.precision_at_k(["a", "b", "c"], {"a", "c"}) == 2 / 3
    assert m.precision_at_k([], {"a"}) is None


def test_coverage_exact_needs_all_topic_lemmas_in_one_chunk():
    chunks = ["Normalization removes redundant data from relational tables."]
    cov = m.coverage(["relational table normalization", "quantum computing"], chunks)
    assert cov["exact"] == 0.5
    assert "quantum computing" in cov["missing"]


def test_replay_filter_matches_recorded_videos():
    cands = {"ok": {"video_id": "ok", "title": "t", "description": "", "link": "l"},
             "short": {"video_id": "short", "title": "t", "description": "", "link": "l"}}
    details = {
        "ok": {"contentDetails": {"duration": "PT10M0S"}, "statistics": {"viewCount": "5000"}, "snippet": {}},
        "short": {"contentDetails": {"duration": "PT30S"}, "statistics": {"viewCount": "5000"}, "snippet": {}},
    }
    fixture = {"candidates": cands, "details": details, "videos": [{"video_id": "ok"}]}
    out = m.replay_filter(fixture)
    assert out["replay_matches"] is True
    assert out["rejected"]["short"] == 1
