import numpy as np
import pytest
import query_input as qi


def test_rank_chunks_orders_by_score_and_caps_at_ten():
    keywords = {f"kw{i}": {"score": 1.0, "freq": 1} for i in range(15)}
    chunk_data = []
    for i in range(15):
        chunk_data.append({
            "context": f"chunk {i}",
            "meta": [{"keyword": f"kw{i}", "score": float(i), "freq": 1}],
        })

    ranked = qi.rank_chunks(keywords, chunk_data)

    assert len(ranked) == 10
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["chunk"] == "chunk 14"  # highest score survives the cap


def test_cosine_sim_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert qi.cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal_vectors_is_zero():
    assert qi.cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


class _FakeNlp:
    """Stand-in for spaCy's nlp(): returns a fixed vector per input text, no model needed."""

    def __init__(self, vectors):
        self._vectors = vectors

    def __call__(self, text):
        return type("Doc", (), {"vector": np.array(self._vectors[text])})()


def test_semantic_dedup_absorbs_similar_chunks_keeping_higher_ranked():
    ranked = [
        {"chunk": "a", "score": 2.0, "keywords": []},
        {"chunk": "b", "score": 1.0, "keywords": []},  # near-duplicate of "a"
        {"chunk": "c", "score": 0.5, "keywords": []},  # distinct
    ]
    nlp = _FakeNlp({"a": [1, 0], "b": [1, 0.001], "c": [0, 1]})

    deduped = qi.semantic_dedup(ranked, nlp, threshold=0.9)

    assert [c["chunk"] for c in deduped] == ["a", "c"]
