"""Pure metric functions for the eval. No network; spaCy only for lemmas/vectors."""
import hashlib
import json
import os

import numpy as np

from pdf_Reader import nlp, LIGHT
from query_input import cosine_sim
from youtube_api import filter_and_rank, MIN_DURATION_SECONDS, MIN_VIEW_COUNT, _duration_seconds, _is_english

SOFT_THRESHOLD = 0.7


def _docs(texts):
    return list(nlp.pipe(texts, disable=LIGHT))


def _lemmas(doc):
    return {t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop}


def coverage(topics, chunk_texts):
    """Share of gold topics found in the selected chunks.
    exact: every content lemma of the topic appears in some single chunk.
    soft:  exact OR topic-vs-chunk vector similarity >= SOFT_THRESHOLD."""
    if not topics:
        return {"exact": None, "soft": None, "missing": []}
    chunk_docs = _docs(chunk_texts)
    chunk_lemmas = [_lemmas(d) for d in chunk_docs]
    exact = soft = 0
    missing = []
    for topic, tdoc in zip(topics, _docs(topics)):
        need = _lemmas(tdoc)
        hit = bool(need) and any(need <= have for have in chunk_lemmas)
        near = any(cosine_sim(tdoc.vector, d.vector) >= SOFT_THRESHOLD for d in chunk_docs)
        exact += hit
        soft += hit or near
        if not (hit or near):
            missing.append(topic)
    return {"exact": exact / len(topics), "soft": soft / len(topics), "missing": missing}


def diversity(chunk_texts):
    """Mean pairwise cosine between selected chunks. Lower = more varied (dedup working)."""
    vecs = [d.vector for d in _docs(chunk_texts)]
    sims = [cosine_sim(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return float(np.mean(sims)) if sims else None


def query_stats(selected, queries):
    """Label-free query checks: fallback rate, formatting artifacts, query-vs-chunk similarity."""
    if not queries:
        return {"n": 0}
    chunk_docs = _docs([c["chunk"] for c in selected])
    query_docs = _docs(queries)
    fallback = artifacts = 0
    sims = []
    for c, q, cd, qd in zip(selected, queries, chunk_docs, query_docs):
        if q == " ".join(kw["keyword"] for kw in c["keywords"]):
            fallback += 1
        if "\n" in q or q.strip() != q or q.startswith(('"', "'")) or len(q.split()) > 12:
            artifacts += 1
        sims.append(cosine_sim(cd.vector, qd.vector))
    n = len(queries)
    return {"n": n, "fallback_rate": fallback / n, "artifact_rate": artifacts / n, "mean_sim": float(np.mean(sims))}


def pair_key(chunk, query):
    return hashlib.sha1(f"{chunk}\n{query}".encode("utf-8")).hexdigest()[:12]


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def judge_scores(selected, queries, judgments):
    """Returns (mean judge score or None, list of (key, chunk, query) still unjudged)."""
    scores, todo = [], []
    for c, q in zip(selected, queries):
        key = pair_key(c["chunk"], q)
        if key in judgments:
            scores.append(judgments[key]["score"])
        else:
            todo.append((key, c["chunk"], q))
    return (float(np.mean(scores)) if scores else None), todo


def replay_filter(fixture):
    """Re-run the pure filter/rank on recorded YouTube data and compare with what was returned,
    plus count how many candidates each rule rejected."""
    cands, details = fixture["candidates"], fixture["details"]
    if not cands:
        return None
    again = [v["video_id"] for v in filter_and_rank(cands, details, topic=fixture.get("topic"))]
    rejected = {"short": 0, "low_views": 0, "non_english": 0}
    for vid in cands:
        item = details.get(vid)
        if not item:
            continue
        rejected["short"] += _duration_seconds(item["contentDetails"].get("duration", "")) < MIN_DURATION_SECONDS
        rejected["low_views"] += int(item["statistics"].get("viewCount", 0)) < MIN_VIEW_COUNT
        rejected["non_english"] += not _is_english(item["snippet"])
    return {
        "candidates": len(cands),
        "returned": len(fixture["videos"]),
        "replay_matches": again == [v["video_id"] for v in fixture["videos"]],
        "rejected": rejected,
    }


def precision_at_k(returned_ids, relevant_ids, k=5):
    top = returned_ids[:k]
    return sum(v in relevant_ids for v in top) / len(top) if top else None
