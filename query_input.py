import logging

import numpy as np
from pdf_Reader import nlp, extract_chunks, LIGHT
from numpy.linalg import norm

logger = logging.getLogger(__name__)

#rank chunks also
def rank_chunks(keywords, chunk_data, scoring="product"):
    """scoring: "product" is the original formula (multiplies by every keyword's global
    score, so chunks with many keywords can rank lower); "sum" adds score*global instead."""
    if scoring == "coverage":
        return _coverage_select(keywords, chunk_data)
    ranked = []
    for chunk in chunk_data:
        if scoring == "sum":
            chunkscore = sum(kw["score"] * keywords[kw["keyword"]]["score"] for kw in chunk["meta"])
        else:
            chunkscore=sum(kw["score"] for kw in chunk["meta"])
            for kw in chunk["meta"]:
                chunkscore*=keywords[kw["keyword"]]["score"]

        ranked.append({
            "chunk": chunk["context"],
            "score": chunkscore,
            "keywords": chunk["meta"],
        })
    
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:10]

MIN_WORDS = 20  # shorter chunks are table headers / captions / page fragments, not teachable content


def _coverage_select(keywords, chunk_data, k=8):
    """Greedy set cover: repeatedly take the chunk adding the most not-yet-covered keyword weight
    (score * freq over the whole document). Spreads picks across topics instead of taking the
    top-k scores and hoping dedup leaves enough variety."""
    pool = [c for c in chunk_data if len(c["context"].split()) >= MIN_WORDS]
    covered, picked = set(), []
    while pool and len(picked) < k:
        def gain(c):
            return sum(keywords[w]["score"] * keywords[w]["freq"] for w in {m["keyword"] for m in c["meta"]} - covered)
        best = max(pool, key=gain)
        g = gain(best)
        if g <= 0:
            break
        picked.append({"chunk": best["context"], "score": g, "keywords": best["meta"]})
        covered |= {m["keyword"] for m in best["meta"]}
        pool.remove(best)
    return picked


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

#for semantic deduplication using cosine similarity of vectors
def semantic_dedup(ranked, nlp, threshold=0.9):
    items = sorted(ranked, key=lambda x: x["score"], reverse=True)
    
    texts = [chunk["chunk"] for chunk in items]
    vectors = {t: d.vector for t, d in zip(texts, nlp.pipe(texts, disable=LIGHT))}
    
    absorbed = set()
    merged = []
    
    for i, chunk_og in enumerate(items):
        if chunk_og["chunk"] in absorbed:
            continue
        merged.append(chunk_og)  #this is the surviving representative
        
        for chunk in items[i+1:]:
            if chunk["chunk"] in absorbed:
                continue
            sim = cosine_sim(vectors[chunk_og["chunk"]], vectors[chunk["chunk"]])
            if sim >= threshold:
                absorbed.add(chunk["chunk"])
    
    return merged


# Picked with `python -m eval.run --sweep` (topic coverage on os_report.pdf: 0.17 -> 0.83).
# Averaged word vectors from en_core_web_md are all >0.9 similar within one document, so
# a lower dedup threshold collapsed the selection to 1-4 chunks.
CHUNK_SIZE = 3
SCORING = "coverage"
DEDUP_THRESHOLD = 0.98


def get_dedup_ranked_chunks(pdf_source):
    """Run the chunk/keyword extraction, ranking, and dedup stages for a PDF."""
    chunk_data, keywords = extract_chunks(pdf_source, CHUNK_SIZE)
    logger.info("extracted %d chunks, %d unique keywords", len(chunk_data), len(keywords))
    ranked = rank_chunks(keywords, chunk_data, SCORING)
    logger.debug("ranked top %d chunks", len(ranked))
    deduped = semantic_dedup(ranked, nlp, DEDUP_THRESHOLD)
    logger.info("%d chunks remain after semantic dedup", len(deduped))
    return deduped


