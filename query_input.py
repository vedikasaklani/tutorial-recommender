import logging

import numpy as np
from pdf_Reader import nlp, extract_chunks
from numpy.linalg import norm

logger = logging.getLogger(__name__)

#rank chunks also
def rank_chunks(keywords, chunk_data):
    ranked = []
    for chunk in chunk_data:
        score=[kw["score"] for kw in chunk["meta"]]
        chunkscore=sum(score)
        for kw in chunk["meta"]:
            chunkscore*=keywords[kw["keyword"]]["score"]

        ranked.append({
            "chunk": chunk["context"],
            "score": chunkscore,
            "keywords": chunk["meta"],
        })
    
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:10]

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

#for semantic deduplication using cosine similarity of vectors
def semantic_dedup(ranked, nlp, threshold=0.9):
    items = sorted(ranked, key=lambda x: x["score"], reverse=True)
    
    vectors = {chunk["chunk"]: nlp(chunk["chunk"]).vector for chunk in items}
    
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


def get_dedup_ranked_chunks(pdf_source):
    """Run the chunk/keyword extraction, ranking, and dedup stages for a PDF."""
    chunk_data, keywords = extract_chunks(pdf_source)
    logger.info("extracted %d chunks, %d unique keywords", len(chunk_data), len(keywords))
    ranked = rank_chunks(keywords, chunk_data)
    logger.debug("ranked top %d chunks", len(ranked))
    deduped = semantic_dedup(ranked, nlp)
    logger.info("%d chunks remain after semantic dedup", len(deduped))
    return deduped


