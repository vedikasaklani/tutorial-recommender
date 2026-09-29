import numpy as np
from pdf_Reader import global_keywords as keywords, nlp, chunk_data
from numpy.linalg import norm

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
ranked=rank_chunks(keywords, chunk_data)

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

dedup_ranked=semantic_dedup(ranked, nlp)


