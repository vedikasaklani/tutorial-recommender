"""Save one live pipeline run as an eval fixture (opt-in via EVAL_RECORD_DIR=eval)."""
import hashlib
import io
import json
import os

from pdf_Reader import extract_chunks
from query_input import rank_chunks, CHUNK_SIZE, SCORING


def _trim_details(details):
    """Keep only the fields filter_and_rank reads, so fixtures stay small."""
    out = {}
    for vid, item in details.items():
        snippet = item.get("snippet", {})
        out[vid] = {
            "contentDetails": {"duration": item["contentDetails"].get("duration", "")},
            "statistics": {"viewCount": item.get("statistics", {}).get("viewCount", "0")},
            "snippet": {k: snippet[k] for k in ("defaultAudioLanguage", "defaultLanguage") if k in snippet},
        }
    return out


def save_fixture(out_dir, pdf_source, selected, queries, trace, videos, topic=None):
    data = pdf_source.getvalue() if hasattr(pdf_source, "getvalue") else open(pdf_source, "rb").read()
    fid = hashlib.sha256(data).hexdigest()[:8]

    chunk_data, global_kw = extract_chunks(io.BytesIO(data), CHUNK_SIZE)
    ranked = rank_chunks(global_kw, chunk_data, SCORING)
    top_kw = sorted(global_kw, key=lambda k: global_kw[k]["score"] * global_kw[k]["freq"], reverse=True)[:30]

    os.makedirs(os.path.join(out_dir, "pdfs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fixtures"), exist_ok=True)
    with open(os.path.join(out_dir, "pdfs", f"{fid}.pdf"), "wb") as f:
        f.write(data)

    fixture = {
        "id": fid,
        "n_chunks": len(chunk_data),
        "global_top_keywords": top_kw,
        "ranked": [{"chunk": c["chunk"], "score": c["score"]} for c in ranked],
        "selected": selected,
        "topic": topic,
        "queries": queries,
        "candidates": trace.get("candidates", {}),
        "details": _trim_details(trace.get("details", {})),
        "videos": videos,
    }
    with open(os.path.join(out_dir, "fixtures", f"{fid}.json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1, ensure_ascii=False)
