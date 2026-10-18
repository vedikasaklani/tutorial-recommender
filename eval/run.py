"""Light eval for the PDF -> video pipeline. Offline: reads recorded fixtures, never calls an API.

  python -m eval.run                  report on every eval/fixtures/*.json
  python -m eval.run --export-judge   write eval/to_judge.json (pairs needing a judge score)
  python -m eval.run --draft-gold     add draft topics to eval/gold.json for you to edit
  python -m eval.run --sweep          parameter sweep (needs the local PDFs in eval/pdfs/)
  python -m eval.run --sweep-rank     video ranking weights vs precision@5 (needs gold relevant_video_ids)
  python -m eval.run --bench          nlp() vs nlp.pipe timing (needs local PDFs)
"""
import argparse
import glob
import json
import os

from eval import metrics as m

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD, JUDGMENTS = os.path.join(HERE, "gold.json"), os.path.join(HERE, "judgments.json")


def _fmt(v):
    return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def _table(rows):
    if not rows:
        return
    cols = list(rows[0])
    widths = [max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(_fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


def _fixtures():
    return [m.load_json(p, {}) for p in sorted(glob.glob(os.path.join(HERE, "fixtures", "*.json")))]


def report():
    gold, judgments = m.load_json(GOLD, {}), m.load_json(JUDGMENTS, {})
    rows, details, todo_all = [], {}, []
    for fx in _fixtures():
        fid, topics = fx["id"], gold.get(fx["id"], {}).get("topics", [])
        texts = [c["chunk"] for c in fx["selected"]]
        cov = m.coverage(topics, texts)
        qs = m.query_stats(fx["selected"], fx["queries"])
        judge, todo = m.judge_scores(fx["selected"], fx["queries"], judgments)
        todo_all += todo
        flt = m.replay_filter(fx)
        relevant = gold.get(fid, {}).get("relevant_video_ids")
        rows.append({
            "pdf": fid, "chunks": len(texts), "cov_exact": cov["exact"], "cov_soft": cov["soft"],
            "diversity": m.diversity(texts), "q_fallback": qs.get("fallback_rate"),
            "q_artifacts": qs.get("artifact_rate"), "q_sim": qs.get("mean_sim"), "judge_1to5": judge,
            "videos": len(fx["videos"]),
            "filter_replay_ok": flt and flt["replay_matches"],
            "p@5": m.precision_at_k([v["video_id"] for v in fx["videos"]], relevant) if relevant else None,
        })
        details[fid] = {"missing_topics": cov["missing"], "filter": flt, "unjudged": len(todo)}
    if not rows:
        print("no fixtures in eval/fixtures/ - record one first (see topics.md)")
        return
    _table(rows)
    print()
    for fid, d in details.items():
        print(f"{fid}: missing topics={d['missing_topics']} unjudged pairs={d['unjudged']} filter={d['filter']}")
    with open(os.path.join(HERE, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "details": details}, f, indent=1)
    if todo_all:
        print(f"\n{len(todo_all)} query pairs have no judge score: run --export-judge and ask Claude to grade them")


def export_judge():
    judgments = m.load_json(JUDGMENTS, {})
    todo = []
    for fx in _fixtures():
        _, t = m.judge_scores(fx["selected"], fx["queries"], judgments)
        todo += [{"key": k, "chunk": c, "query": q} for k, c, q in t]
    path = os.path.join(HERE, "to_judge.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(todo, f, indent=1, ensure_ascii=False)
    print(f"{len(todo)} pairs -> {path}")


def draft_gold():
    gold = m.load_json(GOLD, {})
    for fx in _fixtures():
        entry = gold.setdefault(fx["id"], {})
        if not entry.get("topics"):
            entry["topics"] = fx["global_top_keywords"][:12]
            entry["_note"] = "DRAFT from the PDF's top keywords - edit to the topics a student must see."
    with open(GOLD, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=1)
    print(f"drafted topics for {len(gold)} pdfs -> {GOLD}")


def sweep():
    from pdf_Reader import extract_chunks, nlp
    from query_input import rank_chunks, semantic_dedup
    gold = m.load_json(GOLD, {})
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "pdfs", "*.pdf"))):
        fid = os.path.basename(path)[:-4]
        topics = gold.get(fid, {}).get("topics", [])
        for size in (2, 3, 4):
            chunk_data, kw = extract_chunks(path, size)
            for scoring in ("product", "sum", "coverage"):
                ranked = rank_chunks(kw, chunk_data, scoring)
                for thr in (0.9, 0.95, 0.98):
                    texts = [c["chunk"] for c in semantic_dedup(ranked, nlp, thr)]
                    cov = m.coverage(topics, texts)
                    rows.append({"pdf": fid, "chunk_size": size, "scoring": scoring, "dedup": thr,
                                 "selected": len(texts), "cov_exact": cov["exact"], "cov_soft": cov["soft"],
                                 "diversity": m.diversity(texts)})
    _table(rows) if rows else print("no PDFs in eval/pdfs/")


def sweep_rank():
    """Video-ranking weights vs precision@5 and popularity, replayed from recorded candidates."""
    import statistics
    from youtube_api import filter_and_rank
    gold = m.load_json(GOLD, {})
    rows = []
    for fx in _fixtures():
        relevant = gold.get(fx["id"], {}).get("relevant_video_ids")
        if not relevant or not any("ranks" in c for c in fx["candidates"].values()):
            print(f"{fx['id']}: needs relevant_video_ids in gold.json and a fixture recorded after rank tracking")
            continue
        for w_search in (0, 1.0):
            for w_topic in (0, 0.5, 1.0, 2.0):
                for w_views in (0, 0.5, 1.0, 2.0, 3.0):
                    top = filter_and_rank(fx["candidates"], fx["details"], 5, fx.get("topic"), w_search, w_topic, w_views)
                    rows.append({"pdf": fx["id"], "w_search": w_search, "w_topic": w_topic, "w_views": w_views,
                                 "p@5": m.precision_at_k([v["video_id"] for v in top], set(relevant)),
                                 "median_views": int(statistics.median(v["views"] for v in top)) if top else None})
    rows.sort(key=lambda r: (r["p@5"] or 0, -(r["median_views"] or 0)), reverse=True)
    _table(rows[:int(os.environ.get("TOP", 20))])


def bench():
    from eval import bench as b
    for path in sorted(glob.glob(os.path.join(HERE, "pdfs", "*.pdf"))) or ["dbms_mod_2.pdf"]:
        print(path, b.run(path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ("export-judge", "draft-gold", "sweep", "sweep-rank", "bench"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if a.export_judge:
        export_judge()
    elif a.draft_gold:
        draft_gold()
    elif a.sweep_rank:
        sweep_rank()
    elif a.sweep:
        sweep()
    elif a.bench:
        bench()
    else:
        report()
