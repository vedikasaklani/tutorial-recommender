"""One-off benchmark: per-call nlp() (the original code) vs batched nlp.pipe (current code).
The legacy functions are a verbatim copy of the pre-pipe implementation, kept only here."""
import statistics
import time

from pypdf import PdfReader

from pdf_Reader import nlp, extract_chunks, chunking
from query_input import rank_chunks, semantic_dedup


def legacy_extract_chunks(pdf_path, chunk_size=2):
    pdf = PdfReader(pdf_path)
    tokens = []
    for page in pdf.pages:
        tokens.extend(list(nlp(page.extract_text()).sents))
    chunks = chunking(tokens, chunk_size)
    chunk_data, global_keywords = [], {}
    for chunk in chunks:
        data = {"context": chunk, "meta": []}
        for word, freq, score in nlp(chunk)._.keywords:
            score = float(score)
            if score > 0.5:
                result = word.lower().strip(" ").strip(".").strip(",").strip("!")
                final = " ".join(t.lemma_ for t in nlp(result))
                if final in global_keywords:
                    global_keywords[final]["freq"] += freq
                    global_keywords[final]["score"] = max(global_keywords[final]["score"], score)
                else:
                    global_keywords[final] = {"score": score, "freq": freq}
                data["meta"].append({"keyword": final, "score": score, "freq": freq})
        chunk_data.append(data)
    return chunk_data, global_keywords


class _PerCallNlp:
    """Wraps nlp so .pipe degrades to one nlp() call per text, like the old semantic_dedup."""

    def pipe(self, texts, **kwargs):
        return (nlp(t) for t in texts)


def _time(fn, runs):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def run(pdf_path, runs=3):
    old_t, (old_data, _) = _time(lambda: legacy_extract_chunks(pdf_path), runs)
    new_t, (new_data, kw) = _time(lambda: extract_chunks(pdf_path), runs)
    ranked = rank_chunks(kw, new_data)
    old_d, _ = _time(lambda: semantic_dedup(ranked, _PerCallNlp()), runs)
    new_d, _ = _time(lambda: semantic_dedup(ranked, nlp), runs)
    return {
        "extract_old_s": old_t, "extract_new_s": new_t, "extract_speedup": old_t / new_t,
        "dedup_old_s": old_d, "dedup_new_s": new_d, "dedup_speedup": old_d / new_d if new_d else None,
        "identical_output": old_data == new_data,
        "n_chunks": len(new_data),
    }
