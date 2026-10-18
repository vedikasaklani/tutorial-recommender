# Topics behind the nlp.pipe + eval work

Read this before an interview. Each section: the idea, where it is in the code, and what to say.

## 1. `nlp()` vs `nlp.pipe()`

- `nlp(text)` runs the whole spaCy pipeline on one text. `nlp.pipe(texts)` runs it on a stream in **batches**, so the neural parts (tok2vec, tagger, parser) process many docs per call instead of paying set-up cost per text.
- The old code called `nlp()` once per page, once per chunk, once **per keyword** (just to get a lemma), and once per chunk again in dedup.
- Now (`pdf_Reader.py`, `query_input.py`): pages, chunks, keyword lemmas and dedup vectors each go through one `nlp.pipe` pass.
- **Measured** on `dbms_mod_2.pdf` (151 chunks, `python -m eval.run --bench`, median of 3): extraction 2.19s -> 1.28s (1.7x), dedup vectors 0.097s -> 0.028s (3.5x), output byte-identical to the old code.

## 2. Skipping components with `disable=`

- A spaCy pipeline is a list of components: tok2vec, tagger, parser, ner, attribute_ruler, lemmatizer, plus our custom `keyword_extractor`.
- Sentence splitting needs the parser but not keyword extraction, so pages disable `keyword_extractor`.
- Lemmas and document vectors need none of parser, ner, keyword_extractor (`LIGHT` in `pdf_Reader.py`). Doing less work per text is where the dedup speedup comes from.
- Why it is safe: `Doc.vector` is the mean of static word vectors from the model's vocab, so disabling pipeline components does not change it.

## 3. Unique-then-batch (memoization)

- The same keyword ("database") appears in many chunks. Old code lemmatized it every time.
- Now we collect unique cleaned keywords, lemmatize each once in one batch, and look them up in a dict. Less work and fits `nlp.pipe` naturally.

## 4. What is being evaluated, and why offline

The pipeline has three stages that can each be good or bad:

| Stage | Question | Metric (in `eval/metrics.py`) |
|---|---|---|
| Chunk selection | Do the kept chunks cover what the PDF is about? | topic coverage, diversity |
| Query generation | Is the LLM query on topic and clean? | fallback rate, artifact rate, query-vs-chunk similarity, judge score |
| Video filtering | Do the filter and sort rules behave? | filter replay, rejection counts, precision@5 |

**Why offline:** the YouTube API has a daily quota and the HF model is nondeterministic. So a live run is *recorded once* into a fixture, and the eval replays it for free, deterministically, in CI.

## 5. Record / replay fixtures

- `EVAL_RECORD_DIR=eval python app.py`, then use the extension on a PDF. `pipeline.recommend` saves `eval/fixtures/<id>.json` (selected chunks, queries, YouTube candidates and details, returned videos) and the PDF to `eval/pdfs/<id>.pdf` (gitignored).
- To make filtering replayable, `build_video_pool` was split: network part, then pure `filter_and_rank(candidates, details)`. Pure functions are easy to test and replay.
- The `trace` argument is a small hook so the recorder can see raw API data without changing return values.

## 6. Topic coverage (a recall-style metric)

- Gold labels: a list of topics a student must see (`eval/gold.json`). `--draft-gold` pre-fills it from the PDF's top global keywords. **Edit it by hand**, because a draft built from the system's own keywords is partly circular.
- **Exact match:** every content lemma of the topic appears in a single selected chunk (lemmatized, stop words removed, so "normalizing tables" matches "normalization table").
- **Soft match:** exact, or vector similarity >= 0.7.
- Coverage = matched topics / all topics. It is recall: did we miss anything important?
- Diversity = mean pairwise cosine between selected chunks (lower is more varied).

## 7. LLM-as-judge

- Embedding similarity says "same words/topic", not "would this query find a good tutorial". A judge reads both and scores 1-5.
- Setup here: `--export-judge` writes `eval/to_judge.json`; Claude grades each (chunk, query) pair against a fixed rubric and writes `eval/judgments.json`, keyed by a hash of chunk + query. The eval itself stays offline and just reads the file.
- Why not the HF model: it **wrote** the queries. Judging your own output is biased (self-preference).
- Known weaknesses: judge scores are noisy and rubric-dependent, so keep the rubric fixed and read the written reasons. Re-judge only the pairs whose hash changed.

## 8. Parameter sweep

`--sweep` tries chunk size (2/3/4) x scoring (`product`/`sum`) x dedup threshold (0.8/0.85/0.9) and reports coverage, selected count and diversity. It reports only; changing defaults is your decision. Needs the local PDFs because changing chunk size means re-chunking.

## 9. Two issues the eval already points at

1. **`rank_chunks` multiplies by every keyword's global score.** Scores are <= 1, so more keywords can mean a *lower* chunk score. `scoring="sum"` is the alternative (`rank_chunks(..., scoring="sum")`). In a quick smoke run on one PDF, `sum` gave higher coverage than `product` (0.42 vs 0.25 at chunk size 2). That was a single PDF with auto-drafted topics, so treat it as a lead, not a result.
2. **Dedup at 0.9 removes most chunks.** The same smoke run kept only 4 of the top 10. Averaged word vectors from `en_core_web_md` give high cosine between almost any two passages from one document, so a fixed threshold is blunt. Sentence embeddings would separate topics better.

## 9b. What the eval changed (the story to tell)

| Problem seen | How the eval showed it | Fix | Result on os_report.pdf |
|---|---|---|---|
| Off-topic videos | topic coverage 0.17; only 4 of 10 chunks survived dedup, some were table headers | greedy **coverage selection** (`_coverage_select`), min 20 words, chunk size 3, dedup 0.98 | coverage 0.83 exact / 0.92 soft |
| LLM silently failing | `q_fallback` = 1.00 (every query was the raw keyword fallback) | log LLM errors; HF returned 402 (no credits), so switched to Groq `gpt-oss-20b` | `q_fallback` 0.00, judge mean 3.38/5 |
| Generic results ("design patterns", WordPress 503) | queries did not say what the document was about | one extra LLM call, `document_topic`, anchors every query: concept first, then 1-2 subject words | queries like "critical section backend load balancer" |
| Popular but irrelevant ranking | precision@5 stayed 0.4 for every views/topic weighting | **fused score**: YouTube's rank per query (summed over queries) + topic overlap + log(views); `--sweep-rank` tunes the weights | precision@5 0.40 -> 1.00 (weights unchanged after `--sweep-rank`; with no YouTube-rank signal it falls to 0.6) |

Key lesson: once every query contains the same subject words ("load balancer"), topic overlap cannot separate videos any more. The useful signal was YouTube's own per-query order, which the old code discarded by re-sorting everything by views.

## 9c. Rank fusion in one paragraph

Each video gets `W_SEARCH * sum(1/(rank+1) over queries) + W_TOPIC * topic_overlap + W_VIEWS * log10(views)/7`. A video ranked #1 for two different concept queries beats a 900k-view video that appeared once at #5. `log10` stops a 10M-view video from being worth 1000x a 10k one. The weights in `youtube_api.py` are hand-picked; `python -m eval.run --sweep-rank` replays recorded candidates against `relevant_video_ids` in `eval/gold.json` to tune them.

## 10. Honest limitations (say these first)

- Only the recorded PDFs are measured; a small gold set means small samples.
- View count is still the video ranking signal, not relevance. precision@5 only works if you add `relevant_video_ids` to the gold file.
- The judge is Claude grading in-session, so it is not automatic in CI.
- `relevant_video_ids` were labelled from titles only, by Claude, after seeing the results, so they are biased toward what the system found. Treat p@5 as a regression signal, not a benchmark.
- Everything is tuned on one PDF; record 2-3 more before trusting the defaults.

## 11. Likely interview questions

- *Why is `nlp.pipe` faster?* Batching amortizes per-call overhead and lets the neural components process many docs together; plus `disable=` skips unneeded components.
- *How do you know the refactor did not change behavior?* `--bench` compares old and new output for equality (identical on the sample PDF).
- *How do you evaluate an LLM-generated query?* Cheap label-free checks (fallback rate, artifacts, similarity) plus a judge with a fixed rubric from a different model than the generator.
- *Why record fixtures instead of calling the APIs?* Cost, quota, nondeterminism and CI.
- *What would you improve first?* Sentence embeddings for dedup, relevance-based video ranking, and a bigger labelled set.

## 12. Resume line (once you have your own numbers)

"Batched all spaCy stages with `nlp.pipe` (1.7x faster extraction, 3.5x faster dedup, identical output) and built an offline eval harness with recorded fixtures, topic-coverage metrics and an LLM-judge workflow."

## 13. Before / after on os_report.pdf (same PDF, same eval)

| Metric | Before (original pipeline) | After |
|---|---|---|
| Topic coverage (exact / soft) | 0.17 / 0.25 | 0.83 / 0.92 |
| Queries that fell back to raw keywords | 100% (LLM out of credits, unnoticed) | 0% |
| Judge mean (1-5) | 2.75 (4 old pairs) | 3.62 (8 pairs) |
| precision@5 of returned videos | 0.40 | 1.00 |
| Extraction time (dbms_mod_2.pdf) | 2.19s | 1.28s (nlp.pipe) |

Caveats to say out loud: one PDF; topics, judge scores and relevant videos were all labelled by Claude (videos from titles only, after seeing results), so the numbers are optimistic. They show direction and catch regressions; they are not a benchmark.
