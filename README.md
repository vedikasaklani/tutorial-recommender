# PDF-to-Video Learning Recommender

A backend-oriented project that processes PDF content and recommends relevant YouTube learning resources.

## How to Run

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

Create a `.env` file in the project root with:

```
GROQ_API_KEY=your_groq_api_key
YOUTUBE_API_KEY=your_youtube_data_api_key
```

```
python app.py
```

This serves a JSON API at `http://127.0.0.1:5000`:

- `POST /recommend` — multipart form with a `pdf` file field, returns a JSON list of recommended videos.

The frontend is a browser extension, not this Flask app: [extension/](extension/) for Chrome (side panel), [extension-firefox/](extension-firefox/) for Firefox (sidebar). Load either unpacked (`chrome://extensions` → Load unpacked, or `about:debugging` → Load Temporary Add-on for Firefox), open a PDF tab, then click the toolbar icon.

## Current Pipeline

PDF  
↓  
Text Extraction  
↓  
Sentence Splitting  
↓  
Fixed-Size Chunking  
↓  
Keyword Extraction  
↓  
Keyword Cleaning & Lemmatization  
↓  
Chunk Ranking  
↓  
Semantic Chunk Deduplication  
↓  
Representative Chunks  
↓  
LLM Query Generation  
↓  
YouTube Search  
↓  
Deduplicated Video Pool  

## Current Implementation

### PDF Processing

PDF text is extracted using `pypdf` and processed with spaCy. The text is split into sentences and grouped into fixed-size chunks.

### Keyword Extraction

Keywords are extracted from each chunk and stored with metadata including:

- Keyword score
- Frequency
- Original chunk context

Low-scoring keywords are pruned, cleaned, and normalized.

### Chunk Ranking and Deduplication

Chunks are ranked using keyword importance. Highly similar ranked chunks are compared using vector similarity.

A greedy semantic deduplication approach retains the higher-ranked chunk while absorbing lower-ranked chunks that exceed a similarity threshold.

### Query Generation

Representative chunks and their keywords are sent to an LLM (`openai/gpt-oss-20b` on Groq's OpenAI-compatible API). The model generates concise, context-aware YouTube search queries.

### YouTube Integration

Generated queries are sent to the YouTube Data API. Results are pooled and deduplicated by video ID, then video details (duration, views, language) are fetched in batches. Shorts (<4 min), videos under 1,000 views and non-English videos are dropped, and the top 5 by view count are returned.

## Tech Stack

- Python, Flask (REST API)
- PyPDF, spaCy (`nlp.pipe` batching for all NLP stages)
- Groq LLM API, YouTube Data API
- NumPy / vector similarity
- Requests, Python Dotenv, Pytest

## Evaluation

A light, offline eval lives in [eval/](eval/). It replays recorded runs, so it never calls an API. See [topics.md](topics.md) for the concepts behind it.

### Structure

```
eval/
  run.py          CLI: report, --export-judge, --draft-gold, --sweep, --sweep-rank, --bench
  metrics.py      pure metric functions (topic coverage, diversity, query checks, filter replay, precision@5)
  record.py       saves one live pipeline run as a fixture (used when EVAL_RECORD_DIR is set)
  bench.py        one-off nlp() vs nlp.pipe timing; holds the legacy implementation as the baseline
  fixtures/       recorded runs, one <pdf id>.json each: selected chunks, queries, YouTube candidates, returned videos (committed, used by CI)
  pdfs/           the recorded PDFs (gitignored; needed only for --sweep and --bench)
  gold.json       hand-labelled ground truth per pdf id: topics, relevant_video_ids
  judgments.json  LLM-judge scores (1-5) per (chunk, query) pair, keyed by a hash
```

### What it measures

| Stage | Metric | Needs |
|---|---|---|
| Chunk selection | topic coverage (exact / soft), diversity | `gold.json` topics |
| Query generation | fallback rate, formatting artifacts, query-vs-chunk similarity, judge score | `judgments.json` for the judge score |
| Video ranking | filter replay, rejection counts, precision@5 | `gold.json` relevant_video_ids |

### Workflow

```
EVAL_RECORD_DIR=eval python app.py   # then use the extension on a PDF: saves eval/fixtures/<id>.json and eval/pdfs/<id>.pdf
python -m eval.run --draft-gold      # drafts topics into eval/gold.json - edit them
python -m eval.run --export-judge    # writes eval/to_judge.json; grade the pairs into eval/judgments.json
python -m eval.run                   # report: coverage, diversity, query checks, judge score, filter replay, p@5
python -m eval.run --sweep           # chunk size / scoring / dedup threshold (needs local PDFs)
python -m eval.run --sweep-rank      # video-ranking weights vs precision@5
python -m eval.run --bench           # nlp() vs nlp.pipe timing
```

On Windows PowerShell, set the variable first: `$env:EVAL_RECORD_DIR = "eval"; python app.py`. CI runs `python -m eval.run` on the committed fixtures.

## Future Deliverables

- Rank videos by relevance to the query/chunk, not only view count.
- Replace averaged word vectors with sentence embeddings for chunk deduplication.
- Support recommendations from user-selected PDF content.
- Add relevant website recommendations alongside YouTube videos.

## Status

Working end to end (PDF -> chunks -> queries -> filtered top 5) behind a Flask API and a Chrome/Firefox extension.
