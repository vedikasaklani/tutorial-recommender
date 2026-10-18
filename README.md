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

```
EVAL_RECORD_DIR=eval python app.py   # then use the extension on a PDF: saves eval/fixtures/<id>.json and eval/pdfs/<id>.pdf
python -m eval.run --draft-gold      # drafts topics into eval/gold.json - edit them
python -m eval.run --export-judge    # pairs for an LLM judge; fill eval/judgments.json
python -m eval.run                   # report: topic coverage, diversity, query checks, judge score, filter replay
python -m eval.run --sweep           # chunk size / scoring / dedup threshold (needs local PDFs)
python -m eval.run --bench           # nlp() vs nlp.pipe timing
```

## Future Deliverables

- Rank videos by relevance to the query/chunk, not only view count.
- Replace averaged word vectors with sentence embeddings for chunk deduplication.
- Support recommendations from user-selected PDF content.
- Add relevant website recommendations alongside YouTube videos.

## Status

Working end to end (PDF -> chunks -> queries -> filtered top 5) behind a Flask API and a Chrome/Firefox extension.
