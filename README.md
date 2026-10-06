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
HF_TOKEN=your_hugging_face_token
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

Representative chunks and their keywords are sent to a Hugging Face model using HTTP API requests. The model generates concise, context-aware YouTube search queries.

### YouTube Integration

Generated queries are sent to the YouTube Data API. Search results are collected into a shared candidate pool and deduplicated using video IDs.

Each candidate currently stores:

- Video ID
- Title
- Description
- YouTube link

## Tech Stack

- Python
- PyPDF
- spaCy
- Hugging Face API
- YouTube Data API
- NumPy / Vector Similarity
- Requests
- Python Dotenv

## Future Deliverables

- Retrieve video statistics such as views, likes, and comments.
- Filter and rank candidate videos based on relevance and engagement.
- Return the final top 5 recommendations.
- Integrate the pipeline into a FastAPI backend.
- Support recommendations from user-selected PDF content.
- Add relevant website recommendations alongside YouTube videos.

## Status

**Work in Progress**

Current focus:

Video Candidate Pool  
↓  
Fetch Video Statistics  
↓  
Filter and Rank Videos  
↓  
Return Top 5 Recommendations
