import logging

from query_input import get_dedup_ranked_chunks
from hugging_face import generate_queries
from youtube_api import build_video_pool, YOUTUBE_API_KEY

logger = logging.getLogger(__name__)


def recommend(pdf_source):
    """Run the PDF -> video recommendation pipeline. pdf_source: path or file-like object."""
    try:
        ranked_chunks = get_dedup_ranked_chunks(pdf_source)
    except Exception as e:
        raise RuntimeError(f"chunking/dedup failed: {e}") from e

    try:
        queries = generate_queries(ranked_chunks)
    except Exception as e:
        raise RuntimeError(f"query generation failed: {e}") from e
    logger.info("generated %d youtube queries", len(queries))

    try:
        videos = build_video_pool(queries, YOUTUBE_API_KEY, max_results=5)
    except Exception as e:
        raise RuntimeError(f"youtube search failed: {e}") from e
    logger.info("found %d videos after filtering", len(videos))

    return videos
