import logging
import os

from query_input import get_dedup_ranked_chunks
from hugging_face import generate_queries, document_topic
from pdf_Reader import first_page_text
from youtube_api import build_video_pool, YOUTUBE_API_KEY

logger = logging.getLogger(__name__)


def recommend(pdf_source):
    """Run the PDF -> video recommendation pipeline. pdf_source: path or file-like object."""
    try:
        ranked_chunks = get_dedup_ranked_chunks(pdf_source)
    except Exception as e:
        raise RuntimeError(f"chunking/dedup failed: {e}") from e

    try:
        top_kw = [kw["keyword"] for c in ranked_chunks for kw in c["keywords"]]
        topic = document_topic(first_page_text(pdf_source), top_kw)
        logger.info("document topic: %s", topic)
        queries = generate_queries(ranked_chunks, topic)  # one per chunk (eval pairs these with chunks)
    except Exception as e:
        raise RuntimeError(f"query generation failed: {e}") from e
    logger.info("generated %d youtube queries", len(queries))

    trace = {}
    try:
        # the plain subject ("HTTP load balancer") also gets searched: concept queries alone surface
        # obscure videos, the subject query brings in the popular overviews
        search_queries = list(dict.fromkeys([topic, *queries]))
        videos = build_video_pool(search_queries, YOUTUBE_API_KEY, max_results=5, trace=trace, topic=topic)
    except Exception as e:
        raise RuntimeError(f"youtube search failed: {e}") from e
    logger.info("found %d videos after filtering", len(videos))

    if os.environ.get("EVAL_RECORD_DIR"):  # opt-in: save this run as an eval fixture
        from eval.record import save_fixture
        save_fixture(os.environ["EVAL_RECORD_DIR"], pdf_source, ranked_chunks, queries, trace, videos, topic)

    return videos
