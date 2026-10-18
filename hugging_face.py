import logging

from dotenv import load_dotenv
import os, requests
load_dotenv()
# Groq (OpenAI-compatible). HF_TOKEN is accepted as a fallback name for the same key.
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

logger = logging.getLogger(__name__)


#making the query for youtube
def _build_prompt(chunk, topic=None):
    prompt="Write one YouTube search query (3-5 words) that a student would type to learn the main topic of this text. Name the concept itself, not details like table numbers or results.\n"
    if topic:
        prompt+=(f"The text comes from a document about: {topic}. Lead with the specific concept this passage explains, "
                 f"then add the subject so results stay on it, e.g. 'health checks in a load balancer'. "
                 f"Use common search terms a tutorial would use, 3-5 words, no filler like overview or tutorial, "
                 f"and do not just repeat the subject.\n")
    prompt+=f" Context: {chunk['chunk']}\n, Keywords:"
    for kw in chunk["keywords"]:
        prompt+=f" {kw["keyword"]}, "
    prompt+="\nReturn only the query, no quotes."
    return prompt


def _chat(prompt):
    resp = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json={
        "messages": [{"role": "user", "content": prompt}],
        "model": MODEL,
    }, timeout=90)
    return resp, resp.json()


def document_topic(first_page_text, keywords):
    """One short LLM call: what is this document about? Falls back to the top keywords."""
    fallback = " ".join(keywords[:3])
    try:
        resp, response = _chat(
            "Give the subject of the document that starts with the text below as a 2-4 word YouTube search phrase "
            "a student would type (example: 'HTTP load balancer'). Reply with only the phrase, no quotes.\n\n" + first_page_text[:1500])
        if "choices" in response:
            return response["choices"][0]["message"]["content"].strip() or fallback
        logger.warning("topic: LLM returned no choices (HTTP %s: %s)", resp.status_code, str(response)[:200])
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("topic failed (%s); using keywords", e)
    return fallback


def generate_queries(ranked_chunks, topic=None):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    queries=[]
    for i, chunk in enumerate(ranked_chunks):
        # unique words, in order: keywords overlap ("main focus", "main focus key") and repeats hurt search
        fallback = " ".join(dict.fromkeys(w for kw in chunk["keywords"] for w in kw["keyword"].split()))
        try:
            resp = requests.post(API_URL, headers=headers, json={
                "messages": [{"role": "user", "content": _build_prompt(chunk, topic)}],
                "model": MODEL,
            }, timeout=90)
            response = resp.json()
            if "choices" in response:
                query = response["choices"][0]["message"]["content"]
            else:
                # e.g. 402 out of credits, 429 rate limit: say so instead of silently degrading
                logger.warning("query %d/%d: LLM returned no choices (HTTP %s: %s); using keywords",
                               i + 1, len(ranked_chunks), getattr(resp, "status_code", "?"), str(response)[:200])
                query = fallback
        except (requests.RequestException, ValueError, KeyError) as e:
            # ponytail: LLM unreachable/slow/malformed -> plain keywords as the query
            logger.warning("query %d/%d failed (%s); falling back to keywords", i + 1, len(ranked_chunks), e)
            query = fallback
        logger.debug("query %d/%d: %s", i + 1, len(ranked_chunks), query)
        queries.append(query)
    return queries
