import logging

from dotenv import load_dotenv
import os, requests
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://router.huggingface.co/v1/chat/completions"

logger = logging.getLogger(__name__)


#making the query for youtube using gemma
def _build_prompt(chunk):
    prompt="Generate a relevant youtube query for given context and keywords\n"
    prompt+=f" Context: {chunk['chunk']}\n, Keywords:"
    for kw in chunk["keywords"]:
        prompt+=f" {kw["keyword"]}, "
    prompt+="\nReturn only the youtube query."
    return prompt


def generate_queries(ranked_chunks):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    queries=[]
    for i, chunk in enumerate(ranked_chunks):
        fallback = " ".join(kw["keyword"] for kw in chunk["keywords"])
        try:
            resp = requests.post(API_URL, headers=headers, json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _build_prompt(chunk)
                            },
                        ]
                    }
                ],
                "model": "google/gemma-4-31B-it:novita"
            }, timeout=90)
            response = resp.json()
            query = response["choices"][0]["message"]["content"] if "choices" in response else fallback
        except (requests.RequestException, ValueError, KeyError) as e:
            # ponytail: HF unreachable/slow/malformed -> plain keywords as the query
            logger.warning("query %d/%d failed (%s); falling back to keywords", i + 1, len(ranked_chunks), e)
            query = fallback
        logger.debug("query %d/%d: %s", i + 1, len(ranked_chunks), query)
        queries.append(query)
    return queries
