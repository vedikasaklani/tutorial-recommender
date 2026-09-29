from query_input import  dedup_ranked as ranked_chunks
from dotenv import load_dotenv
import os, requests
load_dotenv() 
HF_TOKEN = os.getenv("HF_TOKEN") 

prompts=[]

#making the query for youtube using gemma 
for chunk in ranked_chunks:
    prompt="Generate a relevant youtube query for given context and keywords\n"
    prompt+=f" Context: {chunk['chunk']}\n, Keywords:"
    for kw in chunk["keywords"]:
        prompt+=f" {kw["keyword"]}, "
    prompt+="\nReturn only the youtube query."
    prompts.append(prompt)


API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}
queries=[]
for prompt in prompts:
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    response = query({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }
        ],
        "model": "google/gemma-4-31B-it:novita"
    })
    if "choices" in response:
        queries.append(response["choices"][0]["message"]["content"])
    else:  # ponytail: LLM unavailable (e.g. out of credits) -> plain keywords as the query
        queries.append(" ".join(kw["keyword"] for kw in ranked_chunks[len(queries)]["keywords"]))

