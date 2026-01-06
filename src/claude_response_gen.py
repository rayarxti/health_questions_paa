import os
import sys
import dotenv
from openai import OpenAI
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import pandas as pd
import tqdm
import json
# Global Constants
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TEMPERATURE = 0.01
DEFAULT_ENDPOINT = "/v1/responses"

def annotater_send_async(
    data: pd.DataFrame,
):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    batch_jsonl = []
    for i in tqdm.trange(len(data)):
        idx = data.index[i]
        batch_jsonl.append(
            Request(
                custom_id="request-{}".format(idx),
                params=MessageCreateParamsNonStreaming(
                    model=DEFAULT_MODEL,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": data.loc[idx, 'question']},
                    ],
                )
            )
        )
    message_batch = client.messages.batches.create(requests=batch_jsonl)
    print("Created batch job with ID:", message_batch.id)

def annotate_get_async(
    batch_id,
):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message_batch = client.messages.batches.retrieve(batch_id)
    print(message_batch)

def annotate_retrieve_async(
    batch_id,
    data,
):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    results = {}
    results_batch = client.messages.batches.results(batch_id)
    for result in results_batch:
        if result.result.type == "succeeded":
            custom_id = int(result.custom_id.split('-')[1])
            results[custom_id] = result.result.message.content[0].text
        
    data['response'] = None

    for i in tqdm.trange(len(data)):
        try: 
            idx = data.index[i]
            data.loc[idx, 'response'] = results[idx]
        except KeyError:
            print(f"Warning: Missing result for index {idx}", file=sys.stderr)
        
    data.to_csv("./data/results(paa)/paa_Claude_Haiku_4_5_results.csv")