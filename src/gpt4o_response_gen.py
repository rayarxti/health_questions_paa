import os
import sys
import dotenv
from openai import OpenAI
from anthropic import Anthropic
import pandas as pd
import tqdm
import json
# Global Constants
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
DEFAULT_MODEL = "gpt-5-2025-08-07"
DEFAULT_TEMPERATURE = 0.01
DEFAULT_ENDPOINT = "/v1/responses"

def annotater_send_async(
    data: pd.DataFrame,
):
    batch_jsonl = []
    batch_jsonl_filename = "./src/batchinput.jsonl"
    for i in tqdm.trange(len(data)):
        idx = data.index[i]
        batch_jsonl.append(
            {
                "custom_id": "request-{}".format(idx),
                "method": "POST",
                "url": DEFAULT_ENDPOINT,
                "body": {
                    "model": DEFAULT_MODEL,
                    "input": [
                        {"role": "user", "content": data.loc[idx, 'question']},
                    ],
                    # "temperature": DEFAULT_TEMPERATURE,
                }
            }
        )
    jsonl_str = "\n".join([json.dumps(item) for item in batch_jsonl])
    with open(batch_jsonl_filename, "w") as f:
        f.write(jsonl_str)
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    batch_input_file = client.files.create(
        file=open(batch_jsonl_filename, "rb"),
        purpose="batch"
    )
    print("Created batch input file with ID:", batch_input_file.id)
    batch_input_file_id = batch_input_file.id
    a = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=DEFAULT_ENDPOINT,
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    print("Created batch job with ID:", a.id)

def annotate_retrieve_async(
    file_id,
    data,
    error_inspection=False,
):
    client = OpenAI(api_key=OPENAI_API_KEY)
    file = client.files.content(file_id)
    if error_inspection:
        print(file.text)
        return None
    else:
        results = {}
        for line in file.text.split("\n"):
            if line:
                returned_json = json.loads(line)
                custom_id = int(returned_json['custom_id'].split('-')[1])
                output_list = returned_json["response"]["body"]["output"]
                for output in output_list:
                    if output["type"] == "message":
                        results[custom_id] = output["content"][0]["text"]
                        break
                else:
                    raise Exception("No message type output found in the response")
        
        data['response'] = None

        for i in tqdm.trange(len(data)):
            try: 
                idx = data.index[i]
                data.loc[idx, 'response'] = results[idx]
            except KeyError:
                print(f"Warning: Missing result for index {idx}", file=sys.stderr)
        
        data.to_csv("./data/results(paa)/paa_GPT_5_results.csv")