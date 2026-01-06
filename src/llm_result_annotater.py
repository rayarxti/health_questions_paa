import pandas as pd
import tqdm
from sklearn.metrics import cohen_kappa_score
from llm_as_a_judge_utils import annotate_sync, annotater_send_async, annotate_get_async, annotate_retrieve_async
import argparse
import sys
from typing import Optional

DEFAULT_FILENAME = './data/llm_results.csv'
DEFAULT_SUFFIX = '_pred'

def llm_response_annotate_sync(
    filename=None,
    suffix=None,
    testing=False
):
    if filename is None:
        filename = DEFAULT_FILENAME
    if suffix is None:
        suffix = DEFAULT_SUFFIX
    output_filename = filename.replace('.csv', suffix + '.csv')

    data = pd.read_csv(filename, index_col=0)
    data['reasoning'] = None
    data['correctness'] = None

    for i in tqdm.trange(len(data)):
        idx = data.index[i]
        query = data.loc[idx, 'question']
        eval_results = annotate_sync(query, input_type="response")
        data.loc[idx, 'correctness'] = eval_results[0]
        data.loc[idx, 'reasoning'] = eval_results[1]
    data.to_csv(output_filename)
    
    if testing:
        data['gold'] = data['correctness_gold']
        data['pred'] = data['correctness']
        print(pd.crosstab(data['gold'], data['pred']))
        print(cohen_kappa_score(data['gold'], data['pred']))

def llm_response_annotate_send_async(filename):
    data = pd.read_csv(filename, index_col=0)
    annotater_send_async(data, input_type="response")

def llm_response_annotate_get_async(batch_id):
    annotate_get_async(batch_id)

def llm_response_annotate_retrieve_async(
    file_id,
    error=None,
    filename=None,
    suffix=None,
    testing=False
):
    if filename is None:
        filename = DEFAULT_FILENAME
    if suffix is None:
        suffix = DEFAULT_SUFFIX
    output_filename = filename.replace('.csv', suffix + '.csv')
    
    results = annotate_retrieve_async(file_id, error_inspection=error)
    if results is not None:
        data = pd.read_csv(filename, index_col=0)
        data['reasoning'] = None
        data['correctness'] = None

        for i in tqdm.trange(len(data)):
            try:
                idx = data.index[i]
                if pd.isna(data.loc[idx, "response"]) or data.loc[idx, "response"].strip() == "":
                    data.loc[idx, 'correctness'] = False
                    data.loc[idx, 'reasoning'] = "Empty response"
                else:
                    eval_results = results[idx]
                    data.loc[idx, 'correctness'] = eval_results["correctness"]
                    data.loc[idx, 'reasoning'] = eval_results["explanation"]
            except KeyError:
                print(f"Warning: Missing result for index {idx}", file=sys.stderr)
        data['correctness'] = data['correctness'].astype(bool)
        data.to_csv(output_filename)
        if testing:
            data['gold'] = data['correctness_gold']
            data['pred'] = data['correctness']
            print(pd.crosstab(data['gold'], data['pred']))
            print(cohen_kappa_score(data['gold'], data['pred']))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that accepts filename (required), suffix (optional), and flags testing/sync."
    )
    parser.add_argument("--sync", action="store_true", help="Enable sync behavior. Defaults to False.")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode. Defaults to False.")
    parser.add_argument("--filename", type=str, nargs="?", help="The filename to process (required for --sync, --send and --retrieve without --error).")
    parser.add_argument("--suffix", type=str, nargs="?", help="Optional suffix.")
    async_group = parser.add_mutually_exclusive_group()
    async_group.add_argument("--send", action="store_true", help="(async-only) Send data/action.")
    async_group.add_argument("--get", action="store_true", help="(async-only) Get data/action.")
    async_group.add_argument("--retrieve", action="store_true", help="(async-only) Retrieve data/action.")
    parser.add_argument("--batch-id", type=str, nargs="?", help="Batch ID (required when using --get).")
    parser.add_argument("--file-id", type=str, nargs="?", help="File ID (required when using --retrieve).")
    parser.add_argument("--error", action="store_true", help="(retrieve-only) Inspect error file. Defaults to False.")

    args = parser.parse_args()
    sync: bool = args.sync
    testing: bool = args.testing
    filename: Optional[str] = args.filename
    suffix: Optional[str] = args.suffix
    send: bool = args.send
    get: bool = args.get
    retrieve: bool = args.retrieve
    batch_id: Optional[str] = args.batch_id
    file_id: Optional[str] = args.file_id
    error: bool = args.error
    if sync and (send or get or retrieve):
        parser.error("options --send, --get, --retrieve are only valid when not using --sync (async mode).")
    if (sync or send or (retrieve and not error)) and not filename:
        parser.error("--filename is required when using --sync, --send or --retrieve without --error.")
    if get and not batch_id:
        parser.error("--batch-id is required when using --get.")
    if retrieve and not file_id:
        parser.error("--file-id is required when using --retrieve.")
    if filename and not (sync or send or (retrieve and not error)):
        parser.error("--filename is only valid when using --sync, --send or --retrieve without --error.")
    if batch_id and not get:
        parser.error("--batch-id is only valid when using --get.")
    if file_id and not retrieve:
        parser.error("--file-id is only valid when using --retrieve.")
    if error and not retrieve:
        parser.error("--error is only valid when using --retrieve.")

    if sync:
        llm_response_annotate_sync(filename, suffix, testing)
    elif send:
        llm_response_annotate_send_async(filename)
    elif get:
        llm_response_annotate_get_async(batch_id)
    elif retrieve:
        llm_response_annotate_retrieve_async(file_id, error, filename, suffix, testing)