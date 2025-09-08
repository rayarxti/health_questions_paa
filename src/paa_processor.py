import pandas as pd
from tqdm import tqdm

FEATURE_LIST = ['class_1', 'class_2', 'class_3']

data = pd.read_csv("./data/paa_annotated.csv", index_col=0)

data['class_3'] = data['class_3'] + data['class_4']
data = data.drop(columns=['class_4'])

data['question_t-1'] = None
data['depth'] = 1
for feature in FEATURE_LIST:
    data[f"{feature}_t-1"] = None
    data[f"{feature}_prev_cnt"] = 0
    data[f"{feature}_prop"] = None

for idx in tqdm(data.index):
    query_no = data.loc[idx, 'query_no']
    click_history = eval(data.loc[idx, 'click_history'])
    last_question = click_history[-1] if click_history else None
    data.loc[idx, 'question_t-1'] = last_question
    if not pd.isna(last_question):
        last_question_slice = data.loc[idx - 1]
        data.loc[idx, 'depth'] = last_question_slice['depth'] + 1
        for feature in FEATURE_LIST:
            data.loc[idx, f"{feature}_t-1"] = last_question_slice[f"{feature}"]
            data.loc[idx, f"{feature}_prev_cnt"] = last_question_slice[f"{feature}_prev_cnt"] + last_question_slice[f"{feature}"]
            data.loc[idx, f"{feature}_prop"] = data.loc[idx, f"{feature}_prev_cnt"] / last_question_slice['depth']

data.to_csv(f"./data/paa_backtracked.csv", index=False)