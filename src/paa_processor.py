import pandas as pd
from tqdm import tqdm

FEATURE_LIST = ['class_1', 'class_2', 'class_3']

data1 = pd.read_csv("./data/paa_unbiased_pred_gpt5.csv", index_col=0).rename(columns={'explanation': 'explanation_gpt5', 'class_1': 'class_1_gpt5', 'class_2': 'class_2_gpt5', 'class_3': 'class_3_gpt5'})
data2 = pd.read_csv("./data/paa_unbiased_pred_gpt5mini.csv", index_col=0).rename(columns={'explanation': 'explanation_gpt5mini', 'class_1': 'class_1_gpt5mini', 'class_2': 'class_2_gpt5mini', 'class_3': 'class_3_gpt5mini'})
data = pd.concat([data1, data2[['explanation_gpt5mini', 'class_1_gpt5mini', 'class_2_gpt5mini', 'class_3_gpt5mini']]], axis=1)
data['aligned'] = data.apply(
    lambda row: 1 if (row['class_1_gpt5'] == row['class_1_gpt5mini'] and
                      row['class_2_gpt5'] == row['class_2_gpt5mini'] and
                      row['class_3_gpt5'] == row['class_3_gpt5mini']) else 0,
    axis=1
)

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
            data.loc[idx, f"{feature}_t-1"] = last_question_slice[f"{feature}_gpt5"]
            data.loc[idx, f"{feature}_prev_cnt"] = last_question_slice[f"{feature}_prev_cnt"] + last_question_slice[f"{feature}_gpt5"]
            data.loc[idx, f"{feature}_prop"] = data.loc[idx, f"{feature}_prev_cnt"] / last_question_slice['depth']
        
data.to_csv(f"./data/paa_unbiased_backtracked.csv")

data = data[(data['aligned'] == 1) & (data['class_1_gpt5'] + data['class_2_gpt5'] > 0)]
data.to_csv(f"./data/paa_unbiased_eval_subset.csv")
