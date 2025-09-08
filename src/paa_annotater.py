import pandas as pd
import tqdm
from gpt4_utils import evaluate

if __name__ == "__main__":
    data = pd.read_csv('./data/paa_annotated.csv')

    data['explanation'] = None
    data['class_1'] = None
    data['class_2'] = None
    data['class_3'] = None
    data['class_4'] = None

    for i in tqdm.trange(len(data)):
        idx = data.index[i]
        query = data.loc[idx, 'question']
        eval_results = evaluate(query)
        data.loc[idx, 'class_1'] = eval_results[0][0]
        data.loc[idx, 'class_2'] = eval_results[0][1]
        data.loc[idx, 'class_3'] = eval_results[0][2]
        data.loc[idx, 'class_4'] = eval_results[0][3]
        data.loc[idx, 'explanation'] = eval_results[1]
    data.to_csv(f'./data/paa_annotated.csv')