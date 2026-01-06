import pandas as pd

data = pd.read_csv('./data/llm_results_long_pred.csv', index_col=0)

print(data.groupby(['model', 'class_1_gpt5', 'class_2_gpt5'])['correctness'].mean())