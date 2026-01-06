import pandas as pd
import tqdm

dfs = []

MODELS = [
    'LLaMA3_1_70B_Instruct',
    'LLaMA3_3_70B_Instruct',
    'LLaMA3_Instruct',
    'MedAlpaca',
    'Meditron_70B',
    'OpenBioLLM_70B',
    'MedGemma_27B',
    
    'GPT_4o',
    'GPT_5',
    'Claude_Haiku_4_5'
]

for model in tqdm.tqdm(MODELS):
    llm_file_name = './data/results(paa)/paa_' + model + '_results.csv'
    df_llm = pd.read_csv(llm_file_name)
    df_llm = df_llm[['question', 'output_text']].rename(columns={'output_text': 'response'})
    dfs.append(
        (model, df_llm)
    )
    

orig_file_name = './data/paa_unbiased_eval_subset.csv'
df_orig = pd.read_csv(orig_file_name, index_col=0)

df_merged = df_orig.reset_index()
df_merged["row_num"] = df_merged.groupby("question").cumcount()
df_merged['response'] = None

for (model, df) in tqdm.tqdm(dfs):
    df["row_num"] = df.groupby("question").cumcount()
    df_merged = pd.merge(df_merged, df, on=['question', 'row_num'], suffixes=('', f'_{model}'))

df_merged = df_merged.drop(columns=["row_num", "response"])
df_merged.to_csv('./data/llm_results_wide.csv', index=False)
df_merged = pd.melt(
    df_merged,
    id_vars=['index','query_no','query','question_rank','question','click_history','explanation_gpt5','class_1_gpt5','class_2_gpt5','class_3_gpt5','explanation_gpt5mini','question_t-1','depth','class_1_t-1','class_1_prev_cnt','class_1_prop','class_2_t-1','class_2_prev_cnt','class_2_prop','class_3_t-1','class_3_prev_cnt','class_3_prop'],
    value_vars=[f'response_{model}' for model in MODELS],
    var_name='model',
    value_name='response'
)
df_merged['model'] = df_merged['model'].str.replace('response_', '')
df_merged.to_csv('./data/llm_results_long.csv')
