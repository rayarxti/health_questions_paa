import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import statsmodels.api as sm

global data
data = pd.read_csv(f"./data/paa_unbiased_backtracked.csv")

for feature in ['class_1', 'class_2', 'class_3']:
    data[feature] = data[f"{feature}_gpt5"]

def scale_size(arr):
    return (np.log(arr) + 1) * 80

FEATURE_LIST_LIST = [
    ['class_1', 'class_2'],
    ['class_1'],
    ['class_2'],
]
FEATURE_ALIASES = {'class_1': 'A', 'class_2': 'B'}
for FEATURE_LIST in FEATURE_LIST_LIST:
    FEATURE_NAME = '_'.join(FEATURE_LIST)
    data[FEATURE_NAME] = data[FEATURE_LIST].sum(axis=1)
    data[f'{FEATURE_NAME}_t-1'] = data[[f'{feature}_t-1' for feature in FEATURE_LIST]].sum(axis=1)
    data[f'{FEATURE_NAME}_prev_cnt'] = data[[f'{feature}_prev_cnt' for feature in FEATURE_LIST]].sum(axis=1)
    data[f'{FEATURE_NAME}_prop'] = data[f'{FEATURE_NAME}_prev_cnt'] / (data['depth'] - 1)

N_BOOTSTRAP = 100

def prop_comparison():
    for FEATURE_LIST in FEATURE_LIST_LIST:
        FEATURE_NAME = '_'.join(FEATURE_LIST)
        
        n_obs = data.groupby([f'{FEATURE_NAME}_t-1'])[FEATURE_NAME].count().sort_index(ascending=False)
        success_counts = data.groupby([f'{FEATURE_NAME}_t-1'])[FEATURE_NAME].sum().sort_index(ascending=False)
        success_props = success_counts / n_obs
        stat, pval = proportions_ztest(success_counts, n_obs)
        print(f"Type{'s' if len(FEATURE_LIST) > 1 else ''} " +
              " + ".join([FEATURE_ALIASES[feature] for feature in FEATURE_LIST]) +
              " & " +
              " & ".join(f"{p * 100:.2f}\\%" for p in success_props) +
              f" & {stat:.3f}" +
              ''. join([f" & {pval * 3 / rank :.3e}" for rank in range(1, 4)]) +
              "\\\\")

def prop_time_analysis():
    global data
    data = data[~pd.isna(data['depth'])]
    # Define global color scale
    vmin, vmax = 1, 0

    # Create figure and layout
    plt.rc('font', size=20)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 9, figure=fig)

    # Subplots
    axes = [
        fig.add_subplot(gs[0, 0:8]),
        fig.add_subplot(gs[1, 0:4]),
        fig.add_subplot(gs[1, 4:8])
    ]
    scs = []

    for idx, FEATURE_LIST in enumerate(FEATURE_LIST_LIST):
        FEATURE_NAME = '_'.join(FEATURE_LIST)        
        df_1 = data.groupby([f'{FEATURE_NAME}_prop', 'depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
        vmin = min(vmin, df_1["count"].min())
        vmax = max(vmax, df_1["count"].max())

    for idx, FEATURE_LIST in enumerate(FEATURE_LIST_LIST):
        ax = axes[idx]
        FEATURE_NAME = '_'.join(FEATURE_LIST)
        df = data[~pd.isna(data[f'{FEATURE_NAME}_prop'])]
        df_1 = df.groupby([f'{FEATURE_NAME}_prop', 'depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
        df_1["size_scaled"] = scale_size(df_1["count"])

        X = sm.add_constant(df[f"{FEATURE_NAME}_prop"])  # add intercept
        y = df[FEATURE_NAME]

        # Fit logistic regression
        logit_model = sm.Logit(y, X).fit(disp=0)
        print(f"Type{'s' if len(FEATURE_LIST) > 1 else ''} " +
              " + ".join([FEATURE_ALIASES[feature] for feature in FEATURE_LIST]) +
              ": OR 95\\% CI$=$[$" +
              "$,$".join(f"{x:.2f}" for x in np.exp(logit_model.conf_int().iloc[1])) +
              "$], FDR-adjusted p-value$=" +
              ' '. join([f"{logit_model.pvalues.iloc[1] * 3 / rank :.3e}" for rank in range(1, 4)]) +
              "$; ")

        # Generate predictions
        x_pred = np.linspace(0, 1, 200)
        X_new = pd.DataFrame({
            "const": 1.0,
            f"{FEATURE_NAME}_prop": x_pred
        })
        pred = logit_model.get_prediction(X_new)

        # Summary frame with CI
        pred_summary = pred.summary_frame(alpha=0.05)  # 95% CI

        # Probability predictions and CI
        mean_curve = pred_summary["predicted"]           # fitted probabilities
        lower_curve = pred_summary["ci_lower"]   # lower CI
        upper_curve = pred_summary["ci_upper"]

        # --- Plot ---
        sc = ax.scatter(
            df_1[f'{FEATURE_NAME}_prop'], df_1["mean"],
            s=df_1["size_scaled"], c=df_1["count"],
            cmap="viridis",
            alpha=0.5,
            edgecolor="k",
            vmin=vmin, vmax=vmax
        )
        ax.set_ylim(-0.05, 1.05)
        ax.plot(x_pred, mean_curve, color="red")
        ax.fill_between(x_pred, lower_curve, upper_curve, color="red", alpha=0.2)
        scs.append(sc)
        
    # Add one shared colorbar (using one of the scatter plots)
    cax = fig.add_subplot(gs[:, 8])  # spans all rows in the last column
    cbar = fig.colorbar(scs[0], cax=cax)
    cbar.set_label("Sample size of data point")
    fig.supxlabel("Proportion of incorrect history")
    fig.supylabel("Proportion of incorrect current question")
    plt.tight_layout()
    plt.show()

def prop_depth_analysis():
    global data
    data = data[~pd.isna(data['depth'])]
    data['depth'] = data['depth'].astype(int) - 1
    # Define global color scale
    vmin, vmax = 1, 0

    # Create figure and layout
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 9, figure=fig)

    # Subplots
    axes = [
        fig.add_subplot(gs[0, 0:8]),
        fig.add_subplot(gs[1, 0:4]),
        fig.add_subplot(gs[1, 4:8])
    ]
    scs = []

    for idx, FEATURE_LIST in enumerate(FEATURE_LIST_LIST):
        FEATURE_NAME = '_'.join(FEATURE_LIST)        
        df_1 = data.groupby(['depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
        vmin = min(vmin, df_1["count"].min())
        vmax = max(vmax, df_1["count"].max())

    for idx, FEATURE_LIST in enumerate(FEATURE_LIST_LIST):
        ax = axes[idx]
        FEATURE_NAME = '_'.join(FEATURE_LIST)
        df = data
        df_1 = df.groupby(['depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
        df_1["size_scaled"] = scale_size(df_1["count"])
        
        X = sm.add_constant(df["depth"])  # add intercept
        y = df[FEATURE_NAME]

        # Fit logistic regression
        logit_model = sm.Logit(y, X).fit(disp=0)
        print(f"Type{'s' if len(FEATURE_LIST) > 1 else ''} " +
              " + ".join([FEATURE_ALIASES[feature] for feature in FEATURE_LIST]) +
              ": OR 95\\% CI$=$[$" +
              "$,$".join(f"{x:.2f}" for x in np.exp(logit_model.conf_int().iloc[1])) +
              "$], FDR-adjusted p-value$=" +
              ' '. join([f"{logit_model.pvalues.iloc[1] * 3 / rank :.3e}" for rank in range(1, 4)]) +
              "$; ")
        
        # Generate predictions
        x_pred = np.linspace(0, 9, 200)
        X_new = pd.DataFrame({
            "const": 1.0,
            "depth": x_pred
        })
        pred = logit_model.get_prediction(X_new)

        # Summary frame with CI
        pred_summary = pred.summary_frame(alpha=0.05)  # 95% CI

        # Probability predictions and CI
        mean_curve = pred_summary["predicted"]           # fitted probabilities
        lower_curve = pred_summary["ci_lower"]   # lower CI
        upper_curve = pred_summary["ci_upper"]

        # --- Plot ---
        sc = ax.scatter(
            df_1['depth'], df_1["mean"],
            s=df_1["size_scaled"], c=df_1["count"],
            cmap="viridis",
            alpha=0.5,
            edgecolor="k",
            vmin=vmin, vmax=vmax
        )
        ax.set_ylim(-0.01, 0.35)
        ax.plot(x_pred, mean_curve, color="red")
        ax.fill_between(x_pred, lower_curve, upper_curve, color="red", alpha=0.2)
        scs.append(sc)
        
    # Add one shared colorbar (using one of the scatter plots)
    cax = fig.add_subplot(gs[:, 8])  # spans all rows in the last column
    cbar = fig.colorbar(scs[0], cax=cax)
    cbar.set_label("Sample size of data point")
    fig.supxlabel("Depth")
    fig.supylabel("Proportion of incorrect current question")
    plt.tight_layout()
    plt.show()

prop_comparison()
prop_time_analysis()
prop_depth_analysis()