import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import tqdm

global data
data = pd.read_csv(f"./data/paa_backtracked.csv")

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
              ''. join([f" & {pval * 3 / rank :.3f}" for rank in range(1, 4)]) +
              "\\\\")

def prop_time_analysis():
    global data
    data = data[~pd.isna(data['depth'])]
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
        df_1 = data.groupby([f'{FEATURE_NAME}_prop', 'depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
        vmin = min(vmin, df_1["count"].min())
        vmax = max(vmax, df_1["count"].max())

    for idx, FEATURE_LIST in enumerate(FEATURE_LIST_LIST):
        ax = axes[idx]
        FEATURE_NAME = '_'.join(FEATURE_LIST)
        df = data[~pd.isna(data[f'{FEATURE_NAME}_prop'])]
        df_1 = df.groupby([f'{FEATURE_NAME}_prop', 'depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
        df_1["size_scaled"] = np.sqrt(df_1["count"])
        
        X = sm.add_constant(df[f"{FEATURE_NAME}_prop"])  # add intercept
        y = df[FEATURE_NAME]

        # Fit logistic regression
        logit_model = sm.Logit(y, X).fit(disp=0)

        # Generate predictions
        x_pred = np.linspace(0, 1, 200)
        X_pred = sm.add_constant(x_pred)
        linear_pred = X_pred @ logit_model.params

        # Compute standard errors for CI
        cov = logit_model.cov_params()
        se_lin = np.sqrt(np.sum((X_pred @ cov) * X_pred, axis=1))
        crit = stats.norm.ppf(0.975)  # 95% CI
        lower_logit = linear_pred - crit * se_lin
        upper_logit = linear_pred + crit * se_lin

        # Logistic transform
        def logistic(z): return 1 / (1 + np.exp(-z))
        mean_curve = logistic(linear_pred)
        lower_curve = logistic(lower_logit)
        upper_curve = logistic(upper_logit)
        print(upper_curve[len(upper_curve) - 1])

        # --- Plot ---
        sc = ax.scatter(
            df_1[f'{FEATURE_NAME}_prop'], df_1["mean"],
            s=df_1["size_scaled"]*20, c=df_1["count"],
            cmap="viridis",
            alpha=0.5,
            edgecolor="k",
            vmin=vmin, vmax=vmax
        )
        ax.set_ylim(-0.05, 0.55)
        ax.plot(x_pred, mean_curve, color="red")
        ax.fill_between(x_pred, lower_curve, upper_curve, color="red", alpha=0.2)
        scs.append(sc)
        
    # Add one shared colorbar (using one of the scatter plots)
    cax = fig.add_subplot(gs[:, 8])  # spans all rows in the last column
    cbar = fig.colorbar(scs[0], cax=cax)
    cbar.set_label("Sample size of data point")

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
        df_1["size_scaled"] = np.sqrt(df_1["count"])
        
        X = sm.add_constant(df["depth"])  # add intercept
        y = df[FEATURE_NAME]

        # Fit logistic regression
        logit_model = sm.Logit(y, X).fit(disp=0)

        # Generate predictions
        x_pred = np.linspace(0, 9, 200)
        X_pred = sm.add_constant(x_pred)
        linear_pred = X_pred @ logit_model.params

        # Compute standard errors for CI
        cov = logit_model.cov_params()
        se_lin = np.sqrt(np.sum((X_pred @ cov) * X_pred, axis=1))
        crit = stats.norm.ppf(0.975)  # 95% CI
        lower_logit = linear_pred - crit * se_lin
        upper_logit = linear_pred + crit * se_lin

        # Logistic transform
        def logistic(z): return 1 / (1 + np.exp(-z))
        mean_curve = logistic(linear_pred)
        lower_curve = logistic(lower_logit)
        upper_curve = logistic(upper_logit)
        print(upper_curve[len(upper_curve) - 1])

        # --- Plot ---
        sc = ax.scatter(
            df_1['depth'], df_1["mean"],
            s=df_1["size_scaled"]*20, c=df_1["count"],
            cmap="viridis",
            alpha=0.5,
            edgecolor="k",
            vmin=vmin, vmax=vmax
        )
        ax.set_ylim(-0.01, 0.15)
        ax.plot(x_pred, mean_curve, color="red")
        ax.fill_between(x_pred, lower_curve, upper_curve, color="red", alpha=0.2)
        scs.append(sc)
        
    # Add one shared colorbar (using one of the scatter plots)
    cax = fig.add_subplot(gs[:, 8])  # spans all rows in the last column
    cbar = fig.colorbar(scs[0], cax=cax)
    cbar.set_label("Sample size of data point")

    plt.tight_layout()
    plt.show()

prop_comparison()
prop_time_analysis()
prop_depth_analysis()