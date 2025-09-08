import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

data = pd.read_csv(f"paa_backtracked.csv")
data = data[~pd.isna(data['depth'])]

# Define global color scale
vmin, vmax = 1, 0

# Create figure and layout
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 13, figure=fig)

# Subplots
axes = [
    fig.add_subplot(gs[0, 0:4]),
    fig.add_subplot(gs[0, 4:8]),
    fig.add_subplot(gs[0, 8:12]),
    fig.add_subplot(gs[1, 0:6]),
    fig.add_subplot(gs[1, 6:12])
]
scs = []

for idx, FEATURE_LIST in enumerate([
    ['class_1'],
    ['class_2'],
    ['class_3'],
    ['class_1', 'class_2'],
    ['class_1', 'class_2', 'class_3'],
]):
    FEATURE_NAME = '_'.join(FEATURE_LIST)

    data[FEATURE_NAME] = data[FEATURE_LIST].sum(axis=1)
    data[f'{FEATURE_NAME}_t-1'] = data[[f'{feature}_t-1' for feature in FEATURE_LIST]].sum(axis=1)
    data[f'{FEATURE_NAME}_prev_cnt'] = data[[f'{feature}_prev_cnt' for feature in FEATURE_LIST]].sum(axis=1)
    data[f'{FEATURE_NAME}_prop'] = data[f'{FEATURE_NAME}_prev_cnt'] / (data['depth'] - 1)
    
    df_1 = data.groupby([f'{FEATURE_NAME}_prop', 'depth'])[FEATURE_NAME].agg(['count', 'mean']).reset_index()
    vmin = min(vmin, df_1["count"].min())
    vmax = max(vmax, df_1["count"].max())

for idx, FEATURE_LIST in enumerate([
    ['class_1'],
    ['class_2'],
    ['class_3'],
    ['class_1', 'class_2'],
    ['class_1', 'class_2', 'class_3'],
]):
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
    y_pred = logistic(linear_pred)
    y_lower = logistic(lower_logit)
    y_upper = logistic(upper_logit)

    # Scatterplot + logistic regression with CI band
    sc = ax.scatter(
        df_1[f'{FEATURE_NAME}_prop'], df_1["mean"],
        s=df_1["size_scaled"]*20, c=df_1["count"],
        cmap="viridis",
        alpha=0.5,
        edgecolor="k",
        vmin=vmin, vmax=vmax
    )
    ax.set_ylim(-0.05, 1.05)
    ax.plot(x_pred, y_pred, color="red")
    ax.fill_between(x_pred, y_lower, y_upper, color="red", alpha=0.2)
    
    scs.append(sc)
    
# Add one shared colorbar (using one of the scatter plots)
cax = fig.add_subplot(gs[:, 12])  # spans all rows in the last column
cbar = fig.colorbar(scs[0], cax=cax)
cbar.set_label("Sample size of data point")

# fig.set_label("Proportion of incorrect assumptions in prior steps")

plt.tight_layout()
plt.show()






# --- Simulate toy data with schema (query_no, depth, label) ---
np.random.seed(123)
n_queries = 40
data = []
for q in range(n_queries):
    n_hist = np.random.randint(5, 12)  # number of histories in this query
    for depth in range(1, n_hist + 1):
        label = np.random.binomial(1, 1 / (1 + np.exp(-0.3 * (depth - 5))))  # outcome ~ depth
        data.append((q, depth, label))

df_depth = pd.DataFrame(data, columns=["query_no", "depth", "label"])

# --- Bootstrap subsampling with discrete depth predictor ---
n_boot = 300
max_depth = df_depth["depth"].max()
depth_values = np.arange(1, max_depth + 1)

pred_curves = []

for b in range(n_boot):
    # Subsample one history per query_no
    sampled = df_depth.groupby("query_no").sample(1, replace=False)
    
    # Logistic regression with depth as predictor
    X = sm.add_constant(sampled["depth"])
    y = sampled["label"]
    try:
        model = sm.Logit(y, X).fit(disp=0)
        preds = model.predict(sm.add_constant(depth_values))
        pred_curves.append(preds)
    except Exception:
        continue

pred_curves = np.array(pred_curves)

# --- Compute median and CI ---
median_curve = np.median(pred_curves, axis=0)
lower_curve = np.percentile(pred_curves, 2.5, axis=0)
upper_curve = np.percentile(pred_curves, 97.5, axis=0)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.plot(depth_values, median_curve, marker="o", label="Median fitted curve")
plt.fill_between(depth_values, lower_curve, upper_curve, color="gray", alpha=0.3, label="95% CI")
plt.xlabel("History depth (number of preceding questions)")
plt.ylabel("Predicted probability of incorrect question")
plt.title("Bootstrap subsampling logistic regression with discrete depth")
plt.legend()
plt.show()
