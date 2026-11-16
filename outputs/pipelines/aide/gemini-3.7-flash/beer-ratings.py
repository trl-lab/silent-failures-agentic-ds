import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Set random seed
SEED = 42
np.random.seed(SEED)

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "review_overall"
if target_col not in train_df.columns:
    possible_targets = [
        c
        for c in train_df.columns
        if "overall" in c.lower() or "target" in c.lower() or "rating" in c.lower()
    ]
    target_col = possible_targets[0] if possible_targets else train_df.columns[-1]

y = train_df[target_col].values.astype(np.float64)

n_train = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    axis=0,
    ignore_index=True,
)

# Identify aspect/sub-rating columns
aspect_candidates = [
    c
    for c in combined_df.columns
    if any(
        k in c.lower()
        for k in [
            "aroma",
            "appearance",
            "palate",
            "taste",
            "flavor",
            "smell",
            "look",
            "feel",
        ]
    )
]

for col in aspect_candidates:
    combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

# Numerical summary across aspects
if len(aspect_candidates) >= 2:
    combined_df["aspect_mean"] = combined_df[aspect_candidates].mean(axis=1)
    combined_df["aspect_std"] = combined_df[aspect_candidates].std(axis=1)
    combined_df["aspect_min"] = combined_df[aspect_candidates].min(axis=1)
    combined_df["aspect_max"] = combined_df[aspect_candidates].max(axis=1)
    combined_df["aspect_range"] = combined_df["aspect_max"] - combined_df["aspect_min"]

    # Pairwise aspect differences
    for i in range(len(aspect_candidates)):
        for j in range(i + 1, len(aspect_candidates)):
            col_i, col_j = aspect_candidates[i], aspect_candidates[j]
            combined_df[f"diff_{col_i}_sub_{col_j}"] = (
                combined_df[col_i] - combined_df[col_j]
            )

    # Deviation of each aspect from the review's mean aspect score
    for col in aspect_candidates:
        combined_df[f"{col}_dev_aspect_mean"] = (
            combined_df[col] - combined_df["aspect_mean"]
        )

# Identify ABV column if present
abv_cols = [c for c in combined_df.columns if "abv" in c.lower()]
for col in abv_cols:
    combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

# Date/time features
time_cols = [
    c
    for c in combined_df.columns
    if any(k in c.lower() for k in ["time", "date", "timestamp"])
]
for t_col in time_cols:
    dt = pd.to_datetime(combined_df[t_col], errors="coerce")
    if dt.isna().mean() < 0.5:
        combined_df[f"{t_col}_year"] = dt.dt.year.fillna(-1).astype(int)
        combined_df[f"{t_col}_month"] = dt.dt.month.fillna(-1).astype(int)
        combined_df[f"{t_col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
        combined_df[f"{t_col}_hour"] = dt.dt.hour.fillna(-1).astype(int)
    combined_df.drop(columns=[t_col], inplace=True, errors="ignore")

# Group-based aggregation features on entities
entity_cols = [
    c
    for c in combined_df.columns
    if any(
        k in c.lower()
        for k in ["beer_beerid", "beer_name", "brewery", "style", "profile", "user"]
    )
]

agg_targets = aspect_candidates + (
    ["aspect_mean"] if "aspect_mean" in combined_df else []
)

for ent in entity_cols:
    # Review count per entity
    cnt_series = combined_df[ent].map(combined_df[ent].value_counts())
    combined_df[f"{ent}_review_count"] = cnt_series

    # Group mean and deviation for aspect ratings
    for target_agg in agg_targets[:3]:  # Top aspect aggregations to keep memory lean
        grp_mean = combined_df.groupby(ent)[target_agg].transform("mean")
        combined_df[f"{ent}_{target_agg}_grp_mean"] = grp_mean
        combined_df[f"{ent}_{target_agg}_grp_diff"] = combined_df[target_agg] - grp_mean

# Text length features
obj_cols = combined_df.select_dtypes(include=["object", "string"]).columns.tolist()
for c in obj_cols:
    str_series = combined_df[c].astype(str).fillna("")
    if str_series.str.len().mean() > 30:
        combined_df[f"{c}_char_len"] = str_series.str.len()
        combined_df[f"{c}_word_count"] = str_series.str.split().apply(len).astype(int)

# Categorical encoding
cat_cols = combined_df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

for c in cat_cols:
    # Frequency encoding
    freq = combined_df[c].value_counts()
    combined_df[f"{c}_freq"] = combined_df[c].map(freq).fillna(0).astype(float)

    # Ordinal / Label encoding
    combined_df[c] = (
        combined_df[c]
        .astype(str)
        .fillna("__MISSING__")
        .astype("category")
        .cat.codes.astype(int)
    )

# Convert all columns to float32
for c in combined_df.columns:
    combined_df[c] = pd.to_numeric(combined_df[c], errors="coerce").astype(np.float32)

# Separate back into train and test
X_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# Out-of-fold smoothed target encoding
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
global_mean = float(np.mean(y))
smooth = 25.0

for c in cat_cols:
    if c in X_train.columns:
        te_col_name = f"{c}_target_enc"
        X_train[te_col_name] = global_mean
        X_test[te_col_name] = global_mean

        stats = (
            pd.DataFrame({"cat": X_train[c], "target": y})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        test_te = (stats["count"] * stats["mean"] + smooth * global_mean) / (
            stats["count"] + smooth
        )
        X_test[te_col_name] = X_test[c].map(test_te).fillna(global_mean).values

        for tr_idx, val_idx in kf.split(X_train):
            tr_stats = (
                pd.DataFrame({"cat": X_train.iloc[tr_idx][c], "target": y[tr_idx]})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            val_te = (tr_stats["count"] * tr_stats["mean"] + smooth * global_mean) / (
                tr_stats["count"] + smooth
            )
            X_train.loc[val_idx, te_col_name] = (
                X_train.iloc[val_idx][c].map(val_te).fillna(global_mean).values
            )

# 5-fold LightGBM Training
oof_preds = np.zeros(len(y), dtype=np.float64)
test_preds = np.zeros(len(X_test), dtype=np.float64)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 25,
    "random_state": SEED,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "review_overall": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv")
