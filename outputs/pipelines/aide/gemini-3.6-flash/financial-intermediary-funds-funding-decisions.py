import itertools
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# 1. Setup paths and directories
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# 2. Load dataset
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Amount in USD"
y = train_df[target_col].values.astype(np.float64)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

drop_cols = [target_col]
if "id" in train_df.columns:
    drop_cols.append("id")

X_train_raw = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test_raw = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

n_train = len(X_train_raw)
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# 3. Datetime Parsing & Pairwise Duration Engineering
parsed_dates = {}
for col in combined.columns:
    if combined[col].dtype == "object" or str(combined[col].dtype).startswith("string"):
        try:
            parsed = pd.to_datetime(combined[col], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.5:
                parsed_dates[col] = parsed
                combined[f"{col}_year"] = parsed.dt.year
                combined[f"{col}_month"] = parsed.dt.month
                combined[f"{col}_day"] = parsed.dt.day
                combined[f"{col}_dayofweek"] = parsed.dt.dayofweek
                combined[f"{col}_quarter"] = parsed.dt.quarter
                combined.drop(columns=[col], inplace=True)
        except Exception:
            pass

# Compute duration differences between pairs of datetime columns
date_cols = list(parsed_dates.keys())
for col1, col2 in itertools.combinations(date_cols, 2):
    diff_name = f"diff_days_{col1}_vs_{col2}"
    combined[diff_name] = (
        parsed_dates[col2] - parsed_dates[col1]
    ).dt.total_seconds() / 86400.0

# Categorical Frequency Encoding & Group Aggregations
cat_cols = combined.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for col in cat_cols:
    freq = combined[col].value_counts(dropna=False).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq)

num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
group_candidates = [c for c in cat_cols if 2 <= combined[c].nunique() <= 100]

for g_col in group_candidates[:5]:
    for n_col in num_cols[:10]:
        if n_col != f"{g_col}_freq":
            grp = combined.groupby(g_col)[n_col].agg(["mean", "std"]).reset_index()
            grp.columns = [
                g_col,
                f"{g_col}_{n_col}_grp_mean",
                f"{g_col}_{n_col}_grp_std",
            ]
            combined = combined.merge(grp, on=g_col, how="left")

for col in cat_cols:
    combined[col] = combined[col].fillna("Missing").astype(str)
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    combined[col] = combined[col].astype("category")

# Separate back into train and test
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 4. Multi-Seed Cross-Validation Setup with Regularized LightGBM
seeds = [42, 2024, 101]
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

min_target = max(0.0, float(np.min(y)))

for seed_idx, seed in enumerate(seeds):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    seed_oof = np.zeros(n_train)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3500,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "cat_smooth": 10.0,
        "cat_l2": 10.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        val_pred = model.predict(X_va)
        seed_oof[val_idx] = val_pred
        test_preds += model.predict(X_test) / (kf.n_splits * len(seeds))

    oof_preds += seed_oof / len(seeds)

# Enforce target lower bound constraint
oof_preds = np.clip(oof_preds, a_min=min_target, a_max=None)
test_preds = np.clip(test_preds, a_min=min_target, a_max=None)

# 5. Evaluation Metric Calculation
cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold Multi-Seed CV RMSE: {cv_rmse:.4f}")

# 6. Generate Submission File
submission = pd.DataFrame({"id": test_ids, "Amount in USD": test_preds})
submission_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission.shape}")
