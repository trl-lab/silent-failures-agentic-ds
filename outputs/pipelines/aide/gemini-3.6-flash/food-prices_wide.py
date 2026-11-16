import os
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# Set seed for reproducibility
np.random.seed(42)

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "usdprice"
features_to_drop = [target_col]
if "id" in train_df.columns:
    features_to_drop.append("id")

train_len = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

# Parse and extract datetime features
date_cols = []
for col in combined.columns:
    if col in features_to_drop:
        continue
    if "date" in col.lower() or combined[col].dtype == "object":
        try:
            parsed = pd.to_datetime(combined[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(combined):
                date_cols.append(col)
                combined[f"{col}_year"] = parsed.dt.year
                combined[f"{col}_month"] = parsed.dt.month
                combined[f"{col}_day"] = parsed.dt.day
                combined[f"{col}_dayofweek"] = parsed.dt.dayofweek
                combined[f"{col}_quarter"] = parsed.dt.quarter
                combined[f"{col}_dayofyear"] = parsed.dt.dayofyear
                combined[f"{col}_sin_month"] = np.sin(2 * np.pi * parsed.dt.month / 12)
                combined[f"{col}_cos_month"] = np.cos(2 * np.pi * parsed.dt.month / 12)
        except Exception:
            pass

# Categorical encoding and frequency statistics
cat_cols = []
for col in combined.columns:
    if col in features_to_drop or col == target_col or col in date_cols:
        continue
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        cat_cols.append(col)
        freq = combined[col].value_counts(normalize=True).to_dict()
        combined[f"{col}_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype("category").cat.codes

# Group aggregations on key features
num_cols = [
    c
    for c in combined.columns
    if c not in cat_cols
    and c not in features_to_drop
    and c != target_col
    and c not in date_cols
]
for group_col in cat_cols[:5]:
    for num_col in num_cols[:5]:
        if group_col in combined.columns and num_col in combined.columns:
            grp = combined.groupby(group_col)[num_col].agg(["mean", "std"])
            combined[f"{group_col}_{num_col}_mean"] = combined[group_col].map(
                grp["mean"]
            )
            combined[f"{group_col}_{num_col}_std"] = (
                combined[group_col].map(grp["std"]).fillna(0)
            )

# Separate into train and test sets
drop_cols = [c for c in features_to_drop + date_cols if c in combined.columns]
X_full = combined.iloc[:train_len].drop(columns=drop_cols)
y_full = train_df[target_col].values
X_test = combined.iloc[train_len:].drop(columns=drop_cols)

feature_cols = [c for c in X_full.columns if c != target_col and c not in drop_cols]
X_full = X_full[feature_cols]
X_test = X_test[feature_cols]

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros(len(X_full))
oof_preds_cb = np.zeros(len(X_full))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
    X_train, y_train = X_full.iloc[train_idx].copy(), y_full[train_idx]
    X_val, y_val = X_full.iloc[val_idx].copy(), y_full[val_idx]
    X_test_fold = X_test.copy()

    # Out-of-fold target encoding for categoricals
    global_mean = y_train.mean()
    for col in cat_cols:
        if col in feature_cols:
            stats = pd.DataFrame({"cat": X_train[col], "target": y_train})
            group_stats = stats.groupby("cat")["target"].agg(["count", "mean"])
            smooth_te = (
                group_stats["count"] * group_stats["mean"] + 10 * global_mean
            ) / (group_stats["count"] + 10)
            te_dict = smooth_te.to_dict()

            X_train[f"{col}_te"] = X_train[col].map(te_dict).fillna(global_mean)
            X_val[f"{col}_te"] = X_val[col].map(te_dict).fillna(global_mean)
            X_test_fold[f"{col}_te"] = X_test_fold[col].map(te_dict).fillna(global_mean)

    # Train XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test_fold) / kf.n_splits

    # Train CatBoost Regressor
    cb_model = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    oof_preds_cb[val_idx] = cb_model.predict(X_val)
    test_preds_cb += cb_model.predict(X_test_fold) / kf.n_splits

# Optimal ensemble weight search
best_w = 0.5
best_rmse = float("inf")
for w in np.linspace(0, 1, 101):
    blend_oof = w * oof_preds_xgb + (1 - w) * oof_preds_cb
    rmse = np.sqrt(mean_squared_error(y_full, blend_oof))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

final_test_preds = best_w * test_preds_xgb + (1 - best_w) * test_preds_cb

print(f"Validation RMSE: {best_rmse:.5f}")

# Output directory and predictions file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "usdprice": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
