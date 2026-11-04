import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Amount"
y = train_df[target_col].values.astype(np.float64)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Feature Engineering
combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Parse date columns
date_cols = [
    col
    for col in combined.columns
    if "date" in col.lower()
    or "received" in col.lower()
    or "start" in col.lower()
    or "end" in col.lower()
]
for col in date_cols:
    try:
        dt = pd.to_datetime(combined[col], errors="coerce")
        combined[f"{col}_year"] = dt.dt.year
        combined[f"{col}_month"] = dt.dt.month
        combined[f"{col}_day"] = dt.dt.day
        combined[f"{col}_dayofweek"] = dt.dt.dayofweek
        combined[f"{col}_quarter"] = dt.dt.quarter
        combined[f"{col}_dayofyear"] = dt.dt.dayofyear
        combined[f"{col}_timestamp"] = dt.astype("int64") // 10**9
        combined[col] = combined[col].astype(str)
    except Exception:
        pass

# Identify categorical columns
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

# Generate bi-gram interaction features for top categorical columns
top_cats = [c for c in cat_cols if combined[c].nunique() > 1][:4]
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        col_i = top_cats[i]
        col_j = top_cats[j]
        inter_name = f"{col_i}_x_{col_j}"
        combined[inter_name] = (
            combined[col_i].astype(str) + "__" + combined[col_j].astype(str)
        )
        inter_freq = combined[inter_name].value_counts()
        combined[f"{inter_name}_freq"] = combined[inter_name].map(inter_freq)
        combined[inter_name] = combined[inter_name].astype("category")

# Relational group cardinality features
for col in cat_cols:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)
    combined[col] = combined[col].astype("category")

if len(top_cats) >= 2:
    for c1 in top_cats[:2]:
        for c2 in top_cats[2:4]:
            if c1 != c2:
                nunique_map = combined.groupby(c1)[c2].nunique()
                combined[f"{c1}_nunique_{c2}"] = combined[c1].map(nunique_map)

# Split back to train and test
n_train = len(train_df)
X_train_proc = combined.iloc[:n_train].copy()
X_test_proc = combined.iloc[n_train:].copy()

# 3. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 47,
    "max_depth": 8,
    "min_child_samples": 25,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test_proc) / kf.n_splits

    fold_rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Overall OOF RMSE: {cv_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Amount": np.clip(test_preds, 0, None)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv.")
