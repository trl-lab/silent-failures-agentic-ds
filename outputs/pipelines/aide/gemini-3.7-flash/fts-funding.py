import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "amountUSD"
if target_col not in train_df.columns:
    # In case the target is named differently or case differs
    for col in train_df.columns:
        if col.lower() == "amountusd":
            target_col = col
            break

# Ensure index aligns with row numbers for submission
test_df = test_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col], errors="ignore")
X_test = test_df.copy()

# 2. Preprocess & Feature Engineering
combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Parse and extract features from date/datetime columns
date_cols = []
for col in combined.columns:
    if (
        "date" in col.lower()
        or "time" in col.lower()
        or "created" in col.lower()
        or "updated" in col.lower()
    ):
        try:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
            date_cols.append(col)
        except Exception:
            pass

for col in date_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_dayofyear"] = combined[col].dt.dayofyear
    combined[f"{col}_timestamp"] = combined[col].astype("int64") // 10**9
    combined.drop(columns=[col], inplace=True)

# Handle categorical and object columns
cat_cols = combined.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
for col in cat_cols:
    combined[col] = combined[col].astype(str).fillna("missing")
    # Frequency encoding
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)
    # Category type for native LightGBM handling
    combined[col] = combined[col].astype("category")

# Split back to train and test
n_train = len(train_df)
X_train_processed = combined.iloc[:n_train].copy()
X_test_processed = combined.iloc[n_train:].copy()

# 3. Model Training & 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

categorical_features = combined.select_dtypes(include=["category"]).columns.tolist()

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_processed, y)):
    X_tr, y_tr = X_train_processed.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_processed.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_features)
    val_data = lgb.Dataset(
        X_va, label=y_va, reference=trn_data, categorical_feature=categorical_features
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test_processed, num_iteration=model.best_iteration) / n_splits
    )

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "amountUSD": np.clip(test_preds, 0, None)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
