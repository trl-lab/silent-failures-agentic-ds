import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Population for whom broadband available, 2019 (%)"

# Filter out missing targets if any
train_df = train_df[train_df[target_col].notnull()].reset_index(drop=True)

y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Feature alignment
feature_cols = [c for c in X.columns if c in X_test.columns]
X = X[feature_cols].copy()
X_test = X_test[feature_cols].copy()


# Sanitize column names to avoid special characters causing LightGBM JSON errors
def sanitize_columns(cols):
    clean_cols = []
    seen = {}
    for col in cols:
        cleaned = re.sub(r"[^\w]", "_", str(col))
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}_{seen[cleaned]}"
        else:
            seen[cleaned] = 0
        clean_cols.append(cleaned)
    return clean_cols


clean_cols = sanitize_columns(feature_cols)
X.columns = clean_cols
X_test.columns = clean_cols

# Handle categorical features cleanly
for col in clean_cols:
    if X[col].dtype == "object" or X[col].dtype.name == "category":
        cats = pd.concat([X[col], X_test[col]], axis=0).astype(str).unique()
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)

# Setup K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 1500,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / n_splits

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {cv_rmse:.5f}")

# Post-processing predictions within domain target bounds
min_target, max_target = y.min(), y.max()
test_preds = np.clip(test_preds, min_target, max_target)

# Create submission directory and save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
