import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "installed_capacity"
y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Combine for uniform feature engineering
n_train = len(X)
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

# 2. Feature Engineering & Preprocessing
for col in combined.columns:
    # Try converting date/year like strings or objects
    if combined[col].dtype == "object":
        # Check if it looks like a datetime
        try:
            dt_series = pd.to_datetime(combined[col], errors="raise")
            combined[f"{col}_year"] = dt_series.dt.year
            combined[f"{col}_month"] = dt_series.dt.month
            combined[f"{col}_day"] = dt_series.dt.day
            combined = combined.drop(columns=[col])
            continue
        except (ValueError, TypeError):
            pass

        # Frequency encoding for high/medium cardinality categoricals
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype("category")
    elif str(combined[col].dtype).startswith("category"):
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype("category")

X = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

# 3. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

# Ensure capacity predictions are non-negative
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "installed_capacity": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
