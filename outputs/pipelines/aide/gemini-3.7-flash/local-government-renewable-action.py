import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Size of Purchase (MW)"
features = [c for c in train_df.columns if c != target_col]

# 2. Preprocessing & Feature Engineering
combined = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Process columns
for col in features:
    # Try parsing dates if column name or content suggests date
    if (
        "date" in col.lower()
        or "year" in col.lower()
        or combined[col].dtype == "object"
    ):
        try:
            dt_series = pd.to_datetime(combined[col], errors="coerce")
            if dt_series.notnull().sum() > 0.5 * len(combined):
                combined[f"{col}_year"] = dt_series.dt.year
                combined[f"{col}_month"] = dt_series.dt.month
                combined[f"{col}_day"] = dt_series.dt.day
        except Exception:
            pass

    # For remaining object columns, extract numbers if mixed or convert to category
    if combined[col].dtype == "object":
        # Check if it can be converted to purely numeric
        numeric_attempt = pd.to_numeric(
            combined[col]
            .astype(str)
            .str.replace(",", "")
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False),
            errors="coerce",
        )
        if (
            numeric_attempt.notnull().sum() > 0.7 * len(combined)
            and numeric_attempt.nunique() > 10
        ):
            combined[f"{col}_num"] = numeric_attempt

        # Categorical type for LightGBM
        combined[col] = combined[col].astype("category")

# Separate train and test features
n_train = len(train_df)
X = combined.iloc[:n_train].copy()
y = train_df[target_col].values
X_test = combined.iloc[n_train:].copy()

# Ensure categorical columns have category dtype
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# 3. K-Fold Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 10,
    "random_state": 42,
    "verbosity": -1,
    "n_estimators": 1500,
}

# 4. Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 5. Evaluation
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {oof_rmse:.4f}")

# Post-processing: Purchase size in MW cannot be negative
test_preds = np.clip(test_preds, 0, None)

# 6. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Size of Purchase (MW)": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
