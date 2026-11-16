import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "State Rate"

# Clean target variable if necessary
if train_df[target_col].dtype == object:
    train_df[target_col] = (
        train_df[target_col].astype(str).str.replace("%", "").str.strip()
    )
    train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")

# Drop rows where target is NaN in training data
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Prepare feature sets
X = train_df.drop(columns=[target_col])
y = train_df[target_col].values
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Combine to preprocess consistently
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

# Process columns
categorical_cols = []
numerical_cols = []

for col in combined.columns:
    # Check if object / string or categorical
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        # Try converting to numeric if mostly numbers
        num_conv = pd.to_numeric(
            combined[col]
            .astype(str)
            .str.replace("%", "")
            .str.replace(",", "")
            .str.strip(),
            errors="coerce",
        )
        if num_conv.notna().sum() / len(combined) > 0.8:
            combined[col] = num_conv
            numerical_cols.append(col)
        else:
            combined[col] = combined[col].astype(str).fillna("__MISSING__")
            categorical_cols.append(col)
    else:
        numerical_cols.append(col)

# Add frequency encoding for categorical columns
for col in categorical_cols:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq).astype(float)
    combined[col] = combined[col].astype("category")

# Split back into train and test
X_processed = combined.iloc[: len(train_df)].copy().reset_index(drop=True)
X_test_processed = combined.iloc[len(train_df) :].copy().reset_index(drop=True)

# 2. 5-Fold Cross Validation Setup
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
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 3. Model Training & Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
    X_train_fold, y_train_fold = X_processed.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X_processed.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test_processed, num_iteration=model.best_iteration) / n_splits
    )

# 4. Evaluation
cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "State Rate": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
