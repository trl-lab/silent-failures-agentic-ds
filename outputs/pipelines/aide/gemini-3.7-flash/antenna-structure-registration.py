import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Strucht"
train_y = train_df[target_col].values
train_X = train_df.drop(columns=[target_col]).copy()
test_X = test_df.copy()

# Ensure matching columns
common_cols = [c for c in train_X.columns if c in test_X.columns]
train_X = train_X[common_cols]
test_X = test_X[common_cols]

# 2. Preprocessing
combined = pd.concat([train_X, test_X], axis=0).reset_index(drop=True)

# Process date columns
for col in combined.columns:
    if combined[col].dtype == "object" or pd.api.types.is_string_dtype(combined[col]):
        # Check if column looks like a date
        sample = combined[col].dropna().head(50)
        if (
            len(sample) > 0
            and sample.str.contains(
                r"^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}", regex=True
            ).mean()
            > 0.8
        ):
            try:
                date_s = pd.to_datetime(combined[col], errors="coerce")
                combined[f"{col}_year"] = date_s.dt.year
                combined[f"{col}_month"] = date_s.dt.month
                combined[f"{col}_day"] = date_s.dt.day
                combined = combined.drop(columns=[col])
            except Exception:
                pass

# Identify categorical and numeric columns
cat_cols = []
num_cols = []
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        cat_cols.append(col)
        combined[col] = combined[col].astype("category")
    else:
        num_cols.append(col)

# Split back to train and test
n_train = len(train_X)
train_processed = combined.iloc[:n_train].copy()
test_processed = combined.iloc[n_train:].copy()

# 3. 5-Fold Cross-Validation Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_processed))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed, train_y)):
    X_tr, y_tr = train_processed.iloc[train_idx], train_y[train_idx]
    X_val, y_val = train_processed.iloc[val_idx], train_y[val_idx]

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(test_processed, num_iteration=model.best_iteration) / kf.n_splits
    )

# 4. Evaluation
oof_rmse = np.sqrt(mean_squared_error(train_y, oof_preds))
print(f"5-Fold CV Out-of-Fold RMSE: {oof_rmse:.4f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Strucht": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
