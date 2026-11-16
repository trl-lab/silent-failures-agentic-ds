import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Original Principal Amount (US$)"
y = train_df[target_col].values.astype(np.float64)

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Identify date columns
date_cols = []
for col in X_train_raw.columns:
    if (
        "date" in col.lower()
        or "period" in col.lower()
        or np.issubdtype(X_train_raw[col].dtype, np.datetime64)
    ):
        date_cols.append(col)
    else:
        sample_vals = X_train_raw[col].dropna()
        if len(sample_vals) > 0 and isinstance(sample_vals.iloc[0], str):
            try:
                pd.to_datetime(sample_vals.iloc[:20])
                date_cols.append(col)
            except (ValueError, TypeError):
                pass


def engineer_features(df_train, df_test, date_columns):
    combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    # Date feature extraction
    parsed_dates = {}
    for col in date_columns:
        dt_series = pd.to_datetime(combined[col], errors="coerce")
        parsed_dates[col] = dt_series
        combined[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
        combined[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
        combined[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
        combined[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
        combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
        combined[f"{col}_is_month_end"] = dt_series.dt.is_month_end.fillna(
            False
        ).astype(int)

    # Date differences
    date_keys = list(parsed_dates.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            col1, col2 = date_keys[i], date_keys[j]
            diff = (parsed_dates[col2] - parsed_dates[col1]).dt.days
            combined[f"days_diff_{col1}_to_{col2}"] = diff.fillna(-9999).astype(float)

    # Drop raw date columns
    combined.drop(columns=date_columns, inplace=True, errors="ignore")

    # Categorical and numerical handling
    cat_cols = []
    for col in combined.columns:
        if combined[col].dtype == "object" or combined[col].dtype.name == "category":
            combined[f"{col}_freq"] = combined[col].map(
                combined[col].value_counts(dropna=False)
            )
            combined[col] = combined[col].astype(str).astype("category")
            cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = combined[col].fillna(combined[col].median())

    train_feat = combined.iloc[: len(df_train)].reset_index(drop=True)
    test_feat = combined.iloc[len(df_train) :].reset_index(drop=True)

    return train_feat, test_feat, cat_cols


X_train, X_test, cat_features = engineer_features(X_train_raw, X_test_raw, date_cols)

# 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_estimators": 2000,
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
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# Prepare submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
