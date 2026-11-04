import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Amount in USD"
train_len = len(train_df)

# Concatenate for feature engineering
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Extract date features
date_cols = []
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])
            date_cols.append(col)
        except Exception:
            pass

for col in date_cols:
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_quarter"] = df[col].dt.quarter
    df[f"{col}_days"] = (df[col] - pd.Timestamp("1970-01-01")).dt.days
    df = df.drop(columns=[col])

# Categorical processing
cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

for col in cat_cols:
    freq = df[col].value_counts().to_dict()
    df[f"{col}_freq"] = df[col].map(freq)
    df[col] = df[col].astype("category")

# Split back to train and test sets
X = df.iloc[:train_len].copy()
y = train_df[target_col].values
X_test = df.iloc[train_len:].copy()

# Setup cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# Post-processing: Non-negative clipping if target is non-negative
if (y >= 0).all():
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Amount in USD": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
