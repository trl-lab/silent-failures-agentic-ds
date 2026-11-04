import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NADAC Rate"
drop_cols = [target_col]

# Identify feature columns
features = [col for col in train_df.columns if col not in drop_cols]

# 2. Feature Engineering & Preprocessing
for col in features:
    # Process date/datetime columns
    if pd.api.types.is_datetime64_any_dtype(train_df[col]) or "date" in col.lower():
        train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
        test_df[col] = pd.to_datetime(test_df[col], errors="coerce")

        train_df[f"{col}_year"] = train_df[col].dt.year
        train_df[f"{col}_month"] = train_df[col].dt.month
        train_df[f"{col}_day"] = train_df[col].dt.day
        train_df[f"{col}_dayofweek"] = train_df[col].dt.dayofweek

        test_df[f"{col}_year"] = test_df[col].dt.year
        test_df[f"{col}_month"] = test_df[col].dt.month
        test_df[f"{col}_day"] = test_df[col].dt.day
        test_df[f"{col}_dayofweek"] = test_df[col].dt.dayofweek

        train_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)

# Re-list feature columns after datetime expansion
features = [col for col in train_df.columns if col not in drop_cols]

# Handle categorical / object columns
for col in features:
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

X = train_df[features]
y = train_df[target_col]
X_test = test_df[features]

# 3. Model Training with 5-Fold Cross-Validation
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
    "n_estimators": 2000,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / n_splits

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.6f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "NADAC Rate": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
