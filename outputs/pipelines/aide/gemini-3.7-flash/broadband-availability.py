import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Population for whom broadband available, 2019 (%)"

# 2. Separate features and target
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Ensure matching columns
features = [col for col in X.columns if col in X_test.columns]
X = X[features]
X_test = X_test[features]

# Clean column names to eliminate special JSON characters that break LightGBM
clean_col_names = [re.sub(r"[^\w]", "_", col) for col in features]
X.columns = clean_col_names
X_test.columns = clean_col_names

# Convert object/string columns to category type for LightGBM
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 3. Model training with 5-fold CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_preds = model.predict(X_va)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / n_splits

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Target values represent percentages, clip predictions to valid range [0, 100]
test_preds = np.clip(test_preds, 0.0, 100.0)

# 5. Create submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
