import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "salary_in_usd"
# Exclude target and potential raw salary leak columns if they are not in test
feature_cols = [
    col for col in test_df.columns if col in train_df.columns and col != target_col
]

# Identify categorical features
cat_cols = [
    col
    for col in feature_cols
    if train_df[col].dtype == "object"
    or isinstance(train_df[col].dtype, pd.CategoricalDtype)
]

# Convert categorical columns to category dtype for LightGBM
for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")

X = train_df[feature_cols]
y = train_df[target_col]
X_test = test_df[feature_cols]

# 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 15,
    "feature_fraction": 0.8,
    "verbose": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train_f, label=y_train_f, categorical_feature=cat_cols)
    val_data = lgb.Dataset(
        X_val_f, label=y_val_f, reference=train_data, categorical_feature=cat_cols
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val_f, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Generate submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), "salary_in_usd": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
