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

target_col = "IMU Score"
drop_cols = [target_col]

# If there are ID columns that should not be used as features
id_candidates = [
    col for col in train_df.columns if "id" in col.lower() or "code" in col.lower()
]

# Identify feature types
features = [c for c in train_df.columns if c not in drop_cols]
X = train_df[features].copy()
y = train_df[target_col].copy()
X_test = test_df[features].copy()

# Date handling and categorical feature encoding
cat_cols = []
for col in features:
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        for df in [X, X_test]:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df.drop(columns=[col], inplace=True)
    elif X[col].dtype == "object" or str(X[col].dtype) == "category":
        cat_cols.append(col)
        # Convert to category for LightGBM
        X[col] = X[col].astype(str).astype("category")
        X_test[col] = X_test[col].astype(str).astype("category")

# Update features list after transformations
features = list(X.columns)

# 2. Setup 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

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
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 3. Train models across folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# 4. Overall validation performance
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"\nOverall Out-of-Fold RMSE: {cv_rmse:.4f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "IMU Score": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
