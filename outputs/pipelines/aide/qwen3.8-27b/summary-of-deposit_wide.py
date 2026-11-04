import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify target and features
target = "DEPSUM"
y = train[target]
X = train.drop(columns=[target])

# Drop id column if present
if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])

# Drop target from test if accidentally present
if target in test.columns:
    test = test.drop(columns=[target])

# Basic data cleaning
# Drop constant columns
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X = X.drop(columns=constant_cols)
test = test.drop(columns=constant_cols)

# Drop columns with >90% missing
missing_cols = [col for col in X.columns if X[col].isnull().mean() > 0.9]
X = X.drop(columns=missing_cols)
test = test.drop(columns=missing_cols)

# Identify categorical features
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
best_iterations = []

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmse_scores.append(rmse)
    best_iterations.append(model.best_iteration)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")

# Train on full training set
dtrain_full = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
avg_best_iter = int(np.mean(best_iterations))
model_full = lgb.train(params, dtrain_full, num_boost_round=avg_best_iter)

# Predict test
test_preds = model_full.predict(test)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": range(len(test)), "DEPSUM": test_preds})
submission.to_csv("./working/submission.csv", index=False)
