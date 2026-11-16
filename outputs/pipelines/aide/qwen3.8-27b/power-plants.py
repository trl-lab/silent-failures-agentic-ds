import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Create working directory if needed
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "NAMPLT_CAP"
X_train = train.drop(columns=[target])
y_train = train[target]
X_test = test.copy()

# Identify categorical columns
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

# Ensure proper types
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(
        X_va, label=y_va, categorical_feature=cat_cols, reference=dtrain
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    preds_va = model.predict(X_va, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_va, preds_va)
    cv_scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.5f}")

print(f"\nCV Mean RMSE: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")

# Retrain on full training data
dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=5000,
    callbacks=[lgb.log_evaluation(-1)],
)

# Predict test
test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Save submission
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "NAMPLT_CAP": test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
