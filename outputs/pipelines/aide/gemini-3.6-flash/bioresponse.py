import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target and feature columns
target_col = "Activity"
features = [c for c in train.columns if c != target_col and c != "id"]

X = train[features].values
y = train[target_col].values
X_test = test[features].values

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

cv_log_loss = log_loss(y, oof_preds)
print(f"5-Fold CV Log Loss: {cv_log_loss:.5f}")

# Ensure output directory exists and save test predictions
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test)), "Activity": test_preds})
sub.to_csv("./working/submission.csv", index=False)
