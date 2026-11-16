import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import lightgbm as lgb

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop samples where target 'sii' is missing
train_df = train_df.dropna(subset=["sii"]).reset_index(drop=True)
train_df["sii"] = train_df["sii"].astype(int)

# Extract IDs
train_ids = train_df["id"]
test_ids = test_df["id"]

# Identify feature columns
target_col = "sii"
drop_cols = ["id", target_col]
feature_cols = [col for col in train_df.columns if col not in drop_cols]

# Identify categorical features
cat_cols = [
    col
    for col in feature_cols
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category"
]

# Convert categorical columns to category dtype
for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")

X = train_df[feature_cols]
y = train_df[target_col].values
X_test = test_df[feature_cols]


# Threshold optimizer for Quadratic Weighted Kappa
class ThresholdOptimizer:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5]

    def _loss(self, coef, X, y):
        X_p = np.digitize(X, coef)
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        initial_coef = [0.5, 1.5, 2.5]
        res = minimize(self._loss, initial_coef, args=(X, y), method="Nelder-Mead")
        self.coef_ = sorted(res.x)

    def predict(self, X):
        return np.digitize(X, self.coef_)


# 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_estimators": 1000,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / n_splits

# Optimize thresholds on OOF continuous predictions
opt = ThresholdOptimizer()
opt.fit(oof_preds, y)

oof_classes = opt.predict(oof_preds)
oof_qwk = cohen_kappa_score(y, oof_classes, weights="quadratic")

print(f"Optimized Thresholds: {opt.coef_}")
print(f"Out-of-Fold QWK Score: {oof_qwk:.5f}")

# Predict on test set
test_classes = opt.predict(test_preds)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "sii": test_classes.astype(int)})
submission.to_csv("./working/submission.csv", index=False)
print(
    f"Submission saved successfully to ./working/submission.csv with {len(submission)} rows."
)
