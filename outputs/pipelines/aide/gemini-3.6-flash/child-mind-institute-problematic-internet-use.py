import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Filter out rows missing target
train_df = train_df.dropna(subset=["sii"]).reset_index(drop=True)
train_df["sii"] = train_df["sii"].astype(int)

# Identify features present in both train and test sets
ignore_cols = {"id", "sii"}
feature_cols = [
    c for c in test_df.columns if c not in ignore_cols and c in train_df.columns
]

# Convert categorical/object columns
for col in feature_cols:
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

X = train_df[feature_cols]
y = train_df["sii"].values
X_test = test_df[feature_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1000,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / 5.0


class OptimizedRounder:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5]

    def _kappa_loss(self, coef, X, y):
        preds = np.digitize(X, np.sort(coef))
        return -cohen_kappa_score(y, preds, weights="quadratic")

    def fit(self, X, y):
        initial_coef = [0.5, 1.5, 2.5]
        res = minimize(
            self._kappa_loss, initial_coef, args=(X, y), method="Nelder-Mead"
        )
        self.coef_ = np.sort(res.x)

    def predict(self, X):
        return np.digitize(X, self.coef_)


rounder = OptimizedRounder()
rounder.fit(oof_preds, y)

oof_discrete = rounder.predict(oof_preds)
score = cohen_kappa_score(y, oof_discrete, weights="quadratic")
print(f"OOF QWK Score: {score:.5f}")

test_discrete = rounder.predict(test_preds)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "sii": test_discrete})
submission.to_csv("./working/submission.csv", index=False)
