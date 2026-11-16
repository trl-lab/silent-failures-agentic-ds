import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ── Directories ──────────────────────────────────────────────
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ── Load data ────────────────────────────────────────────────
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"Train columns: {train.columns.tolist()}")

TARGET = "installed_capacity"

# ── Strip id columns (we rebuild them 0-based) ──────────────
if "id" in train.columns:
    train = train.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])

# ── Separate target ──────────────────────────────────────────
y = train[TARGET].values
X = train.drop(columns=[TARGET])

# If test accidentally carries the target, drop it
if TARGET in test.columns:
    test = test.drop(columns=[TARGET])


# ── Unify dtypes: non-numeric → category, bool → int ───────
def _coerce(df):
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        elif not np.issubdtype(df[c].dtype, np.number):
            df[c] = df[c].astype("category")
    return df


X = _coerce(X)
test = _coerce(test)

# Align category levels between train and test
for c in X.columns:
    if X[c].dtype.name == "category":
        all_cats = pd.Index(
            list(set(X[c].cat.categories) | set(test[c].cat.categories))
            if c in test.columns
            else X[c].cat.categories
        )
        X[c] = X[c].cat.set_categories(all_cats)
        if c in test.columns:
            test[c] = test[c].cat.set_categories(all_cats)

print(f"Number of features: {X.shape[1]}")
print(
    f"Target stats → mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}"
)
print(
    f"Missing in train: {X.isnull().sum().sum()}, in test: {test.isnull().sum().sum()}"
)

# ── Validation split (stratified by target quantile bins) ───
VAL_FRAC = 0.15
rng = np.random.RandomState(42)
idx = rng.permutation(len(X))
n_val = max(1, int(len(X) * VAL_FRAC))
val_idx = idx[:n_val]
train_idx = idx[n_val:]

X_tr, y_tr = X.iloc[train_idx], y[train_idx]
X_va, y_va = X.iloc[val_idx], y[val_idx]
print(f"Train split: {X_tr.shape}, Val split: {X_va.shape}")

# ── LightGBM parameters ─────────────────────────────────────
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "nthread": -1,
    "verbose": -1,
    "seed": 42,
}

# ── Train with early stopping ────────────────────────────────
ds_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
ds_va = lgb.Dataset(X_va, label=y_va, free_raw_data=False)

model = lgb.train(
    params,
    ds_tr,
    num_boost_round=10000,
    valid_sets=[ds_tr, ds_va],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
)

best_iter = model.best_iteration
print(f"Best iteration: {best_iter}")

# ── Validation RMSE (official metric) ───────────────────────
val_pred = model.predict(X_va, num_iteration=best_iter)
val_rmse = float(np.sqrt(mean_squared_error(y_va, val_pred)))
print(f"Validation RMSE: {val_rmse:.6f}")

# ── Retrain on the FULL training set with the chosen rounds ──
ds_full = lgb.Dataset(X, label=y, free_raw_data=False)
final_model = lgb.train(
    params,
    ds_full,
    num_boost_round=best_iter,
)

# ── Test predictions ─────────────────────────────────────────
test_pred = final_model.predict(test, num_iteration=best_iter)

# Capacity is physically non-negative; clip any tiny negative noise
test_pred = np.clip(test_pred, 0.0, None)

# ── Write submission ─────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "installed_capacity": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved → ./submission/submission.csv  ({len(submission)} rows)")
print(
    f"Pred  → min={test_pred.min():.4f}  max={test_pred.max():.4f}  mean={test_pred.mean():.4f}"
)

# ── Required final line ──────────────────────────────────────
print(f"Final Validation Score: {val_rmse}")
