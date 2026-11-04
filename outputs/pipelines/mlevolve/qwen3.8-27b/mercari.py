import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
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
print(f"Train columns: {list(train.columns)}")

# ── Separate target & drop id columns ────────────────────────
target = "price"
y = train[target].values

# Identify id-like columns to drop (keep target for now)
id_like = [
    c
    for c in train.columns
    if c.lower().replace("_", "").replace(" ", "")
    in ("itemid", "id", "trainid", "listingid")
]
drop_train = [c for c in id_like if c != target] + [target]
X = train.drop(columns=drop_train, errors="ignore").reset_index(drop=True)

drop_test = [c for c in drop_train if c in test.columns]
test_df = test.drop(columns=drop_test, errors="ignore").reset_index(drop=True)

# ── Align columns between train and test ─────────────────────
common_cols = [c for c in X.columns if c in test_df.columns]
X = X[common_cols].reset_index(drop=True)
test_df = test_df[common_cols].reset_index(drop=True)

# ── Validation split (15 % hold-out) ─────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"X_train: {X_tr.shape}  |  X_val: {X_val.shape}")

# ── Feature type detection ───────────────────────────────────
cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_tr.columns if c not in cat_cols]

print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
print(f"Numerical   cols ({len(num_cols)}): {num_cols}")

# ── Encode categoricals (fit on train only) ──────────────────
for col in cat_cols:
    X_tr[col] = X_tr[col].astype(str)
    X_val[col] = X_val[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    le = LabelEncoder()
    le.fit(X_tr[col])
    X_tr[col] = le.transform(X_tr[col])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    X_val[col] = X_val[col].map(mapping).fillna(-1).astype(int)
    test_df[col] = test_df[col].map(mapping).fillna(-1).astype(int)

# ── Impute numerical NaN with train median ───────────────────
for col in num_cols:
    med = X_tr[col].median()
    X_tr[col] = X_tr[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)
    test_df[col] = test_df[col].fillna(med)

# ── Train LightGBM with early stopping ───────────────────────
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
print(f"Best iteration: {model.best_iteration}")

# ── Validation RMSE (official metric) ────────────────────────
val_pred = np.clip(model.predict(X_val), 0.0, None)
val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
print(f"Validation RMSE: {val_rmse:.4f}")

# ── Test predictions ─────────────────────────────────────────
test_pred = np.clip(model.predict(test_df), 0.0, None)

# ── Write submission ─────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": range(len(test_pred)),
        "price": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")
print(f"Final Validation Score: {val_rmse}")