import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "CAPACITY"

# Load data.
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Remove rows with missing target and keep a clean target vector.
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET]).copy()

# Drop identifier-like columns that are not useful features.
id_like = [
    c for c in X.columns if str(c).lower().strip() in {"id", "index", "row_id", "rowid"}
]
X = X.drop(columns=id_like, errors="ignore")
test = test.drop(columns=[TARGET] + id_like, errors="ignore")

# Use only features present in both train and test to avoid shape mismatches.
common_cols = [c for c in X.columns if c in test.columns]
X = X[common_cols].copy()
test = test[common_cols].copy()

# Clean and type-align features.
cat_cols = []
num_cols = []


def mostly_numeric(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    parsed = pd.to_numeric(s.astype(str), errors="coerce")
    return parsed.notna().mean() > 0.9


for col in common_cols:
    train_is_text = X[col].dtype == object or X[col].dtype.name in {
        "category",
        "string",
    }
    test_is_text = test[col].dtype == object or test[col].dtype.name in {
        "category",
        "string",
    }

    if train_is_text or test_is_text:
        # If the column is mostly numeric strings, treat it as numeric.
        if mostly_numeric(X[col]) or mostly_numeric(test[col]):
            X[col] = pd.to_numeric(X[col].astype(str), errors="coerce")
            test[col] = pd.to_numeric(test[col].astype(str), errors="coerce")
            num_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        num_cols.append(col)

# Encode categorical columns using mappings learned from training data only.
cat_mappings = {}
for col in cat_cols:
    values = X[col].dropna().unique()
    cat_mappings[col] = {v: i for i, v in enumerate(values)}
    X[col] = X[col].map(cat_mappings[col]).fillna(-1).astype(np.int32)
    test[col] = test[col].map(cat_mappings[col]).fillna(-1).astype(np.int32)

# Drop columns with no useful signal in the training set.
keep_cols = []
for col in common_cols:
    if X[col].isna().all():
        continue
    if X[col].nunique(dropna=True) <= 1:
        continue
    keep_cols.append(col)

X = X[keep_cols].copy()
test = test[keep_cols].copy()
cat_cols = [c for c in cat_cols if c in keep_cols]
num_cols = [c for c in num_cols if c in keep_cols]

# Optional log1p transform for non-negative, right-skewed targets.
use_log = False
if len(y) > 0 and y.min() >= 0 and y.max() > 0 and y.std() > 0:
    use_log = float(y.skew()) > 1.0

y_model = np.log1p(y.to_numpy()) if use_log else y.to_numpy()


def inverse_transform(pred: np.ndarray) -> np.ndarray:
    if use_log:
        pred = np.expm1(pred)
    if y.min() >= 0:
        pred = np.clip(pred, 0.0, None)
    return pred


# Fallback if no usable features remain.
if len(keep_cols) == 0:
    val_rmse = float(root_mean_squared_error(y, np.full(len(y), y.mean())))
    test_pred = np.full(len(test), y.mean())
    submission = pd.DataFrame(
        {"id": np.arange(len(test), dtype=int), "CAPACITY": test_pred}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Validation RMSE: {val_rmse:.6f}")
    raise SystemExit(0)

# 5-fold cross-validation for RMSE estimation.
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_seed": 42,
    "nthread": -1,
    "verbosity": -1,
}

num_boost_round = 10000
early_stopping_rounds = 50
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_rmses = []
test_pred = np.zeros(len(test), dtype=float)

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y_model[train_idx]
    y_val_original = y.to_numpy()[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(
        X_val, label=y_model[val_idx], categorical_feature=cat_cols, reference=dtrain
    )

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        valid_names=["valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(-1),
        ],
    )

    best_iteration = max(1, model.best_iteration)

    val_pred = model.predict(X_val, num_iteration=best_iteration)
    val_pred = inverse_transform(val_pred)
    fold_rmse = root_mean_squared_error(y_val_original, val_pred)
    fold_rmses.append(fold_rmse)

    test_pred += model.predict(test, num_iteration=best_iteration)

    print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

test_pred /= len(fold_rmses)
test_pred = inverse_transform(test_pred)

submission = pd.DataFrame(
    {"id": np.arange(len(test), dtype=int), "CAPACITY": test_pred}
)
submission.to_csv("./working/submission.csv", index=False)

mean_rmse = float(np.mean(fold_rmses))
std_rmse = float(np.std(fold_rmses))
print(f"Validation RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")
