import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

TARGET = "Amount in USD"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUT_DIR = "./working"
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Keep only rows with finite target values.
if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

train = train.dropna(subset=[TARGET])
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train[np.isfinite(train[TARGET])]

y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET]).copy()

if TARGET in test.columns:
    test = test.drop(columns=[TARGET]).copy()
else:
    test = test.copy()

# Remove obvious identifier-like columns.
id_like = {
    "id",
    "index",
    "row_id",
    "rowid",
    "uid",
    "transaction_id",
    "transfer_id",
    "fund_id",
    "account_id",
    "customer_id",
}
drop_id = [c for c in X.columns if str(c).strip().lower() in id_like]
X = X.drop(columns=drop_id)
test = test.drop(columns=[c for c in drop_id if c in test.columns])

# Drop columns that are all-NaN or constant in the training data.
drop_constant_or_nan = []
for c in X.columns:
    if X[c].isna().all():
        drop_constant_or_nan.append(c)
    elif X[c].nunique(dropna=True) <= 1:
        drop_constant_or_nan.append(c)

X = X.drop(columns=drop_constant_or_nan)
test = test.drop(columns=[c for c in drop_constant_or_nan if c in test.columns])

# Convert datetime and date-like object columns to numeric Unix timestamps.
date_cols = []
for c in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[c]):
        date_cols.append(c)
    elif X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
        parsed = pd.to_datetime(X[c], errors="coerce")
        if parsed.notna().mean() > 0.5:
            date_cols.append(c)


def datetime_to_numeric(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")
    out = series.astype("int64") / 1e9
    # NaT becomes a very negative integer after astype; replace with NaN.
    out = out.where(out > -1e18, np.nan)
    return out


for c in date_cols:
    X[c] = datetime_to_numeric(X[c])
    if c in test.columns:
        test[c] = datetime_to_numeric(test[c])
    else:
        test[c] = np.nan

# Identify categorical columns after date conversion.
cat_cols = [
    c
    for c in X.columns
    if X[c].dtype == object
    or pd.api.types.is_string_dtype(X[c])
    or X[c].dtype.name == "category"
]


def align_and_encode(X_tr: pd.DataFrame, X_te: pd.DataFrame, cat_cols: list):
    """
    Encode categoricals using only categories observed in X_tr.
    Numeric columns are coerced to float and left as NaN for LightGBM.
    """
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    # Ensure both frames have exactly the same columns.
    X_te = X_te.reindex(columns=X_tr.columns)

    for c in X_tr.columns:
        if c in cat_cols:
            cats = X_tr[c].dropna().unique()
            if len(cats) == 0:
                X_tr[c] = -1
                X_te[c] = -1
            else:
                # Preserve original category order for determinism.
                cats = list(cats)
                X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
                X_te[c] = pd.Categorical(X_te[c], categories=cats)
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce")
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce")

    cat_used = [c for c in X_tr.columns if X_tr[c].dtype.name == "category"]
    return X_tr, X_te, cat_used


# Target transformation: log1p for non-negative targets, otherwise identity.
use_log_target = bool((y >= 0).all())
y_train_all = np.log1p(y) if use_log_target else y.values


def inverse_target(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if use_log_target:
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


# Fallback for degenerate cases.
if len(X) == 0 or y.nunique() <= 1:
    pred_test = np.full(len(test), float(y.mean()))
    submission = pd.DataFrame(
        {"id": np.arange(len(test), dtype=int), TARGET: inverse_target(pred_test)}
    )
    submission.to_csv(OUT_PATH, index=False)
    print("CV RMSE: 0.0")
    raise SystemExit

# 5-fold cross-validation estimate.
n_splits = min(5, len(X))
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_rmse = []
best_iters = []

params = {
    "objective": "regression_l2",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "num_threads": -1,
}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
    X_tr_raw = X.iloc[tr_idx]
    X_va_raw = X.iloc[va_idx]
    y_tr = y_train_all[tr_idx]
    y_va_orig = y.values[va_idx]

    X_tr, X_va, cat_used = align_and_encode(X_tr_raw, X_va_raw, cat_cols)

    train_set = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=cat_used, free_raw_data=False
    )
    valid_set = lgb.Dataset(
        X_va, label=y_train_all[va_idx], reference=train_set, free_raw_data=False
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    best_iters.append(max(1, int(model.best_iteration)))

    va_pred = model.predict(X_va, num_iteration=model.best_iteration)
    va_pred_orig = inverse_target(va_pred)
    fold_rmse = float(np.sqrt(mean_squared_error(y_va_orig, va_pred_orig)))
    cv_rmse.append(fold_rmse)

    print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

cv_rmse_mean = float(np.mean(cv_rmse))
cv_rmse_std = float(np.std(cv_rmse))
print(f"CV RMSE: {cv_rmse_mean:.6f} (+/- {cv_rmse_std:.6f})")

# Final model trained on all training data.
X_final_tr, X_test, cat_used_final = align_and_encode(X, test, cat_cols)

final_rounds = int(np.clip(np.mean(best_iters), 100, 3000))
final_train_set = lgb.Dataset(
    X_final_tr,
    label=y_train_all,
    categorical_feature=cat_used_final,
    free_raw_data=False,
)

final_model = lgb.train(
    params,
    final_train_set,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test, num_iteration=final_rounds)
test_pred_orig = inverse_target(test_pred)

submission = pd.DataFrame(
    {"id": np.arange(len(test), dtype=int), TARGET: test_pred_orig}
)
submission.to_csv(OUT_PATH, index=False)
print(f"Saved submission to {OUT_PATH}")
