import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

TARGET = "Disbursements USD (Trust funds)"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Ensure target is numeric and remove rows with missing target.
if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train[TARGET] = train[TARGET].replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

# Exclude target and obvious identifier/index columns from features.
exclude = {TARGET, "id", "Id", "ID", "index", "Index"}
common_cols = [c for c in train.columns if c in test.columns and c not in exclude]

# Fallback if column names do not align perfectly.
if not common_cols:
    common_cols = [c for c in train.columns if c not in exclude]

X = train[common_cols].copy()
y = train[TARGET].astype(float).copy()
X_test = test[common_cols].copy() if common_cols else pd.DataFrame(index=test.index)

# Identify likely categorical columns.
cat_cols = []
identifier_tokens = {
    "id",
    "code",
    "name",
    "type",
    "category",
    "country",
    "recipient",
    "project",
    "sector",
    "currency",
    "region",
    "bank",
    "agency",
    "instrument",
    "status",
    "mode",
    "source",
    "fund",
    "program",
}

for col in common_cols:
    s = X[col]
    if pd.api.types.is_bool_dtype(s):
        cat_cols.append(col)
    elif (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        cat_cols.append(col)
    else:
        lower = col.lower()
        if any(token in lower for token in identifier_tokens):
            cat_cols.append(col)

cat_cols = list(dict.fromkeys(cat_cols))  # preserve order and remove duplicates


# Basic cleaning: numeric coercion for non-categoricals, string cleaning for categoricals.
def clean_frame(df, cat_cols):
    out = df.copy()
    for col in out.columns:
        if col in cat_cols:
            out[col] = out[col].astype("string").fillna("missing").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


X = clean_frame(X, cat_cols)
X_test = clean_frame(X_test, cat_cols)

# Align categorical levels using training data only to avoid test leakage.
for col in cat_cols:
    if col not in X.columns:
        continue
    train_categories = sorted(set(X[col].astype(str).unique()))
    X[col] = X[col].astype("category").cat.set_categories(train_categories)
    X_test[col] = (
        X_test[col]
        .astype(str)
        .where(X_test[col].isin(train_categories), "missing")
        .astype("category")
        .cat.set_categories(train_categories)
    )

# Drop features that are all-NaN or constant in the training set.
keep_cols = []
for col in common_cols:
    if col not in X.columns:
        continue
    if X[col].isna().all():
        continue
    if X[col].nunique(dropna=True) <= 1:
        continue
    keep_cols.append(col)

common_cols = keep_cols
cat_cols = [c for c in cat_cols if c in common_cols]
X = X[common_cols]
X_test = X_test[common_cols]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def sanitize_predictions(pred, non_negative):
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    if non_negative:
        pred = np.clip(pred, 0.0, None)
    return pred


non_negative_target = bool((y >= 0).all())

# If no usable features remain, output a simple baseline.
if len(common_cols) == 0:
    cv_rmse = rmse(y, np.zeros_like(y))
    test_pred = np.zeros(len(test), dtype=float)
else:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
        "force_row_wise": True,
    }

    n_splits = 5 if len(X) >= 5 else max(2, len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = []
    best_iterations = []

    for train_idx, valid_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        dtrain = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid_fold,
            label=y_valid_fold,
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        num_iteration = model.best_iteration if model.best_iteration > 0 else 2000
        valid_pred = model.predict(X_valid_fold, num_iteration=num_iteration)
        valid_pred = sanitize_predictions(valid_pred, non_negative_target)

        cv_scores.append(rmse(y_valid_fold, valid_pred))
        if model.best_iteration > 0:
            best_iterations.append(model.best_iteration)

    cv_rmse = float(np.mean(cv_scores))

    # Retrain on the full training set using a reasonable number of rounds from CV.
    if best_iterations:
        final_rounds = int(np.median(best_iterations)) + 10
    else:
        final_rounds = 2000

    final_rounds = max(50, final_rounds)

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = sanitize_predictions(test_pred, non_negative_target)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_pred,
    }
)

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(f"5-fold CV RMSE: {cv_rmse:.6f}")
