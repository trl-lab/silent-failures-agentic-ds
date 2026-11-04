import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score

TARGET = "Company response to consumer"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

if TARGET in train.columns:
    y = train[TARGET].copy()
    train = train.drop(columns=[TARGET])
else:
    y = pd.Series(index=train.index, dtype=object)

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

# Remove training rows with missing target.
valid_y = y.notna()
train = train.loc[valid_y].reset_index(drop=True)
y = y.loc[valid_y].reset_index(drop=True)
test = test.reset_index(drop=True)

# Keep only columns present in both train and test.
common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols]
test = test[common_cols]

MAX_CAT_UNIQUE = 5000


def is_date_like(s: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(1000)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.5


def datetime_to_days(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return (s - pd.Timestamp("1970-01-01")).dt.days.astype("float64")


def numeric_safe(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    return s.replace([np.inf, -np.inf], np.nan)


numeric_cols = []
categorical_cols = []
drop_cols = []

for col in common_cols:
    s = train[col]
    if s.isna().all():
        drop_cols.append(col)
        continue

    if pd.api.types.is_datetime64_any_dtype(s):
        numeric_cols.append(col)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(col)
    else:
        nuniq = s.nunique(dropna=True)
        if nuniq <= MAX_CAT_UNIQUE:
            categorical_cols.append(col)
        elif is_date_like(s) or ("date" in col.lower() or "time" in col.lower()):
            numeric_cols.append(col)
        else:
            drop_cols.append(col)

for col in drop_cols:
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])

for col in numeric_cols:
    train[col] = (
        datetime_to_days(train[col])
        if pd.api.types.is_datetime64_any_dtype(train[col])
        else numeric_safe(train[col])
    )
    test[col] = (
        datetime_to_days(test[col])
        if pd.api.types.is_datetime64_any_dtype(test[col])
        else numeric_safe(test[col])
    )

for col in categorical_cols:
    cats = train[col].dropna().unique()
    cats = sorted(cats, key=lambda x: str(x))
    dtype = pd.CategoricalDtype(categories=cats, ordered=False)
    train[col] = train[col].astype(dtype)
    test[col] = test[col].astype(dtype)

# Drop constant or all-NaN columns after conversion.
feature_cols = []
for col in train.columns:
    if train[col].isna().all():
        continue
    if train[col].nunique(dropna=True) <= 1:
        continue
    feature_cols.append(col)

train = train[feature_cols]
test = test[feature_cols]
categorical_cols = [c for c in categorical_cols if c in feature_cols]

labels = np.unique(y.values)

# Fallback for degenerate cases.
if len(labels) == 1 or len(feature_cols) == 0:
    cv_acc = (
        1.0 if len(labels) == 1 else accuracy_score(y, np.full(len(y), y.mode()[0]))
    )
    final_pred = np.full(len(test), labels[0] if len(labels) == 1 else y.mode()[0])
    print(f"CV Accuracy: {cv_acc:.6f}")
    pd.DataFrame({"id": np.arange(len(test)), TARGET: final_pred}).to_csv(
        os.path.join(WORK_DIR, "submission.csv"), index=False
    )
    raise SystemExit(0)

label_to_idx = {lab: i for i, lab in enumerate(labels)}
y_idx = np.array([label_to_idx[v] for v in y.values])

base_params = {
    "objective": "multiclass",
    "num_class": len(labels),
    "metric": "multi_error",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
}

# Use stratified 5-fold when class support allows it; otherwise fall back to K-fold.
min_class_count = int(y.value_counts().min())
if min_class_count >= 5:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
else:
    splitter = KFold(
        n_splits=max(2, min(5, min_class_count)), shuffle=True, random_state=42
    )

folds = list(splitter.split(train, y_idx))
seeds = [42, 123, 456, 789, 1024]
fold_accs = []
best_iters_by_seed = {seed: [] for seed in seeds}
num_boost_round = 1000

for tr_idx, va_idx in folds:
    X_tr = train.iloc[tr_idx]
    X_va = train.iloc[va_idx]
    y_tr = y_idx[tr_idx]
    y_va = y_idx[va_idx]

    proba_sum = np.zeros((len(va_idx), len(labels)), dtype=np.float64)

    for seed in seeds:
        params = base_params.copy()
        params["seed"] = seed
        params["feature_fraction_seed"] = seed
        params["bagging_seed"] = seed
        params["data_random_seed"] = seed

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_cols)
        dvalid = lgb.Dataset(
            X_va, label=y_va, categorical_feature=categorical_cols, reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        best_it = model.best_iteration if model.best_iteration > 0 else num_boost_round
        best_iters_by_seed[seed].append(best_it)
        proba_sum += model.predict(X_va, num_iteration=best_it)

    pred = labels[np.argmax(proba_sum, axis=1)]
    fold_accs.append(accuracy_score(y.iloc[va_idx], pred))

cv_acc = float(np.mean(fold_accs))
cv_std = float(np.std(fold_accs))
print(f"CV Accuracy: {cv_acc:.6f} (std={cv_std:.6f})")

if len(test) == 0:
    final_pred = np.array([], dtype=labels.dtype)
else:
    test_proba = np.zeros((len(test), len(labels)), dtype=np.float64)

    for seed in seeds:
        params = base_params.copy()
        params["seed"] = seed
        params["feature_fraction_seed"] = seed
        params["bagging_seed"] = seed
        params["data_random_seed"] = seed

        final_rounds = (
            max(1, int(np.mean(best_iters_by_seed[seed])))
            if best_iters_by_seed[seed]
            else num_boost_round
        )

        dtrain_full = lgb.Dataset(
            train, label=y_idx, categorical_feature=categorical_cols
        )
        final_model = lgb.train(
            params,
            dtrain_full,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(-1)],
        )
        test_proba += final_model.predict(test)

    final_pred = labels[np.argmax(test_proba, axis=1)]

submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: final_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
