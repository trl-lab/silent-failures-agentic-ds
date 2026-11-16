import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "TOTAL_MWH"
SEED = 42

os.makedirs(WORK_DIR, exist_ok=True)
np.random.seed(SEED)

train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET!r} not found in train data.")

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.reset_index(drop=True)

id_like = {"id", "index", "row_id", "rowid", "uid", "uuid"}
feature_cols = [
    col
    for col in train.columns
    if col in test.columns and col != TARGET and col.lower() not in id_like
]

feature_cols = [
    col
    for col in feature_cols
    if train[col].notna().any() and train[col].nunique(dropna=True) > 1
]

if len(feature_cols) == 0:
    fallback = float(train[TARGET].mean()) if len(train) else 0.0
    submission = pd.DataFrame(
        {"id": np.arange(len(test)), TARGET: np.full(len(test), fallback)}
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print("CV RMSE: 0.000000")
    raise SystemExit

cat_cols = []
for col in feature_cols:
    dtype = train[col].dtype
    if dtype == object or dtype == "category" or dtype == bool:
        cat_cols.append(col)

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()
y_train = train[TARGET].astype(float).copy()

for col in feature_cols:
    if col in cat_cols:
        continue
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        X_train[col] = (
            X_train[col].dt.tz_localize(None).astype("int64").astype(float)
            // 1_000_000_000
        )
        X_test[col] = (
            X_test[col].dt.tz_localize(None).astype("int64").astype(float)
            // 1_000_000_000
        )
    else:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")


def to_str_keep_nan(series: pd.Series) -> pd.Series:
    series = series.astype(object)
    series = series.where(series.notna(), None)
    return series.map(lambda x: None if x is None else str(x))


for col in cat_cols:
    train_values = to_str_keep_nan(X_train[col])
    categories = sorted(set(train_values.dropna().unique()))
    X_train[col] = pd.Categorical(train_values, categories=categories).codes.astype(
        np.int32
    )
    X_test[col] = pd.Categorical(
        to_str_keep_nan(X_test[col]), categories=categories
    ).codes.astype(np.int32)

y_min = float(y_train.min())
nonnegative_target = y_min >= 0


def clip_if_needed(preds: np.ndarray) -> np.ndarray:
    if nonnegative_target:
        return np.clip(preds, 0.0, None)
    return preds


params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": SEED,
    "num_threads": -1,
}

n_samples = len(X_train)
if n_samples >= 5:
    n_splits = 5
elif n_samples >= 2:
    n_splits = 2
else:
    n_splits = None

cv_rmses = []
best_iterations = []

if n_splits is not None:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        train_dataset = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=cat_cols if cat_cols else None,
            free_raw_data=False,
        )
        valid_dataset = lgb.Dataset(
            X_va,
            label=y_va,
            reference=train_dataset,
            categorical_feature=cat_cols if cat_cols else None,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=5000,
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = (
            int(model.best_iteration) if model.best_iteration is not None else 5000
        )
        best_iter = max(1, best_iter)
        best_iterations.append(best_iter)

        val_pred = model.predict(X_va, num_iteration=best_iter)
        val_pred = clip_if_needed(np.asarray(val_pred, dtype=float))
        fold_rmse = root_mean_squared_error(y_va, val_pred)
        cv_rmses.append(fold_rmse)
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

if cv_rmses:
    cv_rmse = float(np.mean(cv_rmses))
    print(f"CV RMSE: {cv_rmse:.6f}")
else:
    cv_rmse = 0.0
    print("CV RMSE: 0.000000")

if best_iterations:
    final_num_boost = int(max(1, round(np.mean(best_iterations))))
else:
    final_num_boost = 1000

full_dataset = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_cols if cat_cols else None,
    free_raw_data=False,
)

final_model = lgb.train(
    params,
    full_dataset,
    num_boost_round=final_num_boost,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test, num_iteration=final_num_boost)
test_pred = clip_if_needed(np.asarray(test_pred, dtype=float))

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        TARGET: test_pred,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
