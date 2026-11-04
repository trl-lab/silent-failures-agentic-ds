import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Original Principal Amount (US$)"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Basic cleaning: keep rows with target and reset indices.
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.reset_index(drop=True)

train[TARGET] = train[TARGET].astype(float)
if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

# Remove obvious identifier columns.
id_like = [
    c
    for c in train.columns
    if str(c).lower() in {"id", "index", "idx", "row_id", "rowid"}
]
train = train.drop(columns=id_like, errors="ignore")
test = test.drop(columns=id_like, errors="ignore")

# Use only features present in both train and test.
feature_cols = [c for c in train.columns if c != TARGET and c in test.columns]

# Replace infinities with NaN; LightGBM handles missing values natively.
for df in (train, test):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Identify categorical columns before encoding.
cat_cols = []
for c in feature_cols:
    if (
        train[c].dtype == object
        or train[c].dtype == "category"
        or pd.api.types.is_string_dtype(train[c])
    ):
        cat_cols.append(c)

# Convert booleans to integers.
for c in feature_cols:
    if pd.api.types.is_bool_dtype(train[c]):
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)

# Drop columns with no training information.
feature_cols = [c for c in feature_cols if not train[c].isna().all()]
cat_cols = [c for c in cat_cols if c in feature_cols]


def encode_categorical(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Encode categorical values using only categories observed in train."""
    try:
        categories = train_col.dropna().unique()
        mapping = {v: i for i, v in enumerate(categories)}
        train_enc = train_col.map(mapping).fillna(-1).astype(int)
        test_enc = test_col.map(mapping).fillna(-1).astype(int)
    except TypeError:
        # Fallback for unusual unhashable objects: encode by string representation.
        train_str = train_col.astype(str)
        test_str = test_col.astype(str)
        categories = train_str[train_str != "nan"].unique()
        mapping = {v: i for i, v in enumerate(categories)}
        train_enc = train_str.map(mapping).fillna(-1).astype(int)
        test_enc = test_str.map(mapping).fillna(-1).astype(int)

    return train_enc, test_enc


for c in cat_cols:
    train[c], test[c] = encode_categorical(train[c], test[c])

# If no usable features remain, output the training mean.
if len(feature_cols) == 0:
    cv_rmse = float(np.nan)
    test_pred = np.full(len(test), float(train[TARGET].mean()))
    print(f"CV RMSE: {cv_rmse:.6f}")
    pd.DataFrame(
        {
            "id": np.arange(len(test)),
            TARGET: test_pred,
        }
    ).to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    raise SystemExit(0)

# Target transformation: log1p for non-negative targets.
use_log = bool((train[TARGET] >= 0).all())
y = np.log1p(train[TARGET]) if use_log else train[TARGET].to_numpy(dtype=float)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "n_jobs": -1,
    "seed": 42,
}


def inverse_transform(pred: np.ndarray) -> np.ndarray:
    if use_log:
        pred = np.expm1(pred)
        pred = np.clip(pred, 0.0, None)
    return np.nan_to_num(
        pred, nan=float(np.nanmean(y) if not use_log else train[TARGET].mean())
    )


n_splits = 5 if len(train) >= 50 else (2 if len(train) >= 2 else 1)
fold_rmses = []
best_iterations = []

if n_splits >= 2:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(train):
        dtrain = lgb.Dataset(
            train.loc[tr_idx, feature_cols],
            label=y[tr_idx],
            categorical_feature=cat_cols,
        )
        dvalid = lgb.Dataset(
            train.loc[va_idx, feature_cols],
            label=y[va_idx],
            reference=dtrain,
            categorical_feature=cat_cols,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        pred = model.predict(
            train.loc[va_idx, feature_cols], num_iteration=model.best_iteration
        )
        pred = inverse_transform(pred)
        true = train.loc[va_idx, TARGET].to_numpy(dtype=float)
        fold_rmses.append(mean_squared_error(true, pred, squared=False))
        best_iterations.append(max(1, int(model.best_iteration)))

    cv_rmse = float(np.mean(fold_rmses))
    final_rounds = max(10, int(np.mean(best_iterations)))
else:
    cv_rmse = float("nan")
    final_rounds = 500

print(f"CV RMSE: {cv_rmse:.6f}")

# Retrain on all training data and predict test.
dtrain_full = lgb.Dataset(
    train[feature_cols],
    label=y,
    categorical_feature=cat_cols,
)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
)

test_pred = final_model.predict(test[feature_cols], num_iteration=final_rounds)
test_pred = inverse_transform(test_pred)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        TARGET: test_pred,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
