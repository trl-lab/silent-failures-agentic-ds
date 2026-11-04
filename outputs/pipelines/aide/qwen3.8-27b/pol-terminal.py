import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

TARGET = "CAPACITY"
ID_NAMES = {"id", "index", "row_id", "rowid", "idx", "index_id", "row_index"}
MISSING_TOKEN = "__MISSING__"

os.makedirs("./working", exist_ok=True)


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [c for c in df.columns if str(c).strip().lower() in ID_NAMES]
    return df.drop(columns=cols_to_drop, errors="ignore")


def prepare_encoded(X: pd.DataFrame, encoder, categorical_cols: list) -> pd.DataFrame:
    X = X.copy()
    if categorical_cols and encoder is not None:
        X[categorical_cols] = encoder.transform(X[categorical_cols])
        for c in categorical_cols:
            X[c] = X[c].astype(int).astype("category")
    return X


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = drop_id_columns(train)
test = drop_id_columns(test)

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET!r} not found in train data.")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])

feature_cols = [c for c in train.columns if c != TARGET]

for c in feature_cols:
    if c not in test.columns:
        test[c] = np.nan

# Convert datetime-like columns to integer timestamps.
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(
        train[c]
    ) or pd.api.types.is_datetime64_any_dtype(test[c]):
        train_dt = pd.to_datetime(train[c], errors="coerce")
        test_dt = pd.to_datetime(test[c], errors="coerce")
        train[c] = train_dt.astype(np.int64).where(train_dt.notna(), np.nan)
        test[c] = test_dt.astype(np.int64).where(test_dt.notna(), np.nan)

# Identify categorical columns as non-numeric, non-boolean columns.
categorical_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        continue
    if train[c].dtype == bool or test[c].dtype == bool:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)
        continue
    categorical_cols.append(c)

# Fill categorical missing values with a sentinel so OrdinalEncoder can fit safely.
for c in categorical_cols:
    train[c] = train[c].astype(object).where(train[c].notna(), MISSING_TOKEN)
    test[c] = test[c].astype(object).where(test[c].notna(), MISSING_TOKEN)

# Convert remaining non-categorical features to numeric.
for c in feature_cols:
    if c not in categorical_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")


# Drop all-NaN or constant features based on the training data.
def is_useful(col: str) -> bool:
    s = train[col]
    return s.notna().sum() > 0 and s.nunique(dropna=True) > 1


feature_cols = [c for c in feature_cols if is_useful(c)]
categorical_cols = [c for c in categorical_cols if c in feature_cols]

y = train[TARGET].to_numpy(dtype=float)
X = train[feature_cols].copy()

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
    "n_jobs": -1,
    "verbose": -1,
}

n_splits = 5 if len(train) >= 5 else max(2, len(train))
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rmse_scores = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]

    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(X_train_fold[categorical_cols])
    else:
        encoder = None

    X_train_enc = prepare_encoded(X_train_fold, encoder, categorical_cols)
    X_val_enc = prepare_encoded(X_val_fold, encoder, categorical_cols)

    dtrain = lgb.Dataset(
        X_train_enc, label=y_train_fold, categorical_feature=categorical_cols
    )
    dval = lgb.Dataset(
        X_val_enc,
        label=y_val_fold,
        categorical_feature=categorical_cols,
        reference=dtrain,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iter = model.best_iteration if model.best_iteration > 0 else 3000
    best_iterations.append(best_iter)

    val_pred = model.predict(X_val_enc, num_iteration=best_iter)
    rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, val_pred)))

cv_rmse = float(np.mean(rmse_scores))
cv_std = float(np.std(rmse_scores))
print(f"5-Fold CV RMSE: {cv_rmse:.6f} (std {cv_std:.6f})")

# Final model trained on all training data.
if categorical_cols:
    full_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    full_encoder.fit(train[categorical_cols])
else:
    full_encoder = None

X_train_full_enc = prepare_encoded(train[feature_cols], full_encoder, categorical_cols)
X_test_enc = prepare_encoded(test[feature_cols], full_encoder, categorical_cols)

dtrain_full = lgb.Dataset(
    X_train_full_enc, label=y, categorical_feature=categorical_cols
)
avg_best_iter = int(np.mean(best_iterations)) if best_iterations else 1000
avg_best_iter = max(avg_best_iter, 1)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=avg_best_iter,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test_enc, num_iteration=avg_best_iter)
test_pred = np.nan_to_num(
    test_pred, nan=float(np.mean(y)), posinf=float(np.max(y)), neginf=float(np.min(y))
)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "CAPACITY": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print(f"Saved submission to ./working/submission.csv with {len(submission)} rows.")
