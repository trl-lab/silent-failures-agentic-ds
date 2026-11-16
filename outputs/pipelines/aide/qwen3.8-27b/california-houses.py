import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "Total interior livable area"
os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET])
train = train[np.isfinite(train[TARGET])]

feature_cols = []
for col in train.columns:
    if col == TARGET or col == "id":
        continue
    if col not in test.columns:
        continue
    if train[col].nunique(dropna=True) <= 1:
        continue
    feature_cols.append(col)


def is_categorical_column(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        dtype == object
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


cat_cols = [col for col in feature_cols if is_categorical_column(train[col])]
num_cols = [col for col in feature_cols if col not in cat_cols]

MISSING = "__MISSING__"
for col in cat_cols:
    train[col] = train[col].astype("object")
    test[col] = test[col].astype("object")

    train[col] = train[col].where(train[col].notna(), MISSING)
    test[col] = test[col].where(test[col].notna(), MISSING)

    categories = pd.unique(train[col])
    train[col] = pd.Categorical(train[col], categories=categories)
    test[col] = pd.Categorical(test[col], categories=categories)

for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")
    train[col] = train[col].replace([np.inf, -np.inf], np.nan)
    test[col] = test[col].replace([np.inf, -np.inf], np.nan)

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
}

if cat_cols:
    params["categorical_feature"] = cat_cols

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
    dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
    dvalid = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y.iloc[va_idx], preds)

    scores.append(rmse)
    best_iters.append(model.best_iteration)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

print(f"5-fold CV RMSE: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

final_iteration = max(1, int(round(np.mean(best_iters))))
final_model = lgb.train(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=final_iteration,
)

test_preds = final_model.predict(X_test, num_iteration=final_iteration)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
