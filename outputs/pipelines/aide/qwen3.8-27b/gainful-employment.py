import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "Debt-to-Earnings Annual Rate"
os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

id_like = {"id", "index", "row_id", "index_id", "uid", "identifier"}
drop_cols = [c for c in train.columns if str(c).lower() in id_like and c != TARGET]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

y = pd.to_numeric(train[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = train.drop(columns=[TARGET])

valid_mask = y.notna()
X = X.loc[valid_mask]
y = y.loc[valid_mask]

common_cols = [c for c in X.columns if c in test.columns]
if len(common_cols) == 0:
    common_cols = X.columns.tolist()
    test = test.reindex(columns=common_cols)
else:
    X = X[common_cols]
    test = test[common_cols]


def infer_feature_types(df):
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            numeric_cols.append(col)
        else:
            converted = pd.to_numeric(s, errors="coerce")
            non_null = s.notna()

            if (
                non_null.sum() > 0
                and (converted.notna() & non_null).sum() / non_null.sum() >= 0.8
            ):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols


_, categorical_cols = infer_feature_types(X)


def preprocess(df, cat_cols, categories=None, fit=False):
    out = df.copy()

    if fit:
        categories = {}
    elif categories is None:
        categories = {}

    for col in out.columns:
        s = out[col]

        if col in cat_cols:
            if fit:
                cats = pd.unique(s.dropna())

                if len(cats) == 0:
                    out[col] = np.nan
                    categories[col] = []
                    continue

                try:
                    cats = sorted(cats, key=lambda x: str(x))
                except Exception:
                    cats = list(cats)

                categories[col] = cats
                out[col] = pd.Categorical(s, categories=cats)
            else:
                cats = categories.get(col, [])

                if len(cats) == 0:
                    out[col] = np.nan
                else:
                    out[col] = pd.Categorical(s, categories=cats)
        else:
            if pd.api.types.is_datetime64_any_dtype(s):
                if getattr(s.dtype, "tz", None) is not None:
                    s = s.dt.tz_localize(None)

                int_vals = s.astype("int64")
                out[col] = int_vals.where(s.notna(), np.nan)
            else:
                out[col] = pd.to_numeric(s, errors="coerce")

    for col in out.columns:
        if not pd.api.types.is_categorical_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out, categories


n = len(y)
oof = np.full(n, np.nan)

N_SEEDS = 5
SEEDS = [42, 43, 44, 45, 46]

configs = [
    {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "bagging_fraction": 0.9,
        "feature_fraction": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
    {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "bagging_fraction": 0.85,
        "feature_fraction": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
    },
]
N_CONFIGS = len(configs)
best_iters = {i: [] for i in range(N_CONFIGS)}


def make_params(cfg, seed):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_jobs": -1,
        "max_depth": -1,
        "bagging_freq": 1,
    }
    params.update(cfg)
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    params["data_random_seed"] = seed
    return params


if n >= 2 and len(X.columns) > 0:
    n_splits = 5 if n >= 5 else n
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(X):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        X_tr_proc, cats = preprocess(X_tr, categorical_cols, fit=True)
        X_va_proc, _ = preprocess(X_va, categorical_cols, cats, fit=False)

        val_preds = np.zeros(len(y_va), dtype=float)

        for cfg_idx, cfg in enumerate(configs):
            for seed in SEEDS:
                dtrain = lgb.Dataset(X_tr_proc, label=y_tr, categorical_feature="auto")
                dvalid = lgb.Dataset(
                    X_va_proc,
                    label=y_va,
                    reference=dtrain,
                    categorical_feature="auto",
                )

                model = lgb.train(
                    make_params(cfg, seed),
                    dtrain,
                    num_boost_round=5000,
                    valid_sets=[dvalid],
                    valid_names=["valid"],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
                )

                best_iter = getattr(model, "best_iteration", 0)
                if best_iter is None or best_iter <= 0:
                    best_iter = 100

                best_iters[cfg_idx].append(int(best_iter))
                val_preds += model.predict(X_va_proc, num_iteration=best_iter)

        oof[va_idx] = val_preds / (N_SEEDS * N_CONFIGS)
else:
    oof = y.values.astype(float).copy()
    best_iters = {i: [100] for i in range(N_CONFIGS)}

cv_rmse = float(root_mean_squared_error(y, oof)) if n > 0 else 0.0
print(f"CV RMSE: {cv_rmse:.6f}")

if len(test) == 0:
    pred = np.array([], dtype=float)
elif n > 0 and len(X.columns) > 0:
    X_train_proc, cats_full = preprocess(X, categorical_cols, fit=True)
    X_test_proc, _ = preprocess(test, categorical_cols, cats_full, fit=False)

    final_preds = np.zeros(len(test), dtype=float)

    for cfg_idx, cfg in enumerate(configs):
        final_rounds = (
            int(np.mean(best_iters[cfg_idx])) if best_iters[cfg_idx] else 1000
        )

        for seed in SEEDS:
            dtrain_full = lgb.Dataset(X_train_proc, label=y, categorical_feature="auto")
            final_model = lgb.train(
                make_params(cfg, seed),
                dtrain_full,
                num_boost_round=max(1, final_rounds),
            )
            final_preds += final_model.predict(X_test_proc)

    pred = final_preds / (N_SEEDS * N_CONFIGS)
else:
    pred = np.full(len(test), float(y.mean()) if n > 0 else 0.0)

pred = np.asarray(pred, dtype=float)
pred = np.where(np.isfinite(pred), pred, np.nan)

fill_value = float(y.mean()) if n > 0 else 0.0
if np.isnan(pred).any():
    pred = np.nan_to_num(pred, nan=fill_value)

submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: pred})
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
