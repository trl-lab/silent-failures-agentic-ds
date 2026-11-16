import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)

TARGET = "antibody_truth"
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

y_raw = train[TARGET].copy() if TARGET in train.columns else pd.Series(dtype=float)
train = train.drop(columns=[TARGET], errors="ignore")


def clean_target(y):
    if y is None or len(y) == 0:
        return pd.Series(dtype=int)

    if y.dtype == object or y.dtype.name in {"string", "category"}:
        y = y.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "true": 1,
            "positive": 1,
            "yes": 1,
            "t": 1,
            "pos": 1,
            "0": 0,
            "false": 0,
            "negative": 0,
            "no": 0,
            "f": 0,
            "neg": 0,
        }
        y = y.map(mapping)

    y = pd.to_numeric(y, errors="coerce")
    y = y.dropna()

    if len(y) == 0:
        return pd.Series(dtype=int)

    if set(y.unique()).issubset({0, 1}):
        y = y.astype(int)
    else:
        y = (y > 0).astype(int)

    return y


y = clean_target(y_raw)

if len(y) == 0:
    train = train.iloc[0:0].reset_index(drop=True)
    y = pd.Series(dtype=int)
else:
    train = train.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

id_like = {"id", "index", "row_id", "sample_id", "unnamed: 0"}
drop_cols = [c for c in train.columns if str(c).strip().lower() in id_like]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

common = [c for c in train.columns if c in test.columns]
train = train[common]
test = test[common]

cat_cols = []

for col in common:
    tr = train[col]
    te = test[col]

    if pd.api.types.is_bool_dtype(tr):
        train[col] = tr.astype(int)
        test[col] = te.astype(int)

    elif pd.api.types.is_datetime64_any_dtype(tr):
        valid = tr.dropna()
        if len(valid) > 0:
            min_dt = valid.min()
            train[col] = (tr - min_dt).dt.total_seconds().astype("float64") / 86400.0
            test[col] = (te - min_dt).dt.total_seconds().astype("float64") / 86400.0
        else:
            train[col] = np.nan
            test[col] = np.nan

    elif pd.api.types.is_numeric_dtype(tr):
        train[col] = pd.to_numeric(train[col], errors="coerce").astype("float64")
        test[col] = pd.to_numeric(test[col], errors="coerce").astype("float64")
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

    else:
        tr_str = tr.astype(str)
        te_str = te.astype(str)
        tr_notna = tr.notna()
        te_notna = te.notna()

        unique = tr_str[tr_notna].unique()

        if len(unique) == 0:
            train[col] = np.nan
            test[col] = np.nan
            continue

        unk = "__UNK__"
        if unk in unique:
            categories = list(unique)
        else:
            categories = list(unique) + [unk]

        tr_map = tr_str.where(tr_notna, unk)
        te_map = te_str.where(te_notna, unk)
        te_map = te_map.where(te_map.isin(categories), unk)

        train[col] = pd.Categorical(tr_map, categories=categories)
        test[col] = pd.Categorical(te_map, categories=categories)
        cat_cols.append(col)

keep = []
for col in common:
    if col not in train.columns:
        continue
    if train[col].isna().all():
        continue
    if train[col].nunique() <= 1:
        continue
    keep.append(col)

train = train[keep]
test = test[keep]
cat_cols = [c for c in cat_cols if c in keep]

X = train
y = y.astype(int) if len(y) else pd.Series(dtype=int)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": -1,
    "min_data_in_leaf": 5,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "seed": 42,
    "verbose": -1,
}

cv_auc = 0.5
best_iters = []

if len(X) > 0 and y.nunique() >= 2 and len(keep) > 0:
    min_class = int(y.value_counts().min())
    n_splits = min(5, min_class)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_pred = np.full(len(X), np.nan)

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            if y_va.nunique() < 2:
                oof_pred[va_idx] = 0.5
                continue

            cat_feature = cat_cols if cat_cols else None

            dtrain = lgb.Dataset(
                X_tr,
                label=y_tr,
                categorical_feature=cat_feature,
                free_raw_data=False,
            )
            dvalid = lgb.Dataset(
                X_va,
                label=y_va,
                reference=dtrain,
                categorical_feature=cat_feature,
                free_raw_data=False,
            )

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

            num_iteration = model.best_iteration if model.best_iteration else None
            pred = model.predict(X_va, num_iteration=num_iteration)

            oof_pred[va_idx] = pred
            best_iters.append(model.best_iteration)

        if np.isfinite(oof_pred).all():
            cv_auc = float(roc_auc_score(y, oof_pred))

if len(test) == 0:
    test_pred = np.array([])
elif len(X) == 0 or y.nunique() < 2 or len(keep) == 0:
    test_pred = np.full(len(test), 0.5)
else:
    rounds = int(np.clip(np.mean(best_iters) + 50, 100, 2000)) if best_iters else 500

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_cols if cat_cols else None,
        free_raw_data=False,
    )

    full_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = full_model.predict(test, num_iteration=rounds)
    test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "antibody_truth": test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)

print(f"CV ROC AUC: {cv_auc:.6f}")
