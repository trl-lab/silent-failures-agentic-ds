import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "installed_capacity"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in train.columns:
    y = train[TARGET].astype(float)
    X = train.drop(columns=[TARGET])
else:
    y = pd.Series(dtype=float)
    X = train.copy()

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])


def is_obvious_id(col, s):
    c = str(col).lower()
    if c in {"id", "index", "row_id", "idx", "uid", "identifier"}:
        return True
    if c.endswith("_id") and s.nunique(dropna=True) >= 0.95 * max(1, len(s)):
        return True
    return False


id_cols = [c for c in X.columns if is_obvious_id(c, X[c])]
X = X.drop(columns=id_cols, errors="ignore")
test = test.drop(columns=id_cols, errors="ignore")

mask = y.notna()
X = X.loc[mask]
y = y.loc[mask]


def numeric_like(s):
    if pd.api.types.is_numeric_dtype(s):
        return True
    if s.dtype == object or str(s.dtype).startswith("string"):
        converted = pd.to_numeric(s, errors="coerce")
        return converted.notna().mean() > 0.8
    return False


def prepare_features(Xtr, Xval=None):
    Xtr = Xtr.copy()
    if Xval is not None:
        Xval = Xval.copy().reindex(columns=Xtr.columns)
    else:
        Xval = None

    num_cols, cat_cols = [], []
    for c in Xtr.columns:
        s = Xtr[c]
        if numeric_like(s):
            num_cols.append(c)
        elif s.nunique(dropna=True) <= 500:
            cat_cols.append(c)
        # High-cardinality non-numeric columns are dropped in this simple baseline.

    for c in num_cols:
        Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce")
        if Xval is not None:
            Xval[c] = pd.to_numeric(Xval[c], errors="coerce")

    if cat_cols:
        Xtr_cat = Xtr[cat_cols].astype("string").fillna("missing")
        dtr = pd.get_dummies(
            Xtr_cat,
            prefix=[f"cat_{c}" for c in cat_cols],
            dtype=float,
        )
        if Xval is not None:
            Xval_cat = Xval[cat_cols].astype("string").fillna("missing")
            dval = pd.get_dummies(
                Xval_cat,
                prefix=[f"cat_{c}" for c in cat_cols],
                dtype=float,
            )
            dval = dval.reindex(columns=dtr.columns, fill_value=0.0)
        else:
            dval = None
    else:
        dtr = pd.DataFrame(index=Xtr.index)
        dval = pd.DataFrame(index=Xval.index) if Xval is not None else None

    if num_cols:
        ntr = Xtr[num_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        if Xval is not None:
            nval = Xval[num_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            nval = None
    else:
        ntr = pd.DataFrame(index=Xtr.index)
        nval = pd.DataFrame(index=Xval.index) if Xval is not None else None

    out_tr = pd.concat([ntr, dtr], axis=1)
    out_tr.columns = pd.unique(out_tr.columns)
    out_tr = out_tr.loc[:, out_tr.notna().any()]

    if Xval is not None:
        out_val = pd.concat([nval, dval], axis=1)
        out_val.columns = pd.unique(out_val.columns)
        out_val = out_val.reindex(columns=out_tr.columns, fill_value=np.nan)
        return out_tr, out_val

    return out_tr


if len(X) == 0:
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "installed_capacity": 0.0,
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print("CV RMSE: 0.0")
else:
    if len(X) >= 5:
        n_splits = 5
    elif len(X) >= 2:
        n_splits = 2
    else:
        n_splits = 1

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
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    scores = []
    best_iters = []

    if n_splits > 1:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr_idx, va_idx in kf.split(X):
            Xtr_raw = X.iloc[tr_idx]
            ytr = y.iloc[tr_idx]
            Xva_raw = X.iloc[va_idx]
            yva = y.iloc[va_idx]

            Xtr, Xva = prepare_features(Xtr_raw, Xva_raw)

            if Xtr.shape[1] == 0:
                pred = np.full(len(Xva), float(ytr.mean()))
                scores.append(root_mean_squared_error(yva, pred))
                best_iters.append(1)
                continue

            dtrain = lgb.Dataset(Xtr, label=ytr)
            dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

            pred = model.predict(Xva, num_iteration=model.best_iteration)
            scores.append(root_mean_squared_error(yva, pred))
            best_iters.append(model.best_iteration)
    else:
        Xtr = prepare_features(X)
        if Xtr.shape[1] == 0:
            pred = np.full(len(X), float(y.mean()))
        else:
            dtrain = lgb.Dataset(Xtr, label=y)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(-1)],
            )
            pred = model.predict(Xtr)

        scores.append(root_mean_squared_error(y, pred))
        best_iters.append(100)

    cv_rmse = float(np.mean(scores))
    print(f"CV RMSE: {cv_rmse:.6f}")

    Xtr_all, Xte = prepare_features(X, test)

    if Xtr_all.shape[1] == 0:
        pred_test = np.full(len(test), float(y.mean()))
    else:
        final_rounds = max(1, int(np.median(best_iters))) if best_iters else 100
        dtrain = lgb.Dataset(Xtr_all, label=y)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(-1)],
        )
        pred_test = model.predict(Xte)

    if len(y) > 0 and y.min() >= 0:
        pred_test = np.clip(pred_test, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "installed_capacity": pred_test.astype(float),
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print(f"Saved submission with {len(submission)} rows.")
