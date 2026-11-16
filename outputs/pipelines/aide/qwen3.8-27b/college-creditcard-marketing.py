import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

TARGET = "PAYMENTS_BY_ISSUER"
ID_COLS = ["id", "ID", "Id"]

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET} not found in train.")

y_train = pd.to_numeric(train[TARGET], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
X_train = train.drop(columns=[TARGET] + [c for c in ID_COLS if c in train.columns])
X_test = test.drop(columns=[c for c in [TARGET] + ID_COLS if c in test.columns])

valid = y_train.notna()
y_train = y_train[valid].reset_index(drop=True)
X_train = X_train.loc[valid].reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

if len(y_train) == 0:
    os.makedirs("./working", exist_ok=True)
    pd.DataFrame({"id": np.arange(len(X_test), dtype=int), TARGET: 0.0}).to_csv(
        "./working/submission.csv", index=False
    )
    print("CV RMSE: nan")
    sys.exit(0)

X_train = X_train.dropna(axis=1, how="all")
features = X_train.columns.tolist()
X_test = X_test.reindex(columns=features).reset_index(drop=True)

numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in features if c not in numeric_cols]

if numeric_cols:
    X_train[numeric_cols] = X_train[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    )
    X_test[numeric_cols] = X_test[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    )
    X_train[numeric_cols] = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)

MISSING = "__MISSING__"

if cat_cols:

    def _stringify(s):
        return s.astype(object).where(s.notna(), MISSING).astype(str)

    X_train[cat_cols] = X_train[cat_cols].apply(_stringify)
    X_test[cat_cols] = X_test[cat_cols].apply(_stringify)


def make_cat_pipe():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING)),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )


def process_pair(X_tr, X_te):
    if numeric_cols:
        X_tr_num = X_tr[numeric_cols].copy()
        keep_num = [c for c in numeric_cols if X_tr_num[c].notna().any()]
        X_tr_num = X_tr_num[keep_num]
        X_te_num = X_te[keep_num].copy()
    else:
        X_tr_num = pd.DataFrame(index=X_tr.index)
        X_te_num = pd.DataFrame(index=X_te.index)

    if cat_cols:
        cat_pipe = make_cat_pipe()
        cat_pipe.fit(X_tr[cat_cols])
        X_tr_cat = pd.DataFrame(
            cat_pipe.transform(X_tr[cat_cols]), columns=cat_cols, index=X_tr.index
        )
        X_te_cat = pd.DataFrame(
            cat_pipe.transform(X_te[cat_cols]), columns=cat_cols, index=X_te.index
        )
    else:
        X_tr_cat = pd.DataFrame(index=X_tr.index)
        X_te_cat = pd.DataFrame(index=X_te.index)

    X_tr_out = pd.concat([X_tr_num, X_tr_cat], axis=1)
    X_te_out = pd.concat([X_te_num, X_te_cat], axis=1)
    return X_tr_out, X_te_out


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

fold_rmses = []
best_iters = []

if len(X_train) >= 2:
    n_splits = min(5, len(X_train))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(X_train):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        X_tr_p, X_val_p = process_pair(X_tr, X_val)

        if X_tr_p.shape[1] == 0:
            pred = np.full(len(y_val), y_tr.mean(), dtype=float)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, pred))))
            best_iters.append(1)
            continue

        train_ds = lgb.Dataset(X_tr_p, label=y_tr)
        valid_ds = lgb.Dataset(X_val_p, label=y_val, reference=train_ds)

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=2000,
            valid_sets=[valid_ds],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        num_iter = model.best_iteration if model.best_iteration > 0 else None
        pred = model.predict(X_val_p, num_iteration=num_iter)

        fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, pred))))
        best_iters.append(max(1, model.best_iteration))

if fold_rmses:
    cv_rmse = float(np.mean(fold_rmses))
    cv_std = float(np.std(fold_rmses))
else:
    cv_rmse = float("nan")
    cv_std = float("nan")

final_rounds = int(np.clip(np.mean(best_iters) if best_iters else 1000, 100, 2000))
X_tr_final, X_te_final = process_pair(X_train, X_test)

if X_tr_final.shape[1] == 0:
    test_preds = np.full(len(X_test), y_train.mean(), dtype=float)
else:
    train_ds = lgb.Dataset(X_tr_final, label=y_train)
    model_final = lgb.train(
        params,
        train_ds,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )
    test_preds = model_final.predict(X_te_final)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(X_test), dtype=int),
        TARGET: test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)

print(f"CV RMSE: {cv_rmse:.6f} (std: {cv_std:.6f})")
