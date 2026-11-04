import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "current_status"


def clean_target(y: pd.Series) -> pd.Series:
    y = y.copy()
    if y.dtype == bool:
        return y.astype(int)

    non_null = y.dropna()
    if len(non_null) == 0:
        return pd.Series(np.nan, index=y.index, dtype=float)

    uniq = non_null.unique()
    if len(uniq) <= 1:
        return pd.Series(0.0, index=y.index, dtype=float).where(y.notna(), np.nan)

    if len(uniq) == 2:
        try:
            uniq_sorted = sorted(uniq)
        except TypeError:
            uniq_sorted = sorted(uniq, key=str)
        mapping = {v: i for i, v in enumerate(uniq_sorted)}
        return y.map(mapping).astype(float)

    counts = non_null.value_counts()
    mapping = {counts.index[0]: 0}
    for v in counts.index[1:]:
        mapping[v] = 1
    return y.map(mapping).astype(float)


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list
) -> tuple:
    X = train[feature_cols].copy()
    X_test = test.reindex(columns=feature_cols).copy()

    for c in feature_cols:
        if c not in X_test.columns:
            X_test[c] = np.nan

    for c in X.select_dtypes(include=[np.number]).columns:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
    for c in X_test.select_dtypes(include=[np.number]).columns:
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)

    for c in feature_cols:
        if c not in X.columns:
            continue

        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        elif pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        else:
            non_null = int(X[c].notna().sum())
            if non_null > 0:
                conv = pd.to_numeric(X[c], errors="coerce")
                if conv.notna().sum() >= 0.9 * non_null:
                    X[c] = conv
                    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
                    continue

            try:
                cats = X[c].dropna().unique()
            except TypeError:
                X[c] = X[c].astype(str)
                cats = X[c].dropna().unique()

            if len(cats) == 0:
                X[c] = pd.Categorical(X[c], categories=[])
                X_test[c] = pd.Categorical(np.full(len(X_test), np.nan), categories=[])
            else:
                try:
                    X[c] = X[c].astype("category")
                except TypeError:
                    X[c] = X[c].astype(str).astype("category")

                try:
                    X[c] = X[c].cat.set_categories(cats)
                except (TypeError, ValueError):
                    X[c] = pd.Categorical(X[c], categories=cats)

                try:
                    X_test[c] = X_test[c].astype("category").cat.set_categories(cats)
                except (TypeError, ValueError):
                    X_test[c] = pd.Categorical(
                        X_test[c].astype(object).where(X_test[c].isin(cats), np.nan),
                        categories=cats,
                    )

    drop_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=drop_cols)
    X_test = X_test.reindex(columns=X.columns)

    return X, X_test


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = clean_target(train[TARGET])
mask = y.notna()
train = train.loc[mask].reset_index(drop=True)
y = y.loc[mask].astype(int).reset_index(drop=True)

id_like = {
    "id",
    "index",
    "row_id",
    "rowid",
    "uid",
    "user_id",
    "customer_id",
    "account_id",
    "loan_id",
}
id_cols = [c for c in train.columns if str(c).lower() in id_like]
feature_cols = [c for c in train.columns if c != TARGET and c not in id_cols]

X, X_test = prepare_features(train, test, feature_cols)

if X.shape[1] == 0 or y.nunique() < 2:
    cv_auc = 0.5
    test_pred = np.full(len(test), 0.5)
else:
    base_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "n_jobs": -1,
        "verbose": -1,
    }

    seeds = [42, 7, 123]

    min_class = int(y.value_counts().min())
    n_splits = min(5, min_class)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
    else:
        folds = []

    if folds:
        oof_ensemble = np.zeros(len(X))
        test_preds = []

        for seed in seeds:
            params = base_params.copy()
            params["random_state"] = seed

            oof_seed = np.zeros(len(X))
            best_iters = []

            for tr_idx, va_idx in folds:
                dtrain = lgb.Dataset(
                    X.iloc[tr_idx],
                    label=y.iloc[tr_idx],
                    categorical_feature="auto",
                )
                dvalid = lgb.Dataset(
                    X.iloc[va_idx],
                    label=y.iloc[va_idx],
                    categorical_feature="auto",
                    reference=dtrain,
                )

                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=5000,
                    valid_sets=[dvalid],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )

                best_iter = getattr(model, "best_iteration", 0)
                if best_iter is None or best_iter <= 0:
                    best_iter = 1

                best_iters.append(best_iter)
                oof_seed[va_idx] = model.predict(
                    X.iloc[va_idx], num_iteration=best_iter
                )

            oof_ensemble += oof_seed

            final_rounds = int(np.clip(np.mean(best_iters), 100, 3000))
            dtrain_full = lgb.Dataset(X, label=y, categorical_feature="auto")
            final_model = lgb.train(
                params,
                dtrain_full,
                num_boost_round=final_rounds,
                callbacks=[lgb.log_evaluation(-1)],
            )
            test_preds.append(final_model.predict(X_test))

        try:
            cv_auc = roc_auc_score(y, oof_ensemble / len(seeds))
        except Exception:
            cv_auc = 0.5

        test_pred = np.mean(test_preds, axis=0)
    else:
        cv_auc = 0.5
        test_pred = np.full(len(test), 0.5)

test_pred = np.asarray(test_pred, dtype=float)
test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, 1.0)

if not np.isfinite(cv_auc):
    cv_auc = 0.5

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "current_status": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)

print(f"CV ROC AUC: {cv_auc:.6f}")
