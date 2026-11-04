import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "defects"
DROP_COLS = {"id", "index", "defects", "problems", "class", "target", "label"}


def encode_target(series: pd.Series) -> pd.Series:
    s = series.copy()

    if s.dtype == bool:
        return s.astype(int)

    if s.dtype == object:
        mapping = {
            "true": 1,
            "false": 0,
            "True": 1,
            "False": 0,
            "t": 1,
            "f": 0,
            "1": 1,
            "0": 0,
        }
        mapped = s.map(mapping)

        if mapped.isna().any():
            numeric = pd.to_numeric(s, errors="coerce")
            mapped = mapped.fillna(numeric)

        return (pd.to_numeric(mapped, errors="coerce").fillna(0) > 0).astype(int)

    numeric = pd.to_numeric(s, errors="coerce").fillna(0)
    return (numeric > 0).astype(int)


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    out = df.reindex(columns=feature_cols)

    if out.shape[1] == 0:
        return out

    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def main() -> None:
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.reset_index(drop=True)

    y = encode_target(train[TARGET])
    feature_cols = [c for c in train.columns if c not in DROP_COLS]

    X = prepare_features(train, feature_cols)
    X_test = prepare_features(test, feature_cols)

    keep = []
    for col in feature_cols:
        if X[col].notna().any() and X[col].nunique(dropna=True) > 1:
            keep.append(col)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 10,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_bynode_seed": 42,
        "num_threads": -1,
    }

    max_rounds = 1000
    early_stopping_rounds = 50
    fallback_rounds = 300

    if not keep:
        print("5-fold CV ROC AUC: 0.500000")
        test_pred = np.full(len(test), 0.5)
    elif y.nunique() < 2:
        print("5-fold CV ROC AUC: 0.500000")
        test_pred = np.full(len(test), 0.5)
    else:
        X = X[keep].copy()
        X_test = X_test[keep].copy()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y), dtype=float)
        best_iters = []

        for train_idx, val_idx in skf.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            if y_train.nunique() < 2:
                oof[val_idx] = float(y_train.mean()) if len(y_train) else 0.5
                best_iters.append(fallback_rounds)
                continue

            if y_val.nunique() < 2:
                dtrain = lgb.Dataset(X_train, label=y_train)
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=fallback_rounds,
                    callbacks=[lgb.log_evaluation(-1)],
                )
                oof[val_idx] = booster.predict(X_val)
                best_iters.append(fallback_rounds)
                continue

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=max_rounds,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

            best_iter = int(getattr(booster, "best_iteration", 0) or 0)
            if best_iter <= 0:
                best_iter = 1

            best_iters.append(best_iter)
            oof[val_idx] = booster.predict(X_val, num_iteration=best_iter)

        oof = np.nan_to_num(oof, nan=0.5, posinf=1.0, neginf=0.0)

        try:
            cv_auc = roc_auc_score(y, oof)
        except ValueError:
            cv_auc = 0.5

        print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

        avg_best = int(round(np.mean(best_iters))) if best_iters else fallback_rounds
        if avg_best < 1:
            avg_best = 1

        dtrain_full = lgb.Dataset(X, label=y)
        final_booster = lgb.train(
            params,
            dtrain_full,
            num_boost_round=avg_best,
            callbacks=[lgb.log_evaluation(-1)],
        )

        test_pred = final_booster.predict(X_test)

    test_pred = np.nan_to_num(
        np.asarray(test_pred, dtype=float),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    test_pred = np.clip(test_pred, 0.0, 1.0)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "defects": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
