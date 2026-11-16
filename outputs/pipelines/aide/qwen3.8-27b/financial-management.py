import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

TARGET = "State Share"
SEED = 42


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET} not found in train data.")

    # Clean target: keep only finite numeric values.
    y = pd.to_numeric(train[TARGET], errors="coerce")
    y_values = y.to_numpy(dtype=float)
    valid = np.isfinite(y_values)

    train = train.loc[valid].copy()
    y = pd.Series(y_values[valid], index=train.index, dtype=float)

    X = train.drop(columns=[TARGET])

    # Remove obvious identifier columns if present.
    id_like = [
        c
        for c in X.columns
        if str(c).lower() in {"id", "index", "row_id", "rowid", "idx"}
    ]
    X = X.drop(columns=id_like, errors="ignore")
    test = test.drop(columns=id_like, errors="ignore")

    # Use only features present in both train and test.
    feature_cols = [c for c in X.columns if c in test.columns]
    X = X[feature_cols]
    test = test[feature_cols]

    # Fallback if modelling is not possible.
    if len(y) == 0 or len(feature_cols) == 0 or y.nunique() <= 1:
        rmse = float(y.std(ddof=0)) if len(y) > 1 else 0.0
        print(f"5-Fold CV RMSE: {rmse:.6f}")
        pred = np.full(len(test), float(y.mean()) if len(y) > 0 else 0.0)
        pd.DataFrame({"id": np.arange(len(test)), TARGET: pred}).to_csv(
            "./working/submission.csv", index=False
        )
        return

    # Split features into numeric and categorical.
    cat_cols = []
    num_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    def build_features(X_tr, X_te, medians=None):
        X_tr = X_tr.copy()
        X_te = X_te.copy()

        if num_cols:
            for c in num_cols:
                X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce")
                X_te[c] = pd.to_numeric(X_te[c], errors="coerce")
                X_tr[c] = X_tr[c].replace([np.inf, -np.inf], np.nan)
                X_te[c] = X_te[c].replace([np.inf, -np.inf], np.nan)

            if medians is None:
                medians = X_tr[num_cols].median()
            medians = medians.fillna(0.0)

            X_tr[num_cols] = X_tr[num_cols].fillna(medians)
            X_te[num_cols] = X_te[num_cols].fillna(medians)

            num_tr = X_tr[num_cols].to_numpy(dtype=float)
            num_te = X_te[num_cols].to_numpy(dtype=float)
        else:
            num_tr = np.empty((len(X_tr), 0), dtype=float)
            num_te = np.empty((len(X_te), 0), dtype=float)

        if cat_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_tr = ohe.fit_transform(X_tr[cat_cols].astype(str))
            cat_te = ohe.transform(X_te[cat_cols].astype(str))
        else:
            cat_tr = np.empty((len(X_tr), 0), dtype=float)
            cat_te = np.empty((len(X_te), 0), dtype=float)

        if num_tr.shape[1] == 0:
            return cat_tr, cat_te
        if cat_tr.shape[1] == 0:
            return num_tr, num_te

        return np.hstack((num_tr, cat_tr)), np.hstack((num_te, cat_te))

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "bagging_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_seed": SEED,
        "verbose": -1,
    }

    n_splits = 5 if len(y) >= 5 else max(2, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof = np.zeros(len(y), dtype=float)
    best_iters = []

    for tr_idx, va_idx in kf.split(X):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        medians = X_tr[num_cols].median() if num_cols else None
        X_tr_p, X_va_p = build_features(X_tr, X_va, medians=medians)

        dtrain = lgb.Dataset(X_tr_p, label=y_tr.to_numpy(dtype=float))
        dvalid = lgb.Dataset(X_va_p, label=y_va.to_numpy(dtype=float), reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        oof[va_idx] = model.predict(X_va_p)
        if model.best_iteration:
            best_iters.append(model.best_iteration)

    rmse = root_mean_squared_error(y.to_numpy(dtype=float), oof)
    print(f"{n_splits}-Fold CV RMSE: {rmse:.6f}")

    # Train final model on full training data.
    medians_full = X[num_cols].median() if num_cols else None
    X_tr_p, X_te_p = build_features(X, test, medians=medians_full)

    final_rounds = int(np.median(best_iters)) if best_iters else 500
    final_rounds = max(1, final_rounds)

    dtrain_full = lgb.Dataset(X_tr_p, label=y.to_numpy(dtype=float))
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    pred = final_model.predict(X_te_p)
    pred = np.nan_to_num(
        pred,
        nan=float(y.mean()),
        posinf=float(y.max()),
        neginf=float(y.min()),
    )

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            TARGET: pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
