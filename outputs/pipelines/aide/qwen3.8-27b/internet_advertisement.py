import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(np.float32)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float32)

    s = s.astype(str).str.strip().str.lower()
    s = s.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "none": np.nan,
            "null": np.nan,
            "true": "1",
            "false": "0",
            "yes": "1",
            "no": "0",
        }
    )
    return pd.to_numeric(s, errors="coerce").astype(np.float32)


def _target_to_binary(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(str)

    if pd.api.types.is_bool_dtype(s):
        return s.astype(np.int8)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        uniq = vals.dropna().unique()
        if len(uniq) == 2:
            return (vals == max(uniq)).astype(np.int8)
        return vals.fillna(0).astype(np.int8)

    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"true": "1", "false": "0", "yes": "1", "no": "0"})

    if s.isin(["ad", "nonad"]).any():
        return (s == "ad").astype(np.int8)

    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().any():
        uniq = vals.dropna().unique()
        if len(uniq) == 2:
            return (vals == max(uniq)).astype(np.int8)
        return vals.fillna(0).astype(np.int8)

    return s.str.startswith("ad").astype(np.int8)


def _save_submission(test_pred: np.ndarray) -> None:
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "class": np.clip(test_pred, 0.0, 1.0),
        }
    )
    os.makedirs("./working", exist_ok=True)
    submission.to_csv("./working/submission.csv", index=False)


def main() -> None:
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.drop(columns=["id"], errors="ignore")
    test = test.drop(columns=["id"], errors="ignore")
    test = test.drop(columns=["class"], errors="ignore")

    if "class" not in train.columns:
        raise ValueError("Target column 'class' not found in train data.")

    y_raw = train.pop("class")
    valid = y_raw.notna()
    train = train.loc[valid].reset_index(drop=True)
    y_raw = y_raw.loc[valid].reset_index(drop=True)

    y = _target_to_binary(y_raw).to_numpy(dtype=np.int8)

    feature_names = list(train.columns)
    test = test.reindex(columns=feature_names)

    for col in feature_names:
        train[col] = _to_numeric_series(train[col])
        test[col] = _to_numeric_series(test[col])

    extra_cols = []
    for col in feature_names:
        if train[col].isna().any():
            indicator = f"{col}__missing"
            train[indicator] = train[col].isna().astype(np.int8)
            test[indicator] = test[col].isna().astype(np.int8)
            extra_cols.append(indicator)

    features = feature_names + extra_cols
    X = train[features].astype(np.float32)
    X_test = test[features].astype(np.float32)

    features = [c for c in features if X[c].notna().any() and X[c].nunique() > 1]
    X = X[features]
    X_test = X_test[features]

    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        test_pred = np.full(len(X_test), 0.5, dtype=np.float32)
        print("ROC AUC: 0.500000")
        _save_submission(test_pred)
        return

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }

    seeds = [42, 123, 234, 345, 456]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof = np.zeros(len(y), dtype=np.float32)
    best_iters_by_seed = {seed: [] for seed in seeds}

    for train_idx, valid_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y[valid_idx]

        fold_pred = np.zeros(len(valid_idx), dtype=np.float32)

        for seed in seeds:
            params = base_params.copy()
            params["random_state"] = seed
            params["seed"] = seed
            params["bagging_seed"] = seed
            params["feature_fraction_seed"] = seed

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

            best_it = getattr(model, "best_iteration", None)
            if best_it is None or best_it <= 0:
                best_it = model.num_trees()

            best_iters_by_seed[seed].append(int(best_it))
            fold_pred += model.predict(X_valid, num_iteration=best_it).reshape(-1)

        oof[valid_idx] = fold_pred / len(seeds)

    auc = roc_auc_score(y, oof)
    print(f"ROC AUC: {auc:.6f}")

    test_pred = np.zeros(len(X_test), dtype=np.float32)

    for seed in seeds:
        final_rounds = max(1, int(np.median(best_iters_by_seed[seed])))

        params = base_params.copy()
        params["random_state"] = seed
        params["seed"] = seed
        params["bagging_seed"] = seed
        params["feature_fraction_seed"] = seed

        model = lgb.train(
            params,
            lgb.Dataset(X, label=y),
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(-1)],
        )

        test_pred += model.predict(X_test).reshape(-1)

    test_pred /= len(seeds)
    _save_submission(test_pred)


if __name__ == "__main__":
    main()
