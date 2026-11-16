import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def main():
    os.makedirs("./working", exist_ok=True)
    TARGET = "number_of_officers"

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    y = pd.to_numeric(train[TARGET], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    train = train.drop(columns=[TARGET])

    id_cols = [
        c
        for c in train.columns
        if str(c).strip().lower() in {"id", "index", "row_id", "rowid", "identifier"}
    ]
    train = train.drop(columns=id_cols, errors="ignore")
    test = test.drop(columns=id_cols, errors="ignore")

    mask = y.notna()
    train = train.loc[mask].reset_index(drop=True)
    y = y[mask].to_numpy(dtype=float)

    if len(y) == 0:
        sub = pd.DataFrame(
            {"id": np.arange(len(test)), "number_of_officers": np.zeros(len(test))}
        )
        sub.to_csv("./working/submission.csv", index=False)
        print("CV RMSE: nan")
        return

    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan
    test = test[train.columns].reset_index(drop=True)

    feature_cols = list(train.columns)
    cat_cols = []

    def datetime_to_int(series):
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        dt = dt.dt.tz_localize(None)
        return dt.astype("int64").where(dt.notna(), np.nan)

    for col in feature_cols:
        if pd.api.types.is_bool_dtype(train[col]):
            test_num = pd.to_numeric(test[col], errors="coerce")
            non_missing = test[col].notna()
            if (
                non_missing.sum() == 0
                or (test_num.notna() & non_missing).sum() / non_missing.sum() >= 0.9
            ):
                train[col] = train[col].astype(float)
                test[col] = test_num.astype(float)
            else:
                train[col] = train[col].astype(object)
                test[col] = test[col].astype(object)
                cat_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(
            train[col]
        ) or pd.api.types.is_datetime64_any_dtype(test[col]):
            train[col] = datetime_to_int(train[col])
            test[col] = datetime_to_int(test[col])
        elif pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_numeric_dtype(
            test[col]
        ):
            train[col] = pd.to_numeric(train[col], errors="coerce").astype(float)
            test[col] = pd.to_numeric(test[col], errors="coerce").astype(float)
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)
        else:
            num_tr = pd.to_numeric(train[col], errors="coerce")
            num_te = pd.to_numeric(test[col], errors="coerce")
            non_missing = train[col].notna()
            use_numeric = False
            if non_missing.sum() > 0:
                use_numeric = (
                    num_tr.notna() & non_missing
                ).sum() / non_missing.sum() >= 0.99
            if use_numeric:
                train[col] = num_tr.astype(float)
                test[col] = num_te.astype(float)
                train[col] = train[col].replace([np.inf, -np.inf], np.nan)
                test[col] = test[col].replace([np.inf, -np.inf], np.nan)
            else:
                train[col] = train[col].astype(object)
                test[col] = test[col].astype(object)
                cat_cols.append(col)

    for col in cat_cols:
        train[col] = train[col].apply(lambda x: None if pd.isna(x) else str(x))
        test[col] = test[col].apply(lambda x: None if pd.isna(x) else str(x))

    feature_cols = [c for c in feature_cols if train[c].nunique(dropna=True) > 1]
    cat_cols = [c for c in cat_cols if c in feature_cols]
    train = train[feature_cols]
    test = test[feature_cols]

    if len(feature_cols) == 0:
        pred = float(y.mean())
        sub = pd.DataFrame({"id": np.arange(len(test)), "number_of_officers": pred})
        sub.to_csv("./working/submission.csv", index=False)
        print("CV RMSE: nan")
        return

    use_log = bool(y.min() >= 0 and y.max() > 0)
    y_model = np.log1p(y) if use_log else y.copy()

    def encode_vals(vals, mapping):
        enc = vals.map(mapping).astype(float)
        missing_code = float(len(mapping))
        unknown_code = float(len(mapping) + 1)
        enc[vals.notna() & enc.isna()] = unknown_code
        enc[vals.isna()] = missing_code
        return enc.astype(int)

    def encode_categoricals(X_tr, X_va, X_te):
        if not cat_cols:
            return (
                X_tr[feature_cols].copy(),
                X_va[feature_cols].copy(),
                X_te[feature_cols].copy(),
            )
        numeric_cols = [c for c in feature_cols if c not in cat_cols]
        out_tr = X_tr[numeric_cols].copy()
        out_va = X_va[numeric_cols].copy()
        out_te = X_te[numeric_cols].copy()
        for col in cat_cols:
            vals_tr = X_tr[col]
            cats = vals_tr.dropna().unique()
            mapping = {c: i for i, c in enumerate(cats)}
            out_tr[col] = encode_vals(vals_tr, mapping)
            out_va[col] = encode_vals(X_va[col], mapping)
            out_te[col] = encode_vals(X_te[col], mapping)
        return out_tr[feature_cols], out_va[feature_cols], out_te[feature_cols]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "verbose": -1,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

    if len(y) < 2:
        pred = float(y.mean())
        sub = pd.DataFrame({"id": np.arange(len(test)), "number_of_officers": pred})
        sub.to_csv("./working/submission.csv", index=False)
        print("CV RMSE: nan")
        return

    n_splits = 5 if len(y) >= 5 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(test))

    def predict_model(model, data):
        best_it = model.best_iteration
        if best_it is None or best_it <= 0:
            best_it = None
        return model.predict(data, num_iteration=best_it)

    for tr_idx, va_idx in kf.split(train):
        X_tr, X_va, X_te = encode_categoricals(
            train.iloc[tr_idx], train.iloc[va_idx], test
        )
        dtrain = lgb.Dataset(X_tr, label=y_model[tr_idx], categorical_feature=cat_cols)
        dvalid = lgb.Dataset(
            X_va, label=y_model[va_idx], categorical_feature=cat_cols, reference=dtrain
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        oof[va_idx] = predict_model(model, X_va)
        if len(test) > 0:
            test_preds += predict_model(model, X_te)

    test_preds /= n_splits

    if use_log:
        max_log = np.log(np.finfo(float).max) - 1.0
        oof = np.expm1(np.clip(oof, 0.0, max_log))
        test_preds = np.expm1(np.clip(test_preds, 0.0, max_log))

    if y.min() >= 0:
        oof = np.clip(oof, 0.0, None)
        test_preds = np.clip(test_preds, 0.0, None)

    rmse = root_mean_squared_error(y, oof)
    print(f"CV RMSE: {rmse:.6f}")

    sub = pd.DataFrame({"id": np.arange(len(test)), "number_of_officers": test_preds})
    sub.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
