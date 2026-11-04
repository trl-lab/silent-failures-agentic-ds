import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "IMU Score"
INPUT_DIR = "./input"
WORK_DIR = "./working"


def prepare_data(train: pd.DataFrame, test: pd.DataFrame, target: str):
    train = train.dropna(subset=[target]).reset_index(drop=True)
    test = test.reset_index(drop=True)

    exclude = {target, "id"}
    feature_cols = [c for c in train.columns if c in test.columns and c not in exclude]
    feature_cols = [c for c in feature_cols if train[c].notna().any()]

    cat_cols = []

    # Decide whether non-numeric-looking columns should be treated as numeric.
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            continue

        converted = pd.to_numeric(train[c], errors="coerce")
        non_null = int(train[c].notna().sum())

        if non_null > 0 and converted.notna().sum() / non_null > 0.8:
            train[c] = converted
            test[c] = pd.to_numeric(test[c], errors="coerce")
        else:
            cat_cols.append(c)

    # Align categorical categories using training data only.
    cat_categories = {}
    for c in cat_cols:
        s = train[c].astype("object").fillna("missing")
        cats = s.astype("category").cat.categories

        if "missing" not in cats:
            cats = cats.insert(0, "missing")

        cat_categories[c] = cats
        train[c] = s.astype("category").cat.set_categories(cats)

    for c in cat_cols:
        cats = cat_categories[c]
        s = test[c].astype("object").fillna("missing")
        s = s.where(s.isin(cats), "missing")
        test[c] = s.astype("category").cat.set_categories(cats)

    # Ensure remaining features are numeric and finite.
    for c in feature_cols:
        if c in cat_cols:
            continue

        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")

        train[c] = train[c].replace([np.inf, -np.inf], np.nan)
        test[c] = test[c].replace([np.inf, -np.inf], np.nan)

    return train, test, feature_cols, cat_cols


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    train, test, feature_cols, cat_cols = prepare_data(train, test, TARGET)

    y_train = train[TARGET].astype(float)
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # Fallback if no usable features are found.
    if len(feature_cols) == 0:
        mean_target = float(y_train.mean()) if len(y_train) else 0.0
        if len(y_train):
            cv_rmse = root_mean_squared_error(
                y_train, np.full(len(y_train), mean_target)
            )
        else:
            cv_rmse = np.nan

        print(f"CV RMSE: {cv_rmse:.6f}")

        preds = np.full(len(test), mean_target, dtype=float)
        submission = pd.DataFrame(
            {"id": np.arange(len(test), dtype=int), TARGET: preds}
        )
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        return

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
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    num_round = 3000
    early_stopping_rounds = 50

    n_splits = min(5, len(train))

    if n_splits < 2:
        cv_rmse = np.nan
        best_iters = [num_round]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(X_train), dtype=float)
        best_iters = []

        for tr_idx, va_idx in kf.split(X_train):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            dtrain = lgb.Dataset(
                X_tr,
                label=y_tr,
                categorical_feature=cat_cols,
                free_raw_data=False,
            )
            dvalid = lgb.Dataset(
                X_va,
                label=y_va,
                categorical_feature=cat_cols,
                reference=dtrain,
                free_raw_data=False,
            )

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_round,
                valid_sets=[dvalid],
                valid_metrics=["rmse"],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            best_iter = model.best_iteration
            if best_iter <= 0:
                best_iter = num_round

            best_iters.append(best_iter)
            oof[va_idx] = model.predict(X_va, num_iteration=best_iter)

        cv_rmse = root_mean_squared_error(y_train, oof)

    print(f"CV RMSE: {cv_rmse:.6f}")

    best_iter = int(np.mean(best_iters)) if best_iters else num_round
    best_iter = int(np.clip(best_iter, 1, num_round))

    dtrain_full = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    preds = final_model.predict(X_test, num_iteration=best_iter)
    preds = np.asarray(preds, dtype=float)

    submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: preds})
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
