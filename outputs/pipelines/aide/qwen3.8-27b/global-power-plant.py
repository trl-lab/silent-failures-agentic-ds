import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

TARGET = "capacity_mw"


def safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.dropna().astype(str).nunique(dropna=True))


def main():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")
    os.makedirs("./working", exist_ok=True)

    if TARGET in test_raw.columns:
        test_raw = test_raw.drop(columns=[TARGET])
    if "id" in train_raw.columns:
        train_raw = train_raw.drop(columns=["id"])
    if "id" in test_raw.columns:
        test_raw = test_raw.drop(columns=["id"])

    y_raw = (
        train_raw[TARGET].astype(float)
        if TARGET in train_raw.columns
        else pd.Series(np.nan, index=train_raw.index)
    )
    train_raw = train_raw.drop(columns=[TARGET], errors="ignore")

    valid = y_raw.notna() & np.isfinite(y_raw)
    train_raw = train_raw[valid].reset_index(drop=True)
    y_raw = y_raw[valid].reset_index(drop=True)

    if len(y_raw) == 0:
        print("CV RMSE: 0.000000")
        sub = pd.DataFrame(
            {"id": np.arange(len(test_raw), dtype=int), "capacity_mw": 0.0}
        )
        sub.to_csv("./working/submission.csv", index=False)
        return

    feature_cols = [c for c in train_raw.columns if c in test_raw.columns]
    n_train = len(train_raw)
    kept = []

    for c in feature_cols:
        s = train_raw[c]
        if not s.notna().any():
            continue

        nu = safe_nunique(s)
        if nu <= 1:
            continue

        is_text_like = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        )
        if is_text_like and nu > 0.9 * n_train:
            continue

        kept.append(c)

    feature_cols = kept

    train = train_raw[feature_cols].copy()
    test = test_raw[feature_cols].copy()

    if len(feature_cols) == 0:
        mean_y = float(y_raw.mean())
        cv_rmse = float(np.sqrt(np.mean((y_raw - mean_y) ** 2)))
        print(f"CV RMSE: {cv_rmse:.6f}")

        pred = np.full(len(test), mean_y)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0)

        sub = pd.DataFrame({"id": np.arange(len(test), dtype=int), "capacity_mw": pred})
        sub.to_csv("./working/submission.csv", index=False)
        return

    categorical_cols = []

    for c in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(train[c]):
            train[c] = pd.to_datetime(train[c]).dt.year
            test[c] = pd.to_datetime(test[c]).dt.year
            continue

        if pd.api.types.is_bool_dtype(train[c]):
            train[c] = train[c].astype(int)
            test[c] = test[c].astype(int)
            continue

        if pd.api.types.is_numeric_dtype(train[c]):
            train[c] = pd.to_numeric(train[c], errors="coerce")
            test[c] = pd.to_numeric(test[c], errors="coerce")
            train[c] = train[c].replace([np.inf, -np.inf], np.nan)
            test[c] = test[c].replace([np.inf, -np.inf], np.nan)
            continue

        if pd.api.types.is_categorical_dtype(train[c]):
            train[c] = train[c].astype(object)
            test[c] = test[c].astype(object)
        else:
            train_conv = pd.to_numeric(train[c], errors="coerce")
            test_conv = pd.to_numeric(test[c], errors="coerce")

            if (
                train[c].notna().sum() > 0
                and train_conv.notna().sum() == train[c].notna().sum()
            ):
                train[c] = train_conv
                test[c] = test_conv
                train[c] = train[c].replace([np.inf, -np.inf], np.nan)
                test[c] = test[c].replace([np.inf, -np.inf], np.nan)
                continue

            train[c] = train[c].astype(object)
            test[c] = test[c].astype(object)

        train[c] = train[c].where(train[c].notna(), None)
        test[c] = test[c].where(test[c].notna(), None)

        try:
            cats = pd.unique(pd.concat([train[c], test[c]]))
            cats = [x for x in cats if pd.notna(x)]
            train[c] = pd.Categorical(train[c], categories=cats)
            test[c] = pd.Categorical(test[c], categories=cats)
        except TypeError:
            train_str = train[c].astype(str)
            test_str = test[c].astype(str)
            train_str = train_str.where(train[c].notna(), None)
            test_str = test_str.where(test[c].notna(), None)

            cats = pd.unique(pd.concat([train_str, test_str]))
            cats = [x for x in cats if pd.notna(x)]

            train[c] = pd.Categorical(train_str, categories=cats)
            test[c] = pd.Categorical(test_str, categories=cats)

        categorical_cols.append(c)

    y_log = np.log1p(y_raw.clip(lower=0).to_numpy(dtype=float))

    params = {
        "objective": "reg_l2",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "feature_pre_filter": False,
        "verbosity": -1,
        "seed": 42,
    }

    num_boost_round = 3000
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_preds = np.zeros(len(y_raw), dtype=float)
    best_iters = []

    for tr_idx, va_idx in kf.split(y_raw):
        dtrain = lgb.Dataset(
            train.iloc[tr_idx][feature_cols],
            label=y_log[tr_idx],
            categorical_feature=categorical_cols,
        )
        dval = lgb.Dataset(
            train.iloc[va_idx][feature_cols],
            label=y_log[va_idx],
            categorical_feature=categorical_cols,
            reference=dtrain,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = (
            model.best_iteration if model.best_iteration > 0 else num_boost_round
        )
        best_iters.append(best_iter)

        pred_log = model.predict(
            train.iloc[va_idx][feature_cols], num_iteration=best_iter
        )
        pred = np.expm1(np.clip(pred_log, -700, 700)).clip(min=0)
        val_preds[va_idx] = pred

    cv_rmse = float(np.sqrt(np.mean((y_raw.to_numpy(dtype=float) - val_preds) ** 2)))
    print(f"CV RMSE: {cv_rmse:.6f}")

    final_rounds = max(100, int(np.mean(best_iters)))
    dtrain_full = lgb.Dataset(
        train[feature_cols],
        label=y_log,
        categorical_feature=categorical_cols,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred_log = final_model.predict(test[feature_cols])
    test_pred = np.expm1(np.clip(test_pred_log, -700, 700)).clip(min=0)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "capacity_mw": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
