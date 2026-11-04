import os
import numpy as np
import pandas as pd
import lightgbm as lgb

WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

FEATURES = [
    "date_ord",
    "year",
    "month",
    "dayofmonth",
    "dayofweek",
    "is_weekend",
    "weekofyear",
    "dayofyear",
    "onpromotion",
    "onpromotion_log",
    "store_nbr",
    "family_code",
    "store_family_code",
]

CATEGORICAL = ["store_nbr", "family_code", "store_family_code"]


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def make_features(df, min_date, family_to_code):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["store_nbr"] = out["store_nbr"].astype(int)
    out["onpromotion"] = out["onpromotion"].fillna(0).astype(int)

    out["family_code"] = out["family"].map(family_to_code).fillna(0).astype(int)
    out["store_family_code"] = out["store_nbr"] * 1000 + out["family_code"]

    out["date_ord"] = (out["date"] - min_date).dt.days.astype(np.int32)
    out["year"] = out["date"].dt.year.astype(np.int32)
    out["month"] = out["date"].dt.month.astype(np.int32)
    out["dayofmonth"] = out["date"].dt.day.astype(np.int32)
    out["dayofweek"] = out["date"].dt.dayofweek.astype(np.int32)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(np.int32)
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(np.int32)
    out["dayofyear"] = out["date"].dt.dayofyear.astype(np.int32)
    out["onpromotion_log"] = np.log1p(out["onpromotion"]).astype(np.float32)

    return out


def main():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train_raw["date"] = pd.to_datetime(train_raw["date"])
    test_raw["date"] = pd.to_datetime(test_raw["date"])

    train_raw = train_raw.dropna(subset=["sales"]).reset_index(drop=True)

    min_date = train_raw["date"].min()
    family_to_code = {
        family: i for i, family in enumerate(sorted(train_raw["family"].unique()))
    }

    train = make_features(train_raw, min_date, family_to_code)
    test = make_features(test_raw, min_date, family_to_code)

    train["sales"] = train["sales"].fillna(0).clip(lower=0)
    y_train = np.log1p(train["sales"]).astype(np.float32)

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()

    for col in FEATURES:
        if col in CATEGORICAL:
            X_train[col] = X_train[col].astype(np.int32)
            X_test[col] = X_test[col].astype(np.int32)
        else:
            X_train[col] = X_train[col].astype(np.float32)
            X_test[col] = X_test[col].astype(np.float32)

    unique_dates = np.sort(train["date"].unique())
    n_dates = len(unique_dates)
    part_size = max(1, n_dates // 6)

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.9,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "seed": 42,
        "verbose": -1,
        "num_threads": -1,
    }

    cv_scores = []
    best_iters = []

    for fold in range(1, 6):
        start_idx = part_size * fold
        end_idx = part_size * (fold + 1) if fold < 5 else n_dates

        if start_idx >= n_dates:
            continue

        val_start = pd.Timestamp(unique_dates[start_idx])
        val_end = (
            pd.Timestamp(unique_dates[end_idx])
            if end_idx < n_dates
            else pd.Timestamp(unique_dates[-1]) + pd.Timedelta(days=1)
        )

        train_mask = train["date"] < val_start
        val_mask = (train["date"] >= val_start) & (train["date"] < val_end)

        if not train_mask.any() or not val_mask.any():
            continue

        dtrain = lgb.Dataset(
            X_train[train_mask],
            label=y_train[train_mask],
            categorical_feature=CATEGORICAL,
        )
        dval = lgb.Dataset(
            X_train[val_mask],
            label=y_train[val_mask],
            reference=dtrain,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=400,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        best_iter = model.best_iteration if model.best_iteration > 0 else 400
        best_iters.append(best_iter)

        pred_log = model.predict(X_train[val_mask], num_iteration=best_iter)
        pred = np.expm1(pred_log)
        score = rmsle(train["sales"].values[val_mask], pred)
        cv_scores.append(score)

    if best_iters:
        final_rounds = int(np.clip(np.mean(best_iters) + 50, 100, 800))
    else:
        final_rounds = 400

    dtrain_full = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=CATEGORICAL,
    )
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred_log = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "sales": test_pred.astype(float),
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

    if cv_scores:
        print(f"Temporal 5-fold CV RMSLE: {np.mean(cv_scores):.6f}")
    else:
        print("No CV scores computed.")


if __name__ == "__main__":
    main()
