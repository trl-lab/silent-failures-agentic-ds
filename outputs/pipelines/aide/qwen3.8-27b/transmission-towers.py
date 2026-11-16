import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "SupStruc"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))


def add_datetime_features(
    train: pd.DataFrame, test: pd.DataFrame, col: str, min_dt: pd.Timestamp
) -> None:
    def make_features(s: pd.Series) -> dict:
        return {
            f"{col}_days": (s - min_dt).dt.total_seconds().div(86400.0),
            f"{col}_year": s.dt.year.astype("float64"),
            f"{col}_month": s.dt.month.astype("float64"),
            f"{col}_day": s.dt.day.astype("float64"),
            f"{col}_hour": s.dt.hour.astype("float64"),
            f"{col}_minute": s.dt.minute.astype("float64"),
            f"{col}_second": s.dt.second.astype("float64"),
            f"{col}_dayofweek": s.dt.dayofweek.astype("float64"),
        }

    train_feats = make_features(train[col])
    test_feats = make_features(test[col])
    for k, v in train_feats.items():
        train[k] = v
    for k, v in test_feats.items():
        test[k] = v


def preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str):
    train = train.copy()
    test = test.copy()

    y = None
    if target in train.columns:
        y = train.pop(target).astype("float64")

    id_names = {"id", "index", "row_id", "rowid", "uid", "identifier", "sample_id"}
    drop_id = [c for c in train.columns if str(c).lower() in id_names]
    train = train.drop(columns=drop_id, errors="ignore")
    test = test.drop(columns=drop_id, errors="ignore")

    common = [c for c in train.columns if c in test.columns]
    train = train[common]
    test = test[common]

    if y is not None:
        y = y.iloc[: len(train)]

    date_keys = (
        "date",
        "time",
        "day",
        "month",
        "year",
        "hour",
        "minute",
        "second",
        "timestamp",
        "datetime",
    )
    drop_cols = []

    for col in common:
        if col not in train.columns:
            continue

        tr = train[col]
        is_dt = pd.api.types.is_datetime64_any_dtype(tr)

        if not is_dt and tr.dtype in ("object", "category"):
            name_l = str(col).lower()
            if any(k in name_l for k in date_keys):
                parsed_tr = pd.to_datetime(tr, errors="coerce")
                if parsed_tr.notna().mean() > 0.5:
                    train[col] = parsed_tr
                    test[col] = pd.to_datetime(test[col], errors="coerce")
                    is_dt = True

        if is_dt:
            min_dt = train[col].min(skipna=True)
            if pd.isna(min_dt):
                min_dt = test[col].min(skipna=True)
            if pd.isna(min_dt):
                min_dt = pd.Timestamp("1970-01-01")
            add_datetime_features(train, test, col, min_dt)
            drop_cols.append(col)

    if drop_cols:
        train = train.drop(columns=drop_cols, errors="ignore")
        test = test.drop(columns=drop_cols, errors="ignore")

    cat_cols = []

    for col in train.columns:
        tr = train[col]
        te = test[col]

        if tr.dtype == bool:
            train[col] = tr.astype("int8")
            test[col] = te.astype("int8")
        elif tr.dtype in ("object", "category"):
            sample = tr.dropna()
            if len(sample) > 0:
                parsed_sample = pd.to_numeric(sample, errors="coerce")
                if parsed_sample.notna().mean() > 0.9:
                    train[col] = pd.to_numeric(tr, errors="coerce")
                    test[col] = pd.to_numeric(te, errors="coerce")
                    continue

            train[col] = tr.astype("object").fillna("missing")
            test[col] = te.astype("object").fillna("missing")

            cats = sorted(set(train[col].unique()))
            if "missing" not in cats:
                cats.append("missing")

            test[col] = test[col].where(test[col].isin(cats), "missing")
            train[col] = pd.Categorical(train[col], categories=cats)
            test[col] = pd.Categorical(test[col], categories=cats)
            cat_cols.append(col)
        else:
            if not pd.api.types.is_numeric_dtype(tr):
                train[col] = pd.to_numeric(tr, errors="coerce")
                test[col] = pd.to_numeric(te, errors="coerce")

    drop = []
    for col in train.columns:
        if train[col].isna().all():
            drop.append(col)
        elif train[col].nunique(dropna=True) <= 1:
            drop.append(col)

    if drop:
        train = train.drop(columns=drop, errors="ignore")
        test = test.drop(columns=drop, errors="ignore")

    cat_cols = [c for c in cat_cols if c in train.columns]
    return train, test, y, cat_cols


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train, test, y, cat_cols = preprocess(train, test, TARGET)

    if y is None:
        y = pd.Series(np.zeros(len(train), dtype="float64"))

    mask = y.notna()
    train = train[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if train.shape[1] == 0 or len(y) == 0:
        pred = np.full(len(test), float(y.mean()) if len(y) > 0 else 0.0)
        print("CV RMSE: nan")
        pd.DataFrame({"id": np.arange(len(test)), TARGET: pred}).to_csv(
            os.path.join(WORK_DIR, "submission.csv"), index=False
        )
        return

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbosity": -1,
        "num_threads": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), start=1):
        dtrain = lgb.Dataset(
            train.iloc[tr_idx],
            label=y.iloc[tr_idx],
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            train.iloc[va_idx],
            label=y.iloc[va_idx],
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        pred = model.predict(train.iloc[va_idx], num_iteration=model.best_iteration)
        score = rmse(y.iloc[va_idx], pred)
        cv_scores.append(score)
        best_iters.append(model.best_iteration)
        print(f"Fold {fold} RMSE: {score:.6f}")

    print(f"CV RMSE: {np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")

    final_rounds = int(np.mean(best_iters)) + 50 if best_iters else 1000
    dtrain_all = lgb.Dataset(
        train,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_all,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(test, num_iteration=final_rounds)
    submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: test_pred})
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print(
        f"Saved submission with {len(submission)} rows to {os.path.join(WORK_DIR, 'submission.csv')}"
    )


if __name__ == "__main__":
    main()
