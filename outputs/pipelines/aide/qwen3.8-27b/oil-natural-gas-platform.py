import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "WATERDEPTH"
RANDOM_STATE = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"


def is_id_like(col: str) -> bool:
    return str(col).lower() in {"id", "index", "row_id", "sample_id", "uuid"}


def preprocess_train(train: pd.DataFrame):
    train = train.dropna(subset=[TARGET])
    y = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
    finite = np.isfinite(y)
    train = train.loc[finite].copy()
    y = y.loc[finite].to_numpy(dtype=float)

    train = train.drop(columns=[TARGET], errors="ignore")
    train = train.drop(
        columns=[c for c in train.columns if is_id_like(c)], errors="ignore"
    )

    feature_cols = []
    numeric_cols = []
    categorical_cols = []
    categories = {}

    for col in train.columns:
        s = train[col]

        if s.isna().all():
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                vals = s.dt.tz_localize(None) if s.dt.tz is not None else s
                int_vals = vals.astype("int64") // 1_000_000_000
                int_vals = int_vals.where(s.notna(), np.nan).astype(float)
            except Exception:
                int_vals = pd.to_numeric(s.astype(str), errors="coerce")
            train[col] = int_vals
            if train[col].nunique(dropna=True) <= 1:
                continue
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        if pd.api.types.is_bool_dtype(s):
            train[col] = s.astype("int8")
            if train[col].nunique(dropna=True) <= 1:
                continue
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            train[col] = pd.to_numeric(train[col], errors="coerce")
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            if train[col].nunique(dropna=True) <= 1:
                continue
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        # Non-numeric column: use numeric if mostly parseable, otherwise categorical.
        try:
            converted = pd.to_numeric(s, errors="coerce")
        except Exception:
            converted = pd.Series(np.nan, index=s.index)

        non_null_count = int(s.notna().sum())
        if (
            non_null_count > 0
            and converted.notna().sum() / max(non_null_count, 1) > 0.8
        ):
            train[col] = converted.replace([np.inf, -np.inf], np.nan)
            if train[col].nunique(dropna=True) <= 1:
                continue
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = s.dropna().unique().tolist()
        else:
            cats = s.dropna().astype(str).unique().tolist()

        if len(cats) <= 1:
            continue

        categories[col] = cats
        train[col] = pd.Categorical(train[col], categories=cats)
        feature_cols.append(col)
        categorical_cols.append(col)

    train = train.reindex(columns=feature_cols)
    return train, y, feature_cols, numeric_cols, categorical_cols, categories


def preprocess_test(
    test: pd.DataFrame, feature_cols, numeric_cols, categorical_cols, categories
):
    test = test.drop(columns=[TARGET], errors="ignore")
    test = test.drop(
        columns=[c for c in test.columns if is_id_like(c)], errors="ignore"
    )
    test = test.reindex(columns=feature_cols)

    for col in feature_cols:
        if col in numeric_cols:
            test[col] = pd.to_numeric(test[col], errors="coerce")
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)
        elif col in categorical_cols:
            if not isinstance(test[col].dtype, pd.CategoricalDtype):
                test[col] = pd.Categorical(test[col], categories=categories[col])
            else:
                test[col] = test[col].cat.set_categories(categories[col])
        else:
            test[col] = pd.to_numeric(test[col], errors="coerce")
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)

    return test


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    train, y, feature_cols, numeric_cols, categorical_cols, categories = (
        preprocess_train(train)
    )
    test = preprocess_test(
        test, feature_cols, numeric_cols, categorical_cols, categories
    )

    if len(feature_cols) == 0 or len(train) == 0:
        cv_rmse = float(np.std(y)) if len(y) > 1 else 0.0
        print(f"CV RMSE: {cv_rmse:.6f}")
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                "WATERDEPTH": np.full(
                    len(test), float(np.mean(y)) if len(y) > 0 else 0.0
                ),
            }
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
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    n_splits = 5 if len(train) >= 5 else max(2, len(train))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.full(len(train), np.nan, dtype=float)
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train), start=1):
        X_train_fold = train.iloc[train_idx]
        X_val_fold = train.iloc[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        dtrain = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            categorical_feature=categorical_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = (
            model.best_iteration
            if model.best_iteration and model.best_iteration > 0
            else 1
        )
        best_iterations.append(best_iter)

        val_preds = model.predict(X_val_fold, num_iteration=best_iter)
        oof_preds[val_idx] = val_preds
        fold_rmse = float(np.sqrt(mean_squared_error(y_val_fold, val_preds)))
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

    cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
    print(f"CV RMSE: {cv_rmse:.6f}")

    final_rounds = max(1, int(np.mean(best_iterations)))
    dtrain_full = lgb.Dataset(
        train,
        label=y,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_preds = final_model.predict(test, num_iteration=final_rounds)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "WATERDEPTH": test_preds,
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print(f"Saved submission to {os.path.join(WORK_DIR, 'submission.csv')}")


if __name__ == "__main__":
    main()
