import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "Size of Purchase (MW)"
SEED = 42


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a series to numeric values, replacing non-finite values with NaN."""
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return s.astype("int64") // 1_000_000_000
        except Exception:
            pass

    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype("string")
        s = s.str.replace(r"[$,]", "", regex=True).str.strip()

    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def infer_types(X_fit: pd.DataFrame, cols: list) -> tuple:
    """Infer numeric vs categorical columns using only the fit data."""
    numeric_cols = []
    cat_cols = []

    for col in cols:
        s = X_fit[col]

        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            numeric_cols.append(col)
            continue

        nonnull = s.dropna()
        if len(nonnull) == 0:
            cat_cols.append(col)
            continue

        sample = (
            nonnull if len(nonnull) <= 1000 else nonnull.sample(1000, random_state=SEED)
        )
        converted = to_numeric_series(sample)

        if converted.notna().mean() >= 0.8:
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    return numeric_cols, cat_cols


def fit_preprocessor(X_fit: pd.DataFrame, cols: list) -> tuple:
    """Fit simple imputation and ordinal categorical mappings on fit data only."""
    numeric_cols, cat_cols = infer_types(X_fit, cols)

    medians = {}
    for col in numeric_cols:
        vals = to_numeric_series(X_fit[col])
        med = vals.median()
        medians[col] = 0.0 if pd.isna(med) else float(med)

    mappings = {}
    for col in cat_cols:
        cats = X_fit[col].astype("string").dropna().unique()
        mappings[col] = {c: i for i, c in enumerate(cats)}

    return numeric_cols, cat_cols, medians, mappings


def transform(
    X: pd.DataFrame,
    cols: list,
    numeric_cols: list,
    cat_cols: list,
    medians: dict,
    mappings: dict,
) -> pd.DataFrame:
    """Apply fitted preprocessing to a dataframe."""
    out = pd.DataFrame(index=X.index)

    for col in cols:
        if col in numeric_cols:
            out[col] = to_numeric_series(X[col]).fillna(medians[col])
        else:
            out[col] = X[col].astype("string").map(mappings[col]).fillna(-1).astype(int)

    return out


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])
    if "id" in train.columns:
        train = train.drop(columns=["id"])
    if "id" in test.columns:
        test = test.drop(columns=["id"])

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET} not found in train data.")

    y = to_numeric_series(train[TARGET])
    valid = y.notna()
    train = train[valid].copy()
    y = y[valid].astype(float)

    if len(train) == 0:
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: np.zeros(len(test)),
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        print("CV RMSE: nan")
        return

    train = train.drop(columns=[TARGET])

    feature_cols = [c for c in train.columns if c in test.columns]
    if not feature_cols:
        feature_cols = list(train.columns)
        for c in feature_cols:
            if c not in test.columns:
                test[c] = np.nan

    train = train[feature_cols].copy()
    test = test[feature_cols].copy()

    if len(feature_cols) == 0:
        pred = float(y.mean())
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: np.full(len(test), pred),
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        print("CV RMSE: nan")
        return

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbosity": -1,
        "n_jobs": -1,
    }

    cv_scores = []
    best_iters = []

    n_splits = min(5, len(train))
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
            X_tr_raw = train.iloc[tr_idx]
            X_val_raw = train.iloc[val_idx]
            y_tr = y.iloc[tr_idx]
            y_val = y.iloc[val_idx]

            numeric_cols, cat_cols, medians, mappings = fit_preprocessor(
                X_tr_raw, feature_cols
            )

            X_tr = transform(
                X_tr_raw, feature_cols, numeric_cols, cat_cols, medians, mappings
            )
            X_val = transform(
                X_val_raw, feature_cols, numeric_cols, cat_cols, medians, mappings
            )

            categorical_feature = cat_cols if cat_cols else None

            dtrain = lgb.Dataset(
                X_tr, label=y_tr, categorical_feature=categorical_feature
            )
            dval = lgb.Dataset(
                X_val, label=y_val, categorical_feature=categorical_feature
            )

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

            best_iter = getattr(model, "best_iteration", None)
            pred = model.predict(
                X_val,
                num_iteration=(
                    best_iter if best_iter is not None and best_iter > 0 else None
                ),
            )

            if y.min() >= 0:
                pred = np.clip(pred, 0.0, None)

            cv_scores.append(root_mean_squared_error(y_val, pred))
            best_iters.append(
                max(1, int(best_iter)) if best_iter is not None and best_iter > 0 else 1
            )

    cv_rmse = float(np.mean(cv_scores)) if cv_scores else float("nan")

    numeric_cols, cat_cols, medians, mappings = fit_preprocessor(train, feature_cols)

    X_train = transform(train, feature_cols, numeric_cols, cat_cols, medians, mappings)
    X_test = transform(test, feature_cols, numeric_cols, cat_cols, medians, mappings)

    categorical_feature = cat_cols if cat_cols else None
    dtrain_full = lgb.Dataset(X_train, label=y, categorical_feature=categorical_feature)

    num_round = int(np.mean(best_iters)) if best_iters else 500
    final_model = lgb.train(params, dtrain_full, num_boost_round=max(1, num_round))

    test_pred = final_model.predict(X_test)
    if y.min() >= 0:
        test_pred = np.clip(test_pred, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)

    print(f"CV RMSE: {cv_rmse:.6f}")


if __name__ == "__main__":
    main()
