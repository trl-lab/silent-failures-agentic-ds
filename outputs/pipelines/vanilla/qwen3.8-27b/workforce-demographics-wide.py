import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/test.parquet"
TARGET = "Female (%)"
RANDOM_STATE = 42
SMOOTHING = 20.0


def fit_target_stats(X, y, cat_cols, smoothing=SMOOTHING):
    global_mean = float(np.mean(y)) if len(y) else 0.0
    stats = {}
    for col in cat_cols:
        values = X[col].astype("string").fillna("__missing__")
        tmp = pd.DataFrame({"cat": values, "y": y.to_numpy(dtype=float)})
        agg = tmp.groupby("cat")["y"].agg(["sum", "count"])
        encoded = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
        stats[col] = encoded.to_dict()
    return stats, global_mean


def transform_target(X, cat_cols, stats, global_mean):
    out = X.copy()
    for col in cat_cols:
        values = X[col].astype("string").fillna("__missing__")
        out[col] = values.map(stats[col]).fillna(global_mean).astype(float)
    return out


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
        ),
        "catboost": CatBoostRegressor(
            iterations=1200,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
    }


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    y = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.loc[y.notna()].copy()
    y = y.loc[train.index]

    X = train.drop(columns=[TARGET]).copy()
    test = test.reindex(columns=X.columns)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    keep_cols = []
    for col in X.columns:
        if X[col].isna().all():
            continue
        if X[col].nunique(dropna=True) <= 1:
            continue
        keep_cols.append(col)

    X = X[keep_cols]
    test = test.reindex(columns=keep_cols)

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    for col in cat_cols:
        X[col] = X[col].astype("string")
        test[col] = test[col].astype("string")

    if len(X) == 0:
        preds = np.full(len(test), float(np.mean(y)) if len(y) else 0.0)
        pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: preds}).to_csv(
            "predictions.csv", index=False
        )
        return

    if len(X) >= 20:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
    else:
        X_fit, X_val, y_fit, y_val = X, X, y, y

    val_stats, val_global = fit_target_stats(X_fit, y_fit, cat_cols)
    X_fit_enc = transform_target(X_fit, cat_cols, val_stats, val_global)
    X_val_enc = transform_target(X_val, cat_cols, val_stats, val_global)

    models = make_models()
    val_preds = {}
    val_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_fit_enc, y_fit)
            pred = model.predict(X_val_enc)
            pred = np.asarray(pred, dtype=float)
            pred = np.nan_to_num(pred, nan=float(np.mean(y_fit)), posinf=0.0, neginf=0.0)
            val_preds[name] = pred
            val_scores[name] = float(np.sqrt(mean_squared_error(y_val, pred)))
        except Exception:
            continue

    if not val_scores:
        final_stats, final_global = fit_target_stats(X, y, cat_cols)
        X_test_enc = transform_target(test, cat_cols, final_stats, final_global)
        X_train_enc = transform_target(X, cat_cols, final_stats, final_global)

        try:
            model = LGBMRegressor(
                n_estimators=800,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train_enc, y)
            preds = model.predict(X_test_enc)
        except Exception:
            preds = np.full(len(test), float(np.mean(y)) if len(y) else 0.0)

        preds = np.asarray(preds, dtype=float)
        preds = np.nan_to_num(preds, nan=float(np.mean(y)) if len(y) else 0.0, posinf=0.0, neginf=0.0)
        preds = np.clip(preds, 0.0, 100.0)
        pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: preds}).to_csv(
            "predictions.csv", index=False
        )
        return

    weights = {name: 1.0 / max(score, 1e-8) for name, score in val_scores.items()}
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}

    final_stats, final_global = fit_target_stats(X, y, cat_cols)
    X_train_enc = transform_target(X, cat_cols, final_stats, final_global)
    X_test_enc = transform_target(test, cat_cols, final_stats, final_global)

    final_preds = np.zeros(len(test), dtype=float)
    for name, model in models.items():
        if name not in weights:
            continue
        try:
            model.fit(X_train_enc, y)
            pred = model.predict(X_test_enc)
            pred = np.asarray(pred, dtype=float)
            pred = np.nan_to_num(pred, nan=float(np.mean(y)), posinf=0.0, neginf=0.0)
            final_preds += weights[name] * pred
        except Exception:
            continue

    if not np.any(final_preds):
        final_preds = np.full(len(test), float(np.mean(y)) if len(y) else 0.0)

    final_preds = np.nan_to_num(final_preds, nan=float(np.mean(y)) if len(y) else 0.0, posinf=0.0, neginf=0.0)
    final_preds = np.clip(final_preds, 0.0, 100.0)

    pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: final_preds}).to_csv(
        "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()