import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/test.parquet"
TARGET = "fatalities"
RANDOM_STATE = 42


def get_model(name: str):
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "xgb":
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    if name == "cat":
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )

    raise ValueError(f"Unknown model: {name}")


def prepare_features(X_fit: pd.DataFrame, X_apply: pd.DataFrame, cat_cols: list, num_cols: list):
    Xf = X_fit.copy()
    Xa = X_apply.copy()

    for col in num_cols:
        if col not in Xf.columns:
            Xf[col] = np.nan
        if col not in Xa.columns:
            Xa[col] = np.nan

        Xf[col] = pd.to_numeric(Xf[col], errors="coerce").astype(np.float32)
        Xa[col] = pd.to_numeric(Xa[col], errors="coerce").astype(np.float32)

        Xf[col] = Xf[col].replace([np.inf, -np.inf], np.nan)
        Xa[col] = Xa[col].replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        if col not in Xf.columns:
            Xf[col] = "__missing__"
        if col not in Xa.columns:
            Xa[col] = "__missing__"

        Xf[col] = Xf[col].fillna("__missing__").astype(str)
        Xa[col] = Xa[col].fillna("__missing__").astype(str)

        counts = Xf[col].value_counts()

        Xf[f"freq_{col}"] = Xf[col].map(counts).fillna(0).astype(np.float32)
        Xa[f"freq_{col}"] = Xa[col].map(counts).fillna(0).astype(np.float32)

        if counts.size <= 50:
            categories = sorted(counts.index.tolist())
            mapping = {value: idx for idx, value in enumerate(categories)}

            Xf[f"ord_{col}"] = Xf[col].map(mapping).astype(np.float32)
            Xa[f"ord_{col}"] = Xa[col].map(mapping).fillna(-1).astype(np.float32)

        Xf = Xf.drop(columns=[col])
        Xa = Xa.drop(columns=[col])

    Xa = Xa.reindex(columns=Xf.columns)

    Xf = Xf.astype(np.float32)
    Xa = Xa.astype(np.float32)

    return Xf, Xa


def fit_predict(model, X_train, y_train, X_val, y_val, transform: str):
    if transform == "log":
        y_train_transformed = np.log1p(np.clip(y_train, 0.0, None))
        model.fit(X_train, y_train_transformed)
        pred_val = np.expm1(model.predict(X_val))
    else:
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)

    pred_val = np.asarray(pred_val, dtype=np.float64)
    pred_val = np.nan_to_num(pred_val, nan=0.0, posinf=0.0, neginf=0.0)
    pred_val = np.clip(pred_val, 0.0, None)

    return pred_val


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = train_df.dropna(subset=[TARGET])

    y = train_df[TARGET].astype(np.float64)
    X_raw = train_df.drop(columns=[TARGET])

    cat_cols = [
        col
        for col in X_raw.columns
        if not is_numeric_dtype(X_raw[col]) and not is_bool_dtype(X_raw[col])
    ]
    num_cols = [col for col in X_raw.columns if col not in cat_cols]

    if len(X_raw) >= 100:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
    else:
        X_train_raw, X_val_raw, y_train, y_val = X_raw, X_raw, y, y

    X_train, X_val = prepare_features(X_train_raw, X_val_raw, cat_cols, num_cols)

    model_names = ["lgbm", "xgb", "cat"]
    results = []

    for name in model_names:
        best_rmse = np.inf
        best_transform = "direct"

        for transform in ["direct", "log"]:
            model = get_model(name)
            pred_val = fit_predict(model, X_train, y_train, X_val, y_val, transform)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_transform = transform

        results.append((name, best_transform, best_rmse))

    rmses = np.array([item[2] for item in results], dtype=np.float64)
    finite = np.isfinite(rmses)

    if not finite.any():
        weights = np.ones(len(results), dtype=np.float64) / len(results)
    else:
        inverse_rmse = np.zeros_like(rmses, dtype=np.float64)
        inverse_rmse[finite] = 1.0 / np.maximum(rmses[finite], 1e-8)

        if inverse_rmse.sum() <= 0:
            weights = np.ones(len(results), dtype=np.float64) / len(results)
        else:
            weights = inverse_rmse / inverse_rmse.sum()

    X_full, X_test = prepare_features(X_raw, test_df, cat_cols, num_cols)

    test_predictions = []

    for (name, transform, _), weight in zip(results, weights):
        if weight <= 0:
            continue

        model = get_model(name)

        if transform == "log":
            y_full_transformed = np.log1p(np.clip(y, 0.0, None))
            model.fit(X_full, y_full_transformed)
            pred_test = np.expm1(model.predict(X_test))
        else:
            model.fit(X_full, y)
            pred_test = model.predict(X_test)

        pred_test = np.asarray(pred_test, dtype=np.float64)
        pred_test = np.nan_to_num(pred_test, nan=0.0, posinf=0.0, neginf=0.0)
        pred_test = np.clip(pred_test, 0.0, None)

        test_predictions.append(weight * pred_test)

    if test_predictions:
        final_predictions = sum(test_predictions)
    else:
        final_predictions = np.zeros(len(test_df), dtype=np.float64)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "fatalities": final_predictions.astype(np.float64),
        }
    )

    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()