import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/test.parquet"
TARGET = "ViolentCrimesPerPop"
DROP_COLS = {"state", "county", "community", "communityname", "fold"}
RANDOM_STATE = 42


def clean_dataframe(df: pd.DataFrame, target: str, drop_cols: set) -> pd.DataFrame:
    df = df.copy()

    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df[target] = df[target].replace([np.inf, -np.inf], np.nan)

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in df.columns:
        if col == target:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
            else:
                df[col] = df[col].astype("object")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def make_model(name: str):
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    if name == "catboost":
        return CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            bootstrap_type="MVS",
            subsample=0.8,
            colsample_bylevel=0.8,
        )
    raise ValueError(f"Unknown model: {name}")


def preprocess(X_train: pd.DataFrame, X_other: pd.DataFrame, categorical_cols: list):
    X_train = X_train.copy()
    X_other = X_other.copy()

    if categorical_cols:
        X_train[categorical_cols] = X_train[categorical_cols].fillna("missing").astype(str)
        X_other[categorical_cols] = X_other[categorical_cols].fillna("missing").astype(str)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_other[categorical_cols] = encoder.transform(X_other[categorical_cols])

    X_train = X_train.astype(float)
    X_other = X_other.astype(float)

    return X_train, X_other


def safe_predict(model, X: pd.DataFrame, fallback: float) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=float)
    pred = np.where(np.isfinite(pred), pred, fallback)
    return pred


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = clean_dataframe(train_df, TARGET, DROP_COLS)
    test_df = clean_dataframe(test_df, TARGET, DROP_COLS)

    train_df = train_df.dropna(subset=[TARGET])

    if train_df.empty:
        predictions = np.zeros(len(test_df), dtype=float)
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                TARGET: predictions,
            }
        ).to_csv("predictions.csv", index=False)
        return

    feature_cols = [c for c in train_df.columns if c != TARGET]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    categorical_cols = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    drop_uninformative = []
    for col in feature_cols:
        if col in categorical_cols:
            if train_df[col].nunique(dropna=True) <= 1:
                drop_uninformative.append(col)
        else:
            if train_df[col].notna().sum() == 0 or train_df[col].nunique(dropna=True) <= 1:
                drop_uninformative.append(col)

    feature_cols = [c for c in feature_cols if c not in drop_uninformative]
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in numeric_cols if c in feature_cols]

    if not feature_cols:
        predictions = np.full(len(test_df), float(train_df[TARGET].mean()), dtype=float)
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                TARGET: predictions,
            }
        ).to_csv("predictions.csv", index=False)
        return

    X = train_df[feature_cols].copy()
    y = train_df[TARGET].copy()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_fit_proc, X_val_proc = preprocess(X_fit, X_val, categorical_cols)

    model_names = ["lgbm", "xgb", "catboost"]
    val_predictions = {}
    val_rmses = {}

    for name in model_names:
        try:
            model = make_model(name)
            model.fit(X_fit_proc, y_fit)
            val_pred = safe_predict(model, X_val_proc, float(y_val.mean()))
            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            if np.isfinite(rmse):
                val_predictions[name] = val_pred
                val_rmses[name] = rmse
        except Exception:
            continue

    if not val_rmses:
        X_full_proc, X_test_proc = preprocess(train_df[feature_cols], test_df[feature_cols], categorical_cols)
        predictions = np.full(len(test_df), float(y.mean()), dtype=float)
    else:
        weights = {name: 1.0 / max(rmse, 1e-12) for name, rmse in val_rmses.items()}
        total_weight = sum(weights.values())
        weights = {name: w / total_weight for name, w in weights.items()}

        X_full_proc, X_test_proc = preprocess(train_df[feature_cols], test_df[feature_cols], categorical_cols)

        predictions = np.zeros(len(test_df), dtype=float)
        for name, weight in weights.items():
            try:
                model = make_model(name)
                model.fit(X_full_proc, y)
                pred = safe_predict(model, X_test_proc, float(y.mean()))
                predictions += weight * pred
            except Exception:
                continue

    predictions = np.where(np.isfinite(predictions), predictions, float(y.mean()))

    pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: predictions,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()