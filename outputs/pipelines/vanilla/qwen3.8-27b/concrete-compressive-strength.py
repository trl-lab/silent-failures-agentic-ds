import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/test.parquet"
TARGET = "strength"
RANDOM_STATE = 42


def _is_categorical(series: pd.Series) -> bool:
    if series.dtype == bool:
        return False

    if series.dtype == object or str(series.dtype) == "category":
        non_missing = series.notna().sum()
        if non_missing == 0:
            return True

        converted = pd.to_numeric(series, errors="coerce")
        numeric_ratio = converted.notna().sum() / non_missing
        return numeric_ratio < 0.9

    return False


def _clean_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def _make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers, remainder="drop")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_models() -> dict[str, object]:
    return {
        "lgbm": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    feature_cols = [col for col in train_df.columns if col != TARGET]

    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df.reindex(columns=feature_cols).copy()

    cat_cols = [col for col in feature_cols if _is_categorical(X_train_raw[col])]
    num_cols = [col for col in feature_cols if col not in cat_cols]

    X_train_raw = _clean_numeric_frame(X_train_raw, num_cols)
    X_test_raw = _clean_numeric_frame(X_test_raw, num_cols)

    y_train = train_df[TARGET].astype(float).to_numpy()

    if len(X_train_raw) >= 20:
        X_train_split_raw, X_val_split_raw, y_train_split, y_val_split = train_test_split(
            X_train_raw,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_train_split_raw, X_val_split_raw = X_train_raw, X_train_raw
        y_train_split, y_val_split = y_train, y_train

    val_preprocessor = _make_preprocessor(num_cols, cat_cols)
    X_train_split = val_preprocessor.fit_transform(X_train_split_raw)
    X_val_split = val_preprocessor.transform(X_val_split_raw)

    val_models = _make_models()
    val_rmse = {}

    for name, model in val_models.items():
        model.fit(X_train_split, y_train_split)
        val_pred = model.predict(X_val_split)
        val_rmse[name] = _rmse(y_val_split, val_pred)

    names = list(val_models.keys())
    rmse_values = np.array([val_rmse[name] for name in names], dtype=float)

    valid = np.isfinite(rmse_values) & (rmse_values > 0)
    if not valid.any():
        weights = np.ones(len(names), dtype=float) / len(names)
    else:
        inverse_rmse = np.zeros_like(rmse_values, dtype=float)
        inverse_rmse[valid] = 1.0 / rmse_values[valid]
        if inverse_rmse.sum() == 0:
            weights = np.ones(len(names), dtype=float) / len(names)
        else:
            weights = inverse_rmse / inverse_rmse.sum()

    final_preprocessor = _make_preprocessor(num_cols, cat_cols)
    X_train_final = final_preprocessor.fit_transform(X_train_raw)
    X_test_final = final_preprocessor.transform(X_test_raw)

    final_models = _make_models()
    test_predictions = np.zeros(len(X_test_final), dtype=float)

    for name, model in final_models.items():
        model.fit(X_train_final, y_train)
        test_predictions += weights[names.index(name)] * model.predict(X_test_final)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_predictions), dtype=int),
            "strength": test_predictions,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()