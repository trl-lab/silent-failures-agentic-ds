import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/test.parquet"
TARGET = "heating_load"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    return train_df, test_df


def prepare_data(train_df, test_df):
    id_like = {"id", "index", "row_id", "rowid"}

    train_df = train_df.drop(
        columns=[c for c in train_df.columns if str(c).strip().lower() in id_like],
        errors="ignore",
    )
    test_df = test_df.drop(
        columns=[c for c in test_df.columns if str(c).strip().lower() in id_like],
        errors="ignore",
    )

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET])

    feature_cols = [c for c in train_df.columns if c != TARGET]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan
    test_df = test_df[feature_cols].reset_index(drop=True)

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_numeric_dtype(test_df[col]):
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

        if train_df[col].isna().all():
            train_df[col] = 0
            test_df[col] = 0

    return train_df, test_df, feature_cols


def get_categorical_columns(train_df, feature_cols):
    categorical_cols = []

    for col in feature_cols:
        series = train_df[col]

        if series.dtype == object or series.dtype.name == "category" or series.dtype == bool:
            categorical_cols.append(col)
        elif (
            pd.api.types.is_integer_dtype(series)
            and series.nunique(dropna=True) <= 20
        ):
            categorical_cols.append(col)

    return categorical_cols


def build_preprocessor(feature_cols, categorical_cols):
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
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
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_models():
    return {
        "lightgbm": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train_df, test_df = load_data()
    train_df, test_df, feature_cols = prepare_data(train_df, test_df)

    categorical_cols = get_categorical_columns(train_df, feature_cols)
    preprocessor = build_preprocessor(feature_cols, categorical_cols)

    X = train_df[feature_cols]
    y = train_df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    models = make_models()
    val_predictions = {}
    val_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_train_processed, y_train)
            pred = model.predict(X_val_processed)
            pred = np.asarray(pred, dtype=float)
            pred = np.where(np.isfinite(pred), pred, np.nan)
            if np.isnan(pred).any():
                pred = pd.Series(pred).fillna(y_train.mean()).to_numpy()

            rmse = root_mean_squared_error(y_val, pred)
            val_predictions[name] = pred
            val_scores[name] = float(rmse)
        except Exception:
            continue

    if not val_scores:
        preprocessor.fit(X)
        X_train_processed = preprocessor.transform(X)
        X_test_processed = preprocessor.transform(test_df)

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train_processed, y)
        predictions = model.predict(X_test_processed)
    else:
        inverse_scores = {name: 1.0 / max(score, 1e-8) for name, score in val_scores.items()}
        total = sum(inverse_scores.values())
        weights = {name: value / total for name, value in inverse_scores.items()}

        preprocessor.fit(X)
        X_train_processed = preprocessor.transform(X)
        X_test_processed = preprocessor.transform(test_df)

        final_predictions = np.zeros(len(test_df), dtype=float)

        for name, weight in weights.items():
            model = make_models()[name]
            model.fit(X_train_processed, y)
            pred = model.predict(X_test_processed)
            pred = np.asarray(pred, dtype=float)
            pred = np.where(np.isfinite(pred), pred, np.nan)
            if np.isnan(pred).any():
                pred = pd.Series(pred).fillna(y.mean()).to_numpy()
            final_predictions += weight * pred

        predictions = final_predictions

    predictions = np.asarray(predictions, dtype=float)
    predictions = np.where(np.isfinite(predictions), predictions, y.mean())

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "heating_load": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()