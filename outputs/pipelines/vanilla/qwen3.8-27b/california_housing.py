import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/test.parquet"
TARGET = "medianHouseValue"
RANDOM_STATE = 42


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_categorical_columns(df, target):
    categorical_cols = []
    for col in df.columns:
        if col == target:
            continue
        if (
            pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            categorical_cols.append(col)
    return categorical_cols


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=(denominator > 0),
    )


def add_features(df):
    out = df.copy()

    if {"totalRooms", "households"}.issubset(out.columns):
        out["rooms_per_household"] = safe_divide(out["totalRooms"], out["households"])
    if {"totalBedrooms", "totalRooms"}.issubset(out.columns):
        out["bedrooms_per_room"] = safe_divide(out["totalBedrooms"], out["totalRooms"])
    if {"population", "households"}.issubset(out.columns):
        out["population_per_household"] = safe_divide(out["population"], out["households"])

    for col in ["totalRooms", "population", "households"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(pd.to_numeric(out[col], errors="coerce").clip(lower=0))

    if "medianIncome" in out.columns:
        income = pd.to_numeric(out["medianIncome"], errors="coerce")
        out["income_squared"] = income**2
        out["income_cubed"] = income**3

    return out


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("fill", SimpleImputer(strategy="constant", fill_value=0.0)),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        transformers.append(("num", SimpleImputer(strategy="constant", fill_value=0.0), []))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(name):
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
        )
    if name == "catboost":
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_state=RANDOM_STATE,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def predict_original_scale(pipeline, X):
    pred_log = pipeline.predict(X)
    pred = np.expm1(pred_log)
    return np.clip(pred, 0.0, 500000.0)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    categorical_cols = detect_categorical_columns(train_df, TARGET)

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("object")
        if col in test_df.columns:
            test_df[col] = test_df[col].astype("object")

    numeric_cols = [
        col
        for col in train_df.columns
        if col != TARGET and col not in categorical_cols
    ]

    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    train_df = train_df.dropna(subset=[TARGET])

    feature_cols = numeric_cols + categorical_cols

    X_train_full = train_df.reindex(columns=feature_cols)
    y_train_full = pd.to_numeric(train_df[TARGET], errors="coerce").clip(lower=0.0)
    y_train_full_log = np.log1p(y_train_full)

    X_test = test_df.drop(columns=[TARGET], errors="ignore").reindex(columns=feature_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model_names = ["lgbm", "xgb", "catboost"]
    validation_scores = {}

    for name in model_names:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", make_model(name)),
            ]
        )

        pipeline.fit(X_train, y_train_log)
        val_pred = predict_original_scale(pipeline, X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        if np.isfinite(rmse):
            validation_scores[name] = rmse
        else:
            validation_scores[name] = np.inf

        print(f"{name} validation RMSE: {rmse:.4f}")

    weights = {
        name: (0.0 if not np.isfinite(score) else 1.0 / score)
        for name, score in validation_scores.items()
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {name: 1.0 / len(model_names) for name in model_names}
    else:
        weights = {name: weight / weight_sum for name, weight in weights.items()}

    final_predictions = np.zeros(len(X_test), dtype=float)

    for name in model_names:
        if weights[name] == 0.0:
            continue

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", make_model(name)),
            ]
        )

        pipeline.fit(X_train_full, y_train_full_log)
        test_pred = predict_original_scale(pipeline, X_test)
        final_predictions += weights[name] * test_pred

    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_predictions), dtype=int),
            "medianHouseValue": final_predictions,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()