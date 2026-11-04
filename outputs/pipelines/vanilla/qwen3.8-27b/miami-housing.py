import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/test.parquet"
TARGET = "SALE_PRC"
ID_COL = "PARCELNO"


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_numeric(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def add_features(df):
    df = df.copy()

    positive_cols = [
        "LND_SQFOOT",
        "TOT_LVG_AREA",
        "SPEC_FEAT_VAL",
        "RAIL_DIST",
        "OCEAN_DIST",
        "WATER_DIST",
        "CNTR_DIST",
        "SUBCNTR_DI",
        "HWY_DIST",
    ]

    for col in positive_cols:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0.0))

    if "TOT_LVG_AREA" in df.columns and "LND_SQFOOT" in df.columns:
        df["LVG_PER_LND"] = df["TOT_LVG_AREA"] / df["LND_SQFOOT"].replace(0, np.nan)

    if "month_sold" in df.columns:
        month = df["month_sold"].astype(float)
        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    if "structure_quality" in df.columns:
        df["structure_quality_sq"] = df["structure_quality"] ** 2

    if "age" in df.columns:
        df["age_log"] = np.log1p(df["age"].clip(lower=0.0))

    return df


def prepare_dataframe(df, is_train):
    df = df.copy()

    if is_train:
        if TARGET in df.columns:
            df = df.dropna(subset=[TARGET])
            df[TARGET] = df[TARGET].clip(lower=0.0)
    else:
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    df = clean_numeric(df)
    df = add_features(df)

    return df


def make_preprocessor(X_fit):
    numeric_cols = [
        col for col in X_fit.columns
        if pd.api.types.is_numeric_dtype(X_fit[col])
    ]
    categorical_cols = [
        col for col in X_fit.columns
        if col not in numeric_cols
    ]

    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_onehot()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        transformers.append(("passthrough", "passthrough", X_fit.columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=-1,
        ),
    }


def transform_target(y, transform):
    y = np.asarray(y, dtype=float)
    if transform == "log":
        return np.log1p(np.clip(y, 0.0, None))
    return y


def inverse_transform_target(y, transform):
    y = np.asarray(y, dtype=float)
    if transform == "log":
        y = np.expm1(y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, 0.0, None)


def compute_weights(scores):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.array([1.0])

    weights = np.zeros_like(scores)
    zero_mask = np.isclose(scores, 0.0)
    if zero_mask.any():
        weights[zero_mask] = 1.0
    else:
        finite_mask = np.isfinite(scores) & (scores > 0.0)
        if finite_mask.any():
            weights[finite_mask] = 1.0 / scores[finite_mask]

    total = weights.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(len(scores)) / len(scores)

    return weights / total


def train_predict(models, X_fit, y_fit, X_pred, transform):
    y_fit_t = transform_target(y_fit, transform)
    preds = []

    for model in models.values():
        pipe = Pipeline(
            [
                ("preprocessor", make_preprocessor(X_fit)),
                ("model", model),
            ]
        )
        pipe.fit(X_fit, y_fit_t)
        pred = pipe.predict(X_pred)
        pred = inverse_transform_target(pred, transform)
        preds.append(pred)

    return preds


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = prepare_dataframe(train_df, is_train=True)
    test_df = prepare_dataframe(test_df, is_train=False)

    y_train = train_df[TARGET]
    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.reindex(columns=X_train.columns)

    all_nan_cols = X_train.columns[X_train.isna().all()]
    if len(all_nan_cols):
        X_train = X_train.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols, errors="ignore")

    constant_cols = X_train.columns[X_train.nunique(dropna=True) <= 1]
    if len(constant_cols):
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols, errors="ignore")

    if X_train.shape[1] == 0:
        predictions = pd.DataFrame(
            {
                "id": np.arange(len(X_test), dtype=int),
                "SALE_PRC": np.zeros(len(X_test), dtype=float),
            }
        )
        predictions.to_csv("predictions.csv", index=False)
        return

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )

    models = make_models()
    transforms = ["identity", "log"]
    best = None

    for transform in transforms:
        val_preds = train_predict(models, X_tr, y_tr, X_val, transform)
        val_scores = [rmse(y_val, pred) for pred in val_preds]
        weights = compute_weights(val_scores)
        ensemble_val = np.average(val_preds, axis=0, weights=weights)
        score = rmse(y_val, ensemble_val)

        if best is None or score < best["score"]:
            best = {
                "transform": transform,
                "score": score,
                "weights": weights,
            }

    final_preds = train_predict(
        models,
        X_train,
        y_train,
        X_test,
        best["transform"],
    )
    final_pred = np.average(final_preds, axis=0, weights=best["weights"])
    final_pred = np.nan_to_num(final_pred, nan=0.0, posinf=0.0, neginf=0.0)
    final_pred = np.clip(final_pred, 0.0, None)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "SALE_PRC": final_pred,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()