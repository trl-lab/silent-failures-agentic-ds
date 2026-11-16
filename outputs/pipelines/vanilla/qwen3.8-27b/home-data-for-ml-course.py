import inspect
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/test.parquet"
TARGET = "SalePrice"
RANDOM_STATE = 42
N_ESTIMATORS = 3000
LEARNING_RATE = 0.02


def make_one_hot_encoder():
    kwargs = {"handle_unknown": "ignore"}
    try:
        params = inspect.signature(OneHotEncoder.__init__).parameters
        if "sparse_output" in params:
            kwargs["sparse_output"] = False
        else:
            kwargs["sparse"] = False
    except Exception:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def make_models():
    lgbm = LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    xgb = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    catboost = CatBoostRegressor(
        iterations=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        task_type="CPU",
    )

    return [lgbm, xgb, catboost]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y = train_df[TARGET].astype(float)
    X = train_df.drop(columns=[TARGET, "Id"], errors="ignore")
    X_test = test_df.drop(columns=["Id", TARGET], errors="ignore")

    feature_cols = [col for col in X.columns if col in X_test.columns]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    numeric_cols = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in categorical_cols:
        X[col] = X[col].astype("object")
        X_test[col] = X_test[col].astype("object")

    y_log = np.log1p(y)

    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    validation_models = make_models()
    for model in validation_models:
        model.fit(X_train_proc, y_train_log)

    val_pred_log = np.mean(
        [model.predict(X_val_proc) for model in validation_models], axis=0
    )
    val_pred = np.expm1(val_pred_log)
    val_actual = np.expm1(y_val_log)
    _ = np.sqrt(mean_squared_error(val_actual, val_pred))

    preprocessor.fit(X)
    X_full_proc = preprocessor.transform(X)
    X_test_proc = preprocessor.transform(X_test)

    final_models = make_models()
    for model in final_models:
        model.fit(X_full_proc, y_log)

    test_pred_log = np.mean(
        [model.predict(X_test_proc) for model in final_models], axis=0
    )
    test_pred = np.expm1(test_pred_log)
    test_pred = np.maximum(test_pred, 0.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "SalePrice": test_pred,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()