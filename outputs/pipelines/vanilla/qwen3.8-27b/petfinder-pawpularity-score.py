import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/petfinder-pawpularity-score/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/petfinder-pawpularity-score/data/test.parquet"
TARGET = "Pawpularity"
ID_COL = "Id"


def prepare_features(X, reference_columns=None):
    X = X.copy()

    if reference_columns is not None:
        X = X[[c for c in reference_columns if c in X.columns]]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna(axis=1, how="all")

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("object")
            X[col] = X[col].where(X[col].notna(), "missing")

    return X


def build_preprocessor(X):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        return FunctionTransformer(lambda x: x)

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train_all = train_df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET, ID_COL], errors="ignore")

    feature_cols = [c for c in X_train_all.columns if c in X_test.columns]
    X_train_all = X_train_all[feature_cols]
    X_test = X_test[feature_cols]

    X_train_all = prepare_features(X_train_all)
    X_test = prepare_features(X_test, reference_columns=X_train_all.columns)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_all,
        y_train,
        test_size=0.2,
        random_state=42,
    )

    val_preprocessor = build_preprocessor(X_fit)
    X_fit_proc = val_preprocessor.fit_transform(X_fit)
    X_val_proc = val_preprocessor.transform(X_val)

    models = make_models()
    val_rmses = {}

    for name, model in models.items():
        model.fit(X_fit_proc, y_fit)
        val_pred = model.predict(X_val_proc)
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=0.0, neginf=0.0)
        val_rmses[name] = mean_squared_error(y_val, val_pred, squared=False)

    weights = {name: 1.0 / max(rmse, 1e-8) for name, rmse in val_rmses.items()}
    total_weight = sum(weights.values())
    weights = {name: weight / total_weight for name, weight in weights.items()}

    final_preprocessor = build_preprocessor(X_train_all)
    X_train_proc = final_preprocessor.fit_transform(X_train_all)
    X_test_proc = final_preprocessor.transform(X_test)

    test_pred = np.zeros(len(X_test), dtype=float)

    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        pred = model.predict(X_test_proc)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        test_pred += weights[name] * pred

    test_pred = np.clip(test_pred, 0.0, 100.0)

    pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "Pawpularity": test_pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()