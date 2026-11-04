import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/test.parquet"


def add_features(df):
    df = df.copy()

    def num(col, default=0.0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(default, index=df.index)

    df["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
    df["TotalBath"] = (
        num("FullBath")
        + 0.5 * num("HalfBath")
        + num("BsmtFullBath")
        + 0.5 * num("BsmtHalfBath")
    )
    df["TotalPorch"] = (
        num("WoodDeckSF")
        + num("OpenPorchSF")
        + num("EnclosedPorch")
        + num("3SsnPorch")
        + num("ScreenPorch")
    )
    df["Age"] = num("YrSold") - num("YearBuilt")
    df["RemodAge"] = num("YrSold") - num("YearRemodAdd")
    df["GarageAge"] = num("YrSold") - num("GarageYrBlt")
    df["OverallScore"] = num("OverallQual") * num("OverallCond")
    return df


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_frames(train_df, test_df):
    y = np.log1p(train_df["SalePrice"].astype(float).clip(lower=0.0))

    X_train = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    X_test = test_df.drop(columns=["Id", "SalePrice"], errors="ignore")

    X_train = add_features(X_train)
    X_test = add_features(X_test)

    feature_cols = X_train.columns.tolist()
    X_test = X_test.reindex(columns=feature_cols, fill_value=np.nan)

    for frame in (X_train, X_test):
        for col in frame.columns:
            dtype_str = str(frame[col].dtype)
            if frame[col].dtype == object or dtype_str in {"category", "string"} or dtype_str.startswith("string"):
                frame[col] = frame[col].astype(object)
                frame[col] = frame[col].where(frame[col].notna(), np.nan)

    categorical_cols = [c for c in feature_cols if X_train[c].dtype == object]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X_train, X_test, y, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if categorical_cols:
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", make_onehot()),
            ]
        )
        transformers.append(("cat", cat_transformer, categorical_cols))

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_features(preprocessor, X):
    if preprocessor is None:
        return np.zeros((len(X), 0), dtype=float)

    Xp = preprocessor.transform(X)
    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    Xp = np.asarray(Xp, dtype=float)
    return np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)


def make_models():
    lgbm = LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )

    return {"lgbm": lgbm, "xgb": xgb, "cat": cat}


def rmse_log(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, X_test, y, numeric_cols, categorical_cols = prepare_frames(train_df, test_df)
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    weights = {}
    if len(X_train) > 10:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y, test_size=0.2, random_state=42
        )

        preprocessor.fit(X_tr)
        X_tr_p = transform_features(preprocessor, X_tr)
        X_val_p = transform_features(preprocessor, X_val)

        for name, model in make_models().items():
            try:
                model.fit(X_tr_p, y_tr)
                pred_val = model.predict(X_val_p)
                score = rmse_log(y_val, pred_val)
                weights[name] = 1.0 / score if np.isfinite(score) and score > 0 else 0.0
            except Exception:
                weights[name] = 0.0

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {"lgbm": 1.0}

    preprocessor.fit(X_train)
    X_train_p = transform_features(preprocessor, X_train)
    X_test_p = transform_features(preprocessor, X_test)

    final_preds = []
    final_weights = []

    for name, model in make_models().items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        try:
            model.fit(X_train_p, y)
            pred_test = model.predict(X_test_p)
            pred_test = np.asarray(pred_test, dtype=float).ravel()
            final_preds.append(pred_test)
            final_weights.append(w)
        except Exception:
            pass

    if not final_preds:
        try:
            model = LGBMRegressor(
                objective="regression",
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train_p, y)
            pred_test = model.predict(X_test_p)
            pred_test = np.asarray(pred_test, dtype=float).ravel()
            final_preds.append(pred_test)
            final_weights.append(1.0)
        except Exception:
            pred_test = np.full(len(X_test), float(np.median(y)), dtype=float)
            final_preds.append(pred_test)
            final_weights.append(1.0)

    w = np.asarray(final_weights, dtype=float)
    w = w / w.sum()

    pred_log = np.zeros_like(final_preds[0], dtype=float)
    for wi, pred in zip(w, final_preds):
        pred = np.nan_to_num(pred, nan=float(np.median(y)), posinf=20.0, neginf=-20.0)
        pred_log += wi * pred

    pred_log = np.clip(pred_log, 0.0, 20.0)
    pred_price = np.expm1(pred_log)
    pred_price = np.clip(pred_price, 100.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "SalePrice": pred_price,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()