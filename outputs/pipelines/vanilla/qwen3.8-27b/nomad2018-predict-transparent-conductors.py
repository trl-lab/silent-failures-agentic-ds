import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/nomad2018-predict-transparent-conductors/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/nomad2018-predict-transparent-conductors/data/test.parquet"
TARGET = "formation_energy_ev_natom"
ID_COL = "id"


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, -0.999999, None)
    y_pred = np.clip(y_pred, -0.999999, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def make_log_transform(y):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return (lambda x: x), (lambda x: x), 0.0

    y_min = float(np.nanmin(y))
    if y_min >= 0:
        shift = 0.0

        def transform(x):
            return np.log1p(np.asarray(x, dtype=float))

        def inverse(x):
            return np.expm1(np.asarray(x, dtype=float))
    else:
        shift = -y_min + 1.0

        def transform(x):
            return np.log1p(np.asarray(x, dtype=float) + shift)

        def inverse(x):
            return np.expm1(np.asarray(x, dtype=float)) - shift

    return transform, inverse, y_min


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X_train):
    feature_cols = list(X_train.columns)
    cat_cols = []
    num_cols = []

    for col in feature_cols:
        dtype = X_train[col].dtype
        if dtype == object or str(dtype) == "category" or dtype.name == "string":
            cat_cols.append(col)
        else:
            num_cols.append(col)

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", make_onehot())
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "cat": CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            thread_count=-1,
        ),
    }


def save_predictions(predictions, n_rows):
    predictions = np.asarray(predictions, dtype=float)
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    out = pd.DataFrame({
        "id": np.arange(n_rows, dtype=int),
        TARGET: predictions
    })
    out.to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        save_predictions(np.zeros(len(test_df)), len(test_df))
        return

    y_raw = train_df[TARGET].astype(float)
    finite_mask = np.isfinite(y_raw)
    train_df = train_df.loc[finite_mask].copy()
    y_raw = y_raw.loc[finite_mask]

    if ID_COL in train_df.columns:
        train_df = train_df.drop(columns=[ID_COL])
    if ID_COL in test_df.columns:
        test_df = test_df.drop(columns=[ID_COL])
    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    feature_cols = [c for c in train_df.columns if c != TARGET]
    test_df = test_df.reindex(columns=feature_cols)

    if train_df.empty or len(feature_cols) == 0:
        save_predictions(np.zeros(len(test_df)), len(test_df))
        return

    X_all = train_df[feature_cols].copy()
    y_all = y_raw.values.astype(float)

    for col in feature_cols:
        if X_all[col].dtype != object and str(X_all[col].dtype) != "category":
            X_all[col] = X_all[col].replace([np.inf, -np.inf], np.nan)
        if test_df[col].dtype != object and str(test_df[col].dtype) != "category":
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

    transform, inverse, y_min = make_log_transform(y_all)
    y_all_t = transform(y_all)

    if np.nanvar(y_all_t) == 0:
        save_predictions(np.full(len(test_df), float(np.mean(y_all))), len(test_df))
        return

    if len(X_all) >= 20:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all_t, test_size=0.2, random_state=42
        )
        y_val_orig = inverse(y_val)
    else:
        X_train, X_val, y_train, y_val = X_all, X_all, y_all_t, y_all_t
        y_val_orig = y_all

    preprocessor = build_preprocessor(X_train)
    if preprocessor is None:
        save_predictions(np.full(len(test_df), float(np.mean(y_all))), len(test_df))
        return

    models = get_models()
    val_preds = []

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            val_pred_t = pipe.predict(X_val)
            val_pred = inverse(val_pred_t)
            val_preds.append(val_pred)
        except Exception:
            val_preds.append(np.full(len(X_val), float(np.mean(y_all))))

    errors = np.array([rmsle(y_val_orig, p) for p in val_preds], dtype=float)
    errors = np.where(np.isfinite(errors), errors, np.inf)

    if np.all(np.isinf(errors)):
        weights = np.ones(len(models)) / len(models)
    else:
        inv_errors = 1.0 / (errors + 1e-8)
        weights = inv_errors / inv_errors.sum()

    final_preds = np.zeros(len(test_df), dtype=float)
    for (name, model), weight in zip(models.items(), weights):
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        try:
            pipe.fit(X_all, y_all_t)
            test_pred_t = pipe.predict(test_df)
            test_pred = inverse(test_pred_t)
            final_preds += weight * test_pred
        except Exception:
            final_preds += weight * float(np.mean(y_all))

    if y_min >= 0:
        final_preds = np.clip(final_preds, 0.0, None)
    else:
        final_preds = np.clip(final_preds, y_min, None)

    save_predictions(final_preds, len(test_df))


if __name__ == "__main__":
    main()