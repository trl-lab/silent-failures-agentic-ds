import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/test.parquet"
TARGET = "price"
RANDOM_STATE = 42
N_ESTIMATORS = 2000


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    return train_df, test_df


def prepare_frames(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET not in train_df.columns:
        raise ValueError("Target column not found in train data.")

    train_df = train_df.dropna(subset=[TARGET]).copy()
    train_df = train_df[train_df[TARGET] > 0].copy()

    for col in ["x", "y", "z"]:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in test_df.columns:
            test_df[col] = np.nan

    train_df["xy"] = train_df["x"] * train_df["y"]
    train_df["xyz"] = train_df["x"] * train_df["y"] * train_df["z"]

    test_df = test_df.drop(columns=[TARGET], errors="ignore").copy()
    test_df["xy"] = test_df["x"] * test_df["y"]
    test_df["xyz"] = test_df["x"] * test_df["y"] * test_df["z"]

    feature_cols = [col for col in train_df.columns if col != TARGET]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    test_df = test_df[feature_cols].copy()
    return train_df, test_df, feature_cols


def get_column_types(train_df, feature_cols):
    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols, categorical_cols):
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
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_model():
    return LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_and_score(X_train, y_train, X_val, y_val, log_target):
    model = make_model()

    if log_target:
        y_fit = np.log1p(y_train)
        model.fit(X_train, y_fit)
        y_val_pred = np.expm1(model.predict(X_val))
    else:
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

    return rmse(y_val, y_val_pred)


def main():
    train_df, test_df = load_data()
    train_df, test_df, feature_cols = prepare_frames(train_df, test_df)

    numeric_cols, categorical_cols = get_column_types(train_df, feature_cols)

    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

    X_all = train_df[feature_cols]
    y_all = train_df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    score_raw = train_and_score(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        log_target=False,
    )
    score_log = train_and_score(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        log_target=True,
    )

    use_log = score_log < score_raw

    preprocessor_full = make_preprocessor(numeric_cols, categorical_cols)
    X_full_proc = preprocessor_full.fit_transform(X_all)
    X_test_proc = preprocessor_full.transform(test_df)

    final_model = make_model()

    if use_log:
        final_model.fit(X_full_proc, np.log1p(y_all))
        test_pred = np.expm1(final_model.predict(X_test_proc))
    else:
        final_model.fit(X_full_proc, y_all)
        test_pred = final_model.predict(X_test_proc)

    test_pred = np.asarray(test_pred, dtype=float)
    test_pred = np.where(np.isfinite(test_pred), test_pred, np.nan)

    if np.isnan(test_pred).any():
        fallback = float(np.nanmedian(y_all))
        test_pred = np.where(np.isnan(test_pred), fallback, test_pred)

    test_pred = np.maximum(test_pred, 0.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "price": test_pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()