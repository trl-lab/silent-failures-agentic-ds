import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/grant/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/grant/data/test.parquet"
TARGET = "Federal Funds"
DROP_COLS = ["Log Number", "Description"]
TEXT_FEATURES = ["desc_len", "desc_word_count"]


def make_features(
    df,
    feature_cols,
    numeric_cols,
    categorical_cols,
    medians=None,
    fit=False,
):
    X = df.copy()

    if TARGET in X.columns:
        X = X.drop(columns=[TARGET])

    for col in DROP_COLS:
        if col in X.columns:
            X = X.drop(columns=[col])

    if "Description" in df.columns:
        desc = df["Description"].fillna("").astype(str)
        X["desc_len"] = desc.str.len()
        X["desc_word_count"] = desc.str.split().str.len()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_cols]

    if fit:
        med = X[numeric_cols].median()
        med = med.fillna(0.0)
    else:
        med = medians if medians is not None else pd.Series(dtype=float)

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(med.get(col, 0.0))

    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("missing").astype(object)

    return X, med


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if len(test) == 0:
        pd.DataFrame(
            {
                "id": np.arange(0, dtype=int),
                "Federal Funds": np.array([], dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train[np.isfinite(train[TARGET])]
    train = train.dropna(subset=[TARGET])

    base_feature_cols = [
        col
        for col in train.columns
        if col not in [TARGET, *DROP_COLS, *TEXT_FEATURES]
    ]
    feature_cols = list(dict.fromkeys(base_feature_cols + TEXT_FEATURES))

    numeric_cols = []
    categorical_cols = []

    for col in base_feature_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric_cols = list(dict.fromkeys(numeric_cols + TEXT_FEATURES))
    categorical_cols = list(dict.fromkeys(categorical_cols))

    X_full, medians_full = make_features(
        train,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        fit=True,
    )
    y_full = train[TARGET].astype(float)

    if len(y_full) == 0:
        pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                "Federal Funds": np.zeros(len(test), dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    if len(X_full) > 20:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )
        use_validation = True
    else:
        X_train, X_val, y_train, y_val = X_full, X_full, y_full, y_full
        use_validation = False

    cat_indices = [X_full.columns.get_loc(col) for col in categorical_cols]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )

    if use_validation:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_indices,
            early_stopping_rounds=100,
            verbose=0,
        )

        val_pred = model.predict(X_val)
        _ = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        best_iter = None
        try:
            best_iter = model.get_best_iteration()
        except Exception:
            best_iter = getattr(model, "best_iteration_", None)

        if best_iter is None or best_iter < 0:
            best_iter = 1000
        else:
            best_iter = int(best_iter)
    else:
        model.fit(X_train, y_train, cat_features=cat_indices)
        best_iter = 1000

    final_model = CatBoostRegressor(
        iterations=max(1, best_iter),
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    final_model.fit(X_full, y_full, cat_features=cat_indices)

    X_test, _ = make_features(
        test,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        medians=medians_full,
        fit=False,
    )

    predictions = final_model.predict(X_test)
    predictions = np.asarray(predictions, dtype=float).ravel()

    fallback = float(np.nanmean(y_full))
    high = float(np.nanmax(y_full))
    low = float(np.nanmin(y_full))

    predictions = np.nan_to_num(
        predictions,
        nan=fallback,
        posinf=high,
        neginf=low,
    )

    output = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "Federal Funds": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()