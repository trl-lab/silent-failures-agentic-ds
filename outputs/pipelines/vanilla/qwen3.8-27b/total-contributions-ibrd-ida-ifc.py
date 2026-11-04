import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/test.parquet"
TARGET = "Total Contribution (USD)"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_numeric_columns(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def add_safe_features(df):
    df = df.copy()

    if "Contributions Outstanding (USD)" in df.columns:
        outstanding = pd.to_numeric(df["Contributions Outstanding (USD)"], errors="coerce")
        outstanding = outstanding.replace([np.inf, -np.inf], np.nan)
        df["Contributions Outstanding_log"] = np.log1p(outstanding.clip(lower=0))

    if "Fiscal Year of Agreement" in df.columns:
        df["Fiscal Year of Agreement"] = pd.to_numeric(df["Fiscal Year of Agreement"], errors="coerce")

    return df


def fit_categorical_categories(X, cat_cols):
    categories = {}
    for col in cat_cols:
        if col in X.columns:
            values = X[col].dropna().astype(str).unique()
            categories[col] = sorted(set(values) | {"MISSING", "UNKNOWN"})
        else:
            categories[col] = ["MISSING", "UNKNOWN"]
    return categories


def encode_categoricals(X, cat_cols, categories):
    X = X.copy()
    for col in cat_cols:
        if col not in X.columns:
            continue

        series = X[col]
        series = series.where(series.notna(), "MISSING").astype(str)
        cats = categories.get(col, ["MISSING", "UNKNOWN"])
        series = pd.Categorical(series, categories=cats)
        X[col] = pd.Series(series, index=X.index).fillna("UNKNOWN").astype("category")

    return X


def make_model(cat_cols):
    return LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        categorical_features=cat_cols if cat_cols else "auto",
    )


def train_and_validate(X, y, cat_cols, use_log_target):
    val_rmse = np.inf
    best_iteration = 1000

    if len(X) < 100:
        return val_rmse, best_iteration

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    categories = fit_categorical_categories(X_train, cat_cols)
    X_train = encode_categoricals(X_train, cat_cols, categories)
    X_val = encode_categoricals(X_val, cat_cols, categories)

    if use_log_target:
        y_train_fit = np.log1p(y_train.clip(lower=0))
        y_val_eval = np.log1p(y_val.clip(lower=0))
    else:
        y_train_fit = y_train
        y_val_eval = y_val

    model = make_model(cat_cols)
    model.fit(
        X_train,
        y_train_fit,
        eval_set=[(X_val, y_val_eval)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
    )

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = 1000

    val_pred = model.predict(X_val)
    if use_log_target:
        val_pred = np.expm1(val_pred)

    val_rmse = rmse(y_val, val_pred)
    return val_rmse, best_iteration


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df[TARGET] = train_df[TARGET].replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET])

    y = train_df[TARGET].astype(float)
    X = train_df.drop(columns=[TARGET])

    X = clean_numeric_columns(X, X.select_dtypes(include=["number"]).columns)
    X = add_safe_features(X)

    X_test = test_df.copy()
    if TARGET in X_test.columns:
        X_test = X_test.drop(columns=[TARGET])

    X_test = clean_numeric_columns(X_test, X_test.select_dtypes(include=["number"]).columns)
    X_test = add_safe_features(X_test)

    common_cols = [col for col in X.columns if col in X_test.columns]
    X = X[common_cols]
    X_test = X_test[common_cols]

    drop_cols = [col for col in X.columns if X[col].isna().all()]
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in numeric_cols]

    if len(X) == 0 or len(X.columns) == 0:
        predictions = np.zeros(len(X_test), dtype=float)
    elif len(X) < 50:
        predictions = np.full(len(X_test), float(y.mean()) if len(y) > 0 else 0.0, dtype=float)
    else:
        can_log = bool((y.dropna() >= 0).all())
        candidates = [False]
        if can_log:
            candidates.append(True)

        best_use_log = False
        best_val_rmse = np.inf
        best_iteration = 1000

        for use_log in candidates:
            val_rmse, iteration = train_and_validate(X, y, cat_cols, use_log)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_use_log = use_log
                best_iteration = iteration

        categories = fit_categorical_categories(X, cat_cols)
        X_full = encode_categoricals(X, cat_cols, categories)
        X_test_encoded = encode_categoricals(X_test, cat_cols, categories)

        final_model = make_model(cat_cols)
        final_model.set_params(n_estimators=max(1, int(best_iteration)))

        if best_use_log:
            y_fit = np.log1p(y.clip(lower=0))
        else:
            y_fit = y

        final_model.fit(X_full, y_fit)

        predictions = final_model.predict(X_test_encoded)
        if best_use_log:
            predictions = np.expm1(predictions)

    fill_value = float(y.mean()) if len(y) > 0 else 0.0
    predictions = np.asarray(predictions, dtype=float)
    predictions = np.nan_to_num(predictions, nan=fill_value, posinf=fill_value, neginf=fill_value)

    if can_log:
        predictions = np.maximum(predictions, 0.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            TARGET: predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()