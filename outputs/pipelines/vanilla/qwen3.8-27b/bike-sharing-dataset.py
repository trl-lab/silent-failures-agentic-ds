import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/test.parquet"
TARGET = "cnt"
DATE_COL = "dteday"
EXCLUDE_COLS = {TARGET, DATE_COL, "instant", "id", "casual", "registered"}
RANDOM_STATE = 42


def add_date_features(df, use_date):
    df = df.copy()
    if use_date and DATE_COL in df.columns:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        df[DATE_COL] = dt
        df["date_year"] = dt.dt.year
        df["date_month"] = dt.dt.month
        df["date_day"] = dt.dt.day
        df["date_dayofweek"] = dt.dt.dayofweek
        weekend = (dt.dt.dayofweek >= 5).astype("float")
        weekend[dt.isna()] = np.nan
        df["date_is_weekend"] = weekend
    return df


def get_categorical_columns(df, exclude):
    cat_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        dtype = df[col].dtype
        if dtype == object or isinstance(dtype, pd.CategoricalDtype):
            cat_cols.append(col)
    return cat_cols


def encode_categoricals(train_df, val_df, test_df, cat_cols):
    frames = [df for df in (train_df, val_df, test_df) if df is not None]

    for col in cat_cols:
        if col not in train_df.columns:
            continue

        train_df[col] = train_df[col].astype("object").where(train_df[col].notna(), "missing")

        categories = sorted(train_df[col].unique().tolist(), key=lambda x: str(x))
        mapping = {cat: float(i) for i, cat in enumerate(categories)}

        train_df[col] = train_df[col].map(mapping).astype("float64")

        for df in frames:
            if df is train_df:
                continue

            if col not in df.columns:
                df[col] = "missing"
            else:
                df[col] = df[col].astype("object").where(df[col].notna(), "missing")

            df[col] = df[col].map(mapping).fillna(-1.0).astype("float64")

    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df, cat_cols):
    train_df, val_df, test_df = encode_categoricals(train_df, val_df, test_df, cat_cols)

    feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]

    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

        if val_df is not None:
            if col in val_df.columns:
                val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
            else:
                val_df[col] = np.nan

        if test_df is not None:
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
            else:
                test_df[col] = np.nan

    if test_df is not None:
        test_df = test_df.reindex(columns=feature_cols).copy()

    return train_df, val_df, test_df, feature_cols


def make_time_based_split(train_df, val_fraction=0.2):
    if len(train_df) < 2:
        return train_df.copy(), train_df.copy()

    if DATE_COL in train_df.columns:
        valid_count = train_df[DATE_COL].notna().sum()
        if valid_count >= 20:
            sorted_df = train_df.sort_values(DATE_COL, kind="mergesort")
            split_index = int(len(sorted_df) * (1.0 - val_fraction))
            split_index = max(1, min(split_index, len(sorted_df) - 1))

            train_part = sorted_df.iloc[:split_index].copy()
            val_part = sorted_df.iloc[split_index:].copy()

            if len(train_part) > 0 and len(val_part) > 0:
                return train_part, val_part

    return train_test_split(train_df, test_size=val_fraction, random_state=RANDOM_STATE)


def make_model(n_estimators=1000):
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def train_model(X_train, y_train, use_log_target, n_estimators=1000):
    if use_log_target:
        y_model = np.log1p(y_train.clip(lower=0).astype("float64"))
    else:
        y_model = y_train.astype("float64")

    model = make_model(n_estimators)
    model.fit(X_train, y_model)
    return model


def predict_model(model, X, use_log_target):
    if len(X) == 0:
        return np.zeros(0, dtype="float64")

    pred = model.predict(X)

    if use_log_target:
        pred = np.expm1(pred)

    pred = np.asarray(pred, dtype="float64")
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return np.clip(pred, 0.0, None)


def evaluate_model(model, X_val, y_val, use_log_target):
    if len(y_val) == 0:
        return np.inf

    pred = predict_model(model, X_val, use_log_target)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))

    if not np.isfinite(rmse):
        return np.inf

    return rmse


def save_predictions(pred):
    pred = np.asarray(pred, dtype="float64")
    pred = np.where(np.isfinite(pred), pred, 0.0)
    pred = np.clip(pred, 0.0, None)

    out = pd.DataFrame({
        "id": np.arange(len(pred), dtype=int),
        "cnt": pred
    })
    out.to_csv("predictions.csv", index=False)


def main():
    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    if TARGET not in train_raw.columns:
        save_predictions(np.zeros(len(test_raw), dtype="float64"))
        return

    train_raw = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)

    if len(test_raw) == 0:
        save_predictions(np.zeros(0, dtype="float64"))
        return

    if len(train_raw) == 0:
        save_predictions(np.zeros(len(test_raw), dtype="float64"))
        return

    use_date = DATE_COL in train_raw.columns and DATE_COL in test_raw.columns

    train_df = add_date_features(train_raw, use_date)
    test_df = add_date_features(test_raw, use_date)

    cat_cols = get_categorical_columns(train_df, EXCLUDE_COLS)

    train_part, val_part = make_time_based_split(train_df)

    train_part, val_part, test_part, feature_cols = prepare_features(
        train_part,
        val_part,
        test_df,
        cat_cols
    )

    if len(feature_cols) == 0:
        save_predictions(np.zeros(len(test_part), dtype="float64"))
        return

    X_train = train_part.reindex(columns=feature_cols).copy()
    y_train = train_part[TARGET].astype("float64")
    X_val = val_part.reindex(columns=feature_cols).copy()
    y_val = val_part[TARGET].astype("float64")
    X_test = test_part.reindex(columns=feature_cols).copy()

    if len(X_train) < 2 or not X_train.notna().any().any():
        mean_target = float(y_train.mean()) if len(y_train) > 0 else 0.0
        save_predictions(np.full(len(X_test), mean_target, dtype="float64"))
        return

    raw_model = train_model(X_train, y_train, False)
    log_model = train_model(X_train, y_train, True)

    raw_rmse = evaluate_model(raw_model, X_val, y_val, False)
    log_rmse = evaluate_model(log_model, X_val, y_val, True)

    use_log_target = log_rmse < raw_rmse

    full_train_df = add_date_features(train_raw, use_date)
    full_test_df = add_date_features(test_raw, use_date)

    full_cat_cols = get_categorical_columns(full_train_df, EXCLUDE_COLS)
    full_train_df, _, full_test_df, full_feature_cols = prepare_features(
        full_train_df,
        None,
        full_test_df,
        full_cat_cols
    )

    if len(full_feature_cols) == 0:
        save_predictions(np.zeros(len(full_test_df), dtype="float64"))
        return

    X_full_train = full_train_df.reindex(columns=full_feature_cols).copy()
    y_full_train = full_train_df[TARGET].astype("float64")

    if len(X_full_train) < 2 or not X_full_train.notna().any().any():
        mean_target = float(y_full_train.mean()) if len(y_full_train) > 0 else 0.0
        save_predictions(np.full(len(full_test_df), mean_target, dtype="float64"))
        return

    final_model = train_model(X_full_train, y_full_train, use_log_target)
    test_pred = predict_model(final_model, full_test_df, use_log_target)

    save_predictions(test_pred)


if __name__ == "__main__":
    main()