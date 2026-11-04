import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/sberbank-russian-housing-market/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/sberbank-russian-housing-market/data/test.parquet"
TARGET = "price_doc"


def preprocess(df):
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp_year"] = ts.dt.year
        df["timestamp_month"] = ts.dt.month
        df["timestamp_day"] = ts.dt.day
        df["timestamp_year_month"] = ts.dt.year * 100 + ts.dt.month
        df = df.drop(columns=["timestamp"])

    if "full_sq" in df.columns:
        df["log_full_sq"] = np.log1p(df["full_sq"].clip(lower=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        if "full_sq" in df.columns and "life_sq" in df.columns:
            df["life_sq_ratio"] = df["life_sq"] / df["full_sq"]
        if "full_sq" in df.columns and "kitch_sq" in df.columns:
            df["kitch_sq_ratio"] = df["kitch_sq"] / df["full_sq"]
        if "floor" in df.columns and "max_floor" in df.columns:
            df["floor_ratio"] = df["floor"] / df["max_floor"]
        if "timestamp_year" in df.columns and "build_year" in df.columns:
            df["build_age"] = df["timestamp_year"] - df["build_year"]
        if "full_sq" in df.columns and "num_room" in df.columns:
            df["rooms_per_sqm"] = df["num_room"] / df["full_sq"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def get_categorical_columns(df):
    cat_cols = []
    for col in df.columns:
        if col == TARGET:
            continue
        dtype = df[col].dtype
        if dtype == object or isinstance(dtype, pd.CategoricalDtype):
            cat_cols.append(col)
    return cat_cols


def encode_categoricals(train_df, other_df, cat_cols):
    train_enc = train_df.copy()
    other_enc = other_df.copy() if other_df is not None else None

    for col in cat_cols:
        if col not in train_enc.columns:
            continue

        train_enc[col] = train_enc[col].astype(object)
        train_enc[col] = train_enc[col].fillna("unknown").astype(str)

        if other_enc is not None:
            if col not in other_enc.columns:
                other_enc[col] = "unknown"
            else:
                other_enc[col] = other_enc[col].astype(object)
                other_enc[col] = other_enc[col].fillna("unknown").astype(str)

        categories = pd.unique(train_enc[col])
        if "unknown" not in categories:
            categories = np.append(categories, "unknown")
        categories = np.sort(categories)

        if other_enc is not None:
            other_enc[col] = other_enc[col].where(other_enc[col].isin(categories), "unknown")

        train_enc[col] = pd.Categorical(train_enc[col], categories=categories)
        if other_enc is not None:
            other_enc[col] = pd.Categorical(other_enc[col], categories=categories)

    return train_enc, other_enc


def make_validation_masks(train_df, test_size=0.2, random_state=42):
    n = len(train_df)

    if "timestamp_year_month" in train_df.columns:
        ts = train_df["timestamp_year_month"].dropna()
        if len(ts) > 1000:
            cutoff = ts.quantile(1.0 - test_size)
            val_mask = (train_df["timestamp_year_month"] >= cutoff).fillna(False)
            train_mask = ~val_mask

            if val_mask.sum() > 100 and train_mask.sum() > 100:
                return train_mask, val_mask

    idx = np.arange(n)
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=random_state)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return train_mask, val_mask


def make_model(n_estimators):
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_estimators=n_estimators,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    if TARGET in train_df.columns:
        train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
        train_df = train_df.dropna(subset=[TARGET])
        train_df[TARGET] = train_df[TARGET].clip(lower=0)

    train_cols = [c for c in train_df.columns if c != TARGET]
    test_df = test_df.reindex(columns=train_cols)

    cat_cols = get_categorical_columns(train_df)

    train_mask, val_mask = make_validation_masks(train_df)
    train_part = train_df[train_mask]
    val_part = train_df[val_mask]

    X_train_part, X_val = encode_categoricals(
        train_part.drop(columns=[TARGET]),
        val_part.drop(columns=[TARGET]),
        cat_cols,
    )
    y_train_part = np.log1p(train_part[TARGET])
    y_val = np.log1p(val_part[TARGET])

    model = make_model(3000)
    model.fit(
        X_train_part,
        y_train_part,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature="auto",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = 3000

    X_train_full, X_test = encode_categoricals(
        train_df.drop(columns=[TARGET]),
        test_df,
        cat_cols,
    )
    y_train_full = np.log1p(train_df[TARGET])

    final_model = make_model(int(best_iteration))
    final_model.fit(
        X_train_full,
        y_train_full,
        categorical_feature="auto",
    )

    pred_log = final_model.predict(X_test)
    pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=0.0, neginf=0.0)
    pred_price = np.expm1(pred_log)
    pred_price = np.clip(pred_price, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "price_doc": pred_price.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()