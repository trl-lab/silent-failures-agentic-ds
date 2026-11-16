import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/test.parquet"
TARGET = "WATERDEPTH"
MISSING_CAT = "__MISSING_OR_UNSEEN__"

DROP_COLS = {
    "OBJECTID",
    "PLATFRMID",
    "NAME",
    "COMPLEXID",
    "OWNER",
    "OPERNAME",
    "GlobalID",
    "SOURCEDATE",
    "INSTALL",
    "REMOVAL",
}


def parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in ("INSTALL", "REMOVAL"):
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_YEAR"] = dt.dt.year.astype("float64")
                df = df.drop(columns=[col])

    if "INSTALL_YEAR" in df.columns and "REMOVAL_YEAR" in df.columns:
        df["SERVICE_YEARS"] = df["REMOVAL_YEAR"] - df["INSTALL_YEAR"]

    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols)


def make_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def active_columns(df: pd.DataFrame, feature_cols: list) -> list:
    return [c for c in feature_cols if c in df.columns and df[c].notna().any()]


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list,
    cat_cols: list,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = active_columns(source_df, feature_cols)
    X = df.reindex(columns=cols).copy()

    for col in cols:
        if col in cat_cols:
            source_cats = list(source_df[col].cat.categories)
            if MISSING_CAT not in source_cats:
                source_cats.append(MISSING_CAT)

            valid = set(source_cats)
            vals = X[col].astype(object).where(X[col].notna(), MISSING_CAT)
            vals = vals.where(vals.isin(valid), MISSING_CAT)
            X[col] = pd.Categorical(vals, categories=source_cats)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    return X


def make_model(n_estimators: int = 1000) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET]).copy()
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET]).copy()

    train_df = parse_date_features(train_df)
    test_df = parse_date_features(test_df)

    train_df = drop_unwanted_columns(train_df)
    test_df = drop_unwanted_columns(test_df)

    feature_cols = [c for c in train_df.columns if c != TARGET]
    cat_cols = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])
    ]

    train_df = make_categorical(train_df, cat_cols)

    X_all = train_df[feature_cols].copy()
    y_all = train_df[TARGET].astype("float64")

    if len(X_all) > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=42,
        )

        val_source = make_categorical(X_train.copy(), cat_cols)
        X_train_val = prepare_features(X_train, feature_cols, cat_cols, val_source)
        X_val_val = prepare_features(X_val, feature_cols, cat_cols, val_source)

        val_model = make_model(n_estimators=1000)
        val_model.fit(
            X_train_val,
            y_train,
            categorical_feature=[c for c in cat_cols if c in X_train_val.columns],
        )

        val_pred = val_model.predict(X_val_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        print(f"Validation RMSE: {val_rmse:.6f}")

    final_source = make_categorical(train_df.copy(), cat_cols)
    X_train_final = prepare_features(train_df, feature_cols, cat_cols, final_source)
    X_test_final = prepare_features(test_df, feature_cols, cat_cols, final_source)

    final_model = make_model(n_estimators=1000)
    final_model.fit(
        X_train_final,
        y_all,
        categorical_feature=[c for c in cat_cols if c in X_train_final.columns],
    )

    preds = final_model.predict(X_test_final)
    preds = np.asarray(preds, dtype="float64")

    fallback = float(y_all.mean()) if len(y_all) else 0.0
    preds = np.where(np.isfinite(preds), preds, fallback)
    preds = np.clip(preds, 0.0, None)

    out = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype="int64"),
            "WATERDEPTH": preds,
        }
    )
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()