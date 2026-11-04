import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/test.parquet"
TARGET = "Overall"
RANDOM_STATE = 42
IDENTIFIER_COLS = {"Media", "Domain"}


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def is_categorical(series):
    if series.dtype.name == "category":
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    if pd.api.types.is_string_dtype(series):
        return True
    return False


def add_derived_features(df):
    df = df.copy()

    for col in ("Media", "Domain"):
        if col in df.columns:
            s = df[col].astype("object").where(df[col].notna(), "").astype(str)
            df[f"{col}_len"] = s.str.len().astype(float)
            df[f"{col}_words"] = s.str.split().str.len().fillna(0).astype(float)

    if "Domain" in df.columns:
        s = df["Domain"].astype("object").where(df["Domain"].notna(), "").astype(str)
        tld = s.str.extract(r"\.([a-zA-Z]{2,})$")[0]
        df["Domain_tld"] = tld.fillna("missing").astype(str)

    return df


def preprocess(fit_df, y_fit, transform_df, features):
    fit_df = fit_df.copy()
    transform_df = transform_df.copy()
    y_fit = np.asarray(y_fit, dtype=float)

    global_mean = float(np.nanmean(y_fit)) if y_fit.size else 0.0
    if not np.isfinite(global_mean):
        global_mean = 0.0

    for col in features:
        if col not in fit_df.columns:
            fit_df[col] = np.nan
        if col not in transform_df.columns:
            transform_df[col] = np.nan

        if is_categorical(fit_df[col]) or is_categorical(transform_df[col]):
            fit_values = fit_df[col].astype("object").where(fit_df[col].notna(), "missing").astype(str)
            transform_values = transform_df[col].astype("object").where(transform_df[col].notna(), "missing").astype(str)

            if col in IDENTIFIER_COLS:
                freq = fit_values.value_counts(normalize=True)
                fit_df[col] = fit_values.map(freq).fillna(0.0).astype(float)
                transform_df[col] = transform_values.map(freq).fillna(0.0).astype(float)
            else:
                if y_fit.size == 0:
                    fit_df[col] = 0.0
                    transform_df[col] = 0.0
                    continue

                tmp = pd.DataFrame({"cat": fit_values.values, "y": y_fit})
                agg = tmp.groupby("cat")["y"].agg(["sum", "count"])
                smoothing = 20.0
                encoded = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)

                fit_df[col] = fit_values.map(encoded).fillna(global_mean).astype(float)
                transform_df[col] = transform_values.map(encoded).fillna(global_mean).astype(float)
        else:
            fit_series = clean_numeric(fit_df[col])
            median = fit_series.median()
            if pd.isna(median):
                median = 0.0

            fit_df[col] = fit_series.fillna(median).astype(float)

            transform_series = clean_numeric(transform_df[col])
            transform_df[col] = transform_series.fillna(median).astype(float)

    return fit_df[features], transform_df[features]


def make_lgbm(n_estimators=1500):
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    train_df = add_derived_features(train_df)
    test_df = add_derived_features(test_df)

    if TARGET not in train_df.columns:
        predictions = np.zeros(len(test_df), dtype=float)
    else:
        train_df[TARGET] = clean_numeric(train_df[TARGET])
        train_df = train_df.dropna(subset=[TARGET])

        if len(train_df) == 0:
            predictions = np.zeros(len(test_df), dtype=float)
        else:
            features = [col for col in train_df.columns if col != TARGET]

            if len(features) == 0:
                predictions = np.full(len(test_df), float(train_df[TARGET].mean()), dtype=float)
            else:
                use_validation = len(train_df) >= 200

                if use_validation:
                    train_fit_df, train_val_df = train_test_split(
                        train_df,
                        test_size=0.2,
                        random_state=RANDOM_STATE,
                    )
                else:
                    train_fit_df = train_df
                    train_val_df = None

                X_fit, X_val = preprocess(
                    train_fit_df,
                    train_fit_df[TARGET],
                    train_val_df if train_val_df is not None else train_fit_df,
                    features,
                )
                y_fit = train_fit_df[TARGET]

                try:
                    model = make_lgbm(n_estimators=3000)

                    if train_val_df is not None and len(train_val_df) > 0:
                        model.fit(
                            X_fit,
                            y_fit,
                            eval_set=[(X_val, train_val_df[TARGET])],
                            eval_metric="rmse",
                            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
                        )
                        best_iteration = getattr(model, "best_iteration_", None)
                    else:
                        model.fit(X_fit, y_fit)
                        best_iteration = None

                    X_train_full, X_test = preprocess(train_df, train_df[TARGET], test_df, features)
                    y_train_full = train_df[TARGET]

                    if best_iteration is not None and best_iteration > 0:
                        final_model = make_lgbm(n_estimators=int(best_iteration))
                    else:
                        final_model = make_lgbm(n_estimators=1500)

                    final_model.fit(X_train_full, y_train_full)
                    predictions = final_model.predict(X_test)
                    predictions = np.asarray(predictions, dtype=float)

                    fallback = float(np.nanmean(y_train_full))
                    if not np.isfinite(fallback):
                        fallback = 0.0

                    predictions = np.where(np.isfinite(predictions), predictions, fallback)

                except Exception:
                    predictions = np.full(len(test_df), float(train_df[TARGET].mean()), dtype=float)

    output = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "Overall": predictions,
    })
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()