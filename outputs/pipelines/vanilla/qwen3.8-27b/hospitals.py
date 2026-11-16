import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/test.parquet"
TARGET = "BEDS"

DROP_COLS = {
    "OBJECTID",
    "ID",
    "NAME",
    "ADDRESS",
    "TELEPHONE",
    "SOURCE",
    "WEBSITE",
    "ALT_NAME",
    "STATE_ID",
    "COUNTRY",
    "NAICS_DESC",
    "ZIP4",
    "ZIP",
    "SOURCEDATE",
    "VAL_DATE",
    "ST_FIPS",
}

NUMERIC_CANDIDATES = [
    "LATITUDE",
    "LONGITUDE",
    "x",
    "y",
    "COUNTYFIPS",
    "ST_FIPS_NUM",
    "SOURCEDATE_YEAR",
    "VAL_DATE_YEAR",
]

CATEGORICAL_CANDIDATES = [
    "STATE",
    "COUNTY",
    "TYPE",
    "STATUS",
    "OWNER",
    "TRAUMA",
    "HELIPAD",
    "VAL_METHOD",
    "NAICS_CODE",
]


def save_predictions(n, preds=None):
    if preds is None:
        preds = np.zeros(n, dtype="float64")

    preds = np.asarray(preds, dtype="float64")
    preds = np.where(np.isfinite(preds), preds, 0.0)
    preds = np.clip(preds, 0.0, None)

    pd.DataFrame(
        {
            "id": np.arange(n, dtype=int),
            "BEDS": preds,
        }
    ).to_csv("predictions.csv", index=False)


def clean_frame(df):
    df = df.copy()

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if "ST_FIPS" in df.columns:
        df["ST_FIPS_NUM"] = pd.to_numeric(df["ST_FIPS"], errors="coerce").astype("float32")
        df = df.drop(columns=["ST_FIPS"])

    for date_col, year_col in [
        ("SOURCEDATE", "SOURCEDATE_YEAR"),
        ("VAL_DATE", "VAL_DATE_YEAR"),
    ]:
        if date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            df[year_col] = parsed.dt.year.astype("float32")
            df = df.drop(columns=[date_col])

    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype("object")
            .where(df[col].notna(), "MISSING")
            .astype(str)
        )

    return df, numeric_cols, categorical_cols


def fit_categories(train_df, test_df, categorical_cols):
    category_maps = {}

    for col in categorical_cols:
        if col not in train_df.columns:
            continue

        train_values = train_df[col].astype(str)
        categories = sorted(train_values.unique().tolist())
        if "MISSING" not in categories:
            categories.append("MISSING")

        category_maps[col] = categories
        train_df[col] = pd.Categorical(train_values, categories=categories)

    for col in categorical_cols:
        if col not in category_maps:
            continue

        categories = category_maps[col]

        if col in test_df.columns:
            test_values = test_df[col].astype(str)
            test_values = test_values.where(test_values.isin(categories), "MISSING")
            test_df[col] = pd.Categorical(test_values, categories=categories)
        else:
            test_df[col] = pd.Categorical(["MISSING"] * len(test_df), categories=categories)

    return train_df, test_df, category_maps


def make_model(n_estimators):
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    if len(test_df) == 0:
        save_predictions(0)
        return

    if TARGET not in train_df.columns:
        save_predictions(len(test_df))
        return

    y = pd.to_numeric(train_df[TARGET], errors="coerce").astype("float64").to_numpy()
    train_df = train_df.drop(columns=[TARGET])

    train_df, numeric_cols, categorical_cols = clean_frame(train_df)
    test_df, _, _ = clean_frame(test_df)

    mask = np.isfinite(y)
    train_df = train_df.iloc[mask]
    y = y[mask]

    if len(train_df) == 0:
        save_predictions(len(test_df))
        return

    train_df, test_df, _ = fit_categories(train_df, test_df, categorical_cols)

    feature_cols = [c for c in numeric_cols + categorical_cols if c in train_df.columns]

    fallback = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0

    if not feature_cols:
        save_predictions(len(test_df), np.full(len(test_df), fallback))
        return

    X = train_df[feature_cols].copy()
    X_test = test_df.reindex(columns=feature_cols)

    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").astype("float32")
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    valid_cols = [c for c in feature_cols if X[c].notna().any()]

    if not valid_cols:
        save_predictions(len(test_df), np.full(len(test_df), fallback))
        return

    X = X[valid_cols]
    X_test = X_test[valid_cols]

    try:
        if len(X) >= 100:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            best_n = 700
            best_score = np.inf

            for n_est in [300, 700, 1200]:
                model = make_model(n_est)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                pred = np.where(np.isfinite(pred), pred, fallback)
                score = float(np.sqrt(np.mean((pred - y_val) ** 2)))

                if score < best_score:
                    best_score = score
                    best_n = n_est
        else:
            best_n = 500

        final_model = make_model(best_n)
        final_model.fit(X, y)
        preds = final_model.predict(X_test)

    except Exception:
        preds = np.full(len(test_df), fallback)

    save_predictions(len(test_df), preds)


if __name__ == "__main__":
    main()