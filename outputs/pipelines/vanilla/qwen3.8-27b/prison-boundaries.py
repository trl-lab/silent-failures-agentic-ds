import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/prison-boundaries/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/prison-boundaries/data/test.parquet"

KEYWORDS = [
    "juvenile", "detention", "correctional", "prison", "jail", "secure",
    "non-secure", "nonsecure", "minimum", "medium", "maximum", "transit",
    "hold", "holding", "work", "community", "residential", "school",
    "hospital", "psychiatric", "medical", "adult", "youth", "female",
    "male", "county", "state", "federal", "private", "reentry", "halfway",
    "rehab", "treatment", "assessment", "intake", "processing", "pretrial",
    "pre-trial", "posttrial", "post-trial", "parole", "probation",
    "supervision", "shelter", "camp", "farm", "industrial", "vocational",
    "education", "behavioral", "mental", "substance", "detox", "women",
    "men", "mixed", "single", "multi", "unit", "wing", "block", "building",
    "annex", "branch", "campus", "center", "facility", "institution",
    "complex", "site", "location", "office", "department", "division",
    "bureau", "agency", "commission", "court", "sheriff", "police",
    "corrections", "department of corrections", "state corrections",
    "county corrections", "municipal", "city", "town", "village", "rural",
    "urban", "suburban", "highway", "road", "street", "avenue", "boulevard",
    "drive", "lane", "place", "circle", "parkway", "expressway", "freeway",
    "route"
]

CAT_COLS = ["STATE", "TYPE", "STATUS", "COUNTRY", "VAL_METHOD", "SECURELVL"]
NUM_COLS = [
    "ZIP", "ZIP4", "COUNTYFIPS", "Shape__Area", "Shape__Length",
    "SOURCEDATE_year", "VAL_DATE_year", "name_len", "name_words",
    "Shape__Area_log10", "Shape__Length_log10",
    "Shape__Area_Length", "Shape__Length_Area",
    "Shape__Area_Length_log10", "Shape__Length_Area_log10"
]
DROP_COLS = [
    "OBJECTID", "FACILITYID", "NAME", "ADDRESS", "TELEPHONE",
    "SOURCE", "SOURCEDATE", "VAL_DATE", "WEBSITE", "CAPACITY"
]


def add_date_year(df, col):
    out_col = f"{col}_year"
    if col not in df.columns:
        df[out_col] = np.nan
        return
    dt = pd.to_datetime(df[col], errors="coerce")
    df[out_col] = dt.dt.year


def add_name_features(df):
    if "NAME" not in df.columns:
        df["name_len"] = 0
        df["name_words"] = 0
        for kw in KEYWORDS:
            feat = "name_kw_" + re.sub(r"[^a-z0-9]+", "_", kw.lower()).strip("_")
            df[feat] = 0
        return

    name = df["NAME"].fillna("").astype(str).str.lower()
    df["name_len"] = name.str.len()
    df["name_words"] = name.str.split().str.len().fillna(0)

    for kw in KEYWORDS:
        feat = "name_kw_" + re.sub(r"[^a-z0-9]+", "_", kw.lower()).strip("_")
        df[feat] = name.str.contains(kw, regex=False).astype(int)


def add_shape_features(df):
    area = pd.to_numeric(df.get("Shape__Area", np.nan), errors="coerce")
    length = pd.to_numeric(df.get("Shape__Length", np.nan), errors="coerce")

    area_pos = area.clip(lower=0.0) + 1e-12
    length_pos = length.clip(lower=0.0) + 1e-12

    df["Shape__Area_log10"] = np.log10(area_pos)
    df["Shape__Length_log10"] = np.log10(length_pos)

    df["Shape__Area_Length"] = area_pos / length_pos
    df["Shape__Length_Area"] = length_pos / area_pos

    df["Shape__Area_Length_log10"] = np.log10(
        df["Shape__Area_Length"].clip(lower=1e-12)
    )
    df["Shape__Length_Area_log10"] = np.log10(
        df["Shape__Length_Area"].clip(lower=1e-12)
    )


def preprocess(train, test):
    for df in (train, test):
        add_date_year(df, "SOURCEDATE")
        add_date_year(df, "VAL_DATE")
        add_name_features(df)
        add_shape_features(df)

    cat_cols = [c for c in CAT_COLS if c in train.columns and c in test.columns]

    for col in cat_cols:
        train[col] = train[col].fillna("UNKNOWN").astype(str)
        test[col] = test[col].fillna("UNKNOWN").astype(str)

        cats = set(train[col].unique())
        cats.add("UNKNOWN")
        cats = sorted(cats)

        test[col] = test[col].where(test[col].isin(cats), "UNKNOWN")

        train[col] = train[col].astype("category").cat.set_categories(cats)
        test[col] = test[col].astype("category").cat.set_categories(cats)

    num_cols = [c for c in NUM_COLS if c in train.columns and c in test.columns]
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    feature_cols = num_cols + cat_cols
    feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    return X_train, X_test, cat_cols


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    y = train["CAPACITY"].astype(float)
    train = train.drop(columns=["CAPACITY"], errors="ignore")

    X_train, X_test, cat_cols = preprocess(train, test)

    mask = y.notna()
    X_train = X_train.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    if len(X_train) == 0:
        pred = np.zeros(len(X_test), dtype=float)
    else:
        try:
            if len(X_train) > 50:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y, test_size=0.2, random_state=42
                )
                model = LGBMRegressor(
                    objective="regression",
                    n_estimators=5000,
                    learning_rate=0.03,
                    num_leaves=63,
                    max_depth=-1,
                    min_child_samples=10,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    categorical_feature=cat_cols,
                    callbacks=[
                        lgb.early_stopping(100, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
                best_iter = getattr(model, "best_iteration_", None)
                if best_iter is None or best_iter <= 0:
                    best_iter = 1000
            else:
                best_iter = 1000

            final_model = LGBMRegressor(
                objective="regression",
                n_estimators=int(best_iter),
                learning_rate=0.03,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=10,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            final_model.fit(X_train, y, categorical_feature=cat_cols)
            pred = final_model.predict(X_test)
        except Exception:
            fallback = float(np.nanmean(y)) if len(y) else 0.0
            if not np.isfinite(fallback):
                fallback = 0.0
            pred = np.full(len(X_test), fallback, dtype=float)

    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)

    out = pd.DataFrame({
        "id": np.arange(len(pred), dtype=int),
        "CAPACITY": pred
    })
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()