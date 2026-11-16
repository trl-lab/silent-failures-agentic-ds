import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/global-power-plant/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/global-power-plant/data/test.parquet"
TARGET = "capacity_mw"

DROP_COLS = {
    "name",
    "gppd_idnr",
    "url",
    "wepp_id",
    "owner",
    "country_long",
}

MAX_CARDINALITY = 200
RANDOM_STATE = 42


def clean_categorical_series(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    s = s.where(s.notna(), "missing")
    s = s.astype(str)
    s = s.str.strip()
    s = s.replace({"": "missing", "nan": "missing", "None": "missing", "null": "missing"})
    return s


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, fit_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()
    fit_df = fit_df.copy()

    for df in (train_df, test_df, fit_df):
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

    cols = [c for c in train_df.columns if c not in DROP_COLS]
    for c in cols:
        if c not in test_df.columns:
            test_df[c] = np.nan
        if c not in fit_df.columns:
            fit_df[c] = np.nan

    train_df = train_df[cols]
    test_df = test_df[cols]
    fit_df = fit_df[cols]

    cols = [c for c in cols if not fit_df[c].isna().all()]
    train_df = train_df[cols]
    test_df = test_df[cols]
    fit_df = fit_df[cols]

    cat_cols = [
        c for c in cols
        if is_object_dtype(fit_df[c]) or is_categorical_dtype(fit_df[c])
    ]
    num_cols = [c for c in cols if c not in cat_cols]

    if {"commissioning_year", "year_of_capacity_data"}.issubset(train_df.columns):
        train_df["plant_age"] = train_df["year_of_capacity_data"] - train_df["commissioning_year"]
        test_df["plant_age"] = test_df["year_of_capacity_data"] - test_df["commissioning_year"]
        fit_df["plant_age"] = fit_df["year_of_capacity_data"] - fit_df["commissioning_year"]
        num_cols.append("plant_age")

    fuel_cols = [c for c in ["other_fuel1", "other_fuel2", "other_fuel3"] if c in train_df.columns]
    if fuel_cols:
        train_df["n_other_fuels"] = train_df[fuel_cols].notna().sum(axis=1)
        test_df["n_other_fuels"] = test_df[fuel_cols].notna().sum(axis=1)
        fit_df["n_other_fuels"] = fit_df[fuel_cols].notna().sum(axis=1)
        num_cols.append("n_other_fuels")

    for c in num_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
        fit_df[c] = pd.to_numeric(fit_df[c], errors="coerce")

    final_cat_cols = []
    for c in cat_cols:
        train_df[c] = clean_categorical_series(train_df[c])
        test_df[c] = clean_categorical_series(test_df[c])
        fit_df[c] = clean_categorical_series(fit_df[c])

        n_unique = fit_df[c].nunique(dropna=True)
        if n_unique <= MAX_CARDINALITY:
            categories = sorted(fit_df[c].unique())
            train_df[c] = pd.Categorical(train_df[c], categories=categories)
            test_df[c] = pd.Categorical(test_df[c], categories=categories)
            final_cat_cols.append(c)
        else:
            freq = fit_df[c].value_counts(normalize=True)
            train_df[f"freq_{c}"] = train_df[c].map(freq).fillna(0.0).astype(float)
            test_df[f"freq_{c}"] = test_df[c].map(freq).fillna(0.0).astype(float)
            num_cols.append(f"freq_{c}")
            train_df = train_df.drop(columns=[c])
            test_df = test_df.drop(columns=[c])

    final_cols = num_cols + final_cat_cols
    train_df = train_df[final_cols]
    test_df = test_df[final_cols]
    return train_df, test_df, final_cat_cols


def make_model():
    return lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def predict_with_transform(model, X, transform):
    pred = model.predict(X)
    if transform == "log":
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    X_all = train_df.drop(columns=[TARGET])
    y_all = train_df[TARGET].astype(float)

    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_tr, X_val, cat_cols = prepare_features(X_tr_raw, X_val_raw, X_tr_raw)

    candidates = []
    for transform in ["raw", "log"]:
        y_tr_t = np.log1p(y_tr) if transform == "log" else y_tr
        model = make_model()
        model.fit(X_tr, y_tr_t, categorical_feature=cat_cols if cat_cols else "auto")
        pred_val = predict_with_transform(model, X_val, transform)
        rmse = root_mean_squared_error(y_val, pred_val)
        candidates.append((rmse, transform))
        print(f"Validation RMSE [{transform}]: {rmse:.6f}")

    _, best_transform = min(candidates, key=lambda x: x[0])

    X_train_final, X_test_final, final_cat_cols = prepare_features(X_all, X_test, X_all)
    y_train_final = np.log1p(y_all) if best_transform == "log" else y_all

    final_model = make_model()
    final_model.fit(X_train_final, y_train_final, categorical_feature=final_cat_cols if final_cat_cols else "auto")

    test_pred = predict_with_transform(final_model, X_test_final, best_transform)

    out = pd.DataFrame({
        "id": np.arange(len(test_pred), dtype=int),
        "capacity_mw": test_pred.astype(float),
    })
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()