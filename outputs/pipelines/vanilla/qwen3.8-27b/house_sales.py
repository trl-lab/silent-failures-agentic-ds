import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/house_sales/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/house_sales/data/test.parquet"
TARGET = "price"
RANDOM_STATE = 42


def parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df

    date_col = df["date"]
    dt = None

    if pd.api.types.is_datetime64_any_dtype(date_col):
        dt = date_col
    else:
        for fmt in ["%Y%m%dT%H%M%S", None]:
            try:
                candidate = pd.to_datetime(date_col, format=fmt, errors="coerce")
                if candidate.notna().any():
                    dt = candidate
                    break
            except Exception:
                pass

    if dt is None:
        dt = pd.to_datetime(date_col, errors="coerce")

    origin = pd.Timestamp("2014-01-01")
    df["date_year"] = dt.dt.year.astype("float64")
    df["date_month"] = dt.dt.month.astype("float64")
    df["date_day"] = dt.dt.day.astype("float64")
    df["date_dayofweek"] = dt.dt.dayofweek.astype("float64")
    df["date_days"] = (dt - origin).dt.days.astype("float64")
    df = df.drop(columns=["date"], errors="ignore")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date_year" in df.columns and "yr_built" in df.columns:
        df["age"] = df["date_year"] - df["yr_built"]

    if "yr_renovated" in df.columns:
        df["renovated"] = (df["yr_renovated"] > 0).astype("float64")
        if "date_year" in df.columns:
            df["age_renovated"] = np.where(
                df["yr_renovated"] > 0,
                df["date_year"] - df["yr_renovated"],
                np.nan,
            )

    if "sqft_living" in df.columns and "bedrooms" in df.columns:
        df["sqft_per_bedroom"] = df["sqft_living"] / (df["bedrooms"] + 1)

    if "sqft_living" in df.columns and "bathrooms" in df.columns:
        df["sqft_per_bathroom"] = df["sqft_living"] / (df["bathrooms"] + 1)

    if "sqft_living" in df.columns and "sqft_lot" in df.columns:
        df["lot_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)

    if "sqft_above" in df.columns and "sqft_living" in df.columns:
        df["above_ratio"] = df["sqft_above"] / (df["sqft_living"] + 1)

    if "sqft_basement" in df.columns and "sqft_living" in df.columns:
        df["basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1)

    if "sqft_living15" in df.columns and "sqft_living" in df.columns:
        df["neighborhood_sqft_diff"] = df["sqft_living15"] - df["sqft_living"]

    if "sqft_lot15" in df.columns and "sqft_lot" in df.columns:
        df["neighborhood_lot_diff"] = df["sqft_lot15"] - df["sqft_lot"]

    if "view" in df.columns and "waterfront" in df.columns:
        df["view_waterfront"] = df["view"] * df["waterfront"]

    if "grade" in df.columns and "sqft_living" in df.columns:
        df["grade_sqft"] = df["grade"] * df["sqft_living"]

    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    cat_cols = [
        col
        for col in train_df.columns
        if train_df[col].dtype == object or str(train_df[col].dtype) == "category"
    ]

    if not cat_cols:
        return train_df, test_df

    train_cat = train_df[cat_cols].copy()
    test_cat = test_df[cat_cols].copy()

    train_dum = pd.get_dummies(train_cat, prefix=cat_cols, dummy_na=False)
    test_dum = pd.get_dummies(test_cat, prefix=cat_cols, dummy_na=False)

    test_dum = test_dum.reindex(columns=train_dum.columns, fill_value=0)

    train_df = pd.concat([train_df.drop(columns=cat_cols), train_dum], axis=1)
    test_df = pd.concat([test_df.drop(columns=cat_cols), test_dum], axis=1)

    return train_df, test_df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = train_df.dropna(subset=[TARGET])
    y_train = train_df[TARGET].astype("float64").copy()
    train_df = train_df.drop(columns=[TARGET])
    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    train_df = parse_date_features(train_df)
    test_df = parse_date_features(test_df)

    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    train_df, test_df = encode_categoricals(train_df, test_df)

    for df in (train_df, test_df):
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "category":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = [
        col
        for col in train_df.columns
        if train_df[col].notna().any() and train_df[col].nunique(dropna=True) > 1
    ]

    train_df = train_df[feature_cols]
    test_df = test_df.reindex(columns=feature_cols)

    return train_df, y_train, test_df


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
        ),
        "catboost": CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train, X_test = prepare_data(train_df, test_df)

    if len(X_train) == 0:
        predictions = np.zeros(len(X_test), dtype="float64")
    elif len(X_train) < 10:
        predictions = np.full(len(X_test), float(y_train.mean()), dtype="float64")
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        models = make_models()
        val_scores = {}
        fitted_models = {}

        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                val_pred = model.predict(X_val)
                val_pred = np.asarray(val_pred, dtype="float64")
                val_pred = np.nan_to_num(val_pred, nan=float(y_tr.mean()), posinf=0.0, neginf=0.0)
                rmse = float(mean_squared_error(y_val, val_pred, squared=False))
                if np.isfinite(rmse) and rmse > 0:
                    val_scores[name] = rmse
                    fitted_models[name] = model
            except Exception:
                continue

        if not fitted_models:
            try:
                model = LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            except Exception:
                predictions = np.full(len(X_test), float(y_train.mean()), dtype="float64")
        else:
            weights = {name: 1.0 / score for name, score in val_scores.items()}
            total_weight = sum(weights.values())
            if total_weight <= 0 or not np.isfinite(total_weight):
                weights = {name: 1.0 / len(weights) for name in weights}
                total_weight = sum(weights.values())

            test_predictions = np.zeros(len(X_test), dtype="float64")
            for name, model in fitted_models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    pred = np.asarray(pred, dtype="float64")
                    pred = np.nan_to_num(pred, nan=float(y_train.mean()), posinf=0.0, neginf=0.0)
                    test_predictions += (weights[name] / total_weight) * pred
                except Exception:
                    continue

            predictions = test_predictions

    predictions = np.asarray(predictions, dtype="float64")
    predictions = np.nan_to_num(predictions, nan=float(y_train.mean()), posinf=0.0, neginf=0.0)
    predictions = np.maximum(predictions, 0.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype="int64"),
            "price": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()