import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/test.parquet"

TARGET = "trip_duration"
ID_COL = "id"
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


def bearing_degrees(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


class Preprocessor:
    def __init__(self):
        self.numeric_cols = [
            "vendor_id",
            "passenger_count",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
        self.cat_cols = ["store_and_fwd_flag"]
        self.lon_bounds = (-74.35, -73.60)
        self.lat_bounds = (40.35, 41.05)

        self.medians = {}
        self.cat_mappings = {}
        self.feature_cols = []

    def _clean(self, df):
        df = df.copy()

        drop_cols = [ID_COL, "dropoff_datetime", TARGET]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        for col in self.numeric_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "passenger_count" in df.columns:
            df.loc[df["passenger_count"] == 0, "passenger_count"] = np.nan

        for col, bounds in [
            ("pickup_longitude", self.lon_bounds),
            ("dropoff_longitude", self.lon_bounds),
            ("pickup_latitude", self.lat_bounds),
            ("dropoff_latitude", self.lat_bounds),
        ]:
            valid = df[col].notna() & df[col].between(bounds[0], bounds[1])
            df.loc[~valid, col] = np.nan

        if "pickup_datetime" in df.columns:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        else:
            df["pickup_datetime"] = pd.NaT

        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].astype("object")

        return df

    def _time_features(self, df):
        dt = df["pickup_datetime"]

        df["pickup_hour"] = dt.dt.hour
        df["pickup_minute"] = dt.dt.minute
        df["pickup_dayofweek"] = dt.dt.dayofweek
        df["pickup_is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

        hour = df["pickup_hour"].astype(float)
        df["pickup_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["pickup_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

        return df

    def _geo_features(self, df):
        lon1 = df["pickup_longitude"].to_numpy(dtype=float)
        lat1 = df["pickup_latitude"].to_numpy(dtype=float)
        lon2 = df["dropoff_longitude"].to_numpy(dtype=float)
        lat2 = df["dropoff_latitude"].to_numpy(dtype=float)

        df["haversine"] = haversine_km(lon1, lat1, lon2, lat2)
        df["haversine_sq"] = df["haversine"] ** 2
        df["bearing"] = bearing_degrees(lon1, lat1, lon2, lat2)

        return df

    def _encode_categoricals(self, df, fit):
        if fit:
            for col in self.cat_cols:
                values = df[col].dropna().astype(str).unique()
                categories = sorted(set(values.tolist()))

                mode = df[col].mode(dropna=True)
                mode = str(mode.iloc[0]) if len(mode) else "missing"
                if mode not in categories:
                    categories.append(mode)

                self.cat_mappings[col] = {cat: i for i, cat in enumerate(categories)}
                df[col] = df[col].astype(str).map(self.cat_mappings[col]).fillna(-1).astype(int)
        else:
            for col in self.cat_cols:
                if col not in df.columns:
                    df[col] = "missing"

                mapping = self.cat_mappings.get(col, {})
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        return df

    def fit(self, df):
        df = self._clean(df)
        df = self._time_features(df)
        df = self._geo_features(df)
        df = self._encode_categoricals(df, fit=True)

        df = df.drop(columns=["pickup_datetime"], errors="ignore")

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self.feature_cols = list(df.columns)

        for col in self.feature_cols:
            median = df[col].median()
            self.medians[col] = float(median) if pd.notna(median) else 0.0

        df = df.fillna(self.medians)
        return df

    def transform(self, df):
        df = self._clean(df)
        df = self._time_features(df)
        df = self._geo_features(df)
        df = self._encode_categoricals(df, fit=False)

        df = df.drop(columns=["pickup_datetime"], errors="ignore")
        df = df.reindex(columns=self.feature_cols, fill_value=np.nan)

        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in self.medians:
                df[col] = df[col].fillna(self.medians[col])
            else:
                df[col] = df[col].fillna(0.0)

        return df


def make_lgbm(n_estimators=2000):
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_xgb(n_estimators=2000):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )


def make_catboost(iterations=2000):
    return cb.CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
    )


def train_lgbm(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    best = getattr(model, "best_iteration_", None)
    if best is None or best <= 0:
        best = model.n_estimators
    return int(best)


def train_xgb(model, X_train, y_train, X_val, y_val):
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
    except TypeError:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    best = getattr(model, "best_iteration", None)
    if best is None:
        return None
    return int(best)


def train_catboost(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=0,
    )

    best = model.get_best_iteration()
    if best is None or best < 0:
        return None
    return int(best)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    y_full = train_df[TARGET].clip(lower=0).astype(float)
    X_full_raw = train_df.drop(columns=[TARGET])

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full_raw,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_full_log = np.log1p(y_full)

    val_pre = Preprocessor()
    X_train = val_pre.fit(X_train_raw)
    X_val = val_pre.transform(X_val_raw)

    model_factories = {
        "lgbm": make_lgbm,
        "xgb": make_xgb,
        "catboost": make_catboost,
    }

    trainers = {
        "lgbm": train_lgbm,
        "xgb": train_xgb,
        "catboost": train_catboost,
    }

    val_models = {}
    best_iters = {}
    val_scores = {}

    for name, factory in model_factories.items():
        try:
            model = factory()
            best_iter = trainers[name](model, X_train, y_train_log, X_val, y_val_log)

            val_pred = model.predict(X_val)
            val_pred = np.expm1(np.clip(val_pred, 0.0, 20.0))
            score = rmsle(y_val, val_pred)

            val_models[name] = model
            best_iters[name] = best_iter
            val_scores[name] = score
        except Exception:
            continue

    if not val_models:
        raise RuntimeError("No model could be trained.")

    full_pre = Preprocessor()
    X_full = full_pre.fit(X_full_raw)
    X_test = full_pre.transform(test_df)

    final_models = {}
    for name, model in val_models.items():
        try:
            best_iter = best_iters.get(name)

            if name == "lgbm":
                n_estimators = best_iter if best_iter is not None and best_iter > 0 else 2000
                final_model = make_lgbm(n_estimators=n_estimators)
                final_model.fit(X_full, y_full_log)
            elif name == "xgb":
                n_estimators = (best_iter + 1) if best_iter is not None and best_iter >= 0 else 2000
                final_model = make_xgb(n_estimators=n_estimators)
                final_model.fit(X_full, y_full_log)
            else:
                iterations = (best_iter + 1) if best_iter is not None and best_iter >= 0 else 2000
                final_model = make_catboost(iterations=iterations)
                final_model.fit(X_full, y_full_log, verbose=0)

            final_models[name] = final_model
        except Exception:
            final_models[name] = model

    if not final_models:
        final_models = val_models

    weights = np.array([1.0 / max(val_scores[name], 1e-8) for name in final_models], dtype=float)
    weights = weights / weights.sum()

    log_preds = np.zeros(len(X_test), dtype=float)
    for (name, model), weight in zip(final_models.items(), weights):
        log_preds += weight * model.predict(X_test)

    preds = np.expm1(np.clip(log_preds, 0.0, 20.0))
    preds = np.clip(preds, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "trip_duration": preds,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()