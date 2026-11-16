import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

RANDOM_STATE = 42
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
TARGET = "medianHouseValue"

BASE_FEATURES = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
]

CITY_POINTS = {
    "san_francisco": (37.7749, -122.4194),
    "san_jose": (37.3382, -121.8863),
    "los_angeles": (34.0522, -118.2437),
    "san_diego": (32.7157, -117.1611),
    "sacramento": (38.5816, -121.4944),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def haversine_km(lat, lon, lat_ref, lon_ref):
    """Vectorized haversine distance in kilometers."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_ref = float(lat_ref)
    lon_ref = float(lon_ref)

    lat1 = np.radians(lat)
    lat2 = np.radians(lat_ref)
    dlat = lat1 - lat2
    dlon = np.radians(lon - lon_ref)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return 6371.0 * c


def make_stratify_labels(y: pd.Series, n_bins: int = 10):
    """Create stratification labels from target quantiles, robust to ties/caps."""
    y = pd.Series(y).astype(float)
    if y.nunique() <= 1:
        return pd.Series(0, index=y.index)

    try:
        labels = pd.qcut(y, q=n_bins, duplicates="drop")
        if labels.isna().all():
            return pd.Series(0, index=y.index)
        return labels.cat.codes
    except Exception:
        return (
            pd.Series(y)
            .rank(method="first")
            .astype(int)
            .apply(lambda r: min(int(r * n_bins / max(len(y), 1)), n_bins - 1))
        )


def add_rowwise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, per-row feature engineering."""
    out = df.copy()
    eps = 1e-6

    for col in BASE_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in [
        "housingMedianAge",
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]:
        if col in out.columns:
            out[col] = out[col].clip(lower=0.0)

    total_rooms = out["totalRooms"] + eps
    total_bedrooms = out["totalBedrooms"] + eps
    population = out["population"] + eps
    households = out["households"] + eps

    out["bedroom_ratio"] = out["totalBedrooms"] / total_rooms
    out["room_surplus"] = out["totalRooms"] - out["totalBedrooms"]
    out["rooms_per_household"] = out["totalRooms"] / households
    out["bedrooms_per_household"] = out["totalBedrooms"] / households
    out["rooms_per_population"] = out["totalRooms"] / population
    out["bedrooms_per_population"] = out["totalBedrooms"] / population

    out["household_size"] = out["population"] / households
    out["households_per_population"] = out["households"] / population

    out["income_per_household"] = out["medianIncome"] / households
    out["income_per_capita"] = out["medianIncome"] / population
    out["income_households"] = out["medianIncome"] * out["households"]
    out["income_rooms"] = out["medianIncome"] * out["totalRooms"]
    out["income_population"] = out["medianIncome"] * out["population"]
    out["income_age"] = out["medianIncome"] * out["housingMedianAge"]

    out["age_rooms"] = out["housingMedianAge"] * out["totalRooms"]
    out["age_households"] = out["housingMedianAge"] * out["households"]
    out["age_income"] = out["housingMedianAge"] * out["medianIncome"]

    out["income_sq"] = out["medianIncome"] ** 2
    out["income_cube"] = out["medianIncome"] ** 3
    out["log_income"] = np.log1p(out["medianIncome"])
    out["sqrt_income"] = np.sqrt(out["medianIncome"])

    out["log_rooms"] = np.log1p(out["totalRooms"])
    out["log_bedrooms"] = np.log1p(out["totalBedrooms"])
    out["log_population"] = np.log1p(out["population"])
    out["log_households"] = np.log1p(out["households"])
    out["log_age"] = np.log1p(out["housingMedianAge"])

    lat = out["latitude"].to_numpy(dtype=float)
    lon = out["longitude"].to_numpy(dtype=float)

    for city_name, (lat_ref, lon_ref) in CITY_POINTS.items():
        out[f"dist_km_{city_name}"] = haversine_km(lat, lon, lat_ref, lon_ref)

    dist_cols = [f"dist_km_{city}" for city in CITY_POINTS]
    if dist_cols:
        out["dist_km_min_metro"] = out[dist_cols].min(axis=1)
        out["dist_km_mean_metro"] = out[dist_cols].mean(axis=1)

    return out


class HousingFeatureBuilder:
    """
    Fit stateful transformations on the training split only, then apply
    identical transformations to validation/test splits.
    """

    def __init__(self):
        self.medians = {}
        self.scaler = StandardScaler()
        self.scale_cols = []
        self.centroid = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "HousingFeatureBuilder":
        X_fit = add_rowwise_features(X_train)

        self.medians = {
            col: float(np.nanmedian(X_fit[col].to_numpy(dtype=float)))
            for col in X_fit.columns
            if pd.api.types.is_numeric_dtype(X_fit[col])
        }

        X_fit = self._impute(X_fit)
        self.centroid = (
            float(np.nanmean(X_fit["latitude"].to_numpy(dtype=float))),
            float(np.nanmean(X_fit["longitude"].to_numpy(dtype=float))),
        )
        X_fit["dist_km_train_centroid"] = haversine_km(
            X_fit["latitude"],
            X_fit["longitude"],
            self.centroid[0],
            self.centroid[1],
        )

        candidate_scale_cols = [
            "longitude",
            "latitude",
            "housingMedianAge",
            "totalRooms",
            "totalBedrooms",
            "population",
            "households",
            "medianIncome",
            "bedroom_ratio",
            "rooms_per_household",
            "bedrooms_per_household",
            "household_size",
            "income_per_household",
            "income_per_capita",
            "income_households",
            "income_rooms",
            "income_population",
            "income_age",
            "age_rooms",
            "age_households",
            "age_income",
            "income_sq",
            "income_cube",
            "log_income",
            "sqrt_income",
            "log_rooms",
            "log_bedrooms",
            "log_population",
            "log_households",
            "log_age",
            "dist_km_train_centroid",
            "dist_km_min_metro",
            "dist_km_mean_metro",
        ]
        self.scale_cols = [col for col in candidate_scale_cols if col in X_fit.columns]

        self.scaler.fit(X_fit[self.scale_cols].to_numpy(dtype=float))
        self.feature_names = list(X_fit.columns)
        self.is_fitted = True
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if col in self.medians:
                X[col] = X[col].fillna(self.medians[col])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("HousingFeatureBuilder must be fitted before transform.")

        X = add_rowwise_features(X)
        X = self._impute(X)

        X["dist_km_train_centroid"] = haversine_km(
            X["latitude"],
            X["longitude"],
            self.centroid[0],
            self.centroid[1],
        )

        scaled = self.scaler.transform(X[self.scale_cols].to_numpy(dtype=float))
        for col, value in zip(self.scale_cols, scaled.T):
            X[f"{col}_std"] = value

        missing = [col for col in self.feature_names if col not in X.columns]
        if missing:
            for col in missing:
                X[col] = np.nan
        X = X[self.feature_names]

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


def build_model():
    """LightGBM GBDT regressor with squared-error loss for RMSE alignment."""
    return lgb.LGBMRegressor(
        boosting_type="gbdt",
        objective="regression_l2",
        metric="rmse",
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        max_bin=255,
        min_data_in_bin=5,
        feature_pre_filter=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",
    )


def postprocess_predictions(pred):
    """Conservative domain-consistent post-processing: house values are non-negative."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pred, 0.0, None)


def rmse(y_true, y_pred) -> float:
    """Official task metric: RMSE on original medianHouseValue scale."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    ensure_dir(WORKING_DIR)
    ensure_dir(SUBMISSION_DIR)

    train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    test_ids = np.arange(len(test), dtype=int)

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET} not found in train.parquet")

    y_all = train[TARGET].astype(float).clip(lower=0.0)
    X_all = train.drop(columns=[TARGET])

    if TARGET in test.columns:
        X_test = test.drop(columns=[TARGET])
    else:
        X_test = test.copy()

    for col in BASE_FEATURES:
        if col not in X_all.columns:
            X_all[col] = np.nan
        if col not in X_test.columns:
            X_test[col] = np.nan

    X_all = X_all[BASE_FEATURES]
    X_test = X_test[BASE_FEATURES]

    # Stratified 80/20 split on target deciles before any fitted transformation.
    stratify = make_stratify_labels(y_all, n_bins=10)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    builder = HousingFeatureBuilder()
    X_train = builder.fit_transform(X_train_raw)
    X_val = builder.transform(X_val_raw)
    X_test = builder.transform(X_test)

    model = build_model()
    early_stopping_rounds = 50

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
    num_iteration = best_iteration if best_iteration > 0 else None

    val_pred = postprocess_predictions(
        model.predict(X_val, num_iteration=num_iteration)
    )
    score = rmse(y_val, val_pred)

    test_pred = postprocess_predictions(
        model.predict(X_test, num_iteration=num_iteration)
    )

    submission = pd.DataFrame(
        {
            "id": test_ids,
            TARGET: test_pred,
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
