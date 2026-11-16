import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

INPUT_DIR = "./input"
WORKING_DIR = "./working"
PROCESSED_DIR = os.path.join(WORKING_DIR, "processed")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "traffic_volume"
DATE_COL = "date_time"
NUMERIC_COLS = ["temp", "rain_1h", "snow_1h", "clouds_all"]
CATEGORICAL_COLS = ["holiday", "weather_main", "weather_description"]
EXCLUDED_COLS = {
    DATE_COL,
    TARGET,
    "test_id",
    "id",
    "ID",
    "Id",
    "index",
    "Index",
    "row_id",
    "Unnamed: 0",
}


def clean_raw_data(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    df = df.copy()

    if is_test:
        df["test_id"] = np.arange(len(df), dtype=np.int64)

    if DATE_COL not in df.columns:
        df[DATE_COL] = pd.NaT
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        else:
            df[col] = np.nan

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace("", "Unknown")
                .fillna("Unknown")
            )
        else:
            df[col] = "Unknown"

    if not is_test:
        if TARGET not in df.columns:
            raise ValueError("Training data must contain the target column.")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].clip(lower=0)

    return df


class TrafficFeatureBuilder:
    def __init__(self):
        self.numeric_medians = {}
        self.all_numeric_medians = {}
        self.scaler = None
        self.cat_freq = {}
        self.cat_code = {}
        self.dummy_cols = []
        self.feature_names = []
        self.numeric_feature_cols = []

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = df[DATE_COL]

        df["hour"] = dt.dt.hour.fillna(0).astype(float)
        df["minute"] = dt.dt.minute.fillna(0).astype(float)
        df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(float)
        df["month"] = dt.dt.month.fillna(0).astype(float)
        df["day"] = dt.dt.day.fillna(0).astype(float)
        df["weekofyear"] = dt.dt.isocalendar().week.fillna(0).astype(float)
        df["dayofyear"] = dt.dt.dayofyear.fillna(0).astype(float)
        df["quarter"] = dt.dt.quarter.fillna(0).astype(float)

        df["is_weekend"] = (df["dayofweek"] >= 5).astype(float)
        df["is_month_start"] = dt.dt.is_month_start.fillna(False).astype(float)
        df["is_month_end"] = dt.dt.is_month_end.fillna(False).astype(float)
        df["is_quarter_start"] = dt.dt.is_quarter_start.fillna(False).astype(float)
        df["is_quarter_end"] = dt.dt.is_quarter_end.fillna(False).astype(float)
        df["is_year_start"] = dt.dt.is_year_start.fillna(False).astype(float)
        df["is_year_end"] = dt.dt.is_year_end.fillna(False).astype(float)

        cyclical_periods = {
            "hour": 24,
            "dayofweek": 7,
            "month": 12,
            "weekofyear": 52,
            "dayofyear": 365,
        }
        for col, period in cyclical_periods.items():
            df[f"{col}_sin"] = np.sin(2.0 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2.0 * np.pi * df[col] / period)

        df["hour_dayofweek"] = df["hour"] * 7.0 + df["dayofweek"]
        df["hour_weekend"] = df["hour"] * 2.0 + df["is_weekend"]
        df["dayofweek_month"] = df["dayofweek"] * 12.0 + df["month"]

        return df

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["temp_c"] = df["temp"] - 273.15
        df["precip_total"] = df["rain_1h"] + df["snow_1h"]
        df["precip_log"] = np.log1p(df["precip_total"].clip(lower=0))
        df["rain_flag"] = (df["rain_1h"] > 0).astype(float)
        df["snow_flag"] = (df["snow_1h"] > 0).astype(float)
        df["precip_flag"] = (df["precip_total"] > 0).astype(float)
        df["clouds_log"] = np.log1p(df["clouds_all"].clip(lower=0))

        df["temp_precip"] = df["temp_c"] * df["precip_total"]
        df["temp_snow"] = df["temp_c"] * df["snow_1h"]
        df["cloud_precip"] = df["clouds_all"] * df["precip_total"]
        df["low_temp_precip"] = ((df["temp_c"] < 0) & (df["precip_total"] > 0)).astype(
            float
        )
        df["high_cloud_precip"] = (
            (df["clouds_all"] > 75) & (df["precip_total"] > 0)
        ).astype(float)

        severity_map = {
            "Clear": 0,
            "Few Clouds": 1,
            "Scattered Clouds": 2,
            "Broken Clouds": 3,
            "Overcast Clouds": 4,
            "Mist": 5,
            "Smoke": 5,
            "Haze": 5,
            "Dust": 5,
            "Fog": 5,
            "Sand": 5,
            "Ash": 5,
            "Squall": 5,
            "Tornado": 5,
            "Drizzle": 6,
            "Rain": 7,
            "Snow": 8,
            "Sleet": 8,
            "Shower": 7,
            "Thunderstorm": 9,
            "Unknown": 0,
        }
        df["weather_main_severity"] = (
            df["weather_main"].map(severity_map).fillna(0).astype(float)
        )

        return df

    def _add_categorical_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            for col in CATEGORICAL_COLS:
                counts = df[col].value_counts(normalize=True)
                self.cat_freq[col] = counts.to_dict()

                sorted_cats = sorted(df[col].unique().tolist())
                self.cat_code[col] = {cat: i for i, cat in enumerate(sorted_cats)}

                dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
                self.dummy_cols.extend(dummies.columns.tolist())
                df = pd.concat([df, dummies], axis=1)
        else:
            for col in CATEGORICAL_COLS:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
                df = pd.concat([df, dummies], axis=1)

        for col in CATEGORICAL_COLS:
            df[f"{col}_freq"] = (
                df[col].map(self.cat_freq[col]).fillna(0.0).astype(float)
            )
            df[f"{col}_code"] = df[col].map(self.cat_code[col]).fillna(-1).astype(float)

        return df

    def _get_numeric_feature_cols(self, df: pd.DataFrame) -> list:
        numeric_cols = []
        for col in df.columns:
            if col in EXCLUDED_COLS or col in CATEGORICAL_COLS:
                continue
            if str(col).lower().endswith("_id"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        return numeric_cols

    def _impute_base_numeric(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cols = [c for c in NUMERIC_COLS if c in df.columns]

        if fit:
            medians = df[cols].median().fillna(0.0)
            self.numeric_medians = medians.to_dict()
            df[cols] = df[cols].fillna(medians)
        else:
            df[cols] = df[cols].fillna(self.numeric_medians)

        return df

    def _transform(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        df = self._impute_base_numeric(df, fit)
        df = self._add_time_features(df)
        df = self._add_weather_features(df)
        df = self._add_categorical_features(df, fit)

        if fit:
            df = df.replace([np.inf, -np.inf], np.nan)
            numeric_cols = self._get_numeric_feature_cols(df)

            if len(numeric_cols) == 0:
                raise ValueError("No numeric features were created.")

            self.numeric_feature_cols = numeric_cols
            medians = df[numeric_cols].median().fillna(0.0)
            self.all_numeric_medians = medians.to_dict()
            df[numeric_cols] = df[numeric_cols].fillna(medians)

            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(df[numeric_cols].astype(float))
            df = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)
            self.feature_names = numeric_cols
        else:
            if len(self.numeric_feature_cols) == 0:
                raise ValueError("Feature builder must be fitted before transform.")

            df = df.reindex(columns=self.numeric_feature_cols, fill_value=0.0)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(self.all_numeric_medians)

            scaled = self.scaler.transform(df.astype(float))
            df = pd.DataFrame(scaled, columns=self.numeric_feature_cols, index=df.index)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._transform(df, fit=True)
        self.feature_names = features.columns.tolist()
        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._transform(df, fit=False)
        features = features.reindex(columns=self.feature_names, fill_value=0.0)
        return features

    def save(self, path: str):
        payload = {
            "feature_names": self.feature_names,
            "numeric_feature_cols": self.numeric_feature_cols,
            "numeric_medians": {k: float(v) for k, v in self.numeric_medians.items()},
            "all_numeric_medians": {
                k: float(v) for k, v in self.all_numeric_medians.items()
            },
            "cat_freq": self.cat_freq,
            "cat_code": self.cat_code,
            "dummy_cols": self.dummy_cols,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


train_raw = clean_raw_data(
    pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet")), is_test=False
)
test_raw = clean_raw_data(
    pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet")), is_test=True
)

train_raw = train_raw.dropna(subset=[DATE_COL]).copy()
if len(train_raw) == 0:
    raise ValueError("No usable training rows after cleaning.")

train_raw = train_raw.sort_values(DATE_COL).reset_index(drop=True)

if len(train_raw) < 2:
    raise ValueError("At least two training rows are required for a validation split.")

span_days = (train_raw[DATE_COL].max() - train_raw[DATE_COL].min()).days

if len(train_raw) > 1000 and span_days >= 730:
    split_time = train_raw[DATE_COL].max() - pd.Timedelta(days=365)
else:
    split_idx = max(1, min(len(train_raw) - 1, int(np.floor(len(train_raw) * 0.9))))
    split_time = train_raw[DATE_COL].iloc[split_idx]

train_split = train_raw[train_raw[DATE_COL] < split_time].copy()
val_split = train_raw[train_raw[DATE_COL] >= split_time].copy()

if len(train_split) == 0 or len(val_split) == 0:
    split_idx = max(1, min(len(train_raw) - 1, int(np.floor(len(train_raw) * 0.9))))
    split_time = train_raw[DATE_COL].iloc[split_idx]
    train_split = train_raw[train_raw[DATE_COL] < split_time].copy()
    val_split = train_raw[train_raw[DATE_COL] >= split_time].copy()

if len(val_split) == 0:
    raise ValueError("Validation split is empty after chronological splitting.")

builder = TrafficFeatureBuilder()

train_features = builder.fit_transform(train_split)
val_features = builder.transform(val_split)
test_features = builder.transform(test_raw)

train_targets = train_split[TARGET].to_frame(TARGET).reset_index(drop=True)
val_targets = val_split[TARGET].to_frame(TARGET).reset_index(drop=True)
test_ids = test_raw[["test_id"]].reset_index(drop=True)

assert len(train_features) == len(
    train_targets
), "Train feature/target length mismatch."
assert len(val_features) == len(
    val_targets
), "Validation feature/target length mismatch."
assert len(test_features) == len(test_ids), "Test feature/id length mismatch."

train_features.to_parquet(
    os.path.join(PROCESSED_DIR, "train_features.parquet"), index=False
)
val_features.to_parquet(
    os.path.join(PROCESSED_DIR, "val_features.parquet"), index=False
)
test_features.to_parquet(
    os.path.join(PROCESSED_DIR, "test_features.parquet"), index=False
)

train_targets.to_parquet(
    os.path.join(PROCESSED_DIR, "train_targets.parquet"), index=False
)
val_targets.to_parquet(os.path.join(PROCESSED_DIR, "val_targets.parquet"), index=False)
test_ids.to_parquet(os.path.join(PROCESSED_DIR, "test_ids.parquet"), index=False)

builder.save(os.path.join(PROCESSED_DIR, "feature_builder_state.json"))

with open(os.path.join(PROCESSED_DIR, "split_info.json"), "w") as f:
    json.dump(
        {
            "split_type": "chronological",
            "split_time": str(split_time),
            "train_rows": int(len(train_features)),
            "val_rows": int(len(val_features)),
            "test_rows": int(len(test_features)),
            "feature_count": int(len(builder.feature_names)),
        },
        f,
        indent=2,
    )


def load_processed(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing processed artifact: {path}. "
            "Run the data_processing_and_feature_engineering step first."
        )
    return pd.read_parquet(path)


def sanitize_feature_matrix(df: pd.DataFrame, reference_columns=None) -> pd.DataFrame:
    df = df.copy()

    if reference_columns is not None:
        df = df.reindex(columns=reference_columns, fill_value=0.0)

    df = df.astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df


X_train = load_processed("train_features.parquet")
X_val = load_processed("val_features.parquet")
X_test = load_processed("test_features.parquet")

y_train = load_processed("train_targets.parquet")[TARGET].to_numpy(dtype=np.float64)
y_val = load_processed("val_targets.parquet")[TARGET].to_numpy(dtype=np.float64)

test_ids_df = load_processed("test_ids.parquet")
if "test_id" in test_ids_df.columns:
    test_ids = test_ids_df["test_id"].to_numpy()
else:
    test_ids = np.arange(len(test_ids_df), dtype=np.int64)

feature_names = X_train.columns.tolist()

X_train = sanitize_feature_matrix(X_train, feature_names)
X_val = sanitize_feature_matrix(X_val, feature_names)
X_test = sanitize_feature_matrix(X_test, feature_names)

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

train_mask = np.isfinite(y_train)
y_train = y_train[train_mask]
X_train = X_train.loc[train_mask].reset_index(drop=True)

val_mask = np.isfinite(y_val)
y_val = y_val[val_mask]
X_val = X_val.loc[val_mask].reset_index(drop=True)

if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("No usable training rows after target filtering.")

if len(X_val) == 0 or len(y_val) == 0:
    raise ValueError("No usable validation rows after target filtering.")

if len(X_test) != len(test_ids):
    n = min(len(X_test), len(test_ids))
    X_test = X_test.iloc[:n].reset_index(drop=True)
    test_ids = test_ids[:n]

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "random_seed": 42,
    "nthread": max(1, os.cpu_count() or 1),
    "verbosity": -1,
}

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=feature_names,
    free_raw_data=False,
)

val_data = lgb.Dataset(
    X_val,
    label=y_val,
    feature_name=feature_names,
    free_raw_data=False,
    reference=train_data,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=0),
]

booster = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[val_data],
    valid_names=["val"],
    callbacks=callbacks,
)

best_iteration = int(getattr(booster, "best_iteration", 0) or 0)

model_path = os.path.join(MODEL_DIR, "lgbm_best.txt")
save_num_iteration = best_iteration if best_iteration > 0 else None

try:
    booster.save_model(model_path, num_iteration=save_num_iteration)
except TypeError:
    booster.save_model(model_path)

loaded_model = lgb.Booster(model_file=model_path)


def predict_with_saved_model(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    num_iteration = best_iteration if best_iteration > 0 else None

    try:
        pred = model.predict(X, num_iteration=num_iteration)
    except TypeError:
        pred = model.predict(X)

    pred = np.asarray(pred, dtype=np.float64)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    pred = np.clip(pred, 0.0, None)
    return pred


val_pred = predict_with_saved_model(loaded_model, X_val)
score = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

test_pred = predict_with_saved_model(loaded_model, X_test)

if len(test_pred) != len(test_ids):
    n = min(len(test_pred), len(test_ids))
    test_pred = test_pred[:n]
    test_ids = test_ids[:n]

submission = pd.DataFrame(
    {
        "id": test_ids.astype(np.int64),
        "traffic_volume": test_pred.astype(np.float64),
    }
)

submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(
    f"Validation RMSE: {score:.6f} | best_iteration={best_iteration} | rows={len(submission)}"
)
print(f"Final Validation Score: {score}")
