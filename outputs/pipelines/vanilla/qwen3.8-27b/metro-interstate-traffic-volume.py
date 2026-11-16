import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/test.parquet"
TARGET = "traffic_volume"
DATE_COL = "date_time"
CAT_COLS = ["holiday", "weather_main", "weather_description"]
NUMERIC_COLS = ["temp", "rain_1h", "snow_1h", "clouds_all"]


def sanitize(value):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")


class TrafficPreprocessor:
    def __init__(self):
        self.cat_cols = CAT_COLS
        self.numeric_feature_cols = [
            "temp",
            "rain_1h",
            "snow_1h",
            "clouds_all",
            "precip",
            "temp_c",
            "hour",
            "dayofweek",
            "month",
            "day",
            "year",
            "dayofyear",
            "weekofyear",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_rain",
            "is_snow",
            "is_clear",
        ]
        self.cat_categories = {}
        self.medians = pd.Series(dtype=float)
        self.feature_cols = []

    def _extract(self, df):
        df = df.copy()

        dt = pd.to_datetime(df.get(DATE_COL), errors="coerce")
        hour = dt.dt.hour.astype(float)
        dow = dt.dt.dayofweek.astype(float)
        month = dt.dt.month.astype(float)

        df["hour"] = hour
        df["dayofweek"] = dow
        df["month"] = month
        df["day"] = dt.dt.day.astype(float)
        df["year"] = dt.dt.year.astype(float)
        df["dayofyear"] = dt.dt.dayofyear.astype(float)
        df["weekofyear"] = dt.dt.isocalendar().week.astype(float)

        is_weekend = (dow >= 5).astype(float)
        is_weekend[dow.isna()] = np.nan
        df["is_weekend"] = is_weekend

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        df["precip"] = df["rain_1h"].fillna(0.0) + df["snow_1h"].fillna(0.0)
        df["temp_c"] = df["temp"] - 273.15

        is_rain = (df["rain_1h"] > 0).astype(float)
        is_rain[df["rain_1h"].isna()] = np.nan
        df["is_rain"] = is_rain

        is_snow = (df["snow_1h"] > 0).astype(float)
        is_snow[df["snow_1h"].isna()] = np.nan
        df["is_snow"] = is_snow

        is_clear = (df["weather_main"] == "Clear").astype(float)
        is_clear[df["weather_main"].isna()] = np.nan
        df["is_clear"] = is_clear

        return df

    def _clean_categoricals(self, df):
        df = df.copy()
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].fillna("missing").astype(str).map(sanitize)
        return df

    def fit(self, df):
        df = self._clean_categoricals(self._extract(df))

        self.cat_categories = {
            col: sorted(df[col].unique().tolist()) for col in self.cat_cols
        }

        self.numeric_feature_cols = [
            col for col in self.numeric_feature_cols if col in df.columns
        ]
        self.medians = df[self.numeric_feature_cols].median()

        self.feature_cols = self.numeric_feature_cols.copy()
        for col in self.cat_cols:
            for cat in self.cat_categories[col]:
                self.feature_cols.append(f"{col}_{cat}")

        return self

    def transform(self, df):
        df = self._clean_categoricals(self._extract(df))

        numeric = pd.DataFrame(index=df.index)
        for col in self.numeric_feature_cols:
            numeric[col] = df[col]
        numeric = numeric.fillna(self.medians).fillna(0.0)

        dummy_frames = []
        for col in self.cat_cols:
            cats = self.cat_categories.get(col, [])
            dummy = pd.DataFrame(index=df.index)
            for cat in cats:
                dummy[f"{col}_{cat}"] = (df[col] == cat).astype(np.uint8)
            dummy_frames.append(dummy)

        if dummy_frames:
            dummies = pd.concat(dummy_frames, axis=1)
        else:
            dummies = pd.DataFrame(index=df.index)

        out = pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return out[self.feature_cols]


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if len(test_df) == 0:
        pd.DataFrame({"id": [], "traffic_volume": []}).to_csv("predictions.csv", index=False)
        return

    y_train = train_df[TARGET].astype(float)

    val_size = 0.2 if len(train_df) > 20 else 0.0
    if val_size > 0:
        train_part = train_df.iloc[: int(len(train_df) * (1 - val_size))]
        val_part = train_df.iloc[int(len(train_df) * (1 - val_size)) :]

        val_pre = TrafficPreprocessor().fit(train_part)
        X_val = val_pre.transform(val_part)
        y_val = val_part[TARGET].astype(float)

        val_models = make_models()
        val_scores = {}
        for name, model in val_models.items():
            model.fit(val_pre.transform(train_part), train_part[TARGET].astype(float))
            pred = model.predict(X_val)
            val_scores[name] = float(np.sqrt(mean_squared_error(y_val, pred)))
    else:
        val_scores = {}

    pre = TrafficPreprocessor().fit(train_df)
    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)

    models = make_models()
    test_preds = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.asarray(pred, dtype=float)
        pred = np.clip(pred, 0.0, None)
        test_preds.append(pred)

    final_pred = np.mean(test_preds, axis=0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "traffic_volume": final_pred,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()