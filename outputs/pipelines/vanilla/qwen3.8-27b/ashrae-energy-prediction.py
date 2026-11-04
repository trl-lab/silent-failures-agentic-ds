import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/test.parquet"
TARGET = "meter_reading"
ID_COL = "id"
TS_COL = "timestamp"
RANDOM_STATE = 42
MAX_TRAIN_ROWS = 1_000_000


def read_parquet_safe(path, columns=None):
    try:
        if columns is not None:
            return pd.read_parquet(path, columns=columns)
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path)


def compute_dow(year, month, day):
    dow = pd.Series(np.nan, index=year.index, dtype="float32")
    valid = year.notna() & month.notna() & day.notna()
    if not valid.any():
        return dow

    y = year[valid].astype("int64")
    m = month[valid].astype("int64")
    d = day[valid].astype("int64")

    m_adj = m.copy()
    y_adj = y.copy()
    jan_feb = m <= 2
    m_adj[jan_feb] += 12
    y_adj[jan_feb] -= 1

    K = y_adj % 100
    J = y_adj // 100
    h = (d + (13 * (m_adj + 1)) // 5 + K + K // 4 + J // 4 + 5 * J) % 7
    dow.loc[valid] = ((h + 5) % 7).astype("float32")
    return dow


def extract_features(df):
    df = df.copy()

    if TS_COL in df.columns:
        ts = df[TS_COL]
        if pd.api.types.is_datetime64_any_dtype(ts):
            hour = ts.dt.hour.astype("float32")
            day = ts.dt.day.astype("float32")
            month = ts.dt.month.astype("float32")
            year = ts.dt.year.astype("float32")
            dow = ts.dt.dayofweek.astype("float32")
        else:
            ts_str = ts.astype(str)
            hour = pd.to_numeric(ts_str.str[11:13], errors="coerce").astype("float32")
            day = pd.to_numeric(ts_str.str[8:10], errors="coerce").astype("float32")
            month = pd.to_numeric(ts_str.str[5:7], errors="coerce").astype("float32")
            year = pd.to_numeric(ts_str.str[0:4], errors="coerce").astype("float32")
            dow = compute_dow(year, month, day)
    else:
        hour = pd.Series(np.nan, index=df.index, dtype="float32")
        day = pd.Series(np.nan, index=df.index, dtype="float32")
        month = pd.Series(np.nan, index=df.index, dtype="float32")
        year = pd.Series(np.nan, index=df.index, dtype="float32")
        dow = pd.Series(np.nan, index=df.index, dtype="float32")

    df["hour"] = hour.fillna(-1).astype("float32")
    df["day"] = day.fillna(-1).astype("float32")
    df["month"] = month.fillna(-1).astype("float32")
    df["year"] = year.fillna(-1).astype("float32")
    df["dayofweek"] = dow.fillna(-1).astype("float32")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("float32")

    valid_hour = df["hour"] >= 0
    valid_dow = df["dayofweek"] >= 0
    valid_month = df["month"] >= 0

    df["hour_sin"] = np.where(valid_hour, np.sin(2.0 * np.pi * df["hour"] / 24.0), 0.0).astype("float32")
    df["hour_cos"] = np.where(valid_hour, np.cos(2.0 * np.pi * df["hour"] / 24.0), 0.0).astype("float32")
    df["dow_sin"] = np.where(valid_dow, np.sin(2.0 * np.pi * df["dayofweek"] / 7.0), 0.0).astype("float32")
    df["dow_cos"] = np.where(valid_dow, np.cos(2.0 * np.pi * df["dayofweek"] / 7.0), 0.0).astype("float32")
    df["month_sin"] = np.where(valid_month, np.sin(2.0 * np.pi * df["month"] / 12.0), 0.0).astype("float32")
    df["month_cos"] = np.where(valid_month, np.cos(2.0 * np.pi * df["month"] / 12.0), 0.0).astype("float32")

    if "building_id" in df.columns:
        bid = pd.to_numeric(df["building_id"], errors="coerce")
    else:
        bid = pd.Series(np.nan, index=df.index, dtype="float64")
    if "meter" in df.columns:
        meter = pd.to_numeric(df["meter"], errors="coerce")
    else:
        meter = pd.Series(np.nan, index=df.index, dtype="float64")

    df["building_id_cat"] = (bid.fillna(-1).astype("int32") + 1).astype("category")
    df["meter_cat"] = (meter.fillna(-1).astype("int32") + 1).astype("category")

    feature_cols = [
        "building_id_cat",
        "meter_cat",
        "hour",
        "day",
        "month",
        "year",
        "dayofweek",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    return df[feature_cols]


def make_model():
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_cols = [c for c in ["building_id", "meter", TS_COL, TARGET] if c]
    train_raw = read_parquet_safe(TRAIN_PATH, train_cols)
    test_cols = [c for c in ["building_id", "meter", TS_COL] if c]
    test_raw = read_parquet_safe(TEST_PATH, test_cols)

    if TARGET not in train_raw.columns:
        raise ValueError("Target column not found in train data.")

    train_raw = train_raw.dropna(subset=[TARGET])
    train_raw[TARGET] = train_raw[TARGET].astype("float64")
    train_raw[TARGET] = np.log1p(train_raw[TARGET].clip(lower=0.0))

    if len(train_raw) > MAX_TRAIN_ROWS:
        train_raw = train_raw.iloc[:MAX_TRAIN_ROWS].reset_index(drop=True)

    X_train = extract_features(train_raw)
    y_train = train_raw[TARGET].astype("float32").reset_index(drop=True)

    X_test = extract_features(test_raw)

    if len(X_train) > 1000:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.10, random_state=RANDOM_STATE, shuffle=True
        )
    else:
        X_fit, X_val, y_fit, y_val = X_train, pd.DataFrame(columns=X_train.columns), y_train, pd.Series(dtype="float32")

    model = make_model()
    best_iter = 300

    if len(X_val) > 0:
        try:
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            best_iter = int(max(50, getattr(model, "best_iteration_", 300) + 20))
        except Exception:
            model.fit(X_fit, y_fit)
    else:
        model.fit(X_fit, y_fit)

    final_model = make_model()
    final_model.n_estimators = best_iter
    final_model.fit(X_train, y_train)

    pred = final_model.predict(X_test)
    pred = np.expm1(pred).clip(min=0.0)

    out = pd.DataFrame({
        ID_COL: np.arange(len(test_raw), dtype="int64"),
        TARGET: pred.astype("float64")
    })
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()