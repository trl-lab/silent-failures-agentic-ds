import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TARGET = "target"
ID_OUT = "id"
TARGET_OUT = "target"

BASE_TRAIN = "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers/data/train.parquet"
BASE_TEST = "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers/data/test.parquet"
DATA_DIR = os.path.dirname(BASE_TRAIN)
ROOT_DIR = os.path.dirname(DATA_DIR)

EXTRA_DIRS = []


def add_dir(d):
    if d and os.path.isdir(d) and d not in EXTRA_DIRS:
        EXTRA_DIRS.append(d)


def search_dirs():
    dirs = []
    candidates = [
        os.getcwd(),
        DATA_DIR,
        ROOT_DIR,
        os.path.dirname(ROOT_DIR),
        "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers",
        "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers/data",
    ]
    for d in EXTRA_DIRS + candidates:
        if d and os.path.isdir(d) and d not in dirs:
            dirs.append(d)
    return dirs


def find_file(stem, exts=(".parquet", ".csv")):
    dirs = search_dirs()
    for d in dirs:
        for ext in exts:
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                return p
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                fl = f.lower()
                if fl.startswith(stem.lower()) and any(fl.endswith(ext) for ext in exts):
                    return os.path.join(root, f)
    return None


def load_df(stem):
    p = find_file(stem)
    if p is None:
        return None
    try:
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception:
        return None


def to_datetime(series):
    if series is None:
        return series
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def find_col(df, candidates):
    if df is None or len(df.columns) == 0:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for cand in candidates:
        for c in df.columns:
            if cand in str(c).lower():
                return c
    return None


def find_datetime_col(df):
    if df is None:
        return None
    return find_col(
        df,
        [
            "datetime",
            "time",
            "timestamp",
            "date_time",
            "date",
            "dt",
            "hour",
            "hourly",
            "period",
        ],
    )


def numeric_cols(df, exclude=()):
    exclude = set(exclude)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def categorical_cols(df, exclude=()):
    exclude = set(exclude)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
            cols.append(c)
    return cols


def normalize_key_series(s):
    if pd.api.types.is_datetime64_any_dtype(s) or "date" in str(s.name).lower() or "time" in str(s.name).lower():
        return to_datetime(s)
    return s.astype("string")


def aggregate_by_keys(df, keys):
    if df is None or len(df) == 0:
        return df
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return df
    df = df.dropna(subset=keys)
    if len(df) == 0:
        return df
    num = numeric_cols(df, exclude=keys)
    cat = [c for c in df.columns if c not in keys and c not in num]
    if not num and not cat:
        return df.drop_duplicates(subset=keys, keep="first")
    agg = {c: "mean" for c in num}
    for c in cat:
        agg[c] = "first"
    try:
        out = df.groupby(keys, as_index=False, dropna=False).agg(agg)
    except Exception:
        out = df.drop_duplicates(subset=keys, keep="first")
    return out


def safe_merge(df, ref, left_keys, right_keys, prefix=""):
    if ref is None or len(ref) == 0:
        return df
    if not all(k in df.columns for k in left_keys):
        return df
    if not all(k in ref.columns for k in right_keys):
        return df

    df = df.copy()
    ref = ref.copy()

    for lk, rk in zip(left_keys, right_keys):
        if lk in df.columns:
            df[lk] = normalize_key_series(df[lk])
        if rk in ref.columns:
            ref[rk] = normalize_key_series(ref[rk])

    rename = {}
    for lk, rk in zip(left_keys, right_keys):
        if lk != rk:
            rename[rk] = lk
    if prefix:
        for c in ref.columns:
            if c not in right_keys:
                new_c = f"{prefix}{c}"
                if new_c != c:
                    rename[c] = new_c
    if rename:
        ref = ref.rename(columns=rename)

    ref = ref.dropna(subset=left_keys)
    if len(ref) == 0:
        return df

    for k in left_keys:
        if k in ref.columns and k in df.columns:
            ref = ref[ref[k].isin(df[k])]
    if len(ref) == 0:
        return df

    if ref.duplicated(subset=left_keys).any():
        ref = aggregate_by_keys(ref, left_keys)

    df["_orig_idx"] = np.arange(len(df))
    df = df.merge(ref, on=left_keys, how="left")
    df = df.drop_duplicates(subset="_orig_idx", keep="first").drop(columns="_orig_idx")
    return df


def prepare_weather(weather_df, mapping_df):
    if weather_df is None or len(weather_df) == 0:
        return None
    dt_col = find_datetime_col(weather_df)
    if dt_col is None:
        return None

    weather_df = weather_df.copy()
    weather_df[dt_col] = to_datetime(weather_df[dt_col])

    station_col = find_col(
        weather_df,
        [
            "station_id",
            "weather_station_id",
            "station",
            "station_name",
            "weather_station",
            "station_code",
            "id",
        ],
    )
    county_col = find_col(weather_df, ["county", "county_name", "county_code"])

    if station_col is not None and county_col is None and mapping_df is not None and len(mapping_df) > 0:
        map_station = find_col(
            mapping_df,
            [
                "station_id",
                "weather_station_id",
                "station",
                "station_name",
                "weather_station",
                "station_code",
                "id",
            ],
        )
        map_county = find_col(mapping_df, ["county", "county_name", "county_code"])
        if map_station is not None and map_county is not None:
            m = mapping_df[[map_station, map_county]].dropna()
            m = m.drop_duplicates(subset=map_station, keep="first")
            weather_df = weather_df.merge(m, left_on=station_col, right_on=map_station, how="left")
            county_col = map_county
            if map_station != station_col:
                weather_df = weather_df.drop(columns=[map_station], errors="ignore")

    if county_col is None:
        return None

    weather_df = weather_df.dropna(subset=[dt_col, county_col])
    if len(weather_df) == 0:
        return None

    agg = aggregate_by_keys(weather_df, [dt_col, county_col])
    if agg is None or len(agg) == 0:
        return None

    rename = {}
    for c in agg.columns:
        if c not in (dt_col, county_col):
            rename[c] = f"weather_{c}"
    if rename:
        agg = agg.rename(columns=rename)
    return agg, dt_col, county_col


def add_time_features(df, dt_col):
    if dt_col is None or dt_col not in df.columns:
        return df
    df = df.copy()
    dt = to_datetime(df[dt_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["year"] = dt.dt.year
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"].fillna(12) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"].fillna(12) / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"].fillna(0) / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"].fillna(0) / 7)
    df = df.drop(columns=[dt_col], errors="ignore")
    return df


def clean_numeric_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def make_features(df, dt_col, high_card_cols=None, freq_maps=None):
    df = df.copy()
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    drop_cols = [c for c in ["row_id", "data_block_id", "id", "index"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = add_time_features(df, dt_col)

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    for c in df.columns:
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")

    if high_card_cols and freq_maps:
        for c in high_card_cols:
            if c in df.columns:
                freq = freq_maps.get(c, {})
                df[c] = df[c].map(freq).fillna(0).astype(float)

    num_cols = numeric_cols(df, exclude=[TARGET])
    df = clean_numeric_columns(df, num_cols)

    for c in df.columns:
        if c != TARGET and df[c].isna().all():
            df = df.drop(columns=[c], errors="ignore")

    feature_cols = [c for c in df.columns if c != TARGET]
    cat_cols = [c for c in feature_cols if isinstance(df[c].dtype, pd.CategoricalDtype)]
    return df, feature_cols, cat_cols


def align_categoricals(train, test, feature_cols):
    train = train.copy()
    test = test.copy()
    for c in feature_cols:
        if c not in train.columns:
            continue
        if isinstance(train[c].dtype, pd.CategoricalDtype):
            cats = list(train[c].cat.categories)
            if "unknown" not in cats:
                cats.append("unknown")
            train[c] = pd.Categorical(train[c], categories=cats)
            if c in test.columns:
                test[c] = test[c].astype("string").fillna("unknown")
                test[c] = pd.Categorical(test[c], categories=cats)
            else:
                test[c] = pd.Categorical(["unknown"] * len(test), categories=cats)
    return train, test


def join_prices(df, dt_col, prices, prefix):
    if prices is None or len(prices) == 0 or dt_col is None or dt_col not in df.columns:
        return df
    edt = find_datetime_col(prices)
    if edt is None:
        return df

    p = prices.copy()
    p[edt] = to_datetime(p[edt])

    keys = [dt_col]
    rkeys = [edt]
    if "product_type" in df.columns and "product_type" in p.columns:
        keys.append("product_type")
        rkeys.append("product_type")

    merged = safe_merge(df, p, keys, rkeys, prefix=prefix)
    new_cols = [c for c in merged.columns if c.startswith(prefix) and c not in df.columns]
    if new_cols and not all(merged[c].isna().all() for c in new_cols):
        return merged

    p2 = p.copy()
    p2["price_date"] = p2[edt].dt.date.astype("string")
    df2 = df.copy()
    df2["price_date"] = to_datetime(df2[dt_col]).dt.date.astype("string")
    merged2 = safe_merge(df2, p2, ["price_date"], ["price_date"], prefix=prefix)
    merged2 = merged2.drop(columns=["price_date"], errors="ignore")
    return merged2


def join_common(df, dt_col, client, electricity_prices, forecast_weather, historical_weather, gas_prices, station_map, use_forecast=False):
    if "prediction_unit_id" in df.columns and client is not None and len(client) > 0:
        client_key = None
        for k in ["prediction_unit_id", "client_id", "id"]:
            if k in df.columns and k in client.columns:
                client_key = k
                break
        if client_key is not None:
            df = safe_merge(df, client, [client_key], [client_key], prefix="client_")

    if dt_col is not None and "county" in df.columns:
        first = forecast_weather if use_forecast else historical_weather
        second = historical_weather if use_forecast else forecast_weather
        weather = prepare_weather(first, station_map)
        if weather is None:
            weather = prepare_weather(second, station_map)
        if weather is not None:
            wdf, wdt, wcounty = weather
            df = safe_merge(df, wdf, [dt_col, "county"], [wdt, wcounty], prefix="weather_")

    df = join_prices(df, dt_col, electricity_prices, "elec_")

    if dt_col is not None and gas_prices is not None and len(gas_prices) > 0:
        gdt = find_datetime_col(gas_prices)
        if gdt is not None:
            gp = gas_prices.copy()
            gp[gdt] = to_datetime(gp[gdt])
            gp["gas_date"] = gp[gdt].dt.date.astype("string")
            df = df.copy()
            df["gas_date"] = to_datetime(df[dt_col]).dt.date.astype("string")
            df = safe_merge(df, gp, ["gas_date"], ["gas_date"], prefix="gas_")
            df = df.drop(columns=["gas_date"], errors="ignore")

    return df


def save_predictions(pred, n_rows=None):
    if n_rows is None:
        n_rows = len(pred)
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    out = pd.DataFrame({ID_OUT: np.arange(n_rows, dtype=int), TARGET_OUT: pred[:n_rows]})
    out.to_csv("predictions.csv", index=False)


def main():
    train = load_df("train")
    test = load_df("test")

    if train is not None:
        add_dir(os.path.dirname(find_file("train")))
    if test is not None:
        add_dir(os.path.dirname(find_file("test")))

    if train is None:
        n = len(test) if test is not None else 0
        save_predictions(np.zeros(n), n)
        return

    if test is None:
        n = len(train)
        save_predictions(np.zeros(n), n)
        return

    train = train.dropna(subset=[TARGET])
    if len(train) == 0:
        save_predictions(np.zeros(len(test)), len(test))
        return

    train_dt_col = find_datetime_col(train)
    test_dt_col = find_datetime_col(test)
    if train_dt_col is not None:
        train[train_dt_col] = to_datetime(train[train_dt_col])
    if test_dt_col is not None:
        test[test_dt_col] = to_datetime(test[test_dt_col])

    client = load_df("client")
    electricity_prices = load_df("electricity_prices")
    forecast_weather = load_df("forecast_weather")
    historical_weather = load_df("historical_weather")
    gas_prices = load_df("gas_prices")
    station_map = load_df("weather_station_to_county_mapping")

    train = join_common(
        train,
        train_dt_col,
        client,
        electricity_prices,
        forecast_weather,
        historical_weather,
        gas_prices,
        station_map,
        use_forecast=False,
    )
    test = join_common(
        test,
        test_dt_col,
        client,
        electricity_prices,
        forecast_weather,
        historical_weather,
        gas_prices,
        station_map,
        use_forecast=True,
    )

    exclude = [TARGET, "row_id", "data_block_id", "id", "index"]
    train_cats_raw = categorical_cols(train, exclude=exclude)
    high_card_cols = [c for c in train_cats_raw if train[c].nunique(dropna=True) > 1000]
    freq_maps = {c: train[c].value_counts(dropna=True) for c in high_card_cols}

    train, train_features, train_cats = make_features(train, train_dt_col, high_card_cols, freq_maps)
    test, test_features, test_cats = make_features(test, test_dt_col, high_card_cols, freq_maps)

    common_features = [c for c in train_features if c in test_features]
    if not common_features:
        save_predictions(np.zeros(len(test)), len(test))
        return

    train = train[[c for c in [TARGET] + common_features if c in train.columns]]
    test = test[[c for c in common_features if c in test.columns]]
    train, test = align_categoricals(train, test, common_features)

    train = train.dropna(subset=[TARGET])
    if len(train) == 0:
        save_predictions(np.zeros(len(test)), len(test))
        return

    use_val = len(train) > 100
    if use_val:
        if train_dt_col is not None and train_dt_col in train.columns:
            order = train[train_dt_col].rank(method="first")
            val_size = max(1, int(len(train) * 0.2))
            val_idx = order.nlargest(val_size).index
            tr_idx = order.nsmallest(len(train) - val_size).index
        else:
            tr_idx, val_idx = train_test_split(train.index, test_size=0.2, random_state=42)
    else:
        tr_idx = train.index
        val_idx = train.index

    X_tr = train.loc[tr_idx, common_features]
    y_tr = train.loc[tr_idx, TARGET]
    X_val = train.loc[val_idx, common_features]
    y_val = train.loc[val_idx, TARGET]

    lgb_cats = [c for c in train_cats if c in common_features and c not in high_card_cols]

    params = dict(
        objective="regression_l1",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    best_iter = None
    try:
        if use_val and len(val_idx) > 0 and len(tr_idx) > 0:
            model = LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="l1",
                categorical_feature=lgb_cats,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            best_iter = getattr(model, "best_iteration_", None)
        else:
            model = LGBMRegressor(**{**params, "n_estimators": 1000})
            model.fit(X_tr, y_tr, categorical_feature=lgb_cats)
    except Exception:
        best_iter = None

    if best_iter is not None and best_iter > 0:
        final_params = {**params, "n_estimators": int(best_iter)}
    else:
        final_params = {**params, "n_estimators": 1000}

    try:
        final_model = LGBMRegressor(**final_params)
        final_model.fit(train[common_features], train[TARGET], categorical_feature=lgb_cats)
        pred = final_model.predict(test[common_features])
    except Exception:
        mean_target = train[TARGET].mean()
        if pd.isna(mean_target):
            mean_target = 0.0
        pred = np.full(len(test), float(mean_target), dtype=float)

    save_predictions(pred, len(test))


if __name__ == "__main__":
    main()