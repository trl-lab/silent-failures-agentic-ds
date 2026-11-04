import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT = "./input"
WORK = "./working"
os.makedirs(WORK, exist_ok=True)


def read_table(name: str) -> pd.DataFrame:
    """Load a table from parquet or CSV, checking common layouts."""
    candidates = [
        os.path.join(INPUT, f"{name}.parquet"),
        os.path.join(INPUT, f"{name}.csv"),
        os.path.join(INPUT, name, f"{name}.parquet"),
        os.path.join(INPUT, name, f"{name}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find table: {name}")


def clean_datetime(s: pd.Series) -> pd.Series:
    """Parse datetime and remove timezone info to make joins stable."""
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def add_time_features(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    dt = clean_datetime(df[dt_col])
    df[dt_col] = dt
    df["date_key"] = dt.dt.strftime("%Y-%m-%d")
    df["hour"] = dt.dt.hour.astype("int16")
    df["minute"] = dt.dt.minute.astype("int16")
    df["day"] = dt.dt.day.astype("int16")
    df["month"] = dt.dt.month.astype("int16")
    df["year"] = dt.dt.year.astype("int16")
    df["dayofweek"] = dt.dt.dayofweek.astype("int16")
    df["dayofyear"] = dt.dt.dayofyear.astype("int16")
    df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    df["is_night"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 22)).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25).astype("float32")
    return df


def make_daily_price_features(
    prices: pd.DataFrame, price_col: str, prefix: str
) -> pd.DataFrame:
    """Create daily price features using the latest available origin date for each forecast date."""
    if prices.empty or price_col not in prices.columns:
        return pd.DataFrame(columns=["date_key", f"{prefix}_price"])

    out = prices.copy()
    out["forecast_date"] = clean_datetime(out["forecast_date"]).dt.strftime("%Y-%m-%d")
    if "origin_date" in out.columns:
        out["origin_date"] = clean_datetime(out["origin_date"]).dt.strftime("%Y-%m-%d")
    else:
        out["origin_date"] = out["forecast_date"]

    out = out.sort_values(["forecast_date", "origin_date"], ascending=[True, True])
    out = out.drop_duplicates(subset=["forecast_date"], keep="last")
    out = out.rename(columns={price_col: f"{prefix}_price"})
    out["date_key"] = out["forecast_date"]
    return out[["date_key", f"{prefix}_price"]]


def assign_county_to_weather(
    weather: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Assign each weather row to the nearest county using latitude/longitude."""
    if weather.empty or mapping.empty:
        if "county" not in weather.columns:
            weather["county"] = np.nan
        return weather

    weather = weather.copy()
    if "latitude" not in weather.columns or "longitude" not in weather.columns:
        weather["county"] = np.nan
        return weather

    if "county" not in mapping.columns:
        mapping = mapping.rename(columns={"county_name": "county"})

    m = mapping.dropna(subset=["latitude", "longitude", "county"]).copy()
    if m.empty:
        weather["county"] = np.nan
        return weather

    m = m.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)
    m["latitude"] = m["latitude"].astype("float64")
    m["longitude"] = m["longitude"].astype("float64")

    w = weather.copy()
    w["latitude"] = pd.to_numeric(w["latitude"], errors="coerce")
    w["longitude"] = pd.to_numeric(w["longitude"], errors="coerce")

    # Simple nearest-county assignment in chunks to avoid huge memory use.
    county = np.full(len(w), np.nan, dtype=object)
    valid = w[["latitude", "longitude"]].notna().all(axis=1)
    if valid.any():
        # Use deterministic nearest neighbor via scipy if available, otherwise chunked numpy.
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(m[["latitude", "longitude"]].to_numpy(dtype="float64"))
            idx = tree.query(
                w.loc[valid, ["latitude", "longitude"]].to_numpy(dtype="float64"), k=1
            )[1]
            county[valid.to_numpy()] = m["county"].to_numpy()[idx]
        except Exception:
            # Fallback: chunked brute-force nearest neighbor.
            mxy = m[["latitude", "longitude"]].to_numpy(dtype="float64")
            mcounties = m["county"].to_numpy()
            chunk = 50_000
            valid_idx = np.where(valid.to_numpy())[0]
            for start in range(0, len(valid_idx), chunk):
                stop = min(start + chunk, len(valid_idx))
                pts = w.loc[valid_idx[start:stop], ["latitude", "longitude"]].to_numpy(
                    dtype="float64"
                )
                dlat = pts[:, 0:1] - mxy[:, 0:1].T
                dlon = pts[:, 1:2] - mxy[:, 1:2].T
                dist = dlat * dlat + dlon * dlon
                county[valid_idx[start:stop]] = mcounties[np.argmin(dist, axis=1)]

    w["county"] = county
    return w


def weather_numeric_cols(df: pd.DataFrame, exclude: set) -> list:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def add_weather_features(
    df: pd.DataFrame,
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join historical and forecast weather features by county and target datetime."""
    df = df.copy()

    # Historical weather: same-hour observed weather plus simple lags.
    if not historical.empty and "datetime" in historical.columns:
        hist = historical.copy()
        hist["datetime"] = clean_datetime(hist["datetime"])
        hist = assign_county_to_weather(hist, mapping)
        exclude = {"datetime", "latitude", "longitude", "data_block_id", "county"}
        num_cols = weather_numeric_cols(hist, exclude)
        if num_cols:
            hist_agg = hist.groupby(["county", "datetime"], as_index=False)[
                num_cols
            ].mean()
            hist_agg = hist_agg.rename(columns={c: f"hist_{c}" for c in num_cols})
            df = df.merge(hist_agg, on=["county", "datetime"], how="left")

            # Lagged historical features: previous hour and previous day.
            for lag, name in [(1, "lag1h"), (24, "lag24h")]:
                lagged = hist_agg.copy()
                lagged["datetime"] = lagged["datetime"] + pd.Timedelta(hours=lag)
                lagged = lagged.rename(
                    columns={f"hist_{c}": f"hist_{c}_{name}" for c in num_cols}
                )
                df = df.merge(lagged, on=["county", "datetime"], how="left")

    # Forecast weather: forecast_datetime equals target datetime.
    if not forecast.empty and "forecast_datetime" in forecast.columns:
        fc = forecast.copy()
        fc["forecast_datetime"] = clean_datetime(fc["forecast_datetime"])
        if "origin_datetime" in fc.columns:
            fc["origin_datetime"] = clean_datetime(fc["origin_datetime"])
        if "hours_ahead" in fc.columns:
            fc["hours_ahead"] = pd.to_numeric(fc["hours_ahead"], errors="coerce")
            # Use only forecasts made before the target hour.
            fc = fc[fc["hours_ahead"] > 0]

        fc = assign_county_to_weather(fc, mapping)

        # Derive wind speed/direction from u/v components if present.
        if (
            "10_metre_u_wind_component" in fc.columns
            and "10_metre_v_wind_component" in fc.columns
        ):
            u = pd.to_numeric(fc["10_metre_u_wind_component"], errors="coerce")
            v = pd.to_numeric(fc["10_metre_v_wind_component"], errors="coerce")
            fc["forecast_wind_speed"] = np.sqrt(u**2 + v**2)
            fc["forecast_wind_direction"] = (np.degrees(np.arctan2(u, v)) + 360) % 360

        exclude = {
            "forecast_datetime",
            "origin_datetime",
            "hours_ahead",
            "latitude",
            "longitude",
            "data_block_id",
            "county",
        }
        num_cols = weather_numeric_cols(fc, exclude)
        if num_cols:
            # Average available forecasts for each county and target datetime.
            fc_agg = fc.groupby(["county", "forecast_datetime"], as_index=False)[
                num_cols
            ].mean()
            fc_agg = fc_agg.rename(columns={c: f"fc_{c}" for c in num_cols})
            fc_agg = fc_agg.rename(columns={"forecast_datetime": "datetime"})
            df = df.merge(fc_agg, on=["county", "datetime"], how="left")

    return df


def add_client_features(df: pd.DataFrame, client: pd.DataFrame) -> pd.DataFrame:
    """Add client capacity/count features using safe aggregated joins."""
    df = df.copy()
    if client.empty:
        return df

    client = client.copy()
    if "date" in client.columns:
        client["date_key"] = clean_datetime(client["date"]).dt.strftime("%Y-%m-%d")

    id_like = {
        "product_type",
        "county",
        "is_business",
        "date",
        "date_key",
        "data_block_id",
        "prediction_unit_id",
    }
    num_cols = [
        c
        for c in client.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(client[c])
    ]
    if not num_cols:
        return df

    # 1) Direct prediction_unit_id join if available.
    if "prediction_unit_id" in client.columns and "prediction_unit_id" in df.columns:
        agg = client.groupby("prediction_unit_id", as_index=False)[num_cols].mean()
        agg = agg.rename(columns={c: f"client_{c}" for c in num_cols})
        df = df.merge(agg, on="prediction_unit_id", how="left")

    # 2) If client.data_block_id appears to be the unit id, join it to prediction_unit_id.
    if "data_block_id" in client.columns and "prediction_unit_id" in df.columns:
        sample = df["prediction_unit_id"].dropna().head(20_000)
        if not sample.empty and not client["data_block_id"].dropna().empty:
            overlap = sample.isin(client["data_block_id"].dropna().unique()).mean()
            if overlap > 0.5:
                agg = client.groupby("data_block_id", as_index=False)[num_cols].mean()
                agg = agg.rename(columns={"data_block_id": "prediction_unit_id"})
                agg = agg.rename(columns={c: f"client_db_{c}" for c in num_cols})
                df = df.merge(agg, on="prediction_unit_id", how="left")

    # 3) Fallback aggregate by product_type, county, and date.
    if "date_key" in client.columns and all(
        c in df.columns for c in ["product_type", "county", "date_key"]
    ):
        agg = client.groupby(["product_type", "county", "date_key"], as_index=False)[
            num_cols
        ].mean()
        agg["client_count"] = (
            agg.groupby(["product_type", "county", "date_key"]).size().values
        )
        agg = agg.rename(columns={c: f"client_agg_{c}" for c in num_cols})
        df = df.merge(agg, on=["product_type", "county", "date_key"], how="left")

    return df


def prepare_data():
    train = read_table("train")
    test = read_table("test")

    # Ensure target exists and is numeric.
    train["target"] = pd.to_numeric(train["target"], errors="coerce")
    train = train.dropna(subset=["target"]).reset_index(drop=True)

    # Add time features to both.
    train = add_time_features(train)
    test = add_time_features(test)

    # Load reference tables.
    client = read_table("client")
    elec = read_table("electricity_prices")
    gas = read_table("gas_prices")
    hist = read_table("historical_weather")
    fc = read_table("forecast_weather")
    mapping = read_table("weather_station_to_county_mapping")

    # Daily price features.
    elec_daily = make_daily_price_features(elec, "euros_per_mwh", "elec")
    gas_daily = make_daily_price_features(gas, "lowest_price_per_mwh", "gas_low")
    gas_daily2 = make_daily_price_features(gas, "highest_price_per_mwh", "gas_high")
    gas_daily = gas_daily.merge(gas_daily2, on="date_key", how="outer")

    train = train.merge(elec_daily, on="date_key", how="left")
    train = train.merge(gas_daily, on="date_key", how="left")
    test = test.merge(elec_daily, on="date_key", how="left")
    test = test.merge(gas_daily, on="date_key", how="left")

    # Client features.
    train = add_client_features(train, client)
    test = add_client_features(test, client)

    # Weather features.
    train = add_weather_features(train, hist, fc, mapping)
    test = add_weather_features(test, hist, fc, mapping)

    # Categorical columns: align categories between train and test.
    cat_cols = ["county", "product_type", "is_business", "is_consumption"]
    for c in cat_cols:
        if c in train.columns and c in test.columns:
            all_cats = (
                pd.concat([train[c], test[c]], ignore_index=True)
                .astype("category")
                .cat.categories
            )
            train[c] = train[c].astype("category").cat.set_categories(all_cats)
            test[c] = test[c].astype("category").cat.set_categories(all_cats)
        elif c in train.columns:
            train[c] = train[c].astype("category")
        elif c in test.columns:
            test[c] = test[c].astype("category")

    # Feature list: everything except id/time keys and target.
    exclude = {"target", "datetime", "date_key", "row_id", "id"}
    feature_cols = [
        c
        for c in train.columns
        if c not in exclude and c not in test.columns or c in test.columns
    ]
    feature_cols = [c for c in train.columns if c not in exclude and c in test.columns]

    # Drop features that are all NaN in train.
    feature_cols = [c for c in feature_cols if train[c].notna().any()]

    return train, test, feature_cols


def main():
    train, test, feature_cols = prepare_data()

    # Temporal 5-fold validation: split train into 6 time blocks.
    # Fold i trains on blocks 0..i and validates on block i+1.
    tvals = train["datetime"].astype("int64").to_numpy()
    cuts = np.quantile(tvals, np.linspace(0, 1, 7))
    block = np.clip(np.searchsorted(cuts, tvals, side="right") - 1, 0, 5)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
    }

    maes = []
    best_iters = []

    for i in range(5):
        tr_mask = block <= i
        va_mask = block == (i + 1)
        if tr_mask.sum() < 1000 or va_mask.sum() < 100:
            continue

        dtrain = lgb.Dataset(
            train.loc[tr_mask, feature_cols],
            label=train.loc[tr_mask, "target"],
            categorical_feature=[
                c for c in feature_cols if train[c].dtype.name == "category"
            ],
        )
        dvalid = lgb.Dataset(
            train.loc[va_mask, feature_cols],
            label=train.loc[va_mask, "target"],
            reference=dtrain,
            categorical_feature=[
                c for c in feature_cols if train[c].dtype.name == "category"
            ],
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        pred = model.predict(
            train.loc[va_mask, feature_cols], num_iteration=model.best_iteration
        )
        mae = float(np.mean(np.abs(train.loc[va_mask, "target"].to_numpy() - pred)))
        maes.append(mae)
        best_iters.append(model.best_iteration)
        print(f"Fold {i + 1} MAE: {mae:.6f}, best_iteration: {model.best_iteration}")

    if maes:
        cv_mae = float(np.mean(maes))
        print(f"CV MAE: {cv_mae:.6f}")
    else:
        cv_mae = float("nan")
        print("CV MAE: nan")

    # Final model on all training data.
    dtrain_all = lgb.Dataset(
        train[feature_cols],
        label=train["target"],
        categorical_feature=[
            c for c in feature_cols if train[c].dtype.name == "category"
        ],
    )
    final_rounds = int(np.median(best_iters)) if best_iters else 300
    final_model = lgb.train(
        params,
        dtrain_all,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(test[feature_cols], num_iteration=final_rounds)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "target": test_pred.astype(float),
        }
    )
    submission.to_csv(os.path.join(WORK, "submission.csv"), index=False)
    print(f"Saved submission to {os.path.join(WORK, 'submission.csv')}")


if __name__ == "__main__":
    main()
