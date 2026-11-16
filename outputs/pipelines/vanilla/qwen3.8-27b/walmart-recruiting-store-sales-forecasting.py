import numpy as np
import pandas as pd
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/walmart-recruiting-store-sales-forecasting/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/walmart-recruiting-store-sales-forecasting/data/test.parquet"
TARGET = "Weekly_Sales"
RANDOM_STATE = 42


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    return train, test


def parse_dates(df):
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_store_dept(df):
    df = df.copy()
    if "Store_Dept" not in df.columns:
        if "Store" in df.columns and "Dept" in df.columns:
            store = pd.to_numeric(df["Store"], errors="coerce").fillna(-1).astype(int).astype(str)
            dept = pd.to_numeric(df["Dept"], errors="coerce").fillna(-1).astype(int).astype(str)
            df["Store_Dept"] = store + "_" + dept
        else:
            df["Store_Dept"] = "unknown"
    return df


def add_calendar_features(df):
    df = df.copy()

    if "Date" in df.columns:
        d = df["Date"]

        df["year"] = d.dt.year.astype("float64")
        df["month"] = d.dt.month.astype("float64")
        df["day"] = d.dt.day.astype("float64")
        df["dayofweek"] = d.dt.dayofweek.astype("float64")
        df["quarter"] = d.dt.quarter.astype("float64")

        try:
            df["week"] = d.dt.isocalendar().week.astype("float64")
        except Exception:
            df["week"] = d.dt.week.astype("float64")

        df["dayofyear"] = d.dt.dayofyear.astype("float64")

        df["is_month_start"] = d.dt.is_month_start.astype("float64")
        df["is_month_end"] = d.dt.is_month_end.astype("float64")
        df["is_quarter_start"] = d.dt.is_quarter_start.astype("float64")
        df["is_quarter_end"] = d.dt.is_quarter_end.astype("float64")
        df["is_year_start"] = d.dt.is_year_start.astype("float64")
        df["is_year_end"] = d.dt.is_year_end.astype("float64")

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)

    if "IsHoliday" in df.columns:
        holiday = df["IsHoliday"]
        if holiday.dtype == object:
            df["IsHoliday"] = holiday.astype(str).str.lower().isin(["true", "1", "t", "yes"]).astype("float64")
        else:
            df["IsHoliday"] = holiday.fillna(False).astype("float64")
    else:
        df["IsHoliday"] = 0.0

    return df


def encode_categoricals(df, fit_df, categorical_cols):
    df = df.copy()

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if col in fit_df.columns:
            fit_values = fit_df[col].dropna().astype(str).unique()
        else:
            fit_values = np.array([], dtype=object)

        fill_value = "__missing__"
        categories = list(dict.fromkeys(list(fit_values) + [fill_value]))

        df[col] = df[col].astype(str)
        df[col] = df[col].replace({"": fill_value, "nan": fill_value, "None": fill_value})
        df[col] = df[col].fillna(fill_value)

        df[col] = pd.Categorical(df[col], categories=categories)
        df[col] = df[col].fillna(fill_value)

    return df


def add_aggregates(df, fit_df):
    df = df.copy()
    fit_df = fit_df.copy()

    fit_df = ensure_store_dept(fit_df)
    df = ensure_store_dept(df)

    if TARGET not in fit_df.columns:
        return df

    fit_df[TARGET] = fit_df[TARGET].replace([np.inf, -np.inf], np.nan)
    global_mean = float(fit_df[TARGET].mean())
    if not np.isfinite(global_mean):
        global_mean = 0.0

    for frame in (fit_df, df):
        for col in ["Store", "Dept"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype(int)

    fit_df["Store_Dept_key"] = fit_df["Store_Dept"].astype(str)
    df["Store_Dept_key"] = df["Store_Dept"].astype(str)

    store_dept_agg = (
        fit_df.groupby("Store_Dept_key")[TARGET]
        .agg(
            sd_mean="mean",
            sd_median="median",
            sd_std="std",
            sd_min="min",
            sd_max="max",
            sd_count="count",
            sd_sum="sum",
        )
        .reset_index()
    )

    if "Dept" in fit_df.columns:
        dept_agg = (
            fit_df.groupby("Dept")[TARGET]
            .agg(
                dept_mean="mean",
                dept_median="median",
                dept_std="std",
                dept_min="min",
                dept_max="max",
                dept_count="count",
            )
            .reset_index()
        )
    else:
        dept_agg = pd.DataFrame(
            columns=["Dept", "dept_mean", "dept_median", "dept_std", "dept_min", "dept_max", "dept_count"]
        )

    if "Store" in fit_df.columns:
        store_agg = (
            fit_df.groupby("Store")[TARGET]
            .agg(
                store_mean="mean",
                store_median="median",
                store_std="std",
                store_min="min",
                store_max="max",
                store_count="count",
            )
            .reset_index()
        )
    else:
        store_agg = pd.DataFrame(
            columns=["Store", "store_mean", "store_median", "store_std", "store_min", "store_max", "store_count"]
        )

    df = df.merge(store_dept_agg, on="Store_Dept_key", how="left")

    if not dept_agg.empty:
        df = df.merge(dept_agg, on="Dept", how="left")

    if not store_agg.empty:
        df = df.merge(store_agg, on="Store", how="left")

    fill_map = {
        "sd_mean": global_mean,
        "sd_median": global_mean,
        "sd_std": 0.0,
        "sd_min": global_mean,
        "sd_max": global_mean,
        "sd_count": 0.0,
        "sd_sum": 0.0,
        "dept_mean": global_mean,
        "dept_median": global_mean,
        "dept_std": 0.0,
        "dept_min": global_mean,
        "dept_max": global_mean,
        "dept_count": 0.0,
        "store_mean": global_mean,
        "store_median": global_mean,
        "store_std": 0.0,
        "store_min": global_mean,
        "store_max": global_mean,
        "store_count": 0.0,
    }

    for col, fill_value in fill_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    df["sd_range"] = df["sd_max"] - df["sd_min"]
    df["sd_cv"] = df["sd_std"] / (df["sd_mean"].abs() + 1e-6)

    if "dept_max" in df.columns and "dept_min" in df.columns:
        df["dept_range"] = df["dept_max"] - df["dept_min"]

    if "store_max" in df.columns and "store_min" in df.columns:
        df["store_range"] = df["store_max"] - df["store_min"]

    df = df.drop(columns=["Store_Dept_key"], errors="ignore")

    return df


def prepare_frame(df, fit_df, categorical_cols):
    df = parse_dates(df)
    df = ensure_store_dept(df)
    df = add_calendar_features(df)
    df = encode_categoricals(df, fit_df, categorical_cols)
    df = add_aggregates(df, fit_df)
    return df


def get_feature_columns(train_df, test_df):
    ignore = {TARGET, "Date", "id", "Id", "ID", "index", "Index"}
    common = [c for c in train_df.columns if c in test_df.columns and c not in ignore]

    cols = []
    for c in common:
        if c.endswith("_code"):
            continue

        if train_df[c].dtype == object:
            continue

        if pd.api.types.is_datetime64_any_dtype(train_df[c]):
            continue

        cols.append(c)

    return cols


def clean_matrix(X):
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    return X


def make_sample_weights(df):
    if "Store_Dept" not in df.columns or TARGET not in df.columns:
        return pd.Series(1.0, index=df.index)

    tmp = df.copy()
    tmp[TARGET] = tmp[TARGET].replace([np.inf, -np.inf], np.nan)

    weights = tmp.groupby(tmp["Store_Dept"].astype(str))[TARGET].transform("sum")
    weights = weights.fillna(0.0).clip(lower=0.0)

    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        return pd.Series(1.0, index=df.index)

    return weights / mean_weight


def categorical_feature_names(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def baseline_predictions(test_df, train_df):
    global_mean = 0.0

    if TARGET in train_df.columns:
        target = train_df[TARGET].replace([np.inf, -np.inf], np.nan)
        if target.notna().any():
            global_mean = float(target.mean())
            if not np.isfinite(global_mean):
                global_mean = 0.0

    if "Store_Dept" in test_df.columns and "Store_Dept" in train_df.columns and TARGET in train_df.columns:
        target = train_df[TARGET].replace([np.inf, -np.inf], np.nan)
        agg = train_df.assign(_target=target).groupby("Store_Dept")["_target"].mean()
        preds = test_df["Store_Dept"].astype(str).map(agg)
    else:
        preds = pd.Series(global_mean, index=test_df.index)

    preds = pd.Series(preds, index=test_df.index).astype(float)
    preds = preds.replace([np.inf, -np.inf], np.nan).fillna(global_mean)

    if TARGET in train_df.columns:
        target = train_df[TARGET].replace([np.inf, -np.inf], np.nan)
        if target.notna().any() and float(target.min()) >= 0:
            preds = preds.clip(lower=0.0)

    return preds.values


def main():
    train_raw, test_raw = load_data()

    train_raw = parse_dates(train_raw)
    test_raw = parse_dates(test_raw)

    train_raw = ensure_store_dept(train_raw)
    test_raw = ensure_store_dept(test_raw)

    ignore = {TARGET, "Date", "id", "Id", "ID", "index", "Index"}
    categorical_cols = []

    for col in train_raw.columns:
        if col in ignore:
            continue

        dtype = train_raw[col].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
            or str(dtype) in ("string", "string[python]", "string[pyarrow]")
        ):
            categorical_cols.append(col)

    if "Store_Dept" not in categorical_cols:
        categorical_cols.append("Store_Dept")

    sort_cols = [c for c in ["Date", "Store", "Dept"] if c in train_raw.columns]
    if sort_cols:
        train_sorted = train_raw.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        train_sorted = train_raw.reset_index(drop=True)

    split_idx = int(len(train_sorted) * 0.8)
    train_part = train_sorted.iloc[:split_idx].copy()
    val_part = train_sorted.iloc[split_idx:].copy()

    train_part = prepare_frame(train_part, train_part, categorical_cols)
    val_part = prepare_frame(val_part, train_part, categorical_cols)
    test = prepare_frame(test_raw, train_part, categorical_cols)

    feature_cols = get_feature_columns(train_part, val_part)
    if not feature_cols:
        feature_cols = get_feature_columns(train_part, test)

    if not feature_cols or len(train_part) == 0:
        preds = baseline_predictions(test, train_part)
        submission = pd.DataFrame(
            {
                "id": np.arange(len(preds), dtype=int),
                "Weekly_Sales": preds,
            }
        )
        submission.to_csv("predictions.csv", index=False)
        return

    X_train = clean_matrix(train_part[feature_cols].copy())
    y_train = train_part[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    valid_train = y_train.notna()

    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    w_train = make_sample_weights(train_part).loc[valid_train].values

    X_val = clean_matrix(val_part[feature_cols].copy())
    y_val = val_part[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    valid_val = y_val.notna()

    X_val = X_val.loc[valid_val]
    y_val = y_val.loc[valid_val]
    w_val = make_sample_weights(val_part).loc[valid_val].values

    best_iteration = 5000

    try:
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=5000,
            learning_rate=0.03,
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
            verbosity=-1,
        )

        cat_names = categorical_feature_names(X_train)

        if len(X_val) > 0:
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[w_val],
                eval_metric="l1",
                categorical_feature=cat_names if cat_names else "auto",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = int(model.best_iteration_) if getattr(model, "best_iteration_", 0) > 0 else 5000
        else:
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                categorical_feature=cat_names if cat_names else "auto",
            )
    except Exception:
        best_iteration = 5000

    final_train = prepare_frame(train_raw, train_raw, categorical_cols)
    final_test = prepare_frame(test_raw, train_raw, categorical_cols)

    final_feature_cols = get_feature_columns(final_train, final_test)
    if not final_feature_cols:
        final_feature_cols = feature_cols

    y_final_train = pd.Series(dtype=float)

    try:
        X_final_train = clean_matrix(final_train[final_feature_cols].copy())
        y_final_train = final_train[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
        valid_final = y_final_train.notna()

        X_final_train = X_final_train.loc[valid_final]
        y_final_train = y_final_train.loc[valid_final]
        w_final_train = make_sample_weights(final_train).loc[valid_final].values

        X_final_test = clean_matrix(final_test[final_feature_cols].copy())

        if len(X_final_train) == 0:
            raise ValueError("No valid training rows.")

        final_model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=max(1, best_iteration),
            learning_rate=0.03,
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
            verbosity=-1,
        )

        cat_names_final = categorical_feature_names(X_final_train)
        final_model.fit(
            X_final_train,
            y_final_train,
            sample_weight=w_final_train,
            categorical_feature=cat_names_final if cat_names_final else "auto",
        )

        preds = final_model.predict(X_final_test)
    except Exception:
        preds = baseline_predictions(final_test, final_train)

    preds = np.asarray(preds, dtype=float).reshape(-1)

    if len(preds) != len(final_test):
        if len(preds) < len(final_test):
            fill_value = float(np.nanmean(preds)) if len(preds) > 0 and np.isfinite(np.nanmean(preds)) else 0.0
            preds = np.concatenate([preds, np.full(len(final_test) - len(preds), fill_value, dtype=float)])
        else:
            preds = preds[: len(final_test)]

    global_mean = 0.0
    if len(y_final_train) > 0:
        candidate = float(y_final_train.mean())
        if np.isfinite(candidate):
            global_mean = candidate

    preds = np.where(np.isfinite(preds), preds, global_mean)

    if len(y_final_train) > 0 and float(y_final_train.min()) >= 0:
        preds = np.clip(preds, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(preds), dtype=int),
            "Weekly_Sales": preds,
        }
    )

    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()