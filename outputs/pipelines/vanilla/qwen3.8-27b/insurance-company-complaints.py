import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/test.parquet"
TARGET = "Recovery"

CAT_COLS = [
    "Company",
    "Coverage",
    "SubCoverage",
    "Reason",
    "SubReason",
    "Disposition",
    "Conclusion",
    "Status",
]

DROP_COLS = ["File No.", "Opened", "Closed"]
MISSING = "__MISSING__"
UNKNOWN = "__UNKNOWN__"


def write_predictions(pred: np.ndarray) -> None:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, None)
    pd.DataFrame({
        "id": np.arange(len(pred), dtype=int),
        "Recovery": pred
    }).to_csv("predictions.csv", index=False)


def clean_string_series(s: pd.Series) -> pd.Series:
    s = s.fillna(MISSING).astype(str).str.strip()
    s = s.replace({
        "": MISSING,
        "nan": MISSING,
        "NaN": MISSING,
        "None": MISSING,
        "NaT": MISSING,
        "null": MISSING,
        "NULL": MISSING,
    })
    return s


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Opened", "Closed"]:
        if col not in df.columns:
            continue

        raw = df[col]
        if raw.dtype == object:
            raw = raw.astype(str).str.strip()
            raw = raw.replace({
                "": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "NaT": np.nan,
                "null": np.nan,
                "NULL": np.nan,
            })

        dt = pd.to_datetime(raw, errors="coerce")
        if dt.isna().all() and df[col].notna().any():
            dt2 = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
            dt = dt2.fillna(dt)

        df[f"{col}_dt"] = dt

    if "Opened_dt" in df.columns and "Closed_dt" in df.columns:
        df["days_to_close"] = (df["Closed_dt"] - df["Opened_dt"]).dt.days

    for prefix, col in [("opened", "Opened_dt"), ("closed", "Closed_dt")]:
        if col not in df.columns:
            continue
        df[f"{prefix}_year"] = df[col].dt.year
        df[f"{prefix}_month"] = df[col].dt.month
        df[f"{prefix}_day"] = df[col].dt.day
        df[f"{prefix}_dayofweek"] = df[col].dt.dayofweek
        df[f"{prefix}_quarter"] = df[col].dt.quarter

    dt_cols = df.select_dtypes(include=["datetime"]).columns
    if len(dt_cols):
        df = df.drop(columns=dt_cols)

    return df


def drop_unwanted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
    train_df = parse_dates(train_df)
    val_df = parse_dates(val_df)
    test_df = parse_dates(test_df)

    train_df = drop_unwanted(train_df)
    val_df = drop_unwanted(val_df)
    test_df = drop_unwanted(test_df)

    for df in (train_df, val_df, test_df):
        dt_cols = df.select_dtypes(include=["datetime"]).columns
        if len(dt_cols):
            df.drop(columns=dt_cols, inplace=True)

    cat_cols = [c for c in CAT_COLS if c in train_df.columns]

    for col in cat_cols:
        if col not in val_df.columns:
            val_df[col] = MISSING
        if col not in test_df.columns:
            test_df[col] = MISSING

        train_df[col] = clean_string_series(train_df[col])
        val_df[col] = clean_string_series(val_df[col])
        test_df[col] = clean_string_series(test_df[col])

        cats = sorted(set(train_df[col].unique()) | {MISSING, UNKNOWN})

        val_df[col] = val_df[col].where(val_df[col].isin(cats), UNKNOWN)
        test_df[col] = test_df[col].where(test_df[col].isin(cats), UNKNOWN)

        train_df[col] = pd.Categorical(train_df[col], categories=cats)
        val_df[col] = pd.Categorical(val_df[col], categories=cats)
        test_df[col] = pd.Categorical(test_df[col], categories=cats)

    for df in (train_df, val_df, test_df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            df[num_cols] = df[num_cols].astype(float).replace([np.inf, -np.inf], np.nan)

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols + cat_cols
        if c in val_df.columns and c in test_df.columns
    ]

    feature_cols = [
        c for c in feature_cols
        if c in cat_cols or train_df[c].notna().any()
    ]

    return train_df, val_df, test_df, feature_cols, cat_cols


def make_encoded_frame(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes.astype(int)
        elif out[col].dtype == object:
            codes, _ = pd.factorize(out[col])
            out[col] = codes.astype(int)
        elif out[col].dtype == bool:
            out[col] = out[col].astype(float)
        elif np.issubdtype(out[col].dtype, np.number):
            out[col] = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            out[col] = clean_string_series(out[col])
            codes, _ = pd.factorize(out[col])
            out[col] = codes.astype(int)

    return out


def make_catboost_frame(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype) or out[col].dtype == object:
            out[col] = clean_string_series(out[col])
        elif out[col].dtype == bool:
            out[col] = out[col].astype(float)
        elif np.issubdtype(out[col].dtype, np.number):
            out[col] = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            out[col] = clean_string_series(out[col])

    return out


def clean_predictions(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pred, 0.0, None)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0:
        return np.inf
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    if len(test_df) == 0:
        write_predictions(np.array([], dtype=float))
        return

    if TARGET not in train_df.columns or len(train_df) == 0:
        write_predictions(np.zeros(len(test_df), dtype=float))
        return

    y_all = train_df[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    X_all = train_df.drop(columns=[TARGET])

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42
    )

    X_train, X_val, X_test, feature_cols, cat_cols = prepare_features(
        X_train_raw, X_val_raw, test_df
    )

    feature_cols = [c for c in feature_cols if c in X_train.columns]

    if not feature_cols:
        write_predictions(np.zeros(len(test_df), dtype=float))
        return

    X_train = X_train[feature_cols]
    X_val = X_val[feature_cols]
    X_test = X_test[feature_cols]

    valid_train = y_train.notna()
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train].astype(float)

    valid_val = y_val.notna()
    X_val = X_val.loc[valid_val]
    y_val = y_val.loc[valid_val].astype(float)

    if len(y_val) == 0:
        X_val = X_train.copy()
        y_val = y_train.copy()

    if len(y_train) == 0:
        write_predictions(np.zeros(len(test_df), dtype=float))
        return

    models = []
    test_frames = []
    val_preds = []

    try:
        X_train_lgb = make_encoded_frame(X_train, cat_cols)
        X_val_lgb = make_encoded_frame(X_val, cat_cols)
        X_test_lgb = make_encoded_frame(X_test, cat_cols)

        cat_feature = [c for c in cat_cols if c in X_train_lgb.columns]
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_train_lgb,
            y_train,
            categorical_feature=cat_feature if cat_feature else None
        )

        models.append(lgb_model)
        test_frames.append(X_test_lgb)
        val_preds.append(clean_predictions(lgb_model.predict(X_val_lgb)))
    except Exception:
        pass

    try:
        X_train_xgb = make_encoded_frame(X_train, cat_cols)
        X_val_xgb = make_encoded_frame(X_val, cat_cols)
        X_test_xgb = make_encoded_frame(X_test, cat_cols)

        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
        xgb_model.fit(X_train_xgb, y_train)

        models.append(xgb_model)
        test_frames.append(X_test_xgb)
        val_preds.append(clean_predictions(xgb_model.predict(X_val_xgb)))
    except Exception:
        pass

    try:
        X_train_cb = make_catboost_frame(X_train, cat_cols)
        X_val_cb = make_catboost_frame(X_val, cat_cols)
        X_test_cb = make_catboost_frame(X_test, cat_cols)

        cb_params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
        }

        cb_cat_features = [c for c in cat_cols if c in X_train_cb.columns]
        if cb_cat_features:
            cb_params["cat_features"] = cb_cat_features

        cb_model = cb.CatBoostRegressor(**cb_params)
        cb_model.fit(X_train_cb, y_train)

        models.append(cb_model)
        test_frames.append(X_test_cb)
        val_preds.append(clean_predictions(cb_model.predict(X_val_cb)))
    except Exception:
        pass

    if not models:
        fallback = float(np.mean(y_train)) if len(y_train) else 0.0
        write_predictions(np.full(len(X_test), fallback, dtype=float))
        return

    scores = [safe_rmse(y_val, pred) for pred in val_preds]
    weights = []
    for score in scores:
        if np.isfinite(score):
            if score <= 1e-12:
                weights.append(1e6)
            else:
                weights.append(1.0 / score)
        else:
            weights.append(0.0)

    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0 / len(weights)] * len(weights)
        total_weight = 1.0

    test_pred = np.zeros(len(X_test), dtype=float)
    for model, test_frame, weight in zip(models, test_frames, weights):
        test_pred += clean_predictions(model.predict(test_frame)) * weight

    test_pred = test_pred / total_weight
    write_predictions(test_pred)


if __name__ == "__main__":
    main()