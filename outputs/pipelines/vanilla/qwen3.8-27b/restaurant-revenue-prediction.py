import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/restaurant-revenue-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/restaurant-revenue-prediction/data/test.parquet"
TARGET = "revenue"
ID_COL = "Id"
RANDOM_STATE = 42
MISSING = "__MISSING__"


def parse_date(df):
    df = df.copy()
    if "Open Date" in df.columns:
        raw = df["Open Date"].astype(str)
        dates = pd.to_datetime(raw, errors="coerce", format="%m/%d/%Y")
        if len(dates) > 0 and dates.isna().mean() > 0.5:
            dates = pd.to_datetime(raw, errors="coerce")
        df["Open_Year"] = dates.dt.year
        df["Open_Month"] = dates.dt.month
        df["Open_Day"] = dates.dt.day
        df["Open_DayOfWeek"] = dates.dt.dayofweek
        df = df.drop(columns=["Open Date"])
    return df


def categorical_columns(df):
    return [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]


def to_numeric_frame(df):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def build_mappings(train_df, cat_cols):
    mappings = {}
    for c in cat_cols:
        if c in train_df.columns:
            vals = train_df[c].dropna().astype(str)
            vals = vals[vals != MISSING]
            mappings[c] = {str(v): i for i, v in enumerate(sorted(vals.unique()))}
        else:
            mappings[c] = {}
    return mappings


def add_categorical_features(df, cat_cols, train_df):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            missing = df[c].isna().astype(int)
            if c in train_df.columns:
                freq = train_df[c].fillna(MISSING).astype(str).value_counts(normalize=True)
                freq_vals = df[c].fillna(MISSING).astype(str).map(freq).fillna(0.0)
            else:
                freq_vals = pd.Series(0.0, index=df.index)
        else:
            missing = pd.Series(1, index=df.index)
            freq_vals = pd.Series(0.0, index=df.index)
        df[f"{c}_missing"] = missing
        df[f"{c}_freq"] = freq_vals
    return df


def encode_categoricals(df, cat_cols, mappings):
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            df[c] = -1
            continue
        mapping = mappings.get(c, {})
        if mapping:
            df[c] = df[c].fillna(MISSING).astype(str).map(mapping).fillna(-1).astype(int)
        else:
            df[c] = -1
    return df


def prepare_catboost_frame(df, cat_cols, train_df):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(MISSING).astype(str)
            if c in train_df.columns:
                allowed = set(train_df[c].fillna(MISSING).astype(str).unique())
                df[c] = df[c].where(df[c].isin(allowed), MISSING)
            else:
                df[c] = MISSING
        else:
            df[c] = MISSING
    return df


def preprocess(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    y_train = pd.Series(dtype=float)
    if TARGET in train_df.columns:
        train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        y_train = train_df[TARGET].copy()
        train_df = train_df.dropna(subset=[TARGET])
        y_train = y_train.loc[train_df.index].astype(float)
        train_df = train_df.drop(columns=[TARGET])
    else:
        train_df = train_df.replace([np.inf, -np.inf], np.nan)

    if ID_COL in train_df.columns:
        train_df = train_df.drop(columns=[ID_COL])
    if ID_COL in test_df.columns:
        test_df = test_df.drop(columns=[ID_COL])

    train_df = parse_date(train_df)
    test_df = parse_date(test_df)

    cat_cols = categorical_columns(train_df)
    test_cat_cols = categorical_columns(test_df)
    cat_cols = list(dict.fromkeys(cat_cols + [c for c in test_cat_cols if c not in cat_cols]))

    mappings = build_mappings(train_df, cat_cols)

    train_enc = add_categorical_features(train_df, cat_cols, train_df)
    test_enc = add_categorical_features(test_df, cat_cols, train_df)
    train_enc = encode_categoricals(train_enc, cat_cols, mappings)
    test_enc = encode_categoricals(test_enc, cat_cols, mappings)

    train_enc = to_numeric_frame(train_enc)
    test_enc = to_numeric_frame(test_enc)

    feature_cols = list(train_enc.columns)
    train_enc = train_enc.reindex(columns=feature_cols)
    test_enc = test_enc.reindex(columns=feature_cols, fill_value=np.nan)
    feature_cols = [c for c in train_enc.columns if train_enc[c].notna().any()]
    train_enc = train_enc[feature_cols]
    test_enc = test_enc[feature_cols]

    train_cb = prepare_catboost_frame(train_df, cat_cols, train_df)
    test_cb = prepare_catboost_frame(test_df, cat_cols, train_df)

    cb_feature_cols = list(train_cb.columns)
    train_cb = train_cb.reindex(columns=cb_feature_cols)
    test_cb = test_cb.reindex(columns=cb_feature_cols, fill_value=MISSING)

    cb_numeric_cols = [c for c in train_cb.columns if c not in cat_cols]
    for c in cb_numeric_cols:
        train_cb[c] = pd.to_numeric(train_cb[c], errors="coerce")
        test_cb[c] = pd.to_numeric(test_cb[c], errors="coerce")
    train_cb = train_cb.replace([np.inf, -np.inf], np.nan)
    test_cb = test_cb.replace([np.inf, -np.inf], np.nan)

    cb_feature_cols = [c for c in train_cb.columns if train_cb[c].notna().any()]
    train_cb = train_cb[cb_feature_cols]
    test_cb = test_cb[cb_feature_cols]

    return train_enc, test_enc, train_cb, test_cb, y_train, [c for c in cat_cols if c in train_cb.columns]


def safe_predict(model, X, fallback):
    try:
        pred = model.predict(X)
    except Exception:
        pred = np.full(len(X), np.nan)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if pred.size == 0:
        pred = np.full(len(X), np.nan)
    elif pred.size != len(X):
        pred = np.resize(pred, len(X))
    return np.where(np.isfinite(pred), pred, fallback)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train_enc, X_test_enc, X_train_cb, X_test_cb, y_train, cb_cat_cols = preprocess(train_df, test_df)

    if len(y_train) == 0:
        fallback = 0.0
    else:
        fallback = float(np.nanmean(y_train))
        if not np.isfinite(fallback):
            fallback = 0.0

    if len(X_test_enc) == 0:
        pd.DataFrame({"id": np.array([], dtype=int), "revenue": np.array([], dtype=float)}).to_csv(
            "predictions.csv", index=False
        )
        return

    if X_train_enc.shape[1] == 0 or len(X_train_enc) == 0:
        pred = np.full(len(X_test_enc), fallback)
        pd.DataFrame({"id": np.arange(len(pred), dtype=int), "revenue": pred}).to_csv(
            "predictions.csv", index=False
        )
        return

    if len(X_train_enc) > 20:
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train_enc)), test_size=0.2, random_state=RANDOM_STATE
        )
    else:
        train_idx = np.arange(len(X_train_enc))
        val_idx = train_idx

    X_tr_enc = X_train_enc.iloc[train_idx]
    X_val_enc = X_train_enc.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    X_tr_cb = X_train_cb.iloc[train_idx]
    X_val_cb = X_train_cb.iloc[val_idx]

    val_fallback = float(np.nanmean(y_tr)) if len(y_tr) else fallback
    if not np.isfinite(val_fallback):
        val_fallback = fallback

    candidates = []

    if lgb is not None:
        candidates.append(
            {
                "name": "lgbm",
                "model": lgb.LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "X_train": X_train_enc,
                "X_test": X_test_enc,
                "X_tr": X_tr_enc,
                "X_val": X_val_enc,
            }
        )

    if xgb is not None:
        candidates.append(
            {
                "name": "xgb",
                "model": xgb.XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
                "X_train": X_train_enc,
                "X_test": X_test_enc,
                "X_tr": X_tr_enc,
                "X_val": X_val_enc,
            }
        )

    if CatBoostRegressor is not None:
        candidates.append(
            {
                "name": "catboost",
                "model": CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=RANDOM_STATE,
                    verbose=0,
                    cat_features=cb_cat_cols,
                ),
                "X_train": X_train_cb,
                "X_test": X_test_cb,
                "X_tr": X_tr_cb,
                "X_val": X_val_cb,
            }
        )

    rmse_scores = {}
    for cand in candidates:
        try:
            cand["model"].fit(cand["X_tr"], y_tr)
            pred = safe_predict(cand["model"], cand["X_val"], val_fallback)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            if np.isfinite(rmse):
                rmse_scores[cand["name"]] = rmse
        except Exception:
            continue

    if rmse_scores:
        weights = {name: 1.0 / max(rmse, 1e-8) for name, rmse in rmse_scores.items()}
        total = sum(weights.values())
        if not np.isfinite(total) or total <= 0:
            weights = {name: 1.0 for name in rmse_scores}
            total = float(len(weights))
        weights = {name: w / total for name, w in weights.items()}

        test_preds = []
        for cand in candidates:
            weight = weights.get(cand["name"], 0.0)
            if weight <= 0:
                continue
            try:
                cand["model"].fit(cand["X_train"], y_train)
                pred = safe_predict(cand["model"], cand["X_test"], fallback)
                test_preds.append(weight * pred)
            except Exception:
                continue

        if test_preds:
            test_pred = sum(test_preds)
        else:
            test_pred = np.full(len(X_test_enc), fallback)
    else:
        test_pred = np.full(len(X_test_enc), fallback)

    test_pred = np.asarray(test_pred, dtype=float).reshape(-1)
    test_pred = np.where(np.isfinite(test_pred), test_pred, fallback)
    test_pred = np.clip(test_pred, 0.0, None)

    out = pd.DataFrame({"id": np.arange(len(test_pred), dtype=int), "revenue": test_pred})
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()