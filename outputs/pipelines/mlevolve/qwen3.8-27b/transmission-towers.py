import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
import lightgbm as lgb

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred, **kwargs):
        return float(np.sqrt(mean_squared_error(y_true, y_pred, **kwargs)))


warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
MODEL_DIR = os.path.join(WORKING_DIR, "models")
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "SupStruc"
ID_CANDIDATES = {"id", "idx", "index", "row_id", "rowid", "sample_id", "sampleid"}

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if str(c).strip().lower() in ID_CANDIDATES]
    return df.drop(columns=drop_cols, errors="ignore")


train_raw = drop_id_columns(train_raw)
test_raw = drop_id_columns(test_raw)

if TARGET in test_raw.columns:
    test_raw = test_raw.drop(columns=[TARGET], errors="ignore")

if TARGET not in train_raw.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

train_raw[TARGET] = pd.to_numeric(train_raw[TARGET], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
n_before = len(train_raw)
train_raw = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)
n_dropped = n_before - len(train_raw)

# ---------------------------------------------------------------------------
# 2. Leakage-safe train/validation split BEFORE feature engineering
# ---------------------------------------------------------------------------
y_all = train_raw[TARGET].astype(float).values
n_unique = len(np.unique(y_all))
n_strata = min(10, max(2, n_unique))

try:
    quantiles = np.quantile(y_all, np.linspace(0, 1, n_strata + 1)[1:-1])
    strata = np.digitize(y_all, quantiles)
except Exception:
    strata = np.zeros(len(y_all), dtype=int)

train_part, val_part = train_test_split(
    train_raw,
    test_size=0.2,
    random_state=SEED,
    stratify=strata if n_unique > 1 else None,
)

X_train_raw = train_part.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)
y_train = train_part[TARGET].astype(float).values
X_val_raw = val_part.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)
y_val = val_part[TARGET].astype(float).values


# ---------------------------------------------------------------------------
# 3. Feature engineering utilities, all fitted on train split only
# ---------------------------------------------------------------------------
def _is_numeric_like(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False

    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False

    sample = sample.astype(str)
    if sample.str.match(r"^\s*-?\d+([.]\d+)?\s*$").all():
        return False

    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) > 0.5
    except Exception:
        return False


def _infer_column_types(df: pd.DataFrame, n_train: int):
    numeric_cols, categorical_cols, datetime_cols = [], [], []
    max_cat_card = max(20, int(0.05 * n_train))

    for col in df.columns:
        s = df[col]
        if _is_datetime_like(s):
            datetime_cols.append(col)
        elif _is_numeric_like(s):
            nuniq = s.nunique(dropna=True)
            if nuniq <= max_cat_card and nuniq > 1:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols


def _safe_power_transform(train_values, val_values, test_values):
    try:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train_t = pt.fit_transform(train_values.reshape(-1, 1)).ravel()
        val_t = pt.transform(val_values.reshape(-1, 1)).ravel()
        test_t = pt.transform(test_values.reshape(-1, 1)).ravel()
        return train_t, val_t, test_t
    except Exception:
        return train_values.copy(), val_values.copy(), test_values.copy()


def _target_encode(train_col, y, val_col, test_col, smoothing: float = 10.0):
    global_mean = float(np.mean(y)) if len(y) > 0 else 0.0
    tmp = pd.DataFrame({"cat": train_col.astype(str), "y": y})
    agg = tmp.groupby("cat")["y"].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
        agg["count"] + smoothing
    )

    def _transform(s: pd.Series):
        return s.astype(str).map(enc).fillna(global_mean).astype(float)

    return _transform(train_col), _transform(val_col), _transform(test_col)


def _frequency_encode(train_col, val_col, test_col):
    freq = train_col.astype(str).value_counts()

    def _transform(s: pd.Series):
        return s.astype(str).map(freq).fillna(0).astype(float)

    return _transform(train_col), _transform(val_col), _transform(test_col)


def _one_hot_encode(
    train_col, val_col, test_col, prefix: str, max_categories: int = 50
):
    cats = train_col.astype(str).value_counts().head(max_categories).index.tolist()
    cats = [c for c in cats if c != "missing"]

    def _transform(s: pd.Series) -> pd.DataFrame:
        s = s.astype(str)
        dummies = pd.get_dummies(s, prefix=prefix, dummy_na=False, dtype=float)
        prefix_str = f"{prefix}_"
        keep = [
            c
            for c in dummies.columns
            if c.startswith(prefix_str) and c[len(prefix_str) :] in cats
        ]
        return dummies[keep]

    return _transform(train_col), _transform(val_col), _transform(test_col)


def _ordinal_encode(train_col, val_col, test_col):
    order = train_col.astype(str).value_counts().index.tolist()
    mapping = {c: i for i, c in enumerate(order)}

    def _transform(s: pd.Series):
        return s.astype(str).map(mapping).fillna(-1).astype(float)

    return _transform(train_col), _transform(val_col), _transform(test_col)


def _add_geometric_features(train_df, val_df, test_df, numeric_cols):
    def _find(names, patterns):
        found = []
        for c in names:
            lc = str(c).lower()
            if any(p in lc for p in patterns):
                found.append(c)
        return found

    lat_cols = _find(numeric_cols, ["lat", "latitude"])
    lon_cols = _find(numeric_cols, ["lon", "lng", "long", "longitude"])
    if lat_cols and lon_cols:
        lat, lon = lat_cols[0], lon_cols[0]
        for df, tag in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            dlat = df[lat].astype(float)
            dlon = df[lon].astype(float)
            df[f"{tag}_lat"] = dlat
            df[f"{tag}_lon"] = dlon
            df["lat_lon_distance"] = np.sqrt(dlat**2 + dlon**2)
            df["lat_lon_angle"] = np.arctan2(dlat, dlon)

    x_cols = _find(numeric_cols, ["_x", "x_", "x", "easting", "east"])
    y_cols = _find(numeric_cols, ["_y", "y_", "y", "northing", "north"])
    x_cols = [
        c
        for c in x_cols
        if str(c).lower() in {"x", "x_coord", "xcoord", "easting", "east"}
        or str(c).lower().startswith("x_")
    ]
    y_cols = [
        c
        for c in y_cols
        if str(c).lower() in {"y", "y_coord", "ycoord", "northing", "north"}
        or str(c).lower().startswith("y_")
    ]

    if x_cols and y_cols:
        x, y = x_cols[0], y_cols[0]
        for df, tag in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            dx = df[x].astype(float)
            dy = df[y].astype(float)
            df[f"{tag}_x"] = dx
            df[f"{tag}_y"] = dy
            df["xy_radius"] = np.sqrt(dx**2 + dy**2)
            df["xy_angle"] = np.arctan2(dy, dx)

    angle_cols = _find(
        numeric_cols, ["angle", "azimuth", "bearing", "direction", "deg", "degree"]
    )
    for c in angle_cols[:10]:
        for df, tag in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            rad = np.deg2rad(df[c].astype(float))
            df[f"{c}_sin"] = np.sin(rad)
            df[f"{c}_cos"] = np.cos(rad)

    return train_df, val_df, test_df


def _add_numeric_interactions(
    train_df, val_df, test_df, numeric_cols, max_pairs: int = 10
):
    if len(numeric_cols) < 2:
        return train_df, val_df, test_df

    corr = train_df[numeric_cols].corr().abs().fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            pairs.append((numeric_cols[i], numeric_cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for a, b, r in pairs[:max_pairs]:
        if r < 0.3:
            break
        for df, tag in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            df[f"{a}_x_{b}"] = df[a].astype(float) * df[b].astype(float)
            df[f"{a}_div_{b}"] = df[a].astype(float) / (df[b].astype(float) + 1e-8)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 4. Build engineered features
# ---------------------------------------------------------------------------
numeric_cols, categorical_cols, datetime_cols = _infer_column_types(
    X_train_raw, len(X_train_raw)
)

X_train = X_train_raw.copy()
X_val = X_val_raw.copy()
X_test = test_raw.copy()

for col in X_train_raw.columns:
    if col not in X_val.columns:
        X_val[col] = np.nan
    if col not in X_test.columns:
        X_test[col] = np.nan

# Datetime features
for col in datetime_cols:
    for df, tag in [(X_train, "train"), (X_val, "val"), (X_test, "test")]:
        s = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = s.dt.year.astype(float)
        df[f"{col}_month"] = s.dt.month.astype(float)
        df[f"{col}_day"] = s.dt.day.astype(float)
        df[f"{col}_hour"] = s.dt.hour.astype(float)
        df[f"{col}_dow"] = s.dt.dayofweek.astype(float)

        hour = s.dt.hour.fillna(0)
        dow = s.dt.dayofweek.fillna(0)
        df[f"{col}_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df[f"{col}_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

# Missing indicators
for col in list(X_train.columns):
    if X_train[col].isna().any() or X_val[col].isna().any() or X_test[col].isna().any():
        X_train[f"{col}_is_missing"] = X_train[col].isna().astype(float)
        X_val[f"{col}_is_missing"] = X_val[col].isna().astype(float)
        X_test[f"{col}_is_missing"] = X_test[col].isna().astype(float)

# Numeric features
for col in numeric_cols:
    if col not in X_train.columns:
        continue

    for df in [X_train, X_val, X_test]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    median = float(X_train[col].median()) if X_train[col].notna().any() else 0.0
    for df in [X_train, X_val, X_test]:
        df[col] = df[col].fillna(median)

    train_vals = X_train[col].astype(float).values
    val_vals = X_val[col].astype(float).values
    test_vals = X_test[col].astype(float).values

    train_t, val_t, test_t = _safe_power_transform(train_vals, val_vals, test_vals)
    X_train[f"{col}_yj"] = train_t
    X_val[f"{col}_yj"] = val_t
    X_test[f"{col}_yj"] = test_t

    mean = float(np.mean(train_vals))
    std = float(np.std(train_vals))
    if std < 1e-12:
        std = 1.0

    X_train[f"{col}_robust_z"] = (X_train[col] - mean) / std
    X_val[f"{col}_robust_z"] = (X_val[col] - mean) / std
    X_test[f"{col}_robust_z"] = (X_test[col] - mean) / std

# Categorical features
for col in categorical_cols:
    if col not in X_train.columns:
        continue

    for df in [X_train, X_val, X_test]:
        df[col] = df[col].astype("string").fillna("missing")

    tr_f, va_f, te_f = _frequency_encode(X_train[col], X_val[col], X_test[col])
    X_train[f"{col}_freq"] = tr_f
    X_val[f"{col}_freq"] = va_f
    X_test[f"{col}_freq"] = te_f

    tr_t, va_t, te_t = _target_encode(
        X_train[col], y_train, X_val[col], X_test[col], smoothing=10.0
    )
    X_train[f"{col}_target_enc"] = tr_t
    X_val[f"{col}_target_enc"] = va_t
    X_test[f"{col}_target_enc"] = te_t

    tr_o, va_o, te_o = _ordinal_encode(X_train[col], X_val[col], X_test[col])
    X_train[f"{col}_ordinal"] = tr_o
    X_val[f"{col}_ordinal"] = va_o
    X_test[f"{col}_ordinal"] = te_o

    if X_train[col].nunique(dropna=True) <= 50:
        tr_h, va_h, te_h = _one_hot_encode(
            X_train[col], X_val[col], X_test[col], prefix=col, max_categories=50
        )
        X_train = pd.concat([X_train, tr_h], axis=1)
        X_val = pd.concat([X_val, va_h], axis=1)
        X_test = pd.concat([X_test, te_h], axis=1)

    for df in [X_train, X_val, X_test]:
        df.drop(columns=[col], inplace=True, errors="ignore")

# Geometric and interaction features
X_train, X_val, X_test = _add_geometric_features(X_train, X_val, X_test, numeric_cols)
X_train, X_val, X_test = _add_numeric_interactions(
    X_train, X_val, X_test, numeric_cols, max_pairs=10
)


# ---------------------------------------------------------------------------
# 5. Finalize features
# ---------------------------------------------------------------------------
def _finalize_features(train_df, val_df, test_df):
    all_cols = list(train_df.columns)
    for col in val_df.columns:
        if col not in all_cols:
            all_cols.append(col)
    for col in test_df.columns:
        if col not in all_cols:
            all_cols.append(col)

    seen = set()
    unique_cols = []
    for c in all_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    all_cols = unique_cols

    for df in [train_df, val_df, test_df]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan

    train_df = train_df[all_cols]
    val_df = val_df[all_cols]
    test_df = test_df[all_cols]

    for col in all_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    non_const = [c for c in all_cols if train_df[c].nunique(dropna=True) > 1]
    if not non_const:
        non_const = all_cols

    train_df = train_df[non_const]
    val_df = val_df[non_const]
    test_df = test_df[non_const]

    medians = train_df.median(numeric_only=True)
    for col in non_const:
        med = medians.get(col, 0.0)
        if pd.isna(med):
            med = 0.0
        train_df[col] = train_df[col].fillna(med)
        val_df[col] = val_df[col].fillna(med)
        test_df[col] = test_df[col].fillna(med)

    return train_df, val_df, test_df, non_const


X_train, X_val, X_test, feature_names = _finalize_features(X_train, X_val, X_test)
test_ids = np.arange(len(X_test), dtype=int)

# Optional scaled matrices, kept from the original data-processing step.
if len(feature_names) > 0:
    try:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[feature_names].astype(float)),
            columns=[f"{c}_scaled" for c in feature_names],
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val[feature_names].astype(float)),
            columns=X_train_scaled.columns,
            index=X_val.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test[feature_names].astype(float)),
            columns=X_train_scaled.columns,
            index=X_test.index,
        )

        X_train_scaled.to_parquet(
            os.path.join(WORKING_DIR, "train_features_scaled.parquet"), index=False
        )
        X_val_scaled.to_parquet(
            os.path.join(WORKING_DIR, "val_features_scaled.parquet"), index=False
        )
        X_test_scaled.to_parquet(
            os.path.join(WORKING_DIR, "test_features_scaled.parquet"), index=False
        )
    except Exception:
        pass

# Save artifacts for auditability.
X_train.to_parquet(os.path.join(WORKING_DIR, "train_features.parquet"), index=False)
X_val.to_parquet(os.path.join(WORKING_DIR, "val_features.parquet"), index=False)
X_test.to_parquet(os.path.join(WORKING_DIR, "test_features.parquet"), index=False)
np.save(os.path.join(WORKING_DIR, "train_target.npy"), y_train)
np.save(os.path.join(WORKING_DIR, "val_target.npy"), y_val)
np.save(os.path.join(WORKING_DIR, "test_ids.npy"), test_ids)

metadata = {
    "target": TARGET,
    "train_shape": list(X_train.shape),
    "val_shape": list(X_val.shape),
    "test_shape": list(X_test.shape),
    "feature_names": feature_names,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "datetime_cols": datetime_cols,
    "split": {
        "method": "train_test_split",
        "test_size": 0.2,
        "stratify": "target_quantiles",
        "random_state": SEED,
    },
    "leakage_policy": "All encoders, imputers, scalers, and target encoders fit on train split only.",
}
with open(os.path.join(WORKING_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 6. Model design: LightGBM GBDT regression with squared-error loss / RMSE metric
# ---------------------------------------------------------------------------
def _select_model_features(train_df, val_df, test_df, fallback_names):
    common = [
        c for c in train_df.columns if c in val_df.columns and c in test_df.columns
    ]
    if not common:
        common = [c for c in fallback_names if c in train_df.columns]

    redundant_suffixes = ("_yj", "_robust_z", "_scaled")
    selected = [
        c for c in common if not any(str(c).endswith(s) for s in redundant_suffixes)
    ]
    if not selected:
        selected = common

    try:
        var = train_df[selected].astype(float).var()
        selected = [c for c in selected if float(var[c]) > 1e-12]
    except Exception:
        pass

    if not selected:
        selected = common

    return selected


feature_names = _select_model_features(X_train, X_val, X_test, feature_names)

n = len(y_train)
n_features = len(feature_names)

if n < 5000:
    lr = 0.05
    num_leaves = 15
    min_child_samples = 10
    n_estimators = 3000
    subsample = 0.9
    colsample = 0.8
elif n < 50000:
    lr = 0.03
    num_leaves = 31
    min_child_samples = 20
    n_estimators = 5000
    subsample = 0.9
    colsample = 0.8
else:
    lr = 0.02
    num_leaves = 63
    min_child_samples = 30
    n_estimators = 8000
    subsample = 0.8
    colsample = 0.7

if n_features > 200:
    colsample = min(colsample, 0.65)

params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": lr,
    "num_leaves": num_leaves,
    "max_depth": -1,
    "min_child_samples": min_child_samples,
    "subsample": subsample,
    "subsample_freq": 1,
    "colsample_bytree": colsample,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "num_threads": os.cpu_count() or 1,
    "verbose": -1,
}

# Official validation loss is RMSE. LightGBM's internal optimizer is gradient boosting.
loss_fn = lambda y_true, y_pred: float(root_mean_squared_error(y_true, y_pred))
criterion = root_mean_squared_error
optimizer = lgb.Booster


def _to_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    out = df[cols].astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    return out.values


X_tr = _to_matrix(X_train, feature_names)
X_va = _to_matrix(X_val, feature_names)
X_te = _to_matrix(X_test, feature_names)

# Defensive fallback if validation is empty.
if len(y_val) == 0:
    n_val_internal = max(1, int(round(0.1 * len(y_train))))
    X_va = X_tr[-n_val_internal:]
    y_val = y_train[-n_val_internal:]
    X_tr = X_tr[:-n_val_internal]
    y_train = y_train[:-n_val_internal]

max_rounds = int(max(1000, min(6000, n_estimators)))
es_rounds = int(min(150, max(50, max_rounds // 50)))

if n <= 200000:
    n_seeds = 3
elif n <= 500000:
    n_seeds = 2
else:
    n_seeds = 1

seeds = [SEED + i for i in range(n_seeds)]


def _train_one_seed(seed, params, max_rounds, es_rounds):
    train_set = lgb.Dataset(
        X_tr,
        label=y_train,
        feature_name=feature_names,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_va,
        label=y_val,
        reference=train_set,
        feature_name=feature_names,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        valid_names=["val"],
        callbacks=callbacks,
    )

    best_iter = int(getattr(booster, "best_iteration", 0) or max_rounds)
    best_iter = max(1, min(best_iter, max_rounds))

    val_pred = booster.predict(X_va, num_iteration=best_iter)
    val_pred = np.asarray(val_pred, dtype=float).ravel()
    val_rmse = loss_fn(y_val, val_pred) if len(y_val) > 0 else float("nan")

    return booster, best_iter, val_pred, val_rmse


def _refit_one_seed(seed, params, best_iter):
    all_X = np.vstack([X_tr, X_va]).astype(np.float32)
    all_y = np.concatenate([y_train, y_val]).astype(float)

    train_set = lgb.Dataset(
        all_X,
        label=all_y,
        feature_name=feature_names,
        free_raw_data=False,
    )

    final_rounds = max(100, int(best_iter))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return booster


def _predict_ensemble(models, iters, X):
    preds = []
    for booster, it in zip(models, iters):
        p = booster.predict(X, num_iteration=int(it))
        p = np.asarray(p, dtype=float).ravel()
        preds.append(p)
    return np.mean(np.stack(preds, axis=0), axis=0)


def _clean_predictions(preds, reference_y):
    preds = np.asarray(preds, dtype=float).ravel()
    if preds.size == 0:
        return preds

    ref = np.asarray(reference_y, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]

    if ref.size == 0:
        fill, lo, hi = 0.0, 0.0, 0.0
    else:
        fill = float(np.mean(ref))
        lo = float(np.min(ref))
        hi = float(np.max(ref))

    return np.nan_to_num(preds, nan=fill, posinf=hi, neginf=lo)


# ---------------------------------------------------------------------------
# 7. Train seed ensemble, validate with official RMSE, refit on full labeled split
# ---------------------------------------------------------------------------
models = []
final_models = []
best_iters = []
val_scores = []

for seed in seeds:
    booster, best_iter, val_pred, val_rmse = _train_one_seed(
        seed=seed,
        params=params,
        max_rounds=max_rounds,
        es_rounds=es_rounds,
    )

    models.append(booster)
    best_iters.append(best_iter)
    val_scores.append(val_rmse)

    booster.save_model(os.path.join(MODEL_DIR, f"lgb_val_seed_{seed}.txt"))

    final_booster = _refit_one_seed(seed=seed, params=params, best_iter=best_iter)
    final_models.append(final_booster)
    final_booster.save_model(os.path.join(MODEL_DIR, f"lgb_final_seed_{seed}.txt"))

    print(f"seed={seed} best_iter={best_iter} val_rmse={val_rmse:.6f}")

# Validation uses the same ensemble averaging logic as test inference.
val_ensemble_pred = _predict_ensemble(models, best_iters, X_va)
val_ensemble_pred = _clean_predictions(val_ensemble_pred, y_val)
score = loss_fn(y_val, val_ensemble_pred) if len(y_val) > 0 else float("nan")

# Test inference uses refit models with validation-selected iterations.
test_pred = _predict_ensemble(final_models, best_iters, X_te)
test_pred = _clean_predictions(test_pred, np.concatenate([y_train, y_val]))

if test_ids is None or len(test_ids) != len(test_pred):
    test_ids = np.arange(len(test_pred), dtype=int)

submission = pd.DataFrame(
    {
        "id": test_ids.astype(int),
        TARGET: test_pred.astype(float),
    }
)
submission = submission[["id", TARGET]]
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

training_metadata = {
    "task": "regression",
    "target": TARGET,
    "metric": "rmse",
    "split": {
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(X_test)),
        "source": "previously engineered leakage-safe split",
    },
    "model": {
        "family": "lightgbm_gbdt",
        "objective": "regression_l2",
        "metric": "rmse",
        "seed_ensemble": seeds,
        "best_iterations": best_iters,
        "max_rounds": int(max_rounds),
        "early_stopping_rounds": int(es_rounds),
        "final_refit": "train + validation rows with validation-selected iterations",
    },
    "features": {
        "count": int(len(feature_names)),
        "names": [str(c) for c in feature_names],
    },
    "validation_rmse": score,
    "params": params,
}

with open(os.path.join(WORKING_DIR, "training_metadata.json"), "w") as f:
    json.dump(training_metadata, f, indent=2, default=str)

print(f"Final Validation Score: {score}")
