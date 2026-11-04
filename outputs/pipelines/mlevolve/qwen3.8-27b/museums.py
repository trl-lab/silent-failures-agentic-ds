import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

TARGET = "REVENUE15"
SEED = 42
VALID_FRACTION = 0.2
TARGET_ENCODING_SMOOTHING = 20.0
ONE_HOT_MAX_CARDINALITY = 25
GROUP_STAT_MAX_CARDINALITY = 50
MAX_GROUP_STAT_NUMERIC_FEATURES = 18
WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def safe_name(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] if value else "feature"


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    seen = {}
    for col in df.columns:
        base = str(col)
        if base in seen:
            seen[base] += 1
            cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            cols.append(base)
    df.columns = cols
    return df


def is_id_like_column(name: str) -> bool:
    lower = str(name).lower()
    if lower in {
        "id",
        "index",
        "idx",
        "row_id",
        "sample_id",
        "museum_id",
        "identifier",
    }:
        return True
    return bool(re.fullmatch(r".*(_|^)id(_|$)", lower))


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        out = series.astype(float)
    else:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        out = pd.to_numeric(cleaned, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def looks_like_date_column(name: str) -> bool:
    lower = str(name).lower()
    return any(
        token in lower for token in ["date", "time", "year", "month", "day", "week"]
    )


def is_datetime_column(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    if not looks_like_date_column(name):
        return False
    sample = series.dropna().astype(str).head(100)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.8


def is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    converted = clean_numeric_series(series)
    return converted.notna().sum() > 0.5 * len(series)


def add_to_outputs(train_df, valid_df, test_df, name, tr, va, te):
    train_df[name] = tr
    valid_df[name] = va
    test_df[name] = te


def standardize_fit(train_series: pd.Series):
    mean = float(train_series.mean()) if train_series.notna().any() else 0.0
    std = float(train_series.std(ddof=0)) if train_series.notna().sum() > 1 else 0.0
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def transform_standardized(series: pd.Series, mean: float, std: float) -> pd.Series:
    return (series - mean) / std


def signed_log1p(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(np.abs(series))


def robust_z(series: pd.Series, median: float, iqr: float) -> pd.Series:
    if not np.isfinite(iqr) or iqr < 1e-12:
        iqr = 1.0
    return (series - median) / (1.4826 * iqr)


def rmse_metric(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_target_encoding(series, y, smoothing, global_mean):
    if len(series) == 0:
        return pd.Series(dtype=float)
    target_sum = y.groupby(series, dropna=True).sum()
    target_count = y.groupby(series, dropna=True).count()
    if len(target_sum) == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    enc = (target_sum + smoothing * global_mean) / (target_count + smoothing)
    return series.map(enc)


def target_encode_oof(
    series, y, n_splits=5, smoothing=TARGET_ENCODING_SMOOTHING, global_mean=0.0
):
    out = pd.Series(global_mean, index=series.index, dtype=float)
    n = len(series)
    if n < 5 or series.nunique(dropna=True) == 0:
        return out
    n_splits = max(2, min(n_splits, n))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf.split(np.arange(n)):
        tr_s = series.iloc[tr_idx]
        va_s = series.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        if len(tr_s) == 0:
            continue
        target_sum = y_tr.groupby(tr_s, dropna=True).sum()
        target_count = y_tr.groupby(tr_s, dropna=True).count()
        if len(target_sum) == 0:
            continue
        enc = (target_sum + smoothing * global_mean) / (target_count + smoothing)
        mapped = va_s.map(enc).fillna(global_mean)
        out.iloc[va_idx] = mapped.values
    return out


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    seen = set()
    for col in df.columns:
        name = safe_name(col)
        if name in seen:
            suffix = 1
            while f"{name}_{suffix}" in seen:
                suffix += 1
            name = f"{name}_{suffix}"
        seen.add(name)
        new_cols.append(name)
    df.columns = new_cols
    return df


def drop_constant_columns(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame
):
    keep = []
    for col in df_train.columns:
        train_vals = df_train[col]
        if train_vals.dtype == object:
            continue
        train_vals = pd.to_numeric(train_vals, errors="coerce")
        if train_vals.notna().sum() == 0:
            continue
        if train_vals.nunique(dropna=True) <= 1 and train_vals.isna().sum() == 0:
            continue
        keep.append(col)
    return keep


def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.dtype != object else s
    )
    return out.replace([np.inf, -np.inf], np.nan)


def to_cat(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), np.nan)


def process_data():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train_raw = make_columns_unique(train_raw.reset_index(drop=True))
    test_raw = make_columns_unique(test_raw.reset_index(drop=True))

    if TARGET not in train_raw.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train.parquet.")

    y_full = clean_numeric_series(train_raw[TARGET]).replace([np.inf, -np.inf], np.nan)
    keep = y_full.notna()
    train_raw = train_raw.loc[keep].reset_index(drop=True)
    y_full = y_full.loc[keep].reset_index(drop=True)

    if len(train_raw) == 0:
        raise ValueError("No usable training rows with non-missing target.")

    test_ids = pd.Series(np.arange(len(test_raw), dtype=np.int64), name="id")

    drop_cols = [c for c in train_raw.columns if c == TARGET or is_id_like_column(c)]
    feature_cols = [c for c in train_raw.columns if c not in drop_cols]

    X_full = train_raw[feature_cols].copy()
    X_test_raw = test_raw[[c for c in feature_cols if c in test_raw.columns]].copy()
    X_test_raw = X_test_raw.reindex(columns=feature_cols)

    if len(X_full) <= 1:
        train_idx = np.arange(len(X_full))
        valid_idx = np.arange(len(X_full))
    else:
        stratify = None
        if len(y_full) >= 50 and y_full.nunique() > 1:
            try:
                stratify = pd.qcut(y_full, q=10, labels=False, duplicates="drop")
            except Exception:
                stratify = None
        try:
            train_idx, valid_idx = train_test_split(
                np.arange(len(X_full)),
                test_size=VALID_FRACTION,
                random_state=SEED,
                stratify=stratify,
            )
        except Exception:
            train_idx, valid_idx = train_test_split(
                np.arange(len(X_full)),
                test_size=VALID_FRACTION,
                random_state=SEED,
                stratify=None,
            )
        if len(valid_idx) == 0:
            valid_idx = np.array([len(X_full) - 1])
            train_idx = np.setdiff1d(
                np.arange(len(X_full)), valid_idx, assume_unique=False
            )
        if len(train_idx) == 0:
            raise ValueError("No training rows after split.")

    X_train_raw = X_full.iloc[train_idx].copy().reset_index(drop=True)
    y_train = y_full.iloc[train_idx].astype(float).reset_index(drop=True)
    X_valid_raw = X_full.iloc[valid_idx].copy().reset_index(drop=True)
    y_valid = y_full.iloc[valid_idx].astype(float).reset_index(drop=True)
    X_test_raw = X_test_raw.copy().reset_index(drop=True)

    train_out = pd.DataFrame(index=X_train_raw.index)
    valid_out = pd.DataFrame(index=X_valid_raw.index)
    test_out = pd.DataFrame(index=X_test_raw.index)

    base_numeric = {}
    base_categorical = {}

    for col in feature_cols:
        tr = X_train_raw[col]
        va = X_valid_raw[col]
        te = X_test_raw[col]

        if is_datetime_column(tr, col):
            dt_tr = pd.to_datetime(tr, errors="coerce")
            dt_va = pd.to_datetime(va, errors="coerce")
            dt_te = pd.to_datetime(te, errors="coerce")

            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_year",
                dt_tr.dt.year.astype(float),
                dt_va.dt.year.astype(float),
                dt_te.dt.year.astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_month",
                dt_tr.dt.month.astype(float),
                dt_va.dt.month.astype(float),
                dt_te.dt.month.astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_day",
                dt_tr.dt.day.astype(float),
                dt_va.dt.day.astype(float),
                dt_te.dt.day.astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_dayofweek",
                dt_tr.dt.dayofweek.astype(float),
                dt_va.dt.dayofweek.astype(float),
                dt_te.dt.dayofweek.astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_is_weekend",
                (dt_tr.dt.dayofweek >= 5).astype(float),
                (dt_va.dt.dayofweek >= 5).astype(float),
                (dt_te.dt.dayofweek >= 5).astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_missing",
                dt_tr.isna().astype(float),
                dt_va.isna().astype(float),
                dt_te.isna().astype(float),
            )

            base_numeric[f"{col}_year"] = (
                dt_tr.dt.year.astype(float),
                dt_va.dt.year.astype(float),
                dt_te.dt.year.astype(float),
            )
            base_numeric[f"{col}_month"] = (
                dt_tr.dt.month.astype(float),
                dt_va.dt.month.astype(float),
                dt_te.dt.month.astype(float),
            )
            base_numeric[f"{col}_dayofweek"] = (
                dt_tr.dt.dayofweek.astype(float),
                dt_va.dt.dayofweek.astype(float),
                dt_te.dt.dayofweek.astype(float),
            )
            continue

        if is_numeric_column(tr):
            tr_num = clean_numeric_series(tr)
            va_num = clean_numeric_series(va)
            te_num = clean_numeric_series(te)

            add_to_outputs(train_out, valid_out, test_out, col, tr_num, va_num, te_num)
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_missing",
                tr_num.isna().astype(float),
                va_num.isna().astype(float),
                te_num.isna().astype(float),
            )
            base_numeric[col] = (tr_num, va_num, te_num)
            continue

        tr_cat = to_cat(tr)
        va_cat = to_cat(va)
        te_cat = to_cat(te)

        add_to_outputs(
            train_out,
            valid_out,
            test_out,
            f"{col}_missing",
            tr_cat.isna().astype(float),
            va_cat.isna().astype(float),
            te_cat.isna().astype(float),
        )
        base_categorical[col] = (tr_cat, va_cat, te_cat)

    numeric_cols = list(base_numeric.keys())

    for col in numeric_cols:
        tr, va, te = base_numeric[col]
        tr = tr.astype(float)
        va = va.astype(float)
        te = te.astype(float)

        median = float(tr.median()) if tr.notna().any() else 0.0
        mean, std = standardize_fit(tr)
        q01 = float(tr.quantile(0.01)) if tr.notna().any() else median
        q99 = float(tr.quantile(0.99)) if tr.notna().any() else median
        iqr = float(tr.quantile(0.75) - tr.quantile(0.25)) if tr.notna().any() else 0.0

        tr_imp = tr.fillna(median)
        va_imp = va.fillna(median)
        te_imp = te.fillna(median)

        if np.isfinite(q01) and np.isfinite(q99) and q01 < q99:
            tr_clip = tr_imp.clip(lower=q01, upper=q99)
            va_clip = va_imp.clip(lower=q01, upper=q99)
            te_clip = te_imp.clip(lower=q01, upper=q99)
        else:
            tr_clip = tr_imp
            va_clip = va_imp
            te_clip = te_imp

        add_to_outputs(
            train_out, valid_out, test_out, f"{col}_imp", tr_imp, va_imp, te_imp
        )
        add_to_outputs(
            train_out,
            valid_out,
            test_out,
            f"{col}_std",
            transform_standardized(tr_imp, mean, std),
            transform_standardized(va_imp, mean, std),
            transform_standardized(te_imp, mean, std),
        )
        add_to_outputs(
            train_out, valid_out, test_out, f"{col}_clip", tr_clip, va_clip, te_clip
        )
        add_to_outputs(
            train_out,
            valid_out,
            test_out,
            f"{col}_robust_z",
            robust_z(tr_imp, median, iqr),
            robust_z(va_imp, median, iqr),
            robust_z(te_imp, median, iqr),
        )
        add_to_outputs(
            train_out,
            valid_out,
            test_out,
            f"{col}_signed_log",
            signed_log1p(tr_imp),
            signed_log1p(va_imp),
            signed_log1p(te_imp),
        )

    global_target_mean = float(y_train.mean()) if len(y_train) else 0.0
    categorical_meta = {}

    for col in list(base_categorical.keys()):
        tr, va, te = base_categorical[col]

        tr_str = tr.astype(str).where(tr.notna(), np.nan)
        va_str = va.astype(str).where(va.notna(), np.nan)
        te_str = te.astype(str).where(te.notna(), np.nan)

        train_counts = tr_str.value_counts(dropna=True)
        n_unique = len(train_counts)

        if n_unique == 0:
            categorical_meta[col] = {"type": "empty", "n_unique": 0}
            continue

        if n_unique == 2:
            most_common = train_counts.index[0]
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"{col}_bin",
                (tr_str == most_common).astype(float),
                (va_str == most_common).astype(float),
                (te_str == most_common).astype(float),
            )

        freq = train_counts / max(len(tr_str), 1)
        add_to_outputs(
            train_out,
            valid_out,
            test_out,
            f"{col}_freq",
            tr_str.map(freq).fillna(0.0).astype(float),
            va_str.map(freq).fillna(0.0).astype(float),
            te_str.map(freq).fillna(0.0).astype(float),
        )

        full_target_enc = compute_target_encoding(
            tr_str, y_train, TARGET_ENCODING_SMOOTHING, global_target_mean
        )
        tr_target_enc = target_encode_oof(
            tr_str, y_train, 5, TARGET_ENCODING_SMOOTHING, global_target_mean
        )
        add_to_outputs(
            train_out,
            valid_out,
            test_out,
            f"{col}_target_enc",
            tr_target_enc.fillna(global_target_mean).astype(float),
            va_str.map(full_target_enc).fillna(global_target_mean).astype(float),
            te_str.map(full_target_enc).fillna(global_target_mean).astype(float),
        )

        if n_unique <= ONE_HOT_MAX_CARDINALITY:
            categories = train_counts.index.tolist()
            for cat in categories:
                cat_name = safe_name(f"{col}_{cat}")
                add_to_outputs(
                    train_out,
                    valid_out,
                    test_out,
                    cat_name,
                    (tr_str == cat).astype(float),
                    (va_str == cat).astype(float),
                    (te_str == cat).astype(float),
                )

        categorical_meta[col] = {
            "type": "categorical",
            "n_unique": int(n_unique),
            "one_hot": bool(n_unique <= ONE_HOT_MAX_CARDINALITY),
        }

    group_categoricals = [
        col
        for col, meta in categorical_meta.items()
        if meta.get("type") == "categorical"
        and 2 <= meta.get("n_unique", 0) <= GROUP_STAT_MAX_CARDINALITY
    ]

    numeric_feature_candidates = []
    for col in numeric_cols:
        tr = base_numeric[col][0].astype(float)
        if tr.notna().sum() > 0 and tr.nunique(dropna=True) > 1:
            numeric_feature_candidates.append(col)

    if len(numeric_feature_candidates) > MAX_GROUP_STAT_NUMERIC_FEATURES:
        corr_scores = []
        for col in numeric_feature_candidates:
            tr = base_numeric[col][0].astype(float)
            combined = pd.DataFrame({"y": y_train, "x": tr}).dropna()
            if len(combined) >= 10 and combined["x"].std(ddof=0) > 1e-12:
                score = combined["y"].corr(combined["x"])
                score = abs(float(score)) if np.isfinite(score) else 0.0
            else:
                score = 0.0
            corr_scores.append((score, col))
        corr_scores.sort(key=lambda item: item[0], reverse=True)
        selected_numeric_for_group = [
            col for _, col in corr_scores[:MAX_GROUP_STAT_NUMERIC_FEATURES]
        ]
    else:
        selected_numeric_for_group = numeric_feature_candidates

    for gcol in group_categoricals:
        tr_g, va_g, te_g = base_categorical[gcol]
        tr_g = tr_g.astype(str).where(tr_g.notna(), np.nan)
        va_g = va_g.astype(str).where(va_g.notna(), np.nan)
        te_g = te_g.astype(str).where(te_g.notna(), np.nan)

        for ncol in selected_numeric_for_group:
            tr_n = base_numeric[ncol][0].astype(float)
            va_n = base_numeric[ncol][1].astype(float)
            te_n = base_numeric[ncol][2].astype(float)

            tmp = pd.DataFrame({"g": tr_g, "x": tr_n}).dropna()
            if len(tmp) == 0:
                continue

            grouped = tmp.groupby("g")["x"]
            g_mean = grouped.mean()
            g_median = grouped.median()
            g_std = grouped.std(ddof=0).fillna(0.0)
            g_min = grouped.min()
            g_max = grouped.max()

            global_mean = float(tmp["x"].mean())
            global_median = float(tmp["x"].median())
            global_std = float(tmp["x"].std(ddof=0)) if len(tmp) > 1 else 0.0
            global_min = float(tmp["x"].min())
            global_max = float(tmp["x"].max())

            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"grp_{gcol}_{ncol}_mean",
                tr_g.map(g_mean).fillna(global_mean).astype(float),
                va_g.map(g_mean).fillna(global_mean).astype(float),
                te_g.map(g_mean).fillna(global_mean).astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"grp_{gcol}_{ncol}_median",
                tr_g.map(g_median).fillna(global_median).astype(float),
                va_g.map(g_median).fillna(global_median).astype(float),
                te_g.map(g_median).fillna(global_median).astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"grp_{gcol}_{ncol}_std",
                tr_g.map(g_std).fillna(global_std).astype(float),
                va_g.map(g_std).fillna(global_std).astype(float),
                te_g.map(g_std).fillna(global_std).astype(float),
            )
            add_to_outputs(
                train_out,
                valid_out,
                test_out,
                f"grp_{gcol}_{ncol}_range",
                tr_g.map(g_max - g_min).fillna(global_max - global_min).astype(float),
                va_g.map(g_max - g_min).fillna(global_max - global_min).astype(float),
                te_g.map(g_max - g_min).fillna(global_max - global_min).astype(float),
            )

    train_out = sanitize_columns(train_out)
    valid_out = sanitize_columns(valid_out)
    test_out = sanitize_columns(test_out)

    keep_cols = drop_constant_columns(train_out, valid_out, test_out)
    train_out = train_out[keep_cols]
    valid_out = valid_out[keep_cols]
    test_out = test_out[keep_cols]

    train_out = coerce_numeric_frame(train_out)
    valid_out = coerce_numeric_frame(valid_out)
    test_out = coerce_numeric_frame(test_out)

    return train_out, valid_out, test_out, y_train, y_valid, test_ids


def main():
    train_X, valid_X, test_X, y_train, y_valid, test_ids = process_data()

    if train_X.shape[1] == 0:
        train_X = pd.DataFrame(
            {"fallback_feature": np.arange(len(train_X), dtype=float)},
            index=train_X.index,
        )
        valid_X = pd.DataFrame(
            {"fallback_feature": np.arange(len(valid_X), dtype=float)},
            index=valid_X.index,
        )
        test_X = pd.DataFrame(
            {"fallback_feature": np.arange(len(test_X), dtype=float)},
            index=test_X.index,
        )

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        min_child_weight=1e-3,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        train_X,
        y_train,
        eval_set=[(valid_X, y_valid)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    best_iter = getattr(model, "best_iteration_", 0)
    early_stopped = hasattr(model, "best_score_") and len(model.best_score_) > 0
    if best_iter is None or best_iter <= 0:
        best_iter = 1 if early_stopped else model.n_estimators
    best_iter = int(max(1, best_iter))

    val_pred = model.predict(valid_X)
    val_pred = np.asarray(val_pred, dtype=float).reshape(-1)
    val_pred = np.where(np.isfinite(val_pred), val_pred, 0.0)
    val_pred = np.maximum(val_pred, 0.0)
    score = rmse_metric(y_valid, val_pred)

    X_all = pd.concat([train_X, valid_X], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

    params = model.get_params()
    params["n_estimators"] = best_iter
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X_all, y_all)

    test_pred = final_model.predict(test_X)
    test_pred = np.asarray(test_pred, dtype=float).reshape(-1)
    test_pred = np.where(np.isfinite(test_pred), test_pred, 0.0)
    test_pred = np.maximum(test_pred, 0.0)

    submission = pd.DataFrame({"id": test_ids.astype(int).values, TARGET: test_pred})
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
