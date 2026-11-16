import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================================================================
# Configuration and Reproducibility
# =========================================================================
SEED = 42
np.random.seed(SEED)

INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Target Identification
# =========================================================================
train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = None
for col in train_df.columns:
    if col.lower() == "enrollment_ft_teachers_ratio":
        target_col = col
        break

if target_col is None:
    candidates = [
        c for c in train_df.columns if "ratio" in c.lower() or "teacher" in c.lower()
    ]
    target_col = candidates[0] if candidates else "Enrollment_FT_Teachers_Ratio"

y_train = train_df[target_col].values.astype(np.float64)
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Identify ID column if exists
id_col = None
for col in ["id", "ID", "Id", "OBJECTID", "INDEX"]:
    if col in X_test_raw.columns:
        id_col = col
        break


# =========================================================================
# 2. Grade Parsing & Feature Engineering Helpers
# =========================================================================
def parse_grade(val):
    if pd.isna(val) or val is None:
        return np.nan
    val_str = str(val).strip().upper()
    if val_str in ["PK", "PRE-K", "PREK", "N"]:
        return -2.0
    if val_str in ["KG", "K", "KINDER", "KINDERGARTEN"]:
        return 0.0
    if val_str in ["UG", "UNGRADED", "OTHER"]:
        return np.nan
    digits = re.findall(r"\d+", val_str)
    if digits:
        return float(digits[0])
    return np.nan


def clean_feature_names(df):
    clean_cols = []
    for c in df.columns:
        c_clean = re.sub(r"[\[\]<>,:\s]+", "_", str(c))
        clean_cols.append(c_clean)
    df.columns = clean_cols
    return df


def engineer_features(df_train, df_test):
    n_train = len(df_train)
    combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    lat_col = None
    lon_col = None
    for c in combined.columns:
        c_upper = c.upper()
        if c_upper in ["LATITUDE", "LAT", "Y"]:
            lat_col = c
        elif c_upper in ["LONGITUDE", "LON", "X"]:
            lon_col = c

    low_grade_col = None
    high_grade_col = None
    for c in combined.columns:
        c_upper = c.upper()
        if any(
            k in c_upper for k in ["LOW_GRADE", "START_GRADE", "LOWGRADE", "MIN_GRADE"]
        ):
            low_grade_col = c
        elif any(
            k in c_upper for k in ["HIGH_GRADE", "END_GRADE", "HIGHGRADE", "MAX_GRADE"]
        ):
            high_grade_col = c

    features = pd.DataFrame(index=combined.index)

    # Geographic and Spatial Features
    if lat_col is not None and lon_col is not None:
        lat = pd.to_numeric(combined[lat_col], errors="coerce")
        lon = pd.to_numeric(combined[lon_col], errors="coerce")

        features["geo_is_missing"] = (lat.isna() | lon.isna()).astype(np.float32)

        train_lat_median = lat.iloc[:n_train].median()
        train_lon_median = lon.iloc[:n_train].median()
        lat_clean = lat.fillna(train_lat_median)
        lon_clean = lon.fillna(train_lon_median)

        features["latitude"] = lat_clean
        features["longitude"] = lon_clean

        lat_rad = np.radians(lat_clean)
        lon_rad = np.radians(lon_clean)
        features["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        features["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        features["coord_z"] = np.sin(lat_rad)

        geo_train = np.column_stack(
            [lat_clean.iloc[:n_train], lon_clean.iloc[:n_train]]
        )
        geo_all = np.column_stack([lat_clean, lon_clean])

        n_clusters = min(25, max(2, n_train // 50))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=SEED, batch_size=2048, n_init=3
        )
        kmeans.fit(geo_train)

        features["spatial_cluster"] = kmeans.predict(geo_all)
        cluster_distances = kmeans.transform(geo_all)
        features["dist_to_cluster_center"] = cluster_distances.min(axis=1)

        # Spatial K-Nearest Neighbor Density
        n_neighbors_knn = min(31, n_train)
        if n_neighbors_knn > 5:
            nn = NearestNeighbors(
                n_neighbors=n_neighbors_knn, metric="euclidean", n_jobs=-1
            )
            nn.fit(geo_train)
            distances, _ = nn.kneighbors(geo_all)
            features["knn_dist_1"] = (
                distances[:, 1] if distances.shape[1] > 1 else 0.0
            )
            features["knn_dist_5_mean"] = distances[
                :, 1 : min(6, distances.shape[1])
            ].mean(axis=1)
            features["knn_dist_15_mean"] = distances[
                :, 1 : min(16, distances.shape[1])
            ].mean(axis=1)
            features["knn_dist_30_mean"] = distances[
                :, 1 : min(31, distances.shape[1])
            ].mean(axis=1)
            features["spatial_density_5"] = 1.0 / (
                features["knn_dist_5_mean"] + 1e-4
            )
            features["spatial_density_15"] = 1.0 / (
                features["knn_dist_15_mean"] + 1e-4
            )

    # Grade Span Features
    if low_grade_col is not None and high_grade_col is not None:
        low_g = combined[low_grade_col].apply(parse_grade)
        high_g = combined[high_grade_col].apply(parse_grade)

        features["low_grade_num"] = low_g
        features["high_grade_num"] = high_g
        features["grade_span"] = high_g - low_g + 1.0

        features["is_elementary"] = ((low_g <= 1) & (high_g <= 6)).astype(np.float32)
        features["is_middle"] = ((low_g >= 5) & (high_g <= 9)).astype(np.float32)
        features["is_high_school"] = ((low_g >= 8) & (high_g >= 12)).astype(np.float32)
        features["is_k12_full"] = ((low_g <= 0) & (high_g >= 12)).astype(np.float32)
        features["has_prek"] = (low_g < 0).astype(np.float32)

    # Categorical & Numeric Attributes
    cat_cols = []
    num_cols = []

    for c in combined.columns:
        if c in [lat_col, lon_col, low_grade_col, high_grade_col, id_col]:
            continue
        if (
            combined[c].dtype == "object"
            or isinstance(combined[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(combined[c])
        ):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # Text Keyword Flags and Token Statistics
    text_keywords = [
        "charter", "magnet", "academy", "special", "alternative", "virtual",
        "online", "montessori", "christian", "catholic", "tech", "vocational",
        "prep", "preparatory", "center", "elementary", "middle", "high",
        "junior", "senior", "district", "county", "parochial", "baptist",
        "lutheran", "stem", "arts", "science", "early", "learning", "community",
        "international", "public", "private", "collegiate", "traditional", "institute"
    ]

    if cat_cols:
        full_text_series = (
            combined[cat_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        for kw in text_keywords:
            features[f"kw_{kw}"] = full_text_series.str.contains(
                r"\b" + kw + r"\b", regex=True
            ).astype(np.float32)
        features["text_total_chars"] = full_text_series.str.len().astype(np.float32)
        features["text_word_count"] = full_text_series.str.split().apply(len).astype(np.float32)
        features["text_unique_word_count"] = (
            full_text_series.str.split().apply(lambda x: len(set(x))).astype(np.float32)
        )

        for cc in cat_cols:
            cc_lower = cc.lower()
            if any(k in cc_lower for k in ["name", "sch", "desc", "title"]):
                str_s = combined[cc].fillna("").astype(str)
                features[f"{cc}_char_len"] = str_s.str.len().astype(np.float32)
                features[f"{cc}_word_count"] = (
                    str_s.str.split().apply(len).astype(np.float32)
                )

    for nc in num_cols:
        val = pd.to_numeric(combined[nc], errors="coerce")
        features[f"num_{nc}"] = val
        features[f"num_{nc}_isna"] = val.isna().astype(np.float32)
        val_train = val.iloc[:n_train].dropna()
        if len(val_train) > 0 and (val_train >= 0).all():
            features[f"num_{nc}_log1p"] = np.log1p(val.clip(lower=0))

    # Students-per-grade ratio
    enroll_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["enroll", "member", "totenrl", "student", "total"])
    ]
    if enroll_cols and "grade_span" in features.columns:
        for ec in enroll_cols:
            val_ec = pd.to_numeric(combined[ec], errors="coerce").fillna(0)
            span_safe = features["grade_span"].clip(lower=1.0).fillna(1.0)
            features[f"{ec}_per_grade"] = (val_ec / span_safe).astype(np.float32)

    for cc in cat_cols:
        series_str = combined[cc].fillna("MISSING").astype(str).str.strip().str.upper()

        train_freq = series_str.iloc[:n_train].value_counts(normalize=True).to_dict()
        features[f"freq_{cc}"] = (
            series_str.map(train_freq).fillna(0.0).astype(np.float32)
        )

        unique_train_cats = sorted(series_str.iloc[:n_train].unique())
        cat_mapping = {val: idx + 1 for idx, val in enumerate(unique_train_cats)}
        features[f"ord_{cc}"] = series_str.map(cat_mapping).fillna(0).astype(np.int32)
        features[f"cat_{cc}_isna"] = (combined[cc].isna()).astype(np.float32)

    # Hierarchical Group Aggregations
    group_candidates = []
    for cc in cat_cols:
        cc_lower = cc.lower()
        if any(
            k in cc_lower
            for k in [
                "state", "stabbr", "county", "cnty", "city", "zip",
                "type", "level", "locale", "district", "sch_type", "sch_level"
            ]
        ):
            group_candidates.append(cc)
    if "spatial_cluster" in features.columns:
        group_candidates.append("spatial_cluster")

    zip_3digit_series = None
    for cc in cat_cols:
        if "zip" in cc.lower():
            zip_3digit_series = (
                combined[cc].astype(str).str.extract(r"(\d{3})")[0].fillna("MISSING")
            )
            group_candidates.append("zip_3digit")
            train_freq_z = (
                zip_3digit_series.iloc[:n_train].value_counts(normalize=True).to_dict()
            )
            features["freq_zip_3digit"] = (
                zip_3digit_series.map(train_freq_z).fillna(0.0).astype(np.float32)
            )
            unique_train_z = sorted(zip_3digit_series.iloc[:n_train].unique())
            cat_map_z = {val: idx + 1 for idx, val in enumerate(unique_train_z)}
            features["ord_zip_3digit"] = (
                zip_3digit_series.map(cat_map_z).fillna(0).astype(np.int32)
            )
            break

    agg_num_cols = [
        c
        for c in ["grade_span", "low_grade_num", "high_grade_num"]
        if c in features.columns
    ]
    agg_num_cols += [
        f"num_{nc}" for nc in num_cols if f"num_{nc}" in features.columns
    ][:4]

    for grp_col in set(group_candidates):
        if grp_col == "zip_3digit" and zip_3digit_series is not None:
            grp_series = zip_3digit_series
        elif grp_col in combined.columns:
            grp_series = combined[grp_col].fillna("MISSING").astype(str)
        elif grp_col in features.columns:
            grp_series = features[grp_col].fillna("MISSING").astype(str)
        else:
            continue

        n_unique_grp = grp_series.iloc[:n_train].nunique()
        if 2 <= n_unique_grp <= 3000:
            for num_c in agg_num_cols:
                train_sub = features.iloc[:n_train][[num_c]].copy()
                train_sub["_grp"] = grp_series.iloc[:n_train]
                mean_map = train_sub.groupby("_grp")[num_c].mean().to_dict()
                std_map = train_sub.groupby("_grp")[num_c].std().fillna(0.0).to_dict()
                global_m = float(train_sub[num_c].mean())

                grp_mean = grp_series.map(mean_map).fillna(global_m)
                grp_std = grp_series.map(std_map).fillna(0.0)
                features[f"{num_c}_by_{grp_col}_diff"] = (
                    features[num_c] - grp_mean
                ).astype(np.float32)
                features[f"{num_c}_by_{grp_col}_ratio"] = (
                    (features[num_c] + 1e-3) / (grp_mean + 1e-3)
                ).astype(np.float32)
                features[f"{num_c}_by_{grp_col}_std"] = grp_std.astype(np.float32)

    # Missingness Profile Features
    features["row_missing_count"] = combined.isna().sum(axis=1).astype(np.float32)
    features["row_missing_ratio"] = features["row_missing_count"] / float(
        max(1, combined.shape[1])
    )

    # Ensure only numeric/boolean features are kept in the modeling matrix
    non_num_cols = [
        c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])
    ]
    if non_num_cols:
        features = features.drop(columns=non_num_cols)

    X_tr = clean_feature_names(features.iloc[:n_train].copy().reset_index(drop=True))
    X_te = clean_feature_names(features.iloc[n_train:].copy().reset_index(drop=True))

    raw_cat_df = combined[cat_cols].copy() if cat_cols else pd.DataFrame(index=combined.index)
    if zip_3digit_series is not None:
        raw_cat_df["zip_3digit"] = zip_3digit_series
    if "spatial_cluster" in features.columns:
        raw_cat_df["spatial_cluster"] = features["spatial_cluster"].astype(str)

    # Add interaction categories for target encoding
    state_cols = [c for c in cat_cols if any(k in c.lower() for k in ["state", "stabbr"])]
    type_cols = [c for c in cat_cols if any(k in c.lower() for k in ["type", "level", "locale"])]
    if state_cols and type_cols:
        for sc in state_cols[:1]:
            for tc in type_cols[:2]:
                raw_cat_df[f"{sc}_{tc}"] = (
                    combined[sc].fillna("MISSING").astype(str)
                    + "_"
                    + combined[tc].fillna("MISSING").astype(str)
                )

    raw_cats_train = raw_cat_df.iloc[:n_train].copy().reset_index(drop=True)
    raw_cats_test = raw_cat_df.iloc[n_train:].copy().reset_index(drop=True)

    return X_tr, X_te, raw_cats_train, raw_cats_test


X_train_proc, X_test_proc, raw_cats_train, raw_cats_test = engineer_features(
    X_train_raw, X_test_raw
)

# =========================================================================
# 3. Stratified K-Fold Split
# =========================================================================
n_splits = 5
folds = np.zeros(len(X_train_proc), dtype=int)
n_bins = 10
target_bins = pd.qcut(y_train, q=n_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, target_bins)):
    folds[val_idx] = fold


# =========================================================================
# 4. Model Architectures & Parameter Configurations
# =========================================================================
def get_lightgbm_params(seed: int = 42):
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 8,
        "min_child_samples": 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(seed: int = 42):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42):
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 5. Model Training & Out-of-Fold Evaluation Loop
# =========================================================================
n_samples = len(X_train_proc)
n_test = len(X_test_proc)

oof_lgb = np.zeros(n_samples, dtype=np.float64)
oof_xgb = np.zeros(n_samples, dtype=np.float64)
oof_cb = np.zeros(n_samples, dtype=np.float64)
oof_et = np.zeros(n_samples, dtype=np.float64)

test_lgb = np.zeros(n_test, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)
test_cb = np.zeros(n_test, dtype=np.float64)
test_et = np.zeros(n_test, dtype=np.float64)

te_candidates = [
    c for c in raw_cats_train.columns
    if 2 <= raw_cats_train[c].nunique() <= 3000
]

for fold in range(n_splits):
    train_mask = folds != fold
    val_mask = folds == fold

    y_tr = y_train[train_mask]
    y_va = y_train[val_mask]

    X_tr_f = X_train_proc.iloc[train_mask].copy().reset_index(drop=True)
    X_va_f = X_train_proc.iloc[val_mask].copy().reset_index(drop=True)
    X_te_f = X_test_proc.copy().reset_index(drop=True)

    # Fold-isolated Target Encoding with Bayesian Smoothing
    global_mean = float(np.mean(y_tr))
    smoothing = 10.0

    for col in te_candidates:
        series_tr = raw_cats_train.iloc[train_mask][col].fillna("MISSING").astype(str).reset_index(drop=True)
        series_va = raw_cats_train.iloc[val_mask][col].fillna("MISSING").astype(str).reset_index(drop=True)
        series_te = raw_cats_test[col].fillna("MISSING").astype(str).reset_index(drop=True)

        stats = (
            pd.DataFrame({"cat": series_tr, "target": y_tr})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth_val = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        te_map = smooth_val.to_dict()

        te_col_name = f"te_{col}"
        X_tr_f[te_col_name] = series_tr.map(te_map).fillna(global_mean).astype(np.float32)
        X_va_f[te_col_name] = series_va.map(te_map).fillna(global_mean).astype(np.float32)
        X_te_f[te_col_name] = series_te.map(te_map).fillna(global_mean).astype(np.float32)

    X_tr_f = clean_feature_names(X_tr_f)
    X_va_f = clean_feature_names(X_va_f)
    X_te_f = clean_feature_names(X_te_f)

    # LightGBM
    lgb_params = get_lightgbm_params(seed=SEED + fold)
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_f,
        y_tr,
        eval_set=[(X_va_f, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred_lgb = model_lgb.predict(X_va_f)
    oof_lgb[val_mask] = val_pred_lgb
    test_lgb += model_lgb.predict(X_te_f) / n_splits

    # XGBoost
    xgb_params = get_xgboost_params(seed=SEED + fold)
    xgb_params["early_stopping_rounds"] = 50
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr_f, y_tr, eval_set=[(X_va_f, y_va)], verbose=False)
    val_pred_xgb = model_xgb.predict(X_va_f)
    oof_xgb[val_mask] = val_pred_xgb
    test_xgb += model_xgb.predict(X_te_f) / n_splits

    # CatBoost
    if CATBOOST_AVAILABLE:
        cb_params = get_catboost_params(seed=SEED + fold)
        model_cb = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        model_cb.fit(X_tr_f, y_tr, eval_set=(X_va_f, y_va), verbose=False)
        val_pred_cb = model_cb.predict(X_va_f)
        oof_cb[val_mask] = val_pred_cb
        test_cb += model_cb.predict(X_te_f) / n_splits

    # ExtraTrees Regressor
    med = X_tr_f.median()
    X_tr_et = X_tr_f.fillna(med).fillna(0)
    X_va_et = X_va_f.fillna(med).fillna(0)
    X_te_et = X_te_f.fillna(med).fillna(0)

    model_et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    model_et.fit(X_tr_et, y_tr)
    val_pred_et = model_et.predict(X_va_et)
    oof_et[val_mask] = val_pred_et
    test_et += model_et.predict(X_te_et) / n_splits

# =========================================================================
# 6. Constrained Ensemble Blending Optimization
# =========================================================================
models_oof = [oof_lgb, oof_xgb, oof_et]
models_test = [test_lgb, test_xgb, test_et]
if CATBOOST_AVAILABLE:
    models_oof.append(oof_cb)
    models_test.append(test_cb)

oof_matrix = np.column_stack(models_oof)
test_matrix = np.column_stack(models_test)
n_models = oof_matrix.shape[1]


def ensemble_loss(weights):
    w = np.array(weights)
    s = np.sum(w)
    w_norm = (w / s) if s > 0 else (np.ones_like(w) / len(w))
    pred = np.dot(oof_matrix, w_norm)
    return np.sqrt(mean_squared_error(y_train, pred))


init_weights = np.ones(n_models) / n_models
bounds = [(0, 1) for _ in range(n_models)]
opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)

optimal_weights = (
    (opt_res.x / np.sum(opt_res.x))
    if (opt_res.success and np.sum(opt_res.x) > 0)
    else init_weights
)

oof_final_pred = np.clip(np.dot(oof_matrix, optimal_weights), a_min=0, a_max=None)
test_final_pred = np.clip(np.dot(test_matrix, optimal_weights), a_min=0, a_max=None)

score = float(np.sqrt(mean_squared_error(y_train, oof_final_pred)))

# =========================================================================
# 7. Write Submission File & Report Final Score
# =========================================================================
submission_df = pd.DataFrame(
    {
        "id": np.arange(n_test, dtype=int),
        "Enrollment_FT_Teachers_Ratio": test_final_pred,
    }
)
submission_df.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")