import gc
import math
import os
import joblib
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Set random seeds for exact reproducibility
SEED = 42
np.random.seed(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Setup
# =========================================================================
print("Loading raw parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "DEPSUM"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if col.upper() == "DEPSUM":
            target_col = col
            break

y_train = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Preserve test IDs if present
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df), dtype=int)

X_train_raw["__is_test"] = 0
X_test_raw["__is_test"] = 1

combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# =========================================================================
# 2. Dynamic Feature Classification & Feature Engineering
# =========================================================================
numeric_cols = []
cat_cols = []
text_cols = []

for col in combined_df.columns:
    if col in ["__is_test", "id"]:
        continue
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        nunique = combined_df[col].nunique()
        if nunique < 50 and not np.issubdtype(combined_df[col].dtype, np.floating):
            cat_cols.append(col)
        else:
            numeric_cols.append(col)
    else:
        avg_len = combined_df[col].astype(str).str.len().mean()
        if avg_len > 30 or "NAME" in col.upper() or "ADDR" in col.upper():
            text_cols.append(col)
        cat_cols.append(col)

# Dictionary container for batch concatenation to prevent DataFrame fragmentation
engineered_features = {}

# Spatial & Geographic Engineering
lat_col = next(
    (c for c in combined_df.columns if c.upper() in ["SIMS_LAT", "LATITUDE", "LAT"]),
    None,
)
lon_col = next(
    (
        c
        for c in combined_df.columns
        if c.upper() in ["SIMS_LON", "LONGITUDE", "LON", "LNG"]
    ),
    None,
)

if lat_col and lon_col:
    lat_rad = np.radians(pd.to_numeric(combined_df[lat_col], errors="coerce").fillna(0))
    lon_rad = np.radians(pd.to_numeric(combined_df[lon_col], errors="coerce").fillna(0))
    engineered_features["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    engineered_features["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    engineered_features["geo_z"] = np.sin(lat_rad)

    hubs = {
        "dist_nyc": (40.7128, -74.0060),
        "dist_chi": (41.8781, -87.6298),
        "dist_sf": (37.7749, -122.4194),
        "dist_clt": (35.2271, -80.8431),
    }
    for hub_name, (h_lat, h_lon) in hubs.items():
        h_lat_r, h_lon_r = np.radians(h_lat), np.radians(h_lon)
        dlat = lat_rad - h_lat_r
        dlon = lon_rad - h_lon_r
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_rad) * np.cos(h_lat_r) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        engineered_features[hub_name] = 6371.0 * c

# Financial Columns Identification
asset_col = next(
    (c for c in combined_df.columns if c.upper() in ["ASSET", "ASSETS", "TOTASSET"]),
    None,
)
dep_col = next(
    (c for c in combined_df.columns if c.upper() in ["DEP", "TOTDEP", "DEPOSIT"]), None
)
offices_col = next(
    (c for c in combined_df.columns if c.upper() in ["OFFICES", "NUMOFF", "BRANCHES"]),
    None,
)
cert_col = next(
    (
        c
        for c in combined_df.columns
        if c.upper() in ["CERT", "RSSDID", "BKMO", "INST_ID"]
    ),
    None,
)
county_col = next(
    (c for c in combined_df.columns if c.upper() in ["COUNTY", "CNTYNAME", "FIPS"]),
    None,
)
zip_col = next(
    (c for c in combined_df.columns if c.upper() in ["ZIP", "ZIPCODE", "POSTAL"]), None
)
cbsa_col = next(
    (c for c in combined_df.columns if c.upper() in ["CBSA", "MSABR", "CSA", "METRO"]),
    None,
)
state_col = next(
    (c for c in combined_df.columns if c.upper() in ["STNAME", "STALP", "STATE"]), None
)

if asset_col and offices_col:
    asset_s = pd.to_numeric(combined_df[asset_col], errors="coerce").fillna(0)
    off_s = (
        pd.to_numeric(combined_df[offices_col], errors="coerce").fillna(1).clip(lower=1)
    )
    engineered_features["asset_per_office"] = asset_s / off_s
    engineered_features["log_asset_per_office"] = np.log1p(
        np.maximum(engineered_features["asset_per_office"], 0)
    )

if dep_col and offices_col:
    totdep_s = pd.to_numeric(combined_df[dep_col], errors="coerce").fillna(0)
    off_s = (
        pd.to_numeric(combined_df[offices_col], errors="coerce").fillna(1).clip(lower=1)
    )
    engineered_features["dep_per_office"] = totdep_s / off_s
    engineered_features["log_dep_per_office"] = np.log1p(
        np.maximum(engineered_features["dep_per_office"], 0)
    )

if asset_col and dep_col:
    asset_s = pd.to_numeric(combined_df[asset_col], errors="coerce").fillna(0)
    totdep_s = pd.to_numeric(combined_df[dep_col], errors="coerce").fillna(0)
    engineered_features["dep_to_asset_ratio"] = totdep_s / (asset_s + 1e-5)

# Market Concentration & Institution Density
market_geo_cols = [
    c for c in [county_col, zip_col, cbsa_col, state_col] if c is not None
]
for geo_c in market_geo_cols:
    cnt_series = combined_df.groupby(geo_c)[geo_c].transform("count")
    engineered_features[f"{geo_c}_branch_count"] = cnt_series
    if cert_col:
        unique_banks = combined_df.groupby(geo_c)[cert_col].transform("nunique")
        engineered_features[f"{geo_c}_unique_banks"] = unique_banks
        engineered_features[f"{geo_c}_comp_density"] = unique_banks / (cnt_series + 1e-5)
        # Bank branch market share within geographic market
        pair_cnt = combined_df.groupby([geo_c, cert_col])[cert_col].transform("count")
        engineered_features[f"{geo_c}_{cert_col}_market_share"] = pair_cnt / (cnt_series + 1e-5)

if cert_col:
    engineered_features["bank_total_branches"] = combined_df.groupby(cert_col)[
        cert_col
    ].transform("count")
    if state_col:
        engineered_features["bank_state_count"] = combined_df.groupby(cert_col)[
            state_col
        ].transform("nunique")
    if county_col:
        engineered_features["bank_county_count"] = combined_df.groupby(cert_col)[
            county_col
        ].transform("nunique")

# Hierarchical Group Statistical Aggregations & Within-Market Percentile Ranks
hierarchical_group_cols = [
    c for c in [cert_col, cbsa_col, county_col, state_col] if c is not None and c in combined_df.columns
]
fin_numeric_pairs = [
    (col, name)
    for col, name in [(asset_col, "asset"), (dep_col, "dep"), (offices_col, "offices")]
    if col is not None and col in combined_df.columns
]

for grp in hierarchical_group_cols:
    for num_col, num_name in fin_numeric_pairs:
        val_series = pd.to_numeric(combined_df[num_col], errors="coerce").fillna(0)
        grp_obj = val_series.groupby(combined_df[grp])
        grp_mean = grp_obj.transform("mean")
        grp_std = grp_obj.transform("std").fillna(0)
        grp_min = grp_obj.transform("min")
        grp_max = grp_obj.transform("max")

        engineered_features[f"{grp}_{num_name}_mean"] = grp_mean
        engineered_features[f"{grp}_{num_name}_std"] = grp_std
        engineered_features[f"{grp}_{num_name}_min"] = grp_min
        engineered_features[f"{grp}_{num_name}_max"] = grp_max
        engineered_features[f"{grp}_{num_name}_diff_mean"] = val_series - grp_mean
        engineered_features[f"{grp}_{num_name}_ratio_mean"] = val_series / (grp_mean + 1e-5)
        engineered_features[f"{grp}_{num_name}_pct_rank"] = grp_obj.rank(pct=True).fillna(0.5)

# Text Flag Indicators
for tc in set(text_cols):
    if tc in combined_df.columns:
        s = combined_df[tc].astype(str).fillna("")
        engineered_features[f"{tc}_len"] = s.str.len()
        engineered_features[f"{tc}_words"] = s.str.split().apply(len)
        engineered_features[f"{tc}_has_main"] = s.str.upper().str.contains("MAIN").astype(int)
        engineered_features[f"{tc}_has_branch"] = (
            s.str.upper().str.contains("BRANCH").astype(int)
        )
        engineered_features[f"{tc}_has_bank"] = s.str.upper().str.contains("BANK").astype(int)
        engineered_features[f"{tc}_has_trust"] = s.str.upper().str.contains("TRUST").astype(int)
        engineered_features[f"{tc}_has_nat"] = (
            s.str.upper().str.contains("NATIONAL").astype(int)
        )

# Frequency and Label Encodings
engineered_cat_cols = [c for c in cat_cols if c in combined_df.columns]
for c in engineered_cat_cols:
    freq_map = combined_df[c].astype(str).value_counts(normalize=True).to_dict()
    engineered_features[f"{c}_freq"] = (
        combined_df[c].astype(str).map(freq_map).fillna(0).astype(float)
    )
    le = LabelEncoder()
    engineered_features[f"{c}_le"] = le.fit_transform(combined_df[c].astype(str))

# Single-step concatenation of engineered features to prevent dataframe fragmentation
combined_df = pd.concat([combined_df, pd.DataFrame(engineered_features, index=combined_df.index)], axis=1)

# Separate Train and Test
train_mask = combined_df["__is_test"] == 0
test_mask = combined_df["__is_test"] == 1

train_features = (
    combined_df[train_mask].drop(columns=["__is_test"]).reset_index(drop=True)
)
test_features = (
    combined_df[test_mask].drop(columns=["__is_test"]).reset_index(drop=True)
)

# =========================================================================
# 3. Stratified K-Fold & Leak-Free Out-of-Fold Target Encoding
# =========================================================================
n_splits = 10
num_bins = 10
y_binned = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
folds = list(skf.split(train_features, y_binned))

te_cols = [
    c
    for c in [cert_col, county_col, zip_col, cbsa_col, state_col]
    if c is not None and c in train_features.columns
]
global_mean_target = y_train.mean()
global_mean_log_target = np.log1p(np.maximum(y_train, 0)).mean()
smoothing = 10.0

for col in te_cols:
    train_features[f"{col}_te_mean"] = np.nan
    train_features[f"{col}_te_log_mean"] = np.nan

    for train_idx, val_idx in folds:
        fold_X_tr = train_features.iloc[train_idx]
        fold_y_tr = y_train.iloc[train_idx]
        fold_log_y_tr = np.log1p(np.maximum(fold_y_tr, 0))

        stats = (
            pd.DataFrame(
                {"cat": fold_X_tr[col], "y": fold_y_tr, "log_y": fold_log_y_tr}
            )
            .groupby("cat")
            .agg(["count", "mean"])
        )
        stats.columns = ["cnt", "mean_y", "cnt2", "mean_log_y"]

        smoothed_y = (
            stats["cnt"] * stats["mean_y"] + smoothing * global_mean_target
        ) / (stats["cnt"] + smoothing)
        smoothed_log_y = (
            stats["cnt"] * stats["mean_log_y"] + smoothing * global_mean_log_target
        ) / (stats["cnt"] + smoothing)

        val_cats = train_features.iloc[val_idx][col]
        train_features.loc[val_idx, f"{col}_te_mean"] = (
            val_cats.map(smoothed_y).fillna(global_mean_target).values
        )
        train_features.loc[val_idx, f"{col}_te_log_mean"] = (
            val_cats.map(smoothed_log_y).fillna(global_mean_log_target).values
        )

    full_stats = (
        pd.DataFrame(
            {
                "cat": train_features[col],
                "y": y_train,
                "log_y": np.log1p(np.maximum(y_train, 0)),
            }
        )
        .groupby("cat")
        .agg(["count", "mean"])
    )
    full_stats.columns = ["cnt", "mean_y", "cnt2", "mean_log_y"]
    full_smoothed_y = (
        full_stats["cnt"] * full_stats["mean_y"] + smoothing * global_mean_target
    ) / (full_stats["cnt"] + smoothing)
    full_smoothed_log_y = (
        full_stats["cnt"] * full_stats["mean_log_y"]
        + smoothing * global_mean_log_target
    ) / (full_stats["cnt"] + smoothing)

    test_features[f"{col}_te_mean"] = (
        test_features[col].map(full_smoothed_y).fillna(global_mean_target).values
    )
    test_features[f"{col}_te_log_mean"] = (
        test_features[col]
        .map(full_smoothed_log_y)
        .fillna(global_mean_log_target)
        .values
    )

cols_to_drop = [
    c for c in train_features.columns if train_features[c].dtype == "object"
]
X_train_final = train_features.drop(columns=cols_to_drop)
X_test_final = test_features.drop(columns=cols_to_drop)

for col in X_train_final.columns:
    if col in ["id"]:
        continue
    med_val = X_train_final[col].median()
    if pd.isna(med_val):
        med_val = 0.0
    X_train_final[col] = X_train_final[col].fillna(med_val)
    if col in X_test_final.columns:
        X_test_final[col] = X_test_final[col].fillna(med_val)

ignore_cols = {"id", "cv_fold", "__is_test", "DEPSUM"}
feature_cols = [c for c in X_train_final.columns if c not in ignore_cols]
print(f"Engineered feature set contains {len(feature_cols)} predictor columns.")

X_train_mat = X_train_final[feature_cols].copy()
X_test_mat = X_test_final[feature_cols].copy()
y_train_vec = y_train.values

# =========================================================================
# 4. Model Training & Cross-Validation
# =========================================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "n_estimators": 2500,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "n_estimators": 2500,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.04,
    "iterations": 2500,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "verbose": False,
    "thread_count": -1,
    "early_stopping_rounds": 50,
}

et_params = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 0.7,
    "n_jobs": -1,
    "random_state": SEED,
}

oof_lgb = np.zeros(len(y_train_vec), dtype=np.float64)
oof_xgb = np.zeros(len(y_train_vec), dtype=np.float64)
oof_cat = np.zeros(len(y_train_vec), dtype=np.float64)
oof_et = np.zeros(len(y_train_vec), dtype=np.float64)

test_lgb = np.zeros(len(X_test_mat), dtype=np.float64)
test_xgb = np.zeros(len(X_test_mat), dtype=np.float64)
test_cat = np.zeros(len(X_test_mat), dtype=np.float64)
test_et = np.zeros(len(X_test_mat), dtype=np.float64)

print(f"Beginning {n_splits}-fold cross-validation...")
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_train_mat.iloc[train_idx], y_train_vec[train_idx]
    X_val, y_val = X_train_mat.iloc[val_idx], y_train_vec[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test_mat) / n_splits

    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test_mat) / n_splits

    # CatBoost
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test_mat) / n_splits

    # ExtraTrees
    model_et = ExtraTreesRegressor(**et_params)
    model_et.fit(X_tr, y_tr)
    oof_et[val_idx] = model_et.predict(X_val)
    test_et += model_et.predict(X_test_mat) / n_splits

    f_lgb_rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    f_xgb_rmse = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    f_cat_rmse = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    f_et_rmse = np.sqrt(mean_squared_error(y_val, oof_et[val_idx]))
    print(
        f"Fold {fold_idx + 1}/{n_splits} | LGB: {f_lgb_rmse:.4f} | XGB: {f_xgb_rmse:.4f} | CAT: {f_cat_rmse:.4f} | ET: {f_et_rmse:.4f}"
    )

# =========================================================================
# 5. Out-of-Fold Optimization & Final Evaluation
# =========================================================================
oof_models = [("LightGBM", oof_lgb), ("XGBoost", oof_xgb), ("CatBoost", oof_cat), ("ExtraTrees", oof_et)]
for name, pred in oof_models:
    rmse_val = np.sqrt(mean_squared_error(y_train_vec, pred))
    print(f"Overall OOF {name} RMSE: {rmse_val:.4f}")

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et])


def blend_objective(weights):
    blend = np.dot(oof_matrix, weights)
    blend = np.clip(blend, 0, None)
    return np.sqrt(mean_squared_error(y_train_vec, blend))


init_weights = np.ones(4) / 4.0
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

opt_res = minimize(blend_objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
best_weights = opt_res.x
best_weights = np.maximum(best_weights, 0.0)
best_weights = best_weights / np.sum(best_weights)

print("Optimized Ensemble Weights (LGB, XGB, CAT, ET):", np.round(best_weights, 4))

oof_final = np.dot(oof_matrix, best_weights)
oof_final = np.clip(oof_final, 0, None)
final_score = np.sqrt(mean_squared_error(y_train_vec, oof_final))

# =========================================================================
# 6. Test Inference & Submission Generation
# =========================================================================
test_preds = np.dot(test_matrix, best_weights)
test_preds = np.clip(test_preds, 0, None)

submission_df = pd.DataFrame({"id": test_ids, "DEPSUM": test_preds})
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission successfully written to {submission_path} ({len(submission_df)} rows)."
)

print(f"Final Validation Score: {final_score}")