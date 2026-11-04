import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "capacity_mw"
y = train[target_col].values
y_log = np.log1p(np.maximum(0, y))

# Save test IDs
if "id" in test.columns:
    test_ids = test["id"].values
else:
    test_ids = np.arange(len(test))

train_features = train.drop(columns=[target_col], errors="ignore")
test_features = test.drop(columns=[target_col], errors="ignore")


# 2. Feature Engineering
def feature_engineering(tr, te):
    df = pd.concat([tr, te], axis=0, ignore_index=True)

    # Active fuels count
    fuel_cols = [
        c
        for c in ["primary_fuel", "other_fuel1", "other_fuel2", "other_fuel3"]
        if c in df.columns
    ]
    if fuel_cols:
        df["num_active_fuels"] = df[fuel_cols].notna().sum(axis=1)
        df["is_multi_fuel"] = (df["num_active_fuels"] > 1).astype(int)

    # Generation statistics & Physical MW Estimators
    gen_cols = [
        c for c in df.columns if "generation" in c.lower() or "gwh" in c.lower()
    ]
    gen_num_cols = [c for c in gen_cols if pd.api.types.is_numeric_dtype(df[c])]
    if gen_num_cols:
        df["gen_mean"] = df[gen_num_cols].mean(axis=1)
        df["gen_std"] = df[gen_num_cols].std(axis=1)
        df["gen_sum"] = df[gen_num_cols].sum(axis=1)
        df["gen_min"] = df[gen_num_cols].min(axis=1)
        df["gen_max"] = df[gen_num_cols].max(axis=1)
        df["gen_median"] = df[gen_num_cols].median(axis=1)
        df["gen_non_zero_count"] = (df[gen_num_cols] > 0).sum(axis=1)
        df["gen_isna_count"] = df[gen_num_cols].isna().sum(axis=1)

        # Domain Physical Power Capacity Estimates (Mean & Peak equivalent output MW)
        df["est_capacity_mw"] = (df["gen_mean"] * 1000.0) / (8760.0 * 0.45)
        df["est_peak_capacity_mw"] = (df["gen_max"] * 1000.0) / 8760.0

        # Peak-to-mean and variability continuous ratios
        df["gen_peak_to_mean_ratio"] = df["gen_max"] / (df["gen_mean"] + 1e-3)
        df["gen_std_to_mean_ratio"] = df["gen_std"] / (df["gen_mean"] + 1e-3)

        # Explicit Log-1p Transformed Generation Features for Tree Models
        for g_stat in [
            "gen_mean",
            "gen_std",
            "gen_sum",
            "gen_min",
            "gen_max",
            "gen_median",
            "est_capacity_mw",
            "est_peak_capacity_mw",
        ]:
            df[f"log_{g_stat}"] = np.log1p(np.maximum(0, df[g_stat].fillna(0)))

    # Temporal features
    if "commissioning_year" in df.columns:
        df["plant_age"] = 2024 - df["commissioning_year"]
        df["plant_age_isna"] = df["commissioning_year"].isna().astype(int)
        df["commissioning_decade"] = (df["commissioning_year"] // 10) * 10

    # Spatial 3D coordinates and spatial nearest neighbor aggregations
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_rad = np.radians(df["latitude"].fillna(0))
        lon_rad = np.radians(df["longitude"].fillna(0))
        df["spatial_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        df["spatial_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        df["spatial_z"] = np.sin(lat_rad)

        spatial_coords = df[["spatial_x", "spatial_y", "spatial_z"]].values
        nn = NearestNeighbors(n_neighbors=6, algorithm="kd_tree").fit(spatial_coords)
        distances, indices = nn.kneighbors(spatial_coords)

        # Distance to nearest plant
        df["knn_dist_1"] = distances[:, 1]
        df["knn_dist_mean_5"] = distances[:, 1:6].mean(axis=1)

        # Spatial neighbor continuous generation aggregations
        if "log_gen_mean" in df.columns:
            log_gen_vals = df["log_gen_mean"].values
            neighbor_gen_means = np.array(
                [np.nanmean(log_gen_vals[idx[1:6]]) for idx in indices]
            )
            df["spatial_knn_log_gen_mean"] = neighbor_gen_means

    # Categorical Frequency Encoding & Group Stats
    group_cols = [
        c
        for c in ["country", "country_long", "primary_fuel", "owner", "source"]
        if c in df.columns
    ]
    num_agg_cols = [
        c
        for c in [
            "plant_age",
            "gen_mean",
            "gen_sum",
            "log_gen_mean",
            "log_est_peak_capacity_mw",
            "latitude",
            "longitude",
        ]
        if c in df.columns
    ]

    for g_col in group_cols:
        freq = df[g_col].value_counts()
        df[f"{g_col}_freq"] = df[g_col].map(freq)

        for num_c in num_agg_cols:
            grouped = df.groupby(g_col)[num_c]
            df[f"{g_col}_{num_c}_mean"] = grouped.transform("mean")
            df[f"{g_col}_{num_c}_std"] = grouped.transform("std")

    if "country" in df.columns and "primary_fuel" in df.columns:
        df["country_fuel"] = (
            df["country"].astype(str) + "_" + df["primary_fuel"].astype(str)
        )
        df["country_fuel_freq"] = df["country_fuel"].map(
            df["country_fuel"].value_counts()
        )

    # Label encode categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    tr_out = df.iloc[: len(tr)].copy()
    te_out = df.iloc[len(tr) :].copy()
    return tr_out, te_out


X_train, X_test = feature_engineering(train_features, test_features)


# 3. Target Encoding Function
def add_oof_target_encoding(X_tr, X_te, y_target, cat_cols, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = y_target.mean()

    X_tr = X_tr.copy()
    X_te = X_te.copy()

    for col in cat_cols:
        if col not in X_tr.columns:
            continue
        col_te_name = f"{col}_target_enc"
        X_tr[col_te_name] = np.nan
        X_te[col_te_name] = np.nan

        for train_idx, val_idx in kf.split(X_tr):
            tr_cat = X_tr.iloc[train_idx][col]
            tr_y = y_target[train_idx]
            stats = (
                pd.DataFrame({"cat": tr_cat, "y": tr_y})
                .groupby("cat")["y"]
                .agg(["count", "mean"])
            )
            smooth = (stats["count"] * stats["mean"] + 10 * global_mean) / (
                stats["count"] + 10
            )
            X_tr.iloc[val_idx, X_tr.columns.get_loc(col_te_name)] = (
                X_tr.iloc[val_idx][col].map(smooth).fillna(global_mean)
            )

        stats = (
            pd.DataFrame({"cat": X_tr[col], "y": y_target})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + 10 * global_mean) / (
            stats["count"] + 10
        )
        X_te[col_te_name] = X_te[col].map(smooth).fillna(global_mean)

    return X_tr, X_te


cat_enc_candidates = [
    c
    for c in ["country", "primary_fuel", "owner", "country_fuel"]
    if c in X_train.columns
]
X_train, X_test = add_oof_target_encoding(X_train, X_test, y_log, cat_enc_candidates)

# Drop non-numeric columns if any
non_num = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
if non_num:
    X_train = X_train.drop(columns=non_num)
    X_test = X_test.drop(columns=non_num)

# 4. Cross-Validation & Multi-Model Ensembling (LightGBM, CatBoost, XGBoost)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

test_lgb_preds = np.zeros(len(X_test))
test_cat_preds = np.zeros(len(X_test))
test_xgb_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_estimators": 3000,
    "verbose": -1,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "random_seed": 42,
    "iterations": 3000,
    "verbose": 0,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 3000,
    "tree_method": "hist",
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_log[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb_preds += lgb_model.predict(X_test) / n_splits

    # CatBoost
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    oof_cat[val_idx] = cb_model.predict(X_va)
    test_cat_preds += cb_model.predict(X_test) / n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_xgb_preds += xgb_model.predict(X_test) / n_splits


# 5. Optimal Blending and Post-Processing Scale Alignment
def blend_and_scale_objective(weights):
    w1, w2, w3, scale, bias = weights
    w_sum = w1 + w2 + w3 + 1e-8
    w1_n, w2_n, w3_n = w1 / w_sum, w2 / w_sum, w3 / w_sum

    pred_log = w1_n * oof_lgb + w2_n * oof_cat + w3_n * oof_xgb
    pred_orig = np.expm1(pred_log)
    pred_scaled = np.clip(scale * pred_orig + bias, 0, None)
    return np.sqrt(mean_squared_error(y, pred_scaled))


init_params = [0.33, 0.33, 0.33, 1.0, 0.0]
bounds = [(0, 1), (0, 1), (0, 1), (0.5, 2.0), (-50, 50)]

res = minimize(blend_and_scale_objective, init_params, bounds=bounds, method="L-BFGS-B")
w1_opt, w2_opt, w3_opt, scale_opt, bias_opt = res.x
w_sum_opt = w1_opt + w2_opt + w3_opt
w1_opt, w2_opt, w3_opt = (
    w1_opt / w_sum_opt,
    w2_opt / w_sum_opt,
    w3_opt / w_sum_opt,
)

# Compute OOF final score
oof_blend_log = w1_opt * oof_lgb + w2_opt * oof_cat + w3_opt * oof_xgb
oof_final_pred = np.clip(scale_opt * np.expm1(oof_blend_log) + bias_opt, 0, None)
cv_rmse = np.sqrt(mean_squared_error(y, oof_final_pred))

print(f"Validation Metric (RMSE): {cv_rmse:.4f}")

# 6. Generate Submission File
test_blend_log = (
    w1_opt * test_lgb_preds + w2_opt * test_cat_preds + w3_opt * test_xgb_preds
)
test_final_pred = np.clip(scale_opt * np.expm1(test_blend_log) + bias_opt, 0, None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "capacity_mw": test_final_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
