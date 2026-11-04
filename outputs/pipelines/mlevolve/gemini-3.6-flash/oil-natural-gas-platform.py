import os
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "WATERDEPTH"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if "WATER" in col.upper() and "DEPTH" in col.upper():
            target_col = col
            break

# Identify ID column for submission alignment
id_col = None
for candidate in ["id", "ID", "OBJECTID", "COMPLEX_ID", "STRUCTURE_ID"]:
    if candidate in test_df.columns:
        id_col = candidate
        break

y = train_df[target_col].values.astype(np.float32)
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

if id_col is None:
    test_ids = np.arange(len(X_test_raw))
else:
    test_ids = X_test_raw[id_col].values

# 2. Strict Train/Validation Split (80/20)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)


# 3. Leakage-Free Feature Engineering Pipeline
def feature_engineering(X_tr_in, y_tr_in, X_va_in, X_te_in):
    X_tr = X_tr_in.copy()
    X_va = X_va_in.copy()
    X_te = X_te_in.copy()

    # Identify Spatial Coordinates
    lat_col, lon_col = None, None
    for col in X_tr.columns:
        col_lower = col.lower()
        if "lat" in col_lower:
            lat_col = col
        elif "lon" in col_lower or "lng" in col_lower:
            lon_col = col

    # Spatial coordinate transformations & bathymetric proxies
    if lat_col and lon_col:
        for df in [X_tr, X_va, X_te]:
            lat_rad = np.radians(df[lat_col].fillna(X_tr[lat_col].median()))
            lon_rad = np.radians(df[lon_col].fillna(X_tr[lon_col].median()))

            # 3D Cartesian coordinates
            df["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
            df["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
            df["coord_z"] = np.sin(lat_rad)

            # Rotated axes to capture shelf orientation
            df["rot_45_x"] = df[lat_col] * np.cos(np.pi / 4) - df[lon_col] * np.sin(
                np.pi / 4
            )
            df["rot_45_y"] = df[lat_col] * np.sin(np.pi / 4) + df[lon_col] * np.cos(
                np.pi / 4
            )
            df["rot_30_x"] = df[lat_col] * np.cos(np.pi / 6) - df[lon_col] * np.sin(
                np.pi / 6
            )
            df["rot_60_x"] = df[lat_col] * np.cos(np.pi / 3) - df[lon_col] * np.sin(
                np.pi / 3
            )

            cent_lat = X_tr[lat_col].median()
            cent_lon = X_tr[lon_col].median()
            df["dist_to_center"] = np.sqrt(
                (df[lat_col] - cent_lat) ** 2 + (df[lon_col] - cent_lon) ** 2
            )

        # Spatial K-Nearest Neighbors Bathymetry Features
        coords_tr = X_tr[[lat_col, lon_col]].fillna(0).values
        coords_va = X_va[[lat_col, lon_col]].fillna(0).values
        coords_te = X_te[[lat_col, lon_col]].fillna(0).values

        knn = NearestNeighbors(n_neighbors=10, algorithm="kd_tree")
        knn.fit(coords_tr)

        # Out-of-fold KNN statistics for Train split
        kf_knn = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_mean = np.zeros(len(X_tr))
        oof_std = np.zeros(len(X_tr))
        oof_min = np.zeros(len(X_tr))
        oof_max = np.zeros(len(X_tr))

        for tr_i, val_i in kf_knn.split(coords_tr):
            knn_fold = NearestNeighbors(n_neighbors=10, algorithm="kd_tree")
            knn_fold.fit(coords_tr[tr_i])
            distances, indices = knn_fold.kneighbors(coords_tr[val_i])
            fold_targets = y_tr_in[tr_i][indices]

            oof_mean[val_i] = np.mean(fold_targets, axis=1)
            oof_std[val_i] = np.std(fold_targets, axis=1)
            oof_min[val_i] = np.min(fold_targets, axis=1)
            oof_max[val_i] = np.max(fold_targets, axis=1)

        X_tr["knn_depth_mean"] = oof_mean
        X_tr["knn_depth_std"] = oof_std
        X_tr["knn_depth_min"] = oof_min
        X_tr["knn_depth_max"] = oof_max

        # Transform Validation split
        _, idxs_va = knn.kneighbors(coords_va, n_neighbors=10)
        va_targets = y_tr_in[idxs_va]
        X_va["knn_depth_mean"] = np.mean(va_targets, axis=1)
        X_va["knn_depth_std"] = np.std(va_targets, axis=1)
        X_va["knn_depth_min"] = np.min(va_targets, axis=1)
        X_va["knn_depth_max"] = np.max(va_targets, axis=1)

        # Transform Test split
        _, idxs_te = knn.kneighbors(coords_te, n_neighbors=10)
        te_targets = y_tr_in[idxs_te]
        X_te["knn_depth_mean"] = np.mean(te_targets, axis=1)
        X_te["knn_depth_std"] = np.std(te_targets, axis=1)
        X_te["knn_depth_min"] = np.min(te_targets, axis=1)
        X_te["knn_depth_max"] = np.max(te_targets, axis=1)

    # Date / Time Transformations
    date_cols = [c for c in X_tr.columns if "date" in c.lower() or "year" in c.lower()]
    for col in date_cols:
        for df in [X_tr, X_va, X_te]:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df.drop(columns=[col], inplace=True)
                elif not pd.api.types.is_numeric_dtype(df[col]):
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = parsed.dt.year
                    df[f"{col}_month"] = parsed.dt.month
                    df.drop(columns=[col], inplace=True)

    # Categorical Frequency and Out-Of-Fold Target Encoding
    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = float(np.mean(y_tr_in))

    for c in cat_cols:
        freq = X_tr[c].value_counts(normalize=True).to_dict()
        for df in [X_tr, X_va, X_te]:
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)

        te_mean_arr = np.zeros(len(X_tr))
        for tr_i, val_i in kf_te.split(X_tr):
            grp_means = (
                pd.DataFrame({"grp": X_tr[c].iloc[tr_i], "target": y_tr_in[tr_i]})
                .groupby("grp")["target"]
                .mean()
                .to_dict()
            )
            te_mean_arr[val_i] = (
                X_tr[c].iloc[val_i].map(grp_means).fillna(global_mean).values
            )
        X_tr[f"{c}_te_mean"] = te_mean_arr

        full_grp_means = (
            pd.DataFrame({"grp": X_tr[c], "target": y_tr_in})
            .groupby("grp")["target"]
            .mean()
            .to_dict()
        )
        X_va[f"{c}_te_mean"] = X_va[c].map(full_grp_means).fillna(global_mean)
        X_te[f"{c}_te_mean"] = X_te[c].map(full_grp_means).fillna(global_mean)

        # Label encoding
        le = LabelEncoder()
        combined = pd.concat([X_tr[c], X_va[c], X_te[c]]).astype(str).fillna("MISSING")
        le.fit(combined)
        X_tr[c] = le.transform(X_tr[c].astype(str).fillna("MISSING"))
        X_va[c] = le.transform(X_va[c].astype(str).fillna("MISSING"))
        X_te[c] = le.transform(X_te[c].astype(str).fillna("MISSING"))

    # Numerical missingness and median imputation (fit strictly on train)
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        med = X_tr[col].median()
        if pd.isna(med):
            med = 0.0
        for df in [X_tr, X_va, X_te]:
            if col in df.columns:
                df[f"{col}_isna"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(med)

    return X_tr, X_va, X_te


# 4. Execute Feature Engineering Strategy
X_tr_raw = X_train_raw.iloc[train_idx].reset_index(drop=True)
y_tr = y[train_idx]
X_va_raw = X_train_raw.iloc[val_idx].reset_index(drop=True)
y_va = y[val_idx]
X_te_raw = X_test_raw.copy()

X_tr_proc, X_va_proc, X_te_proc = feature_engineering(
    X_tr_raw, y_tr, X_va_raw, X_te_raw
)

# Ensure Exact Column Alignment
common_cols = [
    c for c in X_tr_proc.columns if c in X_va_proc.columns and c in X_te_proc.columns
]
if id_col and id_col in common_cols:
    common_cols.remove(id_col)

X_tr = X_tr_proc[common_cols].fillna(0)
X_va = X_va_proc[common_cols].fillna(0)
X_te = X_te_proc[common_cols].fillna(0)

# Save Processed Datasets
X_tr_save = X_tr.copy()
X_tr_save["WATERDEPTH"] = y_tr
X_va_save = X_va.copy()
X_va_save["WATERDEPTH"] = y_va

X_tr_save.to_parquet("./working/train_processed.parquet", index=False)
X_va_save.to_parquet("./working/val_processed.parquet", index=False)
X_te.to_parquet("./working/test_processed.parquet", index=False)

# 5. Model Architecture Training and Evaluation

# Model 1: LightGBM Regressor
lgb_model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    n_estimators=3000,
    learning_rate=0.015,
    num_leaves=63,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
val_preds_lgb = lgb_model.predict(X_va)
test_preds_lgb = lgb_model.predict(X_te)
joblib.dump(lgb_model, "./working/models/lgb_model.pkl")

# Model 2: XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=3000,
    learning_rate=0.015,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    tree_method="hist",
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
val_preds_xgb = xgb_model.predict(X_va)
test_preds_xgb = xgb_model.predict(X_te)
joblib.dump(xgb_model, "./working/models/xgb_model.pkl")

# Model 3: CatBoost Regressor
cb_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.015,
    depth=7,
    l2_leaf_reg=4.0,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=50,
    verbose=0,
)
cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
val_preds_cb = cb_model.predict(X_va)
test_preds_cb = cb_model.predict(X_te)
joblib.dump(cb_model, "./working/models/cb_model.pkl")

# Model 4: Ridge Regressor
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_va_sc = scaler.transform(X_va)
X_te_sc = scaler.transform(X_te)

ridge_model = Ridge(alpha=100.0, random_state=42)
ridge_model.fit(X_tr_sc, y_tr)
val_preds_ridge = ridge_model.predict(X_va_sc)
test_preds_ridge = ridge_model.predict(X_te_sc)
joblib.dump(ridge_model, "./working/models/ridge_model.pkl")
joblib.dump(scaler, "./working/models/scaler.pkl")

# 6. SLSQP Optimal Convex Ensemble Weight Optimization
val_preds_matrix = np.column_stack(
    [val_preds_lgb, val_preds_xgb, val_preds_cb, val_preds_ridge]
)
test_preds_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_ridge]
)


def blend_rmse_loss(weights):
    w = weights / np.sum(weights)
    blend_preds = np.dot(val_preds_matrix, w)
    return root_mean_squared_error(y_va, blend_preds)


init_weights = np.ones(val_preds_matrix.shape[1]) / val_preds_matrix.shape[1]
bounds = [(0, 1) for _ in range(val_preds_matrix.shape[1])]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

opt_res = minimize(
    blend_rmse_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x / np.sum(opt_res.x)

# 7. Final Predictions and Physical Non-negativity Constraint (WATERDEPTH >= 0)
final_val_preds = np.clip(np.dot(val_preds_matrix, opt_weights), a_min=0.0, a_max=None)
final_test_preds = np.clip(
    np.dot(test_preds_matrix, opt_weights), a_min=0.0, a_max=None
)

final_score = root_mean_squared_error(y_va, final_val_preds)

# 8. Export Submission File
sub = pd.DataFrame({"id": test_ids, "WATERDEPTH": final_test_preds})
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
