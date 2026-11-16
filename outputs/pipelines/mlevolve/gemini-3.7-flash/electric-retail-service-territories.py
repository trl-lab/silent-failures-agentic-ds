import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb

# 1. Directory and Path Setup
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

print("Loading raw parquet datasets...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = "TOTAL_MWH"
if target_col not in df_train.columns:
    for col in df_train.columns:
        if col.upper() == "TOTAL_MWH":
            target_col = col
            break

print(f"Loaded train shape: {df_train.shape}, test shape: {df_test.shape}")
print(f"Target column: {target_col}")

# 2. Data Cleaning & Sentinel Missing Value Imputation
y_train = df_train[target_col].copy()
train_features = df_train.drop(columns=[target_col]).copy()
test_features = df_test.copy()

train_len = len(train_features)
combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

sentinel_numerics = [-999999, -999999.0, -999, -999.0, -1, -1.0]
sentinel_strings = ["NOT AVAILABLE", "UNKNOWN", "NONE", "NA", "N/A", "NULL", ""]

for col in combined.columns:
    if pd.api.types.is_numeric_dtype(combined[col]):
        combined[col] = combined[col].replace(sentinel_numerics, np.nan)
    elif pd.api.types.is_object_dtype(combined[col]) or pd.api.types.is_string_dtype(
        combined[col]
    ):
        combined[col] = combined[col].astype(str).str.strip()
        combined[col] = combined[col].replace(sentinel_strings, np.nan)

# 3. Dynamic Identification of Column Types and Domain Semantics
numeric_cols = [
    col for col in combined.columns if pd.api.types.is_numeric_dtype(combined[col])
]
categorical_cols = [col for col in combined.columns if col not in numeric_cols]


def find_col(patterns, cols):
    for p in patterns:
        for c in cols:
            if p == c.lower() or p in c.lower():
                return c
    return None


lat_col = find_col(["lat", "latitude", "y", "centroid_y"], numeric_cols)
lon_col = find_col(["lon", "long", "longitude", "x", "centroid_x"], numeric_cols)
area_col = find_col(["shape_area", "area", "sq_miles", "sqmi"], numeric_cols)
len_col = find_col(["shape_leng", "shape_length", "length", "perimeter"], numeric_cols)

res_cust_col = find_col(["residential", "res_cust", "cust_res"], numeric_cols)
com_cust_col = find_col(["commercial", "com_cust", "cust_com"], numeric_cols)
ind_cust_col = find_col(["industrial", "ind_cust", "cust_ind"], numeric_cols)
tot_cust_col = find_col(
    ["customers", "total_cust", "cust_tot", "tot_cust"], numeric_cols
)

state_col = find_col(["state", "st", "state_name"], combined.columns)
type_col = find_col(["type", "utility_type", "ownership"], combined.columns)

# 4. Feature Engineering: Domain Ratios, Geometric Indicators, and Coordinates
if tot_cust_col:
    combined[f"log_{tot_cust_col}"] = np.log1p(
        np.maximum(0, combined[tot_cust_col].fillna(0))
    )

    if res_cust_col:
        combined[f"log_{res_cust_col}"] = np.log1p(
            np.maximum(0, combined[res_cust_col].fillna(0))
        )
        combined["ratio_res_to_tot_cust"] = combined[res_cust_col] / (
            combined[tot_cust_col] + 1.0
        )

    if com_cust_col:
        combined[f"log_{com_cust_col}"] = np.log1p(
            np.maximum(0, combined[com_cust_col].fillna(0))
        )
        combined["ratio_com_to_tot_cust"] = combined[com_cust_col] / (
            combined[tot_cust_col] + 1.0
        )

    if ind_cust_col:
        combined[f"log_{ind_cust_col}"] = np.log1p(
            np.maximum(0, combined[ind_cust_col].fillna(0))
        )
        combined["ratio_ind_to_tot_cust"] = combined[ind_cust_col] / (
            combined[tot_cust_col] + 1.0
        )

    if res_cust_col and com_cust_col:
        combined["ratio_com_to_res_cust"] = (combined[com_cust_col] + 1.0) / (
            combined[res_cust_col] + 1.0
        )
    if res_cust_col and ind_cust_col:
        combined["ratio_ind_to_res_cust"] = (combined[ind_cust_col] + 1.0) / (
            combined[res_cust_col] + 1.0
        )

if area_col:
    combined[f"log_{area_col}"] = np.log1p(np.maximum(0, combined[area_col].fillna(0)))
if len_col:
    combined[f"log_{len_col}"] = np.log1p(np.maximum(0, combined[len_col].fillna(0)))
if area_col and len_col:
    combined["isoperimetric_quotient"] = (
        4 * np.pi * np.maximum(0, combined[area_col].fillna(0))
    ) / (np.maximum(1.0, combined[len_col].fillna(1.0)) ** 2)

if tot_cust_col and area_col:
    combined["customer_density_area"] = combined[tot_cust_col] / (
        combined[area_col] + 1.0
    )
    combined["log_customer_density"] = np.log1p(
        np.maximum(0, combined["customer_density_area"].fillna(0))
    )

if lat_col and lon_col:
    lat_val = combined[lat_col].fillna(combined[lat_col].median())
    lon_val = combined[lon_col].fillna(combined[lon_col].median())
    combined["spatial_dist_origin"] = np.sqrt(lat_val**2 + lon_val**2)
    combined["spatial_rot_45_x"] = (lon_val + lat_val) / np.sqrt(2)
    combined["spatial_rot_45_y"] = (lat_val - lon_val) / np.sqrt(2)

    coords_train = np.column_stack([lat_val.iloc[:train_len], lon_val.iloc[:train_len]])
    coords_all = np.column_stack([lat_val, lon_val])
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(coords_train)
    combined["spatial_cluster"] = kmeans.predict(coords_all)

for cat in categorical_cols:
    combined[cat] = combined[cat].fillna("MISSING").astype(str)
    freq_map = combined.iloc[:train_len][cat].value_counts().to_dict()
    combined[f"{cat}_freq"] = combined[cat].map(freq_map).fillna(1).astype(float)

X_train_df = combined.iloc[:train_len].copy().reset_index(drop=True)
X_test_df = combined.iloc[train_len:].copy().reset_index(drop=True)
y_train_df = y_train.reset_index(drop=True)

# 5. Stratified K-Fold Setup & Leak-Free Out-of-Fold Target Encoding
n_splits = 5
num_bins = 10
binned_y = pd.qcut(y_train_df.rank(method="first"), q=num_bins, labels=False)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = np.zeros(train_len, dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, binned_y)):
    folds[val_idx] = fold

X_train_df["fold"] = folds

target_group_cols = [
    c for c in [state_col, type_col] if c is not None and c in X_train_df.columns
]

for group_col in target_group_cols:
    oof_target_mean = np.zeros(train_len)
    oof_target_std = np.zeros(train_len)

    for fold in range(n_splits):
        tr_mask = (X_train_df["fold"] != fold).values
        va_mask = (X_train_df["fold"] == fold).values

        group_means = (
            y_train_df.loc[tr_mask]
            .groupby(X_train_df.loc[tr_mask, group_col].astype(str), observed=False)
            .mean()
            .to_dict()
        )
        group_stds = (
            y_train_df.loc[tr_mask]
            .groupby(X_train_df.loc[tr_mask, group_col].astype(str), observed=False)
            .std()
            .to_dict()
        )

        global_mean = float(y_train_df.loc[tr_mask].mean())
        global_std = float(y_train_df.loc[tr_mask].std())

        mapped_mean = (
            X_train_df.loc[va_mask, group_col].astype(str).map(group_means).astype(float)
        )
        mapped_std = (
            X_train_df.loc[va_mask, group_col].astype(str).map(group_stds).astype(float)
        )

        oof_target_mean[va_mask] = mapped_mean.fillna(global_mean).values
        oof_target_std[va_mask] = mapped_std.fillna(global_std).values

    X_train_df[f"{group_col}_te_mean"] = oof_target_mean
    X_train_df[f"{group_col}_te_std"] = oof_target_std

    full_means = (
        y_train_df.groupby(X_train_df[group_col].astype(str), observed=False)
        .mean()
        .to_dict()
    )
    full_stds = (
        y_train_df.groupby(X_train_df[group_col].astype(str), observed=False)
        .std()
        .to_dict()
    )
    global_mean_full = float(y_train_df.mean())
    global_std_full = float(y_train_df.std())

    X_test_df[f"{group_col}_te_mean"] = (
        X_test_df[group_col]
        .astype(str)
        .map(full_means)
        .astype(float)
        .fillna(global_mean_full)
    )
    X_test_df[f"{group_col}_te_std"] = (
        X_test_df[group_col]
        .astype(str)
        .map(full_stds)
        .astype(float)
        .fillna(global_std_full)
    )

X_train_df[target_col] = y_train_df
feature_cols = [c for c in X_train_df.columns if c not in [target_col, "fold"]]

# Align categorical dtypes consistently
cat_cols = [
    col
    for col in feature_cols
    if (
        isinstance(X_train_df[col].dtype, pd.CategoricalDtype)
        or X_train_df[col].dtype == "object"
    )
]
for col in cat_cols:
    X_train_df[col] = X_train_df[col].astype(str).astype("category")
    X_test_df[col] = pd.Categorical(
        X_test_df[col].astype(str), categories=X_train_df[col].cat.categories
    )


# 6. Model Training & Evaluation (LightGBM and XGBoost)
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "booster": "gbtree",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "enable_categorical": True,
}

oof_preds_lgb = np.zeros(len(X_train_df))
oof_preds_xgb = np.zeros(len(X_train_df))
test_preds_lgb = np.zeros(len(X_test_df))
test_preds_xgb = np.zeros(len(X_test_df))

print(f"Executing {n_splits}-fold cross-validation...")

for fold in range(n_splits):
    trn_mask = X_train_df["fold"] != fold
    val_mask = X_train_df["fold"] == fold

    X_trn = X_train_df.loc[trn_mask, feature_cols]
    y_trn = X_train_df.loc[trn_mask, target_col].values
    X_val = X_train_df.loc[val_mask, feature_cols]
    y_val = X_train_df.loc[val_mask, target_col].values
    X_tst = X_test_df[feature_cols].copy()

    # LightGBM
    trn_data_lgb = lgb.Dataset(
        X_trn, label=y_trn, categorical_feature=cat_cols, free_raw_data=False
    )
    val_data_lgb = lgb.Dataset(
        X_val,
        label=y_val,
        reference=trn_data_lgb,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    model_lgb = lgb.train(
        lgb_params,
        trn_data_lgb,
        num_boost_round=2500,
        valid_sets=[val_data_lgb],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred_lgb = model_lgb.predict(X_val)
    oof_preds_lgb[val_mask] = val_pred_lgb
    test_preds_lgb += model_lgb.predict(X_tst) / n_splits

    # XGBoost
    dtrn = xgb.DMatrix(X_trn, label=y_trn, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtst = xgb.DMatrix(X_tst, enable_categorical=True)

    model_xgb = xgb.train(
        xgb_params,
        dtrn,
        num_boost_round=2500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred_xgb = model_xgb.predict(dval)
    oof_preds_xgb[val_mask] = val_pred_xgb
    test_preds_xgb += model_xgb.predict(dtst) / n_splits

    fold_lgb_rmse = compute_rmse(y_val, val_pred_lgb)
    fold_xgb_rmse = compute_rmse(y_val, val_pred_xgb)
    fold_blend_rmse = compute_rmse(
        y_val, 0.5 * (np.clip(val_pred_lgb, 0, None) + np.clip(val_pred_xgb, 0, None))
    )
    print(
        f"Fold {fold + 1}/{n_splits} | LGB RMSE: {fold_lgb_rmse:.4f} | XGB RMSE: {fold_xgb_rmse:.4f} | Blend RMSE: {fold_blend_rmse:.4f}"
    )

# 7. Out-of-Fold Optimization & Submission Generation
y_true = X_train_df[target_col].values
oof_preds_lgb = np.clip(oof_preds_lgb, 0, None)
oof_preds_xgb = np.clip(oof_preds_xgb, 0, None)
test_preds_lgb = np.clip(test_preds_lgb, 0, None)
test_preds_xgb = np.clip(test_preds_xgb, 0, None)

best_alpha = 0.5
best_oof_rmse = float("inf")
for alpha in np.linspace(0.0, 1.0, 101):
    blended_oof = alpha * oof_preds_lgb + (1.0 - alpha) * oof_preds_xgb
    current_rmse = compute_rmse(y_true, blended_oof)
    if current_rmse < best_oof_rmse:
        best_oof_rmse = current_rmse
        best_alpha = alpha

oof_final = best_alpha * oof_preds_lgb + (1.0 - best_alpha) * oof_preds_xgb
test_final = best_alpha * test_preds_lgb + (1.0 - best_alpha) * test_preds_xgb

print(f"Optimal Ensemble weights: LGB={best_alpha:.2f}, XGB={1.0 - best_alpha:.2f}")

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_final), dtype=int), "TOTAL_MWH": test_final}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert len(submission_df) == len(
    X_test_df
), f"Expected {len(X_test_df)} rows, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    "TOTAL_MWH",
], f"Incorrect columns: {submission_df.columns}"
assert not submission_df["TOTAL_MWH"].isna().any(), "Submission contains NaN values!"

print(f"Final Validation Score: {best_oof_rmse}")
