import os
import warnings
import numpy as np
import pandas as pd

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Separate target and features
target_col = "CAPACITY"
y = train[target_col].values.astype(float)
train_features = train.drop(columns=[target_col])
test_features = test.copy()

if "id" in train_features.columns:
    train_features = train_features.drop(columns=["id"])
if "id" in test_features.columns:
    test_features = test_features.drop(columns=["id"])

n_train = len(train_features)
df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# Missing value indicators
missing_cols = [c for c in df_all.columns if df_all[c].isnull().sum() > 0]
for col in missing_cols:
    df_all[f"{col}_isna"] = df_all[col].isnull().astype(int)

# Spatial coordinate features
lat_cols = [c for c in df_all.columns if "lat" in c.lower()]
lon_cols = [c for c in df_all.columns if "lon" in c.lower() or "lng" in c.lower()]

if lat_cols and lon_cols:
    lat_name = lat_cols[0]
    lon_name = lon_cols[0]
    df_all["coord_sum"] = df_all[lat_name] + df_all[lon_name]
    df_all["coord_diff"] = df_all[lat_name] - df_all[lon_name]
    mean_lat, mean_lon = df_all[lat_name].mean(), df_all[lon_name].mean()
    df_all["dist_center"] = np.sqrt(
        (df_all[lat_name] - mean_lat) ** 2 + (df_all[lon_name] - mean_lon) ** 2
    )
    df_all["polar_angle"] = np.arctan2(
        df_all[lat_name] - mean_lat, df_all[lon_name] - mean_lon
    )

# Identify base categorical columns
base_cat_cols = [
    c
    for c in df_all.columns
    if df_all[c].dtype == "object"
    or df_all[c].dtype.name == "category"
    or df_all[c].dtype == "bool"
]

for c in base_cat_cols:
    df_all[c] = df_all[c].fillna("missing").astype(str)

# Pairwise categorical interaction features
interaction_cat_cols = []
for i in range(len(base_cat_cols)):
    for j in range(i + 1, len(base_cat_cols)):
        col1, col2 = base_cat_cols[i], base_cat_cols[j]
        inter_name = f"{col1}_x_{col2}"
        df_all[inter_name] = df_all[col1] + "_" + df_all[col2]
        interaction_cat_cols.append(inter_name)

all_cat_cols = base_cat_cols + interaction_cat_cols

# Frequency encoding
for c in all_cat_cols:
    freq = df_all[c].value_counts(normalize=True).to_dict()
    df_all[f"{c}_freq"] = df_all[c].map(freq).astype(float)

# Separate train and test sets for target encoding
X_train_proc = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_proc = df_all.iloc[n_train:].copy().reset_index(drop=True)

# Out-of-fold target encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(np.mean(y))
smooth = 10.0

for c in all_cat_cols:
    if X_train_proc[c].nunique() > 1:
        te_train = np.zeros(n_train)
        for train_idx, val_idx in kf.split(X_train_proc):
            tr_cat = X_train_proc.iloc[train_idx][c]
            tr_df = pd.DataFrame({"cat": tr_cat, "y": y[train_idx]})
            stats = tr_df.groupby("cat")["y"].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (
                stats["count"] + smooth
            )
            te_train[val_idx] = (
                X_train_proc.iloc[val_idx][c].map(smoothed).fillna(global_mean).values
            )

        full_df = pd.DataFrame({"cat": X_train_proc[c], "y": y})
        full_stats = full_df.groupby("cat")["y"].agg(["count", "mean"])
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + smooth * global_mean
        ) / (full_stats["count"] + smooth)
        te_test = X_test_proc[c].map(full_smoothed).fillna(global_mean).values

        X_train_proc[f"{c}_te"] = te_train
        X_test_proc[f"{c}_te"] = te_test

# Label Encode base categoricals and drop raw string interaction columns
for c in base_cat_cols:
    le = LabelEncoder()
    combined_vals = pd.concat([X_train_proc[c], X_test_proc[c]], axis=0).astype(str)
    le.fit(combined_vals)
    X_train_proc[c] = le.transform(X_train_proc[c].astype(str))
    X_test_proc[c] = le.transform(X_test_proc[c].astype(str))

X_train_proc = X_train_proc.drop(columns=interaction_cat_cols)
X_test_proc = X_test_proc.drop(columns=interaction_cat_cols)

# Model Training with 5-Fold Cross Validation (LightGBM + CatBoost Ensemble)
oof_lgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
test_lgb = np.zeros(len(X_test_proc))
test_cat = np.zeros(len(X_test_proc))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    # LightGBM
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_test_proc, num_iteration=model_lgb.best_iteration)
        / kf.n_splits
    )

    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.035,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    model_cat.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test_proc) / kf.n_splits

# Blend predictions and apply bound clipping
oof_preds = np.clip(0.5 * oof_lgb + 0.5 * oof_cat, 0, None)
test_preds = np.clip(0.5 * test_lgb + 0.5 * test_cat, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "CAPACITY": test_preds})
sub.to_csv("./working/submission.csv", index=False)
