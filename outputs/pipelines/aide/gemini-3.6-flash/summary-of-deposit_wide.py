import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Load dataset
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "DEPSUM"
y_train = train_df[target_col].values
y_train_log = np.log1p(np.maximum(0, y_train))

# Combine datasets for consistent feature engineering
train_df["is_train"] = 1
test_df["is_train"] = 0
test_df[target_col] = 0.0

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Identify feature types
ignore_cols = [target_col, "is_train", "id"]
feature_cols = [c for c in combined_df.columns if c not in ignore_cols]

cat_cols = []
num_cols = []

for col in feature_cols:
    if (
        combined_df[col].dtype == "object"
        or combined_df[col].dtype.name == "category"
        or combined_df[col].nunique() <= 50
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

# 1. Frequency Encoding for Categoricals
for col in cat_cols:
    freq_map = combined_df[col].value_counts().to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq_map)

# 2. Group Aggregations
group_keys = [
    c
    for c in cat_cols
    if combined_df[c].nunique() > 1 and combined_df[c].nunique() < 5000
][:3]
numeric_stats_cols = [c for c in num_cols if combined_df[c].nunique() > 10][:5]

for g in group_keys:
    for n in numeric_stats_cols:
        grouped = (
            combined_df.groupby(g)[n].agg(["mean", "std", "min", "max"]).reset_index()
        )
        grouped.columns = [g] + [
            f"{g}_{n}_{stat}" for stat in ["mean", "std", "min", "max"]
        ]
        combined_df = combined_df.merge(grouped, on=g, how="left")

# Prepare final feature set
features = [c for c in combined_df.columns if c not in ignore_cols]

# Standardize categorical types
for c in features:
    if combined_df[c].dtype == "object":
        combined_df[c] = combined_df[c].astype("category")

train_features = combined_df[combined_df["is_train"] == 1][features].reset_index(
    drop=True
)
test_features = combined_df[combined_df["is_train"] == 0][features].reset_index(
    drop=True
)

cat_features = [c for c in features if train_features[c].dtype.name == "category"]

# Data copies tailored for CatBoost
train_catboost = train_features.copy()
test_catboost = test_features.copy()
cat_cols_catboost = [c for c in cat_features]
for c in cat_cols_catboost:
    train_catboost[c] = train_catboost[c].astype(str).fillna("missing")
    test_catboost[c] = test_catboost[c].astype(str).fillna("missing")

# 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(train_features))
xgb_oof = np.zeros(len(train_features))
cat_oof = np.zeros(len(train_features))

lgb_test = np.zeros(len(test_features))
xgb_test = np.zeros(len(test_features))
cat_test = np.zeros(len(test_features))

for fold, (trn_idx, val_idx) in enumerate(kf.split(train_features)):
    X_tr, y_tr = train_features.iloc[trn_idx], y_train_log[trn_idx]
    X_va, y_va = train_features.iloc[val_idx], y_train_log[val_idx]

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = lgb_model.predict(X_va)
    lgb_test += lgb_model.predict(test_features) / n_splits

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict(X_va)
    xgb_test += xgb_model.predict(test_features) / n_splits

    # --- CatBoost ---
    cat_tr_x, cat_va_x = train_catboost.iloc[trn_idx], train_catboost.iloc[val_idx]
    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cat_model.fit(
        cat_tr_x,
        y_tr,
        cat_features=cat_cols_catboost,
        eval_set=(cat_va_x, y_va),
        early_stopping_rounds=50,
    )
    cat_oof[val_idx] = cat_model.predict(cat_va_x)
    cat_test += cat_model.predict(test_catboost) / n_splits


# Optimize Ensemble Weights via SciPy SLSQP directly on DEPSUM RMSE
def loss_func(weights):
    weights = weights / np.sum(weights)
    blend_oof_log = weights[0] * lgb_oof + weights[1] * xgb_oof + weights[2] * cat_oof
    blend_oof = np.maximum(0, np.expm1(blend_oof_log))
    return np.sqrt(mean_squared_error(y_train, blend_oof))


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

best_weights = res.x / np.sum(res.x)

# Evaluate Optimized Blend OOF
blend_oof_log = (
    best_weights[0] * lgb_oof + best_weights[1] * xgb_oof + best_weights[2] * cat_oof
)
blend_oof = np.maximum(0, np.expm1(blend_oof_log))
oof_rmse = np.sqrt(mean_squared_error(y_train, blend_oof))
print(f"Validation RMSE: {oof_rmse:.5f}")

# Generate Optimized Test Predictions
blend_test_log = (
    best_weights[0] * lgb_test + best_weights[1] * xgb_test + best_weights[2] * cat_test
)
blend_test = np.maximum(0, np.expm1(blend_test_log))

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(blend_test)), "DEPSUM": blend_test})
submission.to_csv("./working/submission.csv", index=False)
