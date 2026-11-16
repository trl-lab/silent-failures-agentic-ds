import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "CAPACITY"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y = train_df[target_col].values.astype(np.float32)
X = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors="ignore").copy()

n_train = len(X)
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# 2. Spatial & Datetime Feature Engineering
lat_cols = [c for c in combined.columns if c.upper() in ["LATITUDE", "LAT", "Y"]]
lon_cols = [c for c in combined.columns if c.upper() in ["LONGITUDE", "LON", "X"]]

if lat_cols and lon_cols:
    lat_col, lon_col = lat_cols[0], lon_cols[0]
    lat_s = pd.to_numeric(combined[lat_col], errors="coerce").fillna(0)
    lon_s = pd.to_numeric(combined[lon_col], errors="coerce").fillna(0)

    mean_lat, mean_lon = lat_s.mean(), lon_s.mean()
    combined["dist_to_center"] = np.sqrt(
        (lat_s - mean_lat) ** 2 + (lon_s - mean_lon) ** 2
    )

    # Coordinate rotations
    for angle in [30, 45, 60]:
        rad = np.radians(angle)
        combined[f"rot_{angle}_x"] = lat_s * np.cos(rad) + lon_s * np.sin(rad)
        combined[f"rot_{angle}_y"] = -lat_s * np.sin(rad) + lon_s * np.cos(rad)

# Datetime and Categorical handling
cat_cols = []
for col in list(combined.columns):
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        combined[f"{col}_year"] = combined[col].dt.year
        combined[f"{col}_month"] = combined[col].dt.month
        combined[f"{col}_day"] = combined[col].dt.day
        combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
        combined = combined.drop(columns=[col])
    elif combined[col].dtype == "object" or pd.api.types.is_categorical_dtype(
        combined[col]
    ):
        combined[col] = combined[col].astype(str).fillna("missing")
        # Frequency encoding
        freq = combined[col].value_counts()
        combined[f"{col}_freq"] = combined[col].map(freq).astype(np.float32)
        combined[col] = combined[col].astype("category")
        cat_cols.append(col)

# 3. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_processed = combined.iloc[:n_train].copy()
X_test_processed = combined.iloc[n_train:].copy()

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_preds_lgb = np.zeros(len(X_test_processed))
test_preds_xgb = np.zeros(len(X_test_processed))
test_preds_cat = np.zeros(len(X_test_processed))

# Prepare numeric/category encoded frames for CatBoost and XGBoost
X_catboost = X_processed.copy()
X_test_catboost = X_test_processed.copy()
for col in cat_cols:
    X_catboost[col] = X_catboost[col].astype(str)
    X_test_catboost[col] = X_test_catboost[col].astype(str)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
    # 3.1 LightGBM
    X_tr_lgb, y_tr = X_processed.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_processed.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    val_data = lgb.Dataset(X_va_lgb, label=y_va, reference=trn_data)

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(
        X_va_lgb, num_iteration=model_lgb.best_iteration
    )
    test_preds_lgb += (
        model_lgb.predict(X_test_processed, num_iteration=model_lgb.best_iteration)
        / kf.n_splits
    )

    # 3.2 XGBoost
    model_xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va_lgb)
    test_preds_xgb += model_xgb.predict(X_test_processed) / kf.n_splits

    # 3.3 CatBoost
    X_tr_cat = X_catboost.iloc[train_idx]
    X_va_cat = X_catboost.iloc[val_idx]

    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=False,
    )
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_va_cat, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_va_cat)
    test_preds_cat += model_cat.predict(X_test_catboost) / kf.n_splits

# 4. Optimal Blending Optimization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat])


def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    pred = oof_matrix @ weights
    return root_mean_squared_error(y, pred)


initial_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

opt_res = minimize(
    loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x / np.sum(opt_res.x)

final_oof_preds = oof_matrix @ best_weights
cv_rmse = root_mean_squared_error(y, final_oof_preds)

print(f"Optimal Blend Weights (LGB, XGB, CAT): {best_weights.round(4)}")
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# 5. Final Submission
final_test_preds = np.maximum(0, test_matrix @ best_weights)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "CAPACITY": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
