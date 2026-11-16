import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "heating_load"
feature_cols = [
    col for col in test_df.columns if col in train_df.columns and col != target_col
]
if "cooling_load" in feature_cols:
    feature_cols.remove("cooling_load")


# 2. Domain-Specific Feature Engineering
def create_features(df):
    data = df.copy()

    # Physical Building Geometry & Volume
    if "roof_area" in data.columns and "overall_height" in data.columns:
        data["building_volume"] = data["roof_area"] * data["overall_height"]

    if "surface_area" in data.columns and "building_volume" in data.columns:
        data["surface_to_volume"] = data["surface_area"] / (
            data["building_volume"] + 1e-5
        )

    if "wall_area" in data.columns and "roof_area" in data.columns:
        data["wall_to_roof_ratio"] = data["wall_area"] / (data["roof_area"] + 1e-5)

    if "glazing_area" in data.columns and "wall_area" in data.columns:
        data["effective_glazing"] = data["glazing_area"] * data["wall_area"]
        data["opaque_wall_area"] = data["wall_area"] - data["effective_glazing"]

    if "glazing_area" in data.columns and "surface_area" in data.columns:
        data["glazing_to_surface"] = data["glazing_area"] / (
            data["surface_area"] + 1e-5
        )

    if "relative_compactness" in data.columns and "overall_height" in data.columns:
        data["compactness_height"] = (
            data["relative_compactness"] * data["overall_height"]
        )

    # Cyclical orientation encoding (2=North, 3=East, 4=South, 5=West -> 90 deg steps)
    if "orientation" in data.columns:
        # Map orientations 2,3,4,5 to angles in radians [0, pi/2, pi, 3pi/2]
        angle = (data["orientation"] - 2) * (np.pi / 2.0)
        data["orient_sin"] = np.sin(angle)
        data["orient_cos"] = np.cos(angle)

    return data


X_train_raw = train_df[feature_cols]
y_train = train_df[target_col].values
X_test_raw = test_df[feature_cols]

X_train = create_features(X_train_raw)
X_test = create_features(X_test_raw)

# 3. Stratified/K-Fold Cross-Validation Setup with Multi-Seeding
SEEDS = [42, 1337, 2024]
N_SPLITS = 5

oof_cat = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))

test_cat = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_et = np.zeros(len(test_df))

total_runs = len(SEEDS) * N_SPLITS

for seed in SEEDS:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        # 1. CatBoost Regressor
        cat_model = CatBoostRegressor(
            iterations=1600,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=2.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed + fold,
            verbose=False,
        )
        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=60,
            verbose=False,
        )
        oof_cat[val_idx] += cat_model.predict(X_val) / len(SEEDS)
        test_cat += cat_model.predict(X_test) / total_runs

        # 2. LightGBM Regressor
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": seed + fold,
            "verbose": -1,
            "n_estimators": 1600,
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
        oof_lgb[val_idx] += lgb_model.predict(X_val) / len(SEEDS)
        test_lgb += lgb_model.predict(X_test) / total_runs

        # 3. XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=1600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            early_stopping_rounds=60,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_xgb[val_idx] += xgb_model.predict(X_val) / len(SEEDS)
        test_xgb += xgb_model.predict(X_test) / total_runs

        # 4. ExtraTrees Regressor
        et_model = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        oof_et[val_idx] += et_model.predict(X_val) / len(SEEDS)
        test_et += et_model.predict(X_test) / total_runs

# Compute individual model OOF RMSEs
print(f"CatBoost OOF RMSE: {root_mean_squared_error(y_train, oof_cat):.4f}")
print(f"LightGBM OOF RMSE: {root_mean_squared_error(y_train, oof_lgb):.4f}")
print(f"XGBoost  OOF RMSE: {root_mean_squared_error(y_train, oof_xgb):.4f}")
print(f"ExtraTrees OOF RMSE: {root_mean_squared_error(y_train, oof_et):.4f}")

# 4. Find Optimal Ensemble Blending Weights
OOF_STACK = np.column_stack([oof_cat, oof_lgb, oof_xgb, oof_et])
TEST_STACK = np.column_stack([test_cat, test_lgb, test_xgb, test_et])


def objective(w):
    pred = np.dot(OOF_STACK, w)
    return root_mean_squared_error(y_train, pred)


opt_res = minimize(
    objective,
    x0=[0.25, 0.25, 0.25, 0.25],
    bounds=[(0, 1)] * 4,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    method="SLSQP",
)

weights = opt_res.x
oof_ensemble = np.dot(OOF_STACK, weights)
cv_rmse = root_mean_squared_error(y_train, oof_ensemble)

print(
    f"Ensemble Weights -> CatBoost: {weights[0]:.4f}, LightGBM: {weights[1]:.4f}, XGBoost: {weights[2]:.4f}, ExtraTrees: {weights[3]:.4f}"
)
print(f"5-Fold Cross-Validation Ensemble RMSE: {cv_rmse:.4f}")

# 5. Generate and Save Test Predictions
test_predictions = np.dot(TEST_STACK, weights)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "heating_load": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")
