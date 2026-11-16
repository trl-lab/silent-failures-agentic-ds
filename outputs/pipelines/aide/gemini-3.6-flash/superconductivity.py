import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load input data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "critical_temp"
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()


def build_domain_features(df):
    data = df.copy()
    properties = [
        "atomic_mass",
        "fie",
        "atomic_radius",
        "Density",
        "ElectronAffinity",
        "FusionHeat",
        "ThermalConductivity",
        "Valence",
    ]

    cv_cols = []
    for p in properties:
        data[f"diff_wtd_mean_{p}"] = data[f"wtd_mean_{p}"] - data[f"mean_{p}"]
        data[f"ratio_wtd_mean_{p}"] = data[f"wtd_mean_{p}"] / (data[f"mean_{p}"] + 1e-8)

        cv = data[f"std_{p}"] / (data[f"mean_{p}"] + 1e-8)
        data[f"cv_{p}"] = cv
        cv_cols.append(f"cv_{p}")

        data[f"wtd_cv_{p}"] = data[f"wtd_std_{p}"] / (data[f"wtd_mean_{p}"] + 1e-8)
        data[f"ratio_gmean_{p}"] = data[f"gmean_{p}"] / (data[f"mean_{p}"] + 1e-8)
        data[f"ratio_wtd_gmean_{p}"] = data[f"wtd_gmean_{p}"] / (
            data[f"wtd_mean_{p}"] + 1e-8
        )
        data[f"diff_entropy_{p}"] = data[f"wtd_entropy_{p}"] - data[f"entropy_{p}"]
        data[f"spread_ratio_{p}"] = data[f"range_{p}"] / (data[f"std_{p}"] + 1e-8)

    for p in ["ThermalConductivity", "Density", "FusionHeat"]:
        data[f"log_{p}"] = np.log1p(np.maximum(data[f"mean_{p}"], 0))
        data[f"log_wtd_{p}"] = np.log1p(np.maximum(data[f"wtd_mean_{p}"], 0))

    data["density_thermal"] = data["mean_Density"] * data["mean_ThermalConductivity"]
    data["ea_fie_ratio"] = data["mean_ElectronAffinity"] / (data["mean_fie"] + 1e-8)
    data["valence_elem"] = data["mean_Valence"] * data["number_of_elements"]
    data["wtd_valence_elem"] = data["wtd_mean_Valence"] * data["number_of_elements"]

    entropy_cols = [f"entropy_{p}" for p in properties]
    data["max_entropy"] = data[entropy_cols].max(axis=1)
    data["mean_entropy"] = data[entropy_cols].mean(axis=1)
    data["max_cv"] = data[cv_cols].max(axis=1)
    data["mean_cv"] = data[cv_cols].mean(axis=1)

    return data


X_feat = build_domain_features(X)
X_test_feat = build_domain_features(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_et = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}

et_params = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_feat, y)):
    X_tr, y_tr = X_feat.iloc[train_idx], y[train_idx]
    X_va, y_va = X_feat.iloc[val_idx], y[val_idx]

    # 1. LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test_feat) / 5.0

    # 2. XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test_feat) / 5.0

    # 3. CatBoost
    model_cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test_feat) / 5.0

    # 4. ExtraTrees
    model_et = ExtraTreesRegressor(**et_params)
    model_et.fit(X_tr, y_tr)
    oof_et[val_idx] = model_et.predict(X_va)
    test_et += model_et.predict(X_test_feat) / 5.0

# Non-negative lower bound clipping on single model predictions
oof_lgb = np.clip(oof_lgb, 0, None)
oof_xgb = np.clip(oof_xgb, 0, None)
oof_cat = np.clip(oof_cat, 0, None)
oof_et = np.clip(oof_et, 0, None)

test_lgb = np.clip(test_lgb, 0, None)
test_xgb = np.clip(test_xgb, 0, None)
test_cat = np.clip(test_cat, 0, None)
test_et = np.clip(test_et, 0, None)

# Meta-learner stacking with multi-algorithm diversity
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et])

meta_model = Ridge(alpha=1.0, positive=True)
meta_model.fit(oof_matrix, y)

oof_ensemble = meta_model.predict(oof_matrix)
oof_ensemble = np.clip(oof_ensemble, 0, None)

test_ensemble = meta_model.predict(test_matrix)
test_ensemble = np.clip(test_ensemble, 0, None)

ensemble_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Validation Metric (5-Fold Stacked Ensemble RMSE): {ensemble_rmse:.4f}")

# Save test predictions to working directory
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "critical_temp": test_ensemble}
)
submission.to_csv("./working/submission.csv", index=False)
