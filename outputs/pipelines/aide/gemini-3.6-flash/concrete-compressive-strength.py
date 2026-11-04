import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature engineering incorporating material physics, Feret's law, and Powers' gel-space power functions
def create_features(df):
    data = df.copy()

    c = data["cement"]
    s = data["blast_furnace_slag"]
    fa = data["fly_ash"]
    w = data["water"]
    sp = data["superplasticizer"]
    ca = data["coarse_aggregate"]
    fina = data["fine_aggregate"]
    age = data["age"]

    # Total mixture mass and constituent volumetric fractions (specific gravities)
    total_mass = c + s + fa + w + sp + ca + fina
    v_cement = c / 3.15
    v_slag = s / 2.90
    v_flyash = fa / 2.25
    v_water = w / 1.00
    v_sp = sp / 1.10
    v_coarse = ca / 2.68
    v_fine = fina / 2.64

    v_binder = v_cement + v_slag + v_flyash
    v_paste = v_binder + v_water + v_sp
    v_total = v_paste + v_coarse + v_fine

    # Density and mixture balance features
    data["concrete_density"] = total_mass / (v_total + 1e-6)
    data["fine_aggregate_share"] = fina / (ca + fina + 1e-6)
    data["sp_water_ratio"] = sp / (w + 1e-6)
    data["aggregate_paste_vol_ratio"] = (v_coarse + v_fine) / (v_paste + 1e-6)

    # Total and effective binder calculations (SCM efficiency factors)
    data["total_binder"] = c + s + fa
    data["effective_binder"] = c + 0.65 * s + 0.45 * fa

    # Key water and component ratios
    data["water_binder_ratio"] = w / (data["total_binder"] + 1e-6)
    data["water_effective_binder_ratio"] = w / (data["effective_binder"] + 1e-6)
    data["water_cement_ratio"] = w / (c + 1e-6)
    data["sp_binder_ratio"] = sp / (data["total_binder"] + 1e-6)
    data["slag_binder_ratio"] = s / (data["total_binder"] + 1e-6)
    data["flyash_binder_ratio"] = fa / (data["total_binder"] + 1e-6)

    # Aggregate relations
    data["total_aggregate"] = ca + fina
    data["coarse_fine_ratio"] = ca / (fina + 1e-6)
    data["aggregate_binder_ratio"] = data["total_aggregate"] / (
        data["total_binder"] + 1e-6
    )

    # Volumetric paste fractions
    data["paste_volume_fraction"] = v_paste / (v_total + 1e-6)
    data["water_paste_ratio"] = v_water / (v_paste + 1e-6)

    # Feret's strength efficiency law
    data["feret_ratio"] = (v_binder / (v_binder + v_water + 1e-6)) ** 2

    # Hyperbolic curing age maturity features t / (t + K)
    data["hyperbolic_age_7"] = age / (age + 7.0)
    data["hyperbolic_age_14"] = age / (age + 14.0)
    data["hyperbolic_age_28"] = age / (age + 28.0)

    # Curing age non-linear transformations and interactions
    data["log_age"] = np.log1p(age)
    data["sqrt_age"] = np.sqrt(age)
    data["age_pow025"] = age**0.25
    data["age_wb_ratio"] = data["log_age"] / (data["water_binder_ratio"] + 1e-6)
    data["age_eb_ratio"] = data["log_age"] / (
        data["water_effective_binder_ratio"] + 1e-6
    )

    # Gel-space ratio evolution & Powers' theoretical power-law strength features
    alpha = 1.0 - np.exp(-0.04 * age)
    data["gel_space_ratio"] = (0.68 * alpha * data["total_binder"]) / (
        0.32 * data["total_binder"] + w + 1e-6
    )
    data["gel_space_sq"] = data["gel_space_ratio"] ** 2
    data["powers_gel_space_27"] = data["gel_space_ratio"] ** 2.7
    data["powers_gel_space_30"] = data["gel_space_ratio"] ** 3.0
    data["powers_strength_proxy"] = 140.0 * data["powers_gel_space_27"]

    # SCM presence indicators
    data["has_slag"] = (s > 0).astype(float)
    data["has_flyash"] = (fa > 0).astype(float)
    data["has_sp"] = (sp > 0).astype(float)

    return data


# Apply feature engineering
df_train_fe = create_features(train_df)
df_test_fe = create_features(test_df)

target_col = "strength"
drop_cols = [target_col, "id"] if "id" in df_train_fe.columns else [target_col]
feature_cols = [c for c in df_train_fe.columns if c not in drop_cols]

X = df_train_fe[feature_cols]
y = df_train_fe[target_col]

test_drop = ["id"] if "id" in df_test_fe.columns else []
X_test = df_test_fe[[c for c in df_test_fe.columns if c not in test_drop]]
X_test = X_test[feature_cols]

# 5-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Array to hold out-of-fold predictions for each model family
oof_cat1 = np.zeros(len(X))
oof_cat2 = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

test_cat1 = np.zeros(len(X_test))
test_cat2 = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    # Model 1: CatBoost (depth=6)
    cat1 = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=150,
    )
    cat1.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cat1[val_idx] = cat1.predict(X_va)
    test_cat1 += cat1.predict(X_test) / kf.n_splits

    # Model 2: CatBoost (depth=7)
    cat2 = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.020,
        depth=7,
        l2_leaf_reg=3,
        random_seed=100 + fold,
        verbose=0,
        early_stopping_rounds=150,
    )
    cat2.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cat2[val_idx] = cat2.predict(X_va)
    test_cat2 += cat2.predict(X_test) / kf.n_splits

    # Model 3: LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=200 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test) / kf.n_splits


# Optimize ensemble weights using OOF predictions
def objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = w[0] * oof_cat1 + w[1] * oof_cat2 + w[2] * oof_lgb
    return np.sqrt(mean_squared_error(y, blend_oof))


res = minimize(objective, [1 / 3, 1 / 3, 1 / 3], bounds=[(0, 1)] * 3, method="SLSQP")
best_w = res.x / np.sum(res.x)

final_oof = best_w[0] * oof_cat1 + best_w[1] * oof_cat2 + best_w[2] * oof_lgb
final_test_preds = best_w[0] * test_cat1 + best_w[1] * test_cat2 + best_w[2] * test_lgb

cv_rmse = np.sqrt(mean_squared_error(y, final_oof))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Save submission predictions
os.makedirs("./working", exist_ok=True)
if "id" in test_df.columns:
    sub_id = test_df["id"]
else:
    sub_id = np.arange(len(test_df))

sub = pd.DataFrame({"id": sub_id, "strength": final_test_preds})
sub.to_csv("./working/submission.csv", index=False)
