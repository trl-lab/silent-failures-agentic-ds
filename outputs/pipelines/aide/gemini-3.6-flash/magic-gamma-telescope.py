import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare Target Variable
target_col = "class"
if train_df[target_col].dtype == object or isinstance(
    train_df[target_col].iloc[0], str
):
    target_map = {
        "g": 1,
        "h": 0,
        "gamma": 1,
        "hadron": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
    }
    y = (
        train_df[target_col]
        .map(lambda x: target_map.get(x, 1 if str(x).lower().startswith("g") else 0))
        .values
    )
else:
    y = train_df[target_col].astype(int).values

train_raw = train_df.drop(columns=[target_col], errors="ignore")
test_raw = (
    test_df.drop(columns=[target_col, "id"], errors="ignore")
    if "id" in test_df.columns
    else test_df.drop(columns=[target_col], errors="ignore")
)


# 2. Physics-Informed Domain & Structural Feature Engineering
def create_features(df):
    data = df.copy()
    eps = 1e-6

    # Geometry & Aspect Metrics
    data["aspect_ratio"] = data["fLength"] / (data["fWidth"] + eps)
    data["ellipse_area"] = np.pi * data["fLength"] * data["fWidth"] / 4.0
    ratio_sq = np.clip((data["fWidth"] / (data["fLength"] + eps)) ** 2, 0, 1)
    data["eccentricity"] = np.sqrt(1.0 - ratio_sq)

    # Cherenkov Energy & Light Density Metrics
    data["light_density"] = data["fSize"] / (data["ellipse_area"] + eps)
    data["total_light_est"] = np.power(10.0, np.clip(data["fSize"], 0, 10))
    data["photon_density"] = data["total_light_est"] / (data["ellipse_area"] + eps)

    # Concentration Interactions
    data["conc_diff"] = data["fConc"] - data["fConc1"]
    data["conc_ratio"] = data["fConc"] / (data["fConc1"] + eps)
    data["conc_sum"] = data["fConc"] + data["fConc1"]

    # Asymmetry & Spatial Distance Transformations
    data["asym_rel_length"] = data["fAsym"] / (data["fLength"] + eps)
    data["dist_rel_length"] = data["fDist"] / (data["fLength"] + eps)
    data["dist_rel_width"] = data["fDist"] / (data["fWidth"] + eps)
    data["asym_dist_prod"] = data["fAsym"] * data["fDist"]

    # Third Moment Ratios & Interactions
    data["m3_ratio"] = data["fM3Long"] / (np.abs(data["fM3Trans"]) + eps)
    data["m3_long_length"] = data["fM3Long"] * data["fLength"]
    data["m3_trans_width"] = data["fM3Trans"] * data["fWidth"]
    data["m3_abs_sum"] = np.abs(data["fM3Long"]) + np.abs(data["fM3Trans"])

    # Angular Projections
    alpha_rad = data["fAlpha"] * np.pi / 180.0
    data["sin_alpha"] = np.sin(alpha_rad)
    data["cos_alpha"] = np.cos(alpha_rad)
    data["abs_sin_alpha"] = np.abs(np.sin(alpha_rad))
    data["proj_dist_sin"] = data["fDist"] * data["sin_alpha"]
    data["proj_dist_cos"] = data["fDist"] * data["cos_alpha"]
    data["proj_length_sin"] = data["fLength"] * data["sin_alpha"]

    # Log Transformations of Non-negative Quantities
    data["log_fLength"] = np.log1p(np.maximum(0, data["fLength"]))
    data["log_fWidth"] = np.log1p(np.maximum(0, data["fWidth"]))
    data["log_fDist"] = np.log1p(np.maximum(0, data["fDist"]))

    return data


df_all = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
df_all_fe = create_features(df_all)

# Dimensionality Reduction Features via PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.nan_to_num(df_all_fe.values))
pca = PCA(n_components=3, random_state=SEED)
pca_feats = pca.fit_transform(X_scaled)
for i in range(3):
    df_all_fe[f"pca_{i}"] = pca_feats[:, i]

# Split back to Train and Test
X = df_all_fe.iloc[: len(train_raw)].reset_index(drop=True)
X_test = df_all_fe.iloc[len(train_raw) :].reset_index(drop=True)

# 3. Stratified K-Fold Cross Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_hgb = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_hgb = np.zeros(len(X_test))

# 4. Model Training Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits

    # Model 2: XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="auc",
        random_state=SEED + fold,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    # Model 3: CatBoost
    model_cat = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        early_stopping_rounds=50,
        eval_metric="AUC",
        random_seed=SEED + fold,
        verbose=0,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits

    # Model 4: HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        early_stopping=True,
        random_state=SEED + fold,
    )
    model_hgb.fit(X_tr, y_tr)
    oof_hgb[val_idx] = model_hgb.predict_proba(X_va)[:, 1]
    test_preds_hgb += model_hgb.predict_proba(X_test)[:, 1] / n_splits

# 5. Rank Transformation and Scipy Powell Weight Optimization
oof_matrix = np.column_stack(
    [
        rankdata(oof_lgb) / len(oof_lgb),
        rankdata(oof_xgb) / len(oof_xgb),
        rankdata(oof_cat) / len(oof_cat),
        rankdata(oof_hgb) / len(oof_hgb),
    ]
)

test_matrix = np.column_stack(
    [
        rankdata(test_preds_lgb) / len(test_preds_lgb),
        rankdata(test_preds_xgb) / len(test_preds_xgb),
        rankdata(test_preds_cat) / len(test_preds_cat),
        rankdata(test_preds_hgb) / len(test_preds_hgb),
    ]
)


def objective(weights):
    w = np.maximum(0, weights)
    if np.sum(w) == 0:
        return 0.0
    w = w / np.sum(w)
    pred = np.dot(oof_matrix, w)
    return -roc_auc_score(y, pred)


init_weights = [0.25, 0.25, 0.25, 0.25]
res = minimize(
    objective, init_weights, method="Powell", options={"maxiter": 500, "ftol": 1e-5}
)

opt_weights = np.maximum(0, res.x)
opt_weights = opt_weights / np.sum(opt_weights)

oof_blend = np.dot(oof_matrix, opt_weights)
test_blend = np.dot(test_matrix, opt_weights)

lgb_auc = roc_auc_score(y, oof_lgb)
xgb_auc = roc_auc_score(y, oof_xgb)
cat_auc = roc_auc_score(y, oof_cat)
hgb_auc = roc_auc_score(y, oof_hgb)
final_blend_auc = roc_auc_score(y, oof_blend)

print(f"LightGBM Out-of-Fold ROC-AUC: {lgb_auc:.6f}")
print(f"XGBoost Out-of-Fold ROC-AUC:  {xgb_auc:.6f}")
print(f"CatBoost Out-of-Fold ROC-AUC: {cat_auc:.6f}")
print(f"HistGBM Out-of-Fold ROC-AUC:  {hgb_auc:.6f}")
print(
    f"Optimized Ensemble Weights:   LGB={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, CAT={opt_weights[2]:.4f}, HGB={opt_weights[3]:.4f}"
)
print(f"Optimized Rank Ensemble Out-of-Fold ROC-AUC: {final_blend_auc:.6f}")

# 6. Save Final Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": range(len(X_test)), "class": test_blend})
submission.to_csv("./working/submission.csv", index=False)
