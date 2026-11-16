import os
import json
import random
import copy
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seed and directories
SEED = 42
N_SPLITS = 5
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device configured: {device}")


# =========================================================================
# 1. Domain-Specific Feature Engineering & Data Preparation
# =========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    # 1.1 Hemodynamic and Exercise Dynamics
    feat["pred_max_hr"] = 220.0 - feat["Age"]
    feat["hr_reserve_ratio"] = feat["Max HR"] / (feat["pred_max_hr"] + 1e-5)
    feat["hr_deficit"] = feat["pred_max_hr"] - feat["Max HR"]

    # Rate-Pressure Product (RPP): Index of myocardial oxygen consumption
    feat["rate_pressure_product"] = feat["BP"] * feat["Max HR"]
    feat["rpp_scaled"] = feat["rate_pressure_product"] / 1000.0

    # Duke Treadmill Score Proxy
    feat["dts_proxy"] = (
        feat["Max HR"]
        - (5.0 * feat["ST depression"])
        - (4.0 * feat["Exercise angina"] * 10.0)
    )

    # 1.2 Vascular & Ischemic Indicators
    feat["st_slope_interaction"] = feat["ST depression"] * feat["Slope of ST"]
    feat["st_per_slope"] = feat["ST depression"] / (feat["Slope of ST"] + 1e-5)

    # Cholesterol and Blood Pressure to Age ratios
    feat["chol_to_age"] = feat["Cholesterol"] / (feat["Age"] + 1e-5)
    feat["bp_to_age"] = feat["BP"] / (feat["Age"] + 1e-5)
    feat["chol_bp_ratio"] = feat["Cholesterol"] / (feat["BP"] + 1e-5)
    feat["chol_bp_product"] = feat["Cholesterol"] * feat["BP"]

    # Composite Risk Burden Index (Cardiological flags)
    feat["asymptomatic_angina_flag"] = (feat["Chest pain type"] == 4).astype(int)
    feat["reversible_thallium_flag"] = (feat["Thallium"] == 7).astype(int)
    feat["fixed_thallium_flag"] = (feat["Thallium"] == 6).astype(int)
    feat["abnormal_vessels_flag"] = (feat["Number of vessels fluro"] > 0).astype(int)
    feat["st_ischemia_flag"] = (feat["ST depression"] >= 1.0).astype(int)

    feat["cardio_risk_score"] = (
        feat["asymptomatic_angina_flag"] * 2.0
        + feat["Exercise angina"] * 2.0
        + feat["reversible_thallium_flag"] * 2.5
        + feat["abnormal_vessels_flag"] * 2.0
        + feat["st_ischemia_flag"] * 1.5
        + (feat["FBS over 120"] == 1).astype(int) * 1.0
        + (feat["EKG results"] == 2).astype(int) * 1.0
    )

    # 1.3 Categorical Combinations & Diagnostic Patterns
    feat["angina_exercise_combo"] = (
        feat["Chest pain type"].astype(str) + "_" + feat["Exercise angina"].astype(str)
    )
    feat["thallium_vessels_combo"] = (
        feat["Thallium"].astype(str) + "_" + feat["Number of vessels fluro"].astype(str)
    )
    feat["st_slope_ekg_combo"] = (
        feat["Slope of ST"].astype(str) + "_" + feat["EKG results"].astype(str)
    )
    feat["sex_chest_pain_combo"] = (
        feat["Sex"].astype(str) + "_" + feat["Chest pain type"].astype(str)
    )
    feat["vessels_chest_pain_combo"] = (
        feat["Number of vessels fluro"].astype(str) + "_" + feat["Chest pain type"].astype(str)
    )
    feat["thallium_chest_pain_combo"] = (
        feat["Thallium"].astype(str) + "_" + feat["Chest pain type"].astype(str)
    )

    for cat_col in [
        "angina_exercise_combo",
        "thallium_vessels_combo",
        "st_slope_ekg_combo",
        "sex_chest_pain_combo",
        "vessels_chest_pain_combo",
        "thallium_chest_pain_combo",
    ]:
        freq_map = feat[cat_col].value_counts(normalize=True).to_dict()
        feat[f"{cat_col}_freq"] = feat[cat_col].map(freq_map).astype(float)
        feat[cat_col] = feat[cat_col].astype("category").cat.codes

    # Non-linear transformations
    feat["log_st_depression"] = np.log1p(np.maximum(feat["ST depression"], 0))
    feat["log_cholesterol"] = np.log1p(feat["Cholesterol"])
    feat["log_rpp"] = np.log1p(feat["rate_pressure_product"])

    return feat


# =========================================================================
# 1.4 Fold-Specific Group Statistical Lookups & Target Encodings
# =========================================================================
def compute_group_aggregations(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    group_cols: List[str],
    num_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr_dict = {}
    va_dict = {}
    te_dict = {}

    for g_col in group_cols:
        for num_col in num_cols:
            grp = tr_df.groupby(g_col)[num_col].agg(["mean", "std"])
            mean_map = grp["mean"].to_dict()
            std_map = grp["std"].fillna(1.0).replace(0, 1.0).to_dict()
            global_mean = tr_df[num_col].mean()
            global_std = tr_df[num_col].std() if tr_df[num_col].std() > 0 else 1.0

            tr_mean = tr_df[g_col].map(mean_map).fillna(global_mean).values
            va_mean = va_df[g_col].map(mean_map).fillna(global_mean).values
            te_mean = te_df[g_col].map(mean_map).fillna(global_mean).values

            tr_std = tr_df[g_col].map(std_map).fillna(global_std).values
            va_std = va_df[g_col].map(std_map).fillna(global_std).values
            te_std = te_df[g_col].map(std_map).fillna(global_std).values

            prefix = f"{g_col}_{num_col}"
            tr_dict[f"{prefix}_grp_mean"] = tr_mean.astype(np.float32)
            va_dict[f"{prefix}_grp_mean"] = va_mean.astype(np.float32)
            te_dict[f"{prefix}_grp_mean"] = te_mean.astype(np.float32)

            tr_dict[f"{prefix}_grp_zscore"] = (
                (tr_df[num_col].values - tr_mean) / (tr_std + 1e-5)
            ).astype(np.float32)
            va_dict[f"{prefix}_grp_zscore"] = (
                (va_df[num_col].values - va_mean) / (va_std + 1e-5)
            ).astype(np.float32)
            te_dict[f"{prefix}_grp_zscore"] = (
                (te_df[num_col].values - te_mean) / (te_std + 1e-5)
            ).astype(np.float32)

    tr_res = pd.DataFrame(tr_dict, index=tr_df.index)
    va_res = pd.DataFrame(va_dict, index=va_df.index)
    te_res = pd.DataFrame(te_dict, index=te_df.index)
    return tr_res, va_res, te_res


def compute_target_encodings(
    tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    cat_cols: List[str],
    smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr_dict = {}
    va_dict = {}
    te_dict = {}
    global_mean = float(y_tr.mean())

    for col in cat_cols:
        col_name = f"{col}_te"
        stats = (
            pd.DataFrame({"cat": tr_df[col].values, "target": y_tr})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        smooth_dict = smooth.to_dict()

        tr_dict[col_name] = (
            tr_df[col].map(smooth_dict).fillna(global_mean).values.astype(np.float32)
        )
        va_dict[col_name] = (
            va_df[col].map(smooth_dict).fillna(global_mean).values.astype(np.float32)
        )
        te_dict[col_name] = (
            te_df[col].map(smooth_dict).fillna(global_mean).values.astype(np.float32)
        )

    tr_res = pd.DataFrame(tr_dict, index=tr_df.index)
    va_res = pd.DataFrame(va_dict, index=va_df.index)
    te_res = pd.DataFrame(te_dict, index=te_df.index)
    return tr_res, va_res, te_res


print("Loading raw parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

target_col = "Heart Disease"
if train_df[target_col].dtype == object or isinstance(
    train_df[target_col].iloc[0], str
):
    y = (train_df[target_col] == "Presence").astype(int).values
else:
    y = train_df[target_col].astype(int).values

df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    axis=0,
    ignore_index=True,
)

print("Executing feature engineering pipeline...")
df_all_fe = engineer_features(df_all)

n_train = len(train_df)
n_test = len(test_df)

X_train = df_all_fe.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all_fe.iloc[n_train:].copy().reset_index(drop=True)

feature_cols = [c for c in X_train.columns if c not in ["id", target_col]]

X = X_train[feature_cols].copy()
X_test_data = X_test[feature_cols].copy()

# Stratified 5-Fold split assignment
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = np.zeros(n_train, dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    folds[val_idx] = fold


# =========================================================================
# 2. TabResNet Neural Architecture & Model Factory Definitions
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def get_lgb_model(custom_params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMClassifier(**params)


def get_xgb_model(custom_params: Dict[str, Any] = None) -> xgb.XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBClassifier(**params)


def get_catboost_model(custom_params: Dict[str, Any] = None) -> CatBoostClassifier:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        params.update(custom_params)
    return CatBoostClassifier(**params)


# =========================================================================
# 3. Stratified Cross-Validation & Out-Of-Fold Ensembling
# =========================================================================
oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cat = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cat = np.zeros(n_test, dtype=np.float64)

print(f"Starting {N_SPLITS}-Fold Cross-Validation on {len(feature_cols)} features...")

for fold in range(N_SPLITS):
    idx_tr = np.where(folds != fold)[0]
    idx_va = np.where(folds == fold)[0]

    X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
    X_va, y_va = X.iloc[idx_va], y[idx_va]

    # --- LightGBM ---
    lgb_model = get_lgb_model({"random_state": SEED + fold})
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=lgb_callbacks,
    )
    val_pred_lgb = lgb_model.predict_proba(X_va)[:, 1]
    oof_lgb[idx_va] = val_pred_lgb
    test_preds_lgb += lgb_model.predict_proba(X_test_data)[:, 1] / N_SPLITS

    # --- XGBoost ---
    xgb_model = get_xgb_model(
        {"random_state": SEED + fold, "early_stopping_rounds": 50}
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_pred_xgb = xgb_model.predict_proba(X_va)[:, 1]
    oof_xgb[idx_va] = val_pred_xgb
    test_preds_xgb += xgb_model.predict_proba(X_test_data)[:, 1] / N_SPLITS

    # --- CatBoost ---
    cat_model = get_catboost_model(
        {"random_seed": SEED + fold, "early_stopping_rounds": 50}
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    val_pred_cat = cat_model.predict_proba(X_va)[:, 1]
    oof_cat[idx_va] = val_pred_cat
    test_preds_cat += cat_model.predict_proba(X_test_data)[:, 1] / N_SPLITS

    score_lgb = roc_auc_score(y_va, val_pred_lgb)
    score_xgb = roc_auc_score(y_va, val_pred_xgb)
    score_cat = roc_auc_score(y_va, val_pred_cat)
    print(
        f"Fold {fold} AUC | LGB: {score_lgb:.5f} | XGB: {score_xgb:.5f} | CAT: {score_cat:.5f}"
    )

auc_lgb_overall = roc_auc_score(y, oof_lgb)
auc_xgb_overall = roc_auc_score(y, oof_xgb)
auc_cat_overall = roc_auc_score(y, oof_cat)

print(
    f"OOF Individual Scores -> LGB: {auc_lgb_overall:.5f}, XGB: {auc_xgb_overall:.5f}, CAT: {auc_cat_overall:.5f}"
)

# Optimize ensemble weights to maximize ROC-AUC
oof_list = [oof_lgb, oof_xgb, oof_cat]
test_list = [test_preds_lgb, test_preds_xgb, test_preds_cat]


def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-12)
    blend_pred = sum(w * p for w, p in zip(weights, oof_list))
    return -roc_auc_score(y, blend_pred)


opt_res = minimize(
    blend_objective,
    x0=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    bounds=[(0, 1), (0, 1), (0, 1)],
    method="Nelder-Mead",
)

best_weights = opt_res.x / np.sum(opt_res.x)
print(
    f"Optimized Ensemble Weights: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}"
)

final_oof_preds = sum(w * p for w, p in zip(best_weights, oof_list))
final_test_preds = sum(w * p for w, p in zip(best_weights, test_list))


# =========================================================================
# 4. Generate Submission and Final Metric Output
# =========================================================================
sub = pd.DataFrame(
    {"id": test_df["id"].astype(int).values, "Heart Disease": final_test_preds}
)
sub.to_csv("./submission/submission.csv", index=False)

assert os.path.exists("./submission/submission.csv"), "Submission file missing!"
assert len(sub) == n_test, f"Expected {n_test} rows, found {len(sub)}"
assert list(sub.columns) == ["id", "Heart Disease"], f"Incorrect columns: {sub.columns}"
assert not sub.isna().any().any(), "Submission contains null/nan values!"

final_score = roc_auc_score(y, final_oof_preds)
print(f"Final Validation Score: {final_score}")
