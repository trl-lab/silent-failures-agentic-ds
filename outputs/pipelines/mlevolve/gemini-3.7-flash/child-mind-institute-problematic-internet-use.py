import copy
from functools import partial
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Environment, Reproducibility & Directory Setup
# =========================================================================
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


seed_everything(42)

# =========================================================================
# 2. Evaluation Metric & Threshold Optimizer
# =========================================================================


def compute_qwk(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """Computes the Quadratic Weighted Kappa (QWK) between ground truth labels and predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(
        cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic")
    )


class QWKOptimizer:
    """Optimizes continuous regression predictions into discrete ordinal classes {0, 1, 2, 3} using multi-start search."""

    def __init__(
        self, initial_thresholds: Optional[List[float]] = None, n_classes: int = 4
    ):
        self.n_classes = n_classes
        if initial_thresholds is not None:
            self.thresholds = np.array(initial_thresholds, dtype=float)
        else:
            self.thresholds = np.array([0.5, 1.5, 2.5], dtype=float)

    def _predict(self, y_pred: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        sorted_th = np.sort(thresholds)
        return np.digitize(y_pred, sorted_th)

    def _loss(
        self, thresholds: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray
    ) -> float:
        discrete_preds = self._predict(y_pred, thresholds)
        score = compute_qwk(y_true, discrete_preds)
        return -score

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "QWKOptimizer":
        loss_func = partial(self._loss, y_pred=y_pred, y_true=y_true)
        best_score = -float("inf")
        best_th = self.thresholds

        initial_candidates = [
            list(self.thresholds),
            [0.5, 1.5, 2.5],
            [0.6, 1.3, 2.3],
            [0.4, 1.1, 1.9],
        ]
        if len(y_true) > 0:
            p1 = np.mean(y_true == 0)
            p2 = p1 + np.mean(y_true == 1)
            p3 = p2 + np.mean(y_true == 2)
            try:
                q_th = [
                    float(np.quantile(y_pred, p1)),
                    float(np.quantile(y_pred, p2)),
                    float(np.quantile(y_pred, p3)),
                ]
                initial_candidates.append(q_th)
            except Exception:
                pass

        for init_th in initial_candidates:
            for method in ["Nelder-Mead", "Powell"]:
                try:
                    res = minimize(
                        loss_func,
                        init_th,
                        method=method,
                        options={"maxiter": 500},
                    )
                    score = -self._loss(res.x, y_pred, y_true)
                    if score > best_score:
                        best_score = score
                        best_th = np.sort(res.x)
                except Exception:
                    continue

        self.thresholds = np.sort(best_th)
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        return self._predict(y_pred, self.thresholds)


class EnsembleWeightOptimizer:
    """Optimizes blending weights via bounded Dirichlet random search and local exploration."""

    def __init__(self, n_models: int):
        self.n_models = n_models
        self.weights = np.ones(n_models) / n_models

    def fit(
        self, oof_preds_list: List[np.ndarray], y_true: np.ndarray
    ) -> "EnsembleWeightOptimizer":
        preds_matrix = np.column_stack(oof_preds_list)
        best_qwk = -1.0
        best_weights = np.ones(self.n_models) / self.n_models

        # Single model evaluation
        for i in range(self.n_models):
            w = np.zeros(self.n_models)
            w[i] = 1.0
            blended = preds_matrix @ w
            opt = QWKOptimizer().fit(blended, y_true)
            score = compute_qwk(y_true, opt.predict(blended))
            if score > best_qwk:
                best_qwk = score
                best_weights = w

        # Equal weights baseline
        w_eq = np.ones(self.n_models) / self.n_models
        blended_eq = preds_matrix @ w_eq
        opt_eq = QWKOptimizer().fit(blended_eq, y_true)
        score_eq = compute_qwk(y_true, opt_eq.predict(blended_eq))
        if score_eq > best_qwk:
            best_qwk = score_eq
            best_weights = w_eq

        # Dirichlet random search across the simplex
        rng = np.random.RandomState(42)
        dirichlet_samples = rng.dirichlet(np.ones(self.n_models), size=600)
        for w in dirichlet_samples:
            blended = preds_matrix @ w
            opt = QWKOptimizer().fit(blended, y_true)
            score = compute_qwk(y_true, opt.predict(blended))
            if score > best_qwk:
                best_qwk = score
                best_weights = w

        # Local fine-tuning around best found weights
        for _ in range(60):
            noise = rng.normal(0, 0.06, size=self.n_models)
            w = np.clip(best_weights + noise, 0, None)
            if np.sum(w) > 0:
                w = w / np.sum(w)
                blended = preds_matrix @ w
                opt = QWKOptimizer().fit(blended, y_true)
                score = compute_qwk(y_true, opt.predict(blended))
                if score > best_qwk:
                    best_qwk = score
                    best_weights = w

        self.weights = best_weights
        return self

    def blend(self, preds_list: List[np.ndarray]) -> np.ndarray:
        preds_matrix = np.column_stack(preds_list)
        return preds_matrix @ self.weights


# =========================================================================
# 3. Domain Feature Engineering
# =========================================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain transformations, physiological ratios, questionnaire aggregates, and missingness metrics."""
    data = df.copy()

    # Demographic Interactions
    if "Basic_Demos-Age" in data.columns and "Basic_Demos-Sex" in data.columns:
        data["Age_Sex_Interaction"] = data["Basic_Demos-Age"] * 10 + data[
            "Basic_Demos-Sex"
        ].fillna(-1)
        data["Age_Squared"] = data["Basic_Demos-Age"] ** 2

    # Physical Vitals & Anthropometric Ratios
    has_height = "Physical-Height" in data.columns
    has_weight = "Physical-Weight" in data.columns
    has_waist = "Physical-Waist_Circumference" in data.columns
    has_sys_bp = "Physical-Systolic_BP" in data.columns
    has_dia_bp = "Physical-Diastolic_BP" in data.columns
    has_bmi = "Physical-BMI" in data.columns
    has_hr = "Physical-HeartRate" in data.columns

    if has_height and has_weight:
        height_m = data["Physical-Height"] / 100.0
        data["Calc_BMI"] = data["Physical-Weight"] / (height_m**2 + 1e-6)
        if has_bmi:
            data["BMI_Diff"] = data["Calc_BMI"] - data["Physical-BMI"]

    if has_waist and has_height:
        data["Waist_to_Height_Ratio"] = data["Physical-Waist_Circumference"] / (
            data["Physical-Height"] + 1e-6
        )

    if has_waist and has_bmi:
        data["Waist_to_BMI_Ratio"] = data["Physical-Waist_Circumference"] / (
            data["Physical-BMI"] + 1e-6
        )

    if has_sys_bp and has_dia_bp:
        data["Pulse_Pressure"] = (
            data["Physical-Systolic_BP"] - data["Physical-Diastolic_BP"]
        )
        data["Mean_Arterial_Pressure"] = (
            2.0 * data["Physical-Diastolic_BP"] + data["Physical-Systolic_BP"]
        ) / 3.0
        data["BP_Ratio"] = data["Physical-Systolic_BP"] / (
            data["Physical-Diastolic_BP"] + 1e-6
        )

    if has_hr and has_sys_bp:
        data["Rate_Pressure_Product"] = (
            data["Physical-HeartRate"] * data["Physical-Systolic_BP"]
        )

    # Bioelectrical Impedance Analysis (BIA) & Body Composition Ratios
    if "BIA-BIA_Fat" in data.columns and "BIA-BIA_FFM" in data.columns:
        data["Fat_to_FFM_Ratio"] = data["BIA-BIA_Fat"] / (data["BIA-BIA_FFM"] + 1e-6)

    if has_weight:
        if "BIA-BIA_Fat" in data.columns:
            data["Fat_to_Weight_Ratio"] = data["BIA-BIA_Fat"] / (data["Physical-Weight"] + 1e-6)
        if "BIA-BIA_SMM" in data.columns:
            data["SMM_to_Weight_Ratio"] = data["BIA-BIA_SMM"] / (data["Physical-Weight"] + 1e-6)
        if "BIA-BIA_BMC" in data.columns:
            data["BMC_to_Weight_Ratio"] = data["BIA-BIA_BMC"] / (data["Physical-Weight"] + 1e-6)
        if "BIA-BIA_FFM" in data.columns:
            data["FFM_to_Weight_Ratio"] = data["BIA-BIA_FFM"] / (data["Physical-Weight"] + 1e-6)
        if "BIA-BIA_BMR" in data.columns:
            data["BMR_per_kg"] = data["BIA-BIA_BMR"] / (data["Physical-Weight"] + 1e-6)
        if "BIA-BIA_TBW" in data.columns:
            data["TBW_per_kg"] = data["BIA-BIA_TBW"] / (data["Physical-Weight"] + 1e-6)

    if "BIA-BIA_ICW" in data.columns and "BIA-BIA_ECW" in data.columns:
        data["ICW_to_ECW_Ratio"] = data["BIA-BIA_ICW"] / (data["BIA-BIA_ECW"] + 1e-6)

    # Questionnaire Subscale Aggregations
    subscale_prefixes = ["PAQ_A", "PAQ_C", "SDS", "FGC", "Fitness_Endurance"]
    for prefix in subscale_prefixes:
        sub_cols = [
            c for c in data.columns if c.startswith(prefix) and "Season" not in c
        ]
        numeric_sub_cols = [
            c for c in sub_cols if pd.api.types.is_numeric_dtype(data[c])
        ]
        if len(numeric_sub_cols) > 1:
            data[f"{prefix}_mean"] = data[numeric_sub_cols].mean(axis=1)
            data[f"{prefix}_std"] = data[numeric_sub_cols].std(axis=1)
            data[f"{prefix}_sum"] = data[numeric_sub_cols].sum(axis=1)
            data[f"{prefix}_max"] = data[numeric_sub_cols].max(axis=1)
            data[f"{prefix}_min"] = data[numeric_sub_cols].min(axis=1)
            data[f"{prefix}_non_zero_count"] = (data[numeric_sub_cols] > 0).sum(axis=1)

    # Lifestyle Balance Ratio: Sleep Disturbance to Physical Activity
    paq_cols = [c for c in ["PAQ_A_mean", "PAQ_C_mean"] if c in data.columns]
    if paq_cols and "SDS_mean" in data.columns:
        combined_paq = data[paq_cols].mean(axis=1)
        data["Sleep_to_PAQ_Ratio"] = data["SDS_mean"] / (combined_paq + 1e-6)

    # Age-and-Sex Normative Deviations
    if "Basic_Demos-Age" in data.columns and "Basic_Demos-Sex" in data.columns:
        norm_cols = [
            c
            for c in [
                "Physical-BMI",
                "CGAS-CGAS_Score",
                "BIA-BIA_Fat",
                "BIA-BIA_BMR",
                "BIA-BIA_FFM",
                "SDS_mean",
            ]
            if c in data.columns
        ]
        group_keys = [data["Basic_Demos-Age"].fillna(-1), data["Basic_Demos-Sex"].fillna(-1)]
        for col in norm_cols:
            grp_mean = data.groupby(group_keys)[col].transform("mean")
            grp_std = data.groupby(group_keys)[col].transform("std")
            data[f"{col}_AgeSex_Delta"] = data[col] - grp_mean
            data[f"{col}_AgeSex_ZScore"] = (data[col] - grp_mean) / (grp_std + 1e-6)

    # Missingness Profiling
    data["Total_Missing_Count"] = data.isnull().sum(axis=1)
    data["Total_Missing_Ratio"] = data.isnull().mean(axis=1)

    domain_prefixes = [
        "Physical",
        "BIA",
        "PAQ_A",
        "PAQ_C",
        "SDS",
        "PreInt_EduHx",
        "CGAS",
        "Fitness_Endurance",
        "FGC",
    ]
    for prefix in domain_prefixes:
        dom_cols = [c for c in data.columns if c.startswith(prefix)]
        if len(dom_cols) > 0:
            data[f"{prefix}_Missing_Count"] = data[dom_cols].isnull().sum(axis=1)

    # Seasonality Patterns
    season_cols = [c for c in data.columns if "Season" in c]
    if len(season_cols) > 1:
        data["Unique_Seasons_Count"] = data[season_cols].nunique(axis=1)

    return data


# =========================================================================
# 4. Neural Network Architecture & Loss Definition
# =========================================================================


class OrdinalHuberLoss(nn.Module):

    def __init__(self, delta: float = 1.0):
        super(OrdinalHuberLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred.view(-1), y_true.view(-1))


class TabularResidualBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super(TabularResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.dropout(self.act(self.norm1(self.linear1(x))))
        out = self.norm2(self.linear2(out))
        out = self.act(out + residual)
        return out


class TabularNeuralNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.25,
    ):
        super(TabularNeuralNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        blocks = []
        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            blocks.append(TabularResidualBlock(in_d, out_d, dropout=dropout * 0.8))
        self.res_blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        feat = self.res_blocks(feat)
        out = self.head(feat)
        return out.squeeze(-1)


# =========================================================================
# 5. Model Factory
# =========================================================================


class ModelFactory:

    @staticmethod
    def get_lgbm_params(seed: int = 42) -> Dict[str, Any]:
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }

    @staticmethod
    def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "random_state": seed,
            "n_jobs": -1,
        }

    @staticmethod
    def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
        return {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 3.0,
            "random_seed": seed,
            "verbose": False,
            "thread_count": -1,
        }

    @staticmethod
    def create_pytorch_pipeline(
        input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4
    ) -> Tuple[nn.Module, nn.Module, optim.Optimizer, Any]:
        model = TabularNeuralNet(input_dim=input_dim)
        criterion = OrdinalHuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=1, eta_min=1e-5
        )
        return model, criterion, optimizer, scheduler


# =========================================================================
# 6. Data Loading & Preprocessing
# =========================================================================
print("Loading raw train and test datasets...")
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

train_df = train_raw.dropna(subset=["sii"]).copy().reset_index(drop=True)
train_df["sii"] = train_df["sii"].astype(int)

# Dual-target continuous scale alignment for PCIAT-PCIAT_Total
pciat_target = (
    train_df["PCIAT-PCIAT_Total"].copy()
    if "PCIAT-PCIAT_Total" in train_df.columns
    else pd.Series(np.nan, index=train_df.index)
)
if pciat_target.isnull().any():
    sii_to_pciat_median = train_df.groupby("sii")["PCIAT-PCIAT_Total"].median() if "PCIAT-PCIAT_Total" in train_df.columns else pd.Series()
    fallback_map = {0: 10.0, 1: 35.0, 2: 55.0, 3: 75.0}
    pciat_target = pciat_target.fillna(train_df["sii"].map(sii_to_pciat_median)).fillna(train_df["sii"].map(fallback_map))

pciat_min = float(pciat_target.min())
pciat_max = float(pciat_target.max())
pciat_scale = max(pciat_max - pciat_min, 1e-6)
train_df["pciat_scaled"] = ((pciat_target - pciat_min) / pciat_scale) * 3.0

test_df = test_raw.copy().reset_index(drop=True)

base_features = [
    col for col in test_df.columns if col != "id" and not col.startswith("PCIAT")
]

print("Applying domain feature engineering...")
X_train_eng = engineer_features(train_df[base_features])
X_test_eng = engineer_features(test_df[base_features])

# Encode categorical columns
for col in X_train_eng.columns:
    if X_train_eng[col].dtype == "object" or isinstance(
        X_train_eng[col].dtype, pd.CategoricalDtype
    ):
        categories = list(
            set(X_train_eng[col].dropna().unique()).union(
                set(X_test_eng[col].dropna().unique())
            )
        )
        X_train_eng[col] = pd.Categorical(X_train_eng[col], categories=categories).codes
        X_test_eng[col] = pd.Categorical(X_test_eng[col], categories=categories).codes

X_train_eng = X_train_eng.replace([np.inf, -np.inf], np.nan)
X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan)

feature_cols = list(X_train_eng.columns)
n_features = len(feature_cols)

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_eng, train_df["sii"])):
    fold_assignments[val_idx] = fold

train_df["fold"] = fold_assignments
n_folds = 5

# =========================================================================
# 7. Model Training & Out-Of-Fold Evaluation (Dual-Target GBDT + NN)
# =========================================================================
oof_lgb_sii = np.zeros(len(train_df))
oof_lgb_pciat = np.zeros(len(train_df))
oof_xgb_sii = np.zeros(len(train_df))
oof_xgb_pciat = np.zeros(len(train_df))
oof_cb_sii = np.zeros(len(train_df))
oof_cb_pciat = np.zeros(len(train_df))
oof_nn = np.zeros(len(train_df))

test_preds_lgb_sii = np.zeros(len(test_df))
test_preds_lgb_pciat = np.zeros(len(test_df))
test_preds_xgb_sii = np.zeros(len(test_df))
test_preds_xgb_pciat = np.zeros(len(test_df))
test_preds_cb_sii = np.zeros(len(test_df))
test_preds_cb_pciat = np.zeros(len(test_df))
test_preds_nn = np.zeros(len(test_df))

X_test_raw = X_test_eng.values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n--- Commencing Multi-Model Dual-Target 5-Fold Training ---")

for fold in range(n_folds):
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    X_train = X_train_eng.loc[train_mask].values
    y_train_sii = train_df.loc[train_mask, "sii"].values.astype(np.float32)
    y_train_pciat = train_df.loc[train_mask, "pciat_scaled"].values.astype(np.float32)

    X_val = X_train_eng.loc[val_mask].values
    y_val_sii = train_df.loc[val_mask, "sii"].values.astype(np.float32)
    y_val_pciat = train_df.loc[val_mask, "pciat_scaled"].values.astype(np.float32)
    val_indices = np.where(val_mask)[0]

    # --- LightGBM (sii target) ---
    lgb_train_sii = lgb.Dataset(X_train, label=y_train_sii)
    lgb_val_sii = lgb.Dataset(X_val, label=y_val_sii, reference=lgb_train_sii)
    lgb_params = ModelFactory.get_lgbm_params(seed=42 + fold)
    lgb_model_sii = lgb.train(
        lgb_params,
        lgb_train_sii,
        num_boost_round=1500,
        valid_sets=[lgb_val_sii],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb_sii[val_indices] = lgb_model_sii.predict(X_val)
    test_preds_lgb_sii += lgb_model_sii.predict(X_test_raw) / n_folds

    # --- LightGBM (PCIAT continuous target) ---
    lgb_train_pciat = lgb.Dataset(X_train, label=y_train_pciat)
    lgb_val_pciat = lgb.Dataset(X_val, label=y_val_pciat, reference=lgb_train_pciat)
    lgb_model_pciat = lgb.train(
        lgb_params,
        lgb_train_pciat,
        num_boost_round=1500,
        valid_sets=[lgb_val_pciat],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb_pciat[val_indices] = lgb_model_pciat.predict(X_val)
    test_preds_lgb_pciat += lgb_model_pciat.predict(X_test_raw) / n_folds

    # --- XGBoost (sii target) ---
    xgb_params = ModelFactory.get_xgb_params(seed=42 + fold)
    xgb_model_sii = xgb.XGBRegressor(
        **xgb_params, n_estimators=1500, early_stopping_rounds=50
    )
    xgb_model_sii.fit(X_train, y_train_sii, eval_set=[(X_val, y_val_sii)], verbose=False)
    oof_xgb_sii[val_indices] = xgb_model_sii.predict(X_val)
    test_preds_xgb_sii += xgb_model_sii.predict(X_test_raw) / n_folds

    # --- XGBoost (PCIAT continuous target) ---
    xgb_model_pciat = xgb.XGBRegressor(
        **xgb_params, n_estimators=1500, early_stopping_rounds=50
    )
    xgb_model_pciat.fit(X_train, y_train_pciat, eval_set=[(X_val, y_val_pciat)], verbose=False)
    oof_xgb_pciat[val_indices] = xgb_model_pciat.predict(X_val)
    test_preds_xgb_pciat += xgb_model_pciat.predict(X_test_raw) / n_folds

    # --- CatBoost (sii target) ---
    cb_params = ModelFactory.get_catboost_params(seed=42 + fold)
    cb_model_sii = CatBoostRegressor(**cb_params, iterations=1500, early_stopping_rounds=50)
    cb_model_sii.fit(X_train, y_train_sii, eval_set=(X_val, y_val_sii), verbose=False)
    oof_cb_sii[val_indices] = cb_model_sii.predict(X_val)
    test_preds_cb_sii += cb_model_sii.predict(X_test_raw) / n_folds

    # --- CatBoost (PCIAT continuous target) ---
    cb_model_pciat = CatBoostRegressor(**cb_params, iterations=1500, early_stopping_rounds=50)
    cb_model_pciat.fit(X_train, y_train_pciat, eval_set=(X_val, y_val_pciat), verbose=False)
    oof_cb_pciat[val_indices] = cb_model_pciat.predict(X_val)
    test_preds_cb_pciat += cb_model_pciat.predict(X_test_raw) / n_folds

    # --- PyTorch Tabular Neural Network ---
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    X_train_imp = np.nan_to_num(scaler.fit_transform(imputer.fit_transform(X_train)))
    X_val_imp = np.nan_to_num(scaler.transform(imputer.transform(X_val)))
    X_test_imp = np.nan_to_num(scaler.transform(imputer.transform(X_test_raw)))

    train_tensor_x = torch.tensor(X_train_imp, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_sii, dtype=torch.float32)
    val_tensor_x = torch.tensor(X_val_imp, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )

    nn_model, criterion, optimizer, scheduler = ModelFactory.create_pytorch_pipeline(
        input_dim=n_features, lr=1e-3, weight_decay=1e-4
    )
    nn_model = nn_model.to(device)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_preds_tensor = nn_model(val_tensor_x)
            val_loss = criterion(
                val_preds_tensor, torch.tensor(y_val_sii, dtype=torch.float32).to(device)
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = copy.deepcopy(nn_model.state_dict())

    nn_model.load_state_dict(best_nn_weights)
    nn_model.eval()
    with torch.no_grad():
        oof_nn[val_indices] = nn_model(val_tensor_x).cpu().numpy()
        test_tensor_x = torch.tensor(X_test_imp, dtype=torch.float32).to(device)
        test_preds_nn += nn_model(test_tensor_x).cpu().numpy() / n_folds

    print(
        f"Fold {fold+1}/{n_folds} Complete | "
        f"LGB sii RMSE: {np.sqrt(np.mean((oof_lgb_sii[val_indices] - y_val_sii)**2)):.4f} | "
        f"LGB pciat RMSE: {np.sqrt(np.mean((oof_lgb_pciat[val_indices] - y_val_pciat)**2)):.4f} | "
        f"XGB sii RMSE: {np.sqrt(np.mean((oof_xgb_sii[val_indices] - y_val_sii)**2)):.4f} | "
        f"CB sii RMSE: {np.sqrt(np.mean((oof_cb_sii[val_indices] - y_val_sii)**2)):.4f} | "
        f"NN Huber: {best_val_loss:.4f}"
    )

# =========================================================================
# 8. Ensemble Blending & Nelder-Mead Optimization
# =========================================================================
print("\n--- Ensembling Predictions & Optimizing Thresholds ---")
oof_models = [
    oof_lgb_sii,
    oof_lgb_pciat,
    oof_xgb_sii,
    oof_xgb_pciat,
    oof_cb_sii,
    oof_cb_pciat,
    oof_nn,
]
test_models = [
    test_preds_lgb_sii,
    test_preds_lgb_pciat,
    test_preds_xgb_sii,
    test_preds_xgb_pciat,
    test_preds_cb_sii,
    test_preds_cb_pciat,
    test_preds_nn,
]

y_true_all = train_df["sii"].values.astype(int)

ensemble_opt = EnsembleWeightOptimizer(n_models=len(oof_models))
ensemble_opt.fit(oof_models, y_true_all)
print(f"Optimal Ensemble Weights: {np.round(ensemble_opt.weights, 4)}")

oof_blend = ensemble_opt.blend(oof_models)
test_blend = ensemble_opt.blend(test_models)

qwk_optimizer = QWKOptimizer(initial_thresholds=[0.5, 1.5, 2.5])
qwk_optimizer.fit(oof_blend, y_true_all)
optimized_thresholds = qwk_optimizer.thresholds
print(f"Optimal QWK Thresholds: {np.round(optimized_thresholds, 4)}")

oof_discrete_preds = qwk_optimizer.predict(oof_blend)
test_discrete_preds = qwk_optimizer.predict(test_blend)

cv_qwk_score = compute_qwk(y_true_all, oof_discrete_preds)
print(f"Overall Out-Of-Fold Quadratic Weighted Kappa: {cv_qwk_score:.5f}")

# =========================================================================
# 9. Submission Generation & Verification
# =========================================================================
submission_df = pd.DataFrame(
    {"id": test_df["id"].values, "sii": test_discrete_preds.astype(int)}
)

assert len(submission_df) == len(
    test_df
), "Mismatch in test sample count in submission."
assert list(submission_df.columns) == ["id", "sii"], "Submission columns mismatch."
assert not submission_df.isnull().any().any(), "Submission contains null values."
assert (
    submission_df["sii"].isin([0, 1, 2, 3]).all()
), "Predictions contain invalid ordinal classes."

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission successfully written to {submission_path} ({len(submission_df)} rows)."
)

# Mandatory Final Output Line
print(f"Final Validation Score: {cv_qwk_score}")
