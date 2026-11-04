from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for deterministic execution
RANDOM_STATE = 42
SEEDS = [42, 100, 2024, 777, 999]
N_SPLITS = 10
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# =========================================================================
# 1. Feature Engineering & Preprocessing
# =========================================================================
def calculate_egfr(
    age: pd.Series, sex: pd.Series, serum_creatinine: pd.Series
) -> pd.Series:
    """
    Computes estimated Glomerular Filtration Rate (eGFR) using the CKD-EPI (2021) equation.
    sex: 1 for male, 0 for female
    serum_creatinine: mg/dL
    age: years
    """
    kappa = np.where(sex == 0, 0.7, 0.9)
    alpha = np.where(sex == 0, -0.241, -0.302)
    female_multiplier = np.where(sex == 0, 1.012, 1.0)

    scr_ratio = serum_creatinine / kappa
    min_ratio = np.minimum(scr_ratio, 1.0)
    max_ratio = np.maximum(scr_ratio, 1.0)

    egfr = (
        142.0
        * (min_ratio**alpha)
        * (max_ratio**-1.200)
        * (0.9938**age)
        * female_multiplier
    )
    return egfr


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs domain-informed feature engineering for heart failure clinical prognosis.
    Applies non-linear transformations, cardiorenal interaction terms, and clinical threshold indicators.
    """
    data = df.copy()

    # Standardize column names to lower_case for consistent processing
    col_map = {col: col.lower() for col in data.columns if col.lower() != "death_event"}
    if "DEATH_EVENT" in data.columns:
        col_map["DEATH_EVENT"] = "DEATH_EVENT"
    elif "death_event" in data.columns:
        col_map["death_event"] = "DEATH_EVENT"
    data = data.rename(columns=col_map)

    # 1. Logarithmic transformations for right-skewed lab markers & time
    data["log_cpk"] = np.log1p(np.maximum(0, data["creatinine_phosphokinase"]))
    data["log_serum_creatinine"] = np.log1p(np.maximum(0, data["serum_creatinine"]))
    data["log_platelets"] = np.log1p(np.maximum(0, data["platelets"]))
    data["log_time"] = np.log1p(np.maximum(0, data["time"]))

    # 2. Renal & Cardiorenal function features & Hemodynamic interactions
    data["egfr_ckd_epi"] = calculate_egfr(
        data["age"], data["sex"], data["serum_creatinine"]
    )
    # KDIGO CKD staging: G1 (>=90), G2 (60-89), G3a (45-59), G3b (30-44), G4 (15-29), G5 (<15)
    data["kdigo_stage"] = np.where(
        data["egfr_ckd_epi"] >= 90,
        0,
        np.where(
            data["egfr_ckd_epi"] >= 60,
            1,
            np.where(
                data["egfr_ckd_epi"] >= 45,
                2,
                np.where(
                    data["egfr_ckd_epi"] >= 30,
                    3,
                    np.where(data["egfr_ckd_epi"] >= 15, 4, 5),
                ),
            ),
        ),
    )
    data["is_ckd_stage3_plus"] = (data["egfr_ckd_epi"] < 60).astype(np.int32)

    # Sigmoidal non-linear ejection fraction cardiac failure risk curve
    data["ef_sigmoidal_risk"] = 1.0 / (
        1.0 + np.exp((data["ejection_fraction"] - 35.0) / 6.0)
    )

    data["cardiorenal_index"] = data["serum_creatinine"] / (
        data["ejection_fraction"] / 100.0 + 1e-4
    )
    data["creatinine_to_sodium_ratio"] = (
        data["serum_creatinine"] / (data["serum_sodium"] + 1e-4)
    ) * 100.0
    data["sodium_ef_interaction"] = (
        data["serum_sodium"] * data["ejection_fraction"]
    )
    data["electrolyte_renal_ratio"] = data["serum_sodium"] / (
        data["serum_creatinine"] + 1e-4
    )
    data["electrolyte_balance_proxy"] = (
        data["serum_sodium"] - 140.0
    ) / (data["serum_creatinine"] + 1e-4)
    data["sodium_to_ef_ratio"] = data["serum_sodium"] / (
        data["ejection_fraction"] + 1e-4
    )
    data["renal_age_burden"] = data["serum_creatinine"] * data["age"]
    data["cardiac_age_ratio"] = data["ejection_fraction"] / (data["age"] + 1e-4)

    # 3. Clinical threshold and diagnostic indicator features
    data["is_hyponatremia"] = (data["serum_sodium"] < 135).astype(np.int32)
    data["is_severe_hyponatremia"] = (data["serum_sodium"] < 130).astype(np.int32)
    data["is_hfref"] = (data["ejection_fraction"] < 40).astype(np.int32)
    data["is_severe_hf"] = (data["ejection_fraction"] < 30).astype(np.int32)
    data["is_high_creatinine"] = np.where(
        data["sex"] == 1,
        (data["serum_creatinine"] > 1.3).astype(np.int32),
        (data["serum_creatinine"] > 1.1).astype(np.int32),
    )
    data["is_high_cpk"] = (data["creatinine_phosphokinase"] > 200).astype(np.int32)
    data["is_thrombocytopenia"] = (data["platelets"] < 150000).astype(np.int32)
    data["is_elderly"] = (data["age"] >= 65).astype(np.int32)
    data["is_very_elderly"] = (data["age"] >= 75).astype(np.int32)

    # 4. Multi-morbidity composite risk score & Cardiorenal-Anemia Syndrome (CRAS)
    risk_factors = [
        data["anaemia"],
        data["diabetes"],
        data["high_blood_pressure"],
        data["smoking"],
        data["is_hyponatremia"],
        data["is_severe_hf"],
        data["is_high_creatinine"],
        data["is_elderly"],
    ]
    data["multimorbidity_score"] = np.sum(risk_factors, axis=0)

    # MAGGIC-inspired heart failure composite risk score
    maggic_age_pts = np.clip((data["age"] - 50) / 5.0, 0, 8)
    maggic_ef_pts = np.where(
        data["ejection_fraction"] < 20,
        7.0,
        np.where(
            data["ejection_fraction"] < 25,
            6.0,
            np.where(
                data["ejection_fraction"] < 30,
                5.0,
                np.where(
                    data["ejection_fraction"] < 35,
                    3.0,
                    np.where(data["ejection_fraction"] < 40, 2.0, 0.0),
                ),
            ),
        ),
    )
    maggic_scr_pts = np.where(
        data["serum_creatinine"] >= 2.0,
        4.0,
        np.where(data["serum_creatinine"] >= 1.4, 2.0, 0.0),
    )
    data["maggic_risk_score"] = (
        maggic_age_pts
        + maggic_ef_pts
        + maggic_scr_pts
        + data["diabetes"] * 1.5
        + data["high_blood_pressure"] * 1.0
        + data["smoking"] * 1.0
        + data["anaemia"] * 1.0
        + (data["serum_sodium"] < 135).astype(float) * 2.0
    )

    # CRAS and vascular multi-morbidity interactions
    data["cras_anaemia_log_creatinine"] = data["anaemia"] * data["log_serum_creatinine"]
    data["cras_anaemia_high_creatinine"] = (
        data["anaemia"] * data["is_high_creatinine"]
    )
    data["vascular_diabetes_hbp"] = data["diabetes"] * data["high_blood_pressure"]
    data["ef_squared"] = (data["ejection_fraction"] / 100.0) ** 2

    # 5. Follow-up time hazard proxies & interactions
    data["ef_time_hazard"] = (100.0 - data["ejection_fraction"]) / (
        data["log_time"] + 1.0
    )
    data["maggic_time_hazard"] = data["maggic_risk_score"] / (
        data["log_time"] + 1.0
    )
    data["ef_time_interaction"] = data["ejection_fraction"] * data["log_time"]
    data["time_ef_interaction"] = data["time"] * (data["ejection_fraction"] / 100.0)
    data["time_creatinine_ratio"] = data["time"] / (data["serum_creatinine"] + 1e-4)
    data["creatinine_time_hazard"] = data["serum_creatinine"] / (
        data["log_time"] + 1.0
    )
    data["age_time_hazard"] = data["age"] / (data["log_time"] + 1.0)
    data["log_creatinine_time_hazard"] = data["log_serum_creatinine"] / (
        data["log_time"] + 1.0
    )

    return data


# =========================================================================
# 2. Model Architectures & Configurations
# =========================================================================
@dataclass
class EnsembleHyperparameters:
    """
    Standardized hyperparameters for LightGBM, XGBoost, CatBoost, ExtraTrees,
    and LogisticRegression calibrated for small-sample clinical datasets.
    """

    random_state: int = 42

    def get_lightgbm_params(self) -> Dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 12,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": self.random_state,
            "verbose": -1,
            "n_estimators": 500,
        }

    def get_xgboost_params(self) -> Dict[str, Any]:
        return {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "random_state": self.random_state,
            "n_estimators": 500,
            "tree_method": "hist",
        }

    def get_catboost_params(self) -> Dict[str, Any]:
        return {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.03,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_state,
            "iterations": 500,
            "verbose": False,
        }

    def get_extratrees_model(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=self.random_state,
        )

    def get_random_forest_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.random_state,
        )

    def get_hist_gradient_boosting_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=300,
            max_depth=3,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

    def get_logistic_regression_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=0.2,
            max_iter=1000,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            random_state=self.random_state,
        )

    def get_svm_model(self) -> SVC:
        return SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            gamma="scale",
            random_state=self.random_state,
        )

    def get_mlp_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            penalty="l2",
            solver="lbfgs",
            random_state=self.random_state,
        )


# =========================================================================
# PyTorch Tabular Self-Normalizing ResNet (SNN) Architecture
# =========================================================================
class TabularResNet(nn.Module):
    """
    Tabular Residual Self-Normalizing Neural Network with SELU, AlphaDropout,
    and skip connections for smooth non-linear manifold estimation.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 48, dropout_rate: float = 0.08
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity="linear")
        self.act = nn.SELU()
        self.drop1 = nn.AlphaDropout(dropout_rate)

        # Residual Block 1
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.res1_fc1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.res1_fc2.weight, nonlinearity="linear")
        self.drop2 = nn.AlphaDropout(dropout_rate)

        # Residual Block 2
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res2_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.res2_skip = nn.Linear(hidden_dim, hidden_dim // 2)
        nn.init.kaiming_normal_(self.res2_fc1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.res2_fc2.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.res2_skip.weight, nonlinearity="linear")
        self.drop3 = nn.AlphaDropout(dropout_rate)

        self.out_layer = nn.Linear(hidden_dim // 2, 1)
        nn.init.kaiming_normal_(self.out_layer.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_layer(x))
        h = self.drop1(h)

        # Block 1
        res1 = h
        h = self.act(self.res1_fc1(h))
        h = self.drop2(h)
        h = self.act(self.res1_fc2(h)) + res1

        # Block 2
        res2 = self.res2_skip(h)
        h = self.act(self.res2_fc1(h))
        h = self.drop3(h)
        h = self.act(self.res2_fc2(h)) + res2

        out = torch.sigmoid(self.out_layer(h))
        return out.squeeze(-1)


class PyTorchTabularResNet:
    """
    Scikit-learn compatible wrapper for TabularResNet trained with AdamW and Cosine Annealing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        epochs: int = 120,
        batch_size: int = 32,
        dropout_rate: float = 0.08,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = TabularResNet(
            self.input_dim, self.hidden_dim, self.dropout_rate
        )
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    v_pred = self.model(
                        torch.tensor(X_val, dtype=torch.float32)
                    )
                    v_loss = criterion(
                        v_pred, torch.tensor(y_val, dtype=torch.float32)
                    ).item()
                    if v_loss < best_loss:
                        best_loss = v_loss
                        best_state = {
                            k: v.cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            p1 = self.model(X_t).cpu().numpy()
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Transforms continuous predictions to percentile ranks [0, 1]."""
    return (rankdata(arr) - 1.0) / (len(arr) - 1.0 + 1e-12)


def optimize_ensemble_weights(
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
    l2_reg: float = 2e-3,
    dirichlet_reg: float = 1e-3,
) -> np.ndarray:
    """
    Finds optimal convex combination of model predictions to maximize ROC-AUC
    using SLSQP and Nelder-Mead with Dirichlet and L2 regularization.
    oof_matrix shape: (N_samples, N_models)
    """
    n_models = oof_matrix.shape[1]
    init_weights = np.ones(n_models) / n_models
    bounds = [(0.01, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def loss_func(w: np.ndarray) -> float:
        w_norm = np.maximum(1e-6, w)
        w_norm /= np.sum(w_norm)
        blend = np.dot(oof_matrix, w_norm)
        auc = roc_auc_score(y_true, blend)
        l2_penalty = l2_reg * np.sum(w_norm**2)
        dirichlet_penalty = -dirichlet_reg * np.sum(np.log(w_norm + 1e-6))
        return -auc + l2_penalty + dirichlet_penalty

    best_score = roc_auc_score(y_true, np.mean(oof_matrix, axis=1))
    best_weights = init_weights.copy()

    # SLSQP constrained optimization
    try:
        res_slsqp = minimize(
            loss_func,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        w_slsqp = np.maximum(0.0, res_slsqp.x)
        w_slsqp /= np.sum(w_slsqp)
        score_slsqp = roc_auc_score(y_true, np.dot(oof_matrix, w_slsqp))
        if score_slsqp > best_score:
            best_score = score_slsqp
            best_weights = w_slsqp
    except Exception:
        pass

    # Nelder-Mead direct search optimization
    try:
        res_nm = minimize(
            loss_func,
            init_weights,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
        )
        w_nm = np.maximum(0.0, res_nm.x)
        if np.sum(w_nm) > 0:
            w_nm /= np.sum(w_nm)
            score_nm = roc_auc_score(y_true, np.dot(oof_matrix, w_nm))
            if score_nm > best_score:
                best_weights = w_nm
    except Exception:
        pass

    return best_weights


# =========================================================================
# 3. Main Training, Cross-Validation, and Inference Loop
# =========================================================================
def main():
    train_path = os.path.join(INPUT_DIR, "train.parquet")
    test_path = os.path.join(INPUT_DIR, "test.parquet")

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    target_col = None
    for cand in ["DEATH_EVENT", "death_event"]:
        if cand in train_raw.columns:
            target_col = cand
            break

    if target_col is None:
        raise ValueError("Target column 'DEATH_EVENT' not found in training data.")

    train_y = train_raw[target_col].astype(int).values

    # Feature Engineering
    train_feat = engineer_features(train_raw)
    test_feat = engineer_features(test_raw)

    train_feat["DEATH_EVENT"] = train_y
    if "death_event" in train_feat.columns and "DEATH_EVENT" != "death_event":
        train_feat = train_feat.drop(columns=["death_event"])
    if "death_event" in test_feat.columns:
        test_feat = test_feat.drop(columns=["death_event"])
    if "DEATH_EVENT" in test_feat.columns:
        test_feat = test_feat.drop(columns=["DEATH_EVENT"])

    feature_cols = [c for c in train_feat.columns if c not in ["DEATH_EVENT"]]
    y_train = train_feat["DEATH_EVENT"].values.astype(np.float32)

    try:
        import catboost as cb

        has_catboost = True
    except ImportError:
        has_catboost = False

    n_train = len(train_feat)
    n_test = len(test_feat)
    n_seeds = len(SEEDS)

    oof_lgb_total = np.zeros(n_train)
    oof_xgb_total = np.zeros(n_train)
    oof_cb_total = np.zeros(n_train) if has_catboost else None
    oof_rf_total = np.zeros(n_train)
    oof_hgb_total = np.zeros(n_train)
    oof_et_total = np.zeros(n_train)
    oof_lr_total = np.zeros(n_train)
    oof_svm_total = np.zeros(n_train)
    oof_nn_total = np.zeros(n_train)

    test_lgb_total = np.zeros(n_test)
    test_xgb_total = np.zeros(n_test)
    test_cb_total = np.zeros(n_test) if has_catboost else None
    test_rf_total = np.zeros(n_test)
    test_hgb_total = np.zeros(n_test)
    test_et_total = np.zeros(n_test)
    test_lr_total = np.zeros(n_test)
    test_svm_total = np.zeros(n_test)
    test_nn_total = np.zeros(n_test)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_lgb_seed = np.zeros(n_train)
        oof_xgb_seed = np.zeros(n_train)
        oof_cb_seed = np.zeros(n_train) if has_catboost else None
        oof_rf_seed = np.zeros(n_train)
        oof_hgb_seed = np.zeros(n_train)
        oof_et_seed = np.zeros(n_train)
        oof_lr_seed = np.zeros(n_train)
        oof_svm_seed = np.zeros(n_train)
        oof_nn_seed = np.zeros(n_train)

        for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(train_feat, y_train)):
            X_tr_raw = train_feat.iloc[trn_idx][feature_cols].values
            y_tr = y_train[trn_idx]
            X_val_raw = train_feat.iloc[val_idx][feature_cols].values
            y_val = y_train[val_idx]
            X_test_raw = test_feat[feature_cols].values

            # Strict feature normalization isolation per fold
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)

            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            X_tr_pt = pt.fit_transform(X_tr_raw)
            X_val_pt = pt.transform(X_val_raw)
            X_test_pt = pt.transform(X_test_raw)

            builder = EnsembleHyperparameters(random_state=seed + fold_idx)

            # ---------------------------------------------------------
            # 1. LightGBM
            # ---------------------------------------------------------
            lgb_params = builder.get_lightgbm_params()
            n_est_lgb = lgb_params.pop("n_estimators", 500)
            lgb_trn = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_trn)

            lgb_model = lgb.train(
                lgb_params,
                lgb_trn,
                num_boost_round=n_est_lgb,
                valid_sets=[lgb_val],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            oof_lgb_seed[val_idx] = lgb_model.predict(
                X_val, num_iteration=lgb_model.best_iteration
            )
            test_lgb_total += (
                lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
                / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 2. XGBoost
            # ---------------------------------------------------------
            xgb_params = builder.get_xgboost_params()
            xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            oof_xgb_seed[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
            test_xgb_total += (
                xgb_model.predict_proba(X_test)[:, 1] / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 3. CatBoost (if available)
            # ---------------------------------------------------------
            if has_catboost:
                cb_params = builder.get_catboost_params()
                cb_model = cb.CatBoostClassifier(
                    **cb_params, early_stopping_rounds=50
                )
                cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

                oof_cb_seed[val_idx] = cb_model.predict_proba(X_val)[:, 1]
                test_cb_total += (
                    cb_model.predict_proba(X_test)[:, 1] / (N_SPLITS * n_seeds)
                )

            # ---------------------------------------------------------
            # 4. Random Forest Classifier (Balanced)
            # ---------------------------------------------------------
            rf_model = builder.get_random_forest_model()
            rf_model.fit(X_tr, y_tr)
            oof_rf_seed[val_idx] = rf_model.predict_proba(X_val)[:, 1]
            test_rf_total += (
                rf_model.predict_proba(X_test)[:, 1] / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 5. HistGradientBoosting Classifier
            # ---------------------------------------------------------
            hgb_model = builder.get_hist_gradient_boosting_model()
            hgb_model.fit(X_tr, y_tr)
            oof_hgb_seed[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
            test_hgb_total += (
                hgb_model.predict_proba(X_test)[:, 1] / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 6. ExtraTrees Classifier
            # ---------------------------------------------------------
            et_model = builder.get_extratrees_model()
            et_model.fit(X_tr, y_tr)
            oof_et_seed[val_idx] = et_model.predict_proba(X_val)[:, 1]
            test_et_total += (
                et_model.predict_proba(X_test)[:, 1] / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 7. Logistic Regression (ElasticNet)
            # ---------------------------------------------------------
            lr_model = builder.get_logistic_regression_model()
            lr_model.fit(X_tr_pt, y_tr)
            oof_lr_seed[val_idx] = lr_model.predict_proba(X_val_pt)[:, 1]
            test_lr_total += (
                lr_model.predict_proba(X_test_pt)[:, 1] / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 8. Support Vector Classifier (RBF Kernel)
            # ---------------------------------------------------------
            svm_model = builder.get_svm_model()
            svm_model.fit(X_tr_pt, y_tr)
            oof_svm_seed[val_idx] = svm_model.predict_proba(X_val_pt)[:, 1]
            test_svm_total += (
                svm_model.predict_proba(X_test_pt)[:, 1] / (N_SPLITS * n_seeds)
            )

            # ---------------------------------------------------------
            # 9. PyTorch Tabular ResNet (SNN)
            # ---------------------------------------------------------
            nn_model = PyTorchTabularResNet(
                input_dim=X_tr_pt.shape[1],
                hidden_dim=48,
                lr=2e-3,
                weight_decay=1e-4,
                epochs=120,
                batch_size=32,
                dropout_rate=0.08,
                random_state=seed + fold_idx,
            )
            nn_model.fit(X_tr_pt, y_tr, X_val=X_val_pt, y_val=y_val)
            oof_nn_seed[val_idx] = nn_model.predict_proba(X_val_pt)[:, 1]
            test_nn_total += (
                nn_model.predict_proba(X_test_pt)[:, 1] / (N_SPLITS * n_seeds)
            )

        oof_lgb_total += oof_lgb_seed / n_seeds
        oof_xgb_total += oof_xgb_seed / n_seeds
        if has_catboost:
            oof_cb_total += oof_cb_seed / n_seeds
        oof_rf_total += oof_rf_seed / n_seeds
        oof_hgb_total += oof_hgb_seed / n_seeds
        oof_et_total += oof_et_seed / n_seeds
        oof_lr_total += oof_lr_seed / n_seeds
        oof_svm_total += oof_svm_seed / n_seeds
        oof_nn_total += oof_nn_seed / n_seeds

    # Out-Of-Fold Percentile Rank Averaging & Constrained SLSQP Optimization
    models_oof = [
        oof_lgb_total,
        oof_xgb_total,
        oof_rf_total,
        oof_hgb_total,
        oof_et_total,
        oof_lr_total,
        oof_svm_total,
        oof_nn_total,
    ]
    models_test = [
        test_lgb_total,
        test_xgb_total,
        test_rf_total,
        test_hgb_total,
        test_et_total,
        test_lr_total,
        test_svm_total,
        test_nn_total,
    ]
    if has_catboost:
        models_oof.insert(2, oof_cb_total)
        models_test.insert(2, test_cb_total)

    oof_matrix = np.column_stack([rank_normalize(m) for m in models_oof])
    test_matrix = np.column_stack([rank_normalize(m) for m in models_test])

    opt_weights = optimize_ensemble_weights(oof_matrix, y_train)
    oof_ensemble = np.dot(oof_matrix, opt_weights)
    final_test_preds = np.dot(test_matrix, opt_weights)
    final_score = roc_auc_score(y_train, oof_ensemble)

    # Save final submission
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        id_col = sample_sub.columns[0]
        target_col_name = sample_sub.columns[1]
        submission_df = pd.DataFrame(
            {id_col: sample_sub[id_col], target_col_name: final_test_preds}
        )
    else:
        submission_df = pd.DataFrame(
            {"id": np.arange(len(test_feat), dtype=int), "DEATH_EVENT": final_test_preds}
        )
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
