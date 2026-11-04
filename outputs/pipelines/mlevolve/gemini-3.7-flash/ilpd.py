import copy
import os
from typing import Any, Dict, List, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, yeojohnson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set global deterministic random seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Clinical Feature Engineering
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

column_mapping = {
    "V1": "age",
    "V2": "gender",
    "V3": "total_bilirubin",
    "V4": "direct_bilirubin",
    "V5": "alkaline_phosphatase",
    "V6": "alamine_aminotransferase",  # SGPT / ALT
    "V7": "aspartate_aminotransferase",  # SGOT / AST
    "V8": "total_proteins",
    "V9": "albumin",
    "V10": "albumin_globulin_ratio",
    "Age": "age",
    "Gender": "gender",
    "Total_Bilirubin": "total_bilirubin",
    "Direct_Bilirubin": "direct_bilirubin",
    "Alkaline_Phosphatase": "alkaline_phosphatase",
    "Alamine_Aminotransferase": "alamine_aminotransferase",
    "Aspartate_Aminotransferase": "aspartate_aminotransferase",
    "Total_Protiens": "total_proteins",
    "Total_Proteins": "total_proteins",
    "Albumin": "albumin",
    "Albumin_and_Globulin_Ratio": "albumin_globulin_ratio",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        if col in column_mapping:
            renamed[col] = column_mapping[col]
        elif col.lower() in [k.lower() for k in column_mapping.keys()]:
            for k, v in column_mapping.items():
                if col.lower() == k.lower():
                    renamed[col] = v
                    break
    return df.rename(columns=renamed)


train_df = standardize_columns(train_df)
test_df = standardize_columns(test_df)

target_col = None
for candidate in ["Class", "class", "target", "Target", "Dataset", "dataset"]:
    if candidate in train_df.columns:
        target_col = candidate
        break

if target_col is not None:
    y_raw = train_df[target_col]
    unique_targets = set(y_raw.dropna().unique().tolist())

    # In OpenML 1480 / ILPD: Class 1 = liver patient (positive), Class 2 = non-liver patient
    if unique_targets in [{1, 2}, {"1", "2"}, {1.0, 2.0}]:
        y_train = y_raw.apply(lambda x: 1 if int(float(x)) == 1 else 0).astype(int)
    elif unique_targets in [{0, 1}, {"0", "1"}, {0.0, 1.0}]:
        y_train = y_raw.apply(lambda x: 1 if int(float(x)) == 1 else 0).astype(int)
    else:
        val_counts = y_raw.value_counts()
        primary_val = val_counts.index[0]
        y_train = (y_raw == primary_val).astype(int)

    X_train_raw = train_df.drop(columns=[target_col]).copy()
else:
    raise ValueError("Target column not found in training dataset.")

X_test_raw = test_df.copy()


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Standardize Gender to binary
    if "gender" in data.columns:
        gender_col = data["gender"].astype(str).str.strip().str.lower()
        data["gender_encoded"] = gender_col.apply(
            lambda x: 1.0 if x in ["male", "m", "1", "1.0"] else 0.0
        )
        data = data.drop(columns=["gender"])
    else:
        data["gender_encoded"] = 0.0

    # Ensure numerical columns are properly cast
    num_cols = [
        "age",
        "total_bilirubin",
        "direct_bilirubin",
        "alkaline_phosphatase",
        "alamine_aminotransferase",
        "aspartate_aminotransferase",
        "total_proteins",
        "albumin",
        "albumin_globulin_ratio",
    ]
    for col in num_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill domain-exact missing Albumin/Globulin ratio before computing interactions
    # Globulin = Total Proteins - Albumin
    calc_globulin = (data["total_proteins"] - data["albumin"]).clip(lower=0.01)
    calc_ag_ratio = data["albumin"] / calc_globulin

    if "albumin_globulin_ratio" in data.columns:
        data["albumin_globulin_ratio"] = data["albumin_globulin_ratio"].fillna(
            calc_ag_ratio
        )
    else:
        data["albumin_globulin_ratio"] = calc_ag_ratio

    # Bilirubin metabolism features
    tb = data["total_bilirubin"].clip(lower=0.01)
    db = data["direct_bilirubin"].clip(lower=0.0)
    indirect_bilirubin = (tb - db).clip(lower=0.0)

    data["indirect_bilirubin"] = indirect_bilirubin
    data["direct_to_total_bilirubin_ratio"] = db / (tb + 1e-5)
    data["indirect_to_total_bilirubin_ratio"] = indirect_bilirubin / (tb + 1e-5)
    data["is_direct_hyperbilirubinemia"] = (
        (db > 0.3) & (data["direct_to_total_bilirubin_ratio"] > 0.4)
    ).astype(float)
    data["is_total_hyperbilirubinemia"] = (tb > 1.2).astype(float)
    data["jaundice_severity"] = np.clip(tb / 1.2, 0.0, 10.0)

    # Transaminase & Liver Enzyme features (De Ritis Ratio, R-ratio & Interactions)
    alt = data["alamine_aminotransferase"].clip(lower=0.1)  # SGPT
    ast = data["aspartate_aminotransferase"].clip(lower=0.1)  # SGOT
    alp = data["alkaline_phosphatase"].clip(lower=0.1)  # Alkphos
    alb = data["albumin"].clip(lower=0.1)
    age = data["age"].clip(lower=1.0)

    # Clinically validated ALBI score (Albumin-Bilirubin Score):
    # ALBI = (log10(TB in umol/L) * 0.66) + (Albumin in g/L * -0.085)
    # TB mg/dL to umol/L is ~17.1, Albumin g/dL to g/L is ~10
    albi_score = (np.log10(tb * 17.1 + 1e-5) * 0.66) + ((alb * 10.0) * -0.085)
    data["albi_score"] = albi_score
    data["is_albi_grade_1"] = (albi_score <= -2.60).astype(float)
    data["is_albi_grade_2"] = ((albi_score > -2.60) & (albi_score <= -1.39)).astype(float)
    data["is_albi_grade_3"] = (albi_score > -1.39).astype(float)

    # Bilirubin-Albumin Ratio (BAR)
    data["bilirubin_albumin_ratio"] = tb / (alb + 1e-5)
    data["log_bilirubin_albumin_ratio"] = np.log1p(data["bilirubin_albumin_ratio"])

    # FIB-4 surrogate index: Age * AST / sqrt(ALT)
    data["fib4_surrogate"] = (age * ast) / (np.sqrt(alt) + 1e-5)
    data["log_fib4_surrogate"] = np.log1p(data["fib4_surrogate"])

    # King's score surrogate index: Age * AST * TB / 100
    data["kings_surrogate"] = (age * ast * tb) / 100.0
    data["log_kings_surrogate"] = np.log1p(data["kings_surrogate"])

    # Clinically validated R-ratio (R-factor) for liver injury phenotype classification:
    # ULN_ALT ~ 40 U/L, ULN_ALP ~ 120 U/L
    uln_alt = 40.0
    uln_alp = 120.0
    r_ratio = (alt / uln_alt) / (alp / uln_alp + 1e-5)
    data["r_ratio"] = r_ratio
    data["is_hepatocellular_injury"] = (r_ratio >= 5.0).astype(float)
    data["is_cholestatic_injury"] = (r_ratio <= 2.0).astype(float)
    data["is_mixed_injury"] = ((r_ratio > 2.0) & (r_ratio < 5.0)).astype(float)

    data["de_ritis_ratio"] = ast / alt
    data["is_de_ritis_low"] = (data["de_ritis_ratio"] < 1.0).astype(float)
    data["is_de_ritis_mid"] = ((data["de_ritis_ratio"] >= 1.0) & (data["de_ritis_ratio"] <= 2.0)).astype(float)
    data["is_de_ritis_high"] = (data["de_ritis_ratio"] > 2.0).astype(float)
    data["transaminases_sum"] = alt + ast
    data["transaminases_mean"] = (alt + ast) / 2.0
    data["transaminases_diff"] = ast - alt

    # Protein spectrum and Albumin synthetic deficit index
    data["calculated_globulin"] = calc_globulin
    data["calculated_ag_ratio"] = calc_ag_ratio
    data["ag_ratio_discrepancy"] = data["albumin_globulin_ratio"] - calc_ag_ratio
    data["albumin_fraction_of_protein"] = data["albumin"] / (
        data["total_proteins"].clip(lower=0.1)
    )
    data["albumin_deficit_continuous"] = (3.5 - data["albumin"]).clip(lower=0.0)
    data["albumin_synthetic_deficit_index"] = 3.5 - data["albumin"]
    data["is_albumin_deficient"] = (data["albumin"] < 3.5).astype(float)
    data["is_severe_albumin_deficit"] = (data["albumin"] < 2.8).astype(float)

    # Cross-enzyme ratios and interactions
    data["alp_to_ast_ratio"] = alp / (ast + 1e-5)
    data["alp_to_alt_ratio"] = alp / (alt + 1e-5)
    data["alp_to_transaminases_ratio"] = alp / (data["transaminases_sum"] + 1e-5)
    data["bilirubin_alp_interaction"] = tb * alp
    data["bilirubin_ast_interaction"] = tb * ast

    # Cross-metabolic enzyme and protein interactions
    data["alp_to_albumin"] = alp / (alb + 1e-5)
    data["ast_to_total_protein"] = ast / (data["total_proteins"].clip(lower=0.1))
    data["alt_to_total_protein"] = alt / (data["total_proteins"].clip(lower=0.1))

    # Age and risk stratification features
    data["is_elderly"] = (age >= 60).astype(float)
    data["is_young"] = (age < 30).astype(float)
    data["age_times_tb"] = age * tb
    data["age_times_ast"] = age * ast

    # Log-transformed biomarkers
    data["log_total_bilirubin"] = np.log1p(tb)
    data["log_direct_bilirubin"] = np.log1p(db)
    data["log_indirect_bilirubin"] = np.log1p(indirect_bilirubin)
    data["log_alkaline_phosphatase"] = np.log1p(alp)
    data["log_alamine_aminotransferase"] = np.log1p(alt)
    data["log_aspartate_aminotransferase"] = np.log1p(ast)
    data["log_transaminases_sum"] = np.log1p(data["transaminases_sum"])

    # Yeo-Johnson power transformations on heavily skewed enzyme/bilirubin markers
    for col_name, series in [
        ("yj_total_bilirubin", tb),
        ("yj_direct_bilirubin", db + 1e-5),
        ("yj_alkaline_phosphatase", alp),
        ("yj_alamine_aminotransferase", alt),
        ("yj_aspartate_aminotransferase", ast),
        ("yj_transaminases_sum", data["transaminases_sum"]),
    ]:
        transformed_vals, _ = yeojohnson(series.fillna(series.median()))
        data[col_name] = transformed_vals

    # Keep only numeric columns
    data = data.select_dtypes(include=[np.number])

    return data


X_train_engineered = create_clinical_features(X_train_raw)
X_test_engineered = create_clinical_features(X_test_raw)

feature_cols = [
    col for col in X_train_engineered.columns if col in X_test_engineered.columns
]
X_train_engineered = X_train_engineered[feature_cols].copy()
X_test_engineered = X_test_engineered[feature_cols].copy()

# Handle residual NaNs using median of training set
train_medians = X_train_engineered.median(numeric_only=True)
X_train_engineered = X_train_engineered.fillna(train_medians).astype(np.float32)
X_test_engineered = X_test_engineered.fillna(train_medians).astype(np.float32)

# =========================================================================
# 2. Model Architecture Definitions & Factories
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return torch.sigmoid(logits).squeeze(-1)


class PyTorchTabularResNetClassifier:
    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
        lr: float = 1e-3,
        weight_decay: float = 2e-3,
        epochs: int = 55,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        torch.manual_seed(self.random_state)
        input_dim = X_train.shape[1]
        self.model = TabularResNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        best_val_auc = -1.0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    val_preds = self.model(val_x_t).cpu().numpy()
                    val_auc = roc_auc_score(y_val, val_preds)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model(x_t).cpu().numpy()
        preds = np.clip(preds, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - preds, preds])


def get_tabular_resnet_model(random_state: int = 42) -> PyTorchTabularResNetClassifier:
    return PyTorchTabularResNetClassifier(
        hidden_dim=64,
        num_blocks=2,
        dropout_rate=0.25,
        lr=1e-3,
        weight_decay=2e-3,
        epochs=55,
        batch_size=32,
        random_state=random_state,
    )


def get_lightgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=15,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_model(random_state: int = 42) -> cb.CatBoostClassifier:
    return cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
    )


def get_extratrees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_randomforest_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_logistic_model(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )


def get_svc_model(random_state: int = 42) -> CalibratedClassifierCV:
    base_svc = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=random_state,
    )
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)


def get_ridge_model(random_state: int = 42) -> CalibratedClassifierCV:
    base_ridge = RidgeClassifier(alpha=1.0, random_state=random_state)
    return CalibratedClassifierCV(estimator=base_ridge, method="sigmoid", cv=3)


# =========================================================================
# 3. Leak-Free Cross-Validation & Ensemble Blending
# =========================================================================
X = X_train_engineered.values.astype(np.float32)
y = y_train.values.astype(np.float32)
X_test = X_test_engineered.values.astype(np.float32)

num_samples = len(y)
num_test = len(X_test)
n_splits = 5
n_repeats = 5
total_folds = n_splits * n_repeats

model_names = [
    "lightgbm",
    "xgboost",
    "catboost",
    "extratrees",
    "random_forest",
    "tabular_resnet",
    "calibrated_svc",
    "logistic_regression",
]

oof_predictions = {
    name: np.zeros(num_samples, dtype=np.float64) for name in model_names
}
test_predictions = {name: np.zeros(num_test, dtype=np.float64) for name in model_names}

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X, y)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # Fold-safe standardization for neural and margin-based models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test)

    fold_seed = 42 + fold_idx

    # A. LightGBM
    lgb_clf = get_lightgbm_model(random_state=fold_seed)
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_predictions["lightgbm"][va_idx] += (
        lgb_clf.predict_proba(X_va)[:, 1] / n_repeats
    )
    test_predictions["lightgbm"] += lgb_clf.predict_proba(X_test)[:, 1] / total_folds

    # B. XGBoost
    xgb_clf = get_xgboost_model(random_state=fold_seed)
    xgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_predictions["xgboost"][va_idx] += (
        xgb_clf.predict_proba(X_va)[:, 1] / n_repeats
    )
    test_predictions["xgboost"] += xgb_clf.predict_proba(X_test)[:, 1] / total_folds

    # C. CatBoost
    cat_clf = get_catboost_model(random_state=fold_seed)
    cat_clf.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_predictions["catboost"][va_idx] += (
        cat_clf.predict_proba(X_va)[:, 1] / n_repeats
    )
    test_predictions["catboost"] += cat_clf.predict_proba(X_test)[:, 1] / total_folds

    # D. ExtraTrees
    et_clf = get_extratrees_model(random_state=fold_seed)
    et_clf.fit(X_tr, y_tr)
    oof_predictions["extratrees"][va_idx] += (
        et_clf.predict_proba(X_va)[:, 1] / n_repeats
    )
    test_predictions["extratrees"] += et_clf.predict_proba(X_test)[:, 1] / total_folds

    # E. Random Forest
    rf_clf = get_randomforest_model(random_state=fold_seed)
    rf_clf.fit(X_tr, y_tr)
    oof_predictions["random_forest"][va_idx] += (
        rf_clf.predict_proba(X_va)[:, 1] / n_repeats
    )
    test_predictions["random_forest"] += (
        rf_clf.predict_proba(X_test)[:, 1] / total_folds
    )

    # F. PyTorch Tabular ResNet
    resnet_clf = get_tabular_resnet_model(random_state=fold_seed)
    resnet_clf.fit(X_tr_scaled, y_tr, X_val=X_va_scaled, y_val=y_va)
    oof_predictions["tabular_resnet"][va_idx] += (
        resnet_clf.predict_proba(X_va_scaled)[:, 1] / n_repeats
    )
    test_predictions["tabular_resnet"] += (
        resnet_clf.predict_proba(X_te_scaled)[:, 1] / total_folds
    )

    # G. Calibrated SVC (RBF kernel with Platt scaling)
    svc_clf = get_svc_model(random_state=fold_seed)
    svc_clf.fit(X_tr_scaled, y_tr)
    oof_predictions["calibrated_svc"][va_idx] += (
        svc_clf.predict_proba(X_va_scaled)[:, 1] / n_repeats
    )
    test_predictions["calibrated_svc"] += (
        svc_clf.predict_proba(X_te_scaled)[:, 1] / total_folds
    )

    # H. Logistic Regression
    lr_clf = get_logistic_model(random_state=fold_seed)
    lr_clf.fit(X_tr_scaled, y_tr)
    oof_predictions["logistic_regression"][va_idx] += (
        lr_clf.predict_proba(X_va_scaled)[:, 1] / n_repeats
    )
    test_predictions["logistic_regression"] += (
        lr_clf.predict_proba(X_te_scaled)[:, 1] / total_folds
    )

# Regularized logit-space stacking meta-learner
oof_logits = np.column_stack(
    [
        logit(np.clip(oof_predictions[name], 1e-5, 1.0 - 1e-5))
        for name in model_names
    ]
)
test_logits = np.column_stack(
    [
        logit(np.clip(test_predictions[name], 1e-5, 1.0 - 1e-5))
        for name in model_names
    ]
)

meta_learner = LogisticRegression(
    C=0.2,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)
meta_learner.fit(oof_logits, y)

blended_oof_preds = meta_learner.predict_proba(oof_logits)[:, 1]
final_test_preds = meta_learner.predict_proba(test_logits)[:, 1]
final_score = roc_auc_score(y, blended_oof_preds)

# =========================================================================
# 4. Generate & Verify Final Submission
# =========================================================================
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds)),
        "Class": final_test_preds,
    }
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert submission_df.shape == (num_test, 2), f"Incorrect shape: {submission_df.shape}"
assert list(submission_df.columns) == [
    "id",
    "Class",
], f"Incorrect columns: {submission_df.columns}"
assert (
    submission_df["Class"].isnull().sum() == 0
), "Submission contains null/NaN values!"
assert not np.isinf(submission_df["Class"]).any(), "Submission contains Inf values!"

print(f"Final Validation Score: {final_score}")
