import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Ensure directories exist
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# =========================================================================
# 1. Data Cleaning and Chemoinformatic Feature Engineering
# =========================================================================
def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes column names for downstream model compatibility."""
    df = df.copy()
    cleaned_cols = []
    for col in df.columns:
        c = re.sub(r"[^\w\s]", "_", str(col))
        c = re.sub(r"\s+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        cleaned_cols.append(c)
    df.columns = cleaned_cols
    return df


def map_target(y_raw: pd.Series) -> pd.Series:
    """
    Robustly maps the experimental biodegradation class to binary {0, 1}:
    Positive class (1) = Ready Biodegradable (RB / '1' / 1)
    Negative class (0) = Not Ready Biodegradable (NRB / '2' / 2 / 0)
    """
    unique_vals = set(y_raw.unique())
    if unique_vals.issubset({1, 2}):
        return y_raw.map({1: 1, 2: 0}).astype(int)
    elif unique_vals.issubset({"1", "2"}):
        return y_raw.map({"1": 1, "2": 0}).astype(int)
    elif unique_vals.issubset({"RB", "NRB"}):
        return y_raw.map({"RB": 1, "NRB": 0}).astype(int)
    elif unique_vals.issubset({0, 1}):
        return y_raw.astype(int)
    elif unique_vals.issubset({"0", "1"}):
        return y_raw.astype(int)
    else:
        counts = y_raw.value_counts()
        minority_val = counts.index[-1]
        return (y_raw == minority_val).astype(int)


def engineer_chemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-specific chemoinformatic descriptors and structural ratios
    based on molecular graph theory, electro-topological states, and functional groups.
    """
    X = df.copy()
    eps = 1e-6

    # 1. Heteroatom & Elemental Ratios
    if "nHM" in X.columns:
        if "nO" in X.columns and "nN" in X.columns and "nX" in X.columns:
            X["heteroatom_sum"] = X["nO"] + X["nN"] + X["nX"]
            X["heteroatom_ratio"] = X["heteroatom_sum"] / (X["nHM"] + eps)
            X["halogen_ratio"] = X["nX"] / (X["nHM"] + eps)
            X["nitrogen_oxygen_ratio"] = X["nN"] / (X["nO"] + eps)

        if "nCb_" in X.columns:
            X["aromatic_carbon_ratio"] = X["nCb_"] / (X["nHM"] + eps)

        if "nCIR" in X.columns:
            X["circuit_density"] = X["nCIR"] / (X["nHM"] + eps)

        if "nCp" in X.columns and "nCrt" in X.columns:
            X["aliphatic_branching_ratio"] = (X["nCp"] + X["nCrt"]) / (X["nHM"] + eps)

        if "nHDon" in X.columns and "nO" in X.columns and "nN" in X.columns:
            X["hdon_per_polar_atom"] = X["nHDon"] / (X["nO"] + X["nN"] + eps)

    # 2. Graph Spectral & Topological Interaction Descriptors
    if "SpMax_L" in X.columns and "TI2_L" in X.columns:
        X["laplace_eigen_ratio"] = X["SpMax_L"] / (X["TI2_L"] + eps)

    if "SM6_L" in X.columns and "SpMax_L" in X.columns:
        X["spectral_moment_laplace_ratio"] = X["SM6_L"] / (X["SpMax_L"] + eps)

    if "SpMax_A" in X.columns and "SpMax_B_m_" in X.columns:
        X["adj_burden_eigen_ratio"] = X["SpMax_A"] / (X["SpMax_B_m_"] + eps)

    if "SM6_B_m_" in X.columns and "HyWi_B_m_" in X.columns:
        X["spectral_burden_ratio"] = X["SM6_B_m_"] / (X["HyWi_B_m_"] + eps)

    # 3. Electro-topological & Atomic Property Interactions
    if "Me" in X.columns and "Mi" in X.columns:
        X["electroneg_ioniz_prod"] = X["Me"] * X["Mi"]
        X["electroneg_ioniz_diff"] = X["Me"] - X["Mi"]

    if "SdssC" in X.columns and "SdO" in X.columns:
        X["estate_dssc_do_prod"] = X["SdssC"] * X["SdO"]

    if "Psi_i_1d" in X.columns and "Psi_i_A" in X.columns:
        X["pseudoconn_index_prod"] = X["Psi_i_1d"] * X["Psi_i_A"]

    # 4. Topological Distance & Halogen Burden Aggregations
    cn_freq_cols = [c for c in ["F02_C_N_", "F03_C_N_", "F04_C_N_"] if c in X.columns]
    if cn_freq_cols:
        X["total_CN_freq"] = X[cn_freq_cols].sum(axis=1)
        X["max_CN_freq"] = X[cn_freq_cols].max(axis=1)

    halogen_flag_cols = [
        c for c in ["B01_C_Br_", "B03_C_Cl_", "B04_C_Br_"] if c in X.columns
    ]
    if halogen_flag_cols:
        X["total_halogen_flags"] = X[halogen_flag_cols].sum(axis=1)

    functional_group_cols = [
        c
        for c in ["nArNO2", "nArCOOR", "nCRX3", "nN_N", "N_073", "C_026"]
        if c in X.columns
    ]
    if functional_group_cols:
        X["total_recalcitrant_groups"] = X[functional_group_cols].sum(axis=1)

    # 5. Advanced Molecular Topological & Electronic Interaction Features
    def find_col(name: str):
        if name in X.columns:
            return name
        if f"{name}_" in X.columns:
            return f"{name}_"
        for col in X.columns:
            if col.lower() == name.lower() or col.lower() == f"{name.lower()}_":
                return col
        return None

    j_dz_e_col = find_col("J_Dz_e")
    loc_col = find_col("LOC")
    me_col = find_col("Me")
    spposa_p_col = find_col("SpPosA_B_p")
    spmax_m_col = find_col("SpMax_B_m")
    sdo_col = find_col("SdO")
    no_col = find_col("nO")
    sdssc_col = find_col("SdssC")
    nssssc_col = find_col("NssssC")
    ncrt_col = find_col("nCrt")
    ncp_col = find_col("nCp")
    nhm_col = find_col("nHM")

    if j_dz_e_col and me_col:
        X["balaban_electroneg_prod"] = X[j_dz_e_col] * X[me_col]
    if j_dz_e_col and loc_col:
        X["balaban_loc_ratio"] = X[j_dz_e_col] / (X[loc_col] + eps)
    if spposa_p_col and spmax_m_col:
        X["polar_burden_interaction"] = X[spposa_p_col] * X[spmax_m_col]
    if sdo_col and no_col:
        X["estate_sdo_density"] = X[sdo_col] / (X[no_col] + eps)
    if sdssc_col and nssssc_col:
        X["estate_sdssc_density"] = X[sdssc_col] / (X[nssssc_col] + eps)
    if nssssc_col and ncrt_col and ncp_col and nhm_col:
        X["global_steric_branching"] = (X[nssssc_col] + X[ncrt_col] + X[ncp_col]) / (X[nhm_col] + eps)

    return X


# Load raw datasets
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = [c for c in train_df.columns if c.lower() == "class"][0]
y_train_series = map_target(train_df[target_col])
y_all = y_train_series.values.astype(int)

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

X_train_clean = standardize_col_names(X_train_raw)
X_test_clean = standardize_col_names(X_test_raw)

X_train_eng = engineer_chemical_features(X_train_clean)
X_test_eng = engineer_chemical_features(X_test_clean)

# Align columns
common_cols = [c for c in X_train_eng.columns if c in X_test_eng.columns]
X_train_eng = X_train_eng[common_cols]
X_test_eng = X_test_eng[common_cols]

# Handle infinite and missing values
X_train_eng = X_train_eng.replace([np.inf, -np.inf], np.nan)
X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan)

med_values = X_train_eng.median()
X_train_eng = X_train_eng.fillna(med_values)
X_test_eng = X_test_eng.fillna(med_values)

# Stratified 10-Fold Splitting
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(y_all), dtype=int)

for fold, (_, val_idx) in enumerate(skf.split(X_train_eng, y_all)):
    fold_assignments[val_idx] = fold

# Manifold Projections (PCA & FastICA) with Fold Isolation
pca_components = 6
ica_components = 4

train_pca = np.zeros((len(X_train_eng), pca_components), dtype=np.float32)
train_ica = np.zeros((len(X_train_eng), ica_components), dtype=np.float32)
test_pca_accum = np.zeros((len(X_test_eng), pca_components), dtype=np.float32)
test_ica_accum = np.zeros((len(X_test_eng), ica_components), dtype=np.float32)

for fold in range(n_splits):
    tr_mask = fold_assignments != fold
    val_mask = fold_assignments == fold

    scaler_fold = RobustScaler()
    X_tr_sc = scaler_fold.fit_transform(X_train_eng[tr_mask])
    X_val_sc = scaler_fold.transform(X_train_eng[val_mask])
    X_te_sc = scaler_fold.transform(X_test_eng)

    pca = PCA(n_components=pca_components, random_state=42 + fold)
    ica = FastICA(n_components=ica_components, random_state=42 + fold, max_iter=1000)

    pca.fit(X_tr_sc)
    ica.fit(X_tr_sc)

    train_pca[val_mask] = pca.transform(X_val_sc)
    train_ica[val_mask] = ica.transform(X_val_sc)

    test_pca_accum += pca.transform(X_te_sc) / n_splits
    test_ica_accum += ica.transform(X_te_sc) / n_splits

pca_cols = [f"pca_{i}" for i in range(pca_components)]
ica_cols = [f"ica_{i}" for i in range(ica_components)]

for i, col in enumerate(pca_cols):
    X_train_eng[col] = train_pca[:, i]
    X_test_eng[col] = test_pca_accum[:, i]

for i, col in enumerate(ica_cols):
    X_train_eng[col] = train_ica[:, i]
    X_test_eng[col] = test_ica_accum[:, i]

feature_cols = [c for c in X_train_eng.columns]
X_test_input = X_test_eng[feature_cols].copy()
n_train = len(X_train_eng)
n_test = len(X_test_input)


# =========================================================================
# 2. Model Architectures & Component Definitions
# =========================================================================
class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout1(self.act1(self.norm1(self.linear1(x))))
        out = self.dropout2(self.act2(self.norm2(self.linear2(out))))
        return residual + out


class ChemoTabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.head(x)
        return logits.squeeze(-1)


def get_lightgbm_model(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=15,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=2,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        subsample=0.85,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
    )


def get_random_forest_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def get_extra_trees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def get_calibrated_svm_model(random_state: int = 42) -> CalibratedClassifierCV:
    base_svc = SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3, n_jobs=-1)


def get_logistic_regression_model(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )


MODEL_REGISTRY = {
    "lightgbm": get_lightgbm_model,
    "xgboost": get_xgboost_model,
    "catboost": get_catboost_model,
    "random_forest": get_random_forest_model,
    "extra_trees": get_extra_trees_model,
    "svm": get_calibrated_svm_model,
    "logistic_regression": get_logistic_regression_model,
}

model_names = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "extra_trees",
    "svm",
    "logistic_regression",
    "chemo_resnet",
]

oof_preds = {m: np.zeros(n_train, dtype=np.float64) for m in model_names}
test_preds = {m: np.zeros(n_test, dtype=np.float64) for m in model_names}


# =========================================================================
# 3. Cross-Validation Training Loop
# =========================================================================
for fold in range(n_splits):
    train_idx = np.where(fold_assignments != fold)[0]
    val_idx = np.where(fold_assignments == fold)[0]

    X_tr = X_train_eng.iloc[train_idx][feature_cols].copy()
    y_tr = y_all[train_idx]
    X_val = X_train_eng.iloc[val_idx][feature_cols].copy()
    y_val = y_all[val_idx]

    # --- 1. LightGBM ---
    lgb_clf = MODEL_REGISTRY["lightgbm"](random_state=42 + fold)
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds["lightgbm"][val_idx] = lgb_clf.predict_proba(X_val)[:, 1]
    test_preds["lightgbm"] += lgb_clf.predict_proba(X_test_input)[:, 1] / n_splits

    # --- 2. XGBoost ---
    xgb_clf = MODEL_REGISTRY["xgboost"](random_state=42 + fold)
    xgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_preds["xgboost"][val_idx] = xgb_clf.predict_proba(X_val)[:, 1]
    test_preds["xgboost"] += xgb_clf.predict_proba(X_test_input)[:, 1] / n_splits

    # --- 3. CatBoost ---
    cb_clf = MODEL_REGISTRY["catboost"](random_state=42 + fold)
    cb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_preds["catboost"][val_idx] = cb_clf.predict_proba(X_val)[:, 1]
    test_preds["catboost"] += cb_clf.predict_proba(X_test_input)[:, 1] / n_splits

    # --- 4. Random Forest ---
    rf_clf = MODEL_REGISTRY["random_forest"](random_state=42 + fold)
    rf_clf.fit(X_tr, y_tr)
    oof_preds["random_forest"][val_idx] = rf_clf.predict_proba(X_val)[:, 1]
    test_preds["random_forest"] += rf_clf.predict_proba(X_test_input)[:, 1] / n_splits

    # --- 5. Extra Trees ---
    et_clf = MODEL_REGISTRY["extra_trees"](random_state=42 + fold)
    et_clf.fit(X_tr, y_tr)
    oof_preds["extra_trees"][val_idx] = et_clf.predict_proba(X_val)[:, 1]
    test_preds["extra_trees"] += et_clf.predict_proba(X_test_input)[:, 1] / n_splits

    # --- 6. Calibrated SVM ---
    scaler_svm = StandardScaler()
    X_tr_svm = scaler_svm.fit_transform(X_tr)
    X_val_svm = scaler_svm.transform(X_val)
    X_te_svm = scaler_svm.transform(X_test_input)

    svm_clf = MODEL_REGISTRY["svm"](random_state=42 + fold)
    svm_clf.fit(X_tr_svm, y_tr)
    oof_preds["svm"][val_idx] = svm_clf.predict_proba(X_val_svm)[:, 1]
    test_preds["svm"] += svm_clf.predict_proba(X_te_svm)[:, 1] / n_splits

    # --- 7. Regularized Logistic Regression ---
    scaler_lr = StandardScaler()
    X_tr_lr = scaler_lr.fit_transform(X_tr)
    X_val_lr = scaler_lr.transform(X_val)
    X_te_lr = scaler_lr.transform(X_test_input)

    lr_clf = MODEL_REGISTRY["logistic_regression"](random_state=42 + fold)
    lr_clf.fit(X_tr_lr, y_tr)
    oof_preds["logistic_regression"][val_idx] = lr_clf.predict_proba(X_val_lr)[:, 1]
    test_preds["logistic_regression"] += lr_clf.predict_proba(X_te_lr)[:, 1] / n_splits

    # --- 8. PyTorch ChemoTabularResNet with Label Smoothing ---
    scaler_nn = StandardScaler()
    X_tr_nn = torch.tensor(scaler_nn.fit_transform(X_tr), dtype=torch.float32)
    y_tr_nn = torch.tensor(y_tr, dtype=torch.float32)
    X_val_nn = torch.tensor(scaler_nn.transform(X_val), dtype=torch.float32)
    y_val_nn = torch.tensor(y_val, dtype=torch.float32)
    X_te_nn = torch.tensor(scaler_nn.transform(X_test_input), dtype=torch.float32)

    label_smoothing = 0.05
    y_tr_smooth = y_tr_nn * (1.0 - label_smoothing) + 0.5 * label_smoothing

    torch.manual_seed(42 + fold)
    nn_model = ChemoTabularResNet(
        in_features=X_tr.shape[1],
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.25,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-5)

    dataset_tr = TensorDataset(X_tr_nn, y_tr_smooth)
    loader_tr = DataLoader(dataset_tr, batch_size=48, shuffle=True)

    best_val_auc = -1.0
    best_val_probs = np.zeros(len(y_val))
    best_te_probs = np.zeros(n_test)

    for epoch in range(45):
        nn_model.train()
        for bx, by in loader_tr:
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            v_logits = nn_model(X_val_nn)
            v_probs = torch.sigmoid(v_logits).cpu().numpy()
            v_auc = roc_auc_score(y_val, v_probs)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_val_probs = v_probs
                t_logits = nn_model(X_te_nn)
                best_te_probs = torch.sigmoid(t_logits).cpu().numpy()

    oof_preds["chemo_resnet"][val_idx] = best_val_probs
    test_preds["chemo_resnet"] += best_te_probs / n_splits


# =========================================================================
# 4. Ensemble Weight Optimization & Inference (Rank Normalized)
# =========================================================================
oof_ranked = {m: rankdata(oof_preds[m]) / len(oof_preds[m]) for m in model_names}
test_ranked = {m: rankdata(test_preds[m]) / len(test_preds[m]) for m in model_names}


def ensemble_loss(weights: np.ndarray) -> float:
    w = np.clip(weights, 0.0, None)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones(len(model_names)) / len(model_names)

    blend = np.zeros(n_train, dtype=np.float64)
    for i, m in enumerate(model_names):
        blend += w[i] * oof_ranked[m]
    return -roc_auc_score(y_all, blend)


init_weights = np.ones(len(model_names)) / len(model_names)
res = minimize(
    ensemble_loss,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 1000, "adaptive": True},
)

raw_weights = np.clip(res.x, 0.0, None)
if np.sum(raw_weights) > 0:
    optimal_weights = raw_weights / np.sum(raw_weights)
else:
    optimal_weights = init_weights

final_oof_probs = np.zeros(n_train, dtype=np.float64)
final_test_probs = np.zeros(n_test, dtype=np.float64)

for i, m in enumerate(model_names):
    final_oof_probs += optimal_weights[i] * oof_ranked[m]
    final_test_probs += optimal_weights[i] * test_ranked[m]

final_test_probs = np.clip(final_test_probs, 0.0, 1.0)
final_validation_score = float(roc_auc_score(y_all, final_oof_probs))


# =========================================================================
# 5. Format and Export Submission
# =========================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_probs)), "Class": final_test_probs}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert len(submission_df) == len(test_df), "Submission row count mismatch!"
assert list(submission_df.columns) == ["id", "Class"], "Incorrect columns!"
assert not submission_df.isnull().any().any(), "Submission contains nulls!"

print(f"Final Validation Score: {final_validation_score}")
