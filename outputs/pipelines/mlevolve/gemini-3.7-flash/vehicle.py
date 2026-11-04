import json
import math
import os
import warnings
from catboost import CatBoostClassifier
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Seed
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# -----------------------------------------------------------------------------
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = (
    "class"
    if "class" in train_df.columns
    else [c for c in train_df.columns if c not in test_df.columns][0]
)
id_col = "id" if "id" in test_df.columns else None

raw_feature_cols = [
    c
    for c in train_df.columns
    if c not in [target_col, id_col] and pd.api.types.is_numeric_dtype(train_df[c])
]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[target_col].astype(str))
classes = list(label_encoder.classes_)
num_classes = len(classes)

test_ids = (
    test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))
)


def engineer_features(df, feature_names):
    """Extracts geometric invariants, cross-moment interactions,

    and shape asymmetry ratios from vehicle silhouette descriptors.
    """
    data = df[feature_names].copy()
    norm_cols = {
        c: c.lower().replace(" ", "_").replace(".", "_").replace("-", "_")
        for c in feature_names
    }
    df_renamed = data.rename(columns=norm_cols)

    eps = 1e-6
    engineered = pd.DataFrame(index=df.index)

    def get_col(*patterns):
        for pattern in patterns:
            matches = [c for c in df_renamed.columns if pattern in c]
            if len(matches) > 0:
                return df_renamed[matches[0]]
        return None

    compactness = get_col("compactness")
    circularity = get_col("circularity")
    dist_circ = get_col("distance_circularity", "dist_circ")
    radius_ratio = get_col("radius_ratio")
    pr_axis_aspect = get_col("pr_axis_aspect", "pr_axis_aspect_ratio")
    max_length_aspect = get_col("max_length_aspect", "max_length_aspect_ratio")
    scatter_ratio = get_col("scatter_ratio")
    elongatedness = get_col("elongatedness")
    pr_axis_rect = get_col("pr_axis_rect", "pr_axis_rectangularity")
    max_len_rect = get_col("max_length_rect", "max_length_rectangularity")
    scaled_var_maj = get_col(
        "scaled_variance_along_major", "scaled_var_maj", "variance_major"
    )
    scaled_var_min = get_col(
        "scaled_variance_along_minor", "scaled_var_min", "variance_minor"
    )
    gyration = get_col("scaled_radius_of_gyration", "gyration")
    skew_maj = get_col("skewness_about_major", "skew_maj")
    skew_min = get_col("skewness_about_minor", "skew_min")
    kurt_min = get_col("kurtosis_about_minor", "kurt_min")
    kurt_maj = get_col("kurtosis_about_major", "kurt_maj")
    hollows_ratio = get_col("hollows_ratio", "hollows")

    # A. Eccentricity & Aspect Ratio Relationships
    if pr_axis_aspect is not None:
        engineered["eccentricity_pr_axis"] = np.sqrt(
            np.clip(1.0 - np.square(pr_axis_aspect / 100.0), 0.0, 1.0)
        )
    if pr_axis_aspect is not None and max_length_aspect is not None:
        engineered["aspect_ratio_diff"] = np.abs(pr_axis_aspect - max_length_aspect)
        engineered["aspect_ratio_prod"] = (pr_axis_aspect * max_length_aspect) / 100.0
        engineered["aspect_ratio_quotient"] = pr_axis_aspect / (max_length_aspect + eps)

    # B. Rectangularity & Compactness Interactions
    if pr_axis_rect is not None and max_len_rect is not None:
        engineered["rect_discrepancy"] = np.abs(pr_axis_rect - max_len_rect)
        engineered["rect_ratio"] = pr_axis_rect / (max_len_rect + eps)
    if pr_axis_rect is not None and pr_axis_aspect is not None:
        engineered["rect_aspect_interaction"] = (pr_axis_rect * pr_axis_aspect) / 100.0
    if compactness is not None and circularity is not None:
        engineered["compactness_to_circularity"] = compactness / (circularity + eps)
        engineered["isoperimetric_approx"] = (compactness * circularity) / 1000.0
        engineered["isoperimetric_quotient"] = np.square(circularity) / (np.square(compactness) + eps)
        engineered["compactness_circularity_diff"] = compactness - circularity
    if dist_circ is not None and circularity is not None:
        engineered["circularity_disparity"] = dist_circ / (circularity + eps)
    if dist_circ is not None and compactness is not None:
        engineered["dist_circ_compactness_ratio"] = dist_circ / (compactness + eps)

    # C. Variance & Inertia Moment Features
    if scaled_var_maj is not None and scaled_var_min is not None:
        engineered["major_minor_var_ratio"] = scaled_var_maj / (scaled_var_min + eps)
        engineered["total_scaled_var"] = scaled_var_maj + scaled_var_min
        engineered["var_diff"] = np.abs(scaled_var_maj - scaled_var_min)
        engineered["var_anisotropy"] = (scaled_var_maj - scaled_var_min) / (
            scaled_var_maj + scaled_var_min + eps
        )
    if gyration is not None and scatter_ratio is not None:
        engineered["gyration_scatter_composite"] = (gyration * scatter_ratio) / 1000.0
        engineered["gyration_scatter_ratio"] = gyration / (scatter_ratio + eps)
    if gyration is not None and radius_ratio is not None:
        engineered["gyration_radius_ratio"] = gyration / (radius_ratio + eps)
    if gyration is not None and elongatedness is not None:
        engineered["gyration_elongation_ratio"] = gyration / (elongatedness + eps)

    # D. Higher-Order Moment Asymmetry (Skewness & Kurtosis)
    if skew_maj is not None and kurt_min is not None:
        engineered["asymmetry_maj_min"] = skew_maj / (kurt_min + eps)
    if skew_min is not None and kurt_maj is not None:
        engineered["asymmetry_min_maj"] = skew_min / (kurt_maj + eps)
    if skew_maj is not None and skew_min is not None:
        engineered["skewness_composite"] = np.sqrt(
            np.square(skew_maj) + np.square(skew_min)
        )
        engineered["skew_ratio"] = skew_maj / (skew_min + eps)
        engineered["skewness_anisotropy"] = (skew_maj - skew_min) / (
            np.abs(skew_maj) + np.abs(skew_min) + eps
        )
    if kurt_maj is not None and kurt_min is not None:
        engineered["kurtosis_harmonic_mean"] = (
            2.0 * (kurt_maj * kurt_min) / (kurt_maj + kurt_min + eps)
        )
        engineered["kurt_ratio"] = kurt_maj / (kurt_min + eps)
        engineered["kurtosis_anisotropy"] = (kurt_maj - kurt_min) / (
            kurt_maj + kurt_min + eps
        )
    if skew_maj is not None and kurt_maj is not None:
        engineered["skew_kurt_maj_ratio"] = skew_maj / (kurt_maj + eps)
    if skew_min is not None and kurt_min is not None:
        engineered["skew_kurt_min_ratio"] = skew_min / (kurt_min + eps)

    # E. Hollows & Elongation Dynamics
    if hollows_ratio is not None:
        engineered["solidity"] = 1.0 - (hollows_ratio / 100.0)
    if hollows_ratio is not None and elongatedness is not None:
        engineered["hollow_elongation_interaction"] = hollows_ratio * elongatedness
        engineered["hollow_elongation_ratio"] = hollows_ratio / (elongatedness + eps)
    if hollows_ratio is not None and compactness is not None:
        engineered["hollow_compactness_ratio"] = hollows_ratio / (compactness + eps)
    if radius_ratio is not None and elongatedness is not None:
        engineered["radius_elongation_prod"] = (radius_ratio * elongatedness) / 100.0
    if scatter_ratio is not None and elongatedness is not None:
        engineered["scatter_elongation_prod"] = (scatter_ratio * elongatedness) / 100.0

    # F. Non-linear transforms for original raw features
    for col in feature_names:
        val = data[col].values
        engineered[f"{col}_log1p"] = np.log1p(np.clip(val, a_min=0, a_max=None))
        engineered[f"{col}_sq"] = np.square(val / 100.0)
        engineered[f"{col}_sqrt"] = np.sqrt(np.clip(val, a_min=0, a_max=None))

    combined_df = pd.concat([data, engineered], axis=1)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.fillna(combined_df.median())
    return combined_df


X_train_df = engineer_features(train_df, raw_feature_cols)
X_test_df = engineer_features(test_df, raw_feature_cols)

feature_names = list(X_train_df.columns)
X_train_raw = X_train_df.values
X_test_raw = X_test_df.values

n_samples, input_dim = X_train_raw.shape
n_test = len(X_test_df)


# -----------------------------------------------------------------------------
# 2. Model Architectures & Builders
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout1(self.act1(self.ln1(self.fc1(x))))
        out = self.dropout2(self.act2(self.ln2(self.fc2(out))))
        return residual + out


class SilhouetteResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class FeatureGatedTabularNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.gating(x)
        gated_x = x * gates
        features = self.encoder(gated_x)
        return self.classifier(features)


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(preds, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 80,
    min_lr: float = 1e-6,
):
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optimizer, scheduler


def build_lightgbm_classifier(num_classes: int = 4, random_state: int = 42, **kwargs):
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "n_estimators": 350,
        "learning_rate": 0.025,
        "num_leaves": 28,
        "max_depth": 5,
        "min_child_samples": 12,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


def build_xgboost_classifier(num_classes: int = 4, random_state: int = 42, **kwargs):
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "n_estimators": 350,
        "learning_rate": 0.025,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "eval_metric": "mlogloss",
        "early_stopping_rounds": 40,
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def build_extratrees_classifier(random_state: int = 42, **kwargs):
    params = {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return ExtraTreesClassifier(**params)


def build_catboost_classifier(random_state: int = 42, **kwargs):
    params = {
        "iterations": 400,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }
    params.update(kwargs)
    return CatBoostClassifier(**params)


def build_randomforest_classifier(random_state: int = 42, **kwargs):
    params = {
        "n_estimators": 350,
        "max_depth": 14,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def build_svm_classifier(random_state: int = 42, **kwargs):
    base_svc = SVC(C=3.5, kernel="rbf", gamma="scale", probability=True, random_state=random_state)
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)


def build_poly_svm_classifier(random_state: int = 42, **kwargs):
    base_svc = SVC(C=2.5, kernel="poly", degree=2, coef0=1.0, gamma="scale", probability=True, random_state=random_state)
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)


def build_linear_classifier(random_state: int = 42, **kwargs):
    base_lr = LogisticRegression(C=1.5, penalty="l2", solver="lbfgs", max_iter=1000, random_state=random_state)
    return CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv=3)


def train_neural_model(
    model,
    X_tr_np,
    y_tr_np,
    X_va_np,
    y_va_np,
    X_te_np,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    smoothing=0.05,
):
    model = model.to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, lr=lr, weight_decay=weight_decay, epochs=epochs
    )

    tr_tensor_x = torch.tensor(X_tr_np, dtype=torch.float32)
    tr_tensor_y = torch.tensor(y_tr_np, dtype=torch.long)
    va_tensor_x = torch.tensor(X_va_np, dtype=torch.float32).to(device)
    te_tensor_x = torch.tensor(X_te_np, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(tr_tensor_x, tr_tensor_y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(va_tensor_x)
            va_loss = criterion(
                va_logits, torch.tensor(y_va_np, dtype=torch.long).to(device)
            ).item()

        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_probs = torch.softmax(model(va_tensor_x), dim=-1).detach().cpu().numpy()
        te_probs = torch.softmax(model(te_tensor_x), dim=-1).detach().cpu().numpy()

    return va_probs, te_probs


# -----------------------------------------------------------------------------
# 3. Stratified 10-Fold Cross-Validation Training Loop
# -----------------------------------------------------------------------------
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_resnet = np.zeros((n_samples, num_classes))
test_resnet = np.zeros((n_test, num_classes))

oof_lgb = np.zeros((n_samples, num_classes))
test_lgb = np.zeros((n_test, num_classes))

oof_xgb = np.zeros((n_samples, num_classes))
test_xgb = np.zeros((n_test, num_classes))

oof_cb = np.zeros((n_samples, num_classes))
test_cb = np.zeros((n_test, num_classes))

oof_rf = np.zeros((n_samples, num_classes))
test_rf = np.zeros((n_test, num_classes))

oof_et = np.zeros((n_samples, num_classes))
test_et = np.zeros((n_test, num_classes))

oof_svm = np.zeros((n_samples, num_classes))
test_svm = np.zeros((n_test, num_classes))

oof_poly_svm = np.zeros((n_samples, num_classes))
test_poly_svm = np.zeros((n_test, num_classes))

oof_lr = np.zeros((n_samples, num_classes))
test_lr = np.zeros((n_test, num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    X_tr_raw, y_tr = X_train_raw[train_idx], y_train[train_idx]
    X_va_raw, y_va = X_train_raw[val_idx], y_train[val_idx]

    # Fold-isolated preprocessing
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_va_sc = scaler.transform(X_va_raw)
    X_te_sc = scaler.transform(X_test_raw)

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_tr_pt = pt.fit_transform(X_tr_raw)
    X_va_pt = pt.transform(X_va_raw)
    X_te_pt = pt.transform(X_test_raw)

    # Fold-isolated LDA and PCA projections
    lda = LinearDiscriminantAnalysis(n_components=min(num_classes - 1, X_tr_pt.shape[1]))
    X_tr_lda = lda.fit_transform(X_tr_pt, y_tr)
    X_va_lda = lda.transform(X_va_pt)
    X_te_lda = lda.transform(X_te_pt)

    pca = PCA(n_components=6, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_tr_pt)
    X_va_pca = pca.transform(X_va_pt)
    X_te_pca = pca.transform(X_te_pt)

    # Composite feature representations
    X_tr_augmented = np.hstack([X_tr_raw, X_tr_lda, X_tr_pca])
    X_va_augmented = np.hstack([X_va_raw, X_va_lda, X_va_pca])
    X_te_augmented = np.hstack([X_test_raw, X_te_lda, X_te_pca])

    X_tr_norm_aug = np.hstack([X_tr_pt, X_tr_lda, X_tr_pca])
    X_va_norm_aug = np.hstack([X_va_pt, X_va_lda, X_va_pca])
    X_te_norm_aug = np.hstack([X_te_pt, X_te_lda, X_te_pca])

    # 1. SilhouetteResNet
    resnet_model = SilhouetteResNet(
        input_dim=X_tr_norm_aug.shape[1],
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.2,
    )
    val_preds_rn, test_preds_rn = train_neural_model(
        resnet_model,
        X_tr_norm_aug,
        y_tr,
        X_va_norm_aug,
        y_va,
        X_te_norm_aug,
        epochs=80,
        batch_size=32,
        lr=1.2e-3,
        smoothing=0.05,
    )
    oof_resnet[val_idx] = val_preds_rn
    test_resnet += test_preds_rn / N_SPLITS

    # 2. LightGBM
    lgb_clf = build_lightgbm_classifier(
        num_classes=num_classes, random_state=RANDOM_STATE + fold
    )
    lgb_clf.fit(
        X_tr_augmented,
        y_tr,
        eval_set=[(X_va_augmented, y_va)],
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_idx] = lgb_clf.predict_proba(X_va_augmented)
    test_lgb += lgb_clf.predict_proba(X_te_augmented) / N_SPLITS

    # 3. XGBoost
    xgb_clf = build_xgboost_classifier(
        num_classes=num_classes, random_state=RANDOM_STATE + fold
    )
    xgb_clf.fit(X_tr_augmented, y_tr, eval_set=[(X_va_augmented, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_va_augmented)
    test_xgb += xgb_clf.predict_proba(X_te_augmented) / N_SPLITS

    # 4. CatBoost
    cb_clf = build_catboost_classifier(random_state=RANDOM_STATE + fold)
    cb_clf.fit(
        X_tr_augmented,
        y_tr,
        eval_set=(X_va_augmented, y_va),
        early_stopping_rounds=40,
        verbose=False,
    )
    oof_cb[val_idx] = cb_clf.predict_proba(X_va_augmented)
    test_cb += cb_clf.predict_proba(X_te_augmented) / N_SPLITS

    # 5. Random Forest
    rf_clf = build_randomforest_classifier(random_state=RANDOM_STATE + fold)
    rf_clf.fit(X_tr_augmented, y_tr)
    oof_rf[val_idx] = rf_clf.predict_proba(X_va_augmented)
    test_rf += rf_clf.predict_proba(X_te_augmented) / N_SPLITS

    # 6. ExtraTrees
    et_clf = build_extratrees_classifier(random_state=RANDOM_STATE + fold)
    et_clf.fit(X_tr_augmented, y_tr)
    oof_et[val_idx] = et_clf.predict_proba(X_va_augmented)
    test_et += et_clf.predict_proba(X_te_augmented) / N_SPLITS

    # 7. Support Vector Classifier (RBF Calibrated)
    svm_clf = build_svm_classifier(random_state=RANDOM_STATE + fold)
    svm_clf.fit(X_tr_norm_aug, y_tr)
    oof_svm[val_idx] = svm_clf.predict_proba(X_va_norm_aug)
    test_svm += svm_clf.predict_proba(X_te_norm_aug) / N_SPLITS

    # 8. Polynomial Support Vector Classifier (Calibrated)
    poly_svm_clf = build_poly_svm_classifier(random_state=RANDOM_STATE + fold)
    poly_svm_clf.fit(X_tr_norm_aug, y_tr)
    oof_poly_svm[val_idx] = poly_svm_clf.predict_proba(X_va_norm_aug)
    test_poly_svm += poly_svm_clf.predict_proba(X_te_norm_aug) / N_SPLITS

    # 9. Linear / Logistic Classifier (Calibrated)
    lr_clf = build_linear_classifier(random_state=RANDOM_STATE + fold)
    lr_clf.fit(X_tr_norm_aug, y_tr)
    oof_lr[val_idx] = lr_clf.predict_proba(X_va_norm_aug)
    test_lr += lr_clf.predict_proba(X_te_norm_aug) / N_SPLITS

# -----------------------------------------------------------------------------
# 4. Accuracy-Aligned Ensemble Optimization & Post-Processing
# -----------------------------------------------------------------------------
oof_list = [
    oof_resnet,
    oof_lgb,
    oof_xgb,
    oof_cb,
    oof_rf,
    oof_et,
    oof_svm,
    oof_poly_svm,
    oof_lr,
]
test_list = [
    test_resnet,
    test_lgb,
    test_xgb,
    test_cb,
    test_rf,
    test_et,
    test_svm,
    test_poly_svm,
    test_lr,
]
model_names = ["resnet", "lgb", "xgb", "cb", "rf", "et", "svm", "poly_svm", "lr"]
num_models = len(oof_list)

# Initial baseline log-loss optimization via SLSQP
def logloss_func(weights):
    weights = np.array(weights)
    if np.sum(weights) <= 0:
        return 999.0
    weights = weights / np.sum(weights)
    blend_probs = np.zeros_like(oof_list[0])
    for w, oof_p in zip(weights, oof_list):
        blend_probs += w * oof_p
    blend_probs = np.clip(blend_probs, 1e-15, 1.0 - 1e-15)
    blend_probs = blend_probs / blend_probs.sum(axis=1, keepdims=True)
    return log_loss(y_train, blend_probs)

init_weights = [1.0 / num_models] * num_models
bounds = [(0.0, 1.0) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    logloss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 300, "ftol": 1e-7},
)
slsqp_weights = opt_res.x / np.sum(opt_res.x)

# Direct Accuracy-aligned Nelder-Mead Optimization for weights and class multipliers
def accuracy_neg_objective(params):
    w = np.maximum(params[:num_models], 0.0)
    c = np.maximum(params[num_models:], 0.05)
    if np.sum(w) <= 0:
        return 0.0
    w = w / np.sum(w)
    blend = np.zeros_like(oof_list[0])
    for i in range(num_models):
        blend += w[i] * oof_list[i]
    scaled_probs = blend * c
    preds = np.argmax(scaled_probs, axis=1)
    acc = accuracy_score(y_train, preds)
    norm_blend = np.clip(blend / blend.sum(axis=1, keepdims=True), 1e-15, 1.0 - 1e-15)
    smooth_tie_breaker = log_loss(y_train, norm_blend)
    return -acc + 1e-5 * smooth_tie_breaker

best_acc = -1.0
best_params = None

initial_candidates = [
    np.concatenate([slsqp_weights, np.ones(num_classes)]),
    np.concatenate([[1.0 / num_models] * num_models, np.ones(num_classes)]),
]
for m_idx in range(num_models):
    one_hot_w = np.zeros(num_models)
    one_hot_w[m_idx] = 1.0
    initial_candidates.append(np.concatenate([one_hot_w, np.ones(num_classes)]))

for init_p in initial_candidates:
    res = minimize(
        accuracy_neg_objective,
        init_p,
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-5},
    )
    w_cand = np.maximum(res.x[:num_models], 0.0)
    c_cand = np.maximum(res.x[num_models:], 0.05)
    if np.sum(w_cand) > 0:
        w_cand = w_cand / np.sum(w_cand)
        cand_blend = np.zeros_like(oof_list[0])
        for i in range(num_models):
            cand_blend += w_cand[i] * oof_list[i]
        cand_preds = np.argmax(cand_blend * c_cand, axis=1)
        cand_acc = accuracy_score(y_train, cand_preds)
        if cand_acc > best_acc:
            best_acc = cand_acc
            best_params = (w_cand, c_cand)

optimal_weights, optimal_class_weights = best_params

oof_ensemble_probs = np.zeros_like(oof_list[0])
test_ensemble_probs = np.zeros_like(test_list[0])

for w, oof_p, test_p in zip(optimal_weights, oof_list, test_list):
    oof_ensemble_probs += w * oof_p
    test_ensemble_probs += w * test_p

oof_ensemble_preds = np.argmax(oof_ensemble_probs * optimal_class_weights, axis=1)
final_validation_accuracy = accuracy_score(y_train, oof_ensemble_preds)

test_pred_indices = np.argmax(test_ensemble_probs * optimal_class_weights, axis=1)
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
test_pred_labels = [idx_to_class[idx] for idx in test_pred_indices]

submission_df = pd.DataFrame({"id": test_ids, "class": test_pred_labels})
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_accuracy}")
