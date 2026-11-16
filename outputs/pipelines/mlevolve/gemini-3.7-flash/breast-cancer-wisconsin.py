import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================
# 0. Global Setup and Seed Initialization
# =========================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Feature Engineering
# =========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    base_features = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]

    eps = 1e-8

    # Nuclear Pleomorphism & Relative Variation Features
    for feat in base_features:
        f1 = f"{feat}1"  # Mean
        f2 = f"{feat}2"  # SE
        f3 = f"{feat}3"  # Worst

        if f1 in df_out.columns and f3 in df_out.columns:
            df_out[f"{feat}_worst_to_mean_ratio"] = df_out[f3] / (df_out[f1] + eps)
            df_out[f"{feat}_worst_mean_diff"] = df_out[f3] - df_out[f1]

        if f1 in df_out.columns and f2 in df_out.columns:
            df_out[f"{feat}_relative_se"] = df_out[f2] / (df_out[f1] + eps)

        if f2 in df_out.columns and f3 in df_out.columns:
            df_out[f"{feat}_worst_to_se_ratio"] = df_out[f3] / (df_out[f2] + eps)

    # Geometric & Physical Invariants
    if "area1" in df_out.columns and "perimeter1" in df_out.columns:
        df_out["circularity1"] = (4 * np.pi * df_out["area1"]) / (
            df_out["perimeter1"] ** 2 + eps
        )
    if "area3" in df_out.columns and "perimeter3" in df_out.columns:
        df_out["circularity3"] = (4 * np.pi * df_out["area3"]) / (
            df_out["perimeter3"] ** 2 + eps
        )

    # Theoretical Area Discrepancy: pi * r^2 vs recorded area
    if "radius1" in df_out.columns and "area1" in df_out.columns:
        df_out["area_radius_discrepancy1"] = (np.pi * (df_out["radius1"] ** 2)) / (
            df_out["area1"] + eps
        )
    if "radius3" in df_out.columns and "area3" in df_out.columns:
        df_out["area_radius_discrepancy3"] = (np.pi * (df_out["radius3"] ** 2)) / (
            df_out["area3"] + eps
        )

    # Shape Complexity & Border Irregularity
    if "concavity1" in df_out.columns and "concave_points1" in df_out.columns:
        df_out["concavity_to_points_ratio1"] = df_out["concavity1"] / (
            df_out["concave_points1"] + eps
        )
    if "concavity3" in df_out.columns and "concave_points3" in df_out.columns:
        df_out["concavity_to_points_ratio3"] = df_out["concavity3"] / (
            df_out["concave_points3"] + eps
        )

    if "compactness1" in df_out.columns and "smoothness1" in df_out.columns:
        df_out["compactness_smoothness_prod1"] = (
            df_out["compactness1"] * df_out["smoothness1"]
        )
    if "compactness3" in df_out.columns and "smoothness3" in df_out.columns:
        df_out["compactness_smoothness_prod3"] = (
            df_out["compactness3"] * df_out["smoothness3"]
        )

    # Structural Malignancy Burden
    if all(col in df_out.columns for col in ["radius3", "texture3", "concavity3"]):
        df_out["malignancy_volume_proxy"] = (
            df_out["radius3"] * df_out["texture3"] * df_out["concavity3"]
        )

    # Handle numeric inconsistencies cleanly
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    for col in df_out.select_dtypes(include=[np.number]).columns:
        if df_out[col].isna().any():
            median_val = df_out[col].median()
            df_out[col] = df_out[col].fillna(median_val)

    return df_out


# Load raw data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Diagnosis"
if train_df[target_col].dtype == "object" or isinstance(
    train_df[target_col].iloc[0], str
):
    target_mapping = {"M": 1, "B": 0, "1": 1, "0": 0}
    train_df[target_col] = train_df[target_col].map(target_mapping).astype(int)
else:
    train_df[target_col] = train_df[target_col].astype(int)

# Apply Feature Engineering
train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

if "ID" in train_feat.columns:
    train_feat = train_feat.drop(columns=["ID"])
if "ID" in test_feat.columns:
    test_feat = test_feat.drop(columns=["ID"])

# Assign Stratified K-Fold splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_feat["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_feat, train_feat[target_col])):
    train_feat.loc[val_idx, "fold"] = fold


# =========================================================================
# 2. Model Architectures & Builders
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


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
            nn.Dropout(dropout_rate / 2),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def get_neural_network_components(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=64, num_blocks=2, dropout_rate=0.2
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def get_lightgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_model(random_state: int = 42) -> cb.CatBoostClassifier:
    return cb.CatBoostClassifier(
        iterations=350,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
    )


def get_linear_model(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=0.5,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )


def get_svm_model(random_state: int = 42) -> SVC:
    return SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=random_state,
    )


# =========================================================================
# 3. Neural Network Fold Trainer
# =========================================================================
def train_nn_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    input_dim: int,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    model, criterion, optimizer = get_neural_network_components(
        input_dim, lr=lr, weight_decay=1e-4
    )
    model.to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    tr_tensor_x = torch.tensor(X_tr, dtype=torch.float32)
    tr_tensor_y = torch.tensor(y_tr, dtype=torch.float32)
    va_tensor_x = torch.tensor(X_va, dtype=torch.float32)
    va_tensor_y = torch.tensor(y_va, dtype=torch.float32)
    te_tensor_x = torch.tensor(X_te, dtype=torch.float32)

    tr_dataset = TensorDataset(tr_tensor_x, tr_tensor_y)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

    best_va_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(va_tensor_x.to(device))
            va_loss = criterion(va_logits, va_tensor_y.to(device)).item()

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        va_probs = torch.sigmoid(model(va_tensor_x.to(device))).cpu().numpy().flatten()
        te_probs = torch.sigmoid(model(te_tensor_x.to(device))).cpu().numpy().flatten()

    return va_probs, te_probs


# =========================================================================
# 4. Cross-Validation Pipeline
# =========================================================================
fold_col = "fold"
feature_cols = [c for c in train_feat.columns if c not in [target_col, fold_col]]

num_folds = train_feat[fold_col].nunique()
n_train = len(train_feat)
n_test = len(test_feat)
input_dim = len(feature_cols)

models_list = ["lgb", "xgb", "cat", "linear", "svm", "resnet"]
oof_preds = {m: np.zeros(n_train) for m in models_list}
test_preds = {m: np.zeros((num_folds, n_test)) for m in models_list}

for fold in range(num_folds):
    train_mask = train_feat[fold_col] != fold
    val_mask = train_feat[fold_col] == fold
    val_indices = np.where(val_mask)[0]

    X_train_raw = train_feat.loc[train_mask, feature_cols].values
    y_train = train_feat.loc[train_mask, target_col].values
    X_val_raw = train_feat.loc[val_mask, feature_cols].values
    y_val = train_feat.loc[val_mask, target_col].values
    X_test_raw = test_feat[feature_cols].values

    # Strict isolation: Standardize features fitted only on training fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 1. LightGBM
    lgb_clf = get_lightgbm_model(random_state=SEED + fold)
    lgb_clf.fit(X_train_raw, y_train)
    oof_preds["lgb"][val_indices] = lgb_clf.predict_proba(X_val_raw)[:, 1]
    test_preds["lgb"][fold] = lgb_clf.predict_proba(X_test_raw)[:, 1]

    # 2. XGBoost
    xgb_clf = get_xgboost_model(random_state=SEED + fold)
    xgb_clf.fit(X_train_raw, y_train)
    oof_preds["xgb"][val_indices] = xgb_clf.predict_proba(X_val_raw)[:, 1]
    test_preds["xgb"][fold] = xgb_clf.predict_proba(X_test_raw)[:, 1]

    # 3. CatBoost
    cat_clf = get_catboost_model(random_state=SEED + fold)
    cat_clf.fit(X_train_raw, y_train)
    oof_preds["cat"][val_indices] = cat_clf.predict_proba(X_val_raw)[:, 1]
    test_preds["cat"][fold] = cat_clf.predict_proba(X_test_raw)[:, 1]

    # 4. Logistic Regression
    lr_clf = get_linear_model(random_state=SEED + fold)
    lr_clf.fit(X_train_scaled, y_train)
    oof_preds["linear"][val_indices] = lr_clf.predict_proba(X_val_scaled)[:, 1]
    test_preds["linear"][fold] = lr_clf.predict_proba(X_test_scaled)[:, 1]

    # 5. Support Vector Classifier
    svm_clf = get_svm_model(random_state=SEED + fold)
    svm_clf.fit(X_train_scaled, y_train)
    oof_preds["svm"][val_indices] = svm_clf.predict_proba(X_val_scaled)[:, 1]
    test_preds["svm"][fold] = svm_clf.predict_proba(X_test_scaled)[:, 1]

    # 6. Tabular ResNet
    nn_val, nn_test = train_nn_fold(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        input_dim=input_dim,
        epochs=50,
        batch_size=32,
        lr=1e-3,
    )
    oof_preds["resnet"][val_indices] = nn_val
    test_preds["resnet"][fold] = nn_test

# =========================================================================
# 5. Ensemble Optimization & Validation
# =========================================================================
y_true = train_feat[target_col].values


def loss_func(weights):
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-8)
    blend = sum(weights[i] * oof_preds[models_list[i]] for i in range(len(models_list)))
    return -roc_auc_score(y_true, blend)


init_weights = [1.0 / len(models_list)] * len(models_list)
bounds = [(0.0, 1.0) for _ in models_list]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
optimal_weights = opt_res.x / np.sum(opt_res.x)

oof_blend = sum(
    optimal_weights[i] * oof_preds[models_list[i]] for i in range(len(models_list))
)
final_score = roc_auc_score(y_true, oof_blend)

# =========================================================================
# 6. Test Inference & Submission Generation
# =========================================================================
test_blend = sum(
    optimal_weights[i] * np.mean(test_preds[models_list[i]], axis=0)
    for i in range(len(models_list))
)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_blend), dtype=int), "Diagnosis": test_blend}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
