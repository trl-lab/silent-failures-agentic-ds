import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "cannabis"
id_col = "id"

# Identify non-feature target columns present in dataset
non_feature_cols = {
    id_col,
    target_col,
    "alcohol",
    "amphet",
    "amyl",
    "benzos",
    "caff",
    "choc",
    "coke",
    "crack",
    "ecstasy",
    "heroin",
    "ketamine",
    "legalh",
    "lsd",
    "meth",
    "mushrooms",
    "nicotine",
    "semer",
    "vsa",
}

feature_cols = [
    c for c in train_df.columns if c in test_df.columns and c not in non_feature_cols
]

# Target Label Encoding
le = LabelEncoder()
y_train_full = le.fit_transform(train_df[target_col])
num_classes = len(le.classes_)


# 2. Domain-Specific Feature Engineering Function
def create_domain_features(df):
    data = df.copy()

    p_cols = ["nscore", "escore", "oscore", "ascore", "cscore", "ss"]
    imp_col = (
        "impuslive"
        if "impuslive" in data.columns
        else ("impulsive" if "impulsive" in data.columns else None)
    )
    if imp_col:
        p_cols.append(imp_col)

    # Domain Psychological Indices
    if (
        "ss" in data.columns
        and imp_col
        and "oscore" in data.columns
        and "cscore" in data.columns
        and "ascore" in data.columns
    ):
        data["risk_taking_index"] = (data["ss"] + data[imp_col] + data["oscore"]) / (
            data["cscore"] + data["ascore"] + 10.0
        )
        data["psychoticism_proxy"] = (
            data["ss"] + data[imp_col] - data["ascore"] - data["cscore"]
        )

    if "nscore" in data.columns and "escore" in data.columns:
        data["emotional_instability"] = data["nscore"] - data["escore"]

    if "cscore" in data.columns and imp_col:
        data["constraint_index"] = data["cscore"] - data[imp_col]
        data["sensation_impulsive_prod"] = (
            data["ss"] * data[imp_col] if "ss" in data.columns else 0.0
        )

    if "oscore" in data.columns and "cscore" in data.columns:
        data["openness_conscientiousness_ratio"] = data["oscore"] / (
            data["cscore"] + 10.0
        )

    # Nonlinear transformations for continuous traits
    for col in p_cols:
        if col in data.columns:
            data[f"{col}_sq"] = data[col] ** 2
            data[f"{col}_abs"] = np.abs(data[col])

    return data


X_train_engineered = create_domain_features(train_df[feature_cols])
X_test_engineered = create_domain_features(test_df[feature_cols])

# Psychometric columns selection for fold-based latent PCA/KMeans features
p_cols = ["nscore", "escore", "oscore", "ascore", "cscore", "ss"]
imp_col = (
    "impuslive"
    if "impuslive" in train_df.columns
    else ("impulsive" if "impulsive" in train_df.columns else None)
)
if imp_col and imp_col not in p_cols:
    p_cols.append(imp_col)

p_cols_avail = [c for c in p_cols if c in train_df.columns]


# 3. Model Definitions
class ResidualBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and Dropout for tabular features."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    """Deep Residual Neural Network for Continuous Psychometric Tabular Features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss with optional label smoothing for hard sample focusing."""

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes_loc = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            target_one_hot = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            if self.label_smoothing > 0:
                target_one_hot = (
                    1.0 - self.label_smoothing
                ) * target_one_hot + self.label_smoothing / num_classes_loc

        focal_weight = (1.0 - probs) ** self.gamma
        loss = -target_one_hot * focal_weight * log_probs

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)

        return loss.sum(dim=-1).mean()


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 80,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )
    return optimizer, scheduler


def get_lgbm_model(num_classes: int, random_state: int = 42):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=20,
        max_depth=5,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )


def get_xgb_model(num_classes: int, random_state: int = 42):
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=0,
        eval_metric="mlogloss",
    )


def get_catboost_model(num_classes: int, random_state: int = 42):
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=600,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=0,
    )


# 4. Stratified K-Fold Training & Prediction Pipeline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_samples = len(train_df)
num_folds = skf.get_n_splits()

oof_lgb = np.zeros((num_samples, num_classes))
oof_xgb = np.zeros((num_samples, num_classes))
oof_cb = np.zeros((num_samples, num_classes))
oof_nn = np.zeros((num_samples, num_classes))

test_lgb = np.zeros((len(test_df), num_classes))
test_xgb = np.zeros((len(test_df), num_classes))
test_cb = np.zeros((len(test_df), num_classes))
test_nn = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train_engineered, y_train_full)
):
    X_tr = X_train_engineered.iloc[train_idx].copy()
    y_tr = y_train_full[train_idx]
    X_va = X_train_engineered.iloc[val_idx].copy()
    y_va = y_train_full[val_idx]
    X_te = X_test_engineered.copy()

    # Latent PCA & KMeans computed per fold to avoid data leakage
    scaler_p = StandardScaler()
    p_med = X_tr[p_cols_avail].median()
    tr_p_scaled = scaler_p.fit_transform(X_tr[p_cols_avail].fillna(p_med))
    va_p_scaled = scaler_p.transform(X_va[p_cols_avail].fillna(p_med))
    te_p_scaled = scaler_p.transform(X_te[p_cols_avail].fillna(p_med))

    pca = PCA(n_components=3, random_state=42)
    tr_pca = pca.fit_transform(tr_p_scaled)
    va_pca = pca.transform(va_p_scaled)
    te_pca = pca.transform(te_p_scaled)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    tr_km = kmeans.fit_transform(tr_p_scaled)
    va_km = kmeans.transform(va_p_scaled)
    te_km = kmeans.transform(te_p_scaled)

    for i in range(3):
        X_tr[f"pca_{i}"] = tr_pca[:, i]
        X_va[f"pca_{i}"] = va_pca[:, i]
        X_te[f"pca_{i}"] = te_pca[:, i]

    for i in range(5):
        X_tr[f"kmeans_dist_{i}"] = tr_km[:, i]
        X_va[f"kmeans_dist_{i}"] = va_km[:, i]
        X_te[f"kmeans_dist_{i}"] = te_km[:, i]

    # Per-demographic trait aggregations (Strictly fitted on training fold)
    demo_cols = [
        c for c in ["age", "gender", "country", "education", "ethnicity"] if c in X_tr.columns
    ]
    for d_col in demo_cols:
        for p_col in ["ss", "oscore", "cscore", "nscore"]:
            if p_col in X_tr.columns:
                group_stats = (
                    X_tr.groupby(d_col)[p_col].agg(["mean", "std"]).reset_index()
                )
                group_stats.rename(
                    columns={"mean": f"{p_col}_{d_col}_mean", "std": f"{p_col}_{d_col}_std"},
                    inplace=True,
                )

                X_tr = X_tr.merge(group_stats, on=d_col, how="left")
                X_va = X_va.merge(group_stats, on=d_col, how="left")
                X_te = X_te.merge(group_stats, on=d_col, how="left")

                X_tr[f"{p_col}_{d_col}_std"] = X_tr[f"{p_col}_{d_col}_std"].fillna(0)
                X_va[f"{p_col}_{d_col}_std"] = X_va[f"{p_col}_{d_col}_std"].fillna(0)
                X_te[f"{p_col}_{d_col}_std"] = X_te[f"{p_col}_{d_col}_std"].fillna(0)

                X_tr[f"{p_col}_dev_{d_col}"] = X_tr[p_col] - X_tr[f"{p_col}_{d_col}_mean"]
                X_va[f"{p_col}_dev_{d_col}"] = X_va[p_col] - X_va[f"{p_col}_{d_col}_mean"]
                X_te[f"{p_col}_dev_{d_col}"] = X_te[p_col] - X_te[f"{p_col}_{d_col}_mean"]

    # Impute missing values and scale features
    X_tr_imputed = X_tr.fillna(X_tr.median())
    X_va_imputed = X_va.fillna(X_tr.median())
    X_te_imputed = X_te.fillna(X_tr.median())

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imputed)
    X_va_scaled = scaler.transform(X_va_imputed)
    X_te_scaled = scaler.transform(X_te_imputed)

    # Train Model A: LightGBM
    lgb_model = get_lgbm_model(num_classes, random_state=42 + fold)
    lgb_model.fit(
        X_tr_scaled,
        y_tr,
        eval_set=[(X_va_scaled, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_probs_lgb = lgb_model.predict_proba(X_va_scaled)
    test_probs_lgb = lgb_model.predict_proba(X_te_scaled)

    # Train Model B: XGBoost
    xgb_model = get_xgb_model(num_classes, random_state=42 + fold)
    xgb_model.fit(X_tr_scaled, y_tr, eval_set=[(X_va_scaled, y_va)], verbose=False)
    val_probs_xgb = xgb_model.predict_proba(X_va_scaled)
    test_probs_xgb = xgb_model.predict_proba(X_te_scaled)

    # Train Model C: CatBoost
    cb_model = get_catboost_model(num_classes, random_state=42 + fold)
    cb_model.fit(
        X_tr_scaled,
        y_tr,
        eval_set=(X_va_scaled, y_va),
        early_stopping_rounds=50,
        verbose=0,
    )
    val_probs_cb = cb_model.predict_proba(X_va_scaled)
    test_probs_cb = cb_model.predict_proba(X_te_scaled)

    # Train Model D: TabularResNet (PyTorch)
    input_dim = X_tr_scaled.shape[1]
    nn_model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.25,
    ).to(device)

    criterion = MultiClassFocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer, scheduler = get_optimizer_and_scheduler(
        nn_model, lr=1e-3, weight_decay=1e-4, num_epochs=80
    )

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.long),
    )
    te_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

    tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=64, shuffle=False)
    te_loader = DataLoader(te_dataset, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_nn_state = None

    for epoch in range(80):
        nn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                logits = nn_model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
        val_loss /= len(va_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_state = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_state.items()})
    nn_model.eval()

    val_probs_nn = []
    with torch.no_grad():
        for bx, _ in va_loader:
            bx = bx.to(device)
            logits = nn_model(bx)
            probs = torch.softmax(logits, dim=-1)
            val_probs_nn.append(probs.cpu().numpy())
    val_probs_nn = np.vstack(val_probs_nn)

    test_probs_nn = []
    with torch.no_grad():
        for (bx,) in te_loader:
            bx = bx.to(device)
            logits = nn_model(bx)
            probs = torch.softmax(logits, dim=-1)
            test_probs_nn.append(probs.cpu().numpy())
    test_probs_nn = np.vstack(test_probs_nn)

    oof_lgb[val_idx] = val_probs_lgb
    oof_xgb[val_idx] = val_probs_xgb
    oof_cb[val_idx] = val_probs_cb
    oof_nn[val_idx] = val_probs_nn

    test_lgb += test_probs_lgb / num_folds
    test_xgb += test_probs_xgb / num_folds
    test_cb += test_probs_cb / num_folds
    test_nn += test_probs_nn / num_folds

# 5. Out-Of-Fold Metric Calculation & Powell Optimization for Ensemble Weights and Class Multipliers
def objective(params):
    m_w = np.clip(params[:4], 1e-4, None)
    m_w = m_w / np.sum(m_w)
    c_w = np.clip(params[4:], 1e-4, None)

    P_blend = (
        m_w[0] * oof_lgb
        + m_w[1] * oof_xgb
        + m_w[2] * oof_cb
        + m_w[3] * oof_nn
    )
    P_scaled = P_blend * c_w
    preds = np.argmax(P_scaled, axis=1)
    acc = accuracy_score(y_train_full, preds)
    return -acc

init_params = np.concatenate([
    np.array([0.35, 0.25, 0.25, 0.15]),
    np.ones(num_classes)
])

res = minimize(
    objective,
    init_params,
    method="Powell",
    options={"maxiter": 1000, "ftol": 1e-5}
)

best_params = res.x
best_m_w = np.clip(best_params[:4], 1e-4, None)
best_m_w = best_m_w / np.sum(best_m_w)
best_c_w = np.clip(best_params[4:], 1e-4, None)

oof_blend = (
    best_m_w[0] * oof_lgb
    + best_m_w[1] * oof_xgb
    + best_m_w[2] * oof_cb
    + best_m_w[3] * oof_nn
)
oof_preds = np.argmax(oof_blend * best_c_w, axis=1)
val_accuracy = accuracy_score(y_train_full, oof_preds)

test_blend = (
    best_m_w[0] * test_lgb
    + best_m_w[1] * test_xgb
    + best_m_w[2] * test_cb
    + best_m_w[3] * test_nn
)
test_preds = np.argmax(test_blend * best_c_w, axis=1)
test_labels = le.inverse_transform(test_preds)

sub_id = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame({"id": sub_id, "cannabis": test_labels})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_accuracy}")