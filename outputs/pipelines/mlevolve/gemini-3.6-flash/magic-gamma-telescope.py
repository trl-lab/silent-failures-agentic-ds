import math
import os
import random
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


# Reproducibility seed initialization
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)

# 1. Load Parquet Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Robust Target Parser
def parse_target(y_series):
    if y_series.dtype == "object" or isinstance(y_series.iloc[0], str):
        return (
            y_series.astype(str).str.lower().str.strip().str.startswith("g").astype(int)
        )
    return y_series.astype(int)


target = parse_target(train_df["class"])


# 2. Domain-Specific Hillas Parameter Feature Transformations
def create_hillas_features(df):
    data = df.copy()
    eps = 1e-6

    length = np.maximum(data["fLength"], eps)
    width = np.maximum(data["fWidth"], eps)

    # Log-scaled features for skewed variables
    data["fLogLength"] = np.log1p(np.maximum(data["fLength"], 0))
    data["fLogWidth"] = np.log1p(np.maximum(data["fWidth"], 0))
    data["fLogSize"] = np.log1p(np.maximum(data["fSize"], 0))
    data["fLogDist"] = np.log1p(np.maximum(data["fDist"], 0))
    data["fLogAlpha"] = np.log1p(np.maximum(data["fAlpha"], 0))

    # Exact trigonometric orientation components
    alpha_rad_exact = np.radians(data["fAlpha"])
    data["fCosAlpha"] = np.cos(alpha_rad_exact)
    data["fSinAlpha"] = np.sin(alpha_rad_exact)
    data["fCos2Alpha"] = np.cos(2 * alpha_rad_exact)

    # Aspect ratio and shower eccentricity
    data["fAspectRatio"] = width / length
    ecc_sq = 1.0 - (width / length) ** 2
    data["fEccentricity"] = np.sqrt(np.clip(ecc_sq, 0.0, 1.0))

    # Shower ellipse area and perimeter
    data["fArea"] = np.pi * length * width
    data["fLogArea"] = np.log1p(data["fArea"])
    data["fPerimeter"] = np.pi * (
        3 * (length + width) - np.sqrt((3 * length + width) * (length + 3 * width))
    )

    # Physical shape ratios
    data["fSizePerPerimeter"] = data["fSize"] / (data["fPerimeter"] + eps)

    # Light density & concentration profile
    size_linear = np.power(10.0, np.clip(data["fSize"], 0, 10))
    data["fDensity"] = size_linear / (data["fArea"] + eps)
    data["fLogDensity"] = data["fSize"] - np.log10(data["fArea"] + eps)

    conc = np.maximum(data["fConc"], eps)
    conc1 = np.maximum(data["fConc1"], eps)
    data["fConcDiff"] = conc - conc1
    data["fConcRatio"] = conc1 / conc

    # Geometry & orientation (impact parameter)
    alpha_rad = np.radians(np.clip(data["fAlpha"], 0, 90))
    dist = np.maximum(data["fDist"], eps)

    data["fMiss"] = dist * np.sin(alpha_rad)
    data["fDisplacement"] = dist * np.cos(alpha_rad)
    data["fMissDistRatio"] = data["fMiss"] / dist
    data["fLengthDistRatio"] = length / dist

    # Higher-order moment normalizations & asymmetry
    data["fM3LongNorm"] = data["fM3Long"] / length
    data["fM3TransNorm"] = data["fM3Trans"] / width
    data["fMomentMag"] = np.sqrt(data["fM3Long"] ** 2 + data["fM3Trans"] ** 2)
    data["fLogMomentMag"] = np.log1p(data["fMomentMag"])
    data["fAsymNorm"] = data["fAsym"] / length

    # Energy-to-geometry scaling ratios
    data["fLengthSizeRatio"] = length / (data["fSize"] + eps)
    data["fWidthSizeRatio"] = width / (data["fSize"] + eps)

    # Non-linear physical interactions
    data["fEnergyScaledAspectRatio"] = data["fSize"] * data["fAspectRatio"]
    data["fDirDensityProj"] = data["fSize"] / (data["fArea"] * np.cos(alpha_rad) + eps)
    data["fMomentAnisotropyRatio"] = data["fM3Long"] / (data["fM3Trans"] + eps)

    return data


train_processed = create_hillas_features(train_df)
test_processed = create_hillas_features(test_df)

feature_cols = [c for c in train_processed.columns if c not in ["class", "id"]]


# 3. Neural Architecture Definition: PyTorch TabularResNet
class FeatureAttention(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_features, in_features), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, num_features, hidden_dim=128, num_blocks=3, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(num_features),
            FeatureAttention(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


class TabularDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# 4. Multi-Model Cross-Validation & Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

X_train_all = train_processed[feature_cols]
X_test_all = test_processed[feature_cols]

cb_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 0,
}

oof_lgb = np.zeros(len(train_processed))
oof_xgb = np.zeros(len(train_processed))
oof_cat = np.zeros(len(train_processed))
oof_nn = np.zeros(len(train_processed))

test_lgb = np.zeros(len(test_processed))
test_xgb = np.zeros(len(test_processed))
test_cat = np.zeros(len(test_processed))
test_nn = np.zeros(len(test_processed))

print("Starting 5-Fold Cross-Validation Training & Evaluation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, target)):
    X_tr, y_tr = X_train_all.iloc[train_idx], target.iloc[train_idx]
    X_va, y_va = X_train_all.iloc[val_idx], target.iloc[val_idx]

    # --- 1. LightGBM ---
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test_all)[:, 1] / skf.n_splits
    model_lgb.booster_.save_model(f"./working/lgb_fold_{fold}.txt")

    # --- 2. XGBoost ---
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test_all)[:, 1] / skf.n_splits
    model_xgb.save_model(f"./working/xgb_fold_{fold}.json")

    # --- 3. CatBoost ---
    model_cat = CatBoostClassifier(**cb_params)
    model_cat.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )

    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test_all)[:, 1] / skf.n_splits
    model_cat.save_model(f"./working/cat_fold_{fold}.cbm")

    # --- 4. PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test_all)

    tr_ds = TabularDataset(X_tr_sc, y_tr.values)
    va_ds = TabularDataset(X_va_sc)
    te_ds = TabularDataset(X_te_sc)

    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, drop_last=False)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, drop_last=False)

    model_nn = TabularResNet(
        num_features=X_tr.shape[1], hidden_dim=128, num_blocks=3, dropout=0.15
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_val_loss = float("inf")
    best_nn_weights_path = f"./working/nn_fold_{fold}.pt"

    for epoch in range(40):
        model_nn.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model_nn(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Check validation loss for model checkpointing
        model_nn.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for bx, by in DataLoader(
                TabularDataset(X_va_sc, y_va.values), batch_size=128
            ):
                bx, by = bx.to(device), by.to(device)
                logits = model_nn(bx)
                val_loss += criterion(logits, by).item() * len(by)
                val_samples += len(by)
        val_loss /= val_samples

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_nn.state_dict(), best_nn_weights_path)

    # Load best checkpoint for inference
    model_nn.load_state_dict(torch.load(best_nn_weights_path))
    model_nn.eval()

    val_preds_nn = []
    with torch.no_grad():
        for bx in va_loader:
            bx = bx.to(device)
            logits = model_nn(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds_nn.extend(probs)
    oof_nn[val_idx] = np.array(val_preds_nn)

    test_preds_nn = []
    with torch.no_grad():
        for bx in te_loader:
            bx = bx.to(device)
            logits = model_nn(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds_nn.extend(probs)
    test_nn += np.array(test_preds_nn) / skf.n_splits

    fold_lgb_auc = roc_auc_score(y_va, oof_lgb[val_idx])
    fold_xgb_auc = roc_auc_score(y_va, oof_xgb[val_idx])
    fold_cat_auc = roc_auc_score(y_va, oof_cat[val_idx])
    fold_nn_auc = roc_auc_score(y_va, oof_nn[val_idx])
    print(
        f"Fold {fold+1} Finished | LightGBM AUC: {fold_lgb_auc:.5f} | XGBoost AUC: {fold_xgb_auc:.5f} | CatBoost AUC: {fold_cat_auc:.5f} | TabularResNet AUC: {fold_nn_auc:.5f}"
    )

# 5. Rank-Averaging Ensembling Logic with SciPy Dynamic Weight Optimization
oof_lgb_rank = pd.Series(oof_lgb).rank(pct=True).values
oof_xgb_rank = pd.Series(oof_xgb).rank(pct=True).values
oof_cat_rank = pd.Series(oof_cat).rank(pct=True).values
oof_nn_rank = pd.Series(oof_nn).rank(pct=True).values

oof_ranks = np.column_stack([oof_lgb_rank, oof_xgb_rank, oof_cat_rank, oof_nn_rank])

def objective(weights):
    weights = np.array(weights)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    blend_pred = np.dot(oof_ranks, weights)
    return -roc_auc_score(target, blend_pred)

init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x / np.sum(res.x)

oof_blend_rank = np.dot(oof_ranks, best_weights)
final_auc = roc_auc_score(target, oof_blend_rank)

test_lgb_rank = pd.Series(test_lgb).rank(pct=True).values
test_xgb_rank = pd.Series(test_xgb).rank(pct=True).values
test_cat_rank = pd.Series(test_cat).rank(pct=True).values
test_nn_rank = pd.Series(test_nn).rank(pct=True).values

test_ranks = np.column_stack([test_lgb_rank, test_xgb_rank, test_cat_rank, test_nn_rank])
test_blend_rank = np.dot(test_ranks, best_weights)

# 6. Generate Submission File
sub_id = test_df["id"] if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame({"id": sub_id, "class": test_blend_rank})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_auc}")
