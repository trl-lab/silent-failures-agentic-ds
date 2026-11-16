import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

# Set random seeds for deterministic reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


# ---------------------------------------------------------
# 1. Feature Engineering & Dataset Preparation
# ---------------------------------------------------------
def extract_physics_features(df):
    data = df.copy()
    for i in range(1, 5):
        tau = data[f"tau{i}"]
        g = data[f"g{i}"]
        p = data[f"p{i}"]
        data[f"tau_g_ratio_{i}"] = tau / (g + 1e-8)
        data[f"tau_g_prod_{i}"] = tau * g
        data[f"inertia_proxy_{i}"] = (tau**2) * g
        data[f"energy_load_{i}"] = tau * np.abs(p)
        data[f"p_g_ratio_{i}"] = p / (g + 1e-8)
        data[f"response_index_{i}"] = (tau * p) / (g + 1e-8)

    consumer_taus = data[["tau2", "tau3", "tau4"]]
    consumer_gs = data[["g2", "g3", "g4"]]
    consumer_ps = data[["p2", "p3", "p4"]]
    consumer_ratios = data[["tau_g_ratio_2", "tau_g_ratio_3", "tau_g_ratio_4"]]

    data["cons_tau_mean"] = consumer_taus.mean(axis=1)
    data["cons_tau_std"] = consumer_taus.std(axis=1)
    data["cons_tau_max"] = consumer_taus.max(axis=1)
    data["cons_tau_min"] = consumer_taus.min(axis=1)
    data["cons_tau_spread"] = data["cons_tau_max"] - data["cons_tau_min"]

    data["cons_g_mean"] = consumer_gs.mean(axis=1)
    data["cons_g_std"] = consumer_gs.std(axis=1)
    data["cons_g_max"] = consumer_gs.max(axis=1)
    data["cons_g_min"] = consumer_gs.min(axis=1)

    data["cons_p_sum"] = consumer_ps.sum(axis=1)
    data["cons_p_std"] = consumer_ps.std(axis=1)
    data["power_imbalance"] = np.abs(data["p1"] + data["cons_p_sum"])

    data["prod_cons_tau_ratio"] = data["tau1"] / (data["cons_tau_mean"] + 1e-8)
    data["prod_cons_g_ratio"] = data["g1"] / (data["cons_g_mean"] + 1e-8)
    data["prod_cons_delay_ratio"] = data["tau_g_ratio_1"] / (
        consumer_ratios.mean(axis=1) + 1e-8
    )

    all_ratios = data[[f"tau_g_ratio_{i}" for i in range(1, 5)]]
    data["global_ratio_sum"] = all_ratios.sum(axis=1)
    data["global_ratio_max"] = all_ratios.max(axis=1)
    data["global_ratio_std"] = all_ratios.std(axis=1)

    critical_idx = 0
    for i in range(1, 5):
        critical_idx += (data[f"tau{i}"] * (data[f"p{i}"] ** 2)) / (
            data[f"g{i}"] + 1e-8
        )
    data["critical_stability_index"] = critical_idx

    global_dissipation = 0
    abs_p1 = np.abs(data["p1"]) + 1e-8
    for i in range(1, 5):
        global_dissipation += data[f"g{i}"] / (data[f"tau{i}"] + 1e-8)
        data[f"norm_p_{i}"] = data[f"p{i}"] / abs_p1
        for j in range(1, 5):
            data[f"g{i}_tau{j}"] = data[f"g{i}"] * data[f"tau{j}"]

    data["global_dissipation"] = global_dissipation
    return data


def load_data():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train_feat = extract_physics_features(train_raw)
    test_feat = extract_physics_features(test_raw)

    if "stabf" in train_feat.columns:
        if train_feat["stabf"].dtype == object or isinstance(
            train_feat["stabf"].iloc[0], str
        ):
            y = (
                (train_feat["stabf"].astype(str).str.lower() == "unstable")
                .astype(int)
                .values
            )
        else:
            y = train_feat["stabf"].astype(int).values
    elif "stab" in train_feat.columns:
        y = (train_feat["stab"] > 0).astype(int).values
    else:
        raise ValueError("Target not found.")

    drop_cols = [c for c in ["stab", "stabf", "id"] if c in train_feat.columns]
    X = train_feat.drop(columns=drop_cols)

    test_drop_cols = [c for c in ["stab", "stabf", "id"] if c in test_feat.columns]
    X_test = test_feat.drop(columns=test_drop_cols)

    return X, y, X_test


X, y, X_test = load_data()
input_dim = X.shape[1]
n_samples = len(X)
n_test = len(X_test)


# ---------------------------------------------------------
# 2. PyTorch Model Architecture & Focal Loss
# ---------------------------------------------------------
class GLUBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)
        self.bn = nn.BatchNorm1d(out_features * 2)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        h = self.fc(x)
        h = self.bn(h)
        gate, val = torch.chunk(h, 2, dim=-1)
        out = val * torch.sigmoid(gate)
        out = self.dropout(out)
        return out + res


class PhysicsGLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.glu1 = GLUBlock(hidden_dim, hidden_dim, dropout)
        self.glu2 = GLUBlock(hidden_dim, hidden_dim, dropout)
        self.glu3 = GLUBlock(hidden_dim, hidden_dim // 2, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.glu1(x)
        x = self.glu2(x)
        x = self.glu3(x)
        logits = self.head(x)
        return logits.squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


# ---------------------------------------------------------
# 3. Model Training & Cross-Validation
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_hgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_et = np.zeros(n_samples)
oof_net = np.zeros(n_samples)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_hgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_et = np.zeros(n_test)
test_preds_net = np.zeros(n_test)

X_np = X.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_clf.predict_proba(X_va)[:, 1]
    test_preds_lgb += lgb_clf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 2: XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)[:, 1]
    test_preds_xgb += xgb_clf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 3: HistGradientBoosting
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=63,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42 + fold,
    )
    hgb_clf.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_clf.predict_proba(X_va)[:, 1]
    test_preds_hgb += hgb_clf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 4: CatBoost
    cat_clf = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
    )
    cat_clf.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_clf.predict_proba(X_va)[:, 1]
    test_preds_cat += cat_clf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 5: ExtraTrees
    et_clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=5,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_clf.fit(X_tr, y_tr)
    oof_et[val_idx] = et_clf.predict_proba(X_va)[:, 1]
    test_preds_et += et_clf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 6: PyTorch PhysicsGLUNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_np[train_idx])
    X_va_scaled = scaler.transform(X_np[val_idx])
    X_te_scaled = scaler.transform(X_test_np)

    tr_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    tr_loader = DataLoader(
        tr_ds, batch_size=128, shuffle=True, drop_last=False, num_workers=0
    )
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)

    model = PhysicsGLUNet(input_dim=input_dim, hidden_dim=256, dropout=0.2).to(device)
    criterion = FocalLoss(gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val_auc = -1.0
    best_val_preds = None
    best_fold_test_preds = None

    for epoch in range(30):
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
        val_probs = []
        with torch.no_grad():
            for bx, _ in va_loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)

        val_probs = np.array(val_probs)
        val_auc = roc_auc_score(y_va, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_preds = val_probs

            te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                best_fold_test_preds = torch.sigmoid(model(te_tensor)).cpu().numpy()

    oof_net[val_idx] = best_val_preds
    test_preds_net += best_fold_test_preds / skf.n_splits


# ---------------------------------------------------------
# 4. Ensembling via Constrained Blending & Submission
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_hgb, oof_cat, oof_et, oof_net])
test_matrix = np.column_stack(
    [
        test_preds_lgb,
        test_preds_xgb,
        test_preds_hgb,
        test_preds_cat,
        test_preds_et,
        test_preds_net,
    ]
)


def loss_func(weights):
    weights = np.array(weights)
    weights = np.maximum(0, weights)
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0
    weights /= sum_w
    blend_oof = oof_matrix @ weights
    return -roc_auc_score(y, blend_oof)


initial_weights = [0.2, 0.2, 0.15, 0.2, 0.1, 0.15]
opt_res = minimize(loss_func, initial_weights, method="Nelder-Mead")
best_w = np.maximum(0, opt_res.x)
best_w /= np.sum(best_w)

final_oof_blend = oof_matrix @ best_w
final_cv_score = roc_auc_score(y, final_oof_blend)

final_test_preds = test_matrix @ best_w

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "stabf": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_cv_score}")