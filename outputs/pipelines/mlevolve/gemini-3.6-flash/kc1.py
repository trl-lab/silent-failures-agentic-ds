import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. DATA LOADING AND FEATURE ENGINEERING
# ------------------------------------------------------------------------------

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "defects"
if target_col in train_df.columns:
    raw_target = train_df[target_col]
    if raw_target.dtype == object or raw_target.dtype == bool:
        y_train = (
            raw_target.astype(str)
            .str.lower()
            .isin(["true", "1", "t", "yes"])
            .astype(int)
            .values
        )
    else:
        y_train = raw_target.astype(int).values
    train_df = train_df.drop(columns=[target_col])
else:
    raise ValueError("Target column 'defects' not found in training data.")

if "id" in test_df.columns:
    test_ids = test_df["id"].values
    test_features_df = test_df.drop(columns=["id"])
else:
    test_ids = np.arange(len(test_df))
    test_features_df = test_df.copy()

if "id" in train_df.columns:
    train_features_df = train_df.drop(columns=["id"])
else:
    train_features_df = train_df.copy()


def engineer_software_metrics(df):
    df = df.copy()
    eps = 1e-6

    def get_series(col_name):
        if col_name in df.columns:
            return df[col_name].astype(float)
        return pd.Series(0.0, index=df.index)

    loc = get_series("loc")
    lo_code = get_series("lOCode")
    lo_comment = get_series("lOComment")
    lo_blank = get_series("lOBlank")
    lo_code_comment = get_series("lOCodeAndComment")

    total_lines = lo_code + lo_comment + lo_blank + eps
    df["comment_density"] = lo_comment / (lo_code + eps)
    df["blank_ratio"] = lo_blank / total_lines
    df["code_comment_ratio"] = lo_code_comment / (lo_code + eps)

    v_g = get_series("v(g)")
    ev_g = get_series("ev(g)")
    iv_g = get_series("iv(g)")
    branch = get_series("branchCount")

    df["cyclomatic_density"] = v_g / (lo_code + eps)
    df["essential_density"] = ev_g / (v_g + eps)
    df["design_density"] = iv_g / (v_g + eps)
    df["branch_density"] = branch / (lo_code + eps)

    uniq_op = get_series("uniq_Op")
    uniq_opnd = get_series("uniq_Opnd")
    total_op = get_series("total_Op")
    total_opnd = get_series("total_Opnd")

    df["operator_operand_ratio"] = total_op / (total_opnd + eps)
    df["unique_op_ratio"] = uniq_op / (uniq_opnd + eps)
    vocab = uniq_op + uniq_opnd
    df["vocab_size"] = vocab

    calc_length = uniq_op * np.log2(uniq_op + eps) + uniq_opnd * np.log2(
        uniq_opnd + eps
    )
    n_val = get_series("n")
    df["length_discrepancy"] = n_val - calc_length

    effort = get_series("e")
    df["effort_per_loc"] = effort / (lo_code + eps)
    d_val = get_series("d")
    df["difficulty_per_vocab"] = d_val / (vocab + eps)

    v_val = get_series("v")
    df["halstead_estimated_bugs"] = v_val / 3000.0
    df["halstead_time_efficiency_ratio"] = effort / (d_val * v_val + eps)
    df["mccabe_risk_density_index"] = (v_g / (loc + eps)) + (iv_g / (v_g + eps))
    df["halstead_length_residual_error"] = (n_val - calc_length) / (calc_length + eps)

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    for c in num_cols:
        min_val = df[c].min()
        if min_val >= 0:
            df[f"{c}_log1p"] = np.log1p(df[c])

    return df


X_train_raw = engineer_software_metrics(train_features_df)
X_test_raw = engineer_software_metrics(test_features_df)

X_train_np = X_train_raw.values
X_test_np = X_test_raw.values
n_samples, n_features = X_train_np.shape

# ------------------------------------------------------------------------------
# 2. MODEL DEFINITIONS (PYTORCH ARCHITECTURE & HYPERPARAMETERS)
# ------------------------------------------------------------------------------


class SwiGLUBlock(nn.Module):
    """SwiGLU activation block with Layer Normalization and Residual Skip Connections."""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear_gate = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(out_dim, out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x_norm = self.norm(x)
        u, v = self.linear_gate(x_norm).chunk(2, dim=-1)
        swiglu_out = u * F.silu(v)
        swiglu_out = self.dropout(swiglu_out)
        out = self.proj(swiglu_out)
        return res + out


class MultiSampleDropoutHead(nn.Module):
    """Multi-Sample Dropout Head to accelerate convergence and improve generalization."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        num_samples: int = 5,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0)
        return torch.mean(logits, dim=0)


class TabularSwiGLUNet(nn.Module):
    """Deep Tabular Network incorporating SwiGLU blocks and Multi-Sample Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.25,
        head_dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.LayerNorm(hidden_dims[0])
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                SwiGLUBlock(
                    in_dim=hidden_dims[i],
                    out_dim=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.head = MultiSampleDropoutHead(
            in_dim=hidden_dims[-1],
            out_dim=1,
            num_samples=5,
            dropout_rate=head_dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.backbone(x)
        logits = self.head(x)
        return logits.squeeze(-1)


class FocalBinaryCrossEntropyLoss(nn.Module):
    """Focal Binary Cross Entropy Loss for fine-grained probability calibration."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.01,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = (
            targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_factor * torch.pow(1.0 - p_t, self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
}

cb_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.03,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "iterations": 1000,
    "early_stopping_rounds": 50,
}

# ------------------------------------------------------------------------------
# 3. CROSS-VALIDATION & TRAINING LOOP
# ------------------------------------------------------------------------------

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_hgb = np.zeros(n_samples)
oof_et = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(len(X_test_np))
test_xgb = np.zeros(len(X_test_np))
test_cat = np.zeros(len(X_test_np))
test_hgb = np.zeros(len(X_test_np))
test_et = np.zeros(len(X_test_np))
test_nn = np.zeros(len(X_test_np))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train)):
    qt = QuantileTransformer(n_quantiles=min(len(train_idx), 1000), output_distribution="normal", random_state=42)
    qt_tr = qt.fit_transform(X_train_np[train_idx])
    qt_va = qt.transform(X_train_np[val_idx])
    qt_te = qt.transform(X_test_np)

    X_tr_full = np.hstack([X_train_np[train_idx], qt_tr])
    X_va_full = np.hstack([X_train_np[val_idx], qt_va])
    X_te_full = np.hstack([X_test_np, qt_te])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_full)
    X_va = scaler.transform(X_va_full)
    X_test_fold = scaler.transform(X_te_full)
    y_tr, y_va = y_train[train_idx], y_train[val_idx]

    # 1. LightGBM
    clf_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
    clf_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]
    test_lgb += clf_lgb.predict_proba(X_test_fold)[:, 1] / n_splits

    # 2. XGBoost
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = clf_xgb.predict_proba(X_va)[:, 1]
    test_xgb += clf_xgb.predict_proba(X_test_fold)[:, 1] / n_splits

    # 3. CatBoost
    clf_cat = cb.CatBoostClassifier(**cb_params)
    clf_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]
    test_cat += clf_cat.predict_proba(X_test_fold)[:, 1] / n_splits

    # 4. HistGradientBoosting
    clf_hgb = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=300, min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    clf_hgb.fit(X_tr, y_tr)
    oof_hgb[val_idx] = clf_hgb.predict_proba(X_va)[:, 1]
    test_hgb += clf_hgb.predict_proba(X_test_fold)[:, 1] / n_splits

    # 5. ExtraTrees
    clf_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    clf_et.fit(X_tr, y_tr)
    oof_et[val_idx] = clf_et.predict_proba(X_va)[:, 1]
    test_et += clf_et.predict_proba(X_test_fold)[:, 1] / n_splits

    # 6. PyTorch TabularSwiGLUNet
    nn_model = TabularSwiGLUNet(
        input_dim=X_tr.shape[1], hidden_dims=[256, 128, 64], dropout_rate=0.2
    ).to(device)

    criterion = FocalBinaryCrossEntropyLoss(alpha=0.25, gamma=2.0, label_smoothing=0.01)
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(50):
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
        va_preds_list = []
        with torch.no_grad():
            for bx, _ in va_loader:
                bx = bx.to(device)
                logits = nn_model(bx)
                va_preds_list.extend(torch.sigmoid(logits).cpu().numpy())

        current_auc = roc_auc_score(y_va, va_preds_list)
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    val_preds_nn = []
    with torch.no_grad():
        for bx, _ in va_loader:
            logits = nn_model(bx.to(device))
            val_preds_nn.extend(torch.sigmoid(logits).cpu().numpy())
    oof_nn[val_idx] = np.array(val_preds_nn)

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
        test_preds_nn = torch.sigmoid(nn_model(X_test_tensor)).cpu().numpy()
        test_nn += test_preds_nn / n_splits

# ------------------------------------------------------------------------------
# 4. ENSEMBLE BLENDING AND SUBMISSION
# ------------------------------------------------------------------------------


def rank_normalize(arr):
    return rankdata(arr) / len(arr)


oof_models = [
    rank_normalize(oof_lgb),
    rank_normalize(oof_xgb),
    rank_normalize(oof_cat),
    rank_normalize(oof_hgb),
    rank_normalize(oof_et),
    rank_normalize(oof_nn),
]

test_models = [
    rank_normalize(test_lgb),
    rank_normalize(test_xgb),
    rank_normalize(test_cat),
    rank_normalize(test_hgb),
    rank_normalize(test_et),
    rank_normalize(test_nn),
]

n_models = len(oof_models)


def obj_func(weights):
    weights = np.array(weights)
    sum_w = np.sum(weights)
    if sum_w > 0:
        weights = weights / sum_w
    blend_oof = np.zeros_like(oof_models[0])
    for w, oof_m in zip(weights, oof_models):
        blend_oof += w * oof_m
    return -roc_auc_score(y_train, blend_oof)


init_weights = np.ones(n_models) / n_models
bounds = [(0, 1) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    obj_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-7},
)

best_w = res.x / np.sum(res.x)
best_auc = -res.fun

final_test_preds = np.zeros_like(test_models[0])
for w, test_m in zip(best_w, test_models):
    final_test_preds += w * test_m

submission_df = pd.DataFrame({"id": test_ids, "defects": final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {best_auc}")