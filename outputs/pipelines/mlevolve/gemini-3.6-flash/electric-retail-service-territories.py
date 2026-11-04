import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set seeds for deterministic reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# STEP 1: Data Processing & Feature Engineering
# ==========================================

train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "TOTAL_MWH"
id_col = "id" if "id" in test_raw.columns else None

train_y = train_raw[target_col].values.astype(np.float32)

train_ids = (
    train_raw[id_col].values
    if id_col and id_col in train_raw.columns
    else np.arange(len(train_raw), dtype=int)
)
test_ids = (
    test_raw[id_col].values
    if id_col and id_col in test_raw.columns
    else np.arange(len(test_raw), dtype=int)
)

train_features = train_raw.drop(
    columns=[c for c in [target_col, "id"] if c in train_raw.columns]
)
test_features = test_raw.drop(
    columns=[c for c in [target_col, "id"] if c in test_raw.columns]
)

combined_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)
n_train = len(train_raw)

num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()

# Impute missing values
for col in num_cols:
    median_val = train_features[col].median()
    combined_df[col] = combined_df[col].fillna(
        median_val if not np.isnan(median_val) else 0.0
    )

for col in cat_cols:
    combined_df[col] = combined_df[col].fillna("Unknown").astype(str)

# Domain-specific customer sector composition
customer_cols = [
    c
    for c in num_cols
    if "CUST" in c.upper()
    or "RES" in c.upper()
    or "COM" in c.upper()
    or "IND" in c.upper()
]

if customer_cols:
    combined_df["SECTOR_CUST_SUM"] = combined_df[customer_cols].sum(axis=1)
    for col in customer_cols:
        combined_df[f"{col}_SHARE"] = combined_df[col] / (
            combined_df["SECTOR_CUST_SUM"] + 1e-5
        )

# Log1p non-linear transforms
for col in num_cols:
    if combined_df[col].min() >= 0:
        combined_df[f"{col}_log1p"] = np.log1p(combined_df[col])

# Feature interaction ratios and products
if len(num_cols) >= 2:
    for i in range(min(4, len(num_cols))):
        for j in range(i + 1, min(4, len(num_cols))):
            c1, c2 = num_cols[i], num_cols[j]
            combined_df[f"{c1}_x_{c2}"] = combined_df[c1] * combined_df[c2]
            combined_df[f"{c1}_ratio_{c2}"] = (combined_df[c1] + 1) / (
                combined_df[c2] + 1
            )

# Categorical Frequency Encoding
for col in cat_cols:
    freq_map = train_features[col].value_counts().to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq_map).fillna(1)

# Group-level statistics strictly computed on training set
for grp_col in [c for c in cat_cols if train_features[c].nunique() < 100]:
    for target_num in num_cols[:5]:
        grp_stats = (
            train_features.groupby(grp_col)[target_num]
            .agg(["mean", "std"])
            .reset_index()
        )
        grp_stats.columns = [
            grp_col,
            f"{grp_col}_{target_num}_grp_mean",
            f"{grp_col}_{target_num}_grp_std",
        ]
        combined_df = combined_df.merge(grp_stats, on=grp_col, how="left")
        combined_df[f"{grp_col}_{target_num}_grp_mean"] = combined_df[
            f"{grp_col}_{target_num}_grp_mean"
        ].fillna(0.0)
        combined_df[f"{grp_col}_{target_num}_grp_std"] = combined_df[
            f"{grp_col}_{target_num}_grp_std"
        ].fillna(0.0)

# One-hot encoding for categorical variables
combined_df = pd.get_dummies(combined_df, columns=cat_cols, drop_first=True)
bool_cols = combined_df.select_dtypes(include=["bool"]).columns
combined_df[bool_cols] = combined_df[bool_cols].astype(np.int32)

train_processed = combined_df.iloc[:n_train].copy()
test_processed = combined_df.iloc[n_train:].copy()

train_processed[target_col] = train_y

# Stratified 5-Fold split based on binned log-target values
log_y = np.log1p(np.maximum(0, train_y))
y_bins = pd.qcut(log_y, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_processed["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_processed, y_bins)):
    train_processed.loc[val_idx, "fold"] = fold

# ==========================================
# STEP 2: Model Architecture & Design
# ==========================================


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularDualHeadResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.log_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.raw_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor):
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)

        pred_log = self.log_head(h).squeeze(-1)
        pred_raw = self.raw_head(h).squeeze(-1)

        pred_log_exp = torch.expm1(torch.clamp(pred_log, min=0.0, max=20.0))

        g = torch.sigmoid(self.gate)
        fused_pred = g * pred_log_exp + (1.0 - g) * torch.relu(pred_raw)

        return fused_pred, pred_log, pred_raw


class DualScaleLoss(nn.Module):
    def __init__(self, log_weight: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.log_weight = log_weight
        self.eps = eps

    def forward(
        self,
        fused_pred: torch.Tensor,
        pred_log: torch.Tensor,
        pred_raw: torch.Tensor,
        target_raw: torch.Tensor,
    ) -> torch.Tensor:
        target_log = torch.log1p(torch.clamp(target_raw, min=0.0))

        raw_rmse = torch.sqrt(torch.mean((fused_pred - target_raw) ** 2) + self.eps)
        log_mse = torch.mean((pred_log - target_log) ** 2)
        raw_head_rmse = torch.sqrt(
            torch.mean((torch.relu(pred_raw) - target_raw) ** 2) + self.eps
        )

        return raw_rmse + self.log_weight * log_mse + 0.1 * raw_head_rmse


def create_neural_model_and_optimizer(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4
):
    model = TabularDualHeadResNet(input_dim=input_dim)
    criterion = DualScaleLoss(log_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    return model, criterion, optimizer, scheduler


def get_gbdt_configs():
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
    }

    cat_params = {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
    }

    return lgb_params, xgb_params, cat_params


# ==========================================
# STEP 3: Training & Evaluation Pipeline
# ==========================================

feature_cols = [
    c for c in train_processed.columns if c not in [target_col, "id", "fold"]
]

X_test_raw = test_processed[feature_cols].values.astype(np.float32)
X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

n_train = len(train_processed)
n_test = len(test_processed)
n_folds = train_processed["fold"].nunique()

num_models = 5
oof_preds = np.zeros((n_train, num_models), dtype=np.float32)
test_preds_folds = np.zeros((n_test, num_models), dtype=np.float32)

y_train_all = train_processed[target_col].values.astype(np.float32)

lgb_params, xgb_params, cat_params = get_gbdt_configs()

for fold in range(n_folds):
    tr_idx = train_processed["fold"] != fold
    va_idx = train_processed["fold"] == fold

    X_tr = train_processed.loc[tr_idx, feature_cols].values.astype(np.float32)
    y_tr = train_processed.loc[tr_idx, target_col].values.astype(np.float32)

    X_va = train_processed.loc[va_idx, feature_cols].values.astype(np.float32)
    y_va = train_processed.loc[va_idx, target_col].values.astype(np.float32)

    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)

    # Model 1: LightGBM (Raw)
    lgb_raw = lgb.LGBMRegressor(**lgb_params)
    lgb_raw.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    p_va_lgb_raw = np.maximum(0, lgb_raw.predict(X_va))
    p_te_lgb_raw = np.maximum(0, lgb_raw.predict(X_test_raw))

    # Model 2: XGBoost (Raw)
    xgb_params_fold = xgb_params.copy()
    xgb_params_fold["early_stopping_rounds"] = 50
    xgb_raw = xgb.XGBRegressor(**xgb_params_fold)
    xgb_raw.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_va_xgb_raw = np.maximum(0, xgb_raw.predict(X_va))
    p_te_xgb_raw = np.maximum(0, xgb_raw.predict(X_test_raw))

    # Model 3: CatBoost (Raw)
    cat_raw = CatBoostRegressor(**cat_params)
    cat_raw.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    p_va_cat_raw = np.maximum(0, cat_raw.predict(X_va))
    p_te_cat_raw = np.maximum(0, cat_raw.predict(X_test_raw))

    # Model 4: LightGBM (Log Target)
    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_va_log = np.log1p(np.maximum(0, y_va))
    lgb_log = lgb.LGBMRegressor(**lgb_params)
    lgb_log.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    p_va_lgb_log = np.expm1(np.maximum(0, lgb_log.predict(X_va)))
    p_te_lgb_log = np.expm1(np.maximum(0, lgb_log.predict(X_test_raw)))

    # Model 5: PyTorch TabularDualHeadResNet
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test_raw)

    nn_model, criterion, optimizer, scheduler = create_neural_model_and_optimizer(
        input_dim=len(feature_cols)
    )
    nn_model = nn_model.to(device)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    best_nn_weights = None
    patience, patience_counter = 15, 0

    for epoch in range(80):
        nn_model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            fused_p, p_log, p_raw = nn_model(x_b)
            loss = criterion(fused_p, p_log, p_raw, y_b)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_b, _ in val_loader:
                x_b = x_b.to(device)
                fused_p, _, _ = nn_model(x_b)
                val_preds_list.append(fused_p.cpu().numpy())

        val_preds_epoch = np.concatenate(val_preds_list)
        val_loss = np.sqrt(mean_squared_error(y_va, np.maximum(0, val_preds_epoch)))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = nn_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_nn_weights is not None:
        nn_model.load_state_dict(best_nn_weights)

    nn_model.eval()
    with torch.no_grad():
        p_va_nn, _, _ = nn_model(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
        p_te_nn, _, _ = nn_model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
        p_va_nn = np.maximum(0, p_va_nn.cpu().numpy())
        p_te_nn = np.maximum(0, p_te_nn.cpu().numpy())

    oof_preds[va_idx, 0] = p_va_lgb_raw
    oof_preds[va_idx, 1] = p_va_xgb_raw
    oof_preds[va_idx, 2] = p_va_cat_raw
    oof_preds[va_idx, 3] = p_va_lgb_log
    oof_preds[va_idx, 4] = p_va_nn

    test_preds_folds[:, 0] += p_te_lgb_raw / n_folds
    test_preds_folds[:, 1] += p_te_xgb_raw / n_folds
    test_preds_folds[:, 2] += p_te_cat_raw / n_folds
    test_preds_folds[:, 3] += p_te_lgb_log / n_folds
    test_preds_folds[:, 4] += p_te_nn / n_folds


# SLSQP Non-Negative Meta-Learner Optimization
def loss_func(weights):
    blend_oof = np.dot(oof_preds, weights)
    return np.sqrt(mean_squared_error(y_train_all, np.maximum(0, blend_oof)))


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = res.x

final_oof_preds = np.maximum(0, np.dot(oof_preds, opt_weights))
final_val_score = np.sqrt(mean_squared_error(y_train_all, final_oof_preds))

# Save submission file
final_test_preds = np.maximum(0, np.dot(test_preds_folds, opt_weights))

submission_df = pd.DataFrame({"id": test_ids, target_col: final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

assert os.path.exists("./submission/submission.csv"), "Submission file missing!"
assert len(submission_df) == len(test_raw), "Mismatch in test row count!"
assert not submission_df[target_col].isnull().any(), "Submission contains nulls!"

print(f"Final Validation Score: {final_val_score}")
