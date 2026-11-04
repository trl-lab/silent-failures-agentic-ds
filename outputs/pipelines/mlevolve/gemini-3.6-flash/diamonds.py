import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# -----------------------------------------------------------------------------
# 1. Environment & Setup
# -----------------------------------------------------------------------------
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 2. Data Processing & Feature Engineering
# -----------------------------------------------------------------------------
train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))

target_col = "price"
test_ids = test_df["id"].values

# Clean invalid 0 values in physical dimensions using train set medians
dim_cols = ["x", "y", "z"]
for col in dim_cols:
    train_df[col] = train_df[col].replace(0, np.nan)
    test_df[col] = test_df[col].replace(0, np.nan)

    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

# Ordinal mappings
cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_map = {
    "I1": 1,
    "SI2": 2,
    "SI1": 3,
    "VS2": 4,
    "VS1": 5,
    "VVS2": 6,
    "VVS1": 7,
    "IF": 8,
}


def engineer_features(df):
    df = df.copy()

    df["cut_num"] = df["cut"].map(cut_map).astype(int)
    df["color_num"] = df["color"].map(color_map).astype(int)
    df["clarity_num"] = df["clarity"].map(clarity_map).astype(int)

    df["total_quality"] = df["cut_num"] + df["color_num"] + df["clarity_num"]
    df["quality_product"] = df["cut_num"] * df["color_num"] * df["clarity_num"]

    df["volume"] = df["x"] * df["y"] * df["z"]
    df["surface_area"] = 2 * (df["x"] * df["y"] + df["x"] * df["z"] + df["y"] * df["z"])
    df["carat_density"] = df["carat"] / (df["volume"] + 1e-6)

    df["depth_calc"] = 200 * df["z"] / (df["x"] + df["y"] + 1e-6)
    df["depth_diff"] = np.abs(df["depth"] - df["depth_calc"])

    df["aspect_xy"] = df["x"] / (df["y"] + 1e-6)
    df["aspect_xz"] = df["x"] / (df["z"] + 1e-6)
    df["aspect_yz"] = df["y"] / (df["z"] + 1e-6)
    df["table_mm"] = df["table"] * df["x"] / 100.0

    df["carat_log"] = np.log1p(df["carat"])
    df["carat_sq"] = df["carat"] ** 2
    df["carat_cube"] = df["carat"] ** 3
    df["volume_log"] = np.log1p(df["volume"])

    # Carat milestone proximity features
    thresholds = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    for t in thresholds:
        df[f"carat_dist_{t}"] = df["carat"] - t
        df[f"carat_abs_dist_{t}"] = np.abs(df["carat"] - t)
        df[f"carat_gte_{t}"] = (df["carat"] >= t).astype(int)

    df["carat_cut"] = df["carat"] * df["cut_num"]
    df["carat_color"] = df["carat"] * df["color_num"]
    df["carat_clarity"] = df["carat"] * df["clarity_num"]
    df["carat_quality"] = df["carat"] * df["total_quality"]
    df["volume_quality"] = df["volume"] * df["total_quality"]

    return df


train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

# Train-only group aggregations to prevent data leakage
group_cols = ["cut", "color", "clarity"]
agg_cols = ["carat", "volume"]

for col in agg_cols:
    group_stats = (
        train_fe.groupby(group_cols)[col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    group_stats.columns = group_cols + [f"{col}_grp_mean", f"{col}_grp_std"]

    train_fe = train_fe.merge(group_stats, on=group_cols, how="left")
    test_fe = test_fe.merge(group_stats, on=group_cols, how="left")

    train_fe[f"{col}_grp_mean"] = train_fe[f"{col}_grp_mean"].fillna(train_fe[col].mean())
    test_fe[f"{col}_grp_mean"] = test_fe[f"{col}_grp_mean"].fillna(train_fe[col].mean())
    train_fe[f"{col}_grp_std"] = train_fe[f"{col}_grp_std"].fillna(0)
    test_fe[f"{col}_grp_std"] = test_fe[f"{col}_grp_std"].fillna(0)

# Create 5-Fold Stratified-like CV splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_fe["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_fe)):
    train_fe.loc[val_idx, "fold"] = fold

# Out-of-fold target encoding for cut, color, clarity combinations (mean price per carat)
train_fe["price_per_carat"] = train_fe["price"] / train_fe["carat"]
train_fe["target_enc_ppc"] = np.nan

global_enc = train_fe.groupby(group_cols)["price_per_carat"].mean().to_dict()
global_mean = train_fe["price_per_carat"].mean()

for fold in range(5):
    train_fold = train_fe[train_fe["fold"] != fold]
    val_mask = train_fe["fold"] == fold

    fold_enc = train_fold.groupby(group_cols)["price_per_carat"].mean().to_dict()
    keys = list(zip(train_fe.loc[val_mask, "cut"], train_fe.loc[val_mask, "color"], train_fe.loc[val_mask, "clarity"]))
    train_fe.loc[val_mask, "target_enc_ppc"] = [fold_enc.get(k, global_mean) for k in keys]

test_keys = list(zip(test_fe["cut"], test_fe["color"], test_fe["clarity"]))
test_fe["target_enc_ppc"] = [global_enc.get(k, global_mean) for k in test_keys]


# -----------------------------------------------------------------------------
# 3. Model Architecture Definitions
# -----------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def create_pytorch_components(
    input_dim: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lgbm_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(random_state: int = 42) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=0,
        thread_count=-1,
    )


# -----------------------------------------------------------------------------
# 4. Model Training & Evaluation Loop
# -----------------------------------------------------------------------------
exclude_cols = ["price", "price_per_carat", "fold", "cut", "color", "clarity", "id"]
feature_cols = [c for c in train_fe.columns if c not in exclude_cols]

for col in feature_cols:
    train_fe[col] = train_fe[col].replace([np.inf, -np.inf], np.nan)
    test_fe[col] = test_fe[col].replace([np.inf, -np.inf], np.nan)
    med_val = train_fe[col].median()
    train_fe[col] = train_fe[col].fillna(med_val)
    test_fe[col] = test_fe[col].fillna(med_val)

X_test_raw = test_fe[feature_cols].values

num_folds = 5
num_models = 7

oof_predictions = np.zeros((len(train_fe), num_models))
test_predictions = np.zeros((len(test_fe), num_models))

for fold in range(num_folds):
    train_mask = train_fe["fold"] != fold
    val_mask = train_fe["fold"] == fold
    val_indices = train_fe.index[val_mask].values

    X_train = train_fe.loc[train_mask, feature_cols].values
    y_train_raw = train_fe.loc[train_mask, target_col].values
    y_train_log = np.log1p(y_train_raw)

    X_val = train_fe.loc[val_mask, feature_cols].values
    y_val_raw = train_fe.loc[val_mask, target_col].values

    # Model 0: LightGBM (Log target)
    lgb_model = get_lgbm_model(random_state=42 + fold)
    lgb_model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, np.log1p(y_val_raw))],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred_lgb = np.expm1(lgb_model.predict(X_val))
    test_pred_lgb = np.expm1(lgb_model.predict(X_test_raw))

    # Model 1: XGBoost (Log target)
    xgb_model = get_xgb_model(random_state=42 + fold)
    xgb_model.set_params(early_stopping_rounds=50)
    xgb_model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, np.log1p(y_val_raw))],
        verbose=False,
    )
    val_pred_xgb = np.expm1(xgb_model.predict(X_val))
    test_pred_xgb = np.expm1(xgb_model.predict(X_test_raw))

    # Model 2: CatBoost (Log target)
    cb_model = get_catboost_model(random_state=42 + fold)
    cb_model.fit(
        X_train,
        y_train_log,
        eval_set=(X_val, np.log1p(y_val_raw)),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_pred_cb = np.expm1(cb_model.predict(X_val))
    test_pred_cb = np.expm1(cb_model.predict(X_test_raw))

    # Model 3: LightGBM (Raw target - Direct RMSE)
    lgb_raw = get_lgbm_model(random_state=100 + fold)
    lgb_raw.fit(
        X_train,
        y_train_raw,
        eval_set=[(X_val, y_val_raw)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred_lgb_raw = lgb_raw.predict(X_val)
    test_pred_lgb_raw = lgb_raw.predict(X_test_raw)

    # Model 4: XGBoost (Raw target - Direct RMSE)
    xgb_raw = get_xgb_model(random_state=100 + fold)
    xgb_raw.set_params(early_stopping_rounds=50)
    xgb_raw.fit(
        X_train,
        y_train_raw,
        eval_set=[(X_val, y_val_raw)],
        verbose=False,
    )
    val_pred_xgb_raw = xgb_raw.predict(X_val)
    test_pred_xgb_raw = xgb_raw.predict(X_test_raw)

    # Model 5: CatBoost (Raw target - Direct RMSE)
    cb_raw = get_catboost_model(random_state=100 + fold)
    cb_raw.fit(
        X_train,
        y_train_raw,
        eval_set=(X_val, y_val_raw),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_pred_cb_raw = cb_raw.predict(X_val)
    test_pred_cb_raw = cb_raw.predict(X_test_raw)

    # Model 6: PyTorch Tabular ResNet
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_raw)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(np.log1p(y_val_raw), dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)

    epochs = 40
    nn_model, criterion, optimizer, scheduler = create_pytorch_components(
        input_dim=X_train.shape[1], epochs=epochs
    )
    nn_model.to(device)

    best_val_loss = float("inf")
    best_weights = {
        k: v.cpu().clone() for k, v in nn_model.state_dict().items()
    }

    for epoch in range(epochs):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(bx)
        val_loss /= len(X_val_scaled)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    nn_model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        val_pred_nn = np.expm1(nn_model(val_tensor).cpu().numpy())
        test_pred_nn = np.expm1(nn_model(test_tensor).cpu().numpy())

    oof_predictions[val_indices, 0] = val_pred_lgb
    oof_predictions[val_indices, 1] = val_pred_xgb
    oof_predictions[val_indices, 2] = val_pred_cb
    oof_predictions[val_indices, 3] = val_pred_lgb_raw
    oof_predictions[val_indices, 4] = val_pred_xgb_raw
    oof_predictions[val_indices, 5] = val_pred_cb_raw
    oof_predictions[val_indices, 6] = val_pred_nn

    test_predictions[:, 0] += test_pred_lgb / num_folds
    test_predictions[:, 1] += test_pred_xgb / num_folds
    test_predictions[:, 2] += test_pred_cb / num_folds
    test_predictions[:, 3] += test_pred_lgb_raw / num_folds
    test_predictions[:, 4] += test_pred_xgb_raw / num_folds
    test_predictions[:, 5] += test_pred_cb_raw / num_folds
    test_predictions[:, 6] += test_pred_nn / num_folds

# -----------------------------------------------------------------------------
# 5. Optimization & Output Submission
# -----------------------------------------------------------------------------
y_true_all = train_fe[target_col].values


def ensemble_loss(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = np.dot(oof_predictions, weights)
    return np.sqrt(np.mean((y_true_all - blend) ** 2))


initial_weights = [1.0 / num_models] * num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = res.x / np.sum(res.x)

final_oof = np.dot(oof_predictions, opt_weights)
final_score = np.sqrt(np.mean((y_true_all - final_oof) ** 2))

final_test_preds = np.dot(test_predictions, opt_weights)
final_test_preds = np.clip(final_test_preds, a_min=300.0, a_max=None)

sub_df = pd.DataFrame({"id": test_ids, "price": final_test_preds})
sub_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

print(f"Final Validation Score: {final_score:.5f}")