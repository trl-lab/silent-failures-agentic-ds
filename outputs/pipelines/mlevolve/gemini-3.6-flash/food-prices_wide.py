import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# -----------------------------------------------------------------------------
# 1. Setup & Environment Initialization
# -----------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load raw datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "usdprice"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' missing from training data."

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))


# -----------------------------------------------------------------------------
# 2. Feature Engineering & Preprocessing
# -----------------------------------------------------------------------------
def process_dates(df):
    df = df.copy()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        try:
            dt = pd.to_datetime(df[col])
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_quarter"] = dt.dt.quarter
            df[f"{col}_dayofweek"] = dt.dt.dayofweek
            df[f"{col}_dayofyear"] = dt.dt.dayofyear
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
            df.drop(columns=[col], inplace=True)
        except Exception:
            pass
    return df


train_proc = process_dates(train_df)
test_proc = process_dates(test_df)

# Identify Categorical Columns
cat_cols = [
    c
    for c in train_proc.columns
    if c != target_col
    and (train_proc[c].dtype == "object" or train_proc[c].dtype.name == "category")
]

for c in cat_cols:
    train_proc[c] = train_proc[c].astype(str).fillna("missing")
    test_proc[c] = test_proc[c].astype(str).fillna("missing")

# Create High-Order Composite Interaction Keys
candidate_pairs = [
    ("country", "commodity"),
    ("market", "commodity"),
    ("adm1", "commodity"),
    ("category", "unit"),
]
for col1, col2 in candidate_pairs:
    matching_c1 = [c for c in cat_cols if col1 in c.lower()]
    matching_c2 = [c for c in cat_cols if col2 in c.lower()]
    if matching_c1 and matching_c2:
        c1, c2 = matching_c1[0], matching_c2[0]
        pair_name = f"{c1}_X_{c2}"
        train_proc[pair_name] = train_proc[c1] + "_" + train_proc[c2]
        test_proc[pair_name] = test_proc[c1] + "_" + test_proc[c2]
        if pair_name not in cat_cols:
            cat_cols.append(pair_name)

# Frequency Encoding
for c in cat_cols:
    freq = train_proc[c].value_counts().to_dict()
    train_proc[f"{c}_freq"] = train_proc[c].map(freq).fillna(0).astype(np.float32)
    test_proc[f"{c}_freq"] = test_proc[c].map(freq).fillna(0).astype(np.float32)

# Set up Cross-Validation Folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_proc["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_proc)):
    train_proc.loc[val_idx, "fold"] = fold

# Out-of-Fold Target Aggregations
group_candidates = [
    c
    for c in [
        "commodity",
        "market",
        "country_X_commodity",
        cat_cols[0] if cat_cols else None,
    ]
    if c and c in train_proc.columns
]

for g_col in group_candidates[:3]:
    target_mean_col = f"{g_col}_target_mean"
    target_std_col = f"{g_col}_target_std"
    train_proc[target_mean_col] = np.nan
    train_proc[target_std_col] = np.nan

    for fold in range(5):
        tr_mask = train_proc["fold"] != fold
        va_mask = train_proc["fold"] == fold

        grp_stats = (
            train_proc[tr_mask]
            .groupby(g_col)[target_col]
            .agg(["mean", "std"])
            .reset_index()
        )
        grp_stats.columns = [g_col, "grp_mean", "grp_std"]

        mean_map = grp_stats.set_index(g_col)["grp_mean"].to_dict()
        std_map = grp_stats.set_index(g_col)["grp_std"].to_dict()
        train_proc.loc[va_mask, target_mean_col] = train_proc.loc[va_mask, g_col].map(mean_map).values
        train_proc.loc[va_mask, target_std_col] = train_proc.loc[va_mask, g_col].map(std_map).fillna(0).values

    full_grp = train_proc.groupby(g_col)[target_col].agg(["mean", "std"]).reset_index()
    full_grp.columns = [g_col, "grp_mean", "grp_std"]
    global_mean = train_proc[target_col].mean()

    test_merged = test_proc[[g_col]].merge(full_grp, on=g_col, how="left")
    test_proc[target_mean_col] = test_merged["grp_mean"].fillna(global_mean).values
    test_proc[target_std_col] = test_merged["grp_std"].fillna(0).values

    train_proc[target_mean_col] = train_proc[target_mean_col].fillna(global_mean)
    train_proc[target_std_col] = train_proc[target_std_col].fillna(0)

# Label Encoding
for c in cat_cols:
    mapping = {val: idx for idx, val in enumerate(train_proc[c].unique())}
    train_proc[c] = train_proc[c].map(mapping).fillna(-1).astype(np.int32)
    test_proc[c] = test_proc[c].map(mapping).fillna(-1).astype(np.int32)

# Feature Selection & Imputation
features = [c for c in train_proc.columns if c not in [target_col, "fold", "id"]]

for f in features:
    if train_proc[f].isnull().sum() > 0 or test_proc[f].isnull().sum() > 0:
        med = train_proc[f].median()
        train_proc[f] = train_proc[f].fillna(med)
        test_proc[f] = test_proc[f].fillna(med)

X_train = train_proc[features]
y_train = train_proc[target_col].values
folds = train_proc["fold"].values
X_test = test_proc[features]

n_samples = len(y_train)
n_test = len(X_test)
n_features = len(features)
num_folds = len(np.unique(folds))


# -----------------------------------------------------------------------------
# 3. Model Architecture Definitions
# -----------------------------------------------------------------------------
class TabularResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [TabularResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# Arrays for Out-of-Fold and Test Predictions
oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_nn = np.zeros(n_test)


# -----------------------------------------------------------------------------
# 4. Model Training & Evaluation Loop
# -----------------------------------------------------------------------------
# LightGBM
print("--- Training LightGBM ---")
for fold in range(num_folds):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    model_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_lgb[va_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / num_folds

print(f"LightGBM OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof_lgb)):.5f}")

# XGBoost
print("--- Training XGBoost ---")
for fold in range(num_folds):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    model_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_xgb[va_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / num_folds

print(f"XGBoost OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof_xgb)):.5f}")

# CatBoost
print("--- Training CatBoost ---")
for fold in range(num_folds):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    model_cat = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

    oof_cat[va_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test) / num_folds

print(f"CatBoost OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof_cat)):.5f}")

# PyTorch TabularResNet
print("--- Training PyTorch TabularResNet ---")
for fold in range(num_folds):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train.iloc[tr_idx].values)
    X_va_sc = scaler.transform(X_train.iloc[va_idx].values)
    X_te_sc = scaler.transform(X_test.values)

    y_tr = y_train[tr_idx]
    y_va = y_train[va_idx]

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=256, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=512, shuffle=False)

    model_nn = TabularResNet(
        input_dim=n_features, hidden_dim=128, num_blocks=2, dropout_rate=0.1
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(30):
        model_nn.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model_nn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model_nn.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model_nn(bx)
                val_losses.append(criterion(out, by).item())
        val_rmse_ep = np.sqrt(np.mean(val_losses))

        if val_rmse_ep < best_val_loss:
            best_val_loss = val_rmse_ep
            best_weights = {
                k: v.cpu().clone() for k, v in model_nn.state_dict().items()
            }

    model_nn.load_state_dict(best_weights)
    model_nn.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        oof_nn[va_idx] = model_nn(va_tensor).cpu().numpy()

        te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        test_nn += model_nn(te_tensor).cpu().numpy() / num_folds

print(f"TabularResNet OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof_nn)):.5f}")

# -----------------------------------------------------------------------------
# 5. SLSQP Weight Optimization & Ensembling
# -----------------------------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def ensemble_loss(weights):
    weights = weights / np.sum(weights)
    pred = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y_train, pred))


init_weights = np.ones(4) / 4
bounds = [(0, 1) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}

res = minimize(
    ensemble_loss, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
opt_weights = res.x / np.sum(res.x)

final_oof_pred = oof_matrix @ opt_weights
final_rmse = np.sqrt(mean_squared_error(y_train, final_oof_pred))
print(
    f"Optimized Ensemble Weights: LGB={opt_weights[0]:.3f}, XGB={opt_weights[1]:.3f}, CAT={opt_weights[2]:.3f}, NN={opt_weights[3]:.3f}"
)

# -----------------------------------------------------------------------------
# 6. Final Predictions & Submission Generation
# -----------------------------------------------------------------------------
final_test_pred = test_matrix @ opt_weights
final_test_pred = np.maximum(0.0, final_test_pred)

sub = pd.DataFrame({"id": test_ids, "usdprice": final_test_pred})
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse:.5f}")