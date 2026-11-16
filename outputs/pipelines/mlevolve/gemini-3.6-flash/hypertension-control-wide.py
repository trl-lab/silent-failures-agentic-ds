import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Estimated Percentage of Patients with Controlled Blood Pressure"

# Determine ID column
id_col = None
for col in ["id", "ID", "Id", "index", "row_id"]:
    if col in test_df.columns:
        id_col = col
        break

if id_col is not None:
    test_ids = test_df[id_col].values
    if id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])
    test_df = test_df.drop(columns=[id_col])
else:
    test_ids = np.arange(len(test_df))

# Process Target Variable for Binary Classification
if train_df[target_col].nunique() == 2:
    unique_vals = sorted(train_df[target_col].dropna().unique())
    y_full = (train_df[target_col] == unique_vals[1]).astype(int).values
elif (
    train_df[target_col].dtype == object
    or str(train_df[target_col].dtype) == "category"
):
    y_full = (
        train_df[target_col]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes", "y"])
        .astype(int)
        .values
    )
else:
    raw_vals = train_df[target_col].values
    if np.all((raw_vals == 0) | (raw_vals == 1)):
        y_full = raw_vals.astype(int)
    else:
        med = np.nanmedian(raw_vals)
        y_full = (raw_vals >= med).astype(int)

train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()

# Ensure aligned feature columns
common_cols = [c for c in train_features.columns if c in test_features.columns]
X_train_raw = train_features[common_cols].copy()
X_test_raw = test_features[common_cols].copy()


# Feature Engineering Module
def create_missingness_indicators(df_train, df_test):
    df_tr = df_train.copy()
    df_te = df_test.copy()
    na_cols = [c for c in df_train.columns if df_train[c].isnull().mean() > 0.01]
    for c in na_cols:
        df_tr[f"{c}_isna"] = df_tr[c].isnull().astype(float)
        df_te[f"{c}_isna"] = df_te[c].isnull().astype(float)
    return df_tr, df_te


X_train_raw, X_test_raw = create_missingness_indicators(X_train_raw, X_test_raw)

# Extract calendar features from datetime columns and drop original datetime columns
datetime_cols = X_train_raw.select_dtypes(
    include=["datetime", "datetime64", "datetimetz"]
).columns.tolist()
for c in datetime_cols:
    for df in [X_train_raw, X_test_raw]:
        dt_s = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"] = dt_s.dt.year.astype(float)
        df[f"{c}_month"] = dt_s.dt.month.astype(float)
        df[f"{c}_day"] = dt_s.dt.day.astype(float)
        df[f"{c}_dayofweek"] = dt_s.dt.dayofweek.astype(float)
    X_train_raw = X_train_raw.drop(columns=[c])
    X_test_raw = X_test_raw.drop(columns=[c])

# Log-transform skewed non-negative numeric features (stats computed strictly on train)
num_cols_updated = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
skewed_cols = []
for c in num_cols_updated:
    if X_train_raw[c].min() >= 0:
        skewness = X_train_raw[c].skew()
        if abs(skewness) > 1.5:
            skewed_cols.append(c)

for c in skewed_cols:
    X_train_raw[f"{c}_log1p"] = np.log1p(np.maximum(0, X_train_raw[c].fillna(0)))
    X_test_raw[f"{c}_log1p"] = np.log1p(np.maximum(0, X_test_raw[c].fillna(0)))

# Pairwise feature interaction ratios for numerical variables
top_num_cols = (
    X_train_raw.select_dtypes(include=[np.number])
    .std()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)
for i in range(len(top_num_cols)):
    for j in range(i + 1, len(top_num_cols)):
        c1, c2 = top_num_cols[i], top_num_cols[j]
        if (
            c1.endswith("_log1p")
            or c2.endswith("_log1p")
            or c1.endswith("_isna")
            or c2.endswith("_isna")
        ):
            continue
        X_train_raw[f"ratio_{c1}_div_{c2}"] = (
            X_train_raw[c1] / (X_train_raw[c2].abs() + 1e-5)
        ).replace([np.inf, -np.inf], np.nan)
        X_test_raw[f"ratio_{c1}_div_{c2}"] = (
            X_test_raw[c1] / (X_test_raw[c2].abs() + 1e-5)
        ).replace([np.inf, -np.inf], np.nan)

# Target-independent Frequency Encoding
cat_cols = X_train_raw.select_dtypes(exclude=[np.number]).columns.tolist()
for col in cat_cols:
    freq_map = X_train_raw[col].value_counts(normalize=True, dropna=False).to_dict()
    X_train_raw[f"{col}_freq"] = X_train_raw[col].map(freq_map).fillna(0)
    X_test_raw[f"{col}_freq"] = X_test_raw[col].map(freq_map).fillna(0)

# Ordinal Encoding for Categoricals
for col in cat_cols:
    X_train_raw[col] = X_train_raw[col].astype(str).fillna("Missing")
    X_test_raw[col] = X_test_raw[col].astype(str).fillna("Missing")

if len(cat_cols) > 0:
    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_raw[cat_cols] = ord_encoder.fit_transform(X_train_raw[cat_cols])
    X_test_raw[cat_cols] = ord_encoder.transform(X_test_raw[cat_cols])

X_train_raw = X_train_raw.astype(np.float64)
X_test_raw = X_test_raw.astype(np.float64)

all_cols = X_train_raw.columns.tolist()
X_train_scaled = X_train_raw.copy()
X_test_scaled = X_test_raw.copy()


# PyTorch Tabular Attention ResNet Architecture
class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class TabularGatedResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.se = SqueezeAndExcitationBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.se(out)
        return residual + out


class TabularAttentionResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TabularGatedResBlock(dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def train_pytorch_fold(
    X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, in_features, epochs=50, batch_size=64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_va_np, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_np, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = TabularAttentionResNet(
        in_features=in_features, hidden_dim=128, num_blocks=2, dropout=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_val_preds = None
    best_test_preds = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t.to(device))
            val_loss = criterion(
                val_logits,
                torch.tensor(y_va_np, dtype=torch.float32).unsqueeze(1).to(device),
            ).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_preds = val_probs
                test_logits = model(X_te_t.to(device))
                best_test_preds = torch.sigmoid(test_logits).cpu().numpy().ravel()

    return best_val_preds, best_test_preds


# Cross Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

n_samples = len(X_train_scaled)
n_test = len(X_test_scaled)
n_features = X_train_scaled.shape[1]

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

# Cross Validation Training Loop
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_full)):
    X_tr_raw_f, y_tr = X_train_scaled.iloc[tr_idx].values, y_full[tr_idx]
    X_va_raw_f, y_va = X_train_scaled.iloc[va_idx].values, y_full[va_idx]
    X_te_raw_f = X_test_scaled.values

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr = scaler.fit_transform(imputer.fit_transform(X_tr_raw_f))
    X_va = scaler.transform(imputer.transform(X_va_raw_f))
    X_te = scaler.transform(imputer.transform(X_te_raw_f))

    # 1. LightGBM
    model_lgb = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[va_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_te)[:, 1] / n_splits

    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(
        eval_metric="auc",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_te)[:, 1] / n_splits

    # 3. CatBoost
    model_cat = cb.CatBoostClassifier(
        eval_metric="AUC",
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[va_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_te)[:, 1] / n_splits

    # 4. PyTorch TabularAttentionResNet
    val_preds_nn, test_preds_nn_fold = train_pytorch_fold(
        X_tr, y_tr, X_va, y_va, X_te, in_features=n_features, epochs=50
    )
    oof_nn[va_idx] = val_preds_nn
    test_preds_nn += test_preds_nn_fold / n_splits

# Ensemble Weight Optimization using Constrained SLSQP
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
)


def loss_func(weights):
    weights = weights / np.sum(weights)
    blend = oof_matrix @ weights
    return -roc_auc_score(y_full, blend)


init_weights = np.ones(4) / 4.0
bounds = [(0, 1) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}

opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

oof_final = oof_matrix @ best_weights
test_preds_final = test_matrix @ best_weights

final_score = roc_auc_score(y_full, oof_final)

# Generate Final Submission File
sub = pd.DataFrame(
    {
        "id": test_ids,
        "Estimated Percentage of Patients with Controlled Blood Pressure": (
            test_preds_final
        ),
    }
)
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")