import os
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Setup Directories & Load Data
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Target Processing
target_col = "class"


def parse_target(series):
    s = series.astype(str).str.lower().str.strip().str.rstrip(".")
    return ((s == "ad") | (s == "1") | (s == "true")).astype(int)


y_full = parse_target(train_df[target_col])

if "id" in test_df.columns:
    test_ids = test_df["id"].values
    test_features_df = test_df.drop(columns=["id"])
else:
    test_ids = np.arange(len(test_df))
    test_features_df = test_df.copy()

train_features_df = train_df.drop(
    columns=[target_col] + (["id"] if "id" in train_df.columns else [])
)

feature_names = list(train_features_df.columns)
continuous_cols = feature_names[:3]
binary_cols = feature_names[3:]


def sanitize_continuous(df, cols):
    data = {}
    for col in cols:
        s = df[col].astype(str).str.strip()
        s = s.replace({"?": np.nan, "None": np.nan, "nan": np.nan, "": np.nan})
        data[col] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(data, index=df.index)


def sanitize_binary(df, cols):
    data = {}
    for col in cols:
        s = df[col].astype(str).str.strip()
        s = s.replace({"?": "0", "None": "0", "nan": "0", "": "0"})
        data[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float32)
    return pd.DataFrame(data, index=df.index)


print("Sanitizing feature sets...")
X_train_cont_raw = sanitize_continuous(train_features_df, continuous_cols)
X_test_cont_raw = sanitize_continuous(test_features_df, continuous_cols)

X_train_bin_raw = sanitize_binary(train_features_df, binary_cols)
X_test_bin_raw = sanitize_binary(test_features_df, binary_cols)

# Train/Val Split (Strict Separation before Fitting)
print("Creating stratified train/validation split...")
indices = np.arange(len(train_features_df))
tr_idx, val_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y_full.values
)

y_tr = y_full.iloc[tr_idx].reset_index(drop=True)
y_val = y_full.iloc[val_idx].reset_index(drop=True)

X_tr_cont = X_train_cont_raw.iloc[tr_idx].reset_index(drop=True)
X_val_cont = X_train_cont_raw.iloc[val_idx].reset_index(drop=True)
X_te_cont = X_test_cont_raw.reset_index(drop=True)

X_tr_bin = X_train_bin_raw.iloc[tr_idx].reset_index(drop=True)
X_val_bin = X_train_bin_raw.iloc[val_idx].reset_index(drop=True)
X_te_bin = X_test_bin_raw.reset_index(drop=True)

# Imputation & Feature Engineering
print("Engineering continuous geometric features...")
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

for col in continuous_cols:
    X_tr_cont[f"{col}_is_missing"] = X_tr_cont[col].isna().astype(np.float32)
    X_val_cont[f"{col}_is_missing"] = X_val_cont[col].isna().astype(np.float32)
    X_te_cont[f"{col}_is_missing"] = X_te_cont[col].isna().astype(np.float32)

tr_cont_imputed = imputer.fit_transform(X_tr_cont[continuous_cols])
val_cont_imputed = imputer.transform(X_val_cont[continuous_cols])
te_cont_imputed = imputer.transform(X_te_cont[continuous_cols])


def create_geometric_features(imputed_array, missing_df):
    height = imputed_array[:, 0]
    width = imputed_array[:, 1]
    aratio = imputed_array[:, 2]

    area = height * width
    calc_aratio = width / (height + 1e-5)
    aratio_diff = np.abs(aratio - calc_aratio)

    log_height = np.log1p(np.maximum(0, height))
    log_width = np.log1p(np.maximum(0, width))
    log_area = np.log1p(np.maximum(0, area))

    standard_banners = [
        (468, 60),
        (120, 600),
        (728, 90),
        (300, 250),
        (160, 600),
        (88, 31),
        (120, 90),
        (120, 240),
        (234, 60),
    ]
    is_standard = np.zeros(len(height), dtype=np.float32)
    for h_std, w_std in standard_banners:
        match = (np.abs(height - h_std) <= 2) & (np.abs(width - w_std) <= 2)
        is_standard = np.maximum(is_standard, match.astype(np.float32))

    feat_df = pd.DataFrame(
        {
            "height": height,
            "width": width,
            "aratio": aratio,
            "area": area,
            "calc_aratio": calc_aratio,
            "aratio_diff": aratio_diff,
            "log_height": log_height,
            "log_width": log_width,
            "log_area": log_area,
            "is_standard_banner": is_standard,
        },
        index=missing_df.index,
    )

    missing_cols = [c for c in missing_df.columns if c.endswith("_is_missing")]
    for mc in missing_cols:
        feat_df[mc] = missing_df[mc].values
    return feat_df


X_tr_geo = create_geometric_features(tr_cont_imputed, X_tr_cont)
X_val_geo = create_geometric_features(val_cont_imputed, X_val_cont)
X_te_geo = create_geometric_features(te_cont_imputed, X_te_cont)

scale_cols = [
    "height",
    "width",
    "aratio",
    "area",
    "calc_aratio",
    "aratio_diff",
    "log_height",
    "log_width",
    "log_area",
]
X_tr_geo[scale_cols] = scaler.fit_transform(X_tr_geo[scale_cols])
X_val_geo[scale_cols] = scaler.transform(X_val_geo[scale_cols])
X_te_geo[scale_cols] = scaler.transform(X_te_geo[scale_cols])

# Binary Aggregations & SVD
print("Engineering binary text features and SVD representations...")
tr_bin_dict = {}
val_bin_dict = {}
te_bin_dict = {}

tr_bin_dict["total_active_bin_features"] = X_tr_bin.sum(axis=1).astype(np.float32)
val_bin_dict["total_active_bin_features"] = X_val_bin.sum(axis=1).astype(np.float32)
te_bin_dict["total_active_bin_features"] = X_te_bin.sum(axis=1).astype(np.float32)

tr_bin_dict["active_bin_ratio"] = tr_bin_dict["total_active_bin_features"] / len(binary_cols)
val_bin_dict["active_bin_ratio"] = val_bin_dict["total_active_bin_features"] / len(binary_cols)
te_bin_dict["active_bin_ratio"] = te_bin_dict["total_active_bin_features"] / len(binary_cols)

prefixes = set()
for col in binary_cols:
    if "_" in col:
        prefix = col.split("_")[0]
        if len(prefix) > 1:
            prefixes.add(prefix)
    elif "url" in col.lower():
        prefixes.add("url")
    elif "alt" in col.lower():
        prefixes.add("alt")
    elif "anc" in col.lower():
        prefixes.add("anc")

for prefix in sorted(prefixes):
    group_cols = [c for c in binary_cols if c.startswith(prefix) or prefix in c.lower()]
    if len(group_cols) >= 3:
        tr_bin_dict[f"sum_{prefix}_features"] = X_tr_bin[group_cols].sum(axis=1).astype(np.float32)
        val_bin_dict[f"sum_{prefix}_features"] = X_val_bin[group_cols].sum(axis=1).astype(np.float32)
        te_bin_dict[f"sum_{prefix}_features"] = X_te_bin[group_cols].sum(axis=1).astype(np.float32)

X_tr_bin_stats = pd.DataFrame(tr_bin_dict, index=X_tr_bin.index)
X_val_bin_stats = pd.DataFrame(val_bin_dict, index=X_val_bin.index)
X_te_bin_stats = pd.DataFrame(te_bin_dict, index=X_te_bin.index)

n_svd_components = 16
svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
svd_tr = svd.fit_transform(X_tr_bin)
svd_val = svd.transform(X_val_bin)
svd_te = svd.transform(X_te_bin)

svd_cols = [f"svd_bin_{i}" for i in range(n_svd_components)]
X_tr_svd = pd.DataFrame(svd_tr, columns=svd_cols, index=X_tr_bin.index)
X_val_svd = pd.DataFrame(svd_val, columns=svd_cols, index=X_val_bin.index)
X_te_svd = pd.DataFrame(svd_te, columns=svd_cols, index=X_te_bin.index)

# Combine Features
X_tr_final = pd.concat([X_tr_geo, X_tr_bin_stats, X_tr_svd, X_tr_bin], axis=1)
X_val_final = pd.concat([X_val_geo, X_val_bin_stats, X_val_svd, X_val_bin], axis=1)
X_te_final = pd.concat([X_te_geo, X_te_bin_stats, X_te_svd, X_te_bin], axis=1)

X_tr_final.columns = [str(c) for c in X_tr_final.columns]
X_val_final.columns = [str(c) for c in X_val_final.columns]
X_te_final.columns = [str(c) for c in X_te_final.columns]


# Neural Network Architecture Definition
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularAdNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.3
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.resblock1 = ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
        self.downsample = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
        )
        self.resblock2 = ResidualBlock(hidden_dim // 2, dropout_rate=dropout_rate / 2.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.resblock1(out)
        out = self.downsample(out)
        out = self.resblock2(out)
        return self.head(out)


# Train LightGBM Model
print("Training LightGBM model...")
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_tr_final,
    y_tr,
    eval_set=[(X_val_final, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)
lgb_val_preds = lgb_model.predict_proba(X_val_final)[:, 1]
lgb_test_preds = lgb_model.predict_proba(X_te_final)[:, 1]
lgb_val_auc = roc_auc_score(y_val, lgb_val_preds)
print(f"LightGBM Validation ROC-AUC: {lgb_val_auc:.5f}")

# Train XGBoost Model
print("Training XGBoost model...")
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_tr_final, y_tr, eval_set=[(X_val_final, y_val)], verbose=False)
xgb_val_preds = xgb_model.predict_proba(X_val_final)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_te_final)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_preds)
print(f"XGBoost Validation ROC-AUC: {xgb_val_auc:.5f}")

# Train CatBoost Model
print("Training CatBoost model...")
cb_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}
cb_model = CatBoostClassifier(**cb_params)
cb_model.fit(
    X_tr_final,
    y_tr,
    eval_set=(X_val_final, y_val),
    verbose=False,
)
cb_val_preds = cb_model.predict_proba(X_val_final)[:, 1]
cb_test_preds = cb_model.predict_proba(X_te_final)[:, 1]
cb_val_auc = roc_auc_score(y_val, cb_val_preds)
print(f"CatBoost Validation ROC-AUC: {cb_val_auc:.5f}")

# Train PyTorch TabularAdNet
print("Training PyTorch TabularAdNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tr_tensor = torch.tensor(X_tr_final.values, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_final.values, dtype=torch.float32)
X_te_tensor = torch.tensor(X_te_final.values, dtype=torch.float32)

train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

nn_model = TabularAdNet(input_dim=X_tr_final.shape[1], hidden_dim=256, dropout_rate=0.3)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
nn_model.to(device)

best_val_auc = 0.0
best_weights = None
patience = 12
patience_counter = 0

for epoch in range(1, 41):
    nn_model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = nn_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

    train_loss = running_loss / len(train_dataset)

    nn_model.eval()
    with torch.no_grad():
        val_logits = nn_model(X_val_tensor.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
        val_auc = roc_auc_score(y_val, val_probs)

    scheduler.step(val_auc)
    print(
        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {val_auc:.5f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
nn_model.eval()
with torch.no_grad():
    nn_val_preds = (
        torch.sigmoid(nn_model(X_val_tensor.to(device))).cpu().numpy().flatten()
    )
    nn_test_preds = (
        torch.sigmoid(nn_model(X_te_tensor.to(device))).cpu().numpy().flatten()
    )

print(f"PyTorch TabularAdNet Validation ROC-AUC: {best_val_auc:.5f}")

# Ensembling & Final Evaluation
print("Performing rank transformation and optimal weight search...")
val_ranks = np.column_stack([
    rankdata(lgb_val_preds) / len(lgb_val_preds),
    rankdata(xgb_val_preds) / len(xgb_val_preds),
    rankdata(cb_val_preds) / len(cb_val_preds),
    rankdata(nn_val_preds) / len(nn_val_preds),
])

test_ranks = np.column_stack([
    rankdata(lgb_test_preds) / len(lgb_test_preds),
    rankdata(xgb_test_preds) / len(xgb_test_preds),
    rankdata(cb_test_preds) / len(cb_test_preds),
    rankdata(nn_test_preds) / len(nn_test_preds),
])


def loss_func(weights):
    weights = np.maximum(0, np.array(weights))
    s = np.sum(weights)
    if s == 0:
        return 0.0
    w = weights / s
    blend = np.dot(val_ranks, w)
    return -roc_auc_score(y_val, blend)


init_weights = [0.25, 0.25, 0.25, 0.25]
res = minimize(loss_func, init_weights, method="Nelder-Mead")

opt_weights = np.maximum(0, res.x)
if np.sum(opt_weights) > 0:
    opt_weights = opt_weights / np.sum(opt_weights)
else:
    opt_weights = np.array(init_weights)

print(f"Optimal Ensemble Weights (LGB, XGB, CB, NN): {opt_weights}")

final_val_blend = np.dot(val_ranks, opt_weights)
final_test_blend = np.dot(test_ranks, opt_weights)

final_val_auc = roc_auc_score(y_val, final_val_blend)

submission_df = pd.DataFrame({"id": test_ids, "class": final_test_blend})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_auc}")