import json
import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure reproducibility and required output directories
np.random.seed(42)
torch.manual_seed(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# 1. Data Loading & Feature Engineering Pipeline
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Overall"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

if "id" in train_df.columns:
    train_id = train_df["id"]
    test_id = test_df["id"]
else:
    train_id = train_df.index
    test_id = test_df.index

exclude_cols = [target_col, "id"]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train_df[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Impute missing values leak-free
for col in numeric_cols:
    med = X_train[col].median()
    if pd.isna(med):
        med = 0.0
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

# Log1p transforms for skewed numeric columns
top_num_cols = []
if len(numeric_cols) > 0:
    for col in numeric_cols:
        skew_val = X_train[col].skew()
        if abs(skew_val) > 0.75:
            min_val = X_train[col].min()
            if min_val < 0:
                X_train[f"{col}_log1p"] = np.log1p(X_train[col] - min_val)
                X_test[f"{col}_log1p"] = np.log1p((X_test[col] - min_val).clip(lower=0))
            else:
                X_train[f"{col}_log1p"] = np.log1p(X_train[col])
                X_test[f"{col}_log1p"] = np.log1p(X_test[col].clip(lower=0))

    # Row-wise aggregate statistics over numeric features
    X_train["row_mean"] = X_train[numeric_cols].mean(axis=1)
    X_test["row_mean"] = X_test[numeric_cols].mean(axis=1)

    X_train["row_std"] = X_train[numeric_cols].std(axis=1).fillna(0)
    X_test["row_std"] = X_test[numeric_cols].std(axis=1).fillna(0)

    X_train["row_min"] = X_train[numeric_cols].min(axis=1)
    X_test["row_min"] = X_test[numeric_cols].min(axis=1)

    X_train["row_max"] = X_train[numeric_cols].max(axis=1)
    X_test["row_max"] = X_test[numeric_cols].max(axis=1)

    X_train["row_range"] = X_train["row_max"] - X_train["row_min"]
    X_test["row_range"] = X_test["row_max"] - X_test["row_min"]

    X_train["row_sum"] = X_train[numeric_cols].sum(axis=1)
    X_test["row_sum"] = X_test[numeric_cols].sum(axis=1)

    X_train["row_q25"] = X_train[numeric_cols].quantile(0.25, axis=1)
    X_test["row_q25"] = X_test[numeric_cols].quantile(0.25, axis=1)

    X_train["row_q75"] = X_train[numeric_cols].quantile(0.75, axis=1)
    X_test["row_q75"] = X_test[numeric_cols].quantile(0.75, axis=1)

    # Interaction terms for top numeric features sorted by variance
    variances = X_train[numeric_cols].var()
    top_num_cols = list(variances.sort_values(ascending=False).index[:min(5, len(numeric_cols))])

    for i in range(len(top_num_cols)):
        col1 = top_num_cols[i]
        X_train[f"{col1}_diff_mean"] = X_train[col1] - X_train["row_mean"]
        X_test[f"{col1}_diff_mean"] = X_test[col1] - X_test["row_mean"]

        for j in range(i + 1, len(top_num_cols)):
            col2 = top_num_cols[j]
            X_train[f"{col1}_plus_{col2}"] = X_train[col1] + X_train[col2]
            X_test[f"{col1}_plus_{col2}"] = X_test[col1] + X_test[col2]

            X_train[f"{col1}_minus_{col2}"] = X_train[col1] - X_train[col2]
            X_test[f"{col1}_minus_{col2}"] = X_test[col1] - X_test[col2]

            X_train[f"{col1}_mul_{col2}"] = X_train[col1] * X_train[col2]
            X_test[f"{col1}_mul_{col2}"] = X_test[col1] * X_test[col2]

            X_train[f"{col1}_div_{col2}"] = X_train[col1] / (X_train[col2] + 1e-5)
            X_test[f"{col1}_div_{col2}"] = X_test[col1] / (X_test[col2] + 1e-5)

            X_train[f"{col2}_div_{col1}"] = X_train[col2] / (X_train[col1] + 1e-5)
            X_test[f"{col2}_div_{col1}"] = X_test[col2] / (X_test[col1] + 1e-5)

# Frequency encoding for categoricals
for col in categorical_cols:
    freq = X_train[col].value_counts(normalize=True)
    X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)

# PCA removed prior to CV split to avoid data leakage

# 10-Fold Quantile Stratified Split
n_splits = 10
y_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

X_train["fold"] = -1
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bins)):
    X_train.loc[val_idx, "fold"] = fold

# Out-Of-Fold Smoothed Target Encoding (m-estimate with m=10)
global_mean = float(y_train.mean())
m_weight = 10.0
for col in categorical_cols:
    X_train[f"{col}_te"] = np.nan
    for fold in range(n_splits):
        idx_tr = X_train["fold"] != fold
        idx_va = X_train["fold"] == fold

        fold_stats = y_train[idx_tr].groupby(X_train.loc[idx_tr, col]).agg(["count", "mean"])
        fold_te = (fold_stats["count"] * fold_stats["mean"] + m_weight * global_mean) / (fold_stats["count"] + m_weight)
        X_train.loc[idx_va, f"{col}_te"] = X_train.loc[idx_va, col].map(fold_te)

    X_train[f"{col}_te"] = X_train[f"{col}_te"].fillna(global_mean)

    full_stats = y_train.groupby(X_train[col]).agg(["count", "mean"])
    full_te = (full_stats["count"] * full_stats["mean"] + m_weight * global_mean) / (full_stats["count"] + m_weight)
    X_test[f"{col}_te"] = X_test[col].map(full_te).fillna(global_mean)

# Out-Of-Fold Group-by Categorical Aggregations (mean and std of top numeric features per category)
for col in categorical_cols:
    for num_col in top_num_cols:
        X_train[f"{col}_{num_col}_mean"] = np.nan
        X_train[f"{col}_{num_col}_std"] = np.nan

        for fold in range(n_splits):
            idx_tr = X_train["fold"] != fold
            idx_va = X_train["fold"] == fold

            grp_tr = X_train.loc[idx_tr].groupby(col)[num_col].agg(["mean", "std"])
            X_train.loc[idx_va, f"{col}_{num_col}_mean"] = X_train.loc[idx_va, col].map(grp_tr["mean"])
            X_train.loc[idx_va, f"{col}_{num_col}_std"] = X_train.loc[idx_va, col].map(grp_tr["std"])

        global_num_mean = X_train[num_col].mean()
        X_train[f"{col}_{num_col}_mean"] = X_train[f"{col}_{num_col}_mean"].fillna(global_num_mean)
        X_train[f"{col}_{num_col}_std"] = X_train[f"{col}_{num_col}_std"].fillna(0.0)

        grp_full = X_train.groupby(col)[num_col].agg(["mean", "std"])
        X_test[f"{col}_{num_col}_mean"] = X_test[col].map(grp_full["mean"]).fillna(global_num_mean)
        X_test[f"{col}_{num_col}_std"] = X_test[col].map(grp_full["std"]).fillna(0.0)

final_feature_cols = [
    c for c in X_train.columns if c not in ["fold"] + categorical_cols
]

X_train_proc = X_train[final_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_proc = X_test[final_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
folds = X_train["fold"].values

n_train = len(X_train_proc)
n_test = len(X_test_proc)
n_features = len(final_feature_cols)

# Save Metadata
metadata = {
    "target_col": target_col,
    "feature_cols": final_feature_cols,
    "n_splits": n_splits,
    "n_train": n_train,
    "n_test": n_test,
    "n_features": n_features,
}

with open("./working/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


# ==============================================================================
# 2. PyTorch Tabular ResNet Model Architecture
# ==============================================================================
class SqueezeExcitation1D(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w


class TabularResBlock(nn.Module):

    def __init__(self, in_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, in_features * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(in_features * 2, in_features)
        self.se = SqueezeExcitation1D(in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.se(out)
        out = self.dropout(out)
        return residual + out


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
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


# ==============================================================================
# 3. Hyperparameters Setup
# ==============================================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "iterations": 2000,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "early_stopping_rounds": 50,
}

# ==============================================================================
# 4. Model Training & Cross-Validation Loop
# ==============================================================================
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_torch = np.zeros(n_train)
oof_et = np.zeros(n_train)
oof_ridge = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_torch = np.zeros(n_test)
test_preds_et = np.zeros(n_test)
test_preds_ridge = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(n_splits):
    tr_idx = folds != fold
    va_idx = folds == fold

    X_tr = X_train_proc.iloc[tr_idx].copy()
    y_tr = y_train.iloc[tr_idx].copy()
    X_va = X_train_proc.iloc[va_idx].copy()
    y_va = y_train.iloc[va_idx].copy()
    X_te = X_test_proc.copy()

    # Target Power Transformation (Yeo-Johnson)
    pt_y = PowerTransformer(method="yeo-johnson")
    y_tr_trans = pt_y.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
    y_va_trans = pt_y.transform(y_va.values.reshape(-1, 1)).ravel()

    # Compute fold-wise PCA and SVD decomposition features leak-free
    scaler_decomp = StandardScaler()
    X_tr_scaled_d = scaler_decomp.fit_transform(X_tr)
    X_va_scaled_d = scaler_decomp.transform(X_va)
    X_te_scaled_d = scaler_decomp.transform(X_te)

    n_comp = min(4, X_tr.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)

    tr_pca = pca.fit_transform(X_tr_scaled_d)
    va_pca = pca.transform(X_va_scaled_d)
    te_pca = pca.transform(X_te_scaled_d)

    tr_svd = svd.fit_transform(X_tr_scaled_d)
    va_svd = svd.transform(X_va_scaled_d)
    te_svd = svd.transform(X_te_scaled_d)

    for i in range(n_comp):
        X_tr[f"pca_{i}"] = tr_pca[:, i]
        X_va[f"pca_{i}"] = va_pca[:, i]
        X_te[f"pca_{i}"] = te_pca[:, i]

        X_tr[f"svd_{i}"] = tr_svd[:, i]
        X_va[f"svd_{i}"] = va_svd[:, i]
        X_te[f"svd_{i}"] = te_svd[:, i]

    # Quantile Feature Scaling (RankGauss) fitted strictly on fold training data
    qt = QuantileTransformer(
        output_distribution="normal",
        random_state=42,
        n_quantiles=min(100, len(X_tr)),
    )
    X_tr_qt = pd.DataFrame(qt.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_va_qt = pd.DataFrame(qt.transform(X_va), columns=X_va.columns, index=X_va.index)
    X_te_qt = pd.DataFrame(qt.transform(X_te), columns=X_te.columns, index=X_te.index)

    # Mutual Information Feature Selection to filter uninformative interaction columns
    mi_scores = mutual_info_regression(X_tr_qt, y_tr_trans, random_state=42)
    mi_threshold = np.percentile(mi_scores, 25)
    selected_mask = mi_scores > max(0.0, mi_threshold)
    if np.sum(selected_mask) < 10:
        top_idx = np.argsort(mi_scores)[-min(15, len(mi_scores)):]
        selected_cols = X_tr.columns[top_idx].tolist()
    else:
        selected_cols = X_tr.columns[selected_mask].tolist()

    X_tr_feat = X_tr_qt[selected_cols].copy()
    X_va_feat = X_va_qt[selected_cols].copy()
    X_te_feat = X_te_qt[selected_cols].copy()

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr_feat,
        y_tr_trans,
        eval_set=[(X_va_feat, y_va_trans)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    lgb_va_pred = lgb_model.predict(X_va_feat)
    lgb_te_pred = lgb_model.predict(X_te_feat)
    oof_lgb[va_idx] = pt_y.inverse_transform(lgb_va_pred.reshape(-1, 1)).ravel()
    test_preds_lgb += pt_y.inverse_transform(lgb_te_pred.reshape(-1, 1)).ravel() / n_splits

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr_feat, y_tr_trans, eval_set=[(X_va_feat, y_va_trans)], verbose=False)
    xgb_va_pred = xgb_model.predict(X_va_feat)
    xgb_te_pred = xgb_model.predict(X_te_feat)
    oof_xgb[va_idx] = pt_y.inverse_transform(xgb_va_pred.reshape(-1, 1)).ravel()
    test_preds_xgb += pt_y.inverse_transform(xgb_te_pred.reshape(-1, 1)).ravel() / n_splits

    # --- CatBoost ---
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr_feat, y_tr_trans, eval_set=(X_va_feat, y_va_trans), verbose=False)
    cat_va_pred = cat_model.predict(X_va_feat)
    cat_te_pred = cat_model.predict(X_te_feat)
    oof_cat[va_idx] = pt_y.inverse_transform(cat_va_pred.reshape(-1, 1)).ravel()
    test_preds_cat += pt_y.inverse_transform(cat_te_pred.reshape(-1, 1)).ravel() / n_splits

    # --- PyTorch Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feat)
    X_va_scaled = scaler.transform(X_va_feat)
    X_test_scaled = scaler.transform(X_te_feat)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_trans, dtype=torch.float32),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va_trans, dtype=torch.float32),
    )

    batch_sz = min(128, len(X_tr_feat))
    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_sz, shuffle=True, drop_last=False
    )
    va_loader = DataLoader(va_dataset, batch_size=256, shuffle=False, drop_last=False)

    model = TabularResNet(input_dim=X_tr_feat.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 40
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None
    patience, patience_cnt = 10, 0

    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_cnt = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_cnt += len(by)

        val_rmse = np.sqrt(val_loss / max(1, val_cnt))
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for bx, _ in va_loader:
            bx = bx.to(device)
            val_preds_list.append(model(bx).cpu().numpy())
    torch_va_pred = np.concatenate(val_preds_list)

    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        torch_te_pred = model(test_tensor).cpu().numpy()

    oof_torch[va_idx] = pt_y.inverse_transform(torch_va_pred.reshape(-1, 1)).ravel()
    test_preds_torch += pt_y.inverse_transform(torch_te_pred.reshape(-1, 1)).ravel() / n_splits

    # --- ExtraTreesRegressor ---
    et_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    et_model.fit(X_tr_scaled, y_tr_trans)
    et_va_pred = et_model.predict(X_va_scaled)
    et_te_pred = et_model.predict(X_test_scaled)
    oof_et[va_idx] = pt_y.inverse_transform(et_va_pred.reshape(-1, 1)).ravel()
    test_preds_et += pt_y.inverse_transform(et_te_pred.reshape(-1, 1)).ravel() / n_splits

    # --- Ridge ---
    ridge_model = Ridge(random_state=42)
    ridge_model.fit(X_tr_scaled, y_tr_trans)
    ridge_va_pred = ridge_model.predict(X_va_scaled)
    ridge_te_pred = ridge_model.predict(X_test_scaled)
    oof_ridge[va_idx] = pt_y.inverse_transform(ridge_va_pred.reshape(-1, 1)).ravel()
    test_preds_ridge += pt_y.inverse_transform(ridge_te_pred.reshape(-1, 1)).ravel() / n_splits

# ==============================================================================
# 5. SLSQP Optimization & Ensemble Blending
# ==============================================================================
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_torch, oof_et, oof_ridge])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_torch, test_preds_et, test_preds_ridge]
)


def blend_objective(weights, oofs, y_true):
    pred = np.dot(oofs, weights)
    return np.sqrt(mean_squared_error(y_true, pred))


initial_w = np.ones(6) / 6.0
bounds = [(0.0, 1.0) for _ in range(6)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    blend_objective,
    initial_w,
    args=(oof_matrix, y_train.values),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

opt_weights = res.x / np.sum(res.x)

final_oof_preds = np.dot(oof_matrix, opt_weights)
final_test_preds = np.dot(test_matrix, opt_weights)

# Domain clipping bounds
y_min, y_max = y_train.min(), y_train.max()
final_oof_preds = np.clip(final_oof_preds, y_min, y_max)
final_test_preds = np.clip(final_test_preds, y_min, y_max)

# Compute Validation RMSE
final_rmse = np.sqrt(mean_squared_error(y_train, final_oof_preds))

# ==============================================================================
# 6. Save Submission & Output Final Metric
# ==============================================================================
submission_df = pd.DataFrame({"id": test_id, "Overall": final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse}")