import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Target encoding: 'ad.' / 'ad' / 1 -> 1, 'nonad.' / 'nonad' / 0 -> 0
target_col = "class"
y_raw = df_train[target_col]


def encode_target_series(y_series):
    # Check if direct numeric conversion is valid
    try:
        y_num = pd.to_numeric(y_series)
        if not y_num.isna().any():
            unique_vals = np.sort(y_num.unique())
            if len(unique_vals) == 2:
                return (y_num == unique_vals[1]).astype(np.int32).values
    except Exception:
        pass

    # Normalize string representations
    y_str = y_series.astype(str).str.strip().str.lower().str.rstrip(".")
    if set(y_str.unique()).issubset({"0", "1", "0.0", "1.0"}):
        return y_str.isin({"1", "1.0"}).astype(np.int32).values

    def _map_val(s):
        if s in ("1", "1.0", "true", "yes", "ad"):
            return 1
        if s in ("0", "0.0", "false", "no", "nonad"):
            return 0
        if "ad" in s and "non" not in s:
            return 1
        return 0

    encoded = y_str.apply(_map_val).values.astype(np.int32)
    if len(np.unique(encoded)) < 2:
        codes, _ = pd.factorize(y_series)
        return codes.astype(np.int32)
    return encoded


y_train = encode_target_series(y_raw)
X_train_raw = df_train.drop(columns=[target_col]).copy()
X_test_raw = df_test.drop(columns=[target_col], errors="ignore").copy()


# ---------------------------------------------------------
# 2. Clean Continuous (Geometry) vs. Binary Features
# ---------------------------------------------------------
geom_cols = list(X_train_raw.columns[:3])
binary_cols = list(X_train_raw.columns[3:])

for col in geom_cols:
    X_train_raw[col] = pd.to_numeric(X_train_raw[col], errors="coerce")
    X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors="coerce")

for col in binary_cols:
    X_train_raw[col] = (
        pd.to_numeric(X_train_raw[col], errors="coerce").fillna(0).astype(np.float32)
    )
    X_test_raw[col] = (
        pd.to_numeric(X_test_raw[col], errors="coerce").fillna(0).astype(np.float32)
    )


# ---------------------------------------------------------
# 3. Feature Engineering: Geometry Features
# ---------------------------------------------------------
def create_geometry_features(df, geom_cols):
    h = df[geom_cols[0]].values.astype(np.float64)
    w = df[geom_cols[1]].values.astype(np.float64)
    ar = df[geom_cols[2]].values.astype(np.float64)

    geom_df = pd.DataFrame(index=df.index)

    # Raw values
    geom_df["geom_height"] = h
    geom_df["geom_width"] = w
    geom_df["geom_aspect_ratio"] = ar

    # Missingness indicators
    geom_df["geom_h_missing"] = np.isnan(h).astype(np.float32)
    geom_df["geom_w_missing"] = np.isnan(w).astype(np.float32)
    geom_df["geom_ar_missing"] = np.isnan(ar).astype(np.float32)
    geom_df["geom_missing_count"] = (
        np.isnan(h).astype(int) + np.isnan(w).astype(int) + np.isnan(ar).astype(int)
    ).astype(np.float32)

    # Computed geometric interactions
    safe_h = np.where(np.isnan(h), np.nan, np.maximum(h, 1.0))
    safe_w = np.where(np.isnan(w), np.nan, np.maximum(w, 1.0))

    geom_df["geom_area"] = safe_h * safe_w
    geom_df["geom_perimeter"] = 2.0 * (safe_h + safe_w)
    geom_df["geom_diagonal"] = np.sqrt(safe_h**2 + safe_w**2)
    geom_df["geom_computed_ar"] = safe_w / safe_h
    geom_df["geom_ar_discrepancy"] = np.abs(ar - (safe_w / safe_h))

    # Log transforms for scale invariance
    geom_df["geom_log_h"] = np.log1p(np.maximum(0, safe_h))
    geom_df["geom_log_w"] = np.log1p(np.maximum(0, safe_w))
    geom_df["geom_log_area"] = np.log1p(np.maximum(0, geom_df["geom_area"].values))

    # Granular IAB standard ad banner dimension matching flags
    standard_sizes = [
        ("468x60", 468, 60),
        ("728x90", 728, 90),
        ("300x250", 300, 250),
        ("120x600", 120, 600),
        ("160x600", 160, 600),
        ("336x280", 336, 280),
        ("250x250", 250, 250),
        ("200x200", 200, 200),
        ("180x150", 180, 150),
        ("125x125", 125, 125),
        ("234x60", 234, 60),
        ("88x31", 88, 31),
        ("300x600", 300, 600),
        ("970x90", 970, 90),
        ("970x250", 970, 250),
    ]

    for name, std_w, std_h in standard_sizes:
        geom_df[f"is_iab_{name}"] = (
            (np.abs(w - std_w) <= 6) & (np.abs(h - std_h) <= 6)
        ).astype(np.float32)

    # Standard aspect ratio matches
    geom_df["is_ar_banner_approx"] = (np.abs(ar - 7.8) <= 0.6).astype(np.float32)
    geom_df["is_ar_leaderboard_approx"] = (np.abs(ar - 8.09) <= 0.6).astype(np.float32)
    geom_df["is_ar_rect_approx"] = (np.abs(ar - 1.2) <= 0.25).astype(np.float32)
    geom_df["is_ar_skyscraper_approx"] = (np.abs(ar - 0.23) <= 0.1).astype(np.float32)
    geom_df["is_ar_square_approx"] = (np.abs(ar - 1.0) <= 0.15).astype(np.float32)

    return geom_df


train_geom = create_geometry_features(X_train_raw, geom_cols)
test_geom = create_geometry_features(X_test_raw, geom_cols)


# ---------------------------------------------------------
# 4. Feature Engineering: Binary Sparse Token Features
# ---------------------------------------------------------
# Remove invariant zero-variance binary columns
train_bin_vars = X_train_raw[binary_cols].var(axis=0)
active_binary_cols = [col for col in binary_cols if train_bin_vars[col] > 0.0]


def create_binary_aggregates(df, cols):
    mat = df[cols].values
    agg_df = pd.DataFrame(index=df.index)

    token_sum = np.sum(mat, axis=1)
    token_mean = np.mean(mat, axis=1)
    token_std = np.std(mat, axis=1)

    agg_df["token_sum_active"] = token_sum.astype(np.float32)
    agg_df["token_mean_active"] = token_mean.astype(np.float32)
    agg_df["token_std_active"] = token_std.astype(np.float32)
    agg_df["token_sparsity_ratio"] = (token_sum / float(len(cols))).astype(np.float32)

    return agg_df


train_bin_agg = create_binary_aggregates(X_train_raw, active_binary_cols)
test_bin_agg = create_binary_aggregates(X_test_raw, active_binary_cols)

# Ratio of active tokens to log-area and perimeter
train_geom["token_to_log_area"] = train_bin_agg["token_sum_active"] / (
    train_geom["geom_log_area"].fillna(0.0) + 1.0
)
test_geom["token_to_log_area"] = test_bin_agg["token_sum_active"] / (
    test_geom["geom_log_area"].fillna(0.0) + 1.0
)

X_train_bin = X_train_raw[active_binary_cols].values.astype(np.float32)
X_test_bin = X_test_raw[active_binary_cols].values.astype(np.float32)

# Smoothed token log-odds ad-intent scores strictly computed on train data
pos_mask = y_train == 1
neg_mask = y_train == 0
n_pos = float(np.sum(pos_mask))
n_neg = float(np.sum(neg_mask))
alpha_smooth = 1.0

pos_counts = np.sum(X_train_bin[pos_mask], axis=0)
neg_counts = np.sum(X_train_bin[neg_mask], axis=0)

p_pos = (pos_counts + alpha_smooth) / (n_pos + 2.0 * alpha_smooth)
p_neg = (neg_counts + alpha_smooth) / (n_neg + 2.0 * alpha_smooth)
token_log_odds = np.log(p_pos / p_neg).astype(np.float32)


def compute_token_log_odds_features(mat, weights, index):
    weighted_mat = mat * weights
    odds_sum = np.sum(weighted_mat, axis=1)

    pos_weights = np.maximum(weights, 0.0)
    neg_weights = np.minimum(weights, 0.0)
    odds_pos_sum = np.sum(mat * pos_weights, axis=1)
    odds_neg_sum = np.sum(mat * neg_weights, axis=1)

    active_counts = np.sum(mat, axis=1)
    safe_counts = np.maximum(active_counts, 1.0)
    odds_mean = odds_sum / safe_counts

    mat_pos_inf = np.where(mat > 0, weighted_mat, -999.0)
    mat_neg_inf = np.where(mat > 0, weighted_mat, 999.0)
    odds_max = np.where(active_counts > 0, np.max(mat_pos_inf, axis=1), 0.0)
    odds_min = np.where(active_counts > 0, np.min(mat_neg_inf, axis=1), 0.0)

    df_odds = pd.DataFrame(
        {
            "token_log_odds_sum": odds_sum.astype(np.float32),
            "token_log_odds_mean": odds_mean.astype(np.float32),
            "token_log_odds_max": odds_max.astype(np.float32),
            "token_log_odds_min": odds_min.astype(np.float32),
            "token_log_odds_pos_sum": odds_pos_sum.astype(np.float32),
            "token_log_odds_neg_sum": odds_neg_sum.astype(np.float32),
        },
        index=index,
    )
    return df_odds


train_odds_df = compute_token_log_odds_features(
    X_train_bin, token_log_odds, X_train_raw.index
)
test_odds_df = compute_token_log_odds_features(
    X_test_bin, token_log_odds, X_test_raw.index
)

# Multi-view decomposition: 32-component TruncatedSVD & 16-component NMF
n_svd_components = 32
svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
train_svd = svd.fit_transform(X_train_bin)
test_svd = svd.transform(X_test_bin)
svd_cols = [f"svd_comp_{i}" for i in range(n_svd_components)]
df_train_svd = pd.DataFrame(train_svd, columns=svd_cols, index=X_train_raw.index)
df_test_svd = pd.DataFrame(test_svd, columns=svd_cols, index=X_test_raw.index)

n_nmf_components = 16
nmf = NMF(n_components=n_nmf_components, init="nndsvda", random_state=42, max_iter=300)
train_nmf = nmf.fit_transform(X_train_bin)
test_nmf = nmf.transform(X_test_bin)
nmf_cols = [f"nmf_comp_{i}" for i in range(n_nmf_components)]
df_train_nmf = pd.DataFrame(train_nmf, columns=nmf_cols, index=X_train_raw.index)
df_test_nmf = pd.DataFrame(test_nmf, columns=nmf_cols, index=X_test_raw.index)

# Impute continuous geometry features using median computed strictly on train set
continuous_cols_to_impute = [
    col
    for col in train_geom.columns
    if not col.startswith("is_") and not col.endswith("_missing")
]
imputer = SimpleImputer(strategy="median")
train_geom_imputed = train_geom.copy()
test_geom_imputed = test_geom.copy()

train_geom_imputed[continuous_cols_to_impute] = imputer.fit_transform(
    train_geom[continuous_cols_to_impute]
)
test_geom_imputed[continuous_cols_to_impute] = imputer.transform(
    test_geom[continuous_cols_to_impute]
)

# Assemble Processed Datasets
X_train_processed = pd.concat(
    [
        train_geom_imputed,
        train_bin_agg,
        train_odds_df,
        df_train_svd,
        df_train_nmf,
        X_train_raw[active_binary_cols],
    ],
    axis=1,
)
X_test_processed = pd.concat(
    [
        test_geom_imputed,
        test_bin_agg,
        test_odds_df,
        df_test_svd,
        df_test_nmf,
        X_test_raw[active_binary_cols],
    ],
    axis=1,
)

X_train_processed.columns = [str(c) for c in X_train_processed.columns]
X_test_processed.columns = [str(c) for c in X_test_processed.columns]


# =============================================================================
# 5. Model Architectures & Configurations
# =============================================================================
class TabularGLUBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        val, gate = torch.chunk(h, 2, dim=-1)
        h = self.act(val) * torch.sigmoid(gate)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return residual + h


class AdTabularGLUNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
        out_features: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.glu_blocks = nn.ModuleList(
            [
                TabularGLUBlock(
                    hidden_dim=hidden_dim, dropout_rate=dropout_rate
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.glu_blocks:
            out = block(out)
        logits = self.head(out)
        return logits.squeeze(-1)


def build_torch_model_and_optimizer(
    in_features: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 1.0,
    epochs: int = 28,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = AdTabularGLUNet(
        in_features=in_features, hidden_dim=256, num_blocks=3, dropout_rate=0.25
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return model, criterion, optimizer, scheduler


def get_lgbm_classifier(scale_pos_weight: float = 1.0, random_state: int = 42):
    return lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_classifier(scale_pos_weight: float = 1.0, random_state: int = 42):
    return xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=0.5,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_classifier(scale_pos_weight: float = 1.0, random_state: int = 42):
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        scale_pos_weight=scale_pos_weight,
        random_seed=random_state,
        verbose=0,
        early_stopping_rounds=50,
    )


def get_et_classifier(random_state: int = 42):
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def get_linear_classifier(random_state: int = 42):
    return LogisticRegression(
        penalty="l2",
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )


# =============================================================================
# 6. Cross-Validation, Training & Evaluation Loop
# =============================================================================
feature_cols = [c for c in X_train_processed.columns]
X_train_all = X_train_processed.values.astype(np.float32)
y_train_all = y_train.copy()
X_test_all = X_test_processed.values.astype(np.float32)

n_samples = len(y_train_all)
n_test = len(X_test_all)
n_folds = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_resnet = np.zeros(n_samples, dtype=np.float32)
oof_lgbm = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_catboost = np.zeros(n_samples, dtype=np.float32)
oof_et = np.zeros(n_samples, dtype=np.float32)
oof_lr = np.zeros(n_samples, dtype=np.float32)

test_preds_resnet = np.zeros(n_test, dtype=np.float32)
test_preds_lgbm = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_catboost = np.zeros(n_test, dtype=np.float32)
test_preds_et = np.zeros(n_test, dtype=np.float32)
test_preds_lr = np.zeros(n_test, dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
    X_tr_raw, y_tr = X_train_all[train_idx], y_train_all[train_idx]
    X_val_raw, y_val = X_train_all[val_idx], y_train_all[val_idx]

    # Leak-free scaling for ResNet & Linear models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_all)

    n_pos = int(np.sum(y_tr == 1))
    n_neg = int(np.sum(y_tr == 0))
    pos_ratio = max(float(n_neg) / float(max(n_pos, 1)), 0.01)

    # --- LightGBM ---
    lgb_model = get_lgbm_classifier(scale_pos_weight=pos_ratio, random_state=42 + fold)
    lgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_val_raw, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred_lgb = lgb_model.predict_proba(X_val_raw)[:, 1]
    test_pred_lgb = lgb_model.predict_proba(X_test_all)[:, 1]
    oof_lgbm[val_idx] = val_pred_lgb
    test_preds_lgbm += test_pred_lgb / n_folds

    # --- XGBoost ---
    xgb_model = get_xgb_classifier(scale_pos_weight=pos_ratio, random_state=42 + fold)
    xgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_val_raw, y_val)],
        verbose=False,
    )
    val_pred_xgb = xgb_model.predict_proba(X_val_raw)[:, 1]
    test_pred_xgb = xgb_model.predict_proba(X_test_all)[:, 1]
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += test_pred_xgb / n_folds

    # --- CatBoost ---
    cb_model = get_catboost_classifier(scale_pos_weight=pos_ratio, random_state=42 + fold)
    cb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_val_raw, y_val)],
        verbose=False,
    )
    val_pred_cb = cb_model.predict_proba(X_val_raw)[:, 1]
    test_pred_cb = cb_model.predict_proba(X_test_all)[:, 1]
    oof_catboost[val_idx] = val_pred_cb
    test_preds_catboost += test_pred_cb / n_folds

    # --- Extra Trees ---
    et_model = get_et_classifier(random_state=42 + fold)
    et_model.fit(X_tr_raw, y_tr)
    val_pred_et = et_model.predict_proba(X_val_raw)[:, 1]
    test_pred_et = et_model.predict_proba(X_test_all)[:, 1]
    oof_et[val_idx] = val_pred_et
    test_preds_et += test_pred_et / n_folds

    # --- Logistic Regression ---
    lr_model = get_linear_classifier(random_state=42 + fold)
    lr_model.fit(X_tr_scaled, y_tr)
    val_pred_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
    test_pred_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    oof_lr[val_idx] = val_pred_lr
    test_preds_lr += test_pred_lr / n_folds

    # --- Deep Tabular ResNet ---
    epochs = 28
    batch_size = 128
    torch_model, criterion, optimizer, scheduler = build_torch_model_and_optimizer(
        in_features=X_tr_scaled.shape[1],
        lr=1e-3,
        weight_decay=1e-4,
        pos_weight=1.0,
        epochs=epochs,
        device=device,
    )

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    best_val_auc = -1.0
    best_weights = None

    for epoch in range(epochs):
        torch_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        torch_model.eval()
        with torch.no_grad():
            val_logits = torch_model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            curr_val_auc = roc_auc_score(y_val, val_probs)

            if curr_val_auc > best_val_auc:
                best_val_auc = curr_val_auc
                best_weights = copy.deepcopy(torch_model.state_dict())

    torch_model.load_state_dict(best_weights)
    torch_model.eval()
    with torch.no_grad():
        val_pred_resnet = (
            torch.sigmoid(torch_model(X_val_tensor)).cpu().numpy().flatten()
        )
        test_pred_resnet = (
            torch.sigmoid(torch_model(X_test_tensor)).cpu().numpy().flatten()
        )

    oof_resnet[val_idx] = val_pred_resnet
    test_preds_resnet += test_pred_resnet / n_folds

# =============================================================================
# 7. Overall Out-Of-Fold Evaluation & Ensembling
# =============================================================================
oof_list = [
    oof_lgbm,
    oof_xgb,
    oof_catboost,
    oof_et,
    oof_resnet,
    oof_lr,
]
test_list = [
    test_preds_lgbm,
    test_preds_xgb,
    test_preds_catboost,
    test_preds_et,
    test_preds_resnet,
    test_preds_lr,
]

oof_ranks_matrix = np.column_stack(
    [rankdata(preds) / float(len(preds)) for preds in oof_list]
)
test_ranks_matrix = np.column_stack(
    [rankdata(preds) / float(len(preds)) for preds in test_list]
)

n_models = len(oof_list)


def auc_loss(weights):
    w = np.array(weights)
    sum_w = np.sum(w)
    if sum_w <= 0:
        return 1.0
    w_norm = w / sum_w
    blended_ranks = oof_ranks_matrix @ w_norm
    return -roc_auc_score(y_train_all, blended_ranks)


initial_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    auc_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 200, "ftol": 1e-7},
)

opt_weights = opt_res.x
opt_weights = np.clip(opt_weights, 0.0, None)
opt_weights = opt_weights / np.sum(opt_weights)

oof_ensemble = oof_ranks_matrix @ opt_weights
final_val_auc = float(roc_auc_score(y_train_all, oof_ensemble))

# =============================================================================
# 8. Test Predictions & Submission Generation
# =============================================================================
final_test_preds = test_ranks_matrix @ opt_weights

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds), dtype=np.int64), "class": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_auc}")
