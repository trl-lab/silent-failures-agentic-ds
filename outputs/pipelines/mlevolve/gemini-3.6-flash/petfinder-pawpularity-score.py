import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Environment & Reproducibility Setup
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 2. Data Loading
train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Pawpularity"
id_col = (
    "id"
    if "id" in train_df.columns
    else ("Id" if "Id" in train_df.columns else None)
)
meta_cols = [
    c
    for c in train_df.columns
    if c not in [target_col, id_col] and pd.api.types.is_numeric_dtype(train_df[c])
]


# 3. Domain & Interaction Feature Engineering
def build_engineered_features(df, base_meta_cols):
    df_feat = df.copy()
    base_meta_cols = [
        c for c in base_meta_cols if pd.api.types.is_numeric_dtype(df_feat[c])
    ]

    df_feat["meta_sum"] = df_feat[base_meta_cols].sum(axis=1)
    df_feat["meta_mean"] = df_feat[base_meta_cols].mean(axis=1)
    df_feat["meta_std"] = df_feat[base_meta_cols].std(axis=1)

    clarity_cols = [c for c in ["Eyes", "Face", "Near", "Focus"] if c in base_meta_cols]
    distraction_cols = [
        c for c in ["Accessory", "Group", "Collateral", "Blur"] if c in base_meta_cols
    ]

    if clarity_cols:
        df_feat["clarity_score"] = df_feat[clarity_cols].sum(axis=1)
    if distraction_cols:
        df_feat["distraction_score"] = df_feat[distraction_cols].sum(axis=1)

    if clarity_cols and distraction_cols:
        df_feat["clarity_distraction_ratio"] = (df_feat["clarity_score"] + 1.0) / (
            df_feat["distraction_score"] + 1.0
        )
        df_feat["clarity_distraction_diff"] = (
            df_feat["clarity_score"] - df_feat["distraction_score"]
        )

    if "Focus" in base_meta_cols and "Blur" in base_meta_cols:
        df_feat["focus_no_blur"] = (
            (df_feat["Focus"] == 1) & (df_feat["Blur"] == 0)
        ).astype(int)
        df_feat["focus_blur_clash"] = (
            (df_feat["Focus"] == 1) & (df_feat["Blur"] == 1)
        ).astype(int)

    if "Eyes" in base_meta_cols and "Face" in base_meta_cols:
        df_feat["eyes_and_face"] = (
            (df_feat["Eyes"] == 1) & (df_feat["Face"] == 1)
        ).astype(int)

    if "Group" in base_meta_cols and "Collateral" in base_meta_cols:
        df_feat["multi_subject"] = (
            (df_feat["Group"] == 1) | (df_feat["Collateral"] == 1)
        ).astype(int)

    num_meta = len(base_meta_cols)
    for i in range(num_meta):
        for j in range(i + 1, num_meta):
            col1, col2 = base_meta_cols[i], base_meta_cols[j]
            df_feat[f"{col1}_AND_{col2}"] = (df_feat[col1] * df_feat[col2]).astype(int)
            df_feat[f"{col1}_XOR_{col2}"] = (df_feat[col1] != df_feat[col2]).astype(int)

    df_feat["pattern_bitstring"] = (
        df_feat[base_meta_cols].astype(str).agg("".join, axis=1)
    )

    return df_feat


train_feat_df = build_engineered_features(train_df, meta_cols)
test_feat_df = build_engineered_features(test_df, meta_cols)

# KMeans fitting moved into CV loop to prevent fold leakage

pattern_counts = train_feat_df["pattern_bitstring"].value_counts().to_dict()
train_feat_df["pattern_freq"] = (
    train_feat_df["pattern_bitstring"].map(pattern_counts).fillna(1).astype(float)
)
test_feat_df["pattern_freq"] = (
    test_feat_df["pattern_bitstring"].map(pattern_counts).fillna(1).astype(float)
)

exclude_cols = [target_col, "fold", "bin", "pattern_bitstring"]
if id_col:
    exclude_cols.append(id_col)

feature_cols = [
    c
    for c in train_feat_df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_feat_df[c])
]

# 4. Stratified K-Fold Setup
num_bins = int(np.floor(1 + np.log2(len(train_feat_df))))
train_feat_df["bin"] = pd.qcut(
    train_feat_df[target_col], q=num_bins, labels=False, duplicates="drop"
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_feat_df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_feat_df, train_feat_df["bin"])
):
    train_feat_df.loc[val_idx, "fold"] = fold

train_feat_df.drop(columns=["bin"], inplace=True)

# Cross-fold Ridge linear OOF feature on metadata
train_feat_df["ridge_meta_oof"] = 0.0
test_feat_df["ridge_meta_oof"] = 0.0
ridge_meta = Ridge(alpha=10.0, random_state=SEED)

for fold in range(5):
    tr_mask = train_feat_df["fold"] != fold
    v_mask = train_feat_df["fold"] == fold
    ridge_meta.fit(
        train_feat_df.loc[tr_mask, meta_cols],
        train_feat_df.loc[tr_mask, target_col],
    )
    train_feat_df.loc[v_mask, "ridge_meta_oof"] = ridge_meta.predict(
        train_feat_df.loc[v_mask, meta_cols]
    )
    test_feat_df["ridge_meta_oof"] += (
        ridge_meta.predict(test_feat_df[meta_cols]) / 5.0
    )

feature_cols = [
    c
    for c in train_feat_df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_feat_df[c])
]


# 5. Model Architecture Definitions
class GatedLinearUnit(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc(x)
        val, gate = x_proj.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class TabGLUBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.glu(x)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out


class TabGLUNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.25,
        min_val: float = 1.0,
        max_val: float = 100.0,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )

        self.blocks = nn.ModuleList(
            [TabGLUBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        raw_pred = self.head(x)
        return raw_pred.squeeze(-1)


class RMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


# 6. Stratified Cross-Validation & Modeling Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_train = len(train_feat_df)
n_test = len(test_feat_df)
n_folds = 5

oof_preds_ridge = np.zeros(n_train)
oof_preds_svr = np.zeros(n_train)
oof_preds_nn = np.zeros(n_train)
oof_preds_lgb = np.zeros(n_train)
oof_preds_xgb = np.zeros(n_train)
oof_preds_cat = np.zeros(n_train)

test_preds_ridge = np.zeros(n_test)
test_preds_svr = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)
test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)

y_all = train_feat_df[target_col].values
X_test_base = test_feat_df[feature_cols].values

for fold in range(n_folds):
    train_mask = train_feat_df["fold"] != fold
    val_mask = train_feat_df["fold"] == fold
    val_idx = np.where(val_mask)[0]

    X_train_f = train_feat_df.loc[train_mask, feature_cols].values
    y_train_f = train_feat_df.loc[train_mask, target_col].values

    X_val_f = train_feat_df.loc[val_mask, feature_cols].values
    y_val_f = train_feat_df.loc[val_mask, target_col].values

    # Fit scaling and manifold features strictly on train fold
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    tr_meta_idx = [feature_cols.index(c) for c in meta_cols if c in feature_cols]
    kmeans.fit(X_train_f[:, tr_meta_idx])

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)
    X_te_scaled = scaler.transform(X_test_base)

    pca = PCA(n_components=5, random_state=SEED)
    pca_tr = pca.fit_transform(X_tr_scaled)
    pca_val = pca.transform(X_val_scaled)
    pca_te = pca.transform(X_te_scaled)

    svd = TruncatedSVD(n_components=5, random_state=SEED)
    svd_tr = svd.fit_transform(X_tr_scaled)
    svd_val = svd.transform(X_val_scaled)
    svd_te = svd.transform(X_te_scaled)

    ica = FastICA(n_components=5, random_state=SEED, max_iter=1000)
    ica_tr = ica.fit_transform(X_tr_scaled)
    ica_val = ica.transform(X_val_scaled)
    ica_te = ica.transform(X_te_scaled)

    X_train_fold = np.hstack([X_tr_scaled, pca_tr, svd_tr, ica_tr])
    X_val_fold = np.hstack([X_val_scaled, pca_val, svd_val, ica_val])
    X_test_fold = np.hstack([X_te_scaled, pca_te, svd_te, ica_te])

    # --- Linear Model 1: Ridge ---
    ridge_model = Ridge(alpha=10.0, random_state=SEED)
    ridge_model.fit(X_train_fold, y_train_f)
    oof_preds_ridge[val_idx] = ridge_model.predict(X_val_fold)
    test_preds_ridge += ridge_model.predict(X_test_fold) / n_folds

    # --- Kernel Model 2: SVR ---
    svr_model = SVR(C=1.0, epsilon=0.1)
    svr_model.fit(X_train_fold, y_train_f)
    oof_preds_svr[val_idx] = svr_model.predict(X_val_fold)
    test_preds_svr += svr_model.predict(X_test_fold) / n_folds

    # --- Neural Model: TabGLUNet ---
    train_ds = TensorDataset(
        torch.tensor(X_train_fold, dtype=torch.float32),
        torch.tensor(y_train_f, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_fold, dtype=torch.float32),
        torch.tensor(y_val_f, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    nn_model = TabGLUNet(
        input_dim=X_train_fold.shape[1],
        hidden_dim=128,
        num_blocks=3,
        dropout=0.25,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    patience, patience_counter = 10, 0

    for epoch in range(40):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = nn_model(bx)
            loss = criterion(preds, by / 100.0)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_sq_errors = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = nn_model(bx)
                preds_prob = torch.sigmoid(preds) * 100.0
                sq_err = torch.sum((preds_prob - by) ** 2)
                val_sq_errors.append(sq_err.item())

        epoch_val_rmse = np.sqrt(np.sum(val_sq_errors) / len(y_val_f))

        if epoch_val_rmse < best_val_loss:
            best_val_loss = epoch_val_rmse
            best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
        nn_val_pred = (torch.sigmoid(nn_model(val_tensor)).cpu().numpy()) * 100.0
        nn_test_pred = (torch.sigmoid(nn_model(test_tensor)).cpu().numpy()) * 100.0

    oof_preds_nn[val_idx] = nn_val_pred
    test_preds_nn += nn_test_pred / n_folds

    # --- Tree Model 1: LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_fold,
        y_train_f,
        eval_set=[(X_val_fold, y_val_f)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    lgb_val_pred = lgb_model.predict(X_val_fold)
    oof_preds_lgb[val_idx] = lgb_val_pred
    test_preds_lgb += lgb_model.predict(X_test_fold) / n_folds

    # --- Tree Model 2: XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        early_stopping_rounds=50,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train_fold,
        y_train_f,
        eval_set=[(X_val_fold, y_val_f)],
        verbose=False,
    )
    xgb_val_pred = xgb_model.predict(X_val_fold)
    oof_preds_xgb[val_idx] = xgb_val_pred
    test_preds_xgb += xgb_model.predict(X_test_fold) / n_folds

    # --- Tree Model 3: CatBoost ---
    cat_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1000,
        learning_rate=0.02,
        depth=3,
        l2_leaf_reg=5.0,
        early_stopping_rounds=50,
        random_seed=SEED + fold,
        verbose=0,
    )
    cat_model.fit(X_train_fold, y_train_f, eval_set=(X_val_fold, y_val_f), verbose=0)
    cat_val_pred = cat_model.predict(X_val_fold)
    oof_preds_cat[val_idx] = cat_val_pred
    test_preds_cat += cat_model.predict(X_test_fold) / n_folds

# 7. SLSQP Ensembling & Final Optimization
oof_matrix = np.column_stack(
    [oof_preds_nn, oof_preds_lgb, oof_preds_xgb, oof_preds_cat]
)
test_matrix = np.column_stack(
    [test_preds_nn, test_preds_lgb, test_preds_xgb, test_preds_cat]
)


def blend_loss(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    pred_clipped = np.clip(pred, 1.0, 100.0)
    return np.sqrt(mean_squared_error(y_all, pred_clipped))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

opt_res = minimize(
    blend_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

final_oof_preds = np.clip(np.dot(oof_matrix, best_weights), 1.0, 100.0)
final_test_preds = np.clip(np.dot(test_matrix, best_weights), 1.0, 100.0)

final_score = np.sqrt(mean_squared_error(y_all, final_oof_preds))

# 8. Output Submission CSV
sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df = pd.DataFrame(
    {"id": np.arange(len(test_feat_df)), "Pawpularity": final_test_preds}
)
sub_df.to_csv(sub_path, index=False)

print(f"Final Validation Score: {final_score}")