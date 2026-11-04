import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Set random seeds and execution device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Enrollment_FT_Teachers_Ratio"
id_col = "id" if "id" in train_df.columns else None

# Store test IDs for submission
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# Separate target
y = train_df[target_col].values.astype(np.float64)

# Prepare combined dataframe for consistent feature processing
if id_col and id_col in train_df.columns:
    train_features = train_df.drop(columns=[id_col, target_col])
else:
    train_features = train_df.drop(columns=[target_col])

if id_col and id_col in test_df.columns:
    test_features = test_df.drop(columns=[id_col])
else:
    test_features = test_df.copy()

n_train = len(train_features)
combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# 2. Dynamic Geospatial Feature Extraction
lat_cols = [
    c for c in combined.columns if c.lower() in ["latitude", "lat", "y", "shape_y"]
]
lon_cols = [
    c for c in combined.columns if c.lower() in ["longitude", "lon", "x", "shape_x"]
]

lat_name = lat_cols[0] if lat_cols else None
lon_name = lon_cols[0] if lon_cols else None

if lat_name and lon_name:
    combined[lat_name] = pd.to_numeric(combined[lat_name], errors="coerce")
    combined[lon_name] = pd.to_numeric(combined[lon_name], errors="coerce")

    lat_med = train_features[lat_name].median()
    lon_med = train_features[lon_name].median()
    combined[lat_name] = combined[lat_name].fillna(lat_med)
    combined[lon_name] = combined[lon_name].fillna(lon_med)

    # 3D Spherical Coordinates
    lat_rad = np.radians(combined[lat_name])
    lon_rad = np.radians(combined[lon_name])
    combined["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    combined["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    combined["coord_z"] = np.sin(lat_rad)

    # Coordinate Rotations
    for angle in [30, 45, 60]:
        rad = np.radians(angle)
        combined[f"rot_lat_{angle}"] = combined[lat_name] * np.cos(rad) - combined[
            lon_name
        ] * np.sin(rad)
        combined[f"rot_lon_{angle}"] = combined[lat_name] * np.sin(rad) + combined[
            lon_name
        ] * np.cos(rad)

    # Spatial Nearest Neighbors Features
    train_coords = combined.iloc[:n_train][[lat_name, lon_name]].values
    all_coords = combined[[lat_name, lon_name]].values

    for k in [5, 15, 30]:
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn_model.fit(train_coords)
        distances, _ = nn_model.kneighbors(all_coords)
        dist_k = distances[:, 1:]
        combined[f"knn_dist_mean_{k}"] = np.mean(dist_k, axis=1)
        combined[f"knn_dist_std_{k}"] = np.std(dist_k, axis=1)
        combined[f"knn_dist_min_{k}"] = np.min(dist_k, axis=1)
        combined[f"knn_dist_max_{k}"] = np.max(dist_k, axis=1)

# 3. Text Metadata Extraction
name_cols = [
    c
    for c in combined.columns
    if any(k in c.lower() for k in ["name", "address", "city"])
]
for col in name_cols:
    if combined[col].dtype == "object" or combined[col].dtype == "string":
        str_series = combined[col].astype(str).fillna("")
        combined[f"{col}_char_len"] = str_series.apply(len)
        combined[f"{col}_word_count"] = str_series.apply(lambda x: len(x.split()))
        keywords = ["high", "elementary", "middle", "charter", "academy", "private"]
        for kw in keywords:
            combined[f"{col}_has_{kw}"] = (
                str_series.str.lower().str.contains(kw).astype(int)
            )

# 4. Numeric Transformations and Categorical Identification
num_cols = []
cat_cols = []

for c in combined.columns:
    if (
        pd.api.types.is_numeric_dtype(combined[c])
        and combined[c].nunique() > 20
        and not c.startswith("rot_")
    ):
        num_cols.append(c)
    else:
        cat_cols.append(c)

for c in num_cols:
    if combined[c].isna().sum() > 0:
        combined[f"{c}_isna"] = combined[c].isna().astype(int)
        tr_med = combined.iloc[:n_train][c].median()
        combined[c] = combined[c].fillna(tr_med)

    if (combined[c] >= 0).all() and combined[c].skew() > 1.5:
        combined[f"{c}_log1p"] = np.log1p(combined[c])

# Explicit Domain Interaction and Ratio Features
staff_student_cols = [
    c for c in num_cols if any(k in c.lower() for k in ["student", "teacher", "enroll", "staff", "count", "member", "total"])
]
ratio_cols_to_use = staff_student_cols if len(staff_student_cols) >= 2 else num_cols[:6]
for i in range(len(ratio_cols_to_use)):
    for j in range(i + 1, min(i + 4, len(ratio_cols_to_use))):
        c1, c2 = ratio_cols_to_use[i], ratio_cols_to_use[j]
        combined[f"{c1}_ratio_{c2}"] = (combined[c1] + 1.0) / (combined[c2] + 1.0)
        combined[f"{c1}_diff_{c2}"] = combined[c1] - combined[c2]

# Expanded Group Aggregation Features (mean, std, min, max across state/locale/district)
group_candidates = [
    c
    for c in cat_cols
    if any(k in c.lower() for k in ["state", "county", "type", "level", "locale", "district"])
    and combined[c].nunique() < 200
]
if not group_candidates and cat_cols:
    group_candidates = [c for c in cat_cols if combined[c].nunique() < 200][:3]

for g_col in group_candidates[:4]:
    freq = combined.iloc[:n_train][g_col].value_counts()
    combined[f"{g_col}_freq"] = combined[g_col].map(freq).fillna(0)
    for n_col in num_cols[:5]:
        tr_grp = combined.iloc[:n_train].groupby(g_col)[n_col]
        g_mean = tr_grp.mean()
        g_std = tr_grp.std().fillna(0)
        g_min = tr_grp.min()
        g_max = tr_grp.max()
        combined[f"{n_col}_mean_by_{g_col}"] = combined[g_col].map(g_mean).fillna(0)
        combined[f"{n_col}_std_by_{g_col}"] = combined[g_col].map(g_std).fillna(0)
        combined[f"{n_col}_min_by_{g_col}"] = combined[g_col].map(g_min).fillna(0)
        combined[f"{n_col}_max_by_{g_col}"] = combined[g_col].map(g_max).fillna(0)

# Cast categorical columns
for c in cat_cols:
    combined[c] = combined[c].astype(str).astype("category")

# Split processed data back
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 5. Split Strategy Definition
n_splits = 5
q_bins = pd.qcut(y, q=n_splits, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
te_cols = [c for c in cat_cols if X_train[c].nunique() > 10][:5]


# 6. Neural Network Architecture Definition
class TabularSpatialResBlock(nn.Module):
    """Residual block with LayerNorm, SiLU activation, and Dropout for Tabular Neural Networks."""

    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularSpatialResNet(nn.Module):
    """Deep Tabular Spatial Neural Network architecture featuring categorical embeddings,
    multi-scale continuous feature projection, and residual skip connections.
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list = None,
        embedding_dims: list = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities or []

        # Categorical Embeddings
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if cat_cardinalities:
            for idx, card in enumerate(cat_cardinalities):
                emb_dim = (
                    embedding_dims[idx]
                    if embedding_dims and idx < len(embedding_dims)
                    else min(50, max(4, (card + 1) // 2))
                )
                self.embeddings.append(
                    nn.Embedding(num_embeddings=card + 1, embedding_dim=emb_dim)
                )
                total_emb_dim += emb_dim

        # Feature Projection Layer
        in_dim = num_continuous + total_emb_dim
        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )

        # Residual Feature Extractor
        self.res_blocks = nn.ModuleList(
            [
                TabularSpatialResBlock(dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        # Regression Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        embeddings_list = []
        if self.embeddings and x_cat is not None:
            for idx, emb_layer in enumerate(self.embeddings):
                cat_col = x_cat[:, idx].long().clamp(min=0)
                embeddings_list.append(emb_layer(cat_col))

        if embeddings_list:
            x_cat_emb = torch.cat(embeddings_list, dim=-1)
            x_in = torch.cat([x_num, x_cat_emb], dim=-1)
        else:
            x_in = x_num

        feat = self.input_projection(x_in)
        for block in self.res_blocks:
            feat = block(feat)

        out = self.head(feat)
        return out.squeeze(-1)


class CompositeRMSELoss(nn.Module):
    """Composite loss function combining Huber Loss (outlier robustness)
    and Root Mean Squared Error (exact evaluation metric alignment).
    """

    def __init__(self, alpha: float = 0.7, delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse + 1e-8)
        huber = self.huber(y_pred, y_true)
        return self.alpha * rmse + (1.0 - self.alpha) * huber


def build_model_and_optimizer(
    num_continuous: int,
    cat_cardinalities: list = None,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """Factory function to instantiate TabularSpatialResNet, composite loss, optimizer, and scheduler."""
    model = TabularSpatialResNet(
        num_continuous=num_continuous,
        cat_cardinalities=cat_cardinalities,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    )

    criterion = CompositeRMSELoss(alpha=0.7, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-6
    )

    return model, criterion, optimizer, scheduler


# 7. Model Training & Evaluation
cat_cols_list = [
    c
    for c in X_train.columns
    if (
        str(X_train[c].dtype) == "category"
        or X_train[c].dtype == "object"
        or X_train[c].dtype == "string"
    )
]
num_cols_list = [c for c in X_train.columns if c not in cat_cols_list]

# Categorical integer encoding without data leakage
cat_enc_train = np.zeros((len(X_train), len(cat_cols_list)), dtype=np.int64)
cat_enc_test = np.zeros((len(X_test), len(cat_cols_list)), dtype=np.int64)
cat_cardinalities = []

for idx, col in enumerate(cat_cols_list):
    train_cats = X_train[col].astype(str)
    test_cats = X_test[col].astype(str)

    unique_vals = sorted(list(set(train_cats.unique()) | set(test_cats.unique())))
    val_to_idx = {val: i + 1 for i, val in enumerate(unique_vals)}

    cat_enc_train[:, idx] = train_cats.map(val_to_idx).fillna(0).values
    cat_enc_test[:, idx] = test_cats.map(val_to_idx).fillna(0).values
    cat_cardinalities.append(len(unique_vals) + 2)

# Extract continuous arrays and handle infinite/missing edge values
num_train = X_train[num_cols_list].values.astype(np.float32)
num_test = X_test[num_cols_list].values.astype(np.float32)
num_train = np.nan_to_num(num_train, nan=0.0, posinf=0.0, neginf=0.0)
num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)

oof_preds_nn = np.zeros(len(X_train))
oof_preds_lgb = np.zeros(len(X_train))
oof_preds_xgb = np.zeros(len(X_train))
oof_preds_cb = np.zeros(len(X_train))

test_preds_nn = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))

X_test_cat_tensor = torch.tensor(cat_enc_test, dtype=torch.long).to(device)

# Fold Training Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, q_bins)):
    num_tr_fold = num_train[train_idx].copy()
    num_va_fold = num_train[val_idx].copy()
    num_te_fold = num_test.copy()

    # In-fold Target Encoding to prevent target leakage
    for c in te_cols:
        s_tr = X_train[c].iloc[train_idx].astype(str)
        s_va = X_train[c].iloc[val_idx].astype(str)
        s_te = X_test[c].astype(str)
        g_mean = float(np.mean(y[train_idx]))
        counts = s_tr.value_counts()
        means = pd.Series(y[train_idx]).groupby(s_tr).mean()
        smooth = 10.0
        te_map = (counts * means + smooth * g_mean) / (counts + smooth)

        tr_te = s_tr.map(te_map).fillna(g_mean).values.astype(np.float32).reshape(-1, 1)
        va_te = s_va.map(te_map).fillna(g_mean).values.astype(np.float32).reshape(-1, 1)
        te_te = s_te.map(te_map).fillna(g_mean).values.astype(np.float32).reshape(-1, 1)

        num_tr_fold = np.hstack([num_tr_fold, tr_te])
        num_va_fold = np.hstack([num_va_fold, va_te])
        num_te_fold = np.hstack([num_te_fold, te_te])

    y_tr = y[train_idx]
    y_va = y[val_idx]

    # Tree Model Feature Formats
    X_tr_tree = np.hstack([num_tr_fold, cat_enc_train[train_idx]])
    X_va_tree = np.hstack([num_va_fold, cat_enc_train[val_idx]])
    X_te_tree = np.hstack([num_te_fold, cat_enc_test])

    # 1. LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr_tree,
        y_tr,
        eval_set=[(X_va_tree, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds_lgb[val_idx] = lgb_model.predict(X_va_tree)
    test_preds_lgb += lgb_model.predict(X_te_tree) / n_splits

    # 2. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(
        X_tr_tree,
        y_tr,
        eval_set=[(X_va_tree, y_va)],
        verbose=False,
    )
    oof_preds_xgb[val_idx] = xgb_model.predict(X_va_tree)
    test_preds_xgb += xgb_model.predict(X_te_tree) / n_splits

    # 3. CatBoost Regressor
    cb_model = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_tr_tree,
        y_tr,
        eval_set=(X_va_tree, y_va),
        verbose=False,
    )
    oof_preds_cb[val_idx] = cb_model.predict(X_va_tree)
    test_preds_cb += cb_model.predict(X_te_tree) / n_splits

    # 4. TabularSpatialResNet Neural Network
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(num_tr_fold)
    X_va_num = scaler.transform(num_va_fold)
    X_te_num = scaler.transform(num_te_fold)

    X_tr_num_t = torch.tensor(X_tr_num, dtype=torch.float32)
    X_tr_cat_t = torch.tensor(cat_enc_train[train_idx], dtype=torch.long)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32).to(device)
    X_va_cat_t = torch.tensor(cat_enc_train[val_idx], dtype=torch.long).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)

    X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_tr_num_t, X_tr_cat_t, y_tr_t)
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2
    )

    model, criterion, optimizer, scheduler = build_model_and_optimizer(
        num_continuous=X_tr_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_blocks=3,
        lr=1e-3,
        weight_decay=1e-4,
    )
    model.to(device)

    epochs = 40
    best_val_rmse = float("inf")
    os.makedirs("./working", exist_ok=True)
    best_model_path = f"./working/best_model_fold{fold}.pt"

    for epoch in range(epochs):
        model.train()
        for b_num, b_cat, b_y in train_loader:
            b_num, b_cat, b_y = (
                b_num.to(device),
                b_cat.to(device),
                b_y.to(device),
            )
            optimizer.zero_grad()
            preds = model(b_num, b_cat)
            loss = criterion(preds, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_num_t, X_va_cat_t)
            val_preds_clipped = torch.relu(val_preds)
            val_rmse = torch.sqrt(F.mse_loss(val_preds_clipped, y_va_t)).item()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)

    # Load best NN checkpoint for out-of-fold and test predictions
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with torch.no_grad():
        oof_preds_nn[val_idx] = model(X_va_num_t, X_va_cat_t).cpu().numpy()
        test_preds_nn += model(X_te_num_t, X_test_cat_tensor).cpu().numpy() / n_splits

    print(f"Fold {fold+1} complete.")

# Model Ensembling & Out-of-Fold Optimization
oof_matrix = np.column_stack([oof_preds_nn, oof_preds_lgb, oof_preds_xgb, oof_preds_cb])
test_matrix = np.column_stack([test_preds_nn, test_preds_lgb, test_preds_xgb, test_preds_cb])

def blend_loss(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_oof = oof_matrix @ weights
    blend_oof_clipped = np.maximum(0, blend_oof)
    return np.sqrt(mean_squared_error(y, blend_oof_clipped))

init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
res = minimize(blend_loss, init_weights, bounds=bounds, method="SLSQP")

best_weights = res.x / np.sum(res.x)
print(f"Optimal Model Weights (NN, LGB, XGB, CB): {best_weights}")

# Compute blended OOF and Test predictions
oof_blended = np.maximum(0, oof_matrix @ best_weights)
test_blended = np.maximum(0, test_matrix @ best_weights)

cv_score = np.sqrt(mean_squared_error(y, oof_blended))

# Save Test Predictions Submission File
os.makedirs("./submission", exist_ok=True)
sub = pd.DataFrame({"id": test_ids, "Enrollment_FT_Teachers_Ratio": test_blended})
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {cv_score}")