import os

# Prevent OpenMP runtime initialization errors (pthread_mutex_init failed)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import random
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# 1. Reproducibility & Setup
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load Raw Datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Extract Target and IDs
target_col = "RATIO_ENROLL_EMP"
if target_col in df_train.columns:
    y_train = df_train[target_col].values.astype(np.float32)
    X_train_raw = df_train.drop(columns=[target_col]).copy()
else:
    raise ValueError(f"Target column {target_col} missing in train data.")

if "id" in df_test.columns:
    test_ids = df_test["id"].values
elif "ID" in df_test.columns:
    test_ids = df_test["ID"].values
elif "OBJECTID" in df_test.columns:
    test_ids = df_test["OBJECTID"].values
else:
    test_ids = np.arange(len(df_test))

X_test_raw = df_test.copy()

# 3. Dynamic Column Detection & Feature Engineering
num_cols = X_train_raw.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()
cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()

lat_candidates = [
    c for c in num_cols if any(k in c.upper() for k in ["LATITUDE", "LAT", "Y_COORD"])
]
lon_candidates = [
    c
    for c in num_cols
    if any(k in c.upper() for k in ["LONGITUDE", "LON", "LNG", "X_COORD"])
]

lat_col = lat_candidates[0] if lat_candidates else None
lon_col = lon_candidates[0] if lon_candidates else None

X_tr_feat = pd.DataFrame(index=X_train_raw.index)
X_te_feat = pd.DataFrame(index=X_test_raw.index)

# 3a. Geospatial Features
if lat_col and lon_col:
    for df_raw, df_out in [(X_train_raw, X_tr_feat), (X_test_raw, X_te_feat)]:
        lat = df_raw[lat_col].fillna(df_raw[lat_col].median()).values
        lon = df_raw[lon_col].fillna(df_raw[lon_col].median()).values

        df_out["LAT"] = lat
        df_out["LON"] = lon

        for angle in [30, 45, 60]:
            rad = math.radians(angle)
            df_out[f"rot_lat_{angle}"] = lat * math.cos(rad) - lon * math.sin(rad)
            df_out[f"rot_lon_{angle}"] = lat * math.sin(rad) + lon * math.cos(rad)

        df_out["dist_us_center"] = np.sqrt((lat - 39.8283) ** 2 + (lon - -98.5795) ** 2)

    train_coords = X_tr_feat[["LAT", "LON"]].values
    test_coords = X_te_feat[["LAT", "LON"]].values

    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    X_tr_feat["spatial_cluster"] = kmeans.fit_predict(train_coords)
    X_te_feat["spatial_cluster"] = kmeans.predict(test_coords)

    knn = NearestNeighbors(n_neighbors=6, algorithm="kd_tree")
    knn.fit(train_coords)
    tr_distances, _ = knn.kneighbors(train_coords)
    te_distances, _ = knn.kneighbors(test_coords)

    X_tr_feat["spatial_knn_density"] = np.mean(tr_distances[:, 1:], axis=1)
    X_te_feat["spatial_knn_density"] = np.mean(te_distances[:, 1:], axis=1)

# 3b. Numerical Transformations & Ratios
for c in num_cols:
    if c in [lat_col, lon_col]:
        continue
    val_tr = X_train_raw[c].values
    val_te = X_test_raw[c].values

    X_tr_feat[c] = val_tr
    X_te_feat[c] = val_te

    if np.nanmin(val_tr) >= 0:
        X_tr_feat[f"{c}_log1p"] = np.log1p(np.maximum(0, val_tr))
        X_te_feat[f"{c}_log1p"] = np.log1p(np.maximum(0, val_te))

tot_enroll_col = next(
    (c for c in num_cols if "TOT_ENROLL" in c.upper() or "POP" in c.upper()),
    None,
)
ft_col = next((c for c in num_cols if "FT_ENROLL" in c.upper()), None)
pt_col = next((c for c in num_cols if "PT_ENROLL" in c.upper()), None)
dorm_col = next(
    (c for c in num_cols if "DORM" in c.upper() or "CAP" in c.upper()), None
)

if tot_enroll_col:
    tot_tr = np.maximum(0, X_train_raw[tot_enroll_col].fillna(0).values)
    tot_te = np.maximum(0, X_test_raw[tot_enroll_col].fillna(0).values)

    if ft_col:
        X_tr_feat["ratio_ft_tot"] = X_train_raw[ft_col].fillna(0).values / (
            tot_tr + 1.0
        )
        X_te_feat["ratio_ft_tot"] = X_test_raw[ft_col].fillna(0).values / (tot_te + 1.0)
    if pt_col:
        X_tr_feat["ratio_pt_tot"] = X_train_raw[pt_col].fillna(0).values / (
            tot_tr + 1.0
        )
        X_te_feat["ratio_pt_tot"] = X_test_raw[pt_col].fillna(0).values / (tot_te + 1.0)
    if dorm_col:
        X_tr_feat["ratio_dorm_tot"] = X_train_raw[dorm_col].fillna(0).values / (
            tot_tr + 1.0
        )
        X_te_feat["ratio_dorm_tot"] = X_test_raw[dorm_col].fillna(0).values / (
            tot_te + 1.0
        )

# Explicit Log-Difference Ratio Features between Enrollment and Employment/Staffing
enroll_cols = [c for c in num_cols if any(k in c.upper() for k in ["ENROLL", "POP"])]
emp_cols = [c for c in num_cols if any(k in c.upper() for k in ["EMP", "STAFF", "FACULTY", "WORKER"])]

for enc in enroll_cols:
    for emp in emp_cols:
        if enc != emp:
            tr_enc_val = X_train_raw[enc].values
            te_enc_val = X_test_raw[enc].values
            tr_emp_val = X_train_raw[emp].values
            te_emp_val = X_test_raw[emp].values

            X_tr_feat[f"logdiff_{enc}_{emp}"] = np.log1p(np.maximum(0, tr_enc_val)) - np.log1p(np.maximum(0, tr_emp_val))
            X_te_feat[f"logdiff_{enc}_{emp}"] = np.log1p(np.maximum(0, te_enc_val)) - np.log1p(np.maximum(0, te_emp_val))

# Group-level Numerical Summary Statistics (Group Mean and Std)
group_summary_cols = [c for c in cat_cols if any(k in c.upper() for k in ["STATE", "SECTOR"])]
key_num_cols = [c for c in num_cols if any(k in c.upper() for k in ["ENROLL", "POP", "EMP", "STAFF", "FACULTY", "LAT", "LON"])]
if not key_num_cols:
    key_num_cols = num_cols

for g_col in group_summary_cols:
    for n_col in key_num_cols:
        grp = X_train_raw.groupby(g_col)[n_col]
        mean_map = grp.mean()
        std_map = grp.std()
        global_mean = X_train_raw[n_col].mean()
        global_std = X_train_raw[n_col].std()
        X_tr_feat[f"{g_col}_{n_col}_grp_mean"] = X_train_raw[g_col].map(mean_map).fillna(global_mean).values
        X_tr_feat[f"{g_col}_{n_col}_grp_std"] = X_train_raw[g_col].map(std_map).fillna(global_std).values
        X_te_feat[f"{g_col}_{n_col}_grp_mean"] = X_test_raw[g_col].map(mean_map).fillna(global_mean).values
        X_te_feat[f"{g_col}_{n_col}_grp_std"] = X_test_raw[g_col].map(std_map).fillna(global_std).values

# 3c. Institutional Text Vectorization
text_candidates = [
    c
    for c in cat_cols
    if any(
        k in c.upper()
        for k in [
            "NAME",
            "ADDRESS",
            "CITY",
            "COUNTY",
            "NAICS",
            "DESC",
            "ALIAS",
            "WEBSITE",
        ]
    )
]
if text_candidates:
    train_text = (
        X_train_raw[text_candidates]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )
    test_text = (
        X_test_raw[text_candidates]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    svd = TruncatedSVD(n_components=20, random_state=42)

    tfidf_tr = tfidf.fit_transform(train_text)
    tfidf_te = tfidf.transform(test_text)

    svd_tr = svd.fit_transform(tfidf_tr)
    svd_te = svd.transform(tfidf_te)

    for k in range(20):
        X_tr_feat[f"text_svd_{k}"] = svd_tr[:, k]
        X_te_feat[f"text_svd_{k}"] = svd_te[:, k]

# 3d. Frequency Encoding for Categoricals
for c in cat_cols:
    freq_map = X_train_raw[c].astype(str).value_counts().to_dict()
    X_tr_feat[f"{c}_freq"] = X_train_raw[c].astype(str).map(freq_map).fillna(1).values
    X_te_feat[f"{c}_freq"] = X_test_raw[c].astype(str).map(freq_map).fillna(1).values

# Target encoding will be computed within each CV fold to prevent data leakage.
group_cols = [
    c
    for c in cat_cols
    if any(
        k in c.upper()
        for k in ["STATE", "COUNTY", "SECTOR", "LOCALE", "CONTROL", "DEGREE"]
    )
]

# 4. Imputation & Prep
feature_names = [c for c in X_tr_feat.columns if X_tr_feat[c].nunique() > 1]
X_train_unscaled = X_tr_feat[feature_names].values.astype(np.float32)
X_test_unscaled = X_te_feat[feature_names].values.astype(np.float32)

num_features = X_train_unscaled.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 5. PyTorch DCN-v2 Definition
class ExplicitCrossLayer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        xw = torch.matmul(x, self.weight)
        return x0 * (xw + self.bias) + x


class TabularDCNv2(nn.Module):

    def __init__(
        self, input_dim, hidden_dims=[256, 128, 64], num_cross_layers=3, dropout=0.2
    ):
        super().__init__()
        self.cross_layers = nn.ModuleList(
            [ExplicitCrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        deep_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            deep_layers.append(nn.Linear(in_dim, h_dim))
            deep_layers.append(nn.BatchNorm1d(h_dim))
            deep_layers.append(nn.SiLU())
            deep_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.deep_net = nn.Sequential(*deep_layers)

        comb_dim = input_dim + hidden_dims[-1]
        self.head = nn.Sequential(nn.Linear(comb_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        x0 = x
        x_cross = x
        for cross in self.cross_layers:
            x_cross = cross(x0, x_cross)
        x_deep = self.deep_net(x)
        combined = torch.cat([x_cross, x_deep], dim=1)
        return self.head(combined).squeeze(-1)


# 6. Multi-Model Cross-Validation Pipeline
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_dcn = np.zeros(len(y_train), dtype=np.float32)
oof_lgb = np.zeros(len(y_train), dtype=np.float32)
oof_xgb = np.zeros(len(y_train), dtype=np.float32)
oof_cb = np.zeros(len(y_train), dtype=np.float32)

test_dcn = np.zeros(len(X_test_unscaled), dtype=np.float32)
test_lgb = np.zeros(len(X_test_unscaled), dtype=np.float32)
test_xgb = np.zeros(len(X_test_unscaled), dtype=np.float32)
test_cb = np.zeros(len(X_test_unscaled), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_unscaled)):
    # Fold-level median imputation to prevent leakage
    X_tr_raw, X_val_raw = X_train_unscaled[tr_idx].copy(), X_train_unscaled[val_idx].copy()
    X_te_raw = X_test_unscaled.copy()
    fold_medians = np.nanmedian(X_tr_raw, axis=0)
    fold_medians = np.nan_to_num(fold_medians, nan=0.0)

    for col_idx in range(X_tr_raw.shape[1]):
        X_tr_raw[np.isnan(X_tr_raw[:, col_idx]), col_idx] = fold_medians[col_idx]
        X_val_raw[np.isnan(X_val_raw[:, col_idx]), col_idx] = fold_medians[col_idx]
        X_te_raw[np.isnan(X_te_raw[:, col_idx]), col_idx] = fold_medians[col_idx]

    # Target encoding safely inside fold to avoid validation leakage
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    global_mean_tr = np.mean(y_tr)
    X_tr_enc_list, X_val_enc_list, X_te_enc_list = [], [], []
    for gc in group_cols:
        tr_g = X_train_raw.iloc[tr_idx][gc].astype(str)
        val_g = X_train_raw.iloc[val_idx][gc].astype(str)
        te_g = X_test_raw[gc].astype(str)
        stats = pd.DataFrame({"group": tr_g, "target": y_tr}).groupby("group")["target"].agg(["count", "mean"])
        smooth = 10
        smoothed_mean = (stats["count"] * stats["mean"] + smooth * global_mean_tr) / (stats["count"] + smooth)
        enc_dict = smoothed_mean.to_dict()
        X_tr_enc_list.append(tr_g.map(enc_dict).fillna(global_mean_tr).values.reshape(-1, 1))
        X_val_enc_list.append(val_g.map(enc_dict).fillna(global_mean_tr).values.reshape(-1, 1))
        X_te_enc_list.append(te_g.map(enc_dict).fillna(global_mean_tr).values.reshape(-1, 1))

    if group_cols:
        X_tr_raw = np.hstack([X_tr_raw] + X_tr_enc_list)
        X_val_raw = np.hstack([X_val_raw] + X_val_enc_list)
        X_te_raw = np.hstack([X_te_raw] + X_te_enc_list)

    fold_scaler = RobustScaler()
    X_tr = fold_scaler.fit_transform(X_tr_raw)
    X_val = fold_scaler.transform(X_val_raw)
    X_test_scaled = fold_scaler.transform(X_te_raw)

    # --- Model A: PyTorch DCN-v2 ---
    dcn_model = TabularDCNv2(input_dim=X_tr.shape[1]).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(dcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    tr_loader = DataLoader(
        tr_ds, batch_size=128, shuffle=True, drop_last=False, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    best_dcn_path = f"./working/dcn_fold_{fold}.pt"

    for epoch in range(25):
        dcn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = dcn_model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

        dcn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = dcn_model(bx)
                val_loss += criterion(preds, by).item() * len(by)
        val_loss /= len(y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(dcn_model.state_dict(), best_dcn_path)

    dcn_model.load_state_dict(torch.load(best_dcn_path))
    dcn_model.eval()

    with torch.no_grad():
        val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        oof_dcn[val_idx] = dcn_model(val_x_t).cpu().numpy()
        test_x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_dcn += dcn_model(test_x_t).cpu().numpy() / n_splits

    # --- Model B: LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test_scaled) / n_splits

    # --- Model C: XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test_scaled) / n_splits

    # --- Model D: CatBoost ---
    cb_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        iterations=800,
        learning_rate=0.04,
        depth=6,
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42 + fold,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cb[val_idx] = cb_model.predict(X_val)
    test_cb += cb_model.predict(X_test_scaled) / n_splits

    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    print(f"Fold {fold+1}/{n_splits} Complete. LightGBM Val RMSE: {fold_lgb_rmse:.4f}")

# 7. Meta-Learner Stacking (Non-Negative Ridge Regression)
OOF_matrix = np.column_stack([oof_dcn, oof_lgb, oof_xgb, oof_cb])
TEST_matrix = np.column_stack([test_dcn, test_lgb, test_xgb, test_cb])

meta_model = Ridge(alpha=10.0, positive=True, fit_intercept=True)
meta_model.fit(OOF_matrix, y_train)

oof_preds = meta_model.predict(OOF_matrix)
test_preds = meta_model.predict(TEST_matrix)

# Post-process target domain boundaries
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

final_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))

# 8. Export Submission Output
sub_df = pd.DataFrame({"id": test_ids, "RATIO_ENROLL_EMP": test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse}")