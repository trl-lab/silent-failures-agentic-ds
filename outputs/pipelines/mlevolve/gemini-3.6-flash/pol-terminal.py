import os
import random
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Reproducibility & Setup
# ---------------------------------------------------------
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Dynamic Feature Discovery
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "CAPACITY"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' missing from train data"

y_raw_full = np.maximum(0, train_df[target_col].values)
y_log_full = np.log1p(y_raw_full)

max_y_cap = float(max(np.max(y_raw_full) * 5.0, 1e6))
max_log_cap = float(np.log1p(max_y_cap))
max_sqrt_cap = float(np.sqrt(max_y_cap))

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

feature_cols = [c for c in train_df.columns if c != target_col]
train_features = train_df[feature_cols].copy()
test_features = test_df[feature_cols].copy()

lat_cols = [
    c for c in feature_cols if re.search(r"latitude|lat|y_coord", c, re.IGNORECASE)
]
lon_cols = [
    c for c in feature_cols if re.search(r"longitude|lon|x_coord", c, re.IGNORECASE)
]
lat_col = lat_cols[0] if len(lat_cols) > 0 else None
lon_col = lon_cols[0] if len(lon_cols) > 0 else None

text_and_cat_cols = []
num_cols = []

for c in feature_cols:
    if train_features[c].dtype == "object" or str(train_features[c].dtype).startswith(
        "string"
    ):
        text_and_cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train_features[c]):
        if c not in [lat_col, lon_col]:
            num_cols.append(c)

# Create Holdout Train / Validation Split (80 / 20)
bins = pd.qcut(y_log_full, q=10, labels=False, duplicates="drop")
tr_idx, va_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=SEED, stratify=bins
)

train_sub = train_features.iloc[tr_idx].copy()
val_sub = train_features.iloc[va_idx].copy()
y_train_log = y_log_full[tr_idx]
y_val_log = y_log_full[va_idx]
y_tr_raw = y_raw_full[tr_idx]
y_val_raw = y_raw_full[va_idx]

# ---------------------------------------------------------
# 3. Feature Engineering Function
# ---------------------------------------------------------


def process_features(df_train, df_val, df_test, y_tr_l):
    X_tr_list, X_va_list, X_te_list = [], [], []

    # A) Spatial 3D Cartesian Transformations & Clustering
    if lat_col and lon_col:
        for df in [df_train, df_val, df_test]:
            df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce").fillna(
                df_train[lat_col].median()
            )
            df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce").fillna(
                df_train[lon_col].median()
            )

        for df, x_list in zip(
            [df_train, df_val, df_test], [X_tr_list, X_va_list, X_te_list]
        ):
            lat_r = np.radians(df[lat_col].values)
            lon_r = np.radians(df[lon_col].values)
            df_spatial = pd.DataFrame(index=df.index)
            df_spatial["spatial_x"] = np.cos(lat_r) * np.cos(lon_r)
            df_spatial["spatial_y"] = np.cos(lat_r) * np.sin(lon_r)
            df_spatial["spatial_z"] = np.sin(lat_r)
            x_list.append(df_spatial)

        coords_train = df_train[[lat_col, lon_col]].values
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=SEED, batch_size=256).fit(
            coords_train
        )

        for df, x_list in zip(
            [df_train, df_val, df_test], [X_tr_list, X_va_list, X_te_list]
        ):
            coords = df[[lat_col, lon_col]].values
            dists = kmeans.transform(coords)
            dist_df = pd.DataFrame(
                dists,
                columns=[f"dist_cluster_{i}" for i in range(10)],
                index=df.index,
            )
            dist_df["min_cluster_dist"] = dists.min(axis=1)
            x_list.append(dist_df)

    # B) Text Concatenation, TF-IDF SVD & Connectivity Flags
    if len(text_and_cat_cols) > 0:

        def get_text(df):
            return (
                df[text_and_cat_cols]
                .astype(str)
                .fillna("")
                .agg(" ".join, axis=1)
                .str.upper()
            )

        t_tr = get_text(df_train)
        t_va = get_text(df_val)
        t_te = get_text(df_test)

        tfidf = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words="english"
        )
        svd = TruncatedSVD(n_components=16, random_state=SEED)

        svd_tr = svd.fit_transform(tfidf.fit_transform(t_tr))
        svd_va = svd.transform(tfidf.transform(t_va))
        svd_te = svd.transform(tfidf.transform(t_te))

        X_tr_list.append(
            pd.DataFrame(
                svd_tr,
                columns=[f"text_svd_{i}" for i in range(16)],
                index=df_train.index,
            )
        )
        X_va_list.append(
            pd.DataFrame(
                svd_va,
                columns=[f"text_svd_{i}" for i in range(16)],
                index=df_val.index,
            )
        )
        X_te_list.append(
            pd.DataFrame(
                svd_te,
                columns=[f"text_svd_{i}" for i in range(16)],
                index=df_test.index,
            )
        )

        for kw in ["PIPELINE", "MARINE", "VESSEL", "RAIL", "TRUCK", "BARGE"]:
            X_tr_list.append(
                pd.DataFrame(
                    {f"kw_{kw}": t_tr.str.contains(kw).astype(int)},
                    index=df_train.index,
                )
            )
            X_va_list.append(
                pd.DataFrame(
                    {f"kw_{kw}": t_va.str.contains(kw).astype(int)},
                    index=df_val.index,
                )
            )
            X_te_list.append(
                pd.DataFrame(
                    {f"kw_{kw}": t_te.str.contains(kw).astype(int)},
                    index=df_test.index,
                )
            )

    # C) Frequency Encoding for Categoricals (Leak-Free)
    cat_target_cols = [
        c
        for c in text_and_cat_cols
        if df_train[c].nunique() > 1 and df_train[c].nunique() < 500
    ]
    for col in cat_target_cols[:5]:
        freqs = df_train[col].value_counts(normalize=True)
        tr_fe = df_train[col].map(freqs).fillna(0)
        val_fe = df_val[col].map(freqs).fillna(0)
        test_fe = df_test[col].map(freqs).fillna(0)

        X_tr_list.append(
            pd.DataFrame({f"{col}_freq": tr_fe.values}, index=df_train.index)
        )
        X_va_list.append(pd.DataFrame({f"{col}_freq": val_fe.values}, index=df_val.index))
        X_te_list.append(
            pd.DataFrame({f"{col}_freq": test_fe.values}, index=df_test.index)
        )

    # D) Standardized Numerical Features
    if len(num_cols) > 0:
        scaler = StandardScaler()
        num_tr = scaler.fit_transform(df_train[num_cols].fillna(0))
        num_va = scaler.transform(df_val[num_cols].fillna(0))
        num_te = scaler.transform(df_test[num_cols].fillna(0))

        X_tr_list.append(pd.DataFrame(num_tr, columns=num_cols, index=df_train.index))
        X_va_list.append(pd.DataFrame(num_va, columns=num_cols, index=df_val.index))
        X_te_list.append(pd.DataFrame(num_te, columns=num_cols, index=df_test.index))

    X_tr = pd.concat(X_tr_list, axis=1).fillna(0)
    X_va = pd.concat(X_va_list, axis=1).fillna(0)
    X_te = pd.concat(X_te_list, axis=1).fillna(0)

    return X_tr, X_va, X_te


X_train, X_val, X_test = process_features(
    train_sub, val_sub, test_features, y_train_log
)

clean_cols = [re.sub(r"[\[\]<>{}\",:]", "_", col) for col in X_train.columns]
X_train.columns = clean_cols
X_val.columns = clean_cols
X_test.columns = clean_cols

y_tr_sqrt = np.sqrt(y_tr_raw)


# ---------------------------------------------------------
# 4. Neural Architecture Definition: DenseTabularNet
# ---------------------------------------------------------
class DenseTabularBlock(nn.Module):

    def __init__(self, in_features, growth_rate, dropout_rate=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.act = nn.Mish()
        self.fc = nn.Linear(in_features, growth_rate)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.drop(self.fc(self.act(self.bn(x))))
        return torch.cat([x, out], dim=1)


class PyTorchDenseTabularNet(nn.Module):

    def __init__(
        self, input_dim, hidden_dim=64, growth_rate=32, num_layers=3, dropout=0.1
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        curr_dim = hidden_dim
        layers = []
        for _ in range(num_layers):
            layers.append(DenseTabularBlock(curr_dim, growth_rate, dropout))
            curr_dim += growth_rate
        self.block = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.BatchNorm1d(curr_dim),
            nn.Mish(),
            nn.Linear(curr_dim, 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------
# 5. Multi-Model 5-Fold Cross Validation Stacking Setup
# ---------------------------------------------------------
N_MODELS = 5
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds_raw = np.zeros((len(X_train), N_MODELS))
val_preds_raw = np.zeros((len(X_val), N_MODELS))
test_preds_raw = np.zeros((len(X_test), N_MODELS))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (trn_idx, val_fold_idx) in enumerate(kf.split(X_train)):
    X_tr_f, y_tr_log_f = X_train.iloc[trn_idx], y_train_log[trn_idx]
    X_va_f, y_va_log_f = X_train.iloc[val_fold_idx], y_train_log[val_fold_idx]

    y_tr_sqrt_f = y_tr_sqrt[trn_idx]
    y_va_sqrt_f = y_tr_sqrt[val_fold_idx]

    # Model 0: LightGBM (Log target, MSE loss)
    m0 = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        verbose=-1,
    )
    m0.fit(
        X_tr_f,
        y_tr_log_f,
        eval_set=[(X_va_f, y_va_log_f)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    oof_preds_raw[val_fold_idx, 0] = np.expm1(np.clip(m0.predict(X_va_f), 0, max_log_cap))
    val_preds_raw[:, 0] += np.expm1(np.clip(m0.predict(X_val), 0, max_log_cap)) / N_SPLITS
    test_preds_raw[:, 0] += np.expm1(np.clip(m0.predict(X_test), 0, max_log_cap)) / N_SPLITS

    # Model 1: LightGBM (Log target, Huber loss)
    m1 = lgb.LGBMRegressor(
        objective="huber",
        alpha=1.0,
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=25,
        subsample=0.85,
        colsample_bytree=0.75,
        random_state=SEED + fold + 10,
        verbose=-1,
    )
    m1.fit(
        X_tr_f,
        y_tr_log_f,
        eval_set=[(X_va_f, y_va_log_f)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    oof_preds_raw[val_fold_idx, 1] = np.expm1(np.clip(m1.predict(X_va_f), 0, max_log_cap))
    val_preds_raw[:, 1] += np.expm1(np.clip(m1.predict(X_val), 0, max_log_cap)) / N_SPLITS
    test_preds_raw[:, 1] += np.expm1(np.clip(m1.predict(X_test), 0, max_log_cap)) / N_SPLITS

    # Model 2: CatBoost Regressor (Log target)
    m2 = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        random_seed=SEED + fold,
        verbose=0,
    )
    m2.fit(X_tr_f, y_tr_log_f, eval_set=(X_va_f, y_va_log_f), early_stopping_rounds=30)

    oof_preds_raw[val_fold_idx, 2] = np.expm1(np.clip(m2.predict(X_va_f), 0, max_log_cap))
    val_preds_raw[:, 2] += np.expm1(np.clip(m2.predict(X_val), 0, max_log_cap)) / N_SPLITS
    test_preds_raw[:, 2] += np.expm1(np.clip(m2.predict(X_test), 0, max_log_cap)) / N_SPLITS

    # Model 3: XGBoost (Square-root target scale)
    m3 = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        tree_method="hist",
        early_stopping_rounds=30,
    )
    m3.fit(X_tr_f, y_tr_sqrt_f, eval_set=[(X_va_f, y_va_sqrt_f)], verbose=False)

    oof_preds_raw[val_fold_idx, 3] = np.clip(m3.predict(X_va_f), 0, max_sqrt_cap) ** 2
    val_preds_raw[:, 3] += (np.clip(m3.predict(X_val), 0, max_sqrt_cap) ** 2) / N_SPLITS
    test_preds_raw[:, 3] += (np.clip(m3.predict(X_test), 0, max_sqrt_cap) ** 2) / N_SPLITS

    # Model 4: PyTorch DenseTabularNet (Log target, Smooth L1 Loss)
    tr_t = torch.tensor(X_tr_f.values, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_log_f, dtype=torch.float32)
    va_t = torch.tensor(X_va_f.values, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_log_f, dtype=torch.float32)

    val_hold_t = torch.tensor(X_val.values, dtype=torch.float32)
    test_hold_t = torch.tensor(X_test.values, dtype=torch.float32)

    ds = TensorDataset(tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    nn_m = PyTorchDenseTabularNet(input_dim=X_train.shape[1]).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(nn_m.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_v_loss = float("inf")
    best_nn_w = None

    for epoch in range(40):
        nn_m.train()
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_m(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_m.eval()
        with torch.no_grad():
            v_out = nn_m(va_t.to(device))
            v_l = criterion(v_out, y_va_t.to(device)).item()

        if v_l < best_v_loss:
            best_v_loss = v_l
            best_nn_w = nn_m.state_dict().copy()

    if best_nn_w is not None:
        nn_m.load_state_dict(best_nn_w)

    nn_m.eval()
    with torch.no_grad():
        oof_nn_log = nn_m(va_t.to(device)).cpu().numpy()
        val_nn_log = nn_m(val_hold_t.to(device)).cpu().numpy()
        test_nn_log = nn_m(test_hold_t.to(device)).cpu().numpy()

    oof_preds_raw[val_fold_idx, 4] = np.expm1(np.clip(oof_nn_log, 0, max_log_cap))
    val_preds_raw[:, 4] += np.expm1(np.clip(val_nn_log, 0, max_log_cap)) / N_SPLITS
    test_preds_raw[:, 4] += np.expm1(np.clip(test_nn_log, 0, max_log_cap)) / N_SPLITS

# ---------------------------------------------------------
# 6. SLSQP Stacking Optimization
# ---------------------------------------------------------


def raw_rmse_cost(weights):
    w = np.array(weights)
    pred = np.dot(oof_preds_raw, w)
    return np.sqrt(mean_squared_error(y_tr_raw, pred))


init_w = [1.0 / N_MODELS] * N_MODELS
bounds = [(0, 1)] * N_MODELS
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

opt = minimize(
    raw_rmse_cost, init_w, method="SLSQP", bounds=bounds, constraints=constraints
)
meta_weights = opt.x

# ---------------------------------------------------------
# 7. Final Predictions & Evaluation
# ---------------------------------------------------------
final_val_pred = np.maximum(0, np.dot(val_preds_raw, meta_weights))
final_test_pred = np.maximum(0, np.dot(test_preds_raw, meta_weights))

final_val_pred = np.nan_to_num(final_val_pred, nan=0.0, posinf=0.0, neginf=0.0)
final_test_pred = np.nan_to_num(final_test_pred, nan=0.0, posinf=0.0, neginf=0.0)

val_rmse = np.sqrt(mean_squared_error(y_val_raw, final_val_pred))

# Write official submission file
sub = pd.DataFrame({"id": test_ids, "CAPACITY": final_test_pred})
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_rmse}")