import os
import re
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==============================================================================
# 1. Load Raw Data
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "CAPACITY"
train_y = train_df[target_col].values.astype(np.float32)
train_df_features = train_df.drop(columns=[target_col], errors="ignore")
test_df_features = test_df.copy()


# ==============================================================================
# 2. Feature Engineering Routines
# ==============================================================================
def extract_spatial_features(df):
    """Generate spatial metrics, coordinate rotations, and distances."""
    feats = pd.DataFrame(index=df.index)

    lat_col = next(
        (c for c in df.columns if c.upper() in ["LATITUDE", "LAT", "Y"]), None
    )
    lon_col = next(
        (c for c in df.columns if c.upper() in ["LONGITUDE", "LON", "X"]), None
    )

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce").fillna(39.8283)
        lon = pd.to_numeric(df[lon_col], errors="coerce").fillna(-98.5795)

        feats["lat"] = lat
        feats["lon"] = lon

        for angle in [15, 30, 45, 60, 75]:
            rad = np.radians(angle)
            feats[f"rot_{angle}_x"] = lon * np.cos(rad) - lat * np.sin(rad)
            feats[f"rot_{angle}_y"] = lon * np.sin(rad) + lat * np.cos(rad)

        feats["dist_us_center"] = np.sqrt((lat - 39.8283) ** 2 + (lon - -98.5795) ** 2)
        feats["dist_east_coast"] = np.sqrt((lat - 40.7128) ** 2 + (lon - -74.0060) ** 2)
        feats["dist_west_coast"] = np.sqrt(
            (lat - 34.0522) ** 2 + (lon - -118.2437) ** 2
        )

        feats["polar_r"] = np.sqrt(lat**2 + lon**2)
        feats["polar_theta"] = np.arctan2(lat, lon)

    return feats


def extract_text_features(df):
    """Extract domain keyword indicators and summary length statistics."""
    text_cols = [
        c
        for c in df.columns
        if c.upper()
        in [
            "NAME",
            "ADDRESS",
            "CITY",
            "COUNTY",
            "SOURCE",
            "SECURELVL",
            "TYPE",
            "STATUS",
        ]
    ]

    combined_text = df[text_cols].astype(str).fillna("").agg(" ".join, axis=1)

    keywords = [
        "CORRECTIONAL",
        "PENITENTIARY",
        "JAIL",
        "PRISON",
        "YOUTH",
        "DETENTION",
        "COMMUNITY",
        "FEDERAL",
        "STATE",
        "COUNTY",
        "MAXIMUM",
        "MEDIUM",
        "MINIMUM",
        "CLOSE",
        "WORK",
        "CAMP",
        "REENTRY",
        "TREATMENT",
        "MEDICAL",
    ]

    keyword_df = pd.DataFrame(index=df.index)
    text_upper = combined_text.str.upper()
    for kw in keywords:
        keyword_df[f"kw_{kw}"] = text_upper.str.contains(kw).astype(int)

    keyword_df["text_char_len"] = combined_text.str.len()
    keyword_df["text_word_count"] = combined_text.str.split().str.len()

    return combined_text, keyword_df


def extract_temporal_features(df):
    """Extract temporal calendar features from date and time fields."""
    feats = pd.DataFrame(index=df.index)
    date_cols = [
        c
        for c in df.columns
        if "DATE" in c.upper() or "YEAR" in c.upper() or "TIME" in c.upper()
    ]

    for c in date_cols:
        parsed_dt = pd.to_datetime(df[c], errors="coerce")
        if parsed_dt.notna().sum() > 0:
            feats[f"{c}_year"] = parsed_dt.dt.year.fillna(-1)
            feats[f"{c}_month"] = parsed_dt.dt.month.fillna(-1)
            feats[f"{c}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(-1)

    return feats


# Feature Extraction Execution
# 1. Preserve all original raw numerical features from input dataframes
num_cols = [
    c for c in train_df_features.columns
    if np.issubdtype(train_df_features[c].dtype, np.number)
]
train_raw_num = train_df_features[num_cols].copy()
test_raw_num = test_df_features[num_cols].copy()

# 2. Extract spatial and temporal features
train_spatial = extract_spatial_features(train_df_features)
test_spatial = extract_spatial_features(test_df_features)

train_temporal = extract_temporal_features(train_df_features)
test_temporal = extract_temporal_features(test_df_features)

all_temp_cols = list(set(train_temporal.columns).union(set(test_temporal.columns)))
for col in all_temp_cols:
    if col not in train_temporal.columns:
        train_temporal[col] = -1
    if col not in test_temporal.columns:
        test_temporal[col] = -1

# 3. Label encode and frequency encode categorical features
for c1, c2 in [("STATE", "TYPE"), ("STATE", "SECURELVL"), ("TYPE", "SECURELVL")]:
    if c1 in train_df_features.columns and c2 in train_df_features.columns:
        col_name = f"{c1}_{c2}"
        train_df_features[col_name] = train_df_features[c1].astype(str) + "_" + train_df_features[c2].astype(str)
        test_df_features[col_name] = test_df_features[c1].astype(str) + "_" + test_df_features[c2].astype(str)

cat_cols = [
    c
    for c in train_df_features.columns
    if train_df_features[c].dtype == "object"
    or train_df_features[c].dtype.name == "category"
]

train_cat = pd.DataFrame(index=train_df_features.index)
test_cat = pd.DataFrame(index=test_df_features.index)

for col in cat_cols:
    tr_str = train_df_features[col].astype(str)
    te_str = test_df_features[col].astype(str)

    # Label Encoding
    le = LabelEncoder()
    unique_vals = list(set(tr_str.unique()).union(set(te_str.unique())))
    le.fit(unique_vals)
    train_cat[f"{col}_le"] = le.transform(tr_str)
    test_cat[f"{col}_le"] = le.transform(te_str)

    # Frequency Encoding
    freq = tr_str.value_counts(normalize=True).to_dict()
    train_cat[f"{col}_freq"] = tr_str.map(freq).fillna(0)
    test_cat[f"{col}_freq"] = te_str.map(freq).fillna(0)

# 4. Extract text keyword features
train_text_str, train_kw = extract_text_features(train_df_features)
test_text_str, test_kw = extract_text_features(test_df_features)

# Concatenate base features while deduplicating column names
train_engineered_base = pd.concat(
    [train_raw_num, train_spatial, train_temporal, train_cat, train_kw], axis=1
)
test_engineered_base = pd.concat(
    [test_raw_num, test_spatial, test_temporal, test_cat, test_kw], axis=1
)

train_engineered_base = train_engineered_base.loc[
    :, ~train_engineered_base.columns.duplicated()
]
test_engineered_base = test_engineered_base.loc[
    :, ~test_engineered_base.columns.duplicated()
]

common_cols = sorted(
    list(
        set(train_engineered_base.columns).intersection(
            set(test_engineered_base.columns)
        )
    )
)
train_engineered_base = train_engineered_base[common_cols]
test_engineered_base = test_engineered_base[common_cols]

n_train = len(train_df_features)
n_test = len(test_df_features)


# ==============================================================================
# 3. Cross-Validation Setup
# ==============================================================================
n_splits = 5
num_bins = min(10, len(np.unique(train_y)))
y_binned = pd.qcut(train_y, q=num_bins, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = list(skf.split(train_engineered_base, y_binned))


# ==============================================================================
# 4. Neural Architecture Definition
# ==============================================================================
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# ==============================================================================
# 5. Cross-Validation Training & Inference Loop
# ==============================================================================
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_nn = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    # Extract TF-IDF + SVD strictly on training split
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    tr_tfidf = tfidf.fit_transform(train_text_str.iloc[train_idx])
    va_tfidf = tfidf.transform(train_text_str.iloc[val_idx])
    te_tfidf = tfidf.transform(test_text_str)

    n_svd_comp = min(16, tr_tfidf.shape[1] - 1) if tr_tfidf.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_svd_comp, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    va_svd = svd.transform(va_tfidf)
    te_svd = svd.transform(te_tfidf)

    svd_cols = [f"svd_{i}" for i in range(n_svd_comp)]
    tr_svd_df = pd.DataFrame(tr_svd, columns=svd_cols, index=train_idx)
    va_svd_df = pd.DataFrame(va_svd, columns=svd_cols, index=val_idx)
    te_svd_df = pd.DataFrame(te_svd, columns=svd_cols, index=test_df_features.index)

    y_tr, y_va = train_y[train_idx], train_y[val_idx]

    # Out-of-fold Target Encoding for categorical variables
    te_cols = [c for c in ["STATE", "COUNTY", "TYPE", "SECURELVL", "STATUS"] if c in train_df_features.columns]
    if not te_cols:
        te_cols = cat_cols

    tr_te_df = pd.DataFrame(index=train_idx)
    va_te_df = pd.DataFrame(index=val_idx)
    te_te_df = pd.DataFrame(index=test_df_features.index)

    global_mean = float(np.mean(y_tr))
    global_std = float(np.std(y_tr))
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42 + fold_idx)

    for c in te_cols:
        tr_vals = train_df_features.iloc[train_idx][c].astype(str).values
        va_vals = train_df_features.iloc[val_idx][c].astype(str).values
        te_vals = test_df_features[c].astype(str).values

        # Group statistics on full training fold for val and test mapping
        df_tr_col = pd.DataFrame({"cat": tr_vals, "y": y_tr})
        stats = df_tr_col.groupby("cat")["y"].agg(["mean", "std"])

        va_te_df[f"{c}_te_mean"] = (
            pd.Series(va_vals).map(stats["mean"]).fillna(global_mean).values
        )
        va_te_df[f"{c}_te_std"] = (
            pd.Series(va_vals).map(stats["std"]).fillna(global_std).fillna(0).values
        )

        te_te_df[f"{c}_te_mean"] = (
            pd.Series(te_vals).map(stats["mean"]).fillna(global_mean).values
        )
        te_te_df[f"{c}_te_std"] = (
            pd.Series(te_vals).map(stats["std"]).fillna(global_std).fillna(0).values
        )

        # Inner K-fold target encoding for train_idx to prevent leakage
        tr_mean_arr = np.full(len(train_idx), global_mean)
        tr_std_arr = np.full(len(train_idx), global_std)

        for in_tr_i, in_va_i in inner_kf.split(train_idx):
            in_tr_cats = tr_vals[in_tr_i]
            in_tr_y = y_tr[in_tr_i]
            in_va_cats = tr_vals[in_va_i]

            in_global_mean = float(np.mean(in_tr_y))
            in_global_std = float(np.std(in_tr_y))

            in_df = pd.DataFrame({"cat": in_tr_cats, "y": in_tr_y})
            in_stats = in_df.groupby("cat")["y"].agg(["mean", "std"])

            tr_mean_arr[in_va_i] = (
                pd.Series(in_va_cats)
                .map(in_stats["mean"])
                .fillna(in_global_mean)
                .values
            )
            tr_std_arr[in_va_i] = (
                pd.Series(in_va_cats)
                .map(in_stats["std"])
                .fillna(in_global_std)
                .fillna(0)
                .values
            )

        tr_te_df[f"{c}_te_mean"] = tr_mean_arr
        tr_te_df[f"{c}_te_std"] = tr_std_arr

    tr_eng = pd.concat([train_engineered_base.iloc[train_idx], tr_svd_df, tr_te_df], axis=1)
    va_eng = pd.concat([train_engineered_base.iloc[val_idx], va_svd_df, va_te_df], axis=1)
    te_eng = pd.concat([test_engineered_base, te_svd_df, te_te_df], axis=1)

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(tr_eng)
    X_va = imputer.transform(va_eng)
    n_features = X_tr.shape[1]
    X_test_fold = imputer.transform(te_eng)

    # --- A. LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold_idx,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = np.clip(lgb_model.predict(X_va), 0, None)
    test_lgb += np.clip(lgb_model.predict(X_test_fold), 0, None) / n_splits

    # --- B. XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42 + fold_idx,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = np.clip(xgb_model.predict(X_va), 0, None)
    test_xgb += np.clip(xgb_model.predict(X_test_fold), 0, None) / n_splits

    # --- C. CatBoost ---
    cat_model = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        early_stopping_rounds=50,
        random_seed=42 + fold_idx,
        verbose=0,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = np.clip(cat_model.predict(X_va), 0, None)
    test_cat += np.clip(cat_model.predict(X_test_fold), 0, None) / n_splits

    # --- D. PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_test_fold)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    nn_model = TabularResNet(
        input_dim=n_features, hidden_dim=128, num_blocks=2, dropout_rate=0.2
    ).to(device)
    criterion = nn.HuberLoss(delta=100.0)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(40):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(by)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_nn_weights)
    nn_model.to(device)
    nn_model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        oof_nn[val_idx] = np.clip(nn_model(va_tensor).cpu().numpy(), 0, None)

        te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        test_nn += np.clip(nn_model(te_tensor).cpu().numpy(), 0, None) / n_splits


# ==============================================================================
# 6. Ensemble Weight Optimization & Final Predictions
# ==============================================================================
OOF_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
TEST_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def objective(weights):
    weights = np.array(weights)
    pred = np.clip(OOF_matrix @ weights, 0, None)
    return np.sqrt(np.mean((train_y - pred) ** 2))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

res = minimize(
    objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
opt_weights = res.x

final_oof_preds = np.clip(OOF_matrix @ opt_weights, 0, None)
final_rmse = float(np.sqrt(np.mean((train_y - final_oof_preds) ** 2)))

final_test_preds = np.clip(TEST_matrix @ opt_weights, 0, None)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "CAPACITY": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse}")