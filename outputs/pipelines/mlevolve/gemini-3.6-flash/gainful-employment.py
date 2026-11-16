import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Set random seeds for exact reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create required directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Raw Data & Cleaning
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Debt-to-Earnings Annual Rate"

# Clean target variable if needed
if target_col in train_df.columns:
    if train_df[target_col].dtype == object:
        train_df[target_col] = (
            train_df[target_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Preserve test index
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

train_df["is_train"] = 1
test_df["is_train"] = 0
full_df = pd.concat([train_df, test_df], ignore_index=True)

# Standardize privacy suppressed values and clean numerical strings
suppression_tokens = {
    "PrivacySuppressed",
    "Privacy Suppressed",
    "NULL",
    "N/A",
    "NA",
    "None",
    "nan",
    "PS",
    "Low Turnover",
    "low turnover",
    "Suppressed",
    "",
}

feature_cols = [c for c in full_df.columns if c not in [target_col, "is_train", "id"]]

for col in feature_cols:
    if full_df[col].dtype == object:
        full_df[col] = full_df[col].astype(str).str.strip()

        full_df[col] = full_df[col].apply(
            lambda x: (
                np.nan
                if x in suppression_tokens or x.lower() in suppression_tokens
                else x
            )
        )

        cleaned_col = (
            full_df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        converted = pd.to_numeric(cleaned_col, errors="coerce")
        non_null_orig = full_df[col].dropna().count()
        non_null_conv = converted.dropna().count()
        if non_null_orig > 0 and (non_null_conv / float(non_null_orig)) > 0.5:
            full_df[col] = converted

numeric_cols = []
categorical_cols = []
text_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(full_df[col]):
        numeric_cols.append(col)
    else:
        unique_cnt = full_df[col].nunique()
        if unique_cnt > 100 and full_df[col].astype(str).str.len().mean() > 20:
            text_cols.append(col)
        else:
            categorical_cols.append(col)

# ---------------------------------------------------------
# 2. Domain Feature Engineering
# ---------------------------------------------------------
# A. CIP family prefixes (2-digit and 4-digit abstractions)
cip_cols = [col for col in feature_cols if "cip" in col.lower()]
for col in cip_cols:
    s_clean = (
        full_df[col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.strip()
    )
    digits = s_clean.str.replace(r"\D", "", regex=True)

    cip2_col = f"{col}_cip2"
    cip4_col = f"{col}_cip4"

    full_df[cip2_col] = digits.str[:2]
    full_df[cip4_col] = digits.str[:4]

    full_df[cip2_col] = full_df[cip2_col].replace("", np.nan)
    full_df[cip4_col] = full_df[cip4_col].replace("", np.nan)

    if cip2_col not in categorical_cols:
        categorical_cols.append(cip2_col)
    if cip4_col not in categorical_cols:
        categorical_cols.append(cip4_col)

# B. Missing value flags & Log scale transformations
for col in numeric_cols:
    missing_flag_col = f"{col}_is_missing"
    full_df[missing_flag_col] = full_df[col].isna().astype(int)

    col_min = full_df[col].min()
    if col_min is not np.nan and col_min >= 0:
        if full_df[col].skew() > 1.5:
            full_df[f"{col}_log1p"] = np.log1p(full_df[col].fillna(0))

# C. Interactions & Explicit Domain Debt-to-Earnings Ratios
debt_keyword_cols = [c for c in numeric_cols if any(k in c.lower() for k in ["debt", "loan", "borrow"])]
earn_keyword_cols = [c for c in numeric_cols if any(k in c.lower() for k in ["earn", "income", "pay", "salary", "wages"])]

for d_col in debt_keyword_cols:
    for e_col in earn_keyword_cols:
        full_df[f"{d_col}_ratio_{e_col}"] = full_df[d_col] / (full_df[e_col].abs() + 1e-5)
        full_df[f"{d_col}_mult_{e_col}"] = full_df[d_col] * full_df[e_col]
        full_df[f"{d_col}_diff_{e_col}"] = full_df[d_col] - full_df[e_col]

# D. Group Statistics Aggregations (mean, median, std, min, max) grouped by CIP & Credential level
cip_and_cred_cols = [
    col for col in categorical_cols
    if any(k in col.lower() for k in ["cip", "cred", "level", "degree", "type", "tier"])
]
group_cols = list(dict.fromkeys(cip_and_cred_cols + categorical_cols[:5]))

train_mask = full_df["is_train"] == 1
for group_col in group_cols:
    if group_col in full_df.columns and full_df[group_col].nunique() > 1:
        for num_col in numeric_cols:
            grp = full_df[train_mask].groupby(group_col)[num_col]
            mean_map = grp.mean()
            median_map = grp.median()
            std_map = grp.std()
            min_map = grp.min()
            max_map = grp.max()
            full_df[f"{group_col}_{num_col}_grp_mean"] = full_df[group_col].map(mean_map)
            full_df[f"{group_col}_{num_col}_grp_median"] = full_df[group_col].map(median_map)
            full_df[f"{group_col}_{num_col}_grp_std"] = full_df[group_col].map(std_map)
            full_df[f"{group_col}_{num_col}_grp_min"] = full_df[group_col].map(min_map)
            full_df[f"{group_col}_{num_col}_grp_max"] = full_df[group_col].map(max_map)

# D. Text SVD Embeddings
for tcol in text_cols:
    train_text = full_df.loc[full_df["is_train"] == 1, tcol].fillna("").astype(str)
    all_text = full_df[tcol].fillna("").astype(str)

    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_train = tfidf.fit_transform(train_text)
    tfidf_mat = tfidf.transform(all_text)

    n_components = min(10, tfidf_train.shape[1] - 1)
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(tfidf_train)
        svd_feats = svd.transform(tfidf_mat)
        for k in range(n_components):
            full_df[f"{tcol}_svd_{k}"] = svd_feats[:, k]

# E. Categorical Frequency & Type Casting
for ccol in categorical_cols:
    freq_map = full_df[ccol].value_counts().to_dict()
    full_df[f"{ccol}_freq"] = full_df[ccol].map(freq_map).fillna(0)

for col in full_df.columns:
    if full_df[col].dtype == object or full_df[col].dtype.name == "object":
        full_df[col] = full_df[col].astype("category")

train_processed = (
    full_df[full_df["is_train"] == 1].drop(columns=["is_train"]).reset_index(drop=True)
)
test_processed = (
    full_df[full_df["is_train"] == 0].drop(columns=["is_train"]).reset_index(drop=True)
)

X_train = train_processed.drop(
    columns=[target_col] + (["id"] if "id" in train_processed.columns else [])
)
y_train = train_processed[target_col].values

X_test = test_processed.drop(
    columns=[target_col] + (["id"] if "id" in test_processed.columns else []),
    errors="ignore",
)


# Helper function to convert dataframes to clean numeric matrices for XGB/Cat/NN
def prepare_numeric_df(df, cat_cols):
    df_num = df.copy()
    for col in df_num.columns:
        if df_num[col].dtype.name == "category" or col in cat_cols:
            df_num[col] = df_num[col].astype("category").cat.codes
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    return df_num


# Apply LightGBM feature importance selection to retain top 200 most predictive features
if X_train.shape[1] > 200:
    fs_lgb = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    X_train_fs = prepare_numeric_df(X_train, categorical_cols)
    fs_lgb.fit(X_train_fs, y_train)
    importances = fs_lgb.feature_importances_
    top_indices = np.argsort(importances)[::-1][:200]
    top_cols = list(X_train.columns[top_indices])

    X_train = X_train[top_cols]
    X_test = X_test[top_cols]
    categorical_cols = [c for c in categorical_cols if c in top_cols]


# ---------------------------------------------------------
# 3. Deep Tabular ResNet & GBDT Configurations
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return self.act(out + residual)


class DeepTabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


lgb_config = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_config = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

cat_config = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
}


# ---------------------------------------------------------
# 4. Stratified Cross-Validation & Out-Of-Fold Training
# ---------------------------------------------------------
n_splits = 5
y_binned = pd.qcut(y_train, q=n_splits, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

n_train = len(X_train)
n_test = len(X_test)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_nn = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_binned)):
    X_tr_df, y_tr = X_train.iloc[train_idx].copy(), y_train[train_idx]
    X_va_df, y_va = X_train.iloc[val_idx].copy(), y_train[val_idx]
    X_te_df = X_test.copy()

    global_mean = y_tr.mean()

    # In-fold leakage-free target encoding
    for ccol in categorical_cols:
        te_col_name = f"{ccol}_target_enc"
        fold_stats = (
            X_tr_df.assign(target=y_tr).groupby(ccol)["target"].agg(["count", "mean"])
        )
        smooth = 10
        smoothed_enc = (
            fold_stats["count"] * fold_stats["mean"] + smooth * global_mean
        ) / (fold_stats["count"] + smooth)
        enc_dict = smoothed_enc.to_dict()

        X_tr_df[te_col_name] = (
            X_tr_df[ccol].astype(object).map(enc_dict).fillna(global_mean).astype(float)
        )
        X_va_df[te_col_name] = (
            X_va_df[ccol].astype(object).map(enc_dict).fillna(global_mean).astype(float)
        )
        X_te_df[te_col_name] = (
            X_te_df[ccol].astype(object).map(enc_dict).fillna(global_mean).astype(float)
        )

    # --- Model 1: LightGBM ---
    lgb_model = lgb.LGBMRegressor(**lgb_config)
    lgb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_df)
    test_lgb += lgb_model.predict(X_te_df) / n_splits

    # Numeric preprocessing for XGBoost, CatBoost, and Neural Network
    X_tr_num = prepare_numeric_df(X_tr_df, categorical_cols)
    X_va_num = prepare_numeric_df(X_va_df, categorical_cols)
    X_te_num = prepare_numeric_df(X_te_df, categorical_cols)

    col_medians = X_tr_num.median()
    X_tr_num = X_tr_num.fillna(col_medians).fillna(0)
    X_va_num = X_va_num.fillna(col_medians).fillna(0)
    X_te_num = X_te_num.fillna(col_medians).fillna(0)

    # --- Model 2: XGBoost ---
    xgb_model = xgb.XGBRegressor(**xgb_config)
    xgb_model.fit(
        X_tr_num,
        y_tr,
        eval_set=[(X_va_num, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_va_num)
    test_xgb += xgb_model.predict(X_te_num) / n_splits

    # --- Model 3: CatBoost ---
    cat_model = CatBoostRegressor(**cat_config)
    cat_model.fit(
        X_tr_num,
        y_tr,
        eval_set=(X_va_num, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_va_num)
    test_cat += cat_model.predict(X_te_num) / n_splits

    # --- Model 4: PyTorch Deep Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_num)
    X_va_scaled = scaler.transform(X_va_num)
    X_te_scaled = scaler.transform(X_te_num)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    num_input_features = X_tr_scaled.shape[1]
    nn_model = DeepTabularResNet(
        input_dim=num_input_features, hidden_dim=256, num_blocks=3
    ).to(device)

    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_nn_weights = None
    patience, patience_counter = 8, 0

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = nn_model(batch_x)
                val_losses.append(loss_fn(preds, batch_y).item())

        val_mse = np.mean(val_losses)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_nn_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        oof_nn[val_idx] = nn_model(va_tensor).cpu().numpy()

        te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        test_nn += nn_model(te_tensor).cpu().numpy() / n_splits

# ---------------------------------------------------------
# 5. Ridge CV Stacked Meta-Learner & Submission Output
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])

poly = PolynomialFeatures(degree=2, include_bias=False)
oof_poly = poly.fit_transform(oof_matrix)
test_poly = poly.transform(test_matrix)

scaler_meta = StandardScaler()
oof_poly_scaled = scaler_meta.fit_transform(oof_poly)
test_poly_scaled = scaler_meta.transform(test_poly)

alphas = np.logspace(-3, 5, 100)
meta_model = RidgeCV(alphas=alphas, cv=5, scoring="neg_root_mean_squared_error")
meta_model.fit(oof_poly_scaled, y_train)

oof_blend = np.clip(meta_model.predict(oof_poly_scaled), 0, None)
final_score = np.sqrt(mean_squared_error(y_train, oof_blend))

test_blend = np.clip(meta_model.predict(test_poly_scaled), 0, None)

submission_df = pd.DataFrame(
    {"id": test_ids, "Debt-to-Earnings Annual Rate": test_blend}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")