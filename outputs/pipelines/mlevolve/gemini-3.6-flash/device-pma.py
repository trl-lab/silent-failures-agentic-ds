import copy
import os
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

# Set up directories
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"
os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

# ----------------------------------------------------
# 1. Data Processing and Feature Engineering
# ----------------------------------------------------
print("Loading train and test datasets...")
train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

target_col = "decision_code"

# Encode target if non-numeric
if (
    train_df[target_col].dtype == "object"
    or str(train_df[target_col].dtype) == "category"
):
    target_le = LabelEncoder()
    train_df[target_col] = target_le.fit_transform(train_df[target_col].astype(str))

train_len = len(train_df)
y_train = train_df[target_col].values

df_all = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    ignore_index=True,
)

# Identify Column Types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in df_all.columns:
    if col.lower() in ["id", "index"]:
        continue

    col_str = str(col).lower()

    if "date" in col_str or "dt" in col_str:
        try:
            df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
            date_cols.append(col)
            continue
        except Exception:
            pass

    if df_all[col].dtype == "object" or str(df_all[col].dtype) == "category":
        df_all[col] = df_all[col].astype(str).fillna("missing")
        avg_len = df_all[col].str.len().mean()
        nunique = df_all[col].nunique()
        if avg_len > 30 and nunique > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

print(
    f"Detected {len(date_cols)} date, {len(text_cols)} text, {len(cat_cols)} categorical, and {len(num_cols)} numerical features."
)

# Date Features & Differences
print("Engineering date features...")
for col in date_cols:
    dt = df_all[col]
    df_all[f"{col}_year"] = dt.dt.year.fillna(-1).astype(int)
    df_all[f"{col}_month"] = dt.dt.month.fillna(-1).astype(int)
    df_all[f"{col}_day"] = dt.dt.day.fillna(-1).astype(int)
    df_all[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df_all[f"{col}_dayofyear"] = dt.dt.dayofyear.fillna(-1).astype(int)
    df_all[f"{col}_quarter"] = dt.dt.quarter.fillna(-1).astype(int)

for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        col1, col2 = date_cols[i], date_cols[j]
        diff_col = f"{col1}_minus_{col2}_days"
        df_all[diff_col] = (df_all[col1] - df_all[col2]).dt.total_seconds() / (
            24 * 3600
        )
        df_all[diff_col] = df_all[diff_col].fillna(-999)

df_all.drop(columns=date_cols, inplace=True, errors="ignore")

# Split df_all temporarily for feature extraction to avoid leakage
train_df_temp = df_all.iloc[:train_len].copy()
test_df_temp = df_all.iloc[train_len:].copy()

# Text Features (TF-IDF + SVD)
print("Extracting text features...")
for col in text_cols:
    df_all[f"{col}_char_len"] = df_all[col].str.len()
    df_all[f"{col}_word_count"] = df_all[col].apply(lambda x: len(str(x).split()))

    tfidf = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    tfidf_train = tfidf.fit_transform(train_df_temp[col].astype(str))
    tfidf_mat = tfidf.transform(df_all[col].astype(str))

    n_comp = min(8, tfidf_train.shape[1] - 1)
    if n_comp > 1:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(tfidf_train)
        svd_mat = svd.transform(tfidf_mat)
        for c in range(n_comp):
            df_all[f"{col}_svd_{c}"] = svd_mat[:, c]

df_all.drop(columns=text_cols, inplace=True, errors="ignore")

# Categorical Encodings
print("Encoding categorical features...")
for col in cat_cols:
    freq = train_df_temp[col].value_counts(normalize=True)
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0).astype(float)

    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

# Numerical Adjustments
for col in num_cols:
    df_all[f"{col}_isna"] = df_all[col].isna().astype(int)
    median_val = train_df_temp[col].median()
    df_all[col] = df_all[col].fillna(median_val if not pd.isna(median_val) else 0)

# Split back to Train and Test
train_features = df_all.iloc[:train_len].copy()
test_features = df_all.iloc[train_len:].copy()

train_features[target_col] = y_train

# K-Fold Stratified Splits & Out-Of-Fold Target Encoding
print("Creating Stratified K-Fold splits and Target Encodings...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_features["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_features, train_features[target_col])
):
    train_features.loc[val_idx, "fold"] = fold

for cat in cat_cols:
    te_col = f"{cat}_target_enc"
    train_features[te_col] = np.nan
    global_mean = y_train.mean()

    for fold in range(5):
        tr_idx = train_features["fold"] != fold
        val_idx = train_features["fold"] == fold

        means = train_features.loc[tr_idx].groupby(cat)[target_col].mean()
        train_features.loc[val_idx, te_col] = (
            train_features.loc[val_idx, cat].map(means).fillna(global_mean)
        )

    full_means = train_features.groupby(cat)[target_col].mean()
    test_features[te_col] = test_features[cat].map(full_means).fillna(global_mean)


# ----------------------------------------------------
# 2. Model Definitions
# ----------------------------------------------------
class TabularResNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.2
    ):
        super(TabularResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res1_dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1_act = nn.GELU()
        self.res1_drop = nn.Dropout(dropout_rate)
        self.res1_dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn2 = nn.BatchNorm1d(hidden_dim)

        self.res2_dense1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res2_bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.res2_act = nn.GELU()
        self.res2_drop = nn.Dropout(dropout_rate)
        self.res2_dense2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.res2_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.res2_shortcut = nn.Linear(hidden_dim, hidden_dim // 2)

        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)

        res1 = self.res1_dense1(h)
        res1 = self.res1_bn1(res1)
        res1 = self.res1_act(res1)
        res1 = self.res1_drop(res1)
        res1 = self.res1_dense2(res1)
        res1 = self.res1_bn2(res1)
        h = torch.relu(h + res1)

        res2 = self.res2_dense1(h)
        res2 = self.res2_bn1(res2)
        res2 = self.res2_act(res2)
        res2 = self.res2_drop(res2)
        res2 = self.res2_dense2(res2)
        res2 = self.res2_bn2(res2)
        shortcut = self.res2_shortcut(h)
        h = torch.relu(shortcut + res2)

        logits = self.head(h).squeeze(-1)
        return logits


def get_torch_model(
    input_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.2
) -> nn.Module:
    return TabularResNet(
        input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate
    )


def get_criterion() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def get_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_lgb_params() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params() -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


# ----------------------------------------------------
# 3. Model Training & Evaluation
# ----------------------------------------------------
train_df = train_features.copy()
test_df = test_features.copy()

fold_col = "fold"
feature_cols = [
    c for c in train_df.columns if c not in [target_col, fold_col, "id", "index"]
]

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0)
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce").fillna(0)

n_samples = len(train_df)
n_test = len(test_df)
n_folds = train_df[fold_col].nunique()

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_nn = np.zeros(n_test)

X_test_raw = test_df[feature_cols].values

print(
    f"Starting {n_folds}-Fold Cross-Validation training on {len(feature_cols)} features..."
)

for fold in range(n_folds):
    train_mask = train_df[fold_col] != fold
    val_mask = train_df[fold_col] == fold

    X_tr = train_df.loc[train_mask, feature_cols].values
    y_tr = train_df.loc[train_mask, target_col].values

    X_va = train_df.loc[val_mask, feature_cols].values
    y_va = train_df.loc[val_mask, target_col].values

    val_idx = np.where(val_mask)[0]

    # --- LightGBM ---
    lgb_params = get_lgb_params()
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test_raw) / n_folds

    # --- XGBoost ---
    xgb_params = get_xgb_params()
    xgb_model = xgb.XGBClassifier(
        **xgb_params, n_estimators=1000, early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test_raw)[:, 1] / n_folds

    # --- PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_test_sc = scaler.transform(X_test_raw)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = get_torch_model(input_dim=len(feature_cols)).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(nn_model, lr=1e-3, weight_decay=1e-4)

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(25):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        va_preds = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = nn_model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                va_preds.extend(probs)

        curr_auc = roc_auc_score(y_va, va_preds)
        if curr_auc > best_val_auc:
            best_val_auc = curr_auc
            best_weights = copy.deepcopy(nn_model.state_dict())

    if best_weights is not None:
        nn_model.load_state_dict(best_weights)

    nn_model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        oof_nn[val_idx] = torch.sigmoid(nn_model(va_tensor)).cpu().numpy()

        test_tensor = torch.tensor(X_test_sc, dtype=torch.float32).to(device)
        test_nn += torch.sigmoid(nn_model(test_tensor)).cpu().numpy() / n_folds

# ----------------------------------------------------
# 4. Final Ensembling & Submission Export
# ----------------------------------------------------
oof_ensemble = 0.40 * oof_lgb + 0.40 * oof_xgb + 0.20 * oof_nn
test_ensemble = 0.40 * test_lgb + 0.40 * test_xgb + 0.20 * test_nn

score = roc_auc_score(train_df[target_col], oof_ensemble)

submission_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=int), target_col: test_ensemble}
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {score}")