import json
import os
from typing import Any, Dict

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ---------------------------------------------------------------------------
# Setup Directories and Reproducibility
# ---------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Load Data & Target Identification
# ---------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Status"
if target_col not in train_df.columns:
    candidates = [c for c in train_df.columns if c.lower() == "status"]
    if candidates:
        target_col = candidates[0]
    else:
        raise ValueError(
            f"Target column '{target_col}' not found in train columns: {list(train_df.columns)}"
        )

# Target encoding
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df[target_col].astype(str))
train_df["target"] = y_train

# 5-Fold Stratified Split Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
    train_df.loc[val_idx, "fold"] = fold_idx

# ---------------------------------------------------------------------------
# 2. Feature Engineering Pipeline
# ---------------------------------------------------------------------------
raw_features = [c for c in train_df.columns if c not in [target_col, "target", "fold"]]

datetime_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in raw_features:
    if pd.api.types.is_datetime64_any_dtype(train_df[col]):
        datetime_cols.append(col)
        continue

    if train_df[col].dtype == "object" or str(train_df[col].dtype) == "string":
        sample_non_null = train_df[col].dropna().head(50)
        if len(sample_non_null) > 0:
            try:
                parsed_sample = pd.to_datetime(
                    sample_non_null, errors="coerce", format="mixed"
                )
                if parsed_sample.notna().mean() >= 0.7:
                    datetime_cols.append(col)
                    train_df[col] = pd.to_datetime(
                        train_df[col], errors="coerce", format="mixed"
                    )
                    test_df[col] = pd.to_datetime(
                        test_df[col], errors="coerce", format="mixed"
                    )
                    continue
            except Exception:
                try:
                    parsed_sample = pd.to_datetime(sample_non_null, errors="coerce")
                    if parsed_sample.notna().mean() >= 0.7:
                        datetime_cols.append(col)
                        train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
                        test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
                        continue
                except Exception:
                    pass

        str_lens = train_df[col].dropna().astype(str).str.len()
        avg_len = str_lens.mean() if len(str_lens) > 0 else 0
        word_counts = train_df[col].dropna().astype(str).str.split().str.len()
        avg_words = word_counts.mean() if len(word_counts) > 0 else 0

        if avg_len > 35 or avg_words > 4:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

train_feat = pd.DataFrame(index=train_df.index)
test_feat = pd.DataFrame(index=test_df.index)

train_feat["target"] = train_df["target"]
train_feat["fold"] = train_df["fold"]

# Missingness Indicators
for col in raw_features:
    if train_df[col].isna().sum() > 0 or test_df[col].isna().sum() > 0:
        train_feat[f"{col}_isna"] = train_df[col].isna().astype(np.int8)
        test_feat[f"{col}_isna"] = test_df[col].isna().astype(np.int8)

# Datetime Features & Pairwise Differences
ref_date = pd.Timestamp("2000-01-01")
for col in datetime_cols:
    train_dates = train_df[col]
    test_dates = test_df[col]

    train_feat[f"{col}_year"] = train_dates.dt.year.fillna(-1).astype(np.int16)
    test_feat[f"{col}_year"] = test_dates.dt.year.fillna(-1).astype(np.int16)

    train_feat[f"{col}_month"] = train_dates.dt.month.fillna(-1).astype(np.int8)
    test_feat[f"{col}_month"] = test_dates.dt.month.fillna(-1).astype(np.int8)

    train_feat[f"{col}_day"] = train_dates.dt.day.fillna(-1).astype(np.int8)
    test_feat[f"{col}_day"] = test_dates.dt.day.fillna(-1).astype(np.int8)

    train_feat[f"{col}_dayofweek"] = train_dates.dt.dayofweek.fillna(-1).astype(np.int8)
    test_feat[f"{col}_dayofweek"] = test_dates.dt.dayofweek.fillna(-1).astype(np.int8)

    train_feat[f"{col}_quarter"] = train_dates.dt.quarter.fillna(-1).astype(np.int8)
    test_feat[f"{col}_quarter"] = test_dates.dt.quarter.fillna(-1).astype(np.int8)

    train_feat[f"{col}_days_since_ref"] = (
        (train_dates - ref_date).dt.days.fillna(-9999).astype(np.float32)
    )
    test_feat[f"{col}_days_since_ref"] = (
        (test_dates - ref_date).dt.days.fillna(-9999).astype(np.float32)
    )

for i in range(len(datetime_cols)):
    for j in range(i + 1, len(datetime_cols)):
        d1, d2 = datetime_cols[i], datetime_cols[j]
        diff_train = (train_df[d1] - train_df[d2]).dt.days
        diff_test = (test_df[d1] - test_df[d2]).dt.days

        train_feat[f"datediff_{d1}_minus_{d2}"] = diff_train.fillna(-9999).astype(
            np.float32
        )
        test_feat[f"datediff_{d1}_minus_{d2}"] = diff_test.fillna(-9999).astype(
            np.float32
        )

# Text Decomposition (TF-IDF + SVD)
for col in text_cols:
    train_text = train_df[col].fillna("").astype(str)
    test_text = test_df[col].fillna("").astype(str)

    train_feat[f"{col}_char_len"] = train_text.str.len().astype(np.float32)
    test_feat[f"{col}_char_len"] = test_text.str.len().astype(np.float32)

    train_feat[f"{col}_word_count"] = (
        train_text.str.split().str.len().astype(np.float32)
    )
    test_feat[f"{col}_word_count"] = test_text.str.split().str.len().astype(np.float32)

    train_feat[f"{col}_digit_count"] = train_text.apply(
        lambda s: sum(c.isdigit() for c in s)
    ).astype(np.float32)
    test_feat[f"{col}_digit_count"] = test_text.apply(
        lambda s: sum(c.isdigit() for c in s)
    ).astype(np.float32)

    tfidf = TfidfVectorizer(
        max_features=500, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    n_components = min(15, train_tfidf.shape[1] - 1) if train_tfidf.shape[1] > 2 else 0
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)

        for k in range(n_components):
            train_feat[f"{col}_svd_{k}"] = train_svd[:, k].astype(np.float32)
            test_feat[f"{col}_svd_{k}"] = test_svd[:, k].astype(np.float32)

# Categorical Encodings & Out-of-Fold Target Encodings
global_target_mean = float(y_train.mean())
smooth_weight = 10.0

for col in cat_cols:
    train_s = train_df[col].fillna("__MISSING__").astype(str)
    test_s = test_df[col].fillna("__MISSING__").astype(str)

    freq_map = train_s.value_counts(normalize=True).to_dict()
    train_feat[f"{col}_freq"] = train_s.map(freq_map).fillna(0.0).astype(np.float32)
    test_feat[f"{col}_freq"] = test_s.map(freq_map).fillna(0.0).astype(np.float32)

    le = LabelEncoder()
    le.fit(pd.concat([train_s, test_s]).unique())
    train_feat[f"{col}_code"] = le.transform(train_s).astype(np.int32)
    test_feat[f"{col}_code"] = le.transform(test_s).astype(np.int32)

    train_oof_te = np.full(len(train_df), global_target_mean, dtype=np.float32)
    for fold_idx in range(5):
        val_mask = train_df["fold"] == fold_idx
        trn_mask = ~val_mask

        trn_cat = train_s[trn_mask]
        trn_y = y_train[trn_mask]

        counts = trn_cat.value_counts()
        sums = pd.Series(trn_y, index=trn_cat.index).groupby(trn_cat).sum()
        te_map = (
            (sums + smooth_weight * global_target_mean) / (counts + smooth_weight)
        ).to_dict()

        train_oof_te[val_mask] = (
            train_s[val_mask].map(te_map).fillna(global_target_mean).values
        )

    train_feat[f"{col}_target_enc"] = train_oof_te

    all_counts = train_s.value_counts()
    all_sums = pd.Series(y_train, index=train_s.index).groupby(train_s).sum()
    full_te_map = (
        (all_sums + smooth_weight * global_target_mean) / (all_counts + smooth_weight)
    ).to_dict()
    test_feat[f"{col}_target_enc"] = (
        test_s.map(full_te_map).fillna(global_target_mean).astype(np.float32)
    )

# Numerical Transformations
for col in num_cols:
    median_val = train_df[col].median()
    train_val = train_df[col].fillna(median_val).astype(np.float32)
    test_val = test_df[col].fillna(median_val).astype(np.float32)

    train_feat[col] = train_val
    test_feat[col] = test_val

    if (train_val >= 0).all():
        train_feat[f"{col}_log1p"] = np.log1p(train_val)
        test_feat[f"{col}_log1p"] = np.log1p(np.clip(test_val, 0, None))

# Interaction Features
if len(cat_cols) >= 2:
    col1, col2 = cat_cols[0], cat_cols[1]
    inter_train = train_df[col1].astype(str) + "__" + train_df[col2].astype(str)
    inter_test = test_df[col1].astype(str) + "__" + test_df[col2].astype(str)

    inter_freq = inter_train.value_counts(normalize=True).to_dict()
    train_feat[f"inter_{col1}_{col2}_freq"] = (
        inter_train.map(inter_freq).fillna(0.0).astype(np.float32)
    )
    test_feat[f"inter_{col1}_{col2}_freq"] = (
        inter_test.map(inter_freq).fillna(0.0).astype(np.float32)
    )

feature_columns = [c for c in train_feat.columns if c not in ["target", "fold"]]
test_feat = test_feat[feature_columns]

train_feat[feature_columns] = (
    train_feat[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(-999)
)
test_feat[feature_columns] = (
    test_feat[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(-999)
)


# ---------------------------------------------------------------------------
# 3. Model Architectures & Configurations
# ---------------------------------------------------------------------------
def get_lgb_config(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_config(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_config(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1500,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)
        )
        self.res_blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.head(x).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# 4. Multi-Model 5-Fold Cross Validation Training
# ---------------------------------------------------------------------------
y_all = train_feat["target"].values
folds_all = train_df["fold"].values
n_folds = int(folds_all.max() + 1)
n_train = len(train_feat)
n_test = len(test_feat)

X_train_df = train_feat[feature_columns]
X_test_df = test_feat[feature_columns]

oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cat = np.zeros(n_train, dtype=np.float64)
oof_nn = np.zeros(n_train, dtype=np.float64)

test_lgb = np.zeros(n_test, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)
test_cat = np.zeros(n_test, dtype=np.float64)
test_nn = np.zeros(n_test, dtype=np.float64)

for fold in range(n_folds):
    trn_mask = folds_all != fold
    val_mask = folds_all == fold

    X_tr = X_train_df.loc[trn_mask].values.astype(np.float32)
    y_tr = y_all[trn_mask].astype(np.float32)
    X_val = X_train_df.loc[val_mask].values.astype(np.float32)
    y_val = y_all[val_mask].astype(np.float32)
    X_te = X_test_df.values.astype(np.float32)

    # 4.1 LightGBM
    lgb_params = get_lgb_config(seed=42 + fold)
    lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_data = lgb.Dataset(X_val, label=y_val, reference=lgb_train_data)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=1500,
        valid_sets=[lgb_val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_mask] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_lgb += (
        lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / n_folds
    )

    # 4.2 XGBoost
    xgb_params = get_xgb_config(seed=42 + fold)
    xgb_model = xgb.XGBClassifier(
        **xgb_params, n_estimators=1500, early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_mask] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(X_te)[:, 1] / n_folds

    # 4.3 CatBoost
    cat_params = get_catboost_config(seed=42 + fold)
    cat_model = CatBoostClassifier(**cat_params, early_stopping_rounds=50)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_mask] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_te)[:, 1] / n_folds

    # 4.4 Tabular ResNet
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_te)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_sc, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    nn_model = TabularResNet(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.2,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val_auc = -1.0
    best_nn_val_preds = np.zeros(len(y_val))
    best_model_weights = None

    for epoch in range(30):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        epoch_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                probs = torch.sigmoid(nn_model(batch_x)).cpu().numpy()
                epoch_preds.extend(probs)
        epoch_preds = np.array(epoch_preds)

        try:
            curr_val_auc = roc_auc_score(y_val, epoch_preds)
        except Exception:
            curr_val_auc = 0.5

        if curr_val_auc > best_val_auc:
            best_val_auc = curr_val_auc
            best_nn_val_preds = epoch_preds
            best_model_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    oof_nn[val_mask] = best_nn_val_preds

    if best_model_weights is not None:
        nn_model.load_state_dict(
            {k: v.to(device) for k, v in best_model_weights.items()}
        )
    nn_model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(nn_model(test_tensor)).cpu().numpy()
        test_nn += test_probs / n_folds

    fold_blend = (
        0.35 * oof_lgb[val_mask]
        + 0.35 * oof_xgb[val_mask]
        + 0.20 * oof_cat[val_mask]
        + 0.10 * best_nn_val_preds
    )
    print(f"Fold {fold} ROC-AUC: {roc_auc_score(y_val, fold_blend):.5f}")

# ---------------------------------------------------------------------------
# 5. Optimal ROC-AUC Blending & Evaluation
# ---------------------------------------------------------------------------
lgb_score = roc_auc_score(y_all, oof_lgb)
xgb_score = roc_auc_score(y_all, oof_xgb)
cat_score = roc_auc_score(y_all, oof_cat)
nn_score = roc_auc_score(y_all, oof_nn)

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def blend_objective(weights_raw):
    w = np.exp(weights_raw) / np.sum(np.exp(weights_raw))
    pred = np.dot(oof_matrix, w)
    return -roc_auc_score(y_all, pred)


opt_res = minimize(blend_objective, x0=[1.0, 1.0, 1.0, 0.5], method="Nelder-Mead")
opt_weights = np.exp(opt_res.x) / np.sum(np.exp(opt_res.x))

final_oof_pred = np.dot(oof_matrix, opt_weights)
final_validation_score = roc_auc_score(y_all, final_oof_pred)

print(
    f"Individual OOF AUCs -> LGB: {lgb_score:.5f}, XGB: {xgb_score:.5f}, CAT: {cat_score:.5f}, NN: {nn_score:.5f}"
)
print(f"Optimal Weights: {np.round(opt_weights, 4)}")

# ---------------------------------------------------------------------------
# 6. Generate Submission
# ---------------------------------------------------------------------------
final_test_pred = np.dot(test_matrix, opt_weights)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_pred), dtype=np.int64), "Status": final_test_pred}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_score}")
