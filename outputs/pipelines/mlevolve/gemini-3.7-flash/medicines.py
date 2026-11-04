import os
import re
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 0. Setup & Output Directory
# ---------------------------------------------------------------------------
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Raw Datasets
# ---------------------------------------------------------------------------
print("Loading train and test datasets from ./input/...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

target_col = "Medicine status"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in train data.")

# Encode target variable
raw_target = train_df[target_col]
if raw_target.dtype == "object" or raw_target.dtype.name == "category":
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(raw_target.astype(str))
    print(f"Target classes mapped: {dict(enumerate(target_encoder.classes_))}")
else:
    y_train = raw_target.values.astype(np.int32)

y_train_series = pd.Series(y_train, name="target")

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# ---------------------------------------------------------------------------
# 2. Date and Temporal Feature Engineering
# ---------------------------------------------------------------------------
print("Extracting temporal features and regulatory timeline deltas...")


def detect_and_parse_dates(df):
    date_cols = []
    parsed_dates = pd.DataFrame(index=df.index)
    for col in df.columns:
        col_lower = col.lower()
        if (
            "date" in col_lower
            or "published" in col_lower
            or "time" in col_lower
            or "updated" in col_lower
        ):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notnull().sum() > 0:
                    date_cols.append(col)
                    parsed_dates[col] = parsed
            except Exception:
                pass
    return date_cols, parsed_dates


train_date_cols, train_parsed_dates = detect_and_parse_dates(X_train_raw)
_, test_parsed_dates = detect_and_parse_dates(X_test_raw)


def extract_temporal_features(parsed_dates, date_cols):
    feat_df = pd.DataFrame(index=parsed_dates.index)
    for col in date_cols:
        s = parsed_dates[col]
        clean_name = re.sub(r"[^\w\s]", "_", col).strip()
        feat_df[f"date_year_{clean_name}"] = s.dt.year.fillna(-1).astype(np.int32)
        feat_df[f"date_month_{clean_name}"] = s.dt.month.fillna(-1).astype(np.int32)
        feat_df[f"date_day_{clean_name}"] = s.dt.day.fillna(-1).astype(np.int32)
        feat_df[f"date_dayofweek_{clean_name}"] = s.dt.dayofweek.fillna(-1).astype(
            np.int32
        )
        feat_df[f"date_quarter_{clean_name}"] = s.dt.quarter.fillna(-1).astype(np.int32)
        feat_df[f"date_is_null_{clean_name}"] = s.isnull().astype(np.int32)
        feat_df[f"date_elapsed_days_{clean_name}"] = (
            (pd.to_datetime("2025-01-01") - s).dt.days.fillna(-9999).astype(np.float32)
        )

    # Cross-date duration differentials
    if len(date_cols) >= 2:
        for i in range(len(date_cols)):
            for j in range(i + 1, len(date_cols)):
                col_i = date_cols[i]
                col_j = date_cols[j]
                clean_i = re.sub(r"[^\w\s]", "_", col_i).strip()
                clean_j = re.sub(r"[^\w\s]", "_", col_j).strip()
                diff_days = (
                    (parsed_dates[col_i] - parsed_dates[col_j])
                    .dt.days.fillna(-9999)
                    .astype(np.float32)
                )
                feat_df[f"date_diff_{clean_i}_minus_{clean_j}"] = diff_days

    return feat_df


temporal_feats_train = extract_temporal_features(train_parsed_dates, train_date_cols)
temporal_feats_test = extract_temporal_features(test_parsed_dates, train_date_cols)

# ---------------------------------------------------------------------------
# 3. Missingness Patterns and Topology
# ---------------------------------------------------------------------------
print("Generating missingness topology features...")
missing_train = pd.DataFrame(index=X_train_raw.index)
missing_test = pd.DataFrame(index=X_test_raw.index)

for col in X_train_raw.columns:
    clean_col = re.sub(r"[^\w\s]", "_", col).strip()
    missing_train[f"missing_{clean_col}"] = X_train_raw[col].isnull().astype(np.int32)
    missing_test[f"missing_{clean_col}"] = X_test_raw[col].isnull().astype(np.int32)

missing_train["total_missing_count"] = missing_train.sum(axis=1).astype(np.float32)
missing_test["total_missing_count"] = missing_test.sum(axis=1).astype(np.float32)

# ---------------------------------------------------------------------------
# 4. Domain-Specific Regulatory Keywords & Text Metrics
# ---------------------------------------------------------------------------
print("Extracting domain regulatory keyword signals...")
regulatory_keywords = [
    "orphan",
    "biosimilar",
    "generic",
    "conditional",
    "exceptional",
    "withdrawn",
    "refusal",
    "accelerated",
    "pediatric",
    "safety",
    "monitoring",
    "human",
    "veterinary",
    "biological",
    "immunological",
    "advanced therapy",
    "cancer",
    "cardiovascular",
    "infectious",
    "neurology",
    "rare",
]

object_cols = [
    col
    for col in X_train_raw.columns
    if X_train_raw[col].dtype == "object" or X_train_raw[col].dtype.name == "category"
]
object_cols = [c for c in object_cols if c not in train_date_cols]

train_full_text = (
    X_train_raw[object_cols]
    .fillna("")
    .astype(str)
    .apply(lambda row: " ".join(row), axis=1)
)
test_full_text = (
    X_test_raw[object_cols]
    .fillna("")
    .astype(str)
    .apply(lambda row: " ".join(row), axis=1)
)

kw_feats_train = pd.DataFrame(index=X_train_raw.index)
kw_feats_test = pd.DataFrame(index=X_test_raw.index)

for kw in regulatory_keywords:
    slug = kw.replace(" ", "_")
    pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
    kw_feats_train[f"flag_kw_{slug}"] = train_full_text.str.contains(pattern).astype(
        np.int32
    )
    kw_feats_test[f"flag_kw_{slug}"] = test_full_text.str.contains(pattern).astype(
        np.int32
    )

kw_feats_train["text_char_length"] = train_full_text.str.len().astype(np.float32)
kw_feats_test["text_char_length"] = test_full_text.str.len().astype(np.float32)
kw_feats_train["text_word_count"] = (
    train_full_text.str.split().apply(len).astype(np.float32)
)
kw_feats_test["text_word_count"] = (
    test_full_text.str.split().apply(len).astype(np.float32)
)

# ---------------------------------------------------------------------------
# 5. Semantic Embeddings via TF-IDF + Truncated SVD
# ---------------------------------------------------------------------------
print("Computing TF-IDF and SVD latent semantic components...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=25000,
    min_df=2,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
)
svd = TruncatedSVD(n_components=48, random_state=42)

tfidf_train = tfidf.fit_transform(train_full_text)
tfidf_test = tfidf.transform(test_full_text)

svd_train = svd.fit_transform(tfidf_train)
svd_test = svd.transform(tfidf_test)

svd_feats_train = pd.DataFrame(
    svd_train,
    columns=[f"svd_dim_{i}" for i in range(svd_train.shape[1])],
    index=X_train_raw.index,
)
svd_feats_test = pd.DataFrame(
    svd_test,
    columns=[f"svd_dim_{i}" for i in range(svd_test.shape[1])],
    index=X_test_raw.index,
)

# ---------------------------------------------------------------------------
# 6. Categorical Frequency & Out-Of-Fold Target Encoding
# ---------------------------------------------------------------------------
print("Encoding categorical attributes with leak-free OOF target encoding...")
cat_feats_train = pd.DataFrame(index=X_train_raw.index)
cat_feats_test = pd.DataFrame(index=X_test_raw.index)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in object_cols:
    clean_col = re.sub(r"[^\w\s]", "_", col).strip()
    s_train = X_train_raw[col].fillna("__MISSING__").astype(str)
    s_test = X_test_raw[col].fillna("__MISSING__").astype(str)

    freq_map = s_train.value_counts(normalize=True).to_dict()
    cat_feats_train[f"freq_{clean_col}"] = (
        s_train.map(freq_map).fillna(0.0).astype(np.float32)
    )
    cat_feats_test[f"freq_{clean_col}"] = (
        s_test.map(freq_map).fillna(0.0).astype(np.float32)
    )

    count_map = s_train.value_counts().to_dict()
    cat_feats_train[f"count_{clean_col}"] = (
        s_train.map(count_map).fillna(0.0).astype(np.float32)
    )
    cat_feats_test[f"count_{clean_col}"] = (
        s_test.map(count_map).fillna(0.0).astype(np.float32)
    )

    global_mean = float(y_train_series.mean())
    oof_target_enc = np.zeros(len(s_train), dtype=np.float32)
    smoothing = 10.0

    for tr_idx, val_idx in skf.split(s_train, y_train):
        fold_train_s = s_train.iloc[tr_idx]
        fold_train_y = y_train_series.iloc[tr_idx]

        stats = fold_train_y.groupby(fold_train_s).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        mapping = smoothed.to_dict()

        oof_target_enc[val_idx] = (
            s_train.iloc[val_idx].map(mapping).fillna(global_mean).values
        )

    full_stats = y_train_series.groupby(s_train).agg(["count", "mean"])
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    test_target_enc = s_test.map(full_smoothed.to_dict()).fillna(global_mean).values

    cat_feats_train[f"oof_te_{clean_col}"] = oof_target_enc
    cat_feats_test[f"oof_te_{clean_col}"] = test_target_enc.astype(np.float32)

# ---------------------------------------------------------------------------
# 7. Numerical Column Processing & Standardization
# ---------------------------------------------------------------------------
print("Processing existing numeric features...")
num_cols = [
    col
    for col in X_train_raw.columns
    if X_train_raw[col].dtype in [np.int64, np.float64, np.int32, np.float32]
    and col not in train_date_cols
]

num_feats_train = pd.DataFrame(index=X_train_raw.index)
num_feats_test = pd.DataFrame(index=X_test_raw.index)

if len(num_cols) > 0:
    scaler = RobustScaler()
    train_num_imputed = X_train_raw[num_cols].fillna(X_train_raw[num_cols].median())
    test_num_imputed = X_test_raw[num_cols].fillna(X_train_raw[num_cols].median())

    scaled_train = scaler.fit_transform(train_num_imputed)
    scaled_test = scaler.transform(test_num_imputed)

    for idx, col in enumerate(num_cols):
        clean_col = re.sub(r"[^\w\s]", "_", col).strip()
        num_feats_train[f"num_{clean_col}"] = scaled_train[:, idx].astype(np.float32)
        num_feats_test[f"num_{clean_col}"] = scaled_test[:, idx].astype(np.float32)

# ---------------------------------------------------------------------------
# 8. Feature Assembly & Normalization
# ---------------------------------------------------------------------------
print("Concatenating all engineered feature sets...")
X_train_final = pd.concat(
    [
        num_feats_train,
        temporal_feats_train,
        missing_train,
        kw_feats_train,
        svd_feats_train,
        cat_feats_train,
    ],
    axis=1,
)

X_test_final = pd.concat(
    [
        num_feats_test,
        temporal_feats_test,
        missing_test,
        kw_feats_test,
        svd_feats_test,
        cat_feats_test,
    ],
    axis=1,
)

X_train_final.columns = [
    re.sub(r"[^\w\s]", "_", c).replace(" ", "_") for c in X_train_final.columns
]
X_test_final.columns = [
    re.sub(r"[^\w\s]", "_", c).replace(" ", "_") for c in X_test_final.columns
]

X_train_final = X_train_final.fillna(0.0).astype(np.float32)
X_test_final = X_test_final.fillna(0.0).astype(np.float32)

X_train = X_train_final.values.astype(np.float32)
X_test = X_test_final.values.astype(np.float32)
n_samples, n_features = X_train.shape
n_test = X_test.shape[0]

print(f"Final Engineered Features -> Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 9. Model Architecture Definitions
# ---------------------------------------------------------------------------


class TabularResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = (
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)
        )
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = (
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)
        )
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        out = out + residual
        return self.act2(out)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResidualBlock(
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    use_layer_norm=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits.squeeze(-1)


def get_torch_criterion():
    return nn.BCEWithLogitsLoss()


def get_torch_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 40,
    eta_min: float = 1e-6,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return optimizer, scheduler


def get_lgbm_model(seed: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=50,
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=seed,
        n_jobs=-1,
    )


def get_catboost_model(seed: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


# ---------------------------------------------------------------------------
# 10. Stratified 5-Fold Cross-Validation Ensemble Loop
# ---------------------------------------------------------------------------
cv_splits = list(skf.split(X_train, y_train))
num_folds = len(cv_splits)

oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_cat = np.zeros(n_samples, dtype=np.float32)
oof_nn = np.zeros(n_samples, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)
test_cat = np.zeros(n_test, dtype=np.float32)
test_nn = np.zeros(n_test, dtype=np.float32)

print(
    f"Starting 5-Fold CV ensemble training on {n_samples} samples across 4 diverse architectures..."
)

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # --- A. LightGBM ---
    lgb_clf = get_lgbm_model(seed=42 + fold)
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred_lgb = lgb_clf.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx] = val_pred_lgb
    test_lgb += lgb_clf.predict_proba(X_test)[:, 1] / num_folds

    # --- B. XGBoost ---
    xgb_clf = get_xgb_model(seed=42 + fold)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_xgb = xgb_clf.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = val_pred_xgb
    test_xgb += xgb_clf.predict_proba(X_test)[:, 1] / num_folds

    # --- C. CatBoost ---
    cat_clf = get_catboost_model(seed=42 + fold)
    cat_clf.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_pred_cat = cat_clf.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = val_pred_cat
    test_cat += cat_clf.predict_proba(X_test)[:, 1] / num_folds

    # --- D. PyTorch TabularResNet ---
    nn_model = TabularResNet(
        input_dim=n_features, hidden_dim=256, num_blocks=3, dropout_rate=0.25
    ).to(device)
    criterion = get_torch_criterion()
    optimizer, scheduler = get_torch_optimizer_and_scheduler(
        nn_model, lr=1e-3, weight_decay=1e-4, t_max=40
    )

    tr_dataset = TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(torch.tensor(X_test))

    tr_loader = DataLoader(
        tr_dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    best_val_auc = 0.0
    best_nn_val_preds = None
    best_nn_weights = None
    patience = 12
    patience_counter = 0

    for epoch in range(1, 41):
        nn_model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_logits_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_logits_list.append(nn_model(batch_x).cpu())
        val_probs = torch.sigmoid(torch.cat(val_logits_list, dim=0)).numpy().ravel()
        epoch_auc = roc_auc_score(y_val, val_probs)

        if epoch_auc > best_val_auc:
            best_val_auc = epoch_auc
            best_nn_val_preds = val_probs
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    oof_nn[val_idx] = best_nn_val_preds

    nn_model.load_state_dict(best_nn_weights)
    nn_model.to(device)
    nn_model.eval()
    test_logits_list = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            test_logits_list.append(nn_model(batch_x).cpu())
    test_probs = torch.sigmoid(torch.cat(test_logits_list, dim=0)).numpy().ravel()
    test_nn += test_probs / num_folds

    fold_lgb_auc = roc_auc_score(y_val, val_pred_lgb)
    fold_xgb_auc = roc_auc_score(y_val, val_pred_xgb)
    fold_cat_auc = roc_auc_score(y_val, val_pred_cat)
    print(
        f"Fold {fold+1}/{num_folds} - LGB AUC: {fold_lgb_auc:.5f} | XGB AUC: {fold_xgb_auc:.5f} | CAT AUC: {fold_cat_auc:.5f} | NN AUC: {best_val_auc:.5f}"
    )

# ---------------------------------------------------------------------------
# 11. OOF Optimization & Blending
# ---------------------------------------------------------------------------
auc_lgb = roc_auc_score(y_train, oof_lgb)
auc_xgb = roc_auc_score(y_train, oof_xgb)
auc_cat = roc_auc_score(y_train, oof_cat)
auc_nn = roc_auc_score(y_train, oof_nn)

print(
    f"Individual OOF ROC-AUC Scores -> LGB: {auc_lgb:.5f}, XGB: {auc_xgb:.5f}, CAT: {auc_cat:.5f}, NN: {auc_nn:.5f}"
)

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def objective(weights):
    w = np.maximum(0, weights)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 0.0
    w_norm = w / w_sum
    blend_oof = np.dot(oof_matrix, w_norm)
    return -roc_auc_score(y_train, blend_oof)


init_weights = np.array([0.30, 0.30, 0.30, 0.10])
opt_res = minimize(
    objective,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 500, "disp": False},
)
opt_w = np.maximum(0, opt_res.x)
opt_w = opt_w / np.sum(opt_w)

print(
    f"Optimal Ensemble Weights -> LGB: {opt_w[0]:.4f}, XGB: {opt_w[1]:.4f}, CAT: {opt_w[2]:.4f}, NN: {opt_w[3]:.4f}"
)

oof_final_pred = np.dot(oof_matrix, opt_w)
final_val_auc = float(roc_auc_score(y_train, oof_final_pred))

# ---------------------------------------------------------------------------
# 12. Submission Generation & Metric Output
# ---------------------------------------------------------------------------
test_final_pred = np.dot(test_matrix, opt_w)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_final_pred), dtype=np.int64),
        "Medicine status": test_final_pred,
    }
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Generated submission successfully saved to {submission_path}.")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"Final Validation Score: {final_val_auc}")