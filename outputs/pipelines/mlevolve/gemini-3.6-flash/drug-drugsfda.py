import os
import re
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("--- Starting Data Processing and Feature Engineering ---")

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

target_col = "marketing_status"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training dataset.")

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df[target_col].astype(str))
print(f"Target distribution across classes: {np.bincount(y_train)}")

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

for col in ["id", "ID", "index"]:
    if col in X_train_raw.columns:
        X_train_raw = X_train_raw.drop(columns=[col])
    if col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[col])


# 2. Domain-Specific Feature Extraction
def extract_domain_features(df):
    feats = {}
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            col_str = df[col].astype(str).fillna("")
            feats[f"{col}_ingredient_count"] = col_str.apply(
                lambda s: len(re.split(r"[;/,\+]", s)) if len(s) > 0 else 0
            )
            feats[f"{col}_char_len"] = col_str.str.len()
            feats[f"{col}_word_count"] = col_str.apply(lambda s: len(s.split()))

            def extract_first_num(s):
                matches = re.findall(r"(\d+(?:\.\d+)?)", s)
                return float(matches[0]) if matches else np.nan

            feats[f"{col}_extracted_dosage"] = col_str.apply(extract_first_num)
    return pd.DataFrame(feats, index=df.index)


print("Extracting domain-specific features...")
train_domain = extract_domain_features(X_train_raw)
test_domain = extract_domain_features(X_test_raw)

cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()

train_feat_dict = {}
test_feat_dict = {}

# 3. Frequency & Multi-Scale Out-Of-Fold Target Encoding
skf_prep = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y_train.mean()
smoothing_factors = [5.0, 25.0]

for col in cat_cols:
    freq_map = X_train_raw[col].value_counts(normalize=True).to_dict()
    train_feat_dict[f"{col}_freq"] = X_train_raw[col].map(freq_map).fillna(0)
    test_feat_dict[f"{col}_freq"] = X_test_raw[col].map(freq_map).fillna(0)

    for m_smooth in smoothing_factors:
        m_tag = int(m_smooth)
        te_tr = np.full(len(X_train_raw), np.nan)

        for tr_idx, val_idx in skf_prep.split(X_train_raw, y_train):
            fold_train_cat = X_train_raw.iloc[tr_idx][col]
            fold_df = pd.DataFrame({"cat": fold_train_cat, "target": y_train[tr_idx]})
            stats = fold_df.groupby("cat")["target"].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
                stats["count"] + m_smooth
            )
            enc_map = smoothed.to_dict()
            te_tr[val_idx] = X_train_raw.iloc[val_idx][col].map(enc_map).fillna(global_mean)

        train_feat_dict[f"{col}_target_enc_m{m_tag}"] = te_tr

        full_df = pd.DataFrame({"cat": X_train_raw[col], "target": y_train})
        full_stats = full_df.groupby("cat")["target"].agg(["count", "mean"])
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + m_smooth * global_mean
        ) / (full_stats["count"] + m_smooth)
        test_feat_dict[f"{col}_target_enc_m{m_tag}"] = (
            X_test_raw[col].map(full_smoothed.to_dict()).fillna(global_mean)
        )

# 4. Field-Aware Text Embeddings via Field-Specific TF-IDF + TruncatedSVD
if len(cat_cols) > 0:
    print("Computing field-aware TF-IDF + TruncatedSVD features for categorical columns...")
    for col in cat_cols:
        col_tr_text = X_train_raw[col].astype(str).fillna("")
        col_te_text = X_test_raw[col].astype(str).fillna("")

        # Field-specific word TF-IDF + SVD
        tfidf_word = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), analyzer="word", sublinear_tf=True)
        tfidf_tr_word = tfidf_word.fit_transform(col_tr_text)
        tfidf_te_word = tfidf_word.transform(col_te_text)

        n_comp_w = min(16, tfidf_tr_word.shape[1] - 1) if tfidf_tr_word.shape[1] > 1 else 0
        if n_comp_w > 0:
            svd_word = TruncatedSVD(n_components=n_comp_w, random_state=42)
            svd_tr_w = svd_word.fit_transform(tfidf_tr_word)
            svd_te_w = svd_word.transform(tfidf_te_word)
            for i in range(n_comp_w):
                train_feat_dict[f"svd_word_{col}_{i}"] = svd_tr_w[:, i]
                test_feat_dict[f"svd_word_{col}_{i}"] = svd_te_w[:, i]

        # Field-specific char TF-IDF + SVD
        tfidf_char = TfidfVectorizer(max_features=5000, ngram_range=(2, 5), analyzer="char_wb", sublinear_tf=True)
        tfidf_tr_char = tfidf_char.fit_transform(col_tr_text)
        tfidf_te_char = tfidf_char.transform(col_te_text)

        n_comp_c = min(16, tfidf_tr_char.shape[1] - 1) if tfidf_tr_char.shape[1] > 1 else 0
        if n_comp_c > 0:
            svd_char = TruncatedSVD(n_components=n_comp_c, random_state=42)
            svd_tr_c = svd_char.fit_transform(tfidf_tr_char)
            svd_te_c = svd_char.transform(tfidf_te_char)
            for i in range(n_comp_c):
                train_feat_dict[f"svd_char_{col}_{i}"] = svd_tr_c[:, i]
                test_feat_dict[f"svd_char_{col}_{i}"] = svd_te_c[:, i]

train_extra_df = pd.DataFrame(train_feat_dict, index=X_train_raw.index)
test_extra_df = pd.DataFrame(test_feat_dict, index=X_test_raw.index)

X_train_combined = pd.concat([X_train_raw, train_domain, train_extra_df], axis=1)
X_test_combined = pd.concat([X_test_raw, test_domain, test_extra_df], axis=1)

# Re-identify numerical columns
num_cols = X_train_combined.select_dtypes(include=[np.number]).columns.tolist()

# 5. One-Hot Encoding for Low-Cardinality Categorical Features
low_card_cols = [col for col in cat_cols if X_train_combined[col].nunique() <= 20]
if len(low_card_cols) > 0:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_train = ohe.fit_transform(X_train_combined[low_card_cols].fillna("missing"))
    ohe_test = ohe.transform(X_test_combined[low_card_cols].fillna("missing"))

    ohe_feature_names = ohe.get_feature_names_out(low_card_cols)
    ohe_train_df = pd.DataFrame(
        ohe_train, columns=ohe_feature_names, index=X_train_combined.index
    )
    ohe_test_df = pd.DataFrame(
        ohe_test, columns=ohe_feature_names, index=X_test_combined.index
    )

    X_train_num = pd.concat([X_train_combined[num_cols], ohe_train_df], axis=1)
    X_test_num = pd.concat([X_test_combined[num_cols], ohe_test_df], axis=1)
else:
    X_train_num = X_train_combined[num_cols].copy()
    X_test_num = X_test_combined[num_cols].copy()

# 6. Preprocessing setup (Imputation & Scaling)
print("Preprocessing clean feature matrices...")
empty_cols = [c for c in X_train_num.columns if X_train_num[c].isna().all()]
X_train_num = X_train_num.drop(columns=empty_cols)
X_test_num = X_test_num.drop(columns=empty_cols, errors="ignore")

imputer = SimpleImputer(strategy="median")
X_train_clean = pd.DataFrame(
    imputer.fit_transform(X_train_num),
    columns=X_train_num.columns,
    index=X_train_num.index,
)
X_test_clean = pd.DataFrame(
    imputer.transform(X_test_num),
    columns=X_train_num.columns,
    index=X_test_num.index,
)


# ==========================================
# Model Architecture Definitions
# ==========================================


class TabularResBlock(nn.Module):

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
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def get_lgbm(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=45,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_samples=20,
        random_state=seed,
        objective="binary",
        metric="auc",
        verbose=-1,
    )


def get_xgb(seed=42):
    return xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=seed,
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_cat(seed=42):
    return cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=seed,
        eval_metric="AUC",
        verbose=0,
        early_stopping_rounds=50,
    )


# ==========================================
# Cross Validation and Model Training
# ==========================================

print("--- Starting Training and Cross-Validation ---")

n_samples = len(X_train_clean)
n_features = X_train_clean.shape[1]
n_test = len(X_test_clean)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_nn = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_clean.values, y_train)):
    fold_scaler = StandardScaler()

    X_tr_f = fold_scaler.fit_transform(X_train_clean.iloc[train_idx])
    X_va_f = fold_scaler.transform(X_train_clean.iloc[val_idx])
    X_te_f = fold_scaler.transform(X_test_clean).astype(np.float32)
    y_tr_f, y_va_f = y_train[train_idx], y_train[val_idx]

    # LightGBM
    model_lgb = get_lgbm(seed=42 + fold)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model_lgb.fit(
        X_tr_f,
        y_tr_f,
        eval_set=[(X_va_f, y_va_f)],
        eval_metric="auc",
        callbacks=callbacks,
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va_f)[:, 1]
    test_lgb += model_lgb.predict_proba(X_te_f)[:, 1] / n_splits

    # XGBoost
    model_xgb = get_xgb(seed=42 + fold)
    model_xgb.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va_f)[:, 1]
    test_xgb += model_xgb.predict_proba(X_te_f)[:, 1] / n_splits

    # CatBoost
    model_cat = get_cat(seed=42 + fold)
    model_cat.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va_f)[:, 1]
    test_cat += model_cat.predict_proba(X_te_f)[:, 1] / n_splits

    # PyTorch ResNet
    nn_model = TabularResNet(
        input_dim=n_features, hidden_dim=128, num_blocks=2, dropout_rate=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    train_ds = TensorDataset(
        torch.tensor(X_tr_f, dtype=torch.float32),
        torch.tensor(y_tr_f, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_f, dtype=torch.float32),
        torch.tensor(y_va_f, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_auc = -1.0
    best_val_preds = None

    for epoch in range(20):
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
        val_preds_epoch = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = nn_model(bx)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds_epoch.extend(probs)

        val_preds_epoch = np.array(val_preds_epoch)
        epoch_auc = roc_auc_score(y_va_f, val_preds_epoch)

        if epoch_auc > best_val_auc:
            best_val_auc = epoch_auc
            best_val_preds = val_preds_epoch
            torch.save(nn_model.state_dict(), f"./working/nn_model_fold_{fold}.pt")

    oof_nn[val_idx] = best_val_preds

    # Inference on Test with best saved PyTorch model
    nn_model.load_state_dict(torch.load(f"./working/nn_model_fold_{fold}.pt"))
    nn_model.eval()
    test_ds = TensorDataset(torch.tensor(X_te_f, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    nn_test_preds_fold = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            logits = nn_model(bx)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            nn_test_preds_fold.extend(probs)

    test_nn += np.array(nn_test_preds_fold) / n_splits

    print(
        f"Fold {fold+1}/{n_splits} AUCs -> LGB: {roc_auc_score(y_va_f, oof_lgb[val_idx]):.5f}, "
        f"XGB: {roc_auc_score(y_va_f, oof_xgb[val_idx]):.5f}, "
        f"CAT: {roc_auc_score(y_va_f, oof_cat[val_idx]):.5f}, "
        f"NN: {roc_auc_score(y_va_f, oof_nn[val_idx]):.5f}"
    )

# Ensemble Evaluation with Rank-Normalization and Derivative-Free Optimization (Nelder-Mead)
rank_oof_lgb = rankdata(oof_lgb) / len(oof_lgb)
rank_oof_xgb = rankdata(oof_xgb) / len(oof_xgb)
rank_oof_cat = rankdata(oof_cat) / len(oof_cat)
rank_oof_nn = rankdata(oof_nn) / len(oof_nn)

rank_test_lgb = rankdata(test_lgb) / len(test_lgb)
rank_test_xgb = rankdata(test_xgb) / len(test_xgb)
rank_test_cat = rankdata(test_cat) / len(test_cat)
rank_test_nn = rankdata(test_nn) / len(test_nn)

def loss_func(weights):
    w = np.maximum(0, np.array(weights))
    if np.sum(w) > 0:
        w = w / np.sum(w)
    else:
        w = np.ones(4) / 4
    blend_oof = (
        w[0] * rank_oof_lgb
        + w[1] * rank_oof_xgb
        + w[2] * rank_oof_cat
        + w[3] * rank_oof_nn
    )
    return -roc_auc_score(y_train, blend_oof)

init_weights = [0.25, 0.25, 0.25, 0.25]
res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500})

best_weights = np.maximum(0, res.x)
if np.sum(best_weights) > 0:
    best_weights = best_weights / np.sum(best_weights)
else:
    best_weights = np.ones(4) / 4

print(f"Optimized Ensemble Weights: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}, NN={best_weights[3]:.4f}")

oof_blend = (
    best_weights[0] * rank_oof_lgb
    + best_weights[1] * rank_oof_xgb
    + best_weights[2] * rank_oof_cat
    + best_weights[3] * rank_oof_nn
)
test_blend = (
    best_weights[0] * rank_test_lgb
    + best_weights[1] * rank_test_xgb
    + best_weights[2] * rank_test_cat
    + best_weights[3] * rank_test_nn
)

final_val_score = roc_auc_score(y_train, oof_blend)

# Generate Submission File
submission_path = "./submission/submission.csv"
sub_df = pd.DataFrame(
    {"id": np.asarray(test_ids, dtype=int), "marketing_status": test_blend}
)
sub_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_score:.5f}")