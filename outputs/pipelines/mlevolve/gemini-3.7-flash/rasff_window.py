import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, TensorDataset

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Load Data & Prepare Target
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "risk_decision"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in train.parquet")

y_raw = train_df[target_col].astype(str)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# ---------------------------------------------------------
# 2. Identify Column Types & Feature Engineering
# ---------------------------------------------------------
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in X_train_raw.columns:
    if pd.api.types.is_datetime64_any_dtype(X_train_raw[col]):
        date_cols.append(col)
    elif X_train_raw[col].dtype == object or str(X_train_raw[col].dtype) == "category":
        sample_vals = X_train_raw[col].dropna().astype(str)
        if len(sample_vals) > 0:
            avg_len = sample_vals.map(len).mean()
            num_unique = sample_vals.nunique()
            if avg_len > 35 or (num_unique > 500 and avg_len > 25):
                text_cols.append(col)
            elif any(
                d_kw in col.lower() for d_kw in ["date", "time", "created", "timestamp"]
            ):
                date_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(X_train_raw[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)


def extract_tabular_features(df):
    feats = pd.DataFrame(index=df.index)

    # Date / Temporal features
    for col in date_cols:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        feats[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(np.int32)
        feats[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(np.int32)
        feats[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(np.int32)
        feats[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(np.int32)
        feats[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(np.int32)
        feats[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(
            np.int32
        )

        month_rad = 2 * np.pi * (dt_series.dt.month.fillna(1) / 12.0)
        feats[f"{col}_month_sin"] = np.sin(month_rad).astype(np.float32)
        feats[f"{col}_month_cos"] = np.cos(month_rad).astype(np.float32)

    # Numerical features
    for col in num_cols:
        feats[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # Categorical string lengths and word counts
    for col in cat_cols:
        str_s = df[col].astype(str).fillna("missing")
        feats[f"{col}_len"] = str_s.map(len).astype(np.float32)
        feats[f"{col}_word_count"] = str_s.map(lambda x: len(x.split())).astype(
            np.float32
        )

    # High-risk hazard & safety keyword indicator features
    risk_keywords = [
        "salmonella", "listeria", "ecoli", "escherichia", "norovirus", "campylobacter",
        "bacillus", "aflatoxin", "ochratoxin", "pesticide", "chlorpyrifos", "heavy metal",
        "lead", "cadmium", "mercury", "arsenic", "foreign body", "glass", "metal",
        "plastic", "insect", "allergen", "undeclared", "milk", "egg", "gluten",
        "peanut", "unauthorized", "adulterat", "pathogen", "decay", "mold", "contaminat"
    ]
    all_text_series = df.astype(str).apply(lambda row: " ".join(row).lower(), axis=1)
    for kw in risk_keywords:
        feats[f"kw_{kw}"] = all_text_series.str.contains(kw, regex=False).astype(np.float32)

    # Categorical interaction pairs across prominent categorical features
    n_cat = min(len(cat_cols), 8)
    if n_cat >= 2:
        for i in range(n_cat):
            for j in range(i + 1, n_cat):
                c1, c2 = cat_cols[i], cat_cols[j]
                feats[f"{c1}__x__{c2}"] = (
                    df[c1].astype(str).fillna("NA")
                    + "___"
                    + df[c2].astype(str).fillna("NA")
                )

    return feats


X_train_tab = extract_tabular_features(X_train_raw)
X_test_tab = extract_tabular_features(X_test_raw)

inter_cols = [c for c in X_train_tab.columns if "__x__" in c]
all_cat_features = cat_cols + inter_cols

# Frequency Encoding
for col in all_cat_features:
    if col in X_train_raw.columns:
        freq_map = X_train_raw[col].astype(str).value_counts(normalize=True).to_dict()
        X_train_tab[f"{col}_freq"] = (
            X_train_raw[col].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        )
        X_test_tab[f"{col}_freq"] = (
            X_test_raw[col].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        )
    elif col in X_train_tab.columns:
        freq_map = X_train_tab[col].astype(str).value_counts(normalize=True).to_dict()
        X_train_tab[f"{col}_freq"] = (
            X_train_tab[col].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        )
        X_test_tab[f"{col}_freq"] = (
            X_test_tab[col].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        )

# Categorical Label Encoding for Trees
for col in cat_cols + inter_cols:
    train_vals = (
        X_train_raw[col].astype(str)
        if col in X_train_raw.columns
        else X_train_tab[col].astype(str)
    )
    test_vals = (
        X_test_raw[col].astype(str)
        if col in X_test_raw.columns
        else X_test_tab[col].astype(str)
    )

    le = LabelEncoder()
    le.fit(train_vals)
    cat_map = {val: idx for idx, val in enumerate(le.classes_)}
    X_train_tab[f"{col}_cat_code"] = train_vals.map(cat_map).fillna(-1).astype(np.int32)
    X_test_tab[f"{col}_cat_code"] = test_vals.map(cat_map).fillna(-1).astype(np.int32)

drop_raw_strings = [c for c in inter_cols if c in X_train_tab.columns]
X_train_tab.drop(columns=drop_raw_strings, inplace=True)
X_test_tab.drop(columns=drop_raw_strings, inplace=True)

# Text Feature Engineering (TF-IDF + SVD 32 + OOF Linear Meta-Features)
text_feat_dfs_train = []
text_feat_dfs_test = []

text_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
text_fold_indices = list(text_skf.split(X_train_raw, y))

target_text_cols = list(text_cols)
if len(text_cols) > 1:
    X_train_raw["_all_text_concat"] = X_train_raw[text_cols].astype(str).agg(" ".join, axis=1)
    X_test_raw["_all_text_concat"] = X_test_raw[text_cols].astype(str).agg(" ".join, axis=1)
    target_text_cols.append("_all_text_concat")

for t_col in target_text_cols:
    train_text = X_train_raw[t_col].fillna("").astype(str)
    test_text = X_test_raw[t_col].fillna("").astype(str)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
    )
    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    n_comp = min(32, max(2, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)

    svd_cols = [f"{t_col}_svd_{i}" for i in range(n_comp)]
    text_feat_dfs_train.append(
        pd.DataFrame(train_svd, columns=svd_cols, index=X_train_raw.index).astype(np.float32)
    )
    text_feat_dfs_test.append(
        pd.DataFrame(test_svd, columns=svd_cols, index=X_test_raw.index).astype(np.float32)
    )

    oof_lr_probs = np.zeros((len(train_text), num_classes), dtype=np.float32)
    test_lr_probs = np.zeros((len(test_text), num_classes), dtype=np.float32)

    for tr_idx, val_idx in text_fold_indices:
        clf = LogisticRegression(C=1.0, max_iter=250, solver="lbfgs", random_state=42)
        clf.fit(train_tfidf[tr_idx], y[tr_idx])
        oof_lr_probs[val_idx] = clf.predict_proba(train_tfidf[val_idx])
        test_lr_probs += clf.predict_proba(test_tfidf) / len(text_fold_indices)

    lr_cols = [f"{t_col}_lr_prob_{k}" for k in range(num_classes)]
    text_feat_dfs_train.append(
        pd.DataFrame(oof_lr_probs, columns=lr_cols, index=X_train_raw.index).astype(np.float32)
    )
    text_feat_dfs_test.append(
        pd.DataFrame(test_lr_probs, columns=lr_cols, index=X_test_raw.index).astype(np.float32)
    )

if "_all_text_concat" in X_train_raw.columns:
    X_train_raw.drop(columns=["_all_text_concat"], inplace=True)
    X_test_raw.drop(columns=["_all_text_concat"], inplace=True)

if text_feat_dfs_train:
    X_train_text = pd.concat(text_feat_dfs_train, axis=1)
    X_test_text = pd.concat(text_feat_dfs_test, axis=1)
    X_train_combined = pd.concat([X_train_tab, X_train_text], axis=1)
    X_test_combined = pd.concat([X_test_tab, X_test_text], axis=1)
else:
    X_train_combined = X_train_tab.copy()
    X_test_combined = X_test_tab.copy()

common_cols = [c for c in X_train_combined.columns if c in X_test_combined.columns]
X_train_combined = X_train_combined[common_cols]
X_test_combined = X_test_combined[common_cols]

clean_col_map = {c: re.sub(r"[\[\]<>, \'\"]", "_", str(c)) for c in common_cols}
X_train_combined.rename(columns=clean_col_map, inplace=True)
X_test_combined.rename(columns=clean_col_map, inplace=True)

# ---------------------------------------------------------
# 3. Stratified Folds & Out-of-Fold Target Encoding
# ---------------------------------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_indices = list(skf.split(X_train_combined, y))

te_candidate_cols = [
    c for c in cat_cols
    if c in X_train_raw.columns and 1 < X_train_raw[c].nunique() < 500
]

for col in te_candidate_cols:
    for cls_idx in range(num_classes):
        target_binary = (y == cls_idx).astype(float)
        col_name = f"{col}_te_class_{cls_idx}"
        X_train_combined[col_name] = 0.0
        X_test_combined[col_name] = 0.0

        global_mean = target_binary.mean()

        for train_idx, val_idx in fold_indices:
            tr_cat = X_train_raw[col].iloc[train_idx].astype(str)
            tr_y = target_binary[train_idx]
            val_cat = X_train_raw[col].iloc[val_idx].astype(str)

            stats = (
                pd.DataFrame({"cat": tr_cat, "target": tr_y})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            smoothing = 10.0
            smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            te_dict = smoothed.to_dict()

            X_train_combined.loc[X_train_combined.index[val_idx], col_name] = (
                val_cat.map(te_dict).fillna(global_mean).values.astype(np.float32)
            )

        full_cat = X_train_raw[col].astype(str)
        stats = (
            pd.DataFrame({"cat": full_cat, "target": target_binary})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smoothing = 10.0
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        te_dict = smoothed.to_dict()
        X_test_combined[col_name] = (
            X_test_raw[col]
            .astype(str)
            .map(te_dict)
            .fillna(global_mean)
            .values.astype(np.float32)
        )

# Imputation and Scaling
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_combined),
    columns=X_train_combined.columns,
    index=X_train_combined.index,
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_combined),
    columns=X_test_combined.columns,
    index=X_test_combined.index,
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),
    columns=X_train_combined.columns,
    index=X_train_combined.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed),
    columns=X_test_combined.columns,
    index=X_test_combined.index,
)


# ---------------------------------------------------------
# 4. Neural Network Architecture & Model Configurations
# ---------------------------------------------------------
class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super(GatedResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        g = torch.sigmoid(self.gate(x))
        h = self.act(h * g)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(residual + h)


class GatedTabularNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super(GatedTabularNet, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            if h_dim != current_dim:
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(current_dim, h_dim),
                        nn.LayerNorm(h_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout_rate),
                    )
                )
                current_dim = h_dim
            self.blocks.append(GatedResidualBlock(current_dim, dropout_rate))

        self.classifier = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.LayerNorm(current_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(current_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.classifier(h)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_cls = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target)
        return (1.0 - self.smoothing) * nll + self.smoothing * (loss / n_cls)


def build_neural_optimizer_and_scheduler(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4, num_epochs: int = 35
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    return optimizer, scheduler, criterion


def get_lgb_model_params(num_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(num_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_model_params(num_classes: int, seed: int = 42) -> dict:
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# ---------------------------------------------------------
# 5. Model Training and Cross-Validation
# ---------------------------------------------------------
n_samples = len(y)
n_test = len(X_test_imputed)
input_dim = X_train_scaled.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_lgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_xgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_nn = np.zeros((n_samples, num_classes), dtype=np.float32)

test_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_nn = np.zeros((n_test, num_classes), dtype=np.float32)

if HAS_CATBOOST:
    oof_cb = np.zeros((n_samples, num_classes), dtype=np.float32)
    test_cb = np.zeros((n_test, num_classes), dtype=np.float32)

X_tr_sc_np = X_train_scaled.values.astype(np.float32)
X_te_sc_np = X_test_scaled.values.astype(np.float32)
test_tensor = torch.tensor(X_te_sc_np, dtype=torch.float32).to(device)

for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
    X_tr_tab, y_tr = X_train_imputed.iloc[train_idx], y[train_idx]
    X_val_tab, y_val = X_train_imputed.iloc[val_idx], y[val_idx]

    # 1. LightGBM
    lgb_params = get_lgb_model_params(num_classes=num_classes, seed=42 + fold_idx)
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_tr_tab,
        y_tr,
        eval_set=[(X_val_tab, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_clf.predict_proba(X_val_tab)
    test_lgb += lgb_clf.predict_proba(X_test_imputed) / len(fold_indices)

    # 2. XGBoost
    xgb_params = get_xgb_model_params(num_classes=num_classes, seed=42 + fold_idx)
    xgb_clf = xgb.XGBClassifier(early_stopping_rounds=50, **xgb_params)
    xgb_clf.fit(X_tr_tab, y_tr, eval_set=[(X_val_tab, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_val_tab)
    test_xgb += xgb_clf.predict_proba(X_test_imputed) / len(fold_indices)

    # 3. CatBoost
    if HAS_CATBOOST:
        cb_params = get_catboost_model_params(
            num_classes=num_classes, seed=42 + fold_idx
        )
        cb_clf = cb.CatBoostClassifier(early_stopping_rounds=50, **cb_params)
        cb_clf.fit(X_tr_tab, y_tr, eval_set=(X_val_tab, y_val), verbose=False)
        oof_cb[val_idx] = cb_clf.predict_proba(X_val_tab)
        test_cb += cb_clf.predict_proba(X_test_imputed) / len(fold_indices)

    # 4. GatedTabularNet
    X_tr_nn, y_tr_nn = (
        torch.tensor(X_tr_sc_np[train_idx], dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    X_val_nn, y_val_nn = (
        torch.tensor(X_tr_sc_np[val_idx], dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        TensorDataset(X_tr_nn, y_tr_nn),
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )

    nn_model = GatedTabularNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
    ).to(device)

    num_epochs = 35
    optimizer, scheduler, criterion = build_neural_optimizer_and_scheduler(
        nn_model, lr=1.5e-3, weight_decay=1e-4, num_epochs=num_epochs
    )

    best_val_loss = float("inf")
    best_nn_val_probs = None
    patience = 8
    patience_counter = 0

    X_val_nn_dev = X_val_nn.to(device)
    y_val_nn_dev = y_val_nn.to(device)

    for epoch in range(num_epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(X_val_nn_dev)
            val_loss = F.cross_entropy(val_logits, y_val_nn_dev).item()
            val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_val_probs = val_probs
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    oof_nn[val_idx] = best_nn_val_probs

    nn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    nn_model.eval()
    with torch.no_grad():
        test_nn_logits = nn_model(test_tensor)
        test_nn_fold_probs = F.softmax(test_nn_logits, dim=-1).cpu().numpy() / len(
            fold_indices
        )
        test_nn += test_nn_fold_probs

# ---------------------------------------------------------
# 6. Ensemble Optimization & Submission Generation
# ---------------------------------------------------------
if HAS_CATBOOST:
    oof_models = [oof_lgb, oof_xgb, oof_cb, oof_nn]
    test_models = [test_lgb, test_xgb, test_cb, test_nn]
    init_weights = [0.35, 0.35, 0.20, 0.10]
else:
    oof_models = [oof_lgb, oof_xgb, oof_nn]
    test_models = [test_lgb, test_xgb, test_nn]
    init_weights = [0.45, 0.45, 0.10]


def ensemble_loss(weights):
    weights = np.maximum(weights, 0)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return 999.0
    w_norm = weights / w_sum

    blend_oof = sum(w * m for w, m in zip(w_norm, oof_models))
    eps = 1e-15
    blend_oof = np.clip(blend_oof, eps, 1.0 - eps)
    row_losses = -np.log(blend_oof[np.arange(len(y)), y])
    return np.mean(row_losses)


res = minimize(
    ensemble_loss,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 500, "disp": False},
)

optimal_weights = np.maximum(res.x, 0)
optimal_weights = optimal_weights / np.sum(optimal_weights)

oof_final_probs = sum(w * m for w, m in zip(optimal_weights, oof_models))
oof_final_preds = np.argmax(oof_final_probs, axis=1)
final_score = accuracy_score(y, oof_final_preds)

test_final_probs = sum(w * m for w, m in zip(optimal_weights, test_models))
test_final_preds = np.argmax(test_final_probs, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_final_preds)

sub_df = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "risk_decision": test_pred_labels}
)

submission_path = "./submission/submission.csv"
sub_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")
