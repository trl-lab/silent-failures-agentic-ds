import os
import re
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor

import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Ensure required output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Meta Critic"
y_train = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

id_col = "id" if "id" in X_train_raw.columns else None
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

# Identify column types dynamically
text_cols = [
    c
    for c in X_train_raw.columns
    if c != id_col
    and X_train_raw[c].dtype == "object"
    and X_train_raw[c].str.len().mean() > 30
]
cat_cols = [
    c
    for c in X_train_raw.columns
    if c != id_col and X_train_raw[c].dtype == "object" and c not in text_cols
]
num_cols = [
    c
    for c in X_train_raw.columns
    if c != id_col and np.issubdtype(X_train_raw[c].dtype, np.number)
]

# Flavor Lexicon Dictionary
FLAVOR_LEXICONS = {
    "flavor_peat_smoke": [
        "peat",
        "peated",
        "smoke",
        "smoky",
        "ash",
        "campfire",
        "phenol",
        "iodine",
        "medicinal",
        "charcoal",
        "turf",
        "tar",
    ],
    "flavor_fruit_citrus": [
        "apple",
        "pear",
        "lemon",
        "lime",
        "orange",
        "citrus",
        "raisin",
        "fig",
        "date",
        "plum",
        "berry",
        "cherry",
        "apricot",
        "peach",
    ],
    "flavor_sweet_dessert": [
        "vanilla",
        "caramel",
        "toffee",
        "honey",
        "butterscotch",
        "chocolate",
        "cocoa",
        "maple",
        "sugar",
        "marzipan",
        "fudge",
        "cream",
    ],
    "flavor_wood_spice": [
        "oak",
        "cask",
        "sherry",
        "bourbon",
        "spice",
        "cinnamon",
        "nutmeg",
        "clove",
        "pepper",
        "ginger",
        "tannin",
        "wood",
    ],
    "flavor_floral_herbal": [
        "heather",
        "grass",
        "grassy",
        "herbal",
        "floral",
        "lavender",
        "mint",
        "tea",
        "eucalyptus",
        "pine",
    ],
    "flavor_nutty_grainy": [
        "nut",
        "almond",
        "hazelnut",
        "walnut",
        "malt",
        "cereal",
        "grain",
        "toast",
        "bread",
        "biscuit",
    ],
    "flavor_body_finish": [
        "smooth",
        "complex",
        "rich",
        "full",
        "long",
        "intense",
        "balanced",
        "subtle",
        "oily",
        "velvety",
    ],
}


def extract_handcrafted_features(df_input):
    df = df_input.copy()

    formatted_rows = []
    for idx, row in df.iterrows():
        parts = []
        for col in df.columns:
            if col == id_col:
                continue
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                parts.append(f"{col}: {val}")
        formatted_rows.append(". ".join(parts))
    combined_text = pd.Series(formatted_rows, index=df.index)

    features = pd.DataFrame(index=df.index)

    features["text_char_len"] = combined_text.str.len()
    features["text_word_count"] = combined_text.str.split().str.len()
    features["text_avg_word_len"] = features["text_char_len"] / (
        features["text_word_count"] + 1e-5
    )

    lowered = combined_text.str.lower()
    for flavor_name, lexicon in FLAVOR_LEXICONS.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in lexicon) + r")\b"
        features[flavor_name] = lowered.apply(lambda txt: len(re.findall(pattern, txt)))
        features[f"{flavor_name}_density"] = features[flavor_name] / (
            features["text_word_count"] + 1e-5
        )

    features["regex_age"] = lowered.apply(
        lambda txt: [int(m) for m in re.findall(r"(\d+)\s*(?:year|yr|yo|y\.o\.)", txt)]
    ).apply(lambda lst: lst[0] if lst else np.nan)
    features["regex_abv"] = lowered.apply(
        lambda txt: [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", txt)]
    ).apply(lambda lst: lst[0] if lst else np.nan)
    features["is_cask_strength"] = lowered.apply(
        lambda txt: (
            1
            if any(
                k in txt for k in ["cask strength", "cs", "barrel proof", "single cask"]
            )
            else 0
        )
    )
    features["is_sherry_cask"] = lowered.apply(lambda txt: 1 if "sherry" in txt else 0)
    features["is_peat_heavy"] = lowered.apply(
        lambda txt: 1 if any(k in txt for k in ["peat", "islay", "smoky"]) else 0
    )

    for col in num_cols:
        features[col] = df[col]

    if "Price" in features.columns and "Age" in features.columns:
        features["price_per_age"] = features["Price"] / (
            features["Age"].fillna(0) + 1.0
        )
        features["log_price"] = np.log1p(np.maximum(features["Price"].fillna(0), 0))

    if "ABV" in features.columns and "Age" in features.columns:
        features["abv_age_prod"] = features["ABV"] * features["Age"]

    return features, combined_text


# Feature extraction
train_handcrafted, train_texts = extract_handcrafted_features(X_train_raw)
test_handcrafted, test_texts = extract_handcrafted_features(X_test_raw)

# Group-based summary statistics (Price/ABV mean/std by category)
for col in cat_cols:
    for num_c in ["Price", "ABV"]:
        if num_c in X_train_raw.columns:
            grp_mean = X_train_raw.groupby(col)[num_c].transform("mean")
            grp_std = X_train_raw.groupby(col)[num_c].transform("std").fillna(0)
            train_handcrafted[f"{col}_{num_c}_grp_mean"] = grp_mean.values
            train_handcrafted[f"{col}_{num_c}_grp_std"] = grp_std.values

            mean_map = X_train_raw.groupby(col)[num_c].mean().to_dict()
            std_map = X_train_raw.groupby(col)[num_c].std().fillna(0).to_dict()
            test_handcrafted[f"{col}_{num_c}_grp_mean"] = X_test_raw[col].map(mean_map).fillna(X_train_raw[num_c].mean()).values
            test_handcrafted[f"{col}_{num_c}_grp_std"] = X_test_raw[col].map(std_map).fillna(0).values

# Set up 10-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_indices = list(kf.split(train_texts, y_train))

# TF-IDF n-grams with Out-Of-Fold Ridge predictions
tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
tfidf_tr = tfidf_vec.fit_transform(train_texts)
tfidf_te = tfidf_vec.transform(test_texts)

oof_tfidf_ridge = np.zeros(len(y_train))
test_tfidf_ridge = np.zeros(len(test_df))

for tr_idx, val_idx in fold_indices:
    r_model = Ridge(alpha=1.0)
    r_model.fit(tfidf_tr[tr_idx], y_train[tr_idx])
    oof_tfidf_ridge[val_idx] = r_model.predict(tfidf_tr[val_idx])
    test_tfidf_ridge += r_model.predict(tfidf_te) / 10.0

train_handcrafted["tfidf_ridge_pred"] = oof_tfidf_ridge
test_handcrafted["tfidf_ridge_pred"] = test_tfidf_ridge

# Feature extraction: Text Embeddings using 'sentence-transformers/all-mpnet-base-v2'
try:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    train_embeds = embedder.encode(
        train_texts.tolist(), batch_size=32, show_progress_bar=False
    )
    test_embeds = embedder.encode(
        test_texts.tolist(), batch_size=32, show_progress_bar=False
    )
except Exception:
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_train = tfidf.fit_transform(train_texts)
    tfidf_test = tfidf.transform(test_texts)
    svd = TruncatedSVD(n_components=64, random_state=42)
    train_embeds = svd.fit_transform(tfidf_train)
    test_embeds = svd.transform(tfidf_test)

# Construct Out-Of-Fold Ridge target predictions from text embeddings
oof_text_ridge = np.zeros(len(y_train))
test_text_ridge = np.zeros(len(test_df))

for tr_idx, val_idx in fold_indices:
    emb_ridge = Ridge(alpha=1.0)
    emb_ridge.fit(train_embeds[tr_idx], y_train[tr_idx])
    oof_text_ridge[val_idx] = emb_ridge.predict(train_embeds[val_idx])
    test_text_ridge += emb_ridge.predict(test_embeds) / 10.0

train_handcrafted["text_ridge_pred"] = oof_text_ridge
test_handcrafted["text_ridge_pred"] = test_text_ridge

# Compress embeddings via SVD to 16 dimensions
svd = TruncatedSVD(n_components=16, random_state=42)
train_embeds_svd = svd.fit_transform(train_embeds)
test_embeds_svd = svd.transform(test_embeds)

embed_cols = [f"emb_svd_{i}" for i in range(16)]
train_embed_df = pd.DataFrame(
    train_embeds_svd, columns=embed_cols, index=train_handcrafted.index
)
test_embed_df = pd.DataFrame(
    test_embeds_svd, columns=embed_cols, index=test_handcrafted.index
)

X_train_all = pd.concat([train_handcrafted, train_embed_df], axis=1)
X_test_all = pd.concat([test_handcrafted, test_embed_df], axis=1)

# Categorical Frequency Encoding & Out-Of-Fold Target Encoding
for col in cat_cols:
    freq = X_train_raw[col].value_counts(normalize=True).to_dict()
    X_train_all[f"{col}_freq"] = X_train_raw[col].map(freq).fillna(0)
    X_test_all[f"{col}_freq"] = X_test_raw[col].map(freq).fillna(0)

    X_train_all[f"{col}_te"] = np.nan
    global_mean = y_train.mean()
    te_col_idx = X_train_all.columns.get_loc(f"{col}_te")

    for train_idx, val_idx in fold_indices:
        tr_cat = X_train_raw[col].iloc[train_idx].reset_index(drop=True)
        tr_y = pd.Series(y_train[train_idx]).reset_index(drop=True)
        group = tr_y.groupby(tr_cat).agg(["mean", "count"])
        smoothed = (group["count"] * group["mean"] + 10 * global_mean) / (
            group["count"] + 10
        )
        mapped_val = (
            X_train_raw[col].iloc[val_idx].map(smoothed).fillna(global_mean).values
        )
        X_train_all.iloc[val_idx, te_col_idx] = mapped_val

    full_group = pd.Series(y_train).groupby(X_train_raw[col]).agg(["mean", "count"])
    smoothed_full = (full_group["count"] * full_group["mean"] + 10 * global_mean) / (
        full_group["count"] + 10
    )
    X_test_all[f"{col}_te"] = X_test_raw[col].map(smoothed_full).fillna(global_mean)

# Clean, Impute & Standard Scale
feature_cols = [c for c in X_train_all.columns if c != id_col]
X_tr_mat = X_train_all[feature_cols].copy()
X_te_mat = X_test_all[feature_cols].copy()

medians = X_tr_mat.median()
X_tr_mat = X_tr_mat.fillna(medians).fillna(0)
X_te_mat = X_te_mat.fillna(medians).fillna(0)

X_train = X_tr_mat.values
X_test = X_te_mat.values


# ==============================================================================
# PyTorch Model Architecture: WhiskyGatedNet
# ==============================================================================


class FeatureGatingBlock(nn.Module):

    def __init__(self, in_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class DenseResidualBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.dropout1(self.act1(self.norm1(self.linear1(x))))
        out = self.dropout2(self.act2(self.norm2(self.linear2(out))))
        return out + residual


class WhiskyGatedNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_gate = FeatureGatingBlock(input_dim)
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(DenseResidualBlock(curr_dim, h_dim, dropout=dropout))
            curr_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(curr_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gated = self.input_gate(x)
        feat = self.backbone(x_gated)
        out = self.head(feat)
        return out.squeeze(-1)


def train_pytorch_model(
    model, X_tr, y_tr, X_val, y_val, X_te, epochs=100, batch_size=32, lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_rmse = float("inf")
    best_val_preds = None
    best_test_preds = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_tensor).cpu().numpy()
            val_rmse = np.sqrt(np.mean((y_val - val_out) ** 2))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_val_preds = val_out.copy()
                best_test_preds = model(X_te_tensor).cpu().numpy().copy()

    return best_val_preds, best_test_preds, best_val_rmse


# ==============================================================================
# 10-Fold Cross Validation Loop (LightGBM, XGBoost, CatBoost, WhiskyGatedNet, BayesianRidge, SVR, ExtraTrees)
# ==============================================================================

n_train = len(y_train)
n_test = len(X_test)
n_folds = len(fold_indices)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)
oof_nn = np.zeros(n_train)
oof_bridge = np.zeros(n_train)
oof_svr = np.zeros(n_train)
oof_et = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cb = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)
test_preds_bridge = np.zeros(n_test)
test_preds_svr = np.zeros(n_test)
test_preds_et = np.zeros(n_test)

for fold_idx, (tr_idx, val_idx) in enumerate(fold_indices):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[tr_idx])
    X_val = scaler.transform(X_train[val_idx])
    X_te_fold = scaler.transform(X_test)
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # --- A. LightGBM (Shallow depth = 4) ---
    lgb_model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold_idx,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_te_fold) / n_folds

    # --- B. XGBoost (Shallow depth = 4) ---
    xgb_model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold_idx,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_te_fold) / n_folds

    # --- C. CatBoost (Shallow depth = 4) ---
    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=42 + fold_idx,
        verbose=0,
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )
    oof_cb[val_idx] = cb_model.predict(X_val)
    test_preds_cb += cb_model.predict(X_te_fold) / n_folds

    # --- D. WhiskyGatedNet (PyTorch) ---
    net = WhiskyGatedNet(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64])
    nn_val_preds, nn_test_preds, _ = train_pytorch_model(
        net, X_tr, y_tr, X_val, y_val, X_te_fold, epochs=100, batch_size=32, lr=1e-3
    )
    oof_nn[val_idx] = nn_val_preds
    test_preds_nn += nn_test_preds / n_folds

    # --- E. Bayesian Ridge ---
    bridge_model = BayesianRidge()
    bridge_model.fit(X_tr, y_tr)
    oof_bridge[val_idx] = bridge_model.predict(X_val)
    test_preds_bridge += bridge_model.predict(X_te_fold) / n_folds

    # --- F. Support Vector Regressor (RBF) ---
    svr_model = SVR(C=1.0, kernel="rbf", epsilon=0.1)
    svr_model.fit(X_tr, y_tr)
    oof_svr[val_idx] = svr_model.predict(X_val)
    test_preds_svr += svr_model.predict(X_te_fold) / n_folds

    # --- G. ExtraTrees Regressor (Shallow depth = 4) ---
    et_model = ExtraTreesRegressor(n_estimators=300, max_depth=4, random_state=42 + fold_idx)
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict(X_val)
    test_preds_et += et_model.predict(X_te_fold) / n_folds


# ==============================================================================
# Out-Of-Fold Ensembling & SLSQP Weight Optimization
# ==============================================================================


def rmse_objective(weights):
    w = weights / np.sum(weights)
    blend_oof = (
        w[0] * oof_lgb
        + w[1] * oof_xgb
        + w[2] * oof_cb
        + w[3] * oof_nn
        + w[4] * oof_bridge
        + w[5] * oof_svr
        + w[6] * oof_et
    )
    return np.sqrt(np.mean((y_train - blend_oof) ** 2))


init_weights = np.array([1/7.0] * 7)
bounds = [(0, 1)] * 7
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = opt.minimize(
    rmse_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

if res.success and np.sum(res.x) > 0:
    opt_w = res.x / np.sum(res.x)
else:
    opt_w = init_weights

oof_final = (
    opt_w[0] * oof_lgb
    + opt_w[1] * oof_xgb
    + opt_w[2] * oof_cb
    + opt_w[3] * oof_nn
    + opt_w[4] * oof_bridge
    + opt_w[5] * oof_svr
    + opt_w[6] * oof_et
)
test_final = (
    opt_w[0] * test_preds_lgb
    + opt_w[1] * test_preds_xgb
    + opt_w[2] * test_preds_cb
    + opt_w[3] * test_preds_nn
    + opt_w[4] * test_preds_bridge
    + opt_w[5] * test_preds_svr
    + opt_w[6] * test_preds_et
)

# Constrain target within training bounds
y_min, y_max = np.min(y_train), np.max(y_train)
oof_final = np.clip(oof_final, y_min, y_max)
test_final = np.clip(test_final, y_min, y_max)

final_score = np.sqrt(np.mean((y_train - oof_final) ** 2))

# Write final submission file
submission_df = pd.DataFrame({"id": test_ids, "Meta Critic": test_final})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score:.4f}")