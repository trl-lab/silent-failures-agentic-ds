import copy
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# 1. Load Datasets & Validation Split
# -------------------------------------------------------------------------
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

if "id" in test_raw.columns:
    test_ids = test_raw["id"].values
elif "test_id" in test_raw.columns:
    test_ids = test_raw["test_id"].values
else:
    test_ids = np.arange(len(test_raw))

# Strict validation split prior to stateful processing
train_df, val_df = train_test_split(
    train_raw, test_size=0.15, random_state=42, shuffle=True
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_raw.reset_index(drop=True)

y_train = train_df["price"].values.astype(np.float32)
y_val = val_df["price"].values.astype(np.float32)
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)


# -------------------------------------------------------------------------
# 2. Data Cleaning & Hierarchical Category Extraction
# -------------------------------------------------------------------------
def clean_and_split_categories(df):
    df_clean = df.copy()

    df_clean["category_name"] = (
        df_clean["category_name"].fillna("missing/missing/missing").astype(str)
    )
    df_clean["brand_name"] = (
        df_clean["brand_name"].fillna("missing").astype(str).str.strip().str.lower()
    )
    df_clean["name"] = df_clean["name"].fillna("missing").astype(str)
    df_clean["item_description"] = (
        df_clean["item_description"].fillna("missing").astype(str)
    )
    df_clean["shipping"] = df_clean["shipping"].fillna(0).astype(int)
    df_clean["item_condition_id"] = df_clean["item_condition_id"].fillna(3).astype(int)

    def split_cat(val):
        parts = val.split("/")
        c1 = parts[0].strip().lower() if len(parts) > 0 else "missing"
        c2 = parts[1].strip().lower() if len(parts) > 1 else "missing"
        c3 = parts[2].strip().lower() if len(parts) > 2 else "missing"
        return c1, c2, c3

    cats = [split_cat(c) for c in df_clean["category_name"]]
    df_clean["cat_main"] = [c[0] for c in cats]
    df_clean["cat_sub1"] = [c[1] for c in cats]
    df_clean["cat_sub2"] = [c[2] for c in cats]

    return df_clean


train_clean = clean_and_split_categories(train_df)
val_clean = clean_and_split_categories(val_df)
test_clean = clean_and_split_categories(test_df)


# -------------------------------------------------------------------------
# 3. Dense Feature Engineering
# -------------------------------------------------------------------------
def extract_dense_features(df):
    feats = pd.DataFrame(index=df.index)

    feats["item_condition_id"] = df["item_condition_id"]
    feats["shipping"] = df["shipping"]

    name_str = df["name"]
    desc_str = df["item_description"]

    feats["name_char_len"] = name_str.str.len()
    feats["name_word_count"] = name_str.apply(lambda s: len(s.split()))
    feats["name_upper_count"] = name_str.apply(
        lambda s: sum(1 for c in s if c.isupper())
    )
    feats["name_digit_count"] = name_str.apply(
        lambda s: sum(1 for c in s if c.isdigit())
    )
    feats["name_upper_ratio"] = feats["name_upper_count"] / (
        feats["name_char_len"] + 1e-5
    )

    feats["desc_char_len"] = desc_str.str.len()
    feats["desc_word_count"] = desc_str.apply(lambda s: len(s.split()))
    feats["desc_upper_count"] = desc_str.apply(
        lambda s: sum(1 for c in s if c.isupper())
    )
    feats["desc_digit_count"] = desc_str.apply(
        lambda s: sum(1 for c in s if c.isdigit())
    )
    feats["desc_punct_count"] = desc_str.apply(
        lambda s: len(re.findall(r"[!?,.:;]", s))
    )
    feats["desc_upper_ratio"] = feats["desc_upper_count"] / (
        feats["desc_char_len"] + 1e-5
    )

    feats["has_brand"] = (df["brand_name"] != "missing").astype(int)
    feats["no_desc_flag"] = (
        (df["item_description"] == "missing")
        | (df["item_description"].str.lower().str.contains("no description yet"))
    ).astype(int)

    name_lower = name_str.str.lower()
    desc_lower = desc_str.str.lower()
    full_text_lower = name_lower + " " + desc_lower
    brand_lower = df["brand_name"]

    # Regex-based value indicators
    bundle_pattern = re.compile(r"\b(bundle|lot|pack|set|pcs|pieces|pair|pairs)\b")
    nwt_pattern = re.compile(r"\b(nwt|nwot|nib|bnib|new with tag|brand new|never worn|mint)\b")
    authentic_pattern = re.compile(r"\b(authentic|100%|genuine|real|guaranteed|original)\b")
    material_pattern = re.compile(r"\b(gold|silver|diamond|leather|silk|cashmere|14k|18k|10k|925|sterling|platinum)\b")
    vintage_pattern = re.compile(r"\b(vintage|retro|antique|y2k|deadstock|rare|collectible)\b")

    feats["is_bundle"] = full_text_lower.apply(lambda s: 1 if bundle_pattern.search(s) else 0)
    feats["is_nwt"] = full_text_lower.apply(lambda s: 1 if nwt_pattern.search(s) else 0)
    feats["is_authentic"] = full_text_lower.apply(lambda s: 1 if authentic_pattern.search(s) else 0)
    feats["is_premium_material"] = full_text_lower.apply(lambda s: 1 if material_pattern.search(s) else 0)
    feats["is_vintage"] = full_text_lower.apply(lambda s: 1 if vintage_pattern.search(s) else 0)

    brand_in_name = []
    brand_in_desc = []
    word_overlap = []

    for n, d, b in zip(name_lower, desc_lower, brand_lower):
        if b != "missing" and len(b) > 1:
            brand_in_name.append(1 if b in n else 0)
            brand_in_desc.append(1 if b in d else 0)
        else:
            brand_in_name.append(0)
            brand_in_desc.append(0)

        n_words = set(n.split())
        d_words = set(d.split())
        word_overlap.append(len(n_words.intersection(d_words)))

    feats["brand_in_name"] = brand_in_name
    feats["brand_in_desc"] = brand_in_desc
    feats["name_desc_word_overlap"] = word_overlap

    return feats


train_dense_df = extract_dense_features(train_clean)
val_dense_df = extract_dense_features(val_clean)
test_dense_df = extract_dense_features(test_clean)

# -------------------------------------------------------------------------
# 4. Out-of-Fold Target & Frequency Encoding (Fitted Strictly on Train)
# -------------------------------------------------------------------------
te_cols = ["brand_name", "cat_main", "cat_sub1", "cat_sub2", "category_name"]
global_mean = np.mean(y_train_log)
smoothing_weight = 10.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in te_cols:
    train_encoded = np.full(len(train_clean), fill_value=global_mean, dtype=np.float32)

    for tr_idx, oof_idx in kf.split(train_clean):
        tr_subset = train_clean.iloc[tr_idx]
        y_subset = y_train_log[tr_idx]

        stats = (
            pd.DataFrame({"cat": tr_subset[col], "target": y_subset})
            .groupby("cat")
            .agg(["count", "mean"])["target"]
        )
        counts = stats["count"]
        means = stats["mean"]
        smoothed = (counts * means + smoothing_weight * global_mean) / (
            counts + smoothing_weight
        )

        oof_cats = train_clean.iloc[oof_idx][col]
        train_encoded[oof_idx] = oof_cats.map(smoothed).fillna(global_mean).values

    train_dense_df[f"{col}_te"] = train_encoded

    full_stats = (
        pd.DataFrame({"cat": train_clean[col], "target": y_train_log})
        .groupby("cat")
        .agg(["count", "mean"])["target"]
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing_weight * global_mean
    ) / (full_stats["count"] + smoothing_weight)

    val_dense_df[f"{col}_te"] = (
        val_clean[col].map(full_smoothed).fillna(global_mean).values.astype(np.float32)
    )
    test_dense_df[f"{col}_te"] = (
        test_clean[col].map(full_smoothed).fillna(global_mean).values.astype(np.float32)
    )

for col in te_cols:
    freq_map = train_clean[col].value_counts(normalize=True).to_dict()
    train_dense_df[f"{col}_freq"] = (
        train_clean[col].map(freq_map).fillna(0.0).values.astype(np.float32)
    )
    val_dense_df[f"{col}_freq"] = (
        val_clean[col].map(freq_map).fillna(0.0).values.astype(np.float32)
    )
    test_dense_df[f"{col}_freq"] = (
        test_clean[col].map(freq_map).fillna(0.0).values.astype(np.float32)
    )

# -------------------------------------------------------------------------
# 5. One-Hot Encoding, Dense Scaling & Text TF-IDF Vectorization
# -------------------------------------------------------------------------
ohe_cols = ["item_condition_id", "shipping", "cat_main"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
train_ohe = ohe.fit_transform(train_clean[ohe_cols])
val_ohe = ohe.transform(val_clean[ohe_cols])
test_ohe = ohe.transform(test_clean[ohe_cols])

scaler = StandardScaler()
train_dense_scaled = scaler.fit_transform(train_dense_df.values)
val_dense_scaled = scaler.transform(val_dense_df.values)
test_dense_scaled = scaler.transform(test_dense_df.values)

tfidf_name = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40000,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
    lowercase=True,
)
train_name_tfidf = tfidf_name.fit_transform(train_clean["name"])
val_name_tfidf = tfidf_name.transform(val_clean["name"])
test_name_tfidf = tfidf_name.transform(test_clean["name"])

tfidf_desc = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=60000,
    sublinear_tf=True,
    stop_words="english",
    lowercase=True,
)
train_desc_tfidf = tfidf_desc.fit_transform(train_clean["item_description"])
val_desc_tfidf = tfidf_desc.transform(val_clean["item_description"])
test_desc_tfidf = tfidf_desc.transform(test_clean["item_description"])

tfidf_cat = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w-]+\b",
    max_features=2000,
    sublinear_tf=True,
    lowercase=True,
)
train_cat_tfidf = tfidf_cat.fit_transform(train_clean["category_name"])
val_cat_tfidf = tfidf_cat.transform(val_clean["category_name"])
test_cat_tfidf = tfidf_cat.transform(test_clean["category_name"])

tfidf_brand = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w\.\'-]+\b", max_features=5000, binary=True, lowercase=True
)
train_brand_tfidf = tfidf_brand.fit_transform(train_clean["brand_name"])
val_brand_tfidf = tfidf_brand.transform(val_clean["brand_name"])
test_brand_tfidf = tfidf_brand.transform(test_clean["brand_name"])

# Subword character n-gram TF-IDF for name
tfidf_name_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True,
)
train_name_char_tfidf = tfidf_name_char.fit_transform(train_clean["name"])
val_name_char_tfidf = tfidf_name_char.transform(val_clean["name"])
test_name_char_tfidf = tfidf_name_char.transform(test_clean["name"])

# 32-dimensional TruncatedSVD components on text representations for GBDT
svd_name = TruncatedSVD(n_components=32, algorithm="randomized", random_state=42)
train_name_svd = svd_name.fit_transform(train_name_tfidf).astype(np.float32)
val_name_svd = svd_name.transform(val_name_tfidf).astype(np.float32)
test_name_svd = svd_name.transform(test_name_tfidf).astype(np.float32)

svd_desc = TruncatedSVD(n_components=32, algorithm="randomized", random_state=42)
train_desc_svd = svd_desc.fit_transform(train_desc_tfidf).astype(np.float32)
val_desc_svd = svd_desc.transform(val_desc_tfidf).astype(np.float32)
test_desc_svd = svd_desc.transform(test_desc_tfidf).astype(np.float32)

# Full word sparse representation
X_train_sparse = sparse.hstack(
    [
        train_dense_scaled,
        train_ohe,
        train_name_tfidf,
        train_desc_tfidf,
        train_cat_tfidf,
        train_brand_tfidf,
    ],
    format="csr",
    dtype=np.float32,
)

X_val_sparse = sparse.hstack(
    [
        val_dense_scaled,
        val_ohe,
        val_name_tfidf,
        val_desc_tfidf,
        val_cat_tfidf,
        val_brand_tfidf,
    ],
    format="csr",
    dtype=np.float32,
)

X_test_sparse = sparse.hstack(
    [
        test_dense_scaled,
        test_ohe,
        test_name_tfidf,
        test_desc_tfidf,
        test_cat_tfidf,
        test_brand_tfidf,
    ],
    format="csr",
    dtype=np.float32,
)

# Character sparse representation for character-level Ridge
X_train_char_sparse = sparse.hstack(
    [
        train_dense_scaled,
        train_ohe,
        train_name_char_tfidf,
        train_cat_tfidf,
        train_brand_tfidf,
    ],
    format="csr",
    dtype=np.float32,
)

X_val_char_sparse = sparse.hstack(
    [
        val_dense_scaled,
        val_ohe,
        val_name_char_tfidf,
        val_cat_tfidf,
        val_brand_tfidf,
    ],
    format="csr",
    dtype=np.float32,
)

X_test_char_sparse = sparse.hstack(
    [
        test_dense_scaled,
        test_ohe,
        test_name_char_tfidf,
        test_cat_tfidf,
        test_brand_tfidf,
    ],
    format="csr",
    dtype=np.float32,
)


# -------------------------------------------------------------------------
# 6. Model Architectures & Data Loader
# -------------------------------------------------------------------------
class MercariSparseMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        dropout_rates: list = [0.35, 0.25, 0.15],
    ):
        super(MercariSparseMLP, self).__init__()
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.norm0 = nn.LayerNorm(hidden_dims[0])
        self.act0 = nn.Mish()
        self.drop0 = nn.Dropout(dropout_rates[0])

        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.norm1 = nn.LayerNorm(hidden_dims[1])
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rates[1])

        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.norm2 = nn.LayerNorm(hidden_dims[2])
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(dropout_rates[2])

        self.shortcut = nn.Linear(hidden_dims[0], hidden_dims[2])
        self.head = nn.Linear(hidden_dims[2], 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.drop0(self.act0(self.norm0(self.input_proj(x))))
        h1 = self.drop1(self.act1(self.norm1(self.fc1(h0))))
        h2 = self.drop2(self.act2(self.norm2(self.fc2(h1))))
        h2 = h2 + self.shortcut(h0)
        return self.head(h2).squeeze(-1)


class LogScaleRMSELoss(nn.Module):

    def __init__(self):
        super(LogScaleRMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, y_pred_log: torch.Tensor, y_true_log: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(y_pred_log, y_true_log)


class SparseBatchLoader:

    def __init__(self, X_csr, y=None, batch_size=1024, shuffle=True):
        self.X_csr = X_csr
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X_csr.shape[0]

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.n_samples)
            for i in range(0, self.n_samples, self.batch_size):
                batch_idx = np.sort(perm[i : i + self.batch_size])
                batch_x = (
                    torch.from_numpy(self.X_csr[batch_idx].toarray())
                    .float()
                    .to(device)
                )
                if self.y is not None:
                    batch_y = torch.from_numpy(self.y[batch_idx]).float().to(device)
                    yield batch_x, batch_y
                else:
                    yield batch_x
        else:
            for i in range(0, self.n_samples, self.batch_size):
                batch_idx = slice(i, min(i + self.batch_size, self.n_samples))
                batch_x = (
                    torch.from_numpy(self.X_csr[batch_idx].toarray())
                    .float()
                    .to(device)
                )
                if self.y is not None:
                    batch_y = torch.from_numpy(self.y[batch_idx]).float().to(device)
                    yield batch_x, batch_y
                else:
                    yield batch_x

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# -------------------------------------------------------------------------
# 7. Model Training & Evaluation
# -------------------------------------------------------------------------
feature_dim = X_train_sparse.shape[1]
nn_model = MercariSparseMLP(input_dim=feature_dim).to(device)
criterion = LogScaleRMSELoss()
optimizer = AdamW(
    nn_model.parameters(),
    lr=3e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)
epochs = 8
batch_size = 1024
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-5)

best_val_rmse = float("inf")
best_model_state = None

for epoch in range(1, epochs + 1):
    nn_model.train()
    total_train_loss = 0.0
    num_batches = 0

    train_loader = SparseBatchLoader(
        X_train_sparse, y_train_log, batch_size=batch_size, shuffle=True
    )
    for bx, by in train_loader:
        optimizer.zero_grad()
        preds = nn_model(bx)
        loss = criterion(preds, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_train_loss / max(1, num_batches)

    nn_model.eval()
    val_preds_list = []
    val_loader = SparseBatchLoader(X_val_sparse, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for bx in val_loader:
            val_preds_list.append(nn_model(bx).cpu().numpy())

    val_log_preds_nn = np.concatenate(val_preds_list)
    val_preds_nn = np.clip(np.expm1(val_log_preds_nn), 0.0, None)
    val_rmse_nn = root_mean_squared_error(y_val, val_preds_nn)

    if val_rmse_nn < best_val_rmse:
        best_val_rmse = val_rmse_nn
        best_model_state = copy.deepcopy(nn_model.state_dict())

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Log-Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse_nn:.4f} | Best Val RMSE: {best_val_rmse:.4f}"
    )

if best_model_state is not None:
    nn_model.load_state_dict(best_model_state)

nn_model.eval()
val_preds_list = []
val_loader = SparseBatchLoader(X_val_sparse, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for bx in val_loader:
        val_preds_list.append(nn_model(bx).cpu().numpy())
val_preds_nn = np.clip(np.expm1(np.concatenate(val_preds_list)), 0.0, None)

# -------------------------------------------------------------------------
# 7. Heterogeneous Model Training: Dual Ridge, LightGBM, and NN
# -------------------------------------------------------------------------
# 7a. Ridge Regressor (Word Representation) with 5-Fold OOF Meta-Features
oof_ridge_word = np.zeros(len(train_clean), dtype=np.float32)
kf_models = KFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, oof_idx in kf_models.split(train_clean):
    fold_ridge = Ridge(
        alpha=4.0,
        solver="lsqr",
        fit_intercept=True,
        random_state=42,
        tol=1e-3,
        max_iter=100,
    )
    fold_ridge.fit(X_train_sparse[tr_idx], y_train_log[tr_idx])
    oof_ridge_word[oof_idx] = fold_ridge.predict(X_train_sparse[oof_idx])

ridge_word_full = Ridge(
    alpha=4.0,
    solver="lsqr",
    fit_intercept=True,
    random_state=42,
    tol=1e-3,
    max_iter=100,
)
ridge_word_full.fit(X_train_sparse, y_train_log)
val_log_preds_rw = ridge_word_full.predict(X_val_sparse)
test_log_preds_rw = ridge_word_full.predict(X_test_sparse)

# 7b. Ridge Regressor (Character Representation) with 5-Fold OOF Meta-Features
oof_ridge_char = np.zeros(len(train_clean), dtype=np.float32)
for tr_idx, oof_idx in kf_models.split(train_clean):
    fold_ridge = Ridge(
        alpha=3.0,
        solver="lsqr",
        fit_intercept=True,
        random_state=42,
        tol=1e-3,
        max_iter=100,
    )
    fold_ridge.fit(X_train_char_sparse[tr_idx], y_train_log[tr_idx])
    oof_ridge_char[oof_idx] = fold_ridge.predict(X_train_char_sparse[oof_idx])

ridge_char_full = Ridge(
    alpha=3.0,
    solver="lsqr",
    fit_intercept=True,
    random_state=42,
    tol=1e-3,
    max_iter=100,
)
ridge_char_full.fit(X_train_char_sparse, y_train_log)
val_log_preds_rc = ridge_char_full.predict(X_val_char_sparse)
test_log_preds_rc = ridge_char_full.predict(X_test_char_sparse)

# 7c. LightGBM GBDT on Tabular + SVD + OOF Ridge Meta-Features
X_train_lgb = np.hstack(
    [
        train_dense_scaled,
        train_name_svd,
        train_desc_svd,
        oof_ridge_word[:, None],
        oof_ridge_char[:, None],
    ]
)
X_val_lgb = np.hstack(
    [
        val_dense_scaled,
        val_name_svd,
        val_desc_svd,
        val_log_preds_rw[:, None],
        val_log_preds_rc[:, None],
    ]
)
X_test_lgb = np.hstack(
    [
        test_dense_scaled,
        test_name_svd,
        test_desc_svd,
        test_log_preds_rw[:, None],
        test_log_preds_rc[:, None],
    ]
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.04,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_lgb,
    y_train_log,
    eval_set=[(X_val_lgb, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
)
val_log_preds_lgb = lgb_model.predict(X_val_lgb)
test_log_preds_lgb = lgb_model.predict(X_test_lgb)

# 7d. Convert predictions to natural scale ($e^x - 1$)
val_preds_nn = np.clip(val_preds_nn, 0.0, None)
val_preds_rw = np.clip(np.expm1(val_log_preds_rw), 0.0, None)
val_preds_rc = np.clip(np.expm1(val_log_preds_rc), 0.0, None)
val_preds_lgb = np.clip(np.expm1(val_log_preds_lgb), 0.0, None)

test_loader = SparseBatchLoader(X_test_sparse, batch_size=batch_size, shuffle=False)
test_preds_list = []
nn_model.eval()
with torch.no_grad():
    for bx in test_loader:
        test_preds_list.append(nn_model(bx).cpu().numpy())
test_preds_nn = np.clip(np.expm1(np.concatenate(test_preds_list)), 0.0, None)
test_preds_rw = np.clip(np.expm1(test_log_preds_rw), 0.0, None)
test_preds_rc = np.clip(np.expm1(test_log_preds_rc), 0.0, None)
test_preds_lgb = np.clip(np.expm1(test_log_preds_lgb), 0.0, None)

# 7e. Constrained SLSQP Optimization on Validation Set
V_preds = np.column_stack([val_preds_nn, val_preds_rw, val_preds_rc, val_preds_lgb])
T_preds = np.column_stack([test_preds_nn, test_preds_rw, test_preds_rc, test_preds_lgb])


def loss_func(weights):
    w = weights / np.sum(weights)
    pred = V_preds @ w
    return root_mean_squared_error(y_val, pred)


init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
cons = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=cons
)

opt_weights = res.x / np.sum(res.x)
print(
    f"Optimal ensemble weights (NN, Ridge-Word, Ridge-Char, LightGBM): {opt_weights.round(4)}"
)

final_val_score = float(loss_func(opt_weights))
test_preds_ensemble = np.clip(T_preds @ opt_weights, 0.0, None)

# -------------------------------------------------------------------------
# 8. Test Submission Generation
# -------------------------------------------------------------------------
submission_df = pd.DataFrame(
    {
        "id": test_ids.astype(int),
        "price": test_preds_ensemble.astype(np.float64),
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
