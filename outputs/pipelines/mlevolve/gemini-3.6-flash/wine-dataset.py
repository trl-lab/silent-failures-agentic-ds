import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -------------------------------------------------------------------------
# Load Raw Datasets
# -------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path).reset_index(drop=True)
test_df = pd.read_parquet(test_path).reset_index(drop=True)

if "id" in test_df.columns:
    test_ids = test_df["id"]
else:
    test_ids = pd.Series(test_df.index, name="id")


def extract_vintage_year(df):
    vintage_series = pd.Series(np.nan, index=df.index)
    search_cols = []
    if "title" in df.columns:
        search_cols.append("title")
    if "description" in df.columns:
        search_cols.append("description")
    for col in df.columns:
        if df[col].dtype == "object" and col not in search_cols:
            search_cols.append(col)

    for col in search_cols:
        extracted = df[col].astype(str).str.extract(
            r"\b(19[5-9]\d|20[0-2]\d)\b", expand=False
        )
        extracted_num = pd.to_numeric(extracted, errors="coerce")
        vintage_series = vintage_series.fillna(extracted_num)
    return vintage_series


train_df["vintage"] = extract_vintage_year(train_df)
test_df["vintage"] = extract_vintage_year(test_df)
train_df["wine_age"] = 2024 - train_df["vintage"]
test_df["wine_age"] = 2024 - test_df["vintage"]

target_col = "Price" if "Price" in train_df.columns else "price"
y_train = train_df[target_col].values
y_log = np.log1p(np.maximum(0, y_train))

feature_cols_train = [c for c in train_df.columns if c != target_col and c != "id"]

text_cols = [
    c
    for c in feature_cols_train
    if train_df[c].dtype == "object" and train_df[c].str.len().mean() > 30
]
cat_cols = [
    c
    for c in feature_cols_train
    if train_df[c].dtype == "object" and c not in text_cols
]
num_cols = [c for c in feature_cols_train if c not in text_cols and c not in cat_cols]

# Composite cross-categorical features
composite_pairs = [
    ("country", "variety"),
    ("province", "variety"),
    ("winery", "variety"),
    ("country", "province"),
]
for col1, col2 in composite_pairs:
    if col1 in train_df.columns and col2 in train_df.columns:
        comp_col = f"{col1}_{col2}"
        train_df[comp_col] = (
            train_df[col1].fillna("").astype(str)
            + "_"
            + train_df[col2].fillna("").astype(str)
        )
        test_df[comp_col] = (
            test_df[col1].fillna("").astype(str)
            + "_"
            + test_df[col2].fillna("").astype(str)
        )
        if comp_col not in cat_cols:
            cat_cols.append(comp_col)


# -------------------------------------------------------------------------
# Feature Extraction Pipeline
# -------------------------------------------------------------------------
full_df = pd.concat([train_df, test_df], ignore_index=True)


def extract_domain_features(df, text_columns, num_columns):
    df_feat = pd.DataFrame(index=df.index)

    for c in num_columns:
        df_feat[c] = df[c].astype(float)
        df_feat[f"{c}_is_null"] = df[c].isnull().astype(int)

    if "points" in [c.lower() for c in num_columns]:
        p_col = [c for c in num_columns if c.lower() == "points"][0]
        df_feat[f"{p_col}_sq"] = df_feat[p_col] ** 2
        df_feat[f"{p_col}_log"] = np.log1p(df_feat[p_col])

    # Hierarchical group statistics
    group_cats = ["variety", "province", "winery", "country", "vintage"]
    ref_df = df
    for g_cat in group_cats:
        if g_cat in df.columns:
            for num_c in num_columns:
                grp = ref_df.groupby(g_cat)[num_c]
                grp_mean = df[g_cat].map(grp.mean())
                grp_std = df[g_cat].map(grp.std()).fillna(0)
                grp_min = df[g_cat].map(grp.min())
                grp_max = df[g_cat].map(grp.max())

                df_feat[f"{num_c}_mean_by_{g_cat}"] = grp_mean
                df_feat[f"{num_c}_std_by_{g_cat}"] = grp_std
                df_feat[f"{num_c}_min_by_{g_cat}"] = grp_min
                df_feat[f"{num_c}_max_by_{g_cat}"] = grp_max
                df_feat[f"{num_c}_diff_mean_by_{g_cat}"] = (
                    df[num_c].astype(float) - grp_mean
                )

    # Domain keyword flags and length stats
    domain_keywords = [
        "reserve",
        "estate",
        "oak",
        "vintage",
        "ripe",
        "tannins",
        "finish",
        "cherry",
        "fruit",
        "cabernet",
        "noir",
        "dry",
        "acidity",
        "rich",
    ]
    for c in text_columns:
        text_series = df[c].fillna("").astype(str)
        df_feat[f"{c}_char_len"] = text_series.str.len()
        df_feat[f"{c}_word_count"] = text_series.str.split().str.len()
        df_feat[f"{c}_exclam_count"] = text_series.str.count("!")
        df_feat[f"{c}_upper_count"] = text_series.str.findall(r"[A-Z]").str.len()

        text_lower = text_series.str.lower()
        for kw in domain_keywords:
            df_feat[f"{c}_has_{kw}"] = (
                text_lower.str.contains(kw, regex=False).astype(int)
            )

    return df_feat


train_domain = extract_domain_features(train_df, text_cols, num_cols)
test_domain = extract_domain_features(test_df, text_cols, num_cols)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold Target Encoding for Categorical Columns
oof_target_enc_train = pd.DataFrame(index=train_df.index)
for cat in cat_cols:
    oof_target_enc_train[f"{cat}_te"] = 0.0

overall_global_mean = y_log.mean()

for train_idx, val_idx in kf.split(train_df):
    tr_df, tr_y = train_df.iloc[train_idx], y_log[train_idx]
    fold_global_mean = tr_y.mean()
    for cat in cat_cols:
        cat_means = (
            tr_df.assign(target=tr_y).groupby(cat)["target"].agg(["count", "mean"])
        )
        smooth = 10
        smoothed_means = (
            cat_means["count"] * cat_means["mean"] + smooth * fold_global_mean
        ) / (cat_means["count"] + smooth)

        val_cats = train_df.iloc[val_idx][cat]
        oof_target_enc_train.iloc[
            val_idx, oof_target_enc_train.columns.get_loc(f"{cat}_te")
        ] = (val_cats.map(smoothed_means).fillna(fold_global_mean).values)

full_cat_means = {}
for cat in cat_cols:
    cat_means = (
        train_df.assign(target=y_log).groupby(cat)["target"].agg(["count", "mean"])
    )
    smooth = 10
    smoothed = (
        cat_means["count"] * cat_means["mean"] + smooth * overall_global_mean
    ) / (cat_means["count"] + smooth)
    full_cat_means[cat] = smoothed

test_target_enc = pd.DataFrame(index=test_df.index)
for cat in cat_cols:
    test_target_enc[f"{cat}_te"] = (
        test_df[cat].map(full_cat_means[cat]).fillna(overall_global_mean).values
    )

# Frequency Encoding for Categorical Columns
for cat in cat_cols:
    freq = train_df[cat].value_counts(normalize=True)
    train_domain[f"{cat}_freq"] = train_df[cat].map(freq).fillna(0).values
    test_domain[f"{cat}_freq"] = test_df[cat].map(freq).fillna(0).values

# Text Feature Extraction via Fine-Grained TF-IDF + OOF Ridge & TruncatedSVD
all_text_train = train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
all_text_test = test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

vec_word = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 3))
vec_char = TfidfVectorizer(max_features=10000, analyzer="char", ngram_range=(3, 5))

X_tr_word = vec_word.fit_transform(all_text_train)
X_te_word = vec_word.transform(all_text_test)
X_tr_char = vec_char.fit_transform(all_text_train)
X_te_char = vec_char.transform(all_text_test)

X_tfidf_train = hstack([X_tr_word, X_tr_char]).tocsr()
X_tfidf_test = hstack([X_te_word, X_te_char]).tocsr()

oof_ridge_text = np.zeros(len(train_df))
test_ridge_text = np.zeros(len(test_df))

for tr_i, va_i in kf.split(train_df):
    r_model = Ridge(alpha=1.0, random_state=42)
    r_model.fit(X_tfidf_train[tr_i], y_log[tr_i])
    oof_ridge_text[va_i] = r_model.predict(X_tfidf_train[va_i])
    test_ridge_text += r_model.predict(X_tfidf_test) / 5.0

text_data_train_list = [pd.DataFrame({"tfidf_ridge_pred": oof_ridge_text}, index=train_df.index)]
text_data_test_list = [pd.DataFrame({"tfidf_ridge_pred": test_ridge_text}, index=test_df.index)]

for c in text_cols:
    train_text = train_df[c].fillna("").astype(str).tolist()
    test_text = test_df[c].fillna("").astype(str).tolist()

    vec = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=50, random_state=42)

    train_tfidf = vec.fit_transform(train_text)
    test_tfidf = vec.transform(test_text)

    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)

    svd_cols = [f"{c}_svd_{i}" for i in range(50)]
    text_data_train_list.append(
        pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
    )
    text_data_test_list.append(
        pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)
    )

X_train_full = pd.concat(
    [train_domain, oof_target_enc_train] + text_data_train_list, axis=1
)
X_test_full = pd.concat([test_domain, test_target_enc] + text_data_test_list, axis=1)

for col in X_train_full.columns:
    if X_train_full[col].isnull().any() or X_test_full[col].isnull().any():
        median_val = X_train_full[col].median()
        X_train_full[col] = X_train_full[col].fillna(median_val)
        X_test_full[col] = X_test_full[col].fillna(median_val)

folds = []
for f, (tr, va) in enumerate(kf.split(X_train_full)):
    folds.append({"fold": f, "train_idx": tr, "val_idx": va})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# Neural Network Architecture: Tabular Gated Residual Network
# -------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc_gate = nn.Linear(dim, dim * 2)
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_norm = self.norm(x)
        gates, values = torch.chunk(self.fc_gate(x_norm), 2, dim=-1)
        gated_val = self.act(values) * torch.sigmoid(gates)
        out = self.fc_out(self.dropout(gated_val))
        return res + out


class TabularGatedNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# -------------------------------------------------------------------------
# Multi-Model Cross-Validation Pipeline
# -------------------------------------------------------------------------
n_samples = len(X_train_full)
n_test = len(X_test_full)
num_models = 5  # LightGBM, CatBoost, XGBoost, TabularGatedNet, Ridge

oof_matrix = np.zeros((n_samples, num_models))
test_matrix = np.zeros((n_test, num_models))

X_tr_raw = X_train_full.values
X_te_raw = X_test_full.values

num_epochs = 20
batch_size = 256

for fold_info in folds:
    fold = fold_info["fold"]
    tr_idx, va_idx = fold_info["train_idx"], fold_info["val_idx"]

    X_tr, y_tr_log = X_tr_raw[tr_idx], y_log[tr_idx]
    X_va, y_va_log = X_tr_raw[va_idx], y_log[va_idx]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te_raw)

    # 1. LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        extra_trees=True,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    lgb_val_pred = lgb_model.predict(X_va)
    lgb_test_pred = lgb_model.predict(X_te_raw)

    # 2. CatBoost Regressor
    cb_model = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=(X_va, y_va_log),
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_val_pred = cb_model.predict(X_va)
    cb_test_pred = cb_model.predict(X_te_raw)

    # 3. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        verbose=False,
    )
    xgb_val_pred = xgb_model.predict(X_va)
    xgb_test_pred = xgb_model.predict(X_te_raw)

    # 4. PyTorch TabularGatedNet
    tr_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va_log, dtype=torch.float32).unsqueeze(1),
    )

    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    va_loader = DataLoader(va_dataset, batch_size=512, shuffle=False, drop_last=False)

    nn_model = TabularGatedNet(input_dim=X_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_va_loss = float("inf")
    best_nn_val_pred = None

    for epoch in range(num_epochs):
        nn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(bx)
                val_preds_list.append(out.cpu().numpy())

        val_loss /= len(X_va)
        if val_loss < best_va_loss:
            best_va_loss = val_loss
            best_nn_val_pred = np.vstack(val_preds_list).ravel()
            best_state = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict(best_state)
    nn_model.eval()
    with torch.no_grad():
        nn_test_pred = (
            nn_model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .ravel()
        )

    # 5. L2-Regularized Ridge Regression
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_tr_sc, y_tr_log)
    ridge_val_pred = ridge_model.predict(X_va_sc)
    ridge_test_pred = ridge_model.predict(X_te_sc)

    oof_matrix[va_idx, 0] = lgb_val_pred
    oof_matrix[va_idx, 1] = cb_val_pred
    oof_matrix[va_idx, 2] = xgb_val_pred
    oof_matrix[va_idx, 3] = best_nn_val_pred
    oof_matrix[va_idx, 4] = ridge_val_pred

    test_matrix[:, 0] += lgb_test_pred / len(folds)
    test_matrix[:, 1] += cb_test_pred / len(folds)
    test_matrix[:, 2] += xgb_test_pred / len(folds)
    test_matrix[:, 3] += nn_test_pred / len(folds)
    test_matrix[:, 4] += ridge_test_pred / len(folds)


# -------------------------------------------------------------------------
# SLSQP Optimization for Ensemble Blending
# -------------------------------------------------------------------------
def objective(weights):
    w = weights / np.sum(weights)
    pred_log = oof_matrix @ w
    pred_price = np.expm1(np.maximum(0, pred_log))
    return root_mean_squared_error(y_train, pred_price)


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

best_weights = opt_res.x / np.sum(opt_res.x)

# Evaluate Final Out-Of-Fold Validation Performance
opt_oof_log = oof_matrix @ best_weights
opt_oof_price = np.expm1(np.maximum(0, opt_oof_log))
final_score = root_mean_squared_error(y_train, opt_oof_price)

# Final Test Prediction and Submission Generation
opt_test_log = test_matrix @ best_weights
opt_test_price = np.expm1(np.maximum(0, opt_test_log))

submission_df = pd.DataFrame({"id": test_ids, "Price": opt_test_price})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")