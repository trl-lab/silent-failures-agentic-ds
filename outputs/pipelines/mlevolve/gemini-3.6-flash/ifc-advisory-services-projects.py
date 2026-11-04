import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Estimated Total Budget ($)"
if target_col not in train_df.columns:
    target_candidates = [
        c for c in train_df.columns if "budget" in c.lower() or "total" in c.lower()
    ]
    target_col = target_candidates[0] if target_candidates else train_df.columns[-1]

y_train_raw = train_df[target_col].values.astype(np.float64)
y_train_log = np.log1p(np.maximum(0, y_train_raw)).astype(np.float32)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "ID" in test_df.columns:
    test_ids = test_df["ID"].values
else:
    test_ids = np.arange(len(test_df))

train_features = train_df.drop(columns=[target_col], errors="ignore").copy()
test_features = test_df.copy()

# 2. Categorize Columns into Modalities
text_cols = []
cat_cols = []
num_cols = []

for col in train_features.columns:
    if col.lower() in ["id", "index"]:
        continue
    if train_features[col].dtype == "object" or pd.api.types.is_categorical_dtype(
        train_features[col]
    ):
        sample_vals = train_features[col].dropna().astype(str)
        if len(sample_vals) > 0 and sample_vals.str.len().mean() > 35:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# 3. Unstructured Text Feature Extraction (Pretrained Embeddings + 30 SVD + Financial Regex)
train_text = (
    train_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series([""] * len(train_features))
)
test_text = (
    test_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series([""] * len(test_features))
)

train_features["text_len"] = train_text.str.len()
test_features["text_len"] = test_text.str.len()
train_features["text_word_count"] = train_text.str.split().str.len()
test_features["text_word_count"] = test_text.str.split().str.len()
train_features["text_digit_count"] = train_text.apply(
    lambda s: sum(c.isdigit() for c in s)
)
test_features["text_digit_count"] = test_text.apply(
    lambda s: sum(c.isdigit() for c in s)
)

# Extract financial numbers and budget scales directly from project text using Regex
def extract_financial_regex(text_series):
    max_vals = []
    sum_vals = []
    dollar_counts = []
    pattern = re.compile(r'\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|m|billion|b|thousand|k)?', re.IGNORECASE)

    for text in text_series:
        text_str = str(text)
        matches = pattern.findall(text_str)
        amounts = []
        for val_str, mult in matches:
            if not val_str:
                continue
            try:
                val = float(val_str.replace(',', ''))
            except ValueError:
                continue
            mult = mult.lower()
            if mult in ['million', 'm']:
                val *= 1e6
            elif mult in ['billion', 'b']:
                val *= 1e9
            elif mult in ['thousand', 'k']:
                val *= 1e3
            if val > 1000:
                amounts.append(val)

        max_vals.append(np.log1p(max(amounts)) if amounts else 0.0)
        sum_vals.append(np.log1p(sum(amounts)) if amounts else 0.0)
        dollar_counts.append(text_str.count('$'))
    return max_vals, sum_vals, dollar_counts

tr_max, tr_sum, tr_cnt = extract_financial_regex(train_text)
te_max, te_sum, te_cnt = extract_financial_regex(test_text)

train_features["fin_regex_max"] = tr_max
train_features["fin_regex_sum"] = tr_sum
train_features["fin_regex_dollar_cnt"] = tr_cnt

test_features["fin_regex_max"] = te_max
test_features["fin_regex_sum"] = te_sum
test_features["fin_regex_dollar_cnt"] = te_cnt

# Generate deep sentence embeddings via SentenceTransformers / Transformers
if train_text.str.len().max() > 0:
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb_train = embedder.encode(train_text.tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        emb_test = embedder.encode(test_text.tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    except Exception:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model.eval()
        def get_embeddings(texts):
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    batch_texts = texts[i:i+32]
                    encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                    out = model(**encoded)
                    mask = encoded['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                    sum_emb = torch.sum(out.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    mean_emb = (sum_emb / sum_mask).numpy()
                    all_embs.append(mean_emb)
            return np.vstack(all_embs)
        emb_train = get_embeddings(train_text.tolist())
        emb_test = get_embeddings(test_text.tolist())

    for i in range(emb_train.shape[1]):
        train_features[f"emb_{i}"] = emb_train[:, i]
        test_features[f"emb_{i}"] = emb_test[:, i]

    # Enhanced TF-IDF with 30 TruncatedSVD components
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    tfidf_train = tfidf.fit_transform(train_text)
    tfidf_test = tfidf.transform(test_text)

    svd = TruncatedSVD(n_components=30, random_state=42)
    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)

    for i in range(30):
        train_features[f"svd_{i}"] = svd_train[:, i]
        test_features[f"svd_{i}"] = svd_test[:, i]

# Drop raw text columns after feature extraction
if text_cols:
    train_features.drop(columns=text_cols, inplace=True, errors="ignore")
    test_features.drop(columns=text_cols, inplace=True, errors="ignore")

# 4. Categorical Interactions & Frequency Encoding
base_cat_cols = list(cat_cols)
if len(base_cat_cols) >= 2:
    for i in range(min(len(base_cat_cols), 3)):
        for j in range(i + 1, min(len(base_cat_cols), 3)):
            c1, c2 = base_cat_cols[i], base_cat_cols[j]
            inter_col = f"{c1}_x_{c2}"
            train_features[inter_col] = (
                train_features[c1].astype(str) + "_" + train_features[c2].astype(str)
            )
            test_features[inter_col] = (
                test_features[c1].astype(str) + "_" + test_features[c2].astype(str)
            )
            cat_cols.append(inter_col)

for col in cat_cols:
    freq = train_features[col].value_counts(normalize=True).to_dict()
    train_features[f"{col}_freq"] = (
        train_features[col].map(freq).fillna(0).astype(float)
    )
    test_features[f"{col}_freq"] = test_features[col].map(freq).fillna(0).astype(float)

# Group Aggregations
for cat in cat_cols[:3]:
    for num in ["text_len", "text_word_count"] + num_cols[:2]:
        if num in train_features.columns:
            grp_mean = train_features.groupby(cat)[num].mean().to_dict()
            train_features[f"{cat}_{num}_mean"] = (
                train_features[cat].map(grp_mean).fillna(0).astype(float)
            )
            test_features[f"{cat}_{num}_mean"] = (
                test_features[cat].map(grp_mean).fillna(0).astype(float)
            )

# Label Encoding for Gradient Boosting
for c in cat_cols:
    train_features[c] = train_features[c].fillna("Missing").astype(str)
    test_features[c] = test_features[c].fillna("Missing").astype(str)
    unique_cats = sorted(list(set(train_features[c]).union(set(test_features[c]))))
    cat_type = pd.CategoricalDtype(categories=unique_cats, ordered=False)
    train_features[c] = train_features[c].astype(cat_type).cat.codes
    test_features[c] = test_features[c].astype(cat_type).cat.codes

# Numeric missing values imputation
all_feature_cols = [
    c for c in train_features.columns if c.lower() not in ["id", "index"]
]
for col in all_feature_cols:
    if pd.api.types.is_numeric_dtype(train_features[col]):
        med = train_features[col].median()
        med = 0.0 if pd.isna(med) else float(med)
        train_features[col] = train_features[col].fillna(med)
        test_features[col] = test_features[col].fillna(med)

# 5. Out-of-Fold Leak-Free Target Encoding with Multiple Smoothing Factors
kf = KFold(n_splits=5, shuffle=True, random_state=42)
te_cols = [c for c in cat_cols if not c.endswith("_freq")][:5]
global_mean = float(y_train_log.mean())
m_smooth_list = [2.0, 10.0, 30.0]

for c in te_cols:
    for m_smooth in m_smooth_list:
        col_name = f"{c}_te_m{int(m_smooth)}"
        train_features[col_name] = 0.0
        test_features[col_name] = 0.0
        test_te_sum = np.zeros(len(test_features))

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
            tr_cat = train_features[c].iloc[train_idx]
            tr_y = y_train_log[train_idx]
            stats = (
                pd.DataFrame({c: tr_cat, "y": tr_y}).groupby(c)["y"].agg(["count", "mean"])
            )
            smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
                stats["count"] + m_smooth
            )

            val_cat = train_features[c].iloc[val_idx]
            train_features.loc[train_features.index[val_idx], col_name] = val_cat.map(
                smoothed
            ).fillna(global_mean)
            test_te_sum += test_features[c].map(smoothed).fillna(global_mean).values

        test_features[col_name] = test_te_sum / 5.0

# Extract clean model feature matrix
model_features = [c for c in train_features.columns if c.lower() not in ["id", "index"]]
X_train = train_features[model_features].values.astype(np.float32)
X_test = test_features[model_features].values.astype(np.float32)


# 6. PyTorch Tabular ResNet Neural Network Architecture
class ResBlock(nn.Module):

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 7. Model Hyperparameters & Training Loop Setup (Tuned for High-Dimensional Semantic Embeddings)
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "verbose": -1,
    "n_estimators": 1000,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "n_estimators": 1000,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "rsm": 0.5,
    "verbose": False,
    "iterations": 1000,
    "early_stopping_rounds": 40,
}

ridge_alpha = 10.0

n_train = len(X_train)
n_test = len(X_test)
model_names = ["lgb", "xgb", "catboost", "ridge", "resnet"]
n_models = len(model_names)

oof_log_preds = np.zeros((n_train, n_models))
test_log_preds = np.zeros((n_test, n_models))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8. Cross-Validation Training
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr_log = X_train[tr_idx], y_train_log[tr_idx]
    X_va, y_va_log = X_train[va_idx], y_train_log[va_idx]

    # Model 0: LightGBM
    lgb_p = lgb_params.copy()
    lgb_p["random_state"] = 42 + fold
    m_lgb = lgb.LGBMRegressor(**lgb_p)
    m_lgb.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    oof_log_preds[va_idx, 0] = m_lgb.predict(X_va)
    test_log_preds[:, 0] += m_lgb.predict(X_test) / 5.0

    # Model 1: XGBoost
    xgb_p = xgb_params.copy()
    xgb_p["random_state"] = 42 + fold
    m_xgb = xgb.XGBRegressor(**xgb_p)
    m_xgb.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_log_preds[va_idx, 1] = m_xgb.predict(X_va)
    test_log_preds[:, 1] += m_xgb.predict(X_test) / 5.0

    # Model 2: CatBoost
    cb_p = cb_params.copy()
    cb_p["random_seed"] = 42 + fold
    m_cb = cb.CatBoostRegressor(**cb_p)
    m_cb.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
    oof_log_preds[va_idx, 2] = m_cb.predict(X_va)
    test_log_preds[:, 2] += m_cb.predict(X_test) / 5.0

    # Model 3: Ridge Regression (Standardized)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_test)

    m_ridge = Ridge(alpha=ridge_alpha, random_state=42 + fold)
    m_ridge.fit(X_tr_s, y_tr_log)
    oof_log_preds[va_idx, 3] = m_ridge.predict(X_va_s)
    test_log_preds[:, 3] += m_ridge.predict(X_te_s) / 5.0

    # Model 4: PyTorch Tabular ResNet
    m_resnet = TabularResNet(
        input_dim=X_train.shape[1], hidden_dim=256, num_blocks=2, dropout_rate=0.2
    ).to(device)
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = optim.AdamW(m_resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(40):
        m_resnet.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = m_resnet(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        m_resnet.eval()
        with torch.no_grad():
            vx = torch.tensor(X_va_s, dtype=torch.float32).to(device)
            vy = torch.tensor(y_va_log, dtype=torch.float32).to(device)
            vloss = criterion(m_resnet(vx), vy).item()
            if vloss < best_val_loss:
                best_val_loss = vloss
                best_weights = {
                    k: v.cpu().clone() for k, v in m_resnet.state_dict().items()
                }

    if best_weights is not None:
        m_resnet.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    m_resnet.eval()
    with torch.no_grad():
        vx = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        tx = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        oof_log_preds[va_idx, 4] = m_resnet(vx).cpu().numpy()
        test_log_preds[:, 4] += m_resnet(tx).cpu().numpy() / 5.0

# 9. Back-transformation to Raw Scale
oof_raw_preds = np.expm1(np.maximum(0, oof_log_preds))
test_raw_preds = np.expm1(np.maximum(0, test_log_preds))


# 10. SLSQP Blending & Scale Correction Optimization
def blend_loss(params):
    weights = params[:n_models]
    scale = params[n_models]
    pred = np.dot(oof_raw_preds, weights) * scale
    return np.sqrt(mean_squared_error(y_train_raw, pred))


init_params = np.append(np.ones(n_models) / n_models, 1.0)
bounds = [(0.0, 1.0)] * n_models + [(0.1, 5.0)]
constraints = {"type": "eq", "fun": lambda p: np.sum(p[:n_models]) - 1.0}

opt_res = minimize(
    blend_loss, init_params, method="SLSQP", bounds=bounds, constraints=constraints
)

opt_weights = opt_res.x[:n_models]
opt_scale = opt_res.x[n_models]

# 11. Final OOF Evaluation & Submission Formatting
final_oof_preds = np.dot(oof_raw_preds, opt_weights) * opt_scale
final_test_preds = np.dot(test_raw_preds, opt_weights) * opt_scale
final_test_preds = np.maximum(0.0, final_test_preds)

val_rmse = np.sqrt(mean_squared_error(y_train_raw, final_oof_preds))

submission_df = pd.DataFrame(
    {"id": test_ids, "Estimated Total Budget ($)": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_rmse}")