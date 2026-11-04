import gc
import os
import re
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize, nnls
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Environment & Parameter Setup
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs("./working", exist_ok=True)

# 2. Data Loading & Target/ID Detection
print("Loading train and test datasets...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = None
possible_targets = ["H index", "H_index", "h_index", "H-Index", "H Index"]
for col in possible_targets:
    if col in train_df.columns:
        target_col = col
        break

if target_col is None:
    for col in train_df.columns:
        if "h" in col.lower() and "index" in col.lower():
            target_col = col
            break

id_col = None
for col in ["id", "ID", "Id", "Index", "row_id"]:
    if col in test_df.columns:
        id_col = col
        break

if id_col is None:
    test_df["id"] = np.arange(len(test_df))
    train_df["id"] = np.arange(len(train_df))
    id_col = "id"

y_train_raw = train_df[target_col].values.astype(np.float64)
y_train_log = np.log1p(np.maximum(0, y_train_raw))
test_ids = test_df[id_col].values

train_len = len(train_df)
combined_df = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    axis=0,
    ignore_index=True,
)


# 3. Data Cleaning & Type Casting
def clean_numeric_series(s):
    if s.dtype in [np.float64, np.float32, np.int64, np.int32]:
        return s
    cleaned = s.astype(str).str.replace(",", "").str.replace("%", "").str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


for col in combined_df.columns:
    if col == id_col:
        continue
    if combined_df[col].dtype == object:
        sample = combined_df[col].dropna().head(20).astype(str)
        if sample.str.replace(".", "", regex=False).str.isdigit().all():
            combined_df[col] = clean_numeric_series(combined_df[col])

num_cols = [
    c
    for c in combined_df.columns
    if combined_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    and c != id_col
]
cat_cols = [c for c in combined_df.columns if c not in num_cols and c != id_col]

for c in num_cols:
    med = combined_df[c].median()
    combined_df[c] = combined_df[c].fillna(med if not pd.isna(med) else 0.0)

for c in cat_cols:
    combined_df[c] = combined_df[c].fillna("Missing").astype(str)

# Category Parsing
cat_match = next((c for c in cat_cols if "cat" in c.lower()), None)
if cat_match:

    def get_prim_cat(x):
        p = x.split(";")
        return (
            re.sub(r"\(Q\d\)", "", p[0]).strip()
            if len(p) > 0 and p[0].strip()
            else "Missing"
        )

    combined_df["primary_cat"] = combined_df[cat_match].apply(get_prim_cat)
    combined_df["num_cats"] = combined_df[cat_match].apply(
        lambda x: len(x.split(";")) if x != "Missing" else 0
    )
    cat_cols.append("primary_cat")
    num_cols.append("num_cats")

# 4. Feature Engineering: Physics Bounds & Bibliometric Ratios
doc_col = next(
    (
        c
        for c in num_cols
        if "doc" in c.lower() and "cite" not in c.lower() and "ref" not in c.lower()
    ),
    None,
)
cite_col = next(
    (c for c in num_cols if "cite" in c.lower() and "doc" not in c.lower()), None
)
ref_col = next((c for c in num_cols if "ref" in c.lower()), None)
citable_col = next((c for c in num_cols if "citable" in c.lower()), None)
sjr_col = next((c for c in num_cols if "sjr" in c.lower()), None)

eps = 1e-5

if cite_col:
    combined_df["hirsch_upper_bound"] = np.sqrt(np.maximum(0, combined_df[cite_col]))

if doc_col and cite_col:
    combined_df["cites_per_doc"] = combined_df[cite_col] / (combined_df[doc_col] + eps)
    combined_df["hirsch_approx"] = np.power(
        np.maximum(0, combined_df[doc_col]), 0.4
    ) * np.power(np.maximum(0, combined_df["cites_per_doc"]), 0.6)
    combined_df["cite_doc_log_prod"] = np.log1p(
        np.maximum(0, combined_df[cite_col])
    ) * np.log1p(np.maximum(0, combined_df[doc_col]))
    # Lotka-Hirsch power scaling interaction: doc^(1/3) * cites^(2/3)
    combined_df["lotka_hirsch_power"] = np.power(
        np.maximum(0, combined_df[doc_col]), 1.0 / 3.0
    ) * np.power(np.maximum(0, combined_df[cite_col]), 2.0 / 3.0)

if doc_col and ref_col:
    combined_df["refs_per_doc"] = combined_df[ref_col] / (combined_df[doc_col] + eps)

if cite_col and ref_col:
    combined_df["cites_per_ref"] = combined_df[cite_col] / (combined_df[ref_col] + 1.0)

if citable_col and doc_col:
    combined_df["citable_ratio"] = combined_df[citable_col] / (
        combined_df[doc_col] + eps
    )

if sjr_col and doc_col:
    combined_df["sjr_doc_prod"] = combined_df[sjr_col] * combined_df[doc_col]

if sjr_col and cite_col:
    combined_df["sjr_cite_prod"] = combined_df[sjr_col] * np.log1p(
        np.maximum(0, combined_df[cite_col])
    )

for c in list(num_cols):
    v = combined_df[c].values
    if (v >= 0).all() and (v.max() > 10.0):
        combined_df[f"{c}_log1p"] = np.log1p(v)

num_cols = [
    c
    for c in combined_df.columns
    if combined_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    and c != id_col
]

# 5. Group Aggregations & Text Encodings
group_keys = [c for c in cat_cols if combined_df[c].nunique() < 500]
agg_targets = [
    c
    for c in num_cols
    if "sjr" in c.lower() or "doc" in c.lower() or "cite" in c.lower()
][:4]

for key in group_keys[:3]:
    for target in agg_targets:
        grp = combined_df.groupby(key)[target].transform("mean")
        combined_df[f"{key}_{target}_mean"] = grp
        combined_df[f"{key}_{target}_diff"] = combined_df[target] - grp
        combined_df[f"{key}_{target}_std"] = (
            combined_df.groupby(key)[target].transform("std").fillna(0)
        )

# Discipline-specific percentile rank features
if "primary_cat" in combined_df.columns:
    for target in agg_targets:
        combined_df[f"primary_cat_{target}_pct_rank"] = combined_df.groupby("primary_cat")[
            target
        ].rank(pct=True)

# Multi-Modal LSA + NMF Text Embeddings
text_combined = combined_df[cat_cols[0]].astype(str)
for c in cat_cols[1:]:
    text_combined = text_combined + " " + combined_df[c].astype(str)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
tfidf_mat = tfidf.fit_transform(text_combined)

svd = TruncatedSVD(n_components=24, random_state=SEED)
svd_feats = svd.fit_transform(tfidf_mat)
svd_df = pd.DataFrame(svd_feats, columns=[f"lsa_{i}" for i in range(24)])

nmf = NMF(n_components=8, random_state=SEED, max_iter=200)
nmf_feats = nmf.fit_transform(tfidf_mat)
nmf_df = pd.DataFrame(nmf_feats, columns=[f"nmf_{i}" for i in range(8)])

combined_df = pd.concat([combined_df, svd_df, nmf_df], axis=1)

# Categorical Frequency & Label Encodings
for c in cat_cols:
    freq = combined_df[c].map(combined_df[c].value_counts())
    combined_df[f"{c}_freq"] = freq
    combined_df[c] = combined_df[c].astype("category").cat.codes

feature_cols = [c for c in combined_df.columns if c != id_col]
X_train_full = combined_df.iloc[:train_len][feature_cols].copy()
X_test = combined_df.iloc[train_len:][feature_cols].copy()


# 6. Custom Gauss-Newton Loss & PyTorch Model Architectures
def lgb_gauss_newton_rmse_objective(y_true_log, z_pred):
    y_true = np.expm1(np.maximum(0, y_true_log))
    hat_y = np.expm1(np.maximum(-0.99, z_pred))
    grad = (hat_y - y_true) * (hat_y + 1.0)
    hess = (hat_y + 1.0) ** 2 + 1e-4
    return grad, hess


def lgb_gauss_newton_rmse_eval(y_true_log, z_pred):
    y_true = np.expm1(np.maximum(0, y_true_log))
    hat_y = np.expm1(np.maximum(-0.99, z_pred))
    rmse = np.sqrt(np.mean((hat_y - y_true) ** 2))
    return "raw_rmse", rmse, False


class GatedTabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.drop1 = nn.Dropout(dropout)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.drop2 = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.bn_in(self.in_proj(x)))

        h1 = self.fc1(x)
        gate1, val1 = h1.chunk(2, dim=-1)
        h1 = torch.sigmoid(gate1) * self.act(val1)
        h1 = self.drop1(self.proj1(h1))
        x = x + h1

        h2 = self.fc2(x)
        gate2, val2 = h2.chunk(2, dim=-1)
        h2 = torch.sigmoid(gate2) * self.act(val2)
        h2 = self.drop2(self.proj2(h2))
        x = x + h2

        return self.head(x).squeeze(-1)


# 7. Cross-Validation Training Loop
N_MODELS = 8
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds = np.zeros((train_len, N_MODELS))
test_preds = np.zeros((len(test_df), N_MODELS))
te_keys = [c for c in cat_cols if combined_df[c].nunique() > 10][:2]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_log)):
    X_tr = X_train_full.iloc[train_idx].copy()
    y_tr_log = y_train_log[train_idx]
    y_tr_raw = y_train_raw[train_idx]

    X_va = X_train_full.iloc[val_idx].copy()
    y_va_log = y_train_log[val_idx]
    y_va_raw = y_train_raw[val_idx]

    X_te = X_test.copy()

    for key in te_keys:
        global_mean = y_tr_log.mean()
        te_map = pd.Series(y_tr_log).groupby(X_tr[key]).mean()
        te_counts = pd.Series(y_tr_log).groupby(X_tr[key]).count()
        smoothed = (te_counts * te_map + 10.0 * global_mean) / (te_counts + 10.0)

        X_tr[f"{key}_te"] = X_tr[key].map(smoothed).fillna(global_mean)
        X_va[f"{key}_te"] = X_va[key].map(smoothed).fillna(global_mean)
        X_te[f"{key}_te"] = X_te[key].map(smoothed).fillna(global_mean)

        # Robust Median Target Encoding
        global_med = np.median(y_tr_raw)
        med_map = pd.Series(y_tr_raw).groupby(X_tr[key]).median()
        X_tr[f"{key}_med_te"] = X_tr[key].map(med_map).fillna(global_med)
        X_va[f"{key}_med_te"] = X_va[key].map(med_map).fillna(global_med)
        X_te[f"{key}_med_te"] = X_te[key].map(med_map).fillna(global_med)

        # Robust IQR Target Encoding
        def calc_iqr(s):
            q75, q25 = np.percentile(s, [75, 25])
            return q75 - q25

        global_iqr = np.percentile(y_tr_raw, 75) - np.percentile(y_tr_raw, 25)
        iqr_map = pd.Series(y_tr_raw).groupby(X_tr[key]).apply(calc_iqr)
        X_tr[f"{key}_iqr_te"] = X_tr[key].map(iqr_map).fillna(global_iqr)
        X_va[f"{key}_iqr_te"] = X_va[key].map(iqr_map).fillna(global_iqr)
        X_te[f"{key}_iqr_te"] = X_te[key].map(iqr_map).fillna(global_iqr)

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_va = X_va.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Model 0: LGBM Gauss-Newton
    lgb_custom = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    lgb_custom.set_params(objective=lgb_gauss_newton_rmse_objective)
    lgb_custom.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        eval_metric=lgb_gauss_newton_rmse_eval,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds[val_idx, 0] = np.expm1(np.maximum(0, lgb_custom.predict(X_va)))
    test_preds[:, 0] += np.expm1(np.maximum(0, lgb_custom.predict(X_te))) / 5.0

    # Model 1: LGBM Standard L2 (Log Target)
    lgb_std = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold + 10,
        n_jobs=-1,
    )
    lgb_std.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds[val_idx, 1] = np.expm1(np.maximum(0, lgb_std.predict(X_va)))
    test_preds[:, 1] += np.expm1(np.maximum(0, lgb_std.predict(X_te))) / 5.0

    # Model 2: LGBM Raw Target
    lgb_raw = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold + 30,
        n_jobs=-1,
    )
    lgb_raw.fit(
        X_tr,
        y_tr_raw,
        eval_set=[(X_va, y_va_raw)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds[val_idx, 2] = np.maximum(0, lgb_raw.predict(X_va))
    test_preds[:, 2] += np.maximum(0, lgb_raw.predict(X_te)) / 5.0

    # Model 3: XGBoost Log Target
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        tree_method="hist",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_preds[val_idx, 3] = np.expm1(np.maximum(0, xgb_model.predict(X_va)))
    test_preds[:, 3] += np.expm1(np.maximum(0, xgb_model.predict(X_te))) / 5.0

    # Model 4: XGBoost Raw Target
    xgb_raw = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold + 40,
        tree_method="hist",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_raw.fit(X_tr, y_tr_raw, eval_set=[(X_va, y_va_raw)], verbose=False)
    oof_preds[val_idx, 4] = np.maximum(0, xgb_raw.predict(X_va))
    test_preds[:, 4] += np.maximum(0, xgb_raw.predict(X_te)) / 5.0

    # Model 5: CatBoost Raw Target
    cat_raw = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_raw.fit(X_tr, y_tr_raw, eval_set=(X_va, y_va_raw), verbose=False)
    oof_preds[val_idx, 5] = np.maximum(0, cat_raw.predict(X_va))
    test_preds[:, 5] += np.maximum(0, cat_raw.predict(X_te)) / 5.0

    # Model 6: CatBoost Log Target
    cat_log = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED + fold + 20,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
    oof_preds[val_idx, 6] = np.expm1(np.maximum(0, cat_log.predict(X_va)))
    test_preds[:, 6] += np.expm1(np.maximum(0, cat_log.predict(X_te))) / 5.0

    # Model 5: PyTorch Gated Tabular ResNet
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va_log, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=256, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=512, shuffle=False)

    nn_model = GatedTabularResNet(
        input_dim=X_tr_sc.shape[1], hidden_dim=256, dropout=0.15
    ).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    best_val_loss = float("inf")
    best_nn_preds_va = None
    best_nn_preds_te = None

    for epoch in range(25):
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
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in va_loader:
                bx = bx.to(device)
                val_preds_list.append(nn_model(bx).cpu().numpy())
        val_preds_arr = np.concatenate(val_preds_list)
        val_loss = root_mean_squared_error(y_va_log, val_preds_arr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_preds_va = val_preds_arr
            with torch.no_grad():
                te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
                best_nn_preds_te = nn_model(te_tensor).cpu().numpy()

    oof_preds[val_idx, 7] = np.expm1(np.maximum(0, best_nn_preds_va))
    test_preds[:, 7] += np.expm1(np.maximum(0, best_nn_preds_te)) / 5.0

    fold_gn_rmse = root_mean_squared_error(y_va_raw, oof_preds[val_idx, 0])
    print(f"Fold {fold+1}/5 complete - GN-LGBM Raw RMSE: {fold_gn_rmse:.4f}")
    gc.collect()


# 8. NNLS Meta-Learner & Isotonic Recalibration
nnls_weights, _ = nnls(oof_preds, y_train_raw)
if np.sum(nnls_weights) > 0:
    nnls_weights = nnls_weights / np.sum(nnls_weights)

raw_oof_blend = np.maximum(0, oof_preds @ nnls_weights)
raw_test_blend = np.maximum(0, test_preds @ nnls_weights)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_oof_blend, y_train_raw)

final_oof_predictions = np.maximum(0, iso.predict(raw_oof_blend))
final_test_predictions = np.maximum(0, iso.predict(raw_test_blend))

final_score = root_mean_squared_error(y_train_raw, final_oof_predictions)

# 9. Format & Save Submission
submission_df = pd.DataFrame({"id": test_ids, "H index": final_test_predictions})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission with shape {submission_df.shape} to '{SUBMISSION_PATH}'.")

# Final Validation Output
print(f"Final Validation Score: {final_score}")
