import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Create necessary directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Data & Column Identification
# -----------------------------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "H index"
y_train = train_df[target_col].copy().values.astype(np.float32)
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()


def clean_numeric_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s_clean = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s_clean, errors="coerce")


all_cols = list(X_train_raw.columns)
num_cols = []
cat_cols = []
text_cols = []

for c in all_cols:
    sample_vals = X_train_raw[c].dropna()
    if len(sample_vals) == 0:
        continue
    converted = pd.to_numeric(
        sample_vals.astype(str).str.replace(",", "").str.replace("%", ""),
        errors="coerce",
    )
    if converted.notna().mean() > 0.8:
        num_cols.append(c)
        X_train_raw[c] = clean_numeric_series(X_train_raw[c])
        X_test_raw[c] = clean_numeric_series(X_test_raw[c])
    else:
        unique_ratio = sample_vals.nunique() / len(sample_vals)
        if unique_ratio > 0.3 or sample_vals.astype(str).str.len().mean() > 50:
            text_cols.append(c)
        else:
            cat_cols.append(c)


# -----------------------------------------------------------------------------
# 2. Domain Tabular, Group Aggregations & Text Feature Engineering
# -----------------------------------------------------------------------------
docs_col = next(
    (c for c in num_cols if "total docs" in c.lower() or "docs" in c.lower()), None
)
cites_col = next(
    (c for c in num_cols if "total cites" in c.lower() or "cites" in c.lower()), None
)
citable_col = next((c for c in num_cols if "citable" in c.lower()), None)
refs_col = next(
    (c for c in num_cols if "refs" in c.lower() or "total refs" in c.lower()), None
)
sjr_col = next((c for c in num_cols if "sjr" in c.lower()), None)


def engineer_tabular_features(df):
    fe_df = pd.DataFrame(index=df.index)

    for col in num_cols:
        val = df[col].astype(float)
        fe_df[col] = val
        fe_df[f"{col}_log1p"] = np.log1p(np.maximum(val.fillna(0), 0))
        fe_df[f"{col}_sqrt"] = np.sqrt(np.maximum(val.fillna(0), 0))

    if docs_col and cites_col:
        fe_df["cites_per_total_doc"] = fe_df[cites_col] / (fe_df[docs_col] + 1e-5)
    if docs_col and citable_col:
        fe_df["citable_ratio"] = fe_df[citable_col] / (fe_df[docs_col] + 1e-5)
        fe_df["non_citable_ratio"] = (fe_df[docs_col] - fe_df[citable_col]) / (
            fe_df[docs_col] + 1e-5
        )
    if citable_col and cites_col:
        fe_df["cites_per_citable_doc"] = fe_df[cites_col] / (fe_df[citable_col] + 1e-5)
    if docs_col and refs_col:
        fe_df["refs_per_doc"] = fe_df[refs_col] / (fe_df[docs_col] + 1e-5)
    if cites_col and refs_col:
        fe_df["cite_to_ref_density"] = fe_df[cites_col] / (fe_df[refs_col] + 1e-5)
    if sjr_col and docs_col:
        fe_df["prestige_volume"] = fe_df[sjr_col] * fe_df[docs_col]
        fe_df["prestige_volume_log"] = np.log1p(
            np.maximum(fe_df["prestige_volume"].fillna(0), 0)
        )
    if sjr_col and cites_col:
        fe_df["sjr_cites_interaction"] = fe_df[sjr_col] * fe_df[cites_col]
        fe_df["cites_to_sjr_ratio"] = fe_df[cites_col] / (fe_df[sjr_col] + 1e-5)
    if sjr_col and refs_col:
        fe_df["sjr_refs_interaction"] = fe_df[sjr_col] * fe_df[refs_col]

    fe_df["num_missing"] = df[num_cols].isna().sum(axis=1)
    return fe_df


X_train_num = engineer_tabular_features(X_train_raw)
X_test_num = engineer_tabular_features(X_test_raw)

# Group Aggregations based on domain priors
group_cols = [
    c
    for c in cat_cols
    if any(k in c.lower() for k in ["country", "publisher", "type", "category", "area"])
]
metric_cols = [c for c in [sjr_col, cites_col, docs_col, refs_col] if c is not None]

group_train_dfs = []
group_test_dfs = []

for g in group_cols:
    for m in metric_cols:
        grp = X_train_raw.groupby(X_train_raw[g].astype(str))[m]
        mean_map = grp.mean().to_dict()
        std_map = grp.std().fillna(0).to_dict()

        g_train_mean = X_train_raw[g].astype(str).map(mean_map).fillna(X_train_raw[m].mean())
        g_train_std = X_train_raw[g].astype(str).map(std_map).fillna(0)
        g_test_mean = X_test_raw[g].astype(str).map(mean_map).fillna(X_train_raw[m].mean())
        g_test_std = X_test_raw[g].astype(str).map(std_map).fillna(0)

        df_tr = pd.DataFrame(
            {
                f"{g}_{m}_mean": g_train_mean,
                f"{g}_{m}_std": g_train_std,
                f"{m}_ratio_to_{g}_mean": X_train_num[m] / (g_train_mean + 1e-5),
            },
            index=X_train_raw.index,
        )
        df_te = pd.DataFrame(
            {
                f"{g}_{m}_mean": g_test_mean,
                f"{g}_{m}_std": g_test_std,
                f"{m}_ratio_to_{g}_mean": X_test_num[m] / (g_test_mean + 1e-5),
            },
            index=X_test_raw.index,
        )
        group_train_dfs.append(df_tr)
        group_test_dfs.append(df_te)

# Text & Category NLP Feature Engineering with 30 SVD components
text_dfs_train = []
text_dfs_test = []

candidate_text_cols = list(
    dict.fromkeys(
        text_cols
        + [
            cat
            for cat in cat_cols
            if "category" in cat.lower()
            or "area" in cat.lower()
            or "title" in cat.lower()
        ]
    )
)

for c in candidate_text_cols:
    s_tr = X_train_raw[c].fillna("").astype(str)
    s_te = X_test_raw[c].fillna("").astype(str)

    q1_tr = s_tr.str.count(r"Q1")
    q2_tr = s_tr.str.count(r"Q2")
    q3_tr = s_tr.str.count(r"Q3")
    q4_tr = s_tr.str.count(r"Q4")
    total_q_tr = q1_tr + q2_tr + q3_tr + q4_tr + 1e-5

    q1_te = s_te.str.count(r"Q1")
    q2_te = s_te.str.count(r"Q2")
    q3_te = s_te.str.count(r"Q3")
    q4_te = s_te.str.count(r"Q4")
    total_q_te = q1_te + q2_te + q3_te + q4_te + 1e-5

    text_dfs_train.append(
        pd.DataFrame(
            {
                f"{c}_count_Q1": q1_tr,
                f"{c}_count_Q2": q2_tr,
                f"{c}_count_Q3": q3_tr,
                f"{c}_count_Q4": q4_tr,
                f"{c}_Q1_ratio": q1_tr / total_q_tr,
                f"{c}_high_quartile_sum": q1_tr + q2_tr,
                f"{c}_has_Q1": (q1_tr > 0).astype(float),
                f"{c}_char_len": s_tr.str.len(),
                f"{c}_word_count": s_tr.str.split().str.len(),
            },
            index=X_train_raw.index,
        )
    )
    text_dfs_test.append(
        pd.DataFrame(
            {
                f"{c}_count_Q1": q1_te,
                f"{c}_count_Q2": q2_te,
                f"{c}_count_Q3": q3_te,
                f"{c}_count_Q4": q4_te,
                f"{c}_Q1_ratio": q1_te / total_q_te,
                f"{c}_high_quartile_sum": q1_te + q2_te,
                f"{c}_has_Q1": (q1_te > 0).astype(float),
                f"{c}_char_len": s_te.str.len(),
                f"{c}_word_count": s_te.str.split().str.len(),
            },
            index=X_test_raw.index,
        )
    )

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(2, 4),
        analyzer="char_wb",
        max_features=4000,
        sublinear_tf=True,
        min_df=3,
    )

    word_tr = word_vec.fit_transform(s_tr)
    word_te = word_vec.transform(s_te)
    char_tr = char_vec.fit_transform(s_tr)
    char_te = char_vec.transform(s_te)

    combined_tr = hstack([word_tr, char_tr]).tocsr()
    combined_te = hstack([word_te, char_te]).tocsr()

    n_comp = min(30, max(0, combined_tr.shape[1] - 1))
    if n_comp >= 2:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_tr = svd.fit_transform(combined_tr)
        svd_te = svd.transform(combined_te)

        svd_cols = [f"{c}_svd_{i}" for i in range(n_comp)]
        text_dfs_train.append(
            pd.DataFrame(svd_tr, columns=svd_cols, index=X_train_raw.index)
        )
        text_dfs_test.append(
            pd.DataFrame(svd_te, columns=svd_cols, index=X_test_raw.index)
        )

# Frequency Encoding for Categoricals
cat_dfs_train = []
cat_dfs_test = []

for c in cat_cols:
    freq_map = X_train_raw[c].astype(str).value_counts(normalize=True).to_dict()
    cat_dfs_train.append(
        pd.DataFrame(
            {f"{c}_freq": X_train_raw[c].astype(str).map(freq_map).fillna(0)},
            index=X_train_raw.index,
        )
    )
    cat_dfs_test.append(
        pd.DataFrame(
            {f"{c}_freq": X_test_raw[c].astype(str).map(freq_map).fillna(0)},
            index=X_test_raw.index,
        )
    )

X_train_full = pd.concat(
    [X_train_num] + group_train_dfs + text_dfs_train + cat_dfs_train, axis=1
)
X_test_full = pd.concat(
    [X_test_num] + group_test_dfs + text_dfs_test + cat_dfs_test, axis=1
)

X_train_full = X_train_full.loc[:, ~X_train_full.columns.duplicated()]
X_test_full = X_test_full.loc[:, ~X_test_full.columns.duplicated()]

for c in X_train_full.columns:
    if c not in X_test_full.columns:
        X_test_full[c] = 0.0
X_test_full = X_test_full[X_train_full.columns]

# Sanitize feature column names for gradient boosting libraries
orig_cols = list(X_train_full.columns)
clean_cols = [re.sub(r"[\[\]<>, \"]+", "_", str(c)) for c in orig_cols]
col_map = dict(zip(orig_cols, clean_cols))
X_train_full = X_train_full.rename(columns=col_map)
X_test_full = X_test_full.rename(columns=col_map)
feature_cols = list(X_train_full.columns)


# -----------------------------------------------------------------------------
# 3. Model Architectures & Definitions
# -----------------------------------------------------------------------------
def get_lgbm_model(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(seed: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


# -----------------------------------------------------------------------------
# 4. Cross-Validation & Multi-Model Training
# -----------------------------------------------------------------------------
n_splits = 5
num_bins = min(10, len(np.unique(y_train)))
binned_y = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train), dtype=np.float64)
oof_xgb = np.zeros(len(y_train), dtype=np.float64)
oof_cat = np.zeros(len(y_train), dtype=np.float64)
oof_nn = np.zeros(len(y_train), dtype=np.float64)

test_preds_lgb = np.zeros(len(X_test_full), dtype=np.float64)
test_preds_xgb = np.zeros(len(X_test_full), dtype=np.float64)
test_preds_cat = np.zeros(len(X_test_full), dtype=np.float64)
test_preds_nn = np.zeros(len(X_test_full), dtype=np.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, binned_y)):
    X_tr_df = X_train_full.iloc[train_idx][feature_cols]
    y_tr = y_train[train_idx]
    X_va_df = X_train_full.iloc[val_idx][feature_cols]
    y_va = y_train[val_idx]
    X_te_df = X_test_full[feature_cols]

    # --- LightGBM ---
    lgb_model = get_lgbm_model(seed=42 + fold_idx)
    lgb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred_lgb = lgb_model.predict(X_va_df)
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_te_df) / n_splits

    # --- XGBoost ---
    xgb_model = get_xgb_model(seed=42 + fold_idx)
    xgb_model.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)
    val_pred_xgb = xgb_model.predict(X_va_df)
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_te_df) / n_splits

    # --- CatBoost ---
    cat_model = get_catboost_model(seed=42 + fold_idx)
    cat_model.fit(
        X_tr_df, y_tr, eval_set=(X_va_df, y_va), early_stopping_rounds=50, verbose=False
    )
    val_pred_cat = cat_model.predict(X_va_df)
    oof_cat[val_idx] = val_pred_cat
    test_preds_cat += cat_model.predict(X_te_df) / n_splits

    # --- Tabular ResNet (PyTorch) ---
    scaler = StandardScaler()
    X_tr_nn = np.nan_to_num(scaler.fit_transform(X_tr_df), nan=0.0)
    X_va_nn = np.nan_to_num(scaler.transform(X_va_df), nan=0.0)
    X_te_nn = np.nan_to_num(scaler.transform(X_te_df), nan=0.0)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_nn, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )

    nn_model = TabularResNet(
        input_dim=X_tr_nn.shape[1], hidden_dim=256, num_blocks=2, dropout_rate=0.2
    ).to(device)
    epochs = 40
    criterion = nn.MSELoss()
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_nn_val_pred = None
    best_nn_te_pred = None

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = nn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            va_t = torch.tensor(X_va_nn, dtype=torch.float32).to(device)
            va_pred = nn_model(va_t).cpu().numpy()
            val_loss = mean_squared_error(y_va, va_pred)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_val_pred = va_pred
                te_t = torch.tensor(X_te_nn, dtype=torch.float32).to(device)
                best_nn_te_pred = nn_model(te_t).cpu().numpy()

    oof_nn[val_idx] = best_nn_val_pred
    test_preds_nn += best_nn_te_pred / n_splits

    fold_rmse = np.sqrt(
        mean_squared_error(
            y_va,
            (val_pred_lgb + val_pred_xgb + val_pred_cat + best_nn_val_pred) / 4.0,
        )
    )
    print(f"Fold {fold_idx + 1}/{n_splits} Blend RMSE: {fold_rmse:.4f}")

# -----------------------------------------------------------------------------
# 5. Ensemble Optimization & Final Submission
# -----------------------------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
)


def loss_func(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y_train, np.maximum(pred, 0.0)))


initial_weights = np.array([0.35, 0.35, 0.20, 0.10])
bounds = [(0.0, 1.0)] * 4
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
opt_res = minimize(
    loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
)

opt_weights = opt_res.x / np.sum(opt_res.x)

final_oof = np.maximum(np.dot(oof_matrix, opt_weights), 0.0)
final_test = np.maximum(np.dot(test_matrix, opt_weights), 0.0)

final_score = np.sqrt(mean_squared_error(y_train, final_oof))

sub_df = pd.DataFrame({"id": np.arange(len(final_test)), "H index": final_test})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")