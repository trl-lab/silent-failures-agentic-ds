import os
import pickle
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set device and seeds for strict reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Directory configuration
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"
os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

# 1. Load Data
train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Original Principal Amount (US$)"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' missing from train_df."

y_train_raw = train_df[target_col].values.astype(np.float64)
y_train_log = np.log1p(np.maximum(0.0, y_train_raw))

df_train_feat = train_df.drop(columns=[target_col]).copy()
df_test_feat = test_df.copy()

df_train_feat["is_train"] = 1
df_test_feat["is_train"] = 0

combined = pd.concat([df_train_feat, df_test_feat], axis=0, ignore_index=True)

# 2. Date and Temporal Feature Engineering
date_cols = [
    c
    for c in combined.columns
    if any(
        k in c.lower() for k in ["date", "approval", "signing", "effective", "closed"]
    )
]

date_parsed_df = pd.DataFrame(index=combined.index)
for col in date_cols:
    parsed = pd.to_datetime(combined[col], errors="coerce")
    date_parsed_df[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(np.float32)
    date_parsed_df[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(np.float32)
    date_parsed_df[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(np.float32)
    date_parsed_df[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(
        np.float32
    )
    date_parsed_df[f"{col}_quarter"] = parsed.dt.quarter.fillna(-1).astype(np.float32)
    date_parsed_df[f"{col}_sin_month"] = np.sin(
        2 * np.pi * parsed.dt.month / 12.0
    ).fillna(0.0)
    date_parsed_df[f"{col}_cos_month"] = np.cos(
        2 * np.pi * parsed.dt.month / 12.0
    ).fillna(0.0)

if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            d1 = pd.to_datetime(combined[col1], errors="coerce")
            d2 = pd.to_datetime(combined[col2], errors="coerce")
            diff_days = (d2 - d1).dt.days
            date_parsed_df[f"days_diff_{col1}_to_{col2}"] = diff_days.fillna(
                -999
            ).astype(np.float32)

# 3. Categorical & Text Feature Extraction
text_cols = [
    c
    for c in combined.columns
    if combined[c].dtype == "object" or combined[c].dtype == "string"
]
text_cols = [c for c in text_cols if c not in date_cols and c != "is_train"]

train_mask = combined["is_train"] == 1
test_mask = combined["is_train"] == 0

combined_text = combined[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
train_text = combined_text[train_mask]
test_text = combined_text[test_mask]

from scipy.sparse import hstack

tfidf_word = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", analyzer="word")
tfidf_char = TfidfVectorizer(max_features=5000, ngram_range=(2, 4), analyzer="char")

tfidf_mat_train_word = tfidf_word.fit_transform(train_text)
tfidf_mat_test_word = tfidf_word.transform(test_text)
tfidf_mat_train_char = tfidf_char.fit_transform(train_text)
tfidf_mat_test_char = tfidf_char.transform(test_text)

tfidf_mat_train = hstack([tfidf_mat_train_word, tfidf_mat_train_char])
tfidf_mat_test = hstack([tfidf_mat_test_word, tfidf_mat_test_char])

svd = TruncatedSVD(n_components=32, random_state=42)
svd_train = svd.fit_transform(tfidf_mat_train)
svd_test = svd.transform(tfidf_mat_test)
svd_df = pd.DataFrame(np.vstack([svd_train, svd_test]), columns=[f"svd_text_{i}" for i in range(32)])

freq_df = pd.DataFrame(index=combined.index)
for col in text_cols:
    freq_map = combined.loc[train_mask, col].value_counts(normalize=True).to_dict()
    freq_df[f"{col}_freq"] = combined[col].map(freq_map).fillna(0.0).astype(np.float32)

ord_df = pd.DataFrame(index=combined.index)
for col in text_cols:
    le = LabelEncoder()
    le.fit(combined[col].astype(str).fillna("missing"))
    ord_df[f"{col}_encoded"] = le.transform(combined[col].astype(str).fillna("missing"))

# 4. Numerical Features & Skew Transformation
num_cols = [
    c
    for c in combined.columns
    if c not in text_cols and c not in date_cols and c != "is_train"
]
num_df = combined[num_cols].copy()

for col in num_cols:
    if num_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        val_min = num_df.loc[train_mask, col].min()
        if val_min >= 0:
            skew = num_df.loc[train_mask, col].skew()
            if skew > 1.0:
                num_df[f"{col}_log1p"] = np.log1p(num_df[col])

key_cat_cols = [
    c for c in text_cols if combined[c].nunique() > 1 and combined[c].nunique() < 500
]
group_agg_dfs = []
for cat_col in key_cat_cols:
    for num_c in num_cols:
        if num_df[num_c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            train_grp = combined.loc[train_mask].groupby(cat_col)[num_c]
            mean_map = combined[cat_col].map(train_grp.mean()).astype(np.float32)
            std_map = combined[cat_col].map(train_grp.std()).fillna(0.0).astype(np.float32)
            group_agg_dfs.append(pd.Series(mean_map, name=f"{num_c}_grp_mean_{cat_col}"))
            group_agg_dfs.append(pd.Series(std_map, name=f"{num_c}_grp_std_{cat_col}"))

if group_agg_dfs:
    group_agg_df = pd.concat(group_agg_dfs, axis=1)
    all_features = pd.concat([num_df, group_agg_df, date_parsed_df, svd_df, freq_df, ord_df], axis=1)
else:
    all_features = pd.concat([num_df, date_parsed_df, svd_df, freq_df, ord_df], axis=1)

all_features = all_features.replace([np.inf, -np.inf], np.nan)

X_train = all_features[train_mask].reset_index(drop=True)
X_test = all_features[test_mask].reset_index(drop=True)
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians).fillna(0.0)
X_test = X_test.fillna(train_medians).fillna(0.0)

# 5. Out-of-fold Target Encoding (Leak-Free)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
target_enc_cols = [
    c for c in text_cols if combined[c].nunique() > 3 and combined[c].nunique() < 1000
]

te_train = pd.DataFrame(index=X_train.index)
te_test = pd.DataFrame(index=X_test.index)

global_mean = float(np.mean(y_train_log))
global_std = float(np.std(y_train_log))
global_min = float(np.min(y_train_log))
global_max = float(np.max(y_train_log))
m_smooth = 10.0

for col in target_enc_cols:
    for stat in ["mean", "std", "min", "max"]:
        te_train[f"{col}_te_{stat}"] = np.nan

    test_cat_series = df_test_feat[col].astype(str)
    test_folds_stat = {"mean": [], "std": [], "min": [], "max": []}

    for train_idx, val_idx in kf.split(X_train):
        cat_tr = df_train_feat.iloc[train_idx][col].astype(str)
        y_tr = y_train_log[train_idx]

        tr_series = pd.Series(y_tr, index=cat_tr.index)
        cat_counts = cat_tr.value_counts()
        cat_means = tr_series.groupby(cat_tr).mean()
        smoothed_means = (cat_counts * cat_means + m_smooth * global_mean) / (cat_counts + m_smooth)

        cat_stds = tr_series.groupby(cat_tr).std().fillna(global_std)
        cat_mins = tr_series.groupby(cat_tr).min().fillna(global_min)
        cat_maxs = tr_series.groupby(cat_tr).max().fillna(global_max)

        cat_val = df_train_feat.iloc[val_idx][col].astype(str)

        te_train.iloc[val_idx, te_train.columns.get_loc(f"{col}_te_mean")] = cat_val.map(smoothed_means).fillna(global_mean)
        te_train.iloc[val_idx, te_train.columns.get_loc(f"{col}_te_std")] = cat_val.map(cat_stds).fillna(global_std)
        te_train.iloc[val_idx, te_train.columns.get_loc(f"{col}_te_min")] = cat_val.map(cat_mins).fillna(global_min)
        te_train.iloc[val_idx, te_train.columns.get_loc(f"{col}_te_max")] = cat_val.map(cat_maxs).fillna(global_max)

        test_folds_stat["mean"].append(test_cat_series.map(smoothed_means).fillna(global_mean))
        test_folds_stat["std"].append(test_cat_series.map(cat_stds).fillna(global_std))
        test_folds_stat["min"].append(test_cat_series.map(cat_mins).fillna(global_min))
        test_folds_stat["max"].append(test_cat_series.map(cat_maxs).fillna(global_max))

    for stat in ["mean", "std", "min", "max"]:
        te_test[f"{col}_te_{stat}"] = np.mean(test_folds_stat[stat], axis=0)

X_train = pd.concat([X_train, te_train], axis=1)
X_test = pd.concat([X_test, te_test], axis=1)

global_target_mean = float(np.mean(y_train_log))
X_train = X_train.fillna(global_target_mean)
X_test = X_test.fillna(global_target_mean)

for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype(np.float32)
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce").astype(np.float32)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians).fillna(0.0)
X_test = X_test.fillna(train_medians).fillna(0.0)

n_samples_train = len(X_train)
n_samples_test = len(X_test)
n_features = X_train.shape[1]

# Prediction storage arrays
oof_lgb = np.zeros(n_samples_train, dtype=np.float64)
oof_xgb = np.zeros(n_samples_train, dtype=np.float64)
oof_cb = np.zeros(n_samples_train, dtype=np.float64)
oof_nn = np.zeros(n_samples_train, dtype=np.float64)

test_lgb = np.zeros(n_samples_test, dtype=np.float64)
test_xgb = np.zeros(n_samples_test, dtype=np.float64)
test_cb = np.zeros(n_samples_test, dtype=np.float64)
test_nn = np.zeros(n_samples_test, dtype=np.float64)


# PyTorch Tabular ResNet Architecture
class TabularResBlock(nn.Module):

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=128, num_blocks=2, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# 6. Model Training Loop across CV Splits
cv_splits = list(kf.split(X_train))
num_folds = len(cv_splits)

sample_weights_full = 1.0 + (y_train_raw / np.mean(y_train_raw))

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_log[val_idx]
    w_tr = sample_weights_full[train_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test) / num_folds

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_test) / num_folds

    # CatBoost
    cb_model = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = cb_model.predict(X_va)
    test_cb += cb_model.predict(X_test) / num_folds

    # PyTorch Tabular ResNet
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.values)
    X_va_s = scaler.transform(X_va.values)
    X_te_s = scaler.transform(X_test.values)

    ds_tr = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader_tr = DataLoader(
        ds_tr, batch_size=256, shuffle=True, drop_last=False, num_workers=2
    )

    nn_model = TabularResNet(in_features=n_features).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_weights = None
    patience_cnt = 0

    for epoch in range(60):
        nn_model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(X_va_s, dtype=torch.float32).to(device)
            val_preds_t = nn_model(val_x_t)
            val_loss = criterion(
                val_preds_t, torch.tensor(y_va, dtype=torch.float32).to(device)
            ).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                break

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        val_x_t = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        oof_nn[val_idx] = nn_model(val_x_t).cpu().numpy().astype(np.float64)

        test_x_t = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        test_nn += nn_model(test_x_t).cpu().numpy().astype(np.float64) / num_folds

    print(f"Fold {fold + 1}/{num_folds} completed.")

# 7. SLSQP Ensemble Optimization on Raw USD RMSE
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cb, test_nn])


def raw_usd_rmse_objective(params):
    w = params[:4]
    alpha = params[4]
    w_norm = w / (np.sum(w) + 1e-8)
    pred_log = oof_matrix @ w_norm
    pred_raw = np.maximum(0.0, alpha * np.expm1(pred_log))
    return np.sqrt(np.mean((pred_raw - y_train_raw) ** 2))


init_params = np.array([0.25, 0.25, 0.25, 0.25, 1.0])
bounds = [(0.0, 1.0)] * 4 + [(0.5, 2.0)]
constraints = {"type": "eq", "fun": lambda p: np.sum(p[:4]) - 1.0}

opt_res = minimize(
    raw_usd_rmse_objective,
    init_params,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

if opt_res.success:
    opt_w = opt_res.x[:4] / np.sum(opt_res.x[:4])
    opt_alpha = opt_res.x[4]
else:
    opt_w = np.array([0.25, 0.25, 0.25, 0.25])
    opt_alpha = 1.0

# Compute Final Validation Score
final_oof_log = oof_matrix @ opt_w
final_oof_raw = np.maximum(0.0, opt_alpha * np.expm1(final_oof_log))
final_val_rmse = float(np.sqrt(np.mean((final_oof_raw - y_train_raw) ** 2)))

# 8. Final Test Predictions and Submission Export
final_test_log = test_matrix @ opt_w
final_test_raw = np.maximum(0.0, opt_alpha * np.expm1(final_test_log))

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_raw), dtype=np.int64),
        "Original Principal Amount (US$)": final_test_raw,
    }
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not written!"
assert len(submission_df) == len(
    X_test
), f"Row count mismatch: {len(submission_df)} vs {len(X_test)}"
assert list(submission_df.columns) == [
    "id",
    "Original Principal Amount (US$)",
], f"Incorrect column names: {submission_df.columns}"

print(f"Final Validation Score: {final_val_rmse}")