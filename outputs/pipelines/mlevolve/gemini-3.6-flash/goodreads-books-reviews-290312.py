import gc
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data & Create Train/Val Split
# ---------------------------------------------------------
raw_train = pd.read_parquet("./input/train.parquet")
raw_test = pd.read_parquet("./input/test.parquet")

# Clean target
raw_train = raw_train.dropna(subset=["rating"]).reset_index(drop=True)
raw_train["rating"] = raw_train["rating"].astype(np.float32)

train_df, val_df = train_test_split(
    raw_train, test_size=0.2, random_state=42, stratify=raw_train["rating"].astype(int)
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = raw_test.reset_index(drop=True)

y_train = train_df["rating"].values
y_val = val_df["rating"].values

# ---------------------------------------------------------
# 2. Feature Engineering Helpers
# ---------------------------------------------------------
MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def extract_date_features(df, col_name):
    s = df[col_name].fillna("Mon Jan 01 00:00:00 -0000 1900").astype(str)
    year = pd.to_numeric(s.str[-4:], errors="coerce").fillna(1900)
    month = s.str[4:7].map(MONTH_MAP).fillna(1)
    day = pd.to_numeric(s.str[8:10], errors="coerce").fillna(1)
    hour = pd.to_numeric(s.str[11:13], errors="coerce").fillna(0)
    is_missing = df[col_name].isna().astype(np.float32)

    return pd.DataFrame(
        {
            f"{col_name}_year": year.astype(np.float32),
            f"{col_name}_month": month.astype(np.float32),
            f"{col_name}_day": day.astype(np.float32),
            f"{col_name}_hour": hour.astype(np.float32),
            f"{col_name}_is_null": is_missing,
        },
        index=df.index,
    )


def extract_time_deltas(df):
    read_at_dt = pd.to_datetime(df["read_at"], utc=True, errors="coerce")
    started_at_dt = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    date_added_dt = pd.to_datetime(df["date_added"], utc=True, errors="coerce")
    date_updated_dt = pd.to_datetime(
        df["date_updated"], utc=True, errors="coerce"
    )

    read_minus_started = (
        read_at_dt - started_at_dt
    ).dt.total_seconds() / 86400.0
    read_minus_added = (read_at_dt - date_added_dt).dt.total_seconds() / 86400.0
    updated_minus_added = (
        date_updated_dt - date_added_dt
    ).dt.total_seconds() / 86400.0

    return pd.DataFrame(
        {
            "delta_read_minus_started": read_minus_started.astype(np.float32),
            "delta_read_minus_added": read_minus_added.astype(np.float32),
            "delta_updated_minus_added": updated_minus_added.astype(np.float32),
        },
        index=df.index,
    )


def extract_text_metrics(df, col_name="review_text"):
    s = df[col_name].fillna("").astype(str)
    char_len = s.str.len()
    word_count = s.str.split().str.len()
    avg_word_len = char_len / (word_count + 1.0)

    num_excl = s.str.count("!")
    num_quest = s.str.count(r"\?")
    num_digits = s.str.count(r"\d")
    num_upper = s.str.count(r"[A-Z]")
    upper_ratio = num_upper / (char_len + 1.0)

    is_boilerplate = (
        s.str.lower()
        .str.contains("review to come|to be updated|tbr|will update")
        .astype(np.float32)
    )

    return pd.DataFrame(
        {
            "text_char_len": char_len.astype(np.float32),
            "text_word_count": word_count.astype(np.float32),
            "text_avg_word_len": avg_word_len.astype(np.float32),
            "text_num_excl": num_excl.astype(np.float32),
            "text_num_quest": num_quest.astype(np.float32),
            "text_num_digits": num_digits.astype(np.float32),
            "text_upper_ratio": upper_ratio.astype(np.float32),
            "text_is_boilerplate": is_boilerplate,
        },
        index=df.index,
    )


def compute_cf_bias_stats(
    train_data, val_data, test_data, col, target_col="rating", smoothing=10.0
):
    global_mean = np.float32(train_data[target_col].mean())
    global_std = np.float32(train_data[target_col].std())

    tr_mean = np.zeros(len(train_data), dtype=np.float32)
    tr_std = np.zeros(len(train_data), dtype=np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(train_data):
        tr = train_data.iloc[tr_idx]
        va = train_data.iloc[va_idx]

        stats = tr.groupby(col)[target_col].agg(["count", "mean", "std"])
        mean_smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        std_filled = stats["std"].fillna(global_std)
        std_smooth = (stats["count"] * std_filled + smoothing * global_std) / (
            stats["count"] + smoothing
        )

        tr_mean[va_idx] = (
            va[col].map(mean_smooth).fillna(global_mean).values.astype(np.float32)
        )
        tr_std[va_idx] = (
            va[col].map(std_smooth).fillna(global_std).values.astype(np.float32)
        )

    full_stats = train_data.groupby(col)[target_col].agg(["count", "mean", "std"])
    full_mean_smooth = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    full_std_filled = full_stats["std"].fillna(global_std)
    full_std_smooth = (
        full_stats["count"] * full_std_filled + smoothing * global_std
    ) / (full_stats["count"] + smoothing)

    va_mean = (
        val_data[col]
        .map(full_mean_smooth)
        .fillna(global_mean)
        .values.astype(np.float32)
    )
    va_std = (
        val_data[col]
        .map(full_std_smooth)
        .fillna(global_std)
        .values.astype(np.float32)
    )

    te_mean = (
        test_data[col]
        .map(full_mean_smooth)
        .fillna(global_mean)
        .values.astype(np.float32)
    )
    te_std = (
        test_data[col]
        .map(full_std_smooth)
        .fillna(global_std)
        .values.astype(np.float32)
    )

    return tr_mean, va_mean, te_mean, tr_std, va_std, te_std


def compute_count_encoding(train_data, val_data, test_data, col):
    counts = train_data[col].value_counts()
    tr_cnt = train_data[col].map(counts).fillna(0).astype(np.float32)
    va_cnt = val_data[col].map(counts).fillna(0).astype(np.float32)
    te_cnt = test_data[col].map(counts).fillna(0).astype(np.float32)
    return tr_cnt.values, va_cnt.values, te_cnt.values


# ---------------------------------------------------------
# 3. Assemble Feature DataFrames
# ---------------------------------------------------------
train_feat = pd.DataFrame(index=train_df.index)
val_feat = pd.DataFrame(index=val_df.index)
test_feat = pd.DataFrame(index=test_df.index)

# A. Numerical & Engagement Features
for df_split, feat_df in [
    (train_df, train_feat),
    (val_df, val_feat),
    (test_df, test_feat),
]:
    votes = np.maximum(df_split["n_votes"].fillna(0).values, 0)
    comments = np.maximum(df_split["n_comments"].fillna(0).values, 0)

    feat_df["n_votes"] = votes.astype(np.float32)
    feat_df["n_comments"] = comments.astype(np.float32)
    feat_df["log_n_votes"] = np.log1p(votes).astype(np.float32)
    feat_df["log_n_comments"] = np.log1p(comments).astype(np.float32)
    feat_df["comments_per_vote"] = (comments / (votes + 1.0)).astype(np.float32)

# B. Text Metrics
tr_text_m = extract_text_metrics(train_df)
va_text_m = extract_text_metrics(val_df)
te_text_m = extract_text_metrics(test_df)

for c in tr_text_m.columns:
    train_feat[c] = tr_text_m[c].values
    val_feat[c] = va_text_m[c].values
    test_feat[c] = te_text_m[c].values

train_feat["votes_per_word"] = (
    train_feat["n_votes"] / (train_feat["text_word_count"] + 1.0)
).astype(np.float32)
val_feat["votes_per_word"] = (
    val_feat["n_votes"] / (val_feat["text_word_count"] + 1.0)
).astype(np.float32)
test_feat["votes_per_word"] = (
    test_feat["n_votes"] / (test_feat["text_word_count"] + 1.0)
).astype(np.float32)

# C. Date Features
date_cols = ["date_added", "date_updated", "read_at", "started_at"]
for d_col in date_cols:
    tr_d = extract_date_features(train_df, d_col)
    va_d = extract_date_features(val_df, d_col)
    te_d = extract_date_features(test_df, d_col)

    for c in tr_d.columns:
        train_feat[c] = tr_d[c].values
        val_feat[c] = va_d[c].values
        test_feat[c] = te_d[c].values

# D. Target and Count Encodings
(
    train_feat["user_target_enc"],
    val_feat["user_target_enc"],
    test_feat["user_target_enc"],
    _,
    _,
    _,
) = compute_cf_bias_stats(train_df, val_df, test_df, "user_id", smoothing=10.0)

(
    train_feat["book_target_enc"],
    val_feat["book_target_enc"],
    test_feat["book_target_enc"],
    _,
    _,
    _,
) = compute_cf_bias_stats(train_df, val_df, test_df, "book_id", smoothing=5.0)

train_feat["user_count"], val_feat["user_count"], test_feat["user_count"] = (
    compute_count_encoding(train_df, val_df, test_df, "user_id")
)

train_feat["book_count"], val_feat["book_count"], test_feat["book_count"] = (
    compute_count_encoding(train_df, val_df, test_df, "book_id")
)

# E. TF-IDF & SVD Feature Engineering
tfidf = TfidfVectorizer(
    max_features=15000, min_df=3, ngram_range=(1, 2), sublinear_tf=True
)
X_tr_tfidf = tfidf.fit_transform(train_df["review_text"].fillna(""))
X_va_tfidf = tfidf.transform(val_df["review_text"].fillna(""))
X_te_tfidf = tfidf.transform(test_df["review_text"].fillna(""))

svd = TruncatedSVD(n_components=16, random_state=42)
svd_tr = svd.fit_transform(X_tr_tfidf)
svd_va = svd.transform(X_va_tfidf)
svd_te = svd.transform(X_te_tfidf)

for i in range(16):
    train_feat[f"svd_{i}"] = svd_tr[:, i].astype(np.float32)
    val_feat[f"svd_{i}"] = svd_va[:, i].astype(np.float32)
    test_feat[f"svd_{i}"] = svd_te[:, i].astype(np.float32)

# F. Out-Of-Fold Ridge Regression on TF-IDF Text
ridge_oof = np.zeros(len(train_df), dtype=np.float32)
ridge_val = np.zeros(len(val_df), dtype=np.float32)
ridge_test = np.zeros(len(test_df), dtype=np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_i, va_i in kf.split(train_df):
    r_model = Ridge(alpha=1.0)
    r_model.fit(X_tr_tfidf[tr_i], y_train[tr_i])
    ridge_oof[va_i] = r_model.predict(X_tr_tfidf[va_i])
    ridge_val += r_model.predict(X_va_tfidf) / 5.0
    ridge_test += r_model.predict(X_te_tfidf) / 5.0

train_feat["tfidf_ridge_pred"] = ridge_oof
val_feat["tfidf_ridge_pred"] = ridge_val
test_feat["tfidf_ridge_pred"] = ridge_test

del X_tr_tfidf, X_va_tfidf, X_te_tfidf, svd_tr, svd_va, svd_te
gc.collect()


# ---------------------------------------------------------
# 4. Neural Network Architecture & Model Parameters
# ---------------------------------------------------------
class TabularResBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DeepTabularNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def build_pytorch_model(input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4):
    model = DeepTabularNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.25
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    return model, criterion, optimizer, scheduler


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_estimators": 1000,
    "learning_rate": 0.08,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

catboost_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1000,
    "learning_rate": 0.08,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "early_stopping_rounds": 50,
    "verbose": 0,
}

# ---------------------------------------------------------
# 5. Model Training & Validation
# ---------------------------------------------------------
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    train_feat,
    y_train,
    eval_set=[(val_feat, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
val_p_lgb = lgb_model.predict(val_feat)
test_p_lgb = lgb_model.predict(test_feat)

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    train_feat,
    y_train,
    eval_set=[(val_feat, y_val)],
    verbose=False,
)
val_p_xgb = xgb_model.predict(val_feat)
test_p_xgb = xgb_model.predict(test_feat)

cat_model = CatBoostRegressor(**catboost_params)
cat_model.fit(
    train_feat,
    y_train,
    eval_set=(val_feat, y_val),
    verbose=0,
)
val_p_cat = cat_model.predict(val_feat)
test_p_cat = cat_model.predict(test_feat)

# PyTorch Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(np.nan_to_num(train_feat.values))
X_va_scaled = scaler.transform(np.nan_to_num(val_feat.values))
X_te_scaled = scaler.transform(np.nan_to_num(test_feat.values))

tr_dataset = TensorDataset(
    torch.tensor(X_tr_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
va_dataset = TensorDataset(
    torch.tensor(X_va_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)

tr_loader = DataLoader(
    tr_dataset, batch_size=2048, shuffle=True, drop_last=False, num_workers=2
)
va_loader = DataLoader(va_dataset, batch_size=4096, shuffle=False, num_workers=2)

nn_model, criterion, optimizer, scheduler = build_pytorch_model(
    input_dim=X_tr_scaled.shape[1]
)
nn_model.to(device)

epochs = 10
best_nn_val_rmse = float("inf")
best_nn_val_preds = None

for epoch in range(epochs):
    nn_model.train()
    running_loss = 0.0
    for bx, by in tr_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = nn_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(by)
    scheduler.step()

    nn_model.eval()
    va_preds_list = []
    with torch.no_grad():
        for bx, _ in va_loader:
            bx = bx.to(device)
            out = nn_model(bx)
            va_preds_list.append(out.cpu().numpy())
    va_preds_epoch = np.concatenate(va_preds_list)
    val_rmse_epoch = np.sqrt(mean_squared_error(y_val, va_preds_epoch))

    if val_rmse_epoch < best_nn_val_rmse:
        best_nn_val_rmse = val_rmse_epoch
        best_nn_val_preds = va_preds_epoch
        torch.save(nn_model.state_dict(), "./working/best_nn_model.pt")

nn_model.load_state_dict(torch.load("./working/best_nn_model.pt"))
nn_model.eval()

te_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))
te_loader = DataLoader(te_dataset, batch_size=4096, shuffle=False, num_workers=2)

test_p_nn_list = []
with torch.no_grad():
    for (bx,) in te_loader:
        bx = bx.to(device)
        out = nn_model(bx)
        test_p_nn_list.append(out.cpu().numpy())

val_p_nn = best_nn_val_preds
test_p_nn = np.concatenate(test_p_nn_list)


# ---------------------------------------------------------
# 6. Optimization & Ensembling
# ---------------------------------------------------------
def loss_func(weights):
    w = weights / np.sum(weights)
    blend_val = w[0] * val_p_lgb + w[1] * val_p_xgb + w[2] * val_p_cat + w[3] * val_p_nn
    blend_val = np.clip(blend_val, 0.0, 5.0)
    return np.sqrt(mean_squared_error(y_val, blend_val))


init_weights = [0.35, 0.25, 0.25, 0.15]
bounds = [(0, 1)] * 4
res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)

best_weights = res.x / np.sum(res.x)

val_ensemble = (
    best_weights[0] * val_p_lgb
    + best_weights[1] * val_p_xgb
    + best_weights[2] * val_p_cat
    + best_weights[3] * val_p_nn
)
val_ensemble = np.clip(val_ensemble, 0.0, 5.0)

test_ensemble = (
    best_weights[0] * test_p_lgb
    + best_weights[1] * test_p_xgb
    + best_weights[2] * test_p_cat
    + best_weights[3] * test_p_nn
)
test_ensemble = np.clip(test_ensemble, 0.0, 5.0)

score = np.sqrt(mean_squared_error(y_val, val_ensemble))

# ---------------------------------------------------------
# 7. Submission Output
# ---------------------------------------------------------
submission = pd.DataFrame(
    {"id": np.arange(len(test_ensemble)), "rating": test_ensemble}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")