import gc
import json
import os
import pickle
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure working and submission directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==============================================================================
# 1. DATA LOADING AND MODALITY IDENTIFICATION
# ==============================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "availability"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

target_encoder = LabelEncoder()
train_df["target"] = target_encoder.fit_transform(train_df[target_col].astype(str))
num_classes = len(target_encoder.classes_)
target_mapping = {
    int(i): str(cls_name) for i, cls_name in enumerate(target_encoder.classes_)
}

feature_cols = [c for c in train_df.columns if c not in [target_col, "target"]]
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in feature_cols:
    sample_vals = train_df[col].dropna().head(100)
    if sample_vals.empty:
        continue

    if pd.api.types.is_datetime64_any_dtype(train_df[col]):
        date_cols.append(col)
        continue
    elif train_df[col].dtype == "object" or isinstance(sample_vals.iloc[0], str):
        is_date = False
        try:
            parsed = pd.to_datetime(sample_vals, errors="coerce")
            if parsed.notna().mean() > 0.8:
                is_date = True
        except Exception:
            pass

        if is_date:
            date_cols.append(col)
        else:
            avg_len = sample_vals.astype(str).str.len().mean()
            nunique = train_df[col].nunique()
            if avg_len > 35 or nunique > 500:
                text_cols.append(col)
            else:
                cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def extract_date_features(df, date_columns):
    res_df = pd.DataFrame(index=df.index)
    date_series_dict = {}

    for col in date_columns:
        dt_s = pd.to_datetime(df[col], errors="coerce")
        date_series_dict[col] = dt_s

        res_df[f"{col}_year"] = dt_s.dt.year.fillna(-1).astype(np.int32)
        res_df[f"{col}_month"] = dt_s.dt.month.fillna(-1).astype(np.int32)
        res_df[f"{col}_day"] = dt_s.dt.day.fillna(-1).astype(np.int32)
        res_df[f"{col}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1).astype(np.int32)
        res_df[f"{col}_dayofyear"] = dt_s.dt.dayofyear.fillna(-1).astype(np.int32)
        res_df[f"{col}_quarter"] = dt_s.dt.quarter.fillna(-1).astype(np.int32)
        res_df[f"{col}_is_month_start"] = dt_s.dt.is_month_start.fillna(False).astype(
            np.int32
        )
        res_df[f"{col}_is_month_end"] = dt_s.dt.is_month_end.fillna(False).astype(
            np.int32
        )
        res_df[f"{col}_is_na"] = df[col].isna().astype(np.int32)

        valid_mask = dt_s.notna()
        res_df[f"{col}_month_sin"] = np.where(
            valid_mask, np.sin(2 * np.pi * dt_s.dt.month / 12.0), 0.0
        )
        res_df[f"{col}_month_cos"] = np.where(
            valid_mask, np.cos(2 * np.pi * dt_s.dt.month / 12.0), 0.0
        )
        res_df[f"{col}_dow_sin"] = np.where(
            valid_mask, np.sin(2 * np.pi * dt_s.dt.dayofweek / 7.0), 0.0
        )
        res_df[f"{col}_dow_cos"] = np.where(
            valid_mask, np.cos(2 * np.pi * dt_s.dt.dayofweek / 7.0), 0.0
        )

    date_col_list = list(date_columns)
    for i in range(len(date_col_list)):
        for j in range(i + 1, len(date_col_list)):
            col1 = date_col_list[i]
            col2 = date_col_list[j]
            diff_days = (
                date_series_dict[col1] - date_series_dict[col2]
            ).dt.total_seconds() / (24 * 3600)
            res_df[f"date_diff_{col1}_minus_{col2}"] = diff_days.fillna(0.0).astype(
                np.float32
            )
            res_df[f"date_diff_valid_{col1}_{col2}"] = (
                date_series_dict[col1].notna() & date_series_dict[col2].notna()
            ).astype(np.int32)

    return res_df


def extract_text_metadata(df, text_columns):
    res_df = pd.DataFrame(index=df.index)
    for col in text_columns:
        s = df[col].fillna("").astype(str)
        res_df[f"{col}_char_len"] = s.str.len().astype(np.float32)
        res_df[f"{col}_word_count"] = s.apply(lambda x: len(x.split())).astype(
            np.float32
        )
        res_df[f"{col}_digit_count"] = s.apply(
            lambda x: sum(c.isdigit() for c in x)
        ).astype(np.float32)
        res_df[f"{col}_upper_count"] = s.apply(
            lambda x: sum(c.isupper() for c in x)
        ).astype(np.float32)
        res_df[f"{col}_upper_ratio"] = (
            res_df[f"{col}_upper_count"] / (res_df[f"{col}_char_len"] + 1.0)
        ).astype(np.float32)
        res_df[f"{col}_punct_count"] = s.apply(
            lambda x: len(re.findall(r"[^\w\s]", x))
        ).astype(np.float32)
        res_df[f"{col}_is_empty"] = (s.str.strip() == "").astype(np.int32)
    return res_df


train_date_feats = extract_date_features(train_df, date_cols)
test_date_feats = extract_date_features(test_df, date_cols)

train_text_meta = extract_text_metadata(train_df, text_cols)
test_text_meta = extract_text_metadata(test_df, text_cols)

# SVD Semantic Features (Fit only on Train)
svd_feats_train_list = []
svd_feats_test_list = []
if text_cols:
    train_combined_text = (
        train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    test_combined_text = test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    n_components = min(30, max(5, len(train_df) // 20))
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    train_tfidf = tfidf.fit_transform(train_combined_text)
    test_tfidf = tfidf.transform(test_combined_text)

    n_comp_actual = (
        min(n_components, max(1, train_tfidf.shape[1] - 1))
        if train_tfidf.shape[1] > 1
        else 1
    )
    if n_comp_actual >= 1:
        svd = TruncatedSVD(n_components=n_comp_actual, random_state=42)
        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)

        for i in range(n_comp_actual):
            feat_name = f"tfidf_svd_comp_{i}"
            svd_feats_train_list.append(
                pd.Series(train_svd[:, i], index=train_df.index, name=feat_name)
            )
            svd_feats_test_list.append(
                pd.Series(test_svd[:, i], index=test_df.index, name=feat_name)
            )

train_svd_df = (
    pd.concat(svd_feats_train_list, axis=1)
    if svd_feats_train_list
    else pd.DataFrame(index=train_df.index)
)
test_svd_df = (
    pd.concat(svd_feats_test_list, axis=1)
    if svd_feats_test_list
    else pd.DataFrame(index=test_df.index)
)

# Categorical Frequency and Ordinal Features
train_cat_df = pd.DataFrame(index=train_df.index)
test_cat_df = pd.DataFrame(index=test_df.index)

cat_encoders = {}
for col in cat_cols:
    freq_map = train_df[col].value_counts(normalize=True).to_dict()
    train_cat_df[f"{col}_freq"] = (
        train_df[col].map(freq_map).fillna(0.0).astype(np.float32)
    )
    test_cat_df[f"{col}_freq"] = (
        test_df[col].map(freq_map).fillna(0.0).astype(np.float32)
    )

    train_cat_df[f"{col}_isna"] = train_df[col].isna().astype(np.int32)
    test_cat_df[f"{col}_isna"] = test_df[col].isna().astype(np.int32)

    le = LabelEncoder()
    train_vals = train_df[col].fillna("__MISSING__").astype(str)
    test_vals = test_df[col].fillna("__MISSING__").astype(str)

    le.fit(train_vals)
    cat_encoders[col] = {v: idx for idx, v in enumerate(le.classes_)}

    train_cat_df[f"{col}_encoded"] = train_vals.map(
        lambda x: cat_encoders[col].get(x, -1)
    ).astype(np.int32)
    test_cat_df[f"{col}_encoded"] = test_vals.map(
        lambda x: cat_encoders[col].get(x, -1)
    ).astype(np.int32)

# Numerical Transformations
train_num_df = pd.DataFrame(index=train_df.index)
test_num_df = pd.DataFrame(index=test_df.index)

for col in num_cols:
    train_num_df[f"{col}_isna"] = train_df[col].isna().astype(np.int32)
    test_num_df[f"{col}_isna"] = test_df[col].isna().astype(np.int32)

    median_val = train_df[col].median()
    if pd.isna(median_val):
        median_val = 0.0

    train_filled = train_df[col].fillna(median_val).astype(np.float32)
    test_filled = test_df[col].fillna(median_val).astype(np.float32)

    train_num_df[f"{col}_raw"] = train_filled
    test_num_df[f"{col}_raw"] = test_filled

    if (train_filled >= 0).all():
        train_num_df[f"{col}_log1p"] = np.log1p(train_filled)
        test_num_df[f"{col}_log1p"] = np.log1p(test_filled.clip(lower=0))

# Assemble Feature Matrices
X_train_full = pd.concat(
    [train_date_feats, train_text_meta, train_svd_df, train_cat_df, train_num_df],
    axis=1,
)
X_test_full = pd.concat(
    [test_date_feats, test_text_meta, test_svd_df, test_cat_df, test_num_df], axis=1
)

feature_columns = list(X_train_full.columns)
X_test_full = X_test_full[feature_columns]
y_train_full = train_df["target"].values

# Stratified Holdout Split (80% Train, 20% Validation)
train_indices, val_indices = train_test_split(
    np.arange(len(train_df)), test_size=0.2, stratify=y_train_full, random_state=42
)

# Out-of-fold multi-class target encoding computed strictly on train split
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(train_indices), dtype=np.int32)

y_train = y_train_full[train_indices].astype(np.int64)
y_val = y_train_full[val_indices].astype(np.int64)

for fold, (trn_idx, v_idx) in enumerate(skf.split(train_indices, y_train)):
    fold_assignments[v_idx] = fold

train_df_split = train_df.iloc[train_indices].reset_index(drop=True)
val_df_split = train_df.iloc[val_indices].reset_index(drop=True)

for col in cat_cols:
    if train_df[col].nunique() > 5:
        for c in range(num_classes):
            feat_name = f"{col}_te_class_{c}"
            X_train_full[feat_name] = 0.0
            X_test_full[feat_name] = 0.0
            global_mean = float((y_train == c).mean())

            train_te_col = np.zeros(len(train_df_split), dtype=np.float32)
            for fold in range(n_splits):
                val_mask = fold_assignments == fold
                trn_mask = fold_assignments != fold

                trn_cat = train_df_split.loc[trn_mask, col].fillna("__MISSING__")
                trn_y = pd.Series(
                    (y_train[trn_mask] == c).astype(float), index=trn_cat.index
                )

                stats = trn_y.groupby(trn_cat).agg(["count", "mean"])
                smoothing = 10.0
                smooth_te = (
                    stats["count"] * stats["mean"] + smoothing * global_mean
                ) / (stats["count"] + smoothing)

                val_cat = train_df_split.loc[val_mask, col].fillna("__MISSING__")
                train_te_col[val_mask] = (
                    val_cat.map(smooth_te).fillna(global_mean).values
                )

            X_train_full.loc[train_df.index[train_indices], feat_name] = train_te_col

            full_cat = train_df_split[col].fillna("__MISSING__")
            full_y = pd.Series((y_train == c).astype(float), index=full_cat.index)
            stats_full = full_y.groupby(full_cat).agg(["count", "mean"])
            smoothing = 10.0
            smooth_te_full = (
                stats_full["count"] * stats_full["mean"] + smoothing * global_mean
            ) / (stats_full["count"] + smoothing)

            val_cat_full = val_df_split[col].fillna("__MISSING__")
            X_train_full.loc[train_df.index[val_indices], feat_name] = (
                val_cat_full.map(smooth_te_full).fillna(global_mean).values
            )

            test_cat = test_df[col].fillna("__MISSING__")
            X_test_full[feat_name] = (
                test_cat.map(smooth_te_full).fillna(global_mean).values
            )

all_model_features = list(X_train_full.columns)
num_features = len(all_model_features)

X_train = X_train_full.iloc[train_indices][all_model_features].values.astype(np.float32)
X_val = X_train_full.iloc[val_indices][all_model_features].values.astype(np.float32)

X_test = X_test_full[all_model_features].values.astype(np.float32)

# Handle NaNs strictly using training medians
col_medians = np.nanmedian(X_train, axis=0)
col_medians = np.nan_to_num(col_medians, nan=0.0)

for c in range(X_train.shape[1]):
    mask_trn = np.isnan(X_train[:, c])
    mask_val = np.isnan(X_val[:, c])
    mask_tst = np.isnan(X_test[:, c])
    if mask_trn.any():
        X_train[mask_trn, c] = col_medians[c]
    if mask_val.any():
        X_val[mask_val, c] = col_medians[c]
    if mask_tst.any():
        X_test[mask_tst, c] = col_medians[c]

# Feature Scaling strictly on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# 3. MODEL ARCHITECTURES & TRAINING
# ==============================================================================
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TabularResNet Training ---
batch_size = 64
train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

nn_model = TabularResNet(
    input_dim=num_features,
    num_classes=num_classes,
    hidden_dim=256,
    num_blocks=3,
    dropout_rate=0.2,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

epochs = 40
best_nn_val_loss = float("inf")
best_nn_weights = None

for epoch in range(1, epochs + 1):
    nn_model.train()
    total_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = nn_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item() * len(batch_y)

    scheduler.step()
    train_loss = total_train_loss / len(X_train_scaled)

    nn_model.eval()
    total_val_loss = 0.0
    val_preds_list = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            total_val_loss += loss.item() * len(batch_y)
            val_preds_list.append(torch.softmax(logits, dim=1).cpu().numpy())

    val_loss = total_val_loss / len(X_val_scaled)
    val_probs_epoch = np.concatenate(val_preds_list, axis=0)
    val_acc = accuracy_score(y_val, np.argmax(val_probs_epoch, axis=1))

    if val_loss < best_nn_val_loss:
        best_nn_val_loss = val_loss
        best_nn_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    print(
        f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
    )

if best_nn_weights is not None:
    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
nn_model.eval()

with torch.no_grad():
    val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    nn_val_probs = torch.softmax(nn_model(val_tensor), dim=1).cpu().numpy()

    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    nn_test_probs = torch.softmax(nn_model(test_tensor), dim=1).cpu().numpy()

# --- LightGBM Training ---
lgb_train_data = lgb.Dataset(
    X_train, label=y_train, feature_name=all_model_features, free_raw_data=False
)
lgb_val_data = lgb.Dataset(
    X_val,
    label=y_val,
    feature_name=all_model_features,
    reference=lgb_train_data,
    free_raw_data=False,
)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

lgb_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train_data,
    num_boost_round=300,
    valid_sets=[lgb_train_data, lgb_val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(-1)],
)

lgb_val_probs = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_test_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)


# ==============================================================================
# 4. ENSEMBLE EVALUATION & SUBMISSION GENERATION
# ==============================================================================
best_acc = 0.0
best_w = 0.5
for w in np.linspace(0.0, 1.0, 21):
    blend_val = w * lgb_val_probs + (1.0 - w) * nn_val_probs
    pred_val = np.argmax(blend_val, axis=1)
    acc = accuracy_score(y_val, pred_val)
    if acc > best_acc:
        best_acc = acc
        best_w = w

final_val_probs = best_w * lgb_val_probs + (1.0 - best_w) * nn_val_probs
final_val_preds = np.argmax(final_val_probs, axis=1)
final_val_score = float(accuracy_score(y_val, final_val_preds))

final_test_probs = best_w * lgb_test_probs + (1.0 - best_w) * nn_test_probs
final_test_pred_indices = np.argmax(final_test_probs, axis=1)
final_test_pred_labels = [
    target_mapping.get(idx, str(idx)) for idx in final_test_pred_indices
]

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_pred_labels)),
        "availability": final_test_pred_labels,
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")