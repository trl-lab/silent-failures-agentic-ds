import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.sparse import hstack
from scipy.optimize import minimize

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Loading raw dataset...")
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

# Define Target and ID
TARGET_COL = "Task Assigned"
if "id" in test_raw.columns:
    test_ids = test_raw["id"].astype(int).values
else:
    test_ids = np.arange(len(test_raw), dtype=int)
    test_raw["id"] = test_ids

if "id" not in train_raw.columns:
    train_raw["id"] = np.arange(len(train_raw))

# Ensure target is clean
train_raw = train_raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)
target_le = LabelEncoder()
train_raw[TARGET_COL] = target_le.fit_transform(train_raw[TARGET_COL].astype(str))
y = train_raw[TARGET_COL].values

print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

# Identify column types dynamically
all_cols = [c for c in train_raw.columns if c not in [TARGET_COL, "id"]]

text_cols = []
cat_cols = []
num_cols = []
date_cols = []

for c in all_cols:
    dtype = train_raw[c].dtype
    if (
        np.issubdtype(dtype, np.datetime64)
        or "date" in c.lower()
        or "time" in c.lower()
    ):
        date_cols.append(c)
    elif dtype == "object" or dtype.name == "category":
        sample_str = train_raw[c].dropna().astype(str)
        if len(sample_str) > 0 and sample_str.str.len().mean() > 30:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        if train_raw[c].nunique() < 20 and "code" not in c.lower():
            cat_cols.append(c)
        else:
            num_cols.append(c)

print(
    f"Detected columns - Text: {len(text_cols)}, Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}, Date: {len(date_cols)}"
)


# Feature Engineering Functions
def extract_domain_text_features(df, text_columns):
    features = pd.DataFrame(index=df.index)
    hazard_keywords = {
        "hazard_fall": r"\b(fall|ladder|scaffold|roof|height|dropping)\b",
        "hazard_trench": r"\b(trench|excavation|cave-in|collapse|soil)\b",
        "hazard_amputat": r"\b(amputat|finger|hand|blade|saw|cut)\b",
        "hazard_electr": r"\b(electr|power|wire|shock|voltage|arc)\b",
        "hazard_chemical": r"\b(chemical|toxic|gas|vapor|fume|acid|burn)\b",
        "hazard_crush": r"\b(crush|caught|between|pinned|press|struck)\b",
        "hazard_machin": r"\b(machine|guard|conveyor|roller|gear|engine)\b",
        "hazard_ppe": r"\b(ppe|harness|goggles|mask|respirator|helmet)\b",
        "hazard_vehicle": r"\b(forklift|crane|truck|vehicle|loader|boom)\b",
        "hazard_fatal": r"\b(fatal|death|died|killed|severe|criti)\b",
    }

    for c in text_columns:
        text_series = df[c].fillna("").astype(str).str.lower()
        features[f"{c}_char_len"] = text_series.str.len()
        features[f"{c}_word_count"] = text_series.str.split().str.len()
        features[f"{c}_avg_word_len"] = features[f"{c}_char_len"] / (
            features[f"{c}_word_count"] + 1
        )
        features[f"{c}_digit_count"] = text_series.str.count(r"\d")
        features[f"{c}_upper_count"] = df[c].fillna("").astype(str).str.count(r"[A-Z]")

        for kw_name, pattern in hazard_keywords.items():
            features[f"{c}_{kw_name}"] = text_series.str.contains(
                pattern, regex=True
            ).astype(int)

    return features


train_domain_text = extract_domain_text_features(train_raw, text_cols)
test_domain_text = extract_domain_text_features(test_raw, text_cols)


def extract_hierarchical_codes(df, categorical_columns):
    features = pd.DataFrame(index=df.index)
    for c in categorical_columns:
        str_col = df[c].fillna("Missing").astype(str)
        features[f"{c}_prefix_2"] = str_col.str[:2]
        features[f"{c}_prefix_3"] = str_col.str[:3]
    return features


train_hier_codes = extract_hierarchical_codes(train_raw, cat_cols)
test_hier_codes = extract_hierarchical_codes(test_raw, cat_cols)

all_cat_cols = cat_cols + list(train_hier_codes.columns)
train_cats = pd.concat([train_raw[cat_cols].astype(str), train_hier_codes], axis=1)
test_cats = pd.concat([test_raw[cat_cols].astype(str), test_hier_codes], axis=1)

for c in all_cat_cols:
    freq_map = train_cats[c].value_counts(normalize=True).to_dict()
    train_cats[f"{c}_freq"] = train_cats[c].map(freq_map).fillna(0).astype(np.float32)
    test_cats[f"{c}_freq"] = test_cats[c].map(freq_map).fillna(0).astype(np.float32)


def extract_date_features(df, date_columns):
    features = pd.DataFrame(index=df.index)
    for c in date_columns:
        dt_col = pd.to_datetime(df[c], errors="coerce")
        features[f"{c}_year"] = dt_col.dt.year.fillna(-1).astype(int)
        features[f"{c}_month"] = dt_col.dt.month.fillna(-1).astype(int)
        features[f"{c}_day"] = dt_col.dt.day.fillna(-1).astype(int)
        features[f"{c}_dayofweek"] = dt_col.dt.dayofweek.fillna(-1).astype(int)
        features[f"{c}_is_weekend"] = (dt_col.dt.dayofweek >= 5).astype(int)
        features[f"{c}_sin_month"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0).fillna(
            0
        )
        features[f"{c}_cos_month"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0).fillna(
            0
        )
    return features


train_dates = extract_date_features(train_raw, date_cols)
test_dates = extract_date_features(test_raw, date_cols)

train_nums = train_raw[num_cols].copy()
test_nums = test_raw[num_cols].copy()

for c in num_cols:
    if train_nums[c].min() >= 0 and train_nums[c].skew() > 1.5:
        train_nums[f"{c}_log1p"] = np.log1p(train_nums[c])
        test_nums[f"{c}_log1p"] = np.log1p(test_nums[c])

# Categorical Group-by Aggregations of Numerical Features
group_agg_train = pd.DataFrame(index=train_raw.index)
group_agg_test = pd.DataFrame(index=test_raw.index)

main_cat_cols = cat_cols[:5]
for c in main_cat_cols:
    for n in num_cols:
        group_agg_train[f"{c}_{n}_grp_mean"] = train_raw.groupby(c)[n].transform("mean")
        group_agg_train[f"{c}_{n}_grp_std"] = train_raw.groupby(c)[n].transform("std")

        mean_map = train_raw.groupby(c)[n].mean().to_dict()
        std_map = train_raw.groupby(c)[n].std().to_dict()
        group_agg_test[f"{c}_{n}_grp_mean"] = test_raw[c].map(mean_map)
        group_agg_test[f"{c}_{n}_grp_std"] = test_raw[c].map(std_map)

train_cat_encoded = pd.DataFrame(index=train_raw.index)
test_cat_encoded = pd.DataFrame(index=test_raw.index)

for c in all_cat_cols:
    le = LabelEncoder()
    unique_vals = list(set(train_cats[c].unique()).union(set(test_cats[c].unique())))
    le.fit(unique_vals)
    train_cat_encoded[f"{c}_le"] = le.transform(train_cats[c])
    test_cat_encoded[f"{c}_le"] = le.transform(test_cats[c])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_target_enc = pd.DataFrame(index=train_raw.index)
test_target_enc = pd.DataFrame(index=test_raw.index)

global_mean = y.mean()
smooth_weight = 10.0

for c in cat_cols:
    train_target_enc[f"{c}_te"] = np.nan
    stats = train_raw.groupby(c)[TARGET_COL].agg(["count", "mean"])
    smooth_te = (stats["count"] * stats["mean"] + smooth_weight * global_mean) / (
        stats["count"] + smooth_weight
    )
    test_target_enc[f"{c}_te"] = (
        test_raw[c].map(smooth_te).fillna(global_mean).astype(np.float32)
    )

    for train_idx, val_idx in skf.split(train_raw, y):
        fold_train = train_raw.iloc[train_idx]
        fold_stats = fold_train.groupby(c)[TARGET_COL].agg(["count", "mean"])
        fold_smooth_te = (
            fold_stats["count"] * fold_stats["mean"] + smooth_weight * global_mean
        ) / (fold_stats["count"] + smooth_weight)
        train_target_enc.iloc[val_idx, train_target_enc.columns.get_loc(f"{c}_te")] = (
            train_raw.iloc[val_idx][c].map(fold_smooth_te).fillna(global_mean).values
        )

train_target_enc = train_target_enc.astype(np.float32)

print("Extracting out-of-fold text predictions via Ridge & Logistic Regression on TF-IDF...")
train_text_oof = pd.DataFrame(index=train_raw.index)
test_text_oof = pd.DataFrame(index=test_raw.index)

skf_text = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for c in text_cols:
    train_text_corpus = train_raw[c].fillna("").astype(str)
    test_text_corpus = test_raw[c].fillna("").astype(str)

    word_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english", sublinear_tf=True)
    char_vec = TfidfVectorizer(max_features=10000, ngram_range=(2, 5), analyzer="char", sublinear_tf=True)

    X_word_tr = word_vec.fit_transform(train_text_corpus)
    X_word_te = word_vec.transform(test_text_corpus)

    X_char_tr = char_vec.fit_transform(train_text_corpus)
    X_char_te = char_vec.transform(test_text_corpus)

    X_tfidf_tr = hstack([X_word_tr, X_char_tr]).tocsr()
    X_tfidf_te = hstack([X_word_te, X_char_te]).tocsr()

    oof_ridge_col = np.zeros(len(train_raw))
    test_ridge_col = np.zeros(len(test_raw))
    oof_lr_col = np.zeros(len(train_raw))
    test_lr_col = np.zeros(len(test_raw))

    for tr_i, va_i in skf_text.split(X_tfidf_tr, y):
        X_tr_f, y_tr_f = X_tfidf_tr[tr_i], y[tr_i]
        X_va_f = X_tfidf_tr[va_i]

        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X_tr_f, y_tr_f)
        oof_ridge_col[va_i] = ridge.predict(X_va_f)
        test_ridge_col += ridge.predict(X_tfidf_te) / 5.0

        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        lr.fit(X_tr_f, y_tr_f)
        oof_lr_col[va_i] = lr.predict_proba(X_va_f)[:, 1]
        test_lr_col += lr.predict_proba(X_tfidf_te)[:, 1] / 5.0

    train_text_oof[f"{c}_ridge_oof"] = oof_ridge_col
    test_text_oof[f"{c}_ridge_oof"] = test_ridge_col
    train_text_oof[f"{c}_lr_oof"] = oof_lr_col
    test_text_oof[f"{c}_lr_oof"] = test_lr_col

X_train_df = pd.concat(
    [
        train_domain_text,
        train_cats.filter(like="_freq"),
        train_cat_encoded,
        train_target_enc,
        train_dates,
        train_nums,
        group_agg_train,
        train_text_oof,
    ],
    axis=1,
)

X_test_df = pd.concat(
    [
        test_domain_text,
        test_cats.filter(like="_freq"),
        test_cat_encoded,
        test_target_enc,
        test_dates,
        test_nums,
        group_agg_test,
        test_text_oof,
    ],
    axis=1,
)

feature_names = X_train_df.columns.tolist()
X = X_train_df.values.astype(np.float32)
X_test_raw = X_test_df.values.astype(np.float32)

print(
    f"Dataset ready -> Features: {X.shape[1]}, Train samples: {X.shape[0]}, Test samples: {X_test_raw.shape[0]}"
)


# ==============================================================================
# PYTORCH TABULAR RESNET
# ==============================================================================
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.norm2(out)
        out = self.drop2(out)

        return self.act(out + residual)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        logits = self.head(out)
        return logits.squeeze(-1)


# Cross-Validation Model Training
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_nn = np.zeros(len(X))

test_lgb = np.zeros(len(X_test_raw))
test_xgb = np.zeros(len(X_test_raw))
test_cat = np.zeros(len(X_test_raw))
test_nn = np.zeros(len(X_test_raw))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting 5-Fold Cross Validation Training on {device}...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = imputer.fit_transform(X[train_idx])
    X_va_imp = imputer.transform(X[val_idx])
    X_te_imp = imputer.transform(X_test_raw)

    X_tr = scaler.fit_transform(X_tr_imp).astype(np.float32)
    X_va = scaler.transform(X_va_imp).astype(np.float32)
    X_test = scaler.transform(X_te_imp).astype(np.float32)

    y_tr, y_va = y[train_idx], y[val_idx]

    # 1. LIGHTGBM
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": SEED + fold,
        "n_estimators": 1000,
        "verbose": -1,
        "n_jobs": -1,
    }
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # 2. XGBOOST
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED + fold,
        "n_estimators": 1000,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # 3. CATBOOST
    cat_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.04,
        "depth": 6,
        "random_seed": SEED + fold,
        "iterations": 1000,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    model_cat = cb.CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS

    # 4. PYTORCH TABULAR RESNET
    nn_model = TabularResNet(
        input_dim=X.shape[1], hidden_dim=128, num_blocks=2, dropout_rate=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_val_auc = -1.0
    best_nn_preds_val = np.zeros(len(val_idx), dtype=np.float32)
    best_nn_preds_test = np.zeros(len(X_test), dtype=np.float32)
    patience_counter = 0

    for epoch in range(25):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = nn_model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.extend(probs)

        val_preds_arr = np.array(val_preds_list)
        val_auc = roc_auc_score(y_va, val_preds_arr)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_nn_preds_val = val_preds_arr

            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                test_logits = nn_model(test_tensor)
                best_nn_preds_test = torch.sigmoid(test_logits).cpu().numpy()
        else:
            patience_counter += 1
            if patience_counter >= 7:
                break

    oof_nn[val_idx] = best_nn_preds_val
    test_nn += best_nn_preds_test / N_SPLITS

    fold_lgb_auc = roc_auc_score(y_va, oof_lgb[val_idx])
    fold_xgb_auc = roc_auc_score(y_va, oof_xgb[val_idx])
    fold_cat_auc = roc_auc_score(y_va, oof_cat[val_idx])
    fold_nn_auc = roc_auc_score(y_va, oof_nn[val_idx])

    print(
        f"Fold {fold+1}/{N_SPLITS} -> LGB: {fold_lgb_auc:.5f} | XGB: {fold_xgb_auc:.5f} | CAT: {fold_cat_auc:.5f} | NN: {fold_nn_auc:.5f}"
    )

# Ensemble Blending
weights = [0.35, 0.25, 0.25, 0.15]
oof_ensemble = (
    weights[0] * oof_lgb
    + weights[1] * oof_xgb
    + weights[2] * oof_cat
    + weights[3] * oof_nn
)
test_ensemble = (
    weights[0] * test_lgb
    + weights[1] * test_xgb
    + weights[2] * test_cat
    + weights[3] * test_nn
)

final_score = roc_auc_score(y, oof_ensemble)

# Save Submission File
sub_df = pd.DataFrame({"id": test_ids, "Task Assigned": test_ensemble})
sub_df.to_csv("./submission/submission.csv", index=False)
print(f"Saved predictions to ./submission/submission.csv ({sub_df.shape})")

# Final metric output
print(f"Final Validation Score: {final_score}")