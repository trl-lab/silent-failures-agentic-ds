import gc
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# 1. Data Loading & Feature Engineering
# ==============================================================================
print("Loading data from ./input/...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_ids = train_df["id"].values
test_ids = test_df["id"].values
y_train = train_df["target"].values

train_df = train_df.drop(columns=["id", "target"])
test_df = test_df.drop(columns=["id"])

n_train = len(train_df)
n_test = len(test_df)
print(f"Train samples: {n_train}, Test samples: {n_test}")

# Explicit column groupings
bin_cols = [f"bin_{i}" for i in range(5)]
nom_cols = [f"nom_{i}" for i in range(10)]
ord_cols = [f"ord_{i}" for i in range(6)]
time_cols = ["day", "month"]
all_features = bin_cols + nom_cols + ord_cols + time_cols

# Impute / standardize string representations
for col in all_features:
    train_df[col] = train_df[col].fillna("MISSING").astype(str)
    test_df[col] = test_df[col].fillna("MISSING").astype(str)

# Ordinal encodings mapping based on domain hierarchy
ord_1_map = {
    "Novice": 1,
    "Contributor": 2,
    "Expert": 3,
    "Master": 4,
    "Grandmaster": 5,
    "MISSING": 0,
}
ord_2_map = {
    "Freezing": 1,
    "Cold": 2,
    "Warm": 3,
    "Hot": 4,
    "Boiling Hot": 5,
    "Lava Hot": 6,
    "MISSING": 0,
}


def map_ord_3(val):
    if val == "MISSING" or len(val) == 0:
        return 0
    return ord(val[0]) - ord("a") + 1


def map_ord_4(val):
    if val == "MISSING" or len(val) == 0:
        return 0
    return ord(val[0]) - ord("A") + 1


# String decomposition for ord_5
train_df["ord_5_char1"] = train_df["ord_5"].apply(
    lambda x: x[0] if x != "MISSING" else "MISSING"
)
train_df["ord_5_char2"] = train_df["ord_5"].apply(
    lambda x: x[1] if (x != "MISSING" and len(x) > 1) else "MISSING"
)
test_df["ord_5_char1"] = test_df["ord_5"].apply(
    lambda x: x[0] if x != "MISSING" else "MISSING"
)
test_df["ord_5_char2"] = test_df["ord_5"].apply(
    lambda x: x[1] if (x != "MISSING" and len(x) > 1) else "MISSING"
)

# Unique sorted mapping for ord_5 string comparison
unique_ord_5 = sorted(
    [
        v
        for v in set(train_df["ord_5"].unique()).union(set(test_df["ord_5"].unique()))
        if v != "MISSING"
    ]
)
ord_5_map = {v: i + 1 for i, v in enumerate(unique_ord_5)}
ord_5_map["MISSING"] = 0

# Binary mappings
bin_3_map = {"T": 1, "F": 0, "1.0": 1, "0.0": 0, "MISSING": -1}
bin_4_map = {"Y": 1, "N": 0, "1.0": 1, "0.0": 0, "MISSING": -1}
bin_num_map = {"1.0": 1, "1": 1, "0.0": 0, "0": 0, "MISSING": -1}

# Create dense engineered numerical features
df_dense_train = pd.DataFrame(index=train_df.index)
df_dense_test = pd.DataFrame(index=test_df.index)

df_dense_train["bin_0"] = train_df["bin_0"].map(bin_num_map).fillna(-1)
df_dense_test["bin_0"] = test_df["bin_0"].map(bin_num_map).fillna(-1)
df_dense_train["bin_1"] = train_df["bin_1"].map(bin_num_map).fillna(-1)
df_dense_test["bin_1"] = test_df["bin_1"].map(bin_num_map).fillna(-1)
df_dense_train["bin_2"] = train_df["bin_2"].map(bin_num_map).fillna(-1)
df_dense_test["bin_2"] = test_df["bin_2"].map(bin_num_map).fillna(-1)
df_dense_train["bin_3"] = train_df["bin_3"].map(bin_3_map).fillna(-1)
df_dense_test["bin_3"] = test_df["bin_3"].map(bin_3_map).fillna(-1)
df_dense_train["bin_4"] = train_df["bin_4"].map(bin_4_map).fillna(-1)
df_dense_test["bin_4"] = test_df["bin_4"].map(bin_4_map).fillna(-1)

df_dense_train["ord_0"] = train_df["ord_0"].apply(
    lambda x: float(x) if x != "MISSING" else 0.0
)
df_dense_test["ord_0"] = test_df["ord_0"].apply(
    lambda x: float(x) if x != "MISSING" else 0.0
)
df_dense_train["ord_1"] = train_df["ord_1"].map(ord_1_map).fillna(0)
df_dense_test["ord_1"] = test_df["ord_1"].map(ord_1_map).fillna(0)
df_dense_train["ord_2"] = train_df["ord_2"].map(ord_2_map).fillna(0)
df_dense_test["ord_2"] = test_df["ord_2"].map(ord_2_map).fillna(0)
df_dense_train["ord_3"] = train_df["ord_3"].apply(map_ord_3)
df_dense_test["ord_3"] = test_df["ord_3"].apply(map_ord_3)
df_dense_train["ord_4"] = train_df["ord_4"].apply(map_ord_4)
df_dense_test["ord_4"] = test_df["ord_4"].apply(map_ord_4)
df_dense_train["ord_5"] = train_df["ord_5"].map(ord_5_map).fillna(0)
df_dense_test["ord_5"] = test_df["ord_5"].map(ord_5_map).fillna(0)

# Cyclical trigonometric features for day and month
day_train_num = train_df["day"].apply(lambda x: float(x) if x != "MISSING" else np.nan)
day_test_num = test_df["day"].apply(lambda x: float(x) if x != "MISSING" else np.nan)
month_train_num = train_df["month"].apply(
    lambda x: float(x) if x != "MISSING" else np.nan
)
month_test_num = test_df["month"].apply(
    lambda x: float(x) if x != "MISSING" else np.nan
)

df_dense_train["day_sin"] = np.sin(2 * np.pi * day_train_num / 7.0).fillna(0.0)
df_dense_train["day_cos"] = np.cos(2 * np.pi * day_train_num / 7.0).fillna(0.0)
df_dense_test["day_sin"] = np.sin(2 * np.pi * day_test_num / 7.0).fillna(0.0)
df_dense_test["day_cos"] = np.cos(2 * np.pi * day_test_num / 7.0).fillna(0.0)

df_dense_train["month_sin"] = np.sin(2 * np.pi * month_train_num / 12.0).fillna(0.0)
df_dense_train["month_cos"] = np.cos(2 * np.pi * month_train_num / 12.0).fillna(0.0)
df_dense_test["month_sin"] = np.sin(2 * np.pi * month_test_num / 12.0).fillna(0.0)
df_dense_test["month_cos"] = np.cos(2 * np.pi * month_test_num / 12.0).fillna(0.0)

# 2-way categorical interaction features
interactions = [
    ("day", "month"),
    ("nom_0", "nom_1"),
    ("nom_0", "nom_2"),
    ("nom_1", "nom_2"),
    ("ord_1", "ord_2"),
    ("bin_0", "bin_1"),
    ("bin_3", "bin_4"),
]

for col1, col2 in interactions:
    inter_name = f"{col1}_x_{col2}"
    train_df[inter_name] = train_df[col1] + "_" + train_df[col2]
    test_df[inter_name] = test_df[col1] + "_" + test_df[col2]

# Frequency encoding for high cardinality & interaction features
freq_cols = (
    nom_cols
    + ord_cols
    + ["ord_5_char1", "ord_5_char2"]
    + [f"{c1}_x_{c2}" for c1, c2 in interactions]
)

for col in freq_cols:
    freq_map = train_df[col].value_counts(normalize=True).to_dict()
    df_dense_train[f"{col}_freq"] = (
        train_df[col].map(freq_map).fillna(0.0).astype(float)
    )
    df_dense_test[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0.0).astype(float)

# Stratified 5-Fold split and Out-Of-Fold Smoothed Target Encoding
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
fold_assignments = np.zeros(n_train, dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
    fold_assignments[val_idx] = fold

target_enc_cols = [
    "nom_0",
    "nom_1",
    "nom_2",
    "nom_3",
    "nom_4",
    "nom_5",
    "nom_6",
    "nom_7",
    "nom_8",
    "nom_9",
    "ord_5",
    "day",
    "month",
    "day_x_month",
    "nom_0_x_nom_1",
]
smoothing = 10.0
global_mean = float(y_train.mean())
y_train_series = pd.Series(y_train, index=train_df.index)

for col in target_enc_cols:
    df_dense_train[f"{col}_te"] = 0.0
    df_dense_test[f"{col}_te"] = 0.0

    counts = train_df[col].value_counts()
    sums = y_train_series.groupby(train_df[col]).sum()
    test_te = (sums + smoothing * global_mean) / (counts + smoothing)
    df_dense_test[f"{col}_te"] = (
        test_df[col].map(test_te).fillna(global_mean).astype(float)
    )

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        trn_df_fold = train_df.iloc[trn_idx]
        y_trn_fold = y_train[trn_idx]
        fold_mean = float(y_trn_fold.mean())
        y_trn_series_fold = pd.Series(y_trn_fold, index=trn_df_fold.index)

        f_counts = trn_df_fold[col].value_counts()
        f_sums = y_trn_series_fold.groupby(trn_df_fold[col]).sum()
        oof_te = (f_sums + smoothing * fold_mean) / (f_counts + smoothing)

        df_dense_train.loc[df_dense_train.index[val_idx], f"{col}_te"] = (
            train_df.iloc[val_idx][col].map(oof_te).fillna(fold_mean).astype(float)
        )

# Sparse High-Dimensional One-Hot Encoding for linear representations
all_cat_cols = (
    all_features
    + ["ord_5_char1", "ord_5_char2"]
    + [f"{c1}_x_{c2}" for c1, c2 in interactions]
)
print(
    f"Building Sparse One-Hot representation over {len(all_cat_cols)} categorical columns..."
)

ohe = OneHotEncoder(dtype=np.float32, handle_unknown="ignore", min_frequency=2)
ohe.fit(train_df[all_cat_cols])

X_train_ohe = ohe.transform(train_df[all_cat_cols])
X_test_ohe = ohe.transform(test_df[all_cat_cols])

print(f"Sparse OHE shape: Train {X_train_ohe.shape}, Test {X_test_ohe.shape}")
print(f"Dense features shape: Train {df_dense_train.shape}, Test {df_dense_test.shape}")


# ==============================================================================
# 2. Model Architecture Definitions
# ==============================================================================
def get_sparse_logistic_model(
    C: float = 0.12, solver: str = "lbfgs", max_iter: int = 1000, random_state: int = 42
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        penalty="l2",
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )


def get_lightgbm_params(random_state: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout_rate)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.drop(self.act(self.bn(self.fc(x))))
        return out + residual


class SparseDenseCatNN(nn.Module):
    def __init__(
        self,
        sparse_in_dim: int,
        dense_in_dim: int,
        dense_hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.sparse_linear = nn.Linear(sparse_in_dim, 1, bias=True)
        self.dense_bn_in = nn.BatchNorm1d(dense_in_dim)
        dense_layers = []
        prev_dim = dense_in_dim
        for hidden_dim in dense_hidden_dims:
            dense_layers.append(
                ResidualDenseBlock(
                    prev_dim, hidden_dim, dropout_rate=dropout_rate / 2.0
                )
            )
            prev_dim = hidden_dim
        self.dense_trunk = nn.Sequential(*dense_layers)
        self.dense_head = nn.Linear(prev_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        sparse_logit = self.sparse_linear(x_sparse)
        dense_feat = self.dense_bn_in(x_dense)
        dense_feat = self.dense_trunk(dense_feat)
        dense_logit = self.dense_head(dense_feat)
        logit = sparse_logit + dense_logit + self.bias
        return logit.squeeze(-1)


# ==============================================================================
# 3. 5-Fold Training & Ensembling Pipeline
# ==============================================================================
print("Starting 5-fold training and evaluation pipeline...")

oof_preds_lr = np.zeros(n_train, dtype=np.float64)
test_preds_lr = np.zeros(n_test, dtype=np.float64)

oof_preds_lgb = np.zeros(n_train, dtype=np.float64)
test_preds_lgb = np.zeros(n_test, dtype=np.float64)

for fold in range(n_splits):
    trn_idx = np.where(fold_assignments != fold)[0]
    val_idx = np.where(fold_assignments == fold)[0]

    y_trn_fold = y_train[trn_idx]
    y_val_fold = y_train[val_idx]

    # Train Sparse Regularized Logistic Regression
    lr_model = get_sparse_logistic_model(
        C=0.12, solver="lbfgs", max_iter=1000, random_state=42 + fold
    )
    lr_model.fit(X_train_ohe[trn_idx], y_trn_fold)

    val_pred_lr = lr_model.predict_proba(X_train_ohe[val_idx])[:, 1]
    oof_preds_lr[val_idx] = val_pred_lr
    test_preds_lr += (
        lr_model.predict_proba(X_test_ohe)[:, 1].astype(np.float64) / n_splits
    )
    fold_auc_lr = roc_auc_score(y_val_fold, val_pred_lr)

    # Train Gradient Boosted Decision Tree (LightGBM)
    lgb_params = get_lightgbm_params(random_state=42 + fold)
    trn_data = lgb.Dataset(
        df_dense_train.iloc[trn_idx], label=y_trn_fold, free_raw_data=False
    )
    val_data = lgb.Dataset(
        df_dense_train.iloc[val_idx],
        label=y_val_fold,
        reference=trn_data,
        free_raw_data=False,
    )

    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred_lgb = lgb_model.predict(
        df_dense_train.iloc[val_idx], num_iteration=lgb_model.best_iteration
    )
    oof_preds_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += (
        lgb_model.predict(df_dense_test, num_iteration=lgb_model.best_iteration).astype(
            np.float64
        )
        / n_splits
    )
    fold_auc_lgb = roc_auc_score(y_val_fold, val_pred_lgb)

    print(
        f"Fold {fold + 1}/{n_splits} | LR AUC: {fold_auc_lr:.5f} | LGB AUC: {fold_auc_lgb:.5f}"
    )

# Optimize blend weight between LR and LGB to maximize global ROC-AUC
best_weight = 0.5
best_auc = 0.0
for weight in np.linspace(0.0, 1.0, 101):
    blended_oof = weight * oof_preds_lr + (1.0 - weight) * oof_preds_lgb
    current_auc = roc_auc_score(y_train, blended_oof)
    if current_auc > best_auc:
        best_auc = current_auc
        best_weight = weight

print(
    f"Optimal Ensemble Weights -> LR: {best_weight:.2f}, LGB: {1.0 - best_weight:.2f}"
)

# Generate final blended predictions
final_oof_preds = best_weight * oof_preds_lr + (1.0 - best_weight) * oof_preds_lgb
final_test_preds = best_weight * test_preds_lr + (1.0 - best_weight) * test_preds_lgb

assert not np.isnan(final_test_preds).any(), "NaN detected in final test predictions!"
assert len(final_test_preds) == n_test, "Test prediction length mismatch!"

# Save submission file
submission_df = pd.DataFrame({"id": test_ids, "target": final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)

# Compute and report final validation score
final_val_score = roc_auc_score(y_train, final_oof_preds)
print(f"Final Validation Score: {final_val_score}")
