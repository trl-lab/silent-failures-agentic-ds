import gc
import itertools
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure reproducibility and environment setup
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# Step 1: Data Loading & Feature Engineering
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_ids = train_df["id"].values
test_ids = test_df["id"].values
y_train = train_df["target"].values

train_df = train_df.drop(columns=["id", "target"])
test_df = test_df.drop(columns=["id"])

n_train = len(train_df)
n_test = len(test_df)

# Feature Taxonomies
bin_cols = ["bin_0", "bin_1", "bin_2", "bin_3", "bin_4"]
nom_cols = [f"nom_{i}" for i in range(10)]
ord_cols = [f"ord_{i}" for i in range(6)]
cyclical_cols = ["day", "month"]

# Combine for unified non-target feature transformations
all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 1. Binary Features
all_df["bin_0"] = all_df["bin_0"].astype(int)
all_df["bin_1"] = all_df["bin_1"].astype(int)
all_df["bin_2"] = all_df["bin_2"].astype(int)
all_df["bin_3"] = all_df["bin_3"].map({"F": 0, "T": 1}).fillna(0).astype(int)
all_df["bin_4"] = all_df["bin_4"].map({"N": 0, "Y": 1}).fillna(0).astype(int)

# 2. Ordinal Features Semantic Mapping
ord_1_mapping = {
    "Novice": 1,
    "Contributor": 2,
    "Expert": 3,
    "Master": 4,
    "Grandmaster": 5,
}
ord_2_mapping = {
    "Freezing": 1,
    "Cold": 2,
    "Warm": 3,
    "Hot": 4,
    "Boiling Hot": 5,
    "Lava Hot": 6,
}

ord_3_cats = sorted(all_df["ord_3"].dropna().unique())
ord_3_mapping = {val: idx + 1 for idx, val in enumerate(ord_3_cats)}

ord_4_cats = sorted(all_df["ord_4"].dropna().unique())
ord_4_mapping = {val: idx + 1 for idx, val in enumerate(ord_4_cats)}

ord_5_cats = sorted(all_df["ord_5"].dropna().unique())
ord_5_mapping = {val: idx + 1 for idx, val in enumerate(ord_5_cats)}

# Decompose ord_5 into constituent characters and ASCII ordinal mappings
all_df["ord_5_1"] = all_df["ord_5"].astype(str).str[0]
all_df["ord_5_2"] = all_df["ord_5"].astype(str).str[1]

ord_5_1_cats = sorted(all_df["ord_5_1"].dropna().unique())
ord_5_1_mapping = {val: idx + 1 for idx, val in enumerate(ord_5_1_cats)}

ord_5_2_cats = sorted(all_df["ord_5_2"].dropna().unique())
ord_5_2_mapping = {val: idx + 1 for idx, val in enumerate(ord_5_2_cats)}

all_df["ord_0_num"] = all_df["ord_0"].astype(float)
all_df["ord_1_num"] = all_df["ord_1"].map(ord_1_mapping).astype(float)
all_df["ord_2_num"] = all_df["ord_2"].map(ord_2_mapping).astype(float)
all_df["ord_3_num"] = all_df["ord_3"].map(ord_3_mapping).astype(float)
all_df["ord_4_num"] = all_df["ord_4"].map(ord_4_mapping).astype(float)
all_df["ord_5_num"] = all_df["ord_5"].map(ord_5_mapping).astype(float)
all_df["ord_5_1_num"] = all_df["ord_5_1"].map(ord_5_1_mapping).astype(float)
all_df["ord_5_2_num"] = all_df["ord_5_2"].map(ord_5_2_mapping).astype(float)
all_df["ord_5_diff"] = all_df["ord_5_1_num"] - all_df["ord_5_2_num"]

# 3. Cyclical Sine/Cosine Features
all_df["day_sin"] = np.sin(2 * np.pi * all_df["day"] / 7.0)
all_df["day_cos"] = np.cos(2 * np.pi * all_df["day"] / 7.0)
all_df["month_sin"] = np.sin(2 * np.pi * all_df["month"] / 12.0)
all_df["month_cos"] = np.cos(2 * np.pi * all_df["month"] / 12.0)

# 4. Comprehensive Combinatorial Pairwise Interaction Features
interaction_base_cols = [
    "bin_0",
    "bin_1",
    "bin_2",
    "bin_3",
    "bin_4",
    "nom_0",
    "nom_1",
    "nom_2",
    "nom_3",
    "nom_4",
    "ord_0",
    "ord_1",
    "ord_2",
    "ord_3",
    "ord_5_1",
    "ord_5_2",
    "day",
    "month",
]

inter_dict = {}
interaction_cols = []
for col1, col2 in itertools.combinations(interaction_base_cols, 2):
    col_name = f"inter_{col1}_{col2}"
    inter_dict[col_name] = all_df[col1].astype(str) + "_" + all_df[col2].astype(str)
    interaction_cols.append(col_name)

inter_df = pd.DataFrame(inter_dict, index=all_df.index)
all_df = pd.concat([all_df, inter_df], axis=1)

# 5. Frequency / Count Encodings
decomposed_ord_cols = ord_cols + ["ord_5_1", "ord_5_2"]
cat_cols_for_freq = (
    bin_cols + nom_cols + decomposed_ord_cols + cyclical_cols + interaction_cols
)
freq_dict = {}
freq_cols = []
for c in cat_cols_for_freq:
    freq_col_name = f"{c}_freq"
    freq_map = all_df[c].value_counts(normalize=True).to_dict()
    freq_dict[freq_col_name] = all_df[c].map(freq_map).astype(np.float32)
    freq_cols.append(freq_col_name)

freq_df = pd.DataFrame(freq_dict, index=all_df.index)
all_df = pd.concat([all_df, freq_df], axis=1)

# Split back to train and test
train_engineered = all_df.iloc[:n_train].copy().reset_index(drop=True)
test_engineered = all_df.iloc[n_train:].copy().reset_index(drop=True)

# 6. Leak-Free Out-of-Fold (OOF) Dual-Scale Target Encoding & Categorical Indices
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(train_engineered, y_train))

smooth_m5 = 5.0
smooth_m25 = 25.0
global_mean = np.mean(y_train)
target_enc_cols = nom_cols + decomposed_ord_cols + cyclical_cols + interaction_cols

oof_te_m5_train = np.zeros((n_train, len(target_enc_cols)), dtype=np.float32)
oof_te_m25_train = np.zeros((n_train, len(target_enc_cols)), dtype=np.float32)
te_m5_test = np.zeros((n_test, len(target_enc_cols)), dtype=np.float32)
te_m25_test = np.zeros((n_test, len(target_enc_cols)), dtype=np.float32)

for c_idx, col in enumerate(target_enc_cols):
    train_vals = train_engineered[col].astype(str).values
    test_vals = test_engineered[col].astype(str).values

    for fold, (train_idx, val_idx) in enumerate(folds):
        fold_train_vals = train_vals[train_idx]
        fold_y = y_train[train_idx]

        df_fold = pd.DataFrame({"val": fold_train_vals, "target": fold_y})
        stats = df_fold.groupby("val")["target"].agg(["count", "mean"])

        smooth5 = (stats["count"] * stats["mean"] + smooth_m5 * global_mean) / (
            stats["count"] + smooth_m5
        )
        smooth25 = (stats["count"] * stats["mean"] + smooth_m25 * global_mean) / (
            stats["count"] + smooth_m25
        )

        smooth5_dict = smooth5.to_dict()
        smooth25_dict = smooth25.to_dict()

        oof_te_m5_train[val_idx, c_idx] = (
            pd.Series(train_vals[val_idx]).map(smooth5_dict).fillna(global_mean).values
        )
        oof_te_m25_train[val_idx, c_idx] = (
            pd.Series(train_vals[val_idx]).map(smooth25_dict).fillna(global_mean).values
        )

    df_full = pd.DataFrame({"val": train_vals, "target": y_train})
    full_stats = df_full.groupby("val")["target"].agg(["count", "mean"])
    full_smooth5 = (
        full_stats["count"] * full_stats["mean"] + smooth_m5 * global_mean
    ) / (full_stats["count"] + smooth_m5)
    full_smooth25 = (
        full_stats["count"] * full_stats["mean"] + smooth_m25 * global_mean
    ) / (full_stats["count"] + smooth_m25)

    te_m5_test[:, c_idx] = (
        pd.Series(test_vals).map(full_smooth5.to_dict()).fillna(global_mean).values
    )
    te_m25_test[:, c_idx] = (
        pd.Series(test_vals).map(full_smooth25.to_dict()).fillna(global_mean).values
    )

te_m5_names = [f"{col}_te_m5" for col in target_enc_cols]
te_m25_names = [f"{col}_te_m25" for col in target_enc_cols]

train_te_df = pd.DataFrame(
    np.hstack([oof_te_m5_train, oof_te_m25_train]),
    columns=te_m5_names + te_m25_names,
)
test_te_df = pd.DataFrame(
    np.hstack([te_m5_test, te_m25_test]),
    columns=te_m5_names + te_m25_names,
)

# Prepare integer categorical indices with unseen reservation for Embedding Net
cat_embed_cols = bin_cols + nom_cols + ord_cols + cyclical_cols + ["ord_5_1", "ord_5_2"]
cat_cardinalities = []
cat_train_list = []
cat_test_list = []

for col in cat_embed_cols:
    uniques = sorted(all_df[col].astype(str).unique())
    mapping = {val: idx + 1 for idx, val in enumerate(uniques)}
    cardinality = len(uniques) + 1  # 0 reserved for unseen
    emb_dim = min(50, max(4, int(np.ceil(cardinality ** 0.5 * 2))))
    cat_cardinalities.append((cardinality, emb_dim))

    cat_train_list.append(
        train_engineered[col].astype(str).map(mapping).fillna(0).values.astype(np.int64)
    )
    cat_test_list.append(
        test_engineered[col].astype(str).map(mapping).fillna(0).values.astype(np.int64)
    )

X_train_cat = np.column_stack(cat_train_list)
X_test_cat = np.column_stack(cat_test_list)

dense_feature_cols = [
    "bin_0",
    "bin_1",
    "bin_2",
    "bin_3",
    "bin_4",
    "ord_0_num",
    "ord_1_num",
    "ord_2_num",
    "ord_3_num",
    "ord_4_num",
    "ord_5_num",
    "ord_5_1_num",
    "ord_5_2_num",
    "ord_5_diff",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
] + freq_cols

X_train_dense_raw = pd.concat(
    [train_engineered[dense_feature_cols], train_te_df], axis=1
)
X_test_dense_raw = pd.concat([test_engineered[dense_feature_cols], test_te_df], axis=1)

scaler = StandardScaler()
X_train_dense = scaler.fit_transform(X_train_dense_raw).astype(np.float32)
X_test_dense = scaler.transform(X_test_dense_raw).astype(np.float32)

# 7. High-Dimensional Sparse One-Hot Encoding
ohe_cols = (
    bin_cols + nom_cols + decomposed_ord_cols + cyclical_cols + interaction_cols
)
ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
ohe.fit(train_engineered[ohe_cols].astype(str))

X_train_ohe = ohe.transform(train_engineered[ohe_cols].astype(str))
X_test_ohe = ohe.transform(test_engineered[ohe_cols].astype(str))

X_train_sparse = sparse.hstack([sparse.csr_matrix(X_train_dense), X_train_ohe]).tocsr()
X_test_sparse = sparse.hstack([sparse.csr_matrix(X_test_dense), X_test_ohe]).tocsr()

dense_dim = X_train_dense.shape[1]
n_folds = len(folds)


# =========================================================================
# Step 2: Model Definitions
# =========================================================================
class ResidualDenseBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.2,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_features == out_features)

        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = self.dropout1(self.act1(self.norm1(self.linear1(x))))
        h = self.dropout2(self.act2(self.norm2(self.linear2(h))))
        return h + residual


class TabularWideAndDeepNet(nn.Module):
    def __init__(
        self,
        dense_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.20,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(dense_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )

        deep_layers = []
        for i in range(len(hidden_dims) - 1):
            deep_layers.append(
                ResidualDenseBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.deep_backbone = nn.Sequential(*deep_layers)
        self.deep_head = nn.Linear(hidden_dims[-1], 1)
        self.wide_linear = nn.Linear(dense_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_deep = self.input_projection(x)
        h_deep = self.deep_backbone(h_deep)
        out_deep = self.deep_head(h_deep)
        out_wide = self.wide_linear(x)
        logits = out_deep + out_wide + self.bias
        return logits.view(-1)


# =========================================================================
# Step 3: Model Training & Cross-Validation
# =========================================================================

# --- Model 1: Sparse Regularized Logistic Regression ---
oof_preds_lr = np.zeros(n_train, dtype=np.float64)
test_preds_lr = np.zeros(n_test, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_train_sparse[train_idx], y_train[train_idx]
    X_va, y_va = X_train_sparse[val_idx], y_train[val_idx]

    clf = LogisticRegression(
        C=0.12,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42 + fold,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    val_probs = clf.predict_proba(X_va)[:, 1]
    oof_preds_lr[val_idx] = val_probs
    test_preds_lr += clf.predict_proba(X_test_sparse)[:, 1] / n_folds

# --- Model 2: Tabular Wide & Deep PyTorch Network ---
oof_preds_nn = np.zeros(n_train, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)
test_loader = DataLoader(
    TensorDataset(X_test_tensor), batch_size=2048, shuffle=False, num_workers=2
)

batch_size = 1024
epochs = 12

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr = torch.tensor(X_train_dense[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
    X_va = torch.tensor(X_train_dense[val_idx], dtype=torch.float32)
    y_va = torch.tensor(y_train[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va),
        batch_size=2048,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = TabularWideAndDeepNet(
        dense_dim=dense_dim, hidden_dims=[256, 128, 64], dropout_rate=0.20
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=1e-5, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=1e-6
    )

    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_v, _ in val_loader:
                x_v = x_v.to(device)
                logits = model(x_v)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.append(probs)

        val_probs_epoch = np.concatenate(val_preds_list)
        val_auc = roc_auc_score(y_train[val_idx], val_probs_epoch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for x_v, _ in val_loader:
            x_v = x_v.to(device)
            val_preds_list.append(torch.sigmoid(model(x_v)).cpu().numpy())
    oof_preds_nn[val_idx] = np.concatenate(val_preds_list)

    test_fold_preds = []
    with torch.no_grad():
        for (x_t,) in test_loader:
            x_t = x_t.to(device)
            test_fold_preds.append(torch.sigmoid(model(x_t)).cpu().numpy())
    test_preds_nn += np.concatenate(test_fold_preds) / n_folds

# --- Model 3: LightGBM Gradient Boosting ---
oof_preds_lgb = np.zeros(n_train, dtype=np.float64)
test_preds_lgb = np.zeros(n_test, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_train_dense[train_idx], y_train[train_idx]
    X_va, y_va = X_train_dense[val_idx], y_train[val_idx]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    val_probs = lgb_model.predict_proba(X_va)[:, 1]
    oof_preds_lgb[val_idx] = val_probs
    test_preds_lgb += lgb_model.predict_proba(X_test_dense)[:, 1] / n_folds


# =========================================================================
# Ensemble Optimization & Final Blending
# =========================================================================
def ensemble_loss_fn(weights):
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3
    if w_sum <= 0:
        return 1.0
    w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
    blended_oof = w1 * oof_preds_lr + w2 * oof_preds_nn + w3 * oof_preds_lgb
    return -roc_auc_score(y_train, blended_oof)


opt_res = minimize(
    ensemble_loss_fn,
    x0=[0.60, 0.25, 0.15],
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    method="SLSQP",
)

raw_weights = opt_res.x
best_weights = raw_weights / np.sum(raw_weights)

final_oof_preds = (
    best_weights[0] * oof_preds_lr
    + best_weights[1] * oof_preds_nn
    + best_weights[2] * oof_preds_lgb
)

final_test_preds = (
    best_weights[0] * test_preds_lr
    + best_weights[1] * test_preds_nn
    + best_weights[2] * test_preds_lgb
)

final_validation_score = roc_auc_score(y_train, final_oof_preds)

# =========================================================================
# Generate Submission File
# =========================================================================
sub_df = pd.DataFrame({"id": test_ids.astype(int), "target": final_test_preds})
sub_path = "./submission/submission.csv"
sub_df.to_csv(sub_path, index=False)

assert (
    len(sub_df) == n_test
), f"Mismatch in test sample count: {len(sub_df)} vs {n_test}"
assert list(sub_df.columns) == ["id", "target"], f"Incorrect columns: {sub_df.columns}"
assert not sub_df["target"].isna().any(), "Detected NaN values in test predictions"

gc.collect()

print(f"Final Validation Score: {final_validation_score}")