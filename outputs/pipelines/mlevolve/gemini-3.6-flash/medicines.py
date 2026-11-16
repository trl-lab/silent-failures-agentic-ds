import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Directory Setup & Environment Configuration
# ---------------------------------------------------------
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Data Loading & Base Parsing
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Handle ID column
id_col = "id" if "id" in test_df.columns else None
if id_col and id_col in train_df.columns:
    train_ids = train_df[id_col].values
    test_ids = test_df[id_col].values
    train_df = train_df.drop(columns=[id_col])
    test_df = test_df.drop(columns=[id_col])
else:
    train_ids = np.arange(len(train_df))
    test_ids = np.arange(len(test_df))

# Isolate Target
target_col = "Medicine status"
y_raw = train_df[target_col].copy()
train_df = train_df.drop(columns=[target_col])

# Encode Target if categorical or string
if y_raw.dtype == "object" or isinstance(y_raw.iloc[0], str):
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
else:
    y = y_raw.values.astype(int)

# Identify base data types
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()


# ---------------------------------------------------------
# 3. Dynamic Feature Engineering Strategy
# ---------------------------------------------------------
def create_engineered_features(train_data, test_data):
    tr = train_data.copy()
    te = test_data.copy()

    # A. Missingness Indicator Features
    for col in num_cols + cat_cols:
        if tr[col].isnull().sum() > 0 or te[col].isnull().sum() > 0:
            tr[f"{col}_isna"] = tr[col].isnull().astype(int)
            te[f"{col}_isna"] = te[col].isnull().astype(int)

    # B. Categorical Imputation & Frequency Encoding
    for col in cat_cols:
        tr[col] = tr[col].fillna("Missing").astype(str)
        te[col] = te[col].fillna("Missing").astype(str)

        freq = tr[col].value_counts(normalize=True).to_dict()
        tr[f"{col}_freq"] = tr[col].map(freq).fillna(0)
        te[f"{col}_freq"] = te[col].map(freq).fillna(0)

    # C. Numerical Imputation, Skewness Correction & Transforms
    for col in num_cols:
        med_val = tr[col].median()
        tr[col] = tr[col].fillna(med_val)
        te[col] = te[col].fillna(med_val)

        skewness = tr[col].skew()
        if abs(skewness) > 1.0:
            min_val = min(tr[col].min(), te[col].min())
            shift = abs(min_val) + 1.0 if min_val <= 0 else 0
            tr[f"{col}_log1p"] = np.log1p(tr[col] + shift)
            te[f"{col}_log1p"] = np.log1p(te[col] + shift)

    # D. Row-level Summary Aggregations
    if len(num_cols) >= 2:
        tr["num_mean"] = tr[num_cols].mean(axis=1)
        te["num_mean"] = te[num_cols].mean(axis=1)
        tr["num_std"] = tr[num_cols].std(axis=1).fillna(0)
        te["num_std"] = te[num_cols].std(axis=1).fillna(0)
        tr["num_min"] = tr[num_cols].min(axis=1)
        te["num_min"] = te[num_cols].min(axis=1)
        tr["num_max"] = tr[num_cols].max(axis=1)
        te["num_max"] = te[num_cols].max(axis=1)
        tr["num_sum"] = tr[num_cols].sum(axis=1)
        te["num_sum"] = te[num_cols].sum(axis=1)

    # E. Pairwise Numeric Interactions
    if len(num_cols) >= 2:
        top_num = num_cols[: min(5, len(num_cols))]
        for i in range(len(top_num)):
            for j in range(i + 1, len(top_num)):
                c1, c2 = top_num[i], top_num[j]
                tr[f"{c1}_plus_{c2}"] = tr[c1] + tr[c2]
                te[f"{c1}_plus_{c2}"] = te[c1] + te[c2]
                tr[f"{c1}_minus_{c2}"] = tr[c1] - tr[c2]
                te[f"{c1}_minus_{c2}"] = te[c1] - te[c2]
                tr[f"{c1}_ratio_{c2}"] = tr[c1] / (tr[c2].abs() + 1e-5)
                te[f"{c1}_ratio_{c2}"] = te[c1] / (te[c2].abs() + 1e-5)

    # F. Grouped Numerical Aggregations
    if len(cat_cols) > 0 and len(num_cols) > 0:
        for c_col in cat_cols[: min(3, len(cat_cols))]:
            for n_col in num_cols[: min(3, len(num_cols))]:
                grp = tr.groupby(c_col)[n_col].agg(["mean", "std"]).reset_index()
                grp.columns = [
                    c_col,
                    f"{c_col}_{n_col}_grp_mean",
                    f"{c_col}_{n_col}_grp_std",
                ]
                tr = tr.merge(grp, on=c_col, how="left")
                te = te.merge(grp, on=c_col, how="left")
                te[f"{c_col}_{n_col}_grp_mean"] = te[
                    f"{c_col}_{n_col}_grp_mean"
                ].fillna(tr[n_col].mean())
                te[f"{c_col}_{n_col}_grp_std"] = te[f"{c_col}_{n_col}_grp_std"].fillna(
                    0
                )

    # G. Ordinal Label Encoding for Categoricals
    all_cats = tr.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in all_cats:
        le = LabelEncoder()
        comb = pd.concat([tr[col].astype(str), te[col].astype(str)], axis=0)
        le.fit(comb)
        tr[col] = le.transform(tr[col].astype(str))
        te[col] = le.transform(te[col].astype(str))

    return tr, te


X_train, X_test = create_engineered_features(train_df, test_df)

# ---------------------------------------------------------
# 4. Out-Of-Fold Target Encoding (Leakage Free)
# ---------------------------------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
global_target_mean = y.mean()

for col in cat_cols:
    te_col = f"{col}_target_enc"
    X_train[te_col] = 0.0
    X_test[te_col] = 0.0

    # Calculate target encoding on full train for test set
    cat_stats = (
        pd.DataFrame({"cat": train_df[col].astype(str), "target": y})
        .groupby("cat")["target"]
        .agg(["mean", "count"])
    )
    smooth_m = 10.0
    smoothed_te = (
        cat_stats["mean"] * cat_stats["count"] + global_target_mean * smooth_m
    ) / (cat_stats["count"] + smooth_m)
    X_test[te_col] = (
        test_df[col].astype(str).map(smoothed_te).fillna(global_target_mean)
    )

    # OOF calculation for train set
    for tr_idx, va_idx in skf.split(X_train, y):
        tr_cat = train_df[col].iloc[tr_idx].astype(str)
        tr_y = y[tr_idx]
        fold_global_mean = tr_y.mean()
        df_tr = pd.DataFrame({"cat": tr_cat, "target": tr_y})
        stats = df_tr.groupby("cat")["target"].agg(["mean", "count"])
        smoothed_oof = (
            stats["mean"] * stats["count"] + fold_global_mean * smooth_m
        ) / (stats["count"] + smooth_m)
        X_train.loc[X_train.index[va_idx], te_col] = (
            train_df[col]
            .iloc[va_idx]
            .astype(str)
            .map(smoothed_oof)
            .fillna(fold_global_mean)
        )

# Sanitize feature names for LightGBM and XGBoost compatibility
clean_cols = []
seen = {}
for col in X_train.columns:
    c = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
    if c in seen:
        seen[c] += 1
        c = f"{c}_{seen[c]}"
    else:
        seen[c] = 0
    clean_cols.append(c)

X_train.columns = clean_cols
X_test.columns = clean_cols

# Save processed datasets
processed_train = X_train.copy()
processed_train[target_col] = y
processed_train.to_parquet("./working/train_engineered.parquet", index=False)
X_test.to_parquet("./working/test_engineered.parquet", index=False)


# ---------------------------------------------------------
# 5. Model Architecture & Model Factory Definitions
# ---------------------------------------------------------
feature_cols = list(X_train.columns)
nn_cat_cols = [
    c
    for c in feature_cols
    if X_train[c].dtype in [np.int64, np.int32]
    and X_train[c].nunique() < 100
    and not c.endswith("_isna")
]
cat_cols_idx = [feature_cols.index(c) for c in nn_cat_cols]
num_cols_idx = [i for i in range(len(feature_cols)) if i not in cat_cols_idx]

cat_dims = [int(X_train[c].max() + 1) for c in nn_cat_cols]
cat_emb_dims = [max(4, min(32, (x + 1) // 2)) for x in cat_dims]


class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_rate=0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Linear(in_features, hidden_features)
            if in_features != hidden_features
            else nn.Identity()
        )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.drop1(self.act1(self.bn1(self.fc1(x))))
        out = self.drop2(self.act2(self.bn2(self.fc2(out))))
        return out + res


class TabularResNet(nn.Module):
    def __init__(
        self,
        num_cols_idx,
        cat_cols_idx,
        cat_dims,
        cat_emb_dims,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
    ):
        super().__init__()
        self.num_cols_idx = num_cols_idx
        self.cat_cols_idx = cat_cols_idx

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes + 2, emb_dim)
                for num_classes, emb_dim in zip(cat_dims, cat_emb_dims)
            ]
        )

        total_emb_dim = sum(cat_emb_dims)
        num_numerical = len(num_cols_idx)
        self.num_bn = nn.BatchNorm1d(num_numerical) if num_numerical > 0 else None

        in_dim = total_emb_dim + num_numerical

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        curr_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(curr_dim, h_dim, dropout_rate=dropout_rate))
            curr_dim = h_dim
        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x):
        if len(self.cat_cols_idx) > 0:
            x_cat = x[:, self.cat_cols_idx].long()
            emb_outs = []
            for i, emb in enumerate(self.embeddings):
                c_val = torch.clamp(x_cat[:, i], 0, emb.num_embeddings - 1)
                emb_outs.append(emb(c_val))
            x_emb = torch.cat(emb_outs, dim=1)
        else:
            x_emb = None

        if len(self.num_cols_idx) > 0:
            x_num = x[:, self.num_cols_idx].float()
            if self.num_bn is not None:
                x_num = self.num_bn(x_num)
        else:
            x_num = None

        if x_emb is not None and x_num is not None:
            feat = torch.cat([x_emb, x_num], dim=1)
        elif x_emb is not None:
            feat = x_emb
        else:
            feat = x_num

        feat = self.input_proj(feat)
        feat = self.backbone(feat)
        logits = self.head(feat)
        return logits.squeeze(-1)


def create_lgb_model():
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=20,
        random_state=42,
        n_estimators=1500,
        verbose=-1,
    )


def create_xgb_model():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        tree_method="hist",
        random_state=42,
        n_estimators=1500,
        early_stopping_rounds=50,
    )


# ---------------------------------------------------------
# 6. Multi-Model 5-Fold Cross-Validation Execution
# ---------------------------------------------------------
lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))

xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros(len(X_test))

nn_oof = np.zeros(len(X_train))
nn_test_preds = np.zeros(len(X_test))

X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
    X_tr_df, y_tr = X_train.iloc[tr_idx], y[tr_idx]
    X_va_df, y_va = X_train.iloc[va_idx], y[va_idx]

    # --- A. LightGBM ---
    model_lgb = create_lgb_model()
    model_lgb.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_oof[va_idx] = model_lgb.predict_proba(X_va_df)[:, 1]
    lgb_test_preds += model_lgb.predict_proba(X_test)[:, 1] / n_splits

    # --- B. XGBoost ---
    model_xgb = create_xgb_model()
    model_xgb.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        verbose=False,
    )
    xgb_oof[va_idx] = model_xgb.predict_proba(X_va_df)[:, 1]
    xgb_test_preds += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    # --- C. PyTorch TabularResNet ---
    X_tr_tensor = torch.tensor(X_train_np[tr_idx], dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)
    X_va_tensor = torch.tensor(X_train_np[va_idx], dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model_nn = TabularResNet(
        num_cols_idx=num_cols_idx,
        cat_cols_idx=cat_cols_idx,
        cat_dims=cat_dims,
        cat_emb_dims=cat_emb_dims,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
    ).to(device)

    opt_nn = optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt_nn, mode="max", factor=0.5, patience=3
    )

    best_va_auc = 0.0
    best_va_preds = None
    best_test_preds_fold = None
    patience = 12
    patience_counter = 0

    for epoch in range(45):
        model_nn.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            opt_nn.zero_grad()
            logits = model_nn(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt_nn.step()

        model_nn.eval()
        with torch.no_grad():
            va_logits = model_nn(X_va_tensor)
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
            va_auc = roc_auc_score(y_va, va_probs)
            scheduler.step(va_auc)

            if va_auc > best_va_auc:
                best_va_auc = va_auc
                best_va_preds = va_probs
                test_logits = model_nn(X_test_tensor)
                best_test_preds_fold = torch.sigmoid(test_logits).cpu().numpy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    nn_oof[va_idx] = best_va_preds
    nn_test_preds += best_test_preds_fold / n_splits

# ---------------------------------------------------------
# 7. Optimal Ensemble Weight Optimization & Final Blending
# ---------------------------------------------------------
lgb_score = roc_auc_score(y, lgb_oof)
xgb_score = roc_auc_score(y, xgb_oof)
nn_score = roc_auc_score(y, nn_oof)

best_score = 0.0
best_weights = (0.33, 0.33, 0.34)

for w1 in np.linspace(0.0, 1.0, 21):
    for w2 in np.linspace(0.0, 1.0 - w1, 21):
        w3 = max(0.0, 1.0 - w1 - w2)
        oof_blend = w1 * lgb_oof + w2 * xgb_oof + w3 * nn_oof
        score = roc_auc_score(y, oof_blend)
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

w_lgb, w_xgb, w_nn = best_weights
final_test_preds = (
    w_lgb * lgb_test_preds + w_xgb * xgb_test_preds + w_nn * nn_test_preds
)

# ---------------------------------------------------------
# 8. Submission Output & Final Score Reporting
# ---------------------------------------------------------
submission = pd.DataFrame({"id": test_ids, "Medicine status": final_test_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {best_score}")