import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Ensure required output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)

# 2. Load raw datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Extract target column and map to binary {0, 1}
target_candidates = [
    c for c in train_df.columns if c.upper() in ["APPETENCY", "TARGET"]
]
target_col = target_candidates[0] if target_candidates else train_df.columns[-1]

y_raw = train_df[target_col]
y = (
    y_raw.astype(str)
    .str.strip()
    .apply(lambda x: 1 if x in ["1", "1.0", "True", "true", "+1"] else 0)
    .values
)

train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()

# Preserve common feature columns
common_cols = [c for c in train_features.columns if c in test_features.columns]
train_features = train_features[common_cols]
test_features = test_features[common_cols]

# Extract test identifiers
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "ID" in test_df.columns:
    test_ids = test_df["ID"].values
else:
    test_ids = np.arange(len(test_df))

# 3. Categorize Numerical and Categorical Features
num_cols = []
cat_cols = []

for c in common_cols:
    if c.lower().startswith("var"):
        try:
            idx = int(re.findall(r"\d+", c)[0])
            if idx <= 190:
                num_cols.append(c)
            else:
                cat_cols.append(c)
        except Exception:
            if pd.api.types.is_numeric_dtype(train_features[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)
    else:
        if pd.api.types.is_numeric_dtype(train_features[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

for c in num_cols:
    train_features[c] = pd.to_numeric(train_features[c], errors="coerce")
    test_features[c] = pd.to_numeric(test_features[c], errors="coerce")


# 4. Feature Engineering: Missingness Patterns
def compute_missingness(df, n_cols, c_cols):
    df_miss = pd.DataFrame(index=df.index)
    df_miss["num_missing_total"] = df.isnull().sum(axis=1)
    df_miss["pct_missing_total"] = df_miss["num_missing_total"] / float(len(df.columns))

    if len(n_cols) > 0:
        df_miss["num_missing_num"] = df[n_cols].isnull().sum(axis=1)
        df_miss["pct_missing_num"] = df_miss["num_missing_num"] / float(len(n_cols))

    if len(c_cols) > 0:
        df_miss["num_missing_cat"] = df[c_cols].isnull().sum(axis=1)
        df_miss["pct_missing_cat"] = df_miss["num_missing_cat"] / float(len(c_cols))

    return df_miss


train_miss = compute_missingness(train_features, num_cols, cat_cols)
test_miss = compute_missingness(test_features, num_cols, cat_cols)


# 5. Feature Engineering: Row-level Numerical Aggregations
def compute_num_aggregations(df, n_cols):
    if not n_cols:
        return pd.DataFrame(index=df.index)
    arr = df[n_cols].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_val = np.nanmean(arr, axis=1)
        std_val = np.nanstd(arr, axis=1)
        min_val = np.nanmin(arr, axis=1)
        max_val = np.nanmax(arr, axis=1)
        median_val = np.nanmedian(arr, axis=1)
        nz_count = np.nansum(arr != 0, axis=1)

    return pd.DataFrame(
        {
            "row_num_mean": np.nan_to_num(mean_val, nan=0.0),
            "row_num_std": np.nan_to_num(std_val, nan=0.0),
            "row_num_min": np.nan_to_num(min_val, nan=0.0),
            "row_num_max": np.nan_to_num(max_val, nan=0.0),
            "row_num_median": np.nan_to_num(median_val, nan=0.0),
            "row_num_nz_count": nz_count,
        },
        index=df.index,
    )


train_num_stats = compute_num_aggregations(train_features, num_cols)
test_num_stats = compute_num_aggregations(test_features, num_cols)

# 6. Feature Engineering: Categorical Processing & Frequency Encoding
for col in cat_cols:
    train_features[col] = train_features[col].astype(object).fillna("MISSING").astype(str).str.lower()
    test_features[col] = test_features[col].astype(object).fillna("MISSING").astype(str).str.lower()

freq_train = pd.DataFrame(index=train_features.index)
freq_test = pd.DataFrame(index=test_features.index)

for col in cat_cols:
    vc = train_features[col].value_counts()
    freq_train[f"{col}_freq"] = train_features[col].map(vc).fillna(0).astype(np.float32)
    freq_test[f"{col}_freq"] = test_features[col].map(vc).fillna(0).astype(np.float32)

# 7. Feature Engineering: Out-of-Fold Target Encoding
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_train = pd.DataFrame(
    0.0, index=train_features.index, columns=[f"{c}_te" for c in cat_cols]
)
te_test = pd.DataFrame(
    0.0, index=test_features.index, columns=[f"{c}_te" for c in cat_cols]
)

m_smooth = 10.0
global_mean = float(y.mean())

for tr_idx, va_idx in skf_te.split(train_features, y):
    y_fold = y[tr_idx]
    for c in cat_cols:
        s_tr = train_features[c].iloc[tr_idx]
        s_va = train_features[c].iloc[va_idx]
        stats = (
            pd.DataFrame({"cat": s_tr, "target": y_fold})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
            stats["count"] + m_smooth
        )
        mapping = smoothed.to_dict()
        te_train.iloc[va_idx, te_train.columns.get_loc(f"{c}_te")] = s_va.map(
            mapping
        ).fillna(global_mean).values

for c in cat_cols:
    stats = (
        pd.DataFrame({"cat": train_features[c], "target": y})
        .groupby("cat")["target"]
        .agg(["count", "mean"])
    )
    smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
        stats["count"] + m_smooth
    )
    mapping = smoothed.to_dict()
    te_test[f"{c}_te"] = test_features[c].map(mapping).fillna(global_mean)

# 8. Feature Engineering: Ordinal Encoding
cat_ord_train = pd.DataFrame(index=train_features.index)
cat_ord_test = pd.DataFrame(index=test_features.index)

for col in cat_cols:
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
    )
    cat_ord_train[col] = oe.fit_transform(train_features[[col]]).ravel()
    cat_ord_test[col] = oe.transform(test_features[[col]]).ravel()

# 9. Feature Engineering: Numerical Cleaning & Quantile Clipping
clean_num_train = pd.DataFrame(index=train_features.index)
clean_num_test = pd.DataFrame(index=test_features.index)

for col in num_cols:
    s_tr = train_features[col]
    s_te = test_features[col]

    if s_tr.isnull().mean() >= 0.999:
        continue

    valid_vals = s_tr.dropna()
    if len(valid_vals) == 0 or valid_vals.std() == 0:
        continue

    med = valid_vals.median()
    tr_filled = s_tr.fillna(med).astype(np.float32)
    te_filled = s_te.fillna(med).astype(np.float32)

    q_low = valid_vals.quantile(0.005)
    q_high = valid_vals.quantile(0.995)
    if q_high > q_low:
        tr_filled = tr_filled.clip(q_low, q_high)
        te_filled = te_filled.clip(q_low, q_high)

    clean_num_train[col] = tr_filled
    clean_num_test[col] = te_filled

# Assembly of complete feature matrices
X_train = pd.concat(
    [
        clean_num_train,
        cat_ord_train,
        train_miss,
        train_num_stats,
        freq_train,
        te_train,
    ],
    axis=1,
)
X_test = pd.concat(
    [
        clean_num_test,
        cat_ord_test,
        test_miss,
        test_num_stats,
        freq_test,
        te_test,
    ],
    axis=1,
)

X_train.columns = [str(c) for c in X_train.columns]
X_test.columns = [str(c) for c in X_test.columns]

# Dense representations for XGBoost and PyTorch inputs
X_train_clean = X_train.fillna(0.0).values.astype(np.float32)
X_test_clean = X_test.fillna(0.0).values.astype(np.float32)


# 10. PyTorch Deep Tabular MLP Definition
class TabularNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_pytorch_fold(
    X_tr_sc,
    y_tr,
    X_va_sc,
    y_va,
    X_te_sc,
    input_dim,
    device,
    fold,
    epochs=15,
    batch_size=512,
):
    torch.manual_seed(42 + fold)
    model = TabularNN(input_dim).to(device)

    pos_count = (y_tr == 1).sum()
    neg_count = (y_tr == 0).sum()
    pos_weight_val = float(neg_count) / max(1, pos_count)
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    X_va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
    X_te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)

    best_val_auc = -1.0
    best_va_preds = np.zeros(len(X_va_sc))
    best_te_preds = np.zeros(len(X_te_sc))
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_probs = torch.sigmoid(model(X_va_tensor)).cpu().numpy()
            te_probs = torch.sigmoid(model(X_te_tensor)).cpu().numpy()

        val_auc = roc_auc_score(y_va, va_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_va_preds = va_probs
            best_te_preds = te_probs
            best_state_dict = model.state_dict().copy()

    if best_state_dict is not None:
        torch.save(best_state_dict, f"./working/models/pytorch_fold_{fold}.pt")

    return best_va_preds, best_te_preds


# 11. Stratified 5-Fold Ensembling Strategy across 4 Models
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
oof_nn = np.zeros(len(X_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train_clean.shape[1]
scale_pos_weight_val = float((y == 0).sum()) / float((y == 1).sum())

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_clean, y)):
    X_tr_pd, y_tr = X_train.iloc[tr_idx], y[tr_idx]
    X_va_pd, y_va = X_train.iloc[va_idx], y[va_idx]

    X_tr_np, X_va_np = X_train_clean[tr_idx], X_train_clean[va_idx]

    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        is_unbalance=True,
        learning_rate=0.03,
        n_estimators=1000,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr_pd,
        y_tr,
        eval_set=[(X_va_pd, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[va_idx] = model_lgb.predict_proba(X_va_pd)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits
    model_lgb.booster_.save_model(f"./working/models/lgbm_fold_{fold}.txt")

    # Model 2: XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight_val,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
    )
    model_xgb.fit(
        X_tr_np,
        y_tr,
        eval_set=[(X_va_np, y_va)],
        verbose=False,
    )
    oof_xgb[va_idx] = model_xgb.predict_proba(X_va_np)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test_clean)[:, 1] / n_splits
    model_xgb.save_model(f"./working/models/xgb_fold_{fold}.json")

    # Model 3: CatBoost
    model_cb = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cb.fit(
        X_tr_np,
        y_tr,
        eval_set=(X_va_np, y_va),
        use_best_model=True,
    )
    oof_cb[va_idx] = model_cb.predict_proba(X_va_np)[:, 1]
    test_cb += model_cb.predict_proba(X_test_clean)[:, 1] / n_splits
    model_cb.save_model(f"./working/models/catboost_fold_{fold}.cbm")

    # Model 4: PyTorch Deep Tabular MLP
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_np)
    X_va_sc = scaler.transform(X_va_np)
    X_te_sc = scaler.transform(X_test_clean)

    nn_va_preds, nn_te_preds = train_pytorch_fold(
        X_tr_sc,
        y_tr,
        X_va_sc,
        y_va,
        X_te_sc,
        input_dim,
        device,
        fold,
        epochs=15,
        batch_size=512,
    )
    oof_nn[va_idx] = nn_va_preds
    test_nn += nn_te_preds / n_splits

# 12. Rank Transformation and SLSQP Optimization
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_cb_rank = rankdata(oof_cb) / len(oof_cb)
oof_nn_rank = rankdata(oof_nn) / len(oof_nn)

test_lgb_rank = rankdata(test_lgb) / len(test_lgb)
test_xgb_rank = rankdata(test_xgb) / len(test_xgb)
test_cb_rank = rankdata(test_cb) / len(test_cb)
test_nn_rank = rankdata(test_nn) / len(test_nn)


def ensemble_loss(weights):
    weights = np.array(weights)
    weights = np.maximum(weights, 0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)

    blend = (
        weights[0] * oof_lgb_rank
        + weights[1] * oof_xgb_rank
        + weights[2] * oof_cb_rank
        + weights[3] * oof_nn_rank
    )
    return -roc_auc_score(y, blend)


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 100},
)

opt_weights = opt_res.x
opt_weights = np.maximum(opt_weights, 0)
opt_weights /= opt_weights.sum()

oof_ensemble = (
    opt_weights[0] * oof_lgb_rank
    + opt_weights[1] * oof_xgb_rank
    + opt_weights[2] * oof_cb_rank
    + opt_weights[3] * oof_nn_rank
)

test_ensemble = (
    opt_weights[0] * test_lgb_rank
    + opt_weights[1] * test_xgb_rank
    + opt_weights[2] * test_cb_rank
    + opt_weights[3] * test_nn_rank
)

cv_score = roc_auc_score(y, oof_ensemble)

# 13. Format and Output Final Submission
sub = pd.DataFrame({"id": test_ids, "APPETENCY": test_ensemble})
sub.to_csv("./submission/submission.csv", index=False)

assert len(sub) == len(test_df), "Submission row count mismatch!"
assert not sub["APPETENCY"].isnull().any(), "Submission contains NaN values!"

print(f"Final Validation Score: {cv_score}")
