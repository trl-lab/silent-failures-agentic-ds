import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Directory Setup
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 2. Data Loading
print("Loading datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "AwardTotal"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

# Ensure standard 0-based integer id column
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))

y_train = train_df[target_col].values.astype(np.float64)
y_train_log = np.log1p(np.maximum(0, y_train))

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()
if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

# 3. Dynamic Column Identification
feature_cols = [c for c in X_train_raw.columns if c != "id"]
num_cols, cat_cols, text_cols, date_cols = [], [], [], []

for c in feature_cols:
    train_s = X_train_raw[c]

    if pd.api.types.is_datetime64_any_dtype(train_s):
        date_cols.append(c)
        continue

    if pd.api.types.is_numeric_dtype(train_s):
        num_cols.append(c)
        continue

    if pd.api.types.is_string_dtype(train_s) or pd.api.types.is_object_dtype(train_s):
        sample_str = train_s.dropna().astype(str)
        avg_len = sample_str.str.len().mean() if len(sample_str) > 0 else 0
        avg_words = (
            sample_str.str.split().str.len().mean() if len(sample_str) > 0 else 0
        )

        if avg_words > 5 or avg_len > 50:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        cat_cols.append(c)

print(
    f"Discovered features - Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}, Text: {len(text_cols)}, Date: {len(date_cols)}"
)

# 4. Feature Engineering
X_train_feat = pd.DataFrame(index=train_df.index)
X_test_feat = pd.DataFrame(index=test_df.index)

# Date Features
for c in date_cols:
    for df_raw, df_feat in [
        (X_train_raw, X_train_feat),
        (X_test_raw, X_test_feat),
    ]:
        dt_s = pd.to_datetime(df_raw[c], errors="coerce")
        df_feat[f"{c}_year"] = dt_s.dt.year.fillna(-1).astype(np.float32)
        df_feat[f"{c}_month"] = dt_s.dt.month.fillna(-1).astype(np.float32)
        df_feat[f"{c}_day"] = dt_s.dt.day.fillna(-1).astype(np.float32)
        df_feat[f"{c}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1).astype(np.float32)
        df_feat[f"{c}_quarter"] = dt_s.dt.quarter.fillna(-1).astype(np.float32)

# Numeric Features
for c in num_cols:
    tr_vals = X_train_raw[c].values.astype(np.float32)
    te_vals = X_test_raw[c].values.astype(np.float32)

    med = np.nanmedian(tr_vals)
    if np.isnan(med):
        med = 0.0

    tr_imputed = np.where(np.isnan(tr_vals), med, tr_vals)
    te_imputed = np.where(np.isnan(te_vals), med, te_vals)

    X_train_feat[f"num_{c}"] = tr_imputed
    X_test_feat[f"num_{c}"] = te_imputed

    if np.isnan(tr_vals).any() or np.isnan(te_vals).any():
        X_train_feat[f"num_{c}_isna"] = np.isnan(tr_vals).astype(np.float32)
        X_test_feat[f"num_{c}_isna"] = np.isnan(te_vals).astype(np.float32)

    if (tr_imputed >= 0).all() and (np.max(tr_imputed) > 100):
        X_train_feat[f"num_{c}_log1p"] = np.log1p(tr_imputed)
        X_test_feat[f"num_{c}_log1p"] = np.log1p(te_imputed)

# Categorical Features (Frequency & OOF Target Encoding)
kf_target = KFold(n_splits=5, shuffle=True, random_state=42)

for c in cat_cols:
    tr_cat = X_train_raw[c].astype(str).fillna("missing")
    te_cat = X_test_raw[c].astype(str).fillna("missing")

    freq_map = tr_cat.value_counts(normalize=True).to_dict()
    X_train_feat[f"cat_{c}_freq"] = (
        tr_cat.map(freq_map).fillna(0).values.astype(np.float32)
    )
    X_test_feat[f"cat_{c}_freq"] = (
        te_cat.map(freq_map).fillna(0).values.astype(np.float32)
    )

    oof_target_enc = np.zeros(len(train_df), dtype=np.float32)
    global_mean = y_train_log.mean()

    for train_idx, val_idx in kf_target.split(train_df):
        tr_sub_cat = tr_cat.iloc[train_idx]
        tr_sub_y = y_train_log[train_idx]

        cat_means = (
            pd.DataFrame({"cat": tr_sub_cat, "y": tr_sub_y})
            .groupby("cat")["y"]
            .mean()
            .to_dict()
        )
        oof_target_enc[val_idx] = (
            tr_cat.iloc[val_idx].map(cat_means).fillna(global_mean)
        )

    full_cat_means = (
        pd.DataFrame({"cat": tr_cat, "y": y_train_log})
        .groupby("cat")["y"]
        .mean()
        .to_dict()
    )
    X_train_feat[f"cat_{c}_te"] = oof_target_enc
    X_test_feat[f"cat_{c}_te"] = (
        te_cat.map(full_cat_means).fillna(global_mean).values.astype(np.float32)
    )

# Text Features
if len(text_cols) > 0:
    for c in text_cols:
        tr_text = X_train_raw[c].astype(str).fillna("")
        te_text = X_test_raw[c].astype(str).fillna("")

        X_train_feat[f"text_{c}_len"] = tr_text.str.len().values.astype(np.float32)
        X_test_feat[f"text_{c}_len"] = te_text.str.len().values.astype(np.float32)
        X_train_feat[f"text_{c}_words"] = (
            tr_text.str.split().str.len().values.astype(np.float32)
        )
        X_test_feat[f"text_{c}_words"] = (
            te_text.str.split().str.len().values.astype(np.float32)
        )

    tr_combined_text = (
        X_train_raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    te_combined_text = (
        X_test_raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )

    tfidf = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2)
    )
    svd = TruncatedSVD(n_components=30, random_state=42)

    tr_tfidf = tfidf.fit_transform(tr_combined_text)
    te_tfidf = tfidf.transform(te_combined_text)

    tr_svd = svd.fit_transform(tr_tfidf)
    te_svd = svd.transform(te_tfidf)

    svd_cols = [f"text_svd_{i}" for i in range(tr_svd.shape[1])]
    svd_tr_df = pd.DataFrame(tr_svd, columns=svd_cols, index=train_df.index, dtype=np.float32)
    svd_te_df = pd.DataFrame(te_svd, columns=svd_cols, index=test_df.index, dtype=np.float32)
    X_train_feat = pd.concat([X_train_feat, svd_tr_df], axis=1)
    X_test_feat = pd.concat([X_test_feat, svd_te_df], axis=1)

X_train_feat = X_train_feat.fillna(0.0)
X_test_feat = X_test_feat.fillna(0.0)
X_train_feat, X_test_feat = X_train_feat.align(
    X_test_feat, join="left", axis=1, fill_value=0.0
)

# Cross-Validation Scheme (Stratified Binned K-Fold)
num_bins = int(np.floor(1 + np.log2(len(train_df))))
y_binned = pd.qcut(y_train_log, q=num_bins, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(skf.split(X_train_feat, y_binned))


# 5. Model Architecture & Hyperparameters
class TabularResidualBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(TabularResidualBlock(curr_dim, h_dim, dropout=dropout))
            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats)
        return out.squeeze(-1)


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 1500,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 1500,
    "tree_method": "hist",
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "iterations": 1500,
    "verbose": False,
}

# 6. Model Training & Cross-Validation
X_train_arr = X_train_feat.values.astype(np.float32)
X_test_arr = X_test_feat.values.astype(np.float32)
input_dim = X_train_arr.shape[1]
n_train, n_test = len(X_train_arr), len(X_test_arr)
n_splits = len(cv_splits)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_cat = np.zeros(n_train, dtype=np.float32)
oof_nn = np.zeros(n_train, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cat = np.zeros(n_test, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)

print(f"Training ensemble models across {n_splits} folds on {device}...")

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    fold_scaler = StandardScaler()
    X_tr = fold_scaler.fit_transform(X_train_arr[train_idx])
    X_va = fold_scaler.transform(X_train_arr[val_idx])
    X_test_fold = fold_scaler.transform(X_test_arr)
    y_tr_log = y_train_log[train_idx]
    y_va_log = y_train_log[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_preds_lgb += lgb_model.predict(X_test_fold) / n_splits

    # XGBoost
    xgb_params_fold = xgb_params.copy()
    xgb_params_fold["early_stopping_rounds"] = 50
    xgb_model = xgb.XGBRegressor(**xgb_params_fold)
    xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_preds_xgb += xgb_model.predict(X_test_fold) / n_splits

    # CatBoost
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_tr,
        y_tr_log,
        eval_set=(X_va, y_va_log),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_va)
    test_preds_cat += cat_model.predict(X_test_fold) / n_splits

    # PyTorch TabularResNet
    nn_model = TabularResNet(
        input_dim=input_dim, hidden_dims=[256, 128, 64], dropout=0.2
    ).to(device)
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va_log, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    best_nn_state = None

    for epoch in range(40):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_losses.append(loss_fn(out, by).item())
        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_nn_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict(best_nn_state)
    nn_model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
        oof_nn[val_idx] = nn_model(va_tensor).cpu().numpy()

        te_tensor = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
        test_preds_nn += nn_model(te_tensor).cpu().numpy() / n_splits

# 7. Convert OOF and Test Predictions Back to Dollar Scale
oof_lgb_real = np.expm1(np.maximum(0, oof_lgb))
oof_xgb_real = np.expm1(np.maximum(0, oof_xgb))
oof_cat_real = np.expm1(np.maximum(0, oof_cat))
oof_nn_real = np.expm1(np.maximum(0, oof_nn))

test_lgb_real = np.expm1(np.maximum(0, test_preds_lgb))
test_xgb_real = np.expm1(np.maximum(0, test_preds_xgb))
test_cat_real = np.expm1(np.maximum(0, test_preds_cat))
test_nn_real = np.expm1(np.maximum(0, test_preds_nn))

# 8. Ensemble Optimization
oof_matrix = np.column_stack([oof_lgb_real, oof_xgb_real, oof_cat_real, oof_nn_real])
test_matrix = np.column_stack(
    [test_lgb_real, test_xgb_real, test_cat_real, test_nn_real]
)


def blend_objective(weights):
    w = np.array(weights) / np.sum(weights)
    blend = np.dot(oof_matrix, w)
    return root_mean_squared_error(y_train, blend)


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = res.x / np.sum(res.x)

# 9. Validation Evaluation & Submission Generation
final_oof_preds = np.maximum(0, np.dot(oof_matrix, best_weights))
final_score = root_mean_squared_error(y_train, final_oof_preds)

final_test_preds = np.maximum(0, np.dot(test_matrix, best_weights))
test_ids = test_df["id"].values

submission_df = pd.DataFrame({"id": test_ids, "AwardTotal": final_test_preds})
sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)

print(f"Final Validation Score: {final_score}")