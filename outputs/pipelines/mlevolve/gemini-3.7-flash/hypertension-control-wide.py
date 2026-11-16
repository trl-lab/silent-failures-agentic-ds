import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 0. Utility Functions & Seeds
# =========================================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_column_names(df):
    """Sanitize column names to prevent issues with downstream tree models."""
    df = df.copy()
    new_cols = []
    for col in df.columns:
        clean_col = re.sub(r"[^\w\s]", "_", str(col))
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col).strip("_")
        new_cols.append(clean_col)
    df.columns = new_cols
    return df


# =========================================================================
# 1. Deep Tabular Neural Network Architecture (TabularResNet)
# =========================================================================
class ResidualBlock(nn.Module):
    """Residual Block with LayerNorm, Dropout, and Swish Activation for Tabular Data."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """High-capacity Tabular Residual Network with multi-layer skip connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits


def build_pytorch_model_and_optimizer(
    input_dim: int,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    total_epochs: int = 40,
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        output_dim=1,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-6,
    )
    return model, criterion, optimizer, scheduler


def train_pytorch_fold(
    X_train_np,
    y_train_np,
    X_val_np,
    y_val_np,
    X_test_np,
    device,
    epochs=35,
    batch_size=128,
):
    """Train TabularResNet for a single fold with early stopping on validation AUC."""
    input_dim = X_train_np.shape[1]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_np)
    X_va = scaler.transform(X_val_np)
    X_te = scaler.transform(X_test_np)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model, criterion, optimizer, scheduler = build_pytorch_model_and_optimizer(
        input_dim=input_dim,
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.2,
        learning_rate=1e-3,
        total_epochs=epochs,
    )
    model.to(device)

    best_auc = -1.0
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                preds = torch.sigmoid(model(bx)).cpu().numpy().flatten()
                val_preds_list.extend(preds)

        val_auc = roc_auc_score(y_val_np, val_preds_list)
        if val_auc > best_auc:
            best_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        val_t = torch.tensor(X_va, dtype=torch.float32).to(device)
        oof_preds = torch.sigmoid(model(val_t)).cpu().numpy().flatten()

        test_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        test_preds = torch.sigmoid(model(test_t)).cpu().numpy().flatten()

    return oof_preds, test_preds


# =========================================================================
# 2. Model Hyperparameters
# =========================================================================
def get_lgbm_params(seed: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> dict:
    return {
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 5,
        "eval_metric": "AUC",
        "random_seed": seed,
        "verbose": False,
        "early_stopping_rounds": 50,
    }


# =========================================================================
# 3. Main End-to-End Pipeline
# =========================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    target_col = "Estimated Percentage of Patients with Controlled Blood Pressure"
    clean_target_col = re.sub(r"[^\w\s]", "_", target_col)
    clean_target_col = re.sub(r"\s+", "_", clean_target_col)
    clean_target_col = re.sub(r"_+", "_", clean_target_col).strip("_")

    # 1. Load Data
    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    df_train_raw = pd.read_parquet(train_path)
    df_test_raw = pd.read_parquet(test_path)

    test_ids = np.arange(len(df_test_raw))
    if "id" in df_test_raw.columns:
        test_ids = df_test_raw["id"].values
    elif "ID" in df_test_raw.columns:
        test_ids = df_test_raw["ID"].values

    train_df = clean_column_names(df_train_raw)
    test_df = clean_column_names(df_test_raw)

    # 2. Extract and Standardize Target
    if clean_target_col not in train_df.columns:
        candidates = [
            c
            for c in train_df.columns
            if "controlled" in c.lower()
            or "blood_pressure" in c.lower()
            or "hypertension" in c.lower()
        ]
        if candidates:
            clean_target_col = candidates[0]

    y_raw = train_df[clean_target_col].copy()

    if y_raw.dtype == "object" or str(y_raw.dtype) == "category":
        unique_vals = sorted(y_raw.dropna().unique())
        val_map = {val: i for i, val in enumerate(unique_vals)}
        y = y_raw.map(val_map).astype(int)
    else:
        if set(y_raw.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            y = y_raw.astype(int)
        else:
            median_val = y_raw.median()
            y = (y_raw >= median_val).astype(int)

    train_df["target"] = y
    initial_feature_cols = [
        c
        for c in train_df.columns
        if c not in ["target", clean_target_col, "id", "ID", "Id"]
    ]

    X_train = train_df[initial_feature_cols].copy()
    X_test = test_df[[c for c in initial_feature_cols if c in test_df.columns]].copy()

    for col in initial_feature_cols:
        if col not in X_test.columns:
            X_test[col] = np.nan

    # Identify and decompose Datetime / Timestamp columns
    date_cols = [
        c for c in initial_feature_cols
        if pd.api.types.is_datetime64_any_dtype(X_train[c])
        or pd.api.types.is_datetime64_any_dtype(X_test[c])
    ]

    for c in date_cols:
        tr_dt = pd.to_datetime(X_train[c], errors="coerce")
        te_dt = pd.to_datetime(X_test[c], errors="coerce")

        X_train[f"{c}_year"] = tr_dt.dt.year.astype(np.float32)
        X_test[f"{c}_year"] = te_dt.dt.year.astype(np.float32)

        X_train[f"{c}_month"] = tr_dt.dt.month.astype(np.float32)
        X_test[f"{c}_month"] = te_dt.dt.month.astype(np.float32)

        X_train[f"{c}_day"] = tr_dt.dt.day.astype(np.float32)
        X_test[f"{c}_day"] = te_dt.dt.day.astype(np.float32)

        X_train[f"{c}_dayofweek"] = tr_dt.dt.dayofweek.astype(np.float32)
        X_test[f"{c}_dayofweek"] = te_dt.dt.dayofweek.astype(np.float32)

        tr_epoch = (tr_dt.astype("int64") // 10**9).astype(np.float64)
        te_epoch = (te_dt.astype("int64") // 10**9).astype(np.float64)
        X_train[f"{c}_timestamp"] = tr_epoch.where(tr_dt.notnull(), np.nan).astype(np.float32)
        X_test[f"{c}_timestamp"] = te_epoch.where(te_dt.notnull(), np.nan).astype(np.float32)

        X_train.drop(columns=[c], inplace=True)
        X_test.drop(columns=[c], inplace=True)

    feature_cols = [c for c in X_train.columns if c in X_test.columns]

    cat_cols = []
    num_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # 3. Stratified 10-Fold Split
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        train_df.loc[val_idx, "fold"] = fold

    # 4. Missingness Indicators & Numeric Preprocessing
    if num_cols:
        high_missing_num = [c for c in num_cols if X_train[c].isnull().mean() > 0.02]
        for c in high_missing_num:
            X_train[f"{c}_isna"] = X_train[c].isnull().astype(np.float32)
            X_test[f"{c}_isna"] = X_test[c].isnull().astype(np.float32)

        X_train[num_cols] = X_train[num_cols].astype(np.float32)
        X_test[num_cols] = X_test[num_cols].astype(np.float32)

        num_medians = X_train[num_cols].median().fillna(0.0)
        X_train[num_cols] = X_train[num_cols].fillna(num_medians)
        X_test[num_cols] = X_test[num_cols].fillna(num_medians)

        # Robust non-linear quantile transforms for skewed continuous columns
        skewed_cols = [c for c in num_cols if abs(float(X_train[c].skew())) > 1.5]
        if skewed_cols:
            qt = QuantileTransformer(
                n_quantiles=min(1000, len(X_train)),
                output_distribution="normal",
                random_state=42,
            )
            qt_tr = qt.fit_transform(X_train[skewed_cols])
            qt_te = qt.transform(X_test[skewed_cols])
            for idx, c in enumerate(skewed_cols):
                X_train[f"{c}_qt"] = np.nan_to_num(qt_tr[:, idx], nan=0.0).astype(np.float32)
                X_test[f"{c}_qt"] = np.nan_to_num(qt_te[:, idx], nan=0.0).astype(np.float32)

        for c in num_cols:
            min_val = X_train[c].min()
            if pd.notnull(min_val) and min_val >= 0:
                if X_train[c].max() > 100 or X_train[c].skew() > 2.0:
                    X_train[f"{c}_log1p"] = np.log1p(X_train[c].clip(lower=0))
                    X_test[f"{c}_log1p"] = np.log1p(X_test[c].clip(lower=0))

        # Pairwise interaction ratios & relative differences among top continuous metrics
        top_inter_cols = num_cols[:10] if len(num_cols) > 10 else num_cols
        eps = 1e-6
        for i in range(len(top_inter_cols)):
            for j in range(i + 1, len(top_inter_cols)):
                c1, c2 = top_inter_cols[i], top_inter_cols[j]
                X_train[f"{c1}_div_{c2}"] = ((X_train[c1] + eps) / (X_train[c2] + eps)).astype(np.float32)
                X_test[f"{c1}_div_{c2}"] = ((X_test[c1] + eps) / (X_test[c2] + eps)).astype(np.float32)
                X_train[f"{c1}_reldiff_{c2}"] = (
                    (X_train[c1] - X_train[c2]) / (np.abs(X_train[c1]) + np.abs(X_train[c2]) + eps)
                ).astype(np.float32)
                X_test[f"{c1}_reldiff_{c2}"] = (
                    (X_test[c1] - X_test[c2]) / (np.abs(X_test[c1]) + np.abs(X_test[c2]) + eps)
                ).astype(np.float32)

        if len(num_cols) >= 4:
            X_train["num_mean"] = X_train[num_cols].mean(axis=1)
            X_train["num_std"] = X_train[num_cols].std(axis=1)
            X_train["num_max"] = X_train[num_cols].max(axis=1)
            X_train["num_min"] = X_train[num_cols].min(axis=1)
            X_train["num_q25"] = X_train[num_cols].quantile(0.25, axis=1)
            X_train["num_q75"] = X_train[num_cols].quantile(0.75, axis=1)
            X_train["num_iqr"] = X_train["num_q75"] - X_train["num_q25"]

            X_test["num_mean"] = X_test[num_cols].mean(axis=1)
            X_test["num_std"] = X_test[num_cols].std(axis=1)
            X_test["num_max"] = X_test[num_cols].max(axis=1)
            X_test["num_min"] = X_test[num_cols].min(axis=1)
            X_test["num_q25"] = X_test[num_cols].quantile(0.25, axis=1)
            X_test["num_q75"] = X_test[num_cols].quantile(0.75, axis=1)
            X_test["num_iqr"] = X_test["num_q75"] - X_test["num_q25"]

    # 5. Categorical Encoding (Smoothed OOF Target Encoding m=10 & Contextual Group Aggregations)
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("MISSING").astype(str)
        X_test[c] = X_test[c].fillna("MISSING").astype(str)

        # Contextual group aggregations across top continuous metrics
        top_num_cols = num_cols[:8] if len(num_cols) > 8 else num_cols
        for nc in top_num_cols:
            grp_mean = X_train.groupby(c)[nc].mean().to_dict()
            grp_std = X_train.groupby(c)[nc].std().fillna(0.0).to_dict()
            global_nc_mean = float(X_train[nc].mean())

            mean_tr = X_train[c].map(grp_mean).fillna(global_nc_mean).astype(np.float32)
            mean_te = X_test[c].map(grp_mean).fillna(global_nc_mean).astype(np.float32)
            std_tr = X_train[c].map(grp_std).fillna(0.0).astype(np.float32)
            std_te = X_test[c].map(grp_std).fillna(0.0).astype(np.float32)

            X_train[f"{c}_{nc}_grp_mean"] = mean_tr
            X_test[f"{c}_{nc}_grp_mean"] = mean_te
            X_train[f"{c}_{nc}_grp_std"] = std_tr
            X_test[f"{c}_{nc}_grp_std"] = std_te
            X_train[f"{c}_{nc}_diff_mean"] = (X_train[nc] - mean_tr).astype(np.float32)
            X_test[f"{c}_{nc}_diff_mean"] = (X_test[nc] - mean_te).astype(np.float32)

        freq = X_train[c].value_counts(normalize=True).to_dict()
        X_train[f"{c}_freq"] = X_train[c].map(freq).fillna(0).astype(np.float32)
        X_test[f"{c}_freq"] = X_test[c].map(freq).fillna(0).astype(np.float32)

        # Smoothed Out-of-Fold Target Encoding (m-estimate smoothing m=10)
        m = 10.0
        global_mean = float(y.mean())
        X_train[f"{c}_target_enc"] = np.nan

        for fold in range(n_splits):
            train_mask = train_df["fold"] != fold
            val_mask = train_df["fold"] == fold

            fold_counts = train_df.loc[train_mask, c].value_counts()
            fold_sums = train_df.loc[train_mask].groupby(c)["target"].sum()
            smoothed_te = ((fold_sums + m * global_mean) / (fold_counts + m)).to_dict()

            X_train.loc[val_mask, f"{c}_target_enc"] = (
                X_train.loc[val_mask, c].map(smoothed_te).fillna(global_mean).astype(np.float32)
            )

        full_counts = train_df[c].value_counts()
        full_sums = train_df.groupby(c)["target"].sum()
        smoothed_full_te = ((full_sums + m * global_mean) / (full_counts + m)).to_dict()
        X_test[f"{c}_target_enc"] = (
            X_test[c].map(smoothed_full_te).fillna(global_mean).astype(np.float32)
        )

        cat_mapping = {val: idx for idx, val in enumerate(X_train[c].unique())}
        X_train[c] = X_train[c].map(cat_mapping).fillna(-1).astype(np.int32)
        X_test[c] = X_test[c].map(cat_mapping).fillna(-1).astype(np.int32)

    # 6. Feature Pruning (Filtering zero-variance, constant, and non-informative columns)
    candidate_cols = [c for c in X_train.columns if c in X_test.columns]
    valid_features = []
    for c in candidate_cols:
        if X_train[c].nunique() > 1 and not (X_train[c].isna().all()):
            valid_features.append(c)

    final_features = valid_features
    X_train_final = X_train[final_features].copy()
    X_test_final = X_test[final_features].copy()
    X_train_final["target"] = train_df["target"].values
    X_train_final["fold"] = train_df["fold"].values

    # 7. Model Training & Cross-Validation
    n_samples = len(X_train_final)
    n_test = len(X_test_final)
    y_all = X_train_final["target"].values

    oof_lgb = np.zeros(n_samples, dtype=np.float64)
    test_lgb = np.zeros(n_test, dtype=np.float64)

    oof_xgb = np.zeros(n_samples, dtype=np.float64)
    test_xgb = np.zeros(n_test, dtype=np.float64)

    oof_cb = np.zeros(n_samples, dtype=np.float64)
    test_cb = np.zeros(n_test, dtype=np.float64)

    oof_et = np.zeros(n_samples, dtype=np.float64)
    test_et = np.zeros(n_test, dtype=np.float64)

    oof_nn = np.zeros(n_samples, dtype=np.float64)
    test_nn = np.zeros(n_test, dtype=np.float64)

    lgb_params = get_lgbm_params(seed=42)
    xgb_params = get_xgb_params(seed=42)
    cb_params = get_catboost_params(seed=42)

    use_pca = len(num_cols) >= 5
    n_pca_components = min(5, len(num_cols)) if use_pca else 0
    pca_feature_names = [f"pca_dim_{i}" for i in range(n_pca_components)] if use_pca else []
    fold_feature_names = final_features + pca_feature_names

    for fold in range(n_splits):
        train_mask = X_train_final["fold"] != fold
        val_mask = X_train_final["fold"] == fold

        X_tr_df = X_train_final.loc[train_mask, final_features].copy()
        X_va_df = X_train_final.loc[val_mask, final_features].copy()
        X_te_df = X_test_final[final_features].copy()

        if use_pca:
            scaler_pca = StandardScaler()
            pca = PCA(n_components=n_pca_components, random_state=42)
            tr_num = scaler_pca.fit_transform(X_tr_df[num_cols])
            va_num = scaler_pca.transform(X_va_df[num_cols])
            te_num = scaler_pca.transform(X_te_df[num_cols])
            pca_tr = pca.fit_transform(tr_num)
            pca_va = pca.transform(va_num)
            pca_te = pca.transform(te_num)
            for i in range(n_pca_components):
                X_tr_df[f"pca_dim_{i}"] = pca_tr[:, i]
                X_va_df[f"pca_dim_{i}"] = pca_va[:, i]
                X_te_df[f"pca_dim_{i}"] = pca_te[:, i]

        X_tr = X_tr_df[fold_feature_names].values.astype(np.float32)
        y_tr = X_train_final.loc[train_mask, "target"].values
        X_va = X_va_df[fold_feature_names].values.astype(np.float32)
        y_va = X_train_final.loc[val_mask, "target"].values
        X_te = X_te_df[fold_feature_names].values.astype(np.float32)

        val_idx = np.where(val_mask)[0]

        # A. LightGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=fold_feature_names)
        dval = lgb.Dataset(
            X_va, label=y_va, reference=dtrain, feature_name=fold_feature_names
        )

        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        oof_lgb[val_idx] = lgb_model.predict(
            X_va, num_iteration=lgb_model.best_iteration
        )
        test_lgb += (
            lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / n_splits
        )

        # B. XGBoost
        dm_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=fold_feature_names)
        dm_va = xgb.DMatrix(X_va, label=y_va, feature_names=fold_feature_names)
        dm_te = xgb.DMatrix(X_te, feature_names=fold_feature_names)

        xgb_model = xgb.train(
            xgb_params,
            dm_tr,
            num_boost_round=1200,
            evals=[(dm_va, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        oof_xgb[val_idx] = xgb_model.predict(dm_va)
        test_xgb += xgb_model.predict(dm_te) / n_splits

        # C. CatBoost
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            verbose=False,
        )

        oof_cb[val_idx] = cb_model.predict_proba(X_va)[:, 1]
        test_cb += cb_model.predict_proba(X_te)[:, 1] / n_splits

        # D. ExtraTrees Bagging Classifier
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        oof_et[val_idx] = et_model.predict_proba(X_va)[:, 1]
        test_et += et_model.predict_proba(X_te)[:, 1] / n_splits

        # E. PyTorch TabularResNet
        nn_oof, nn_test = train_pytorch_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            device=device,
            epochs=35,
            batch_size=128,
        )
        oof_nn[val_idx] = nn_oof
        test_nn += nn_test / n_splits

    # 8. Out-of-Fold Ensemble Optimization with Rank Normalization & L2 Regularization
    oof_lgb_rank = rankdata(oof_lgb) / n_samples
    oof_xgb_rank = rankdata(oof_xgb) / n_samples
    oof_cb_rank = rankdata(oof_cb) / n_samples
    oof_et_rank = rankdata(oof_et) / n_samples
    oof_nn_rank = rankdata(oof_nn) / n_samples

    test_lgb_rank = rankdata(test_lgb) / n_test
    test_xgb_rank = rankdata(test_xgb) / n_test
    test_cb_rank = rankdata(test_cb) / n_test
    test_et_rank = rankdata(test_et) / n_test
    test_nn_rank = rankdata(test_nn) / n_test

    oof_matrix = np.column_stack([oof_lgb_rank, oof_xgb_rank, oof_cb_rank, oof_et_rank, oof_nn_rank])
    test_matrix = np.column_stack([test_lgb_rank, test_xgb_rank, test_cb_rank, test_et_rank, test_nn_rank])

    def loss_func(weights):
        w = np.array(weights)
        sum_w = np.sum(w)
        if sum_w == 0:
            return 0.0
        w_norm = w / sum_w
        blend_oof = np.dot(oof_matrix, w_norm)
        l2_reg = 0.001 * np.sum((w_norm - 1.0 / len(w_norm)) ** 2)
        return -roc_auc_score(y_all, blend_oof) + l2_reg

    init_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    bounds = [(0.0, 1.0)] * 5
    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
    )

    opt_weights = res.x if res.success else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    opt_weights = opt_weights / np.sum(opt_weights)

    final_oof_preds = np.dot(oof_matrix, opt_weights)
    final_test_preds = np.dot(test_matrix, opt_weights)
    final_score = roc_auc_score(y_all, final_oof_preds)

    # 9. Generate and Save Submission File
    submission_df = pd.DataFrame(
        {"id": test_ids, target_col: final_test_preds.astype(np.float64)}
    )

    sub_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()