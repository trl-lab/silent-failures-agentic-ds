import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import warnings

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. Reproducibility Setup
# ---------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)


# ---------------------------------------------------------
# 2. Data Processing & Feature Engineering
# ---------------------------------------------------------
def engineer_features():
    input_dir = "./input"
    output_dir = "./working"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Amount in USD"
    if target_col not in train_df.columns:
        possible_targets = [
            c for c in train_df.columns if "amount" in c.lower() or "usd" in c.lower()
        ]
        if possible_targets:
            target_col = possible_targets[0]

    # Preserve / assign deterministic integer ID
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    feature_cols = [c for c in train_df.columns if c not in [target_col, "id"]]

    # Identify date-like columns
    date_cols = []
    for col in feature_cols:
        col_lower = col.lower()
        if (
            "date" in col_lower
            or "year" in col_lower
            or "period" in col_lower
            or "time" in col_lower
        ):
            date_cols.append(col)
        elif train_df[col].dtype == "object":
            sample_val = (
                train_df[col].dropna().astype(str).iloc[0]
                if len(train_df[col].dropna()) > 0
                else ""
            )
            if re.search(r"\d{4}[-/]\d{1,2}", sample_val):
                date_cols.append(col)

    def extract_date_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df_out = df.copy()
        for col in cols:
            try:
                parsed_dt = pd.to_datetime(df_out[col], errors="coerce")
                if parsed_dt.notna().sum() > 0:
                    df_out[f"{col}_year"] = parsed_dt.dt.year.fillna(-1).astype(int)
                    df_out[f"{col}_month"] = parsed_dt.dt.month.fillna(-1).astype(int)
                    df_out[f"{col}_quarter"] = parsed_dt.dt.quarter.fillna(-1).astype(
                        int
                    )
                    df_out[f"{col}_day"] = parsed_dt.dt.day.fillna(-1).astype(int)
                    df_out[f"{col}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(
                        -1
                    ).astype(int)
                    df_out[f"{col}_is_month_end"] = parsed_dt.dt.is_month_end.fillna(
                        False
                    ).astype(int)
                    df_out[f"{col}_is_quarter_end"] = (
                        parsed_dt.dt.is_quarter_end.fillna(False).astype(int)
                    )

                    month_rad = 2 * np.pi * (parsed_dt.dt.month.fillna(1) - 1) / 12.0
                    df_out[f"{col}_month_sin"] = np.sin(month_rad)
                    df_out[f"{col}_month_cos"] = np.cos(month_rad)
            except Exception:
                pass
        return df_out

    train_df = extract_date_features(train_df, date_cols)
    test_df = extract_date_features(test_df, date_cols)

    current_cols = [c for c in train_df.columns if c not in [target_col, "id"]]
    cat_cols = []
    num_cols = []

    for col in current_cols:
        if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
            cat_cols.append(col)
        else:
            num_cols.append(col)

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

    # High-order categorical interactions
    interaction_pairs = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, min(i + 4, len(cat_cols))):
            col_a, col_b = cat_cols[i], cat_cols[j]
            pair_name = f"{col_a}_x_{col_b}"
            train_df[pair_name] = train_df[col_a] + "__" + train_df[col_b]
            test_df[pair_name] = test_df[col_a] + "__" + test_df[col_b]
            interaction_pairs.append(pair_name)

    all_cat_cols = cat_cols + interaction_pairs

    # Frequency encoding
    for col in all_cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0.0).astype(float)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0.0).astype(float)

    # Cross-entity diversity features
    for col_a in cat_cols[:4]:
        for col_b in cat_cols[:4]:
            if col_a != col_b:
                div_map = train_df.groupby(col_a)[col_b].nunique().to_dict()
                train_df[f"{col_a}_nunique_{col_b}"] = (
                    train_df[col_a].map(div_map).fillna(0).astype(float)
                )
                test_df[f"{col_a}_nunique_{col_b}"] = (
                    test_df[col_a].map(div_map).fillna(0).astype(float)
                )

    # Target binning for stratified K-Fold
    n_splits = 5
    y_raw = train_df[target_col].values
    train_df["target_log1p"] = np.log1p(np.maximum(0, y_raw))

    try:
        y_binned = pd.qcut(
            y_raw, q=min(10, len(np.unique(y_raw))), labels=False, duplicates="drop"
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds_iter = skf.split(train_df, y_binned)
    except Exception:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds_iter = kf.split(train_df)

    train_df["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(folds_iter):
        train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold_id

    # Strictly isolated Out-Of-Fold target encoding
    smoothing = 10.0
    global_mean_target = train_df[target_col].mean()
    global_mean_log = train_df["target_log1p"].mean()

    for col in cat_cols:
        train_df[f"{col}_te_mean"] = global_mean_target
        train_df[f"{col}_te_log_mean"] = global_mean_log
        test_df[f"{col}_te_mean"] = global_mean_target
        test_df[f"{col}_te_log_mean"] = global_mean_log

        for fold_id in range(n_splits):
            tr_mask = train_df["fold"] != fold_id
            va_mask = train_df["fold"] == fold_id

            stats = (
                train_df[tr_mask]
                .groupby(col)
                .agg(
                    count=(target_col, "count"),
                    sum_target=(target_col, "sum"),
                    sum_log=("target_log1p", "sum"),
                )
            )

            stats["te_val"] = (stats["sum_target"] + smoothing * global_mean_target) / (
                stats["count"] + smoothing
            )
            stats["te_log_val"] = (stats["sum_log"] + smoothing * global_mean_log) / (
                stats["count"] + smoothing
            )

            mapping_val = stats["te_val"].to_dict()
            mapping_log = stats["te_log_val"].to_dict()

            train_df.loc[va_mask, f"{col}_te_mean"] = (
                train_df.loc[va_mask, col].map(mapping_val).fillna(global_mean_target)
            )
            train_df.loc[va_mask, f"{col}_te_log_mean"] = (
                train_df.loc[va_mask, col].map(mapping_log).fillna(global_mean_log)
            )

        full_stats = train_df.groupby(col).agg(
            count=(target_col, "count"),
            sum_target=(target_col, "sum"),
            sum_log=("target_log1p", "sum"),
        )
        full_stats["te_val"] = (
            full_stats["sum_target"] + smoothing * global_mean_target
        ) / (full_stats["count"] + smoothing)
        full_stats["te_log_val"] = (
            full_stats["sum_log"] + smoothing * global_mean_log
        ) / (full_stats["count"] + smoothing)

        test_df[f"{col}_te_mean"] = (
            test_df[col].map(full_stats["te_val"].to_dict()).fillna(global_mean_target)
        )
        test_df[f"{col}_te_log_mean"] = (
            test_df[col].map(full_stats["te_log_val"].to_dict()).fillna(global_mean_log)
        )

    for col in all_cat_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = pd.Categorical(
            test_df[col], categories=train_df[col].cat.categories
        )

    engineered_num_cols = [
        c
        for c in train_df.columns
        if c not in all_cat_cols and c not in [target_col, "target_log1p", "fold", "id"]
    ]
    for col in engineered_num_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    return train_df, test_df, target_col


# ---------------------------------------------------------
# 3. Model Architecture Design
# ---------------------------------------------------------
class TabularResBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_continuous_features: int,
        cat_cardinalities: Optional[List[int]] = None,
        embedding_dims: Optional[List[int]] = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.num_continuous_features = num_continuous_features
        self.use_embeddings = (
            cat_cardinalities is not None and len(cat_cardinalities) > 0
        )

        total_emb_dim = 0
        if self.use_embeddings:
            self.embeddings = nn.ModuleList()
            if embedding_dims is None:
                embedding_dims = [
                    max(4, min(64, int(round(1.6 * (c**0.56)))))
                    for c in cat_cardinalities
                ]

            for card, emb_d in zip(cat_cardinalities, embedding_dims):
                self.embeddings.append(nn.Embedding(card + 1, emb_d))
                total_emb_dim += emb_d
        else:
            self.embeddings = None

        total_in_dim = num_continuous_features + total_emb_dim
        self.input_layer = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = []
        if self.use_embeddings and x_cat is not None:
            emb_outputs = []
            for i, emb_layer in enumerate(self.embeddings):
                cat_idx = x_cat[:, i].clamp(0, emb_layer.num_embeddings - 1)
                emb_outputs.append(emb_layer(cat_idx))
            cat_embedded = torch.cat(emb_outputs, dim=1)
            features.append(cat_embedded)

        if self.num_continuous_features > 0:
            features.append(x_num)

        x = torch.cat(features, dim=1)
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)

        out = self.head(x)
        return out


class TabularDataset(Dataset):

    def __init__(self, num_data, cat_data, targets=None):
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.cat_data = (
            torch.tensor(cat_data, dtype=torch.long)
            if cat_data is not None and cat_data.shape[1] > 0
            else None
        )
        self.targets = (
            torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            if targets is not None
            else None
        )

    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        num_feat = self.num_data[idx]
        cat_feat = (
            self.cat_data[idx]
            if self.cat_data is not None
            else torch.tensor([], dtype=torch.long)
        )
        if self.targets is not None:
            return num_feat, cat_feat, self.targets[idx]
        return num_feat, cat_feat


# ---------------------------------------------------------
# 4. Training, Validation & Inference Execution
# ---------------------------------------------------------
def main():
    train_df, test_df, target_col = engineer_features()

    ignore_cols = [target_col, "target_log1p", "fold", "id"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    cat_cols = [
        c
        for c in feature_cols
        if train_df[c].dtype.name == "category" or train_df[c].dtype == "object"
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    cat_mappings = {}
    cat_cardinalities = []
    for c in cat_cols:
        unique_cats = (
            pd.concat([train_df[c].astype(str), test_df[c].astype(str)])
            .unique()
            .tolist()
        )
        mapping = {val: idx for idx, val in enumerate(unique_cats)}
        cat_mappings[c] = mapping
        cat_cardinalities.append(len(unique_cats))

    n_folds = int(train_df["fold"].nunique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train = len(train_df)
    n_test = len(test_df)

    oof_preds_lgb1 = np.zeros(n_train)
    test_preds_lgb1 = np.zeros(n_test)

    oof_preds_lgb2 = np.zeros(n_train)
    test_preds_lgb2 = np.zeros(n_test)

    oof_preds_nn = np.zeros(n_train)
    test_preds_nn = np.zeros(n_test)

    y_train_all = train_df[target_col].values

    for fold_idx in range(n_folds):
        tr_mask = train_df["fold"] != fold_idx
        va_mask = train_df["fold"] == fold_idx

        X_train_df = train_df[tr_mask]
        y_train = train_df.loc[tr_mask, target_col].values

        X_val_df = train_df[va_mask]
        y_val = train_df.loc[va_mask, target_col].values

        # Model 1: LightGBM GBDT Regressor (Conservative Depth)
        lgb_train_data = lgb.Dataset(
            X_train_df[feature_cols],
            label=y_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        lgb_val_data = lgb.Dataset(
            X_val_df[feature_cols],
            label=y_val,
            reference=lgb_train_data,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )

        params1 = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 15,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold_idx,
            "n_jobs": -1,
            "verbose": -1,
        }

        model_lgb1 = lgb.train(
            params1,
            lgb_train_data,
            num_boost_round=1500,
            valid_sets=[lgb_val_data],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_preds_lgb1[va_mask] = np.maximum(
            0,
            model_lgb1.predict(
                X_val_df[feature_cols], num_iteration=model_lgb1.best_iteration
            ),
        )
        test_preds_lgb1 += (
            np.maximum(
                0,
                model_lgb1.predict(
                    test_df[feature_cols], num_iteration=model_lgb1.best_iteration
                ),
            )
            / n_folds
        )

        # Model 2: LightGBM Deep Regularized Regressor
        params2 = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
            "random_state": 100 + fold_idx,
            "n_jobs": -1,
            "verbose": -1,
        }

        model_lgb2 = lgb.train(
            params2,
            lgb_train_data,
            num_boost_round=1500,
            valid_sets=[lgb_val_data],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_preds_lgb2[va_mask] = np.maximum(
            0,
            model_lgb2.predict(
                X_val_df[feature_cols], num_iteration=model_lgb2.best_iteration
            ),
        )
        test_preds_lgb2 += (
            np.maximum(
                0,
                model_lgb2.predict(
                    test_df[feature_cols], num_iteration=model_lgb2.best_iteration
                ),
            )
            / n_folds
        )

        # Model 3: Tabular Residual Neural Network
        scaler = StandardScaler()
        tr_num = scaler.fit_transform(X_train_df[num_cols].values)
        va_num = scaler.transform(X_val_df[num_cols].values)
        te_num = scaler.transform(test_df[num_cols].values)

        tr_cat = np.zeros((len(X_train_df), len(cat_cols)), dtype=np.int64)
        va_cat = np.zeros((len(X_val_df), len(cat_cols)), dtype=np.int64)
        te_cat = np.zeros((len(test_df), len(cat_cols)), dtype=np.int64)

        for i, c in enumerate(cat_cols):
            mapping = cat_mappings[c]
            tr_cat[:, i] = X_train_df[c].astype(str).map(mapping).fillna(0).values
            va_cat[:, i] = X_val_df[c].astype(str).map(mapping).fillna(0).values
            te_cat[:, i] = test_df[c].astype(str).map(mapping).fillna(0).values

        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-6
        y_train_scaled = (y_train - y_mean) / y_std

        train_ds = TabularDataset(tr_num, tr_cat, y_train_scaled)
        val_ds = TabularDataset(va_num, va_cat, None)
        test_ds = TabularDataset(te_num, te_cat, None)

        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True, drop_last=False, num_workers=0
        )
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

        nn_model = TabularResNet(
            num_continuous_features=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            hidden_dim=128,
            num_blocks=2,
            dropout_rate=0.15,
        ).to(device)

        optimizer = AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_weights = None

        for epoch in range(40):
            nn_model.train()
            for batch_num, batch_cat, batch_y in train_loader:
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device) if len(cat_cols) > 0 else None
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out = nn_model(batch_num, batch_cat)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 2.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for b_num, b_cat in val_loader:
                    b_num = b_num.to(device)
                    b_cat = b_cat.to(device) if len(cat_cols) > 0 else None
                    preds = nn_model(b_num, b_cat).cpu().numpy().flatten()
                    val_preds_list.append(preds)

            val_preds_epoch = np.maximum(
                0, np.concatenate(val_preds_list) * y_std + y_mean
            )
            val_rmse_epoch = np.sqrt(mean_squared_error(y_val, val_preds_epoch))

            if val_rmse_epoch < best_val_rmse:
                best_val_rmse = val_rmse_epoch
                best_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        nn_model.load_state_dict(best_weights)
        nn_model.eval()

        val_preds_list = []
        with torch.no_grad():
            for b_num, b_cat in val_loader:
                b_num = b_num.to(device)
                b_cat = b_cat.to(device) if len(cat_cols) > 0 else None
                preds = nn_model(b_num, b_cat).cpu().numpy().flatten()
                val_preds_list.append(preds)
        oof_preds_nn[va_mask] = np.maximum(
            0, np.concatenate(val_preds_list) * y_std + y_mean
        )

        test_preds_list = []
        with torch.no_grad():
            for b_num, b_cat in test_loader:
                b_num = b_num.to(device)
                b_cat = b_cat.to(device) if len(cat_cols) > 0 else None
                preds = nn_model(b_num, b_cat).cpu().numpy().flatten()
                test_preds_list.append(preds)
        test_preds_nn += (
            np.maximum(0, np.concatenate(test_preds_list) * y_std + y_mean) / n_folds
        )

    # ----------------- Optimal Convex Ensemble -----------------
    oof_matrix = np.column_stack([oof_preds_lgb1, oof_preds_lgb2, oof_preds_nn])
    test_matrix = np.column_stack([test_preds_lgb1, test_preds_lgb2, test_preds_nn])

    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.dot(oof_matrix, w)
        return np.sqrt(mean_squared_error(y_train_all, pred))

    init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    bounds = [(0, 1), (0, 1), (0, 1)]
    opt_res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    )

    opt_w = opt_res.x / np.sum(opt_res.x)
    oof_final = np.maximum(0, np.dot(oof_matrix, opt_w))
    final_test_pred = np.maximum(0, np.dot(test_matrix, opt_w))

    final_score = np.sqrt(mean_squared_error(y_train_all, oof_final))

    # ----------------- Save Submission -----------------
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")

    submission_df = pd.DataFrame(
        {"id": test_df["id"].values.astype(int), target_col: final_test_pred}
    )
    submission_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()