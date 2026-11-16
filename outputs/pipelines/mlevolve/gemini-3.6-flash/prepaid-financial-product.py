import copy
import os
import re
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seed for complete reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. Preprocessing & Feature Engineering
# ==============================================================================
def process_features():
    os.makedirs("./working", exist_ok=True)

    print("Loading raw dataset from ./input...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "current_status"
    id_col = "id" if "id" in train_df.columns else None

    # Extract target and binary format conversion
    y_train = train_df[target_col].copy()
    if y_train.dtype == object or y_train.dtype == bool:
        unique_vals = y_train.dropna().unique()
        mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
        y_train = y_train.map(mapping).astype(int)
    else:
        y_train = y_train.astype(int)

    train_df = train_df.drop(columns=[target_col])

    # Store IDs
    train_ids = train_df[id_col].values if id_col else np.arange(len(train_df))
    test_ids = test_df[id_col].values if id_col else np.arange(len(test_df))

    if id_col and id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])
        test_df = test_df.drop(columns=[id_col])

    n_train = len(train_df)

    # Set up 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        folds[val_idx] = fold

    X_train_list = []
    X_test_list = []

    # Datetime Columns
    datetime_cols = []
    for col in train_df.columns:
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            datetime_cols.append(col)
        elif train_df[col].dtype == object:
            sample = train_df[col].dropna().astype(str).head(100)
            if (
                sample.str.match(r"^\d{4}-\d{2}-\d{2}").all()
                or sample.str.match(r"^\d{2}/\d{2}/\d{4}").all()
            ):
                train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
                test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
                datetime_cols.append(col)

    for col in datetime_cols:
        for df, container in [(train_df, X_train_list), (test_df, X_test_list)]:
            dt_feat = pd.DataFrame(index=df.index)
            dt = df[col]
            dt_feat[f"{col}_year"] = dt.dt.year
            dt_feat[f"{col}_month"] = dt.dt.month
            dt_feat[f"{col}_day"] = dt.dt.day
            dt_feat[f"{col}_dayofweek"] = dt.dt.dayofweek
            dt_feat[f"{col}_quarter"] = dt.dt.quarter
            dt_feat[f"{col}_is_month_end"] = dt.dt.is_month_end.astype(float)

            dt_feat[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
            dt_feat[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
            dt_feat[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
            dt_feat[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)

            container.append(dt_feat)

    non_dt_cols = [c for c in train_df.columns if c not in datetime_cols]
    numeric_cols = []
    categorical_cols = []
    text_cols = []

    for col in non_dt_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            avg_len = (
                train_df[col].dropna().astype(str).str.len().mean()
                if not train_df[col].dropna().empty
                else 0
            )
            if avg_len > 50:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    # Numerical Features
    if numeric_cols:
        num_train = train_df[numeric_cols].copy()
        num_test = test_df[numeric_cols].copy()

        for col in numeric_cols:
            num_train[f"{col}_isna"] = num_train[col].isna().astype(float)
            num_test[f"{col}_isna"] = num_test[col].isna().astype(float)

            min_val = num_train[col].min()
            if min_val >= 0 and (num_train[col].skew() > 1.5):
                num_train[f"{col}_log1p"] = np.log1p(num_train[col])
                num_test[f"{col}_log1p"] = np.log1p(num_test[col])

        X_train_list.append(num_train)
        X_test_list.append(num_test)

    # Categorical Features: Frequency & OOF Target Encodings
    for col in categorical_cols:
        cat_train = train_df[col].astype(str).fillna("missing")
        cat_test = test_df[col].astype(str).fillna("missing")

        freq = cat_train.value_counts(normalize=True).to_dict()
        freq_train = cat_train.map(freq).fillna(0.0).rename(f"{col}_freq")
        freq_test = cat_test.map(freq).fillna(0.0).rename(f"{col}_freq")

        oof_target_enc = np.zeros(n_train)
        test_target_enc = np.zeros(len(test_df))
        global_mean = y_train.mean()

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
            tr_cat = cat_train.iloc[train_idx]
            tr_y = y_train.iloc[train_idx]
            val_cat = cat_train.iloc[val_idx]

            counts = tr_cat.value_counts()
            means = tr_y.groupby(tr_cat).mean()
            smooth = (counts * means + 10 * global_mean) / (counts + 10)
            smooth_dict = smooth.to_dict()

            oof_target_enc[val_idx] = val_cat.map(smooth_dict).fillna(global_mean)
            test_target_enc += cat_test.map(smooth_dict).fillna(global_mean) / 5.0

        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ord_train = oe.fit_transform(cat_train.values.reshape(-1, 1)).ravel()
        ord_test = oe.transform(cat_test.values.reshape(-1, 1)).ravel()

        cat_df_tr = pd.DataFrame(
            {
                f"{col}_freq": freq_train,
                f"{col}_oof_te": oof_target_enc,
                f"{col}_ordinal": ord_train,
            },
            index=train_df.index,
        )
        cat_df_te = pd.DataFrame(
            {
                f"{col}_freq": freq_test,
                f"{col}_oof_te": test_target_enc,
                f"{col}_ordinal": ord_test,
            },
            index=test_df.index,
        )

        X_train_list.append(cat_df_tr)
        X_test_list.append(cat_df_te)

    # Pairwise Numerical Differences and Ratios
    if len(numeric_cols) >= 2:
        num_pair_tr = pd.DataFrame(index=train_df.index)
        num_pair_te = pd.DataFrame(index=test_df.index)
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                num_pair_tr[f"{col1}_minus_{col2}"] = train_df[col1] - train_df[col2]
                num_pair_te[f"{col1}_minus_{col2}"] = test_df[col1] - test_df[col2]
                num_pair_tr[f"{col1}_ratio_{col2}"] = train_df[col1] / (train_df[col2].abs() + 1e-5)
                num_pair_te[f"{col1}_ratio_{col2}"] = test_df[col1] / (test_df[col2].abs() + 1e-5)
        X_train_list.append(num_pair_tr)
        X_test_list.append(num_pair_te)

    # Group Aggregations Across All Categoricals & Numericals
    if categorical_cols and numeric_cols:
        for c_col in categorical_cols:
            for n_col in numeric_cols:
                grp_stats = (
                    train_df.groupby(c_col, observed=False)[n_col]
                    .agg(["mean", "std", "min", "max", "median"])
                    .reset_index()
                )
                grp_stats.columns = [
                    c_col,
                    f"{c_col}_{n_col}_grp_mean",
                    f"{c_col}_{n_col}_grp_std",
                    f"{c_col}_{n_col}_grp_min",
                    f"{c_col}_{n_col}_grp_max",
                    f"{c_col}_{n_col}_grp_median",
                ]

                tr_merged = pd.merge(
                    train_df[[c_col]], grp_stats, on=c_col, how="left"
                ).drop(columns=[c_col])
                te_merged = pd.merge(
                    test_df[[c_col]], grp_stats, on=c_col, how="left"
                ).drop(columns=[c_col])

                tr_merged.index = train_df.index
                te_merged.index = test_df.index

                tr_mean = tr_merged[f"{c_col}_{n_col}_grp_mean"]
                tr_std = tr_merged[f"{c_col}_{n_col}_grp_std"].fillna(0)
                te_mean = te_merged[f"{c_col}_{n_col}_grp_mean"]
                te_std = te_merged[f"{c_col}_{n_col}_grp_std"].fillna(0)

                tr_merged[f"{c_col}_{n_col}_grp_zscore"] = (
                    train_df[n_col] - tr_mean
                ) / (tr_std + 1e-5)
                tr_merged[f"{c_col}_{n_col}_grp_ratio"] = train_df[n_col] / (
                    tr_mean + 1e-5
                )

                te_merged[f"{c_col}_{n_col}_grp_zscore"] = (
                    test_df[n_col] - te_mean
                ) / (te_std + 1e-5)
                te_merged[f"{c_col}_{n_col}_grp_ratio"] = test_df[n_col] / (
                    te_mean + 1e-5
                )

                X_train_list.append(tr_merged)
                X_test_list.append(te_merged)

    # High-order Smooth Target Encodings for Categorical Pairs
    if len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                c1, c2 = categorical_cols[i], categorical_cols[j]
                cat_comb_train = (
                    train_df[c1].astype(str).fillna("missing")
                    + "_"
                    + train_df[c2].astype(str).fillna("missing")
                )
                cat_comb_test = (
                    test_df[c1].astype(str).fillna("missing")
                    + "_"
                    + test_df[c2].astype(str).fillna("missing")
                )

                oof_target_enc = np.zeros(n_train)
                test_target_enc = np.zeros(len(test_df))
                global_mean = y_train.mean()

                for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
                    tr_cat = cat_comb_train.iloc[train_idx]
                    tr_y = y_train.iloc[train_idx]
                    val_cat = cat_comb_train.iloc[val_idx]

                    counts = tr_cat.value_counts()
                    means = tr_y.groupby(tr_cat).mean()
                    smooth = (counts * means + 10 * global_mean) / (counts + 10)
                    smooth_dict = smooth.to_dict()

                    oof_target_enc[val_idx] = val_cat.map(smooth_dict).fillna(global_mean)
                    test_target_enc += cat_comb_test.map(smooth_dict).fillna(global_mean) / 5.0

                comb_df_tr = pd.DataFrame(
                    {f"{c1}_{c2}_pair_te": oof_target_enc},
                    index=train_df.index,
                )
                comb_df_te = pd.DataFrame(
                    {f"{c1}_{c2}_pair_te": test_target_enc},
                    index=test_df.index,
                )

                X_train_list.append(comb_df_tr)
                X_test_list.append(comb_df_te)

    # Text Columns
    for col in text_cols:
        txt_train = train_df[col].fillna("").astype(str)
        txt_test = test_df[col].fillna("").astype(str)

        tr_txt_stats = pd.DataFrame(
            {
                f"{col}_char_len": txt_train.str.len(),
                f"{col}_word_count": txt_train.str.split().str.len(),
                f"{col}_num_digits": txt_train.apply(
                    lambda s: sum(c.isdigit() for c in s)
                ),
                f"{col}_uppercase_ratio": txt_train.apply(
                    lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1e-5)
                ),
            },
            index=train_df.index,
        )

        te_txt_stats = pd.DataFrame(
            {
                f"{col}_char_len": txt_test.str.len(),
                f"{col}_word_count": txt_test.str.split().str.len(),
                f"{col}_num_digits": txt_test.apply(
                    lambda s: sum(c.isdigit() for c in s)
                ),
                f"{col}_uppercase_ratio": txt_test.apply(
                    lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1e-5)
                ),
            },
            index=test_df.index,
        )

        X_train_list.append(tr_txt_stats)
        X_test_list.append(te_txt_stats)

        tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english"
        )
        tfidf_tr = tfidf.fit_transform(txt_train)
        tfidf_te = tfidf.transform(txt_test)

        svd_comp = min(16, tfidf_tr.shape[1] - 1)
        if svd_comp > 1:
            svd = TruncatedSVD(n_components=svd_comp, random_state=42)
            svd_tr = svd.fit_transform(tfidf_tr)
            svd_te = svd.transform(tfidf_te)

            svd_tr_df = pd.DataFrame(
                svd_tr,
                columns=[f"{col}_svd_{i}" for i in range(svd_comp)],
                index=train_df.index,
            )
            svd_te_df = pd.DataFrame(
                svd_te,
                columns=[f"{col}_svd_{i}" for i in range(svd_comp)],
                index=test_df.index,
            )

            X_train_list.append(svd_tr_df)
            X_test_list.append(svd_te_df)

    X_train = pd.concat(X_train_list, axis=1)
    X_test = pd.concat(X_test_list, axis=1)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_clean = pd.DataFrame(
        X_train_imp, columns=X_train.columns, index=X_train.index
    )
    X_test_clean = pd.DataFrame(X_test_imp, columns=X_test.columns, index=X_test.index)

    X_train_clean["fold"] = folds
    X_train_clean["target"] = y_train.values
    X_train_clean["id"] = train_ids
    X_test_clean["id"] = test_ids

    X_train_clean.to_parquet("./working/train_processed.parquet", index=False)
    X_test_clean.to_parquet("./working/test_processed.parquet", index=False)
    print("Data processing complete.")


# ==============================================================================
# 2. PyTorch Neural Architecture & Objective
# ==============================================================================
class TabularResBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout_rate: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return residual + out


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(
                    in_features=hidden_dim,
                    hidden_features=hidden_dim * 2,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        logits = torch.mean(
            torch.stack([self.out_proj(drop(h)) for drop in self.dropouts]), dim=0
        )
        return logits.squeeze(-1)


class FocalBCEWithRankLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.5, rank_weight: float = 0.3
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.rank_weight = rank_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = ((1 - p_t) ** self.gamma) * bce_loss
        main_loss = focal_loss.mean()

        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_probs = probs[pos_mask]
            neg_probs = probs[neg_mask]
            max_pairs = min(1000, len(pos_probs) * len(neg_probs))
            pos_sample = pos_probs[
                torch.randint(0, len(pos_probs), (max_pairs,), device=logits.device)
            ]
            neg_sample = neg_probs[
                torch.randint(0, len(neg_probs), (max_pairs,), device=logits.device)
            ]
            rank_loss = F.margin_ranking_loss(
                pos_sample,
                neg_sample,
                target=torch.ones_like(pos_sample),
                margin=0.1,
            )
            return main_loss + self.rank_weight * rank_loss

        return main_loss


def get_lgb_params() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "random_state": 42,
        "n_estimators": 1500,
    }


def get_hgb_params() -> dict:
    return {
        "learning_rate": 0.03,
        "max_iter": 1500,
        "max_depth": 6,
        "l2_regularization": 1.0,
        "random_state": 42,
        "early_stopping": True,
        "n_iter_no_change": 50,
        "scoring": "roc_auc",
    }


def get_et_params() -> dict:
    return {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    }


def get_xgb_params() -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_estimators": 1500,
        "tree_method": "hist",
    }


def get_catboost_params() -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "iterations": 1500,
        "verbose": False,
    }


def create_nn_model_and_optimizer(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.25
    )
    criterion = FocalBCEWithRankLoss(gamma=2.0, rank_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


# ==============================================================================
# 3. Model Training, Optimization & Submission
# ==============================================================================
def main():
    process_features()

    train_df = pd.read_parquet("./working/train_processed.parquet")
    test_df = pd.read_parquet("./working/test_processed.parquet")

    folds = train_df["fold"].values
    y_train = train_df["target"].values
    feature_cols = [c for c in train_df.columns if c not in ["id", "fold", "target"]]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    n_train = len(train_df)
    n_test = len(test_df)
    num_folds = len(np.unique(folds))

    model_names = [
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gb",
        "extra_trees",
        "nn_resnet",
    ]
    oof_preds = {m: np.zeros(n_train) for m in model_names}
    test_preds = {m: np.zeros(n_test) for m in model_names}

    # Train Tree Models across 5 Folds
    for fold in range(num_folds):
        idx_tr = np.where(folds != fold)[0]
        idx_va = np.where(folds == fold)[0]

        X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
        X_va, y_va = X_train[idx_va], y_train[idx_va]

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**get_lgb_params())
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_preds["lightgbm"][idx_va] = lgb_model.predict_proba(X_va)[:, 1]
        test_preds["lightgbm"] += lgb_model.predict_proba(X_test)[:, 1] / num_folds

        # XGBoost
        xgb_params = get_xgb_params()
        xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds["xgboost"][idx_va] = xgb_model.predict_proba(X_va)[:, 1]
        test_preds["xgboost"] += xgb_model.predict_proba(X_test)[:, 1] / num_folds

        # CatBoost
        cb_params = get_catboost_params()
        cb_model = cb.CatBoostClassifier(**cb_params, early_stopping_rounds=50)
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds["catboost"][idx_va] = cb_model.predict_proba(X_va)[:, 1]
        test_preds["catboost"] += cb_model.predict_proba(X_test)[:, 1] / num_folds

        # HistGradientBoosting
        hgb_model = HistGradientBoostingClassifier(**get_hgb_params())
        hgb_model.fit(X_tr, y_tr)
        oof_preds["hist_gb"][idx_va] = hgb_model.predict_proba(X_va)[:, 1]
        test_preds["hist_gb"] += hgb_model.predict_proba(X_test)[:, 1] / num_folds

        # ExtraTrees
        et_model = ExtraTreesClassifier(**get_et_params())
        et_model.fit(X_tr, y_tr)
        oof_preds["extra_trees"][idx_va] = et_model.predict_proba(X_va)[:, 1]
        test_preds["extra_trees"] += et_model.predict_proba(X_test)[:, 1] / num_folds

    # Train TabularResNet across 5 Folds
    for fold in range(num_folds):
        idx_tr = np.where(folds != fold)[0]
        idx_va = np.where(folds == fold)[0]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train[idx_tr])
        X_va_sc = scaler.transform(X_train[idx_va])
        X_te_sc = scaler.transform(X_test)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_train[idx_tr], dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )

        model, criterion, optimizer = create_nn_model_and_optimizer(
            input_dim=len(feature_cols)
        )
        model.to(device)

        best_auc = 0.0
        best_weights = None
        patience, patience_counter = 5, 0

        for epoch in range(25):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
                val_logits = model(va_tensor)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = roc_auc_score(y_train[idx_va], val_probs)

            if val_auc > best_auc:
                best_auc = val_auc
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            oof_preds["nn_resnet"][idx_va] = (
                torch.sigmoid(model(va_tensor)).cpu().numpy()
            )

            te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            test_preds["nn_resnet"] += (
                torch.sigmoid(model(te_tensor)).cpu().numpy() / num_folds
            )

    # Percentile Rank Transformation for Optimal ROC-AUC Blending
    oof_preds_ranked = {}
    test_preds_ranked = {}
    for m in model_names:
        oof_preds_ranked[m] = rankdata(oof_preds[m]) / len(oof_preds[m])
        test_preds_ranked[m] = rankdata(test_preds[m]) / len(test_preds[m])

    # Ensemble Weight Optimization
    def blend_objective(weights):
        w = np.maximum(0, weights)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones_like(w) / len(w)

        blended_oof = np.zeros(n_train)
        for i, m in enumerate(model_names):
            blended_oof += w[i] * oof_preds_ranked[m]
        return -roc_auc_score(y_train, blended_oof)

    init_weights = np.ones(len(model_names)) / len(model_names)
    res = minimize(blend_objective, init_weights, method="Nelder-Mead")

    opt_w = np.maximum(0, res.x)
    if opt_w.sum() > 0:
        opt_w = opt_w / opt_w.sum()
    else:
        opt_w = np.ones_like(opt_w) / len(opt_w)

    final_oof_preds = np.zeros(n_train)
    final_test_preds = np.zeros(n_test)

    for i, m in enumerate(model_names):
        final_oof_preds += opt_w[i] * oof_preds_ranked[m]
        final_test_preds += opt_w[i] * test_preds_ranked[m]

    final_score = roc_auc_score(y_train, final_oof_preds)

    # Save Final Submission
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame(
        {"id": test_df["id"].astype(int), "current_status": final_test_preds}
    )
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
