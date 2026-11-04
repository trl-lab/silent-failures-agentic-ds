import gc
import os
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    """Sets random seed across all libraries for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GatedTabularBlock(nn.Module):
    """Gated Residual Tabular block with LayerNorm, Swish gating, and Dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.linear1(h)
        h1, h2 = h.chunk(2, dim=-1)
        h = self.act(h1) * torch.sigmoid(h2)
        h = self.dropout(h)
        h = self.linear2(h)
        return residual + h


class GatedResidualTabularNet(nn.Module):
    """Deep Gated Residual Tabular Neural Network with LayerNorm and Swish activations."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                GatedTabularBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def train_tabular_resnet(
    X_tr_norm: np.ndarray,
    y_tr: np.ndarray,
    X_va_norm: np.ndarray,
    y_va: np.ndarray,
    X_te_norm: np.ndarray,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    """Trains a TabularResNet model on fold data with early stopping and cosine scheduling."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_features = X_tr_norm.shape[1]
    model = GatedResidualTabularNet(
        in_features=in_features,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_dataset = TensorDataset(
        torch.tensor(X_tr_norm, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_norm, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    test_tensor = torch.tensor(X_te_norm, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_losses.append(loss.item() * len(y_b))

        val_rmse = np.sqrt(np.sum(val_losses) / len(y_va))
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        val_preds_list = []
        for x_b, _ in val_loader:
            x_b = x_b.to(device)
            val_preds_list.append(model(x_b).cpu().numpy())
        val_preds = np.concatenate(val_preds_list)
        test_preds = model(test_tensor).cpu().numpy()

    return val_preds, test_preds


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def process_and_engineer_features(input_dir: str = "./input"):
    """Loads raw data, performs feature engineering, and constructs cross-validation folds."""
    print("=== Step 1: Loading Data and Feature Engineering ===")

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Cash Disbursements (US$)"
    if target_col not in train_df.columns:
        for col in train_df.columns:
            if "disbursement" in col.lower():
                target_col = col
                break

    print(
        f"Train shape: {train_df.shape} | Test shape: {test_df.shape} | Target: {target_col}"
    )

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    train_df["is_train"] = 1
    test_df["is_train"] = 0
    test_df[target_col] = np.nan

    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_df, test_df
    gc.collect()

    feature_cols = [
        c for c in combined_df.columns if c not in [target_col, "id", "is_train"]
    ]

    date_cols, cat_cols, num_cols = [], [], []
    for col in feature_cols:
        col_dtype = combined_df[col].dtype
        col_sample = combined_df[col].dropna().head(100)

        is_date = False
        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            is_date = True
        elif col_dtype == "object":
            try:
                converted = pd.to_datetime(col_sample, errors="coerce", format="mixed")
                if converted.notnull().mean() > 0.8:
                    is_date = True
            except Exception:
                is_date = False

        if is_date:
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(col_dtype):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # 1. Date Feature Engineering
    for col in date_cols:
        combined_df[col] = pd.to_datetime(
            combined_df[col], errors="coerce", format="mixed"
        )
        col_prefix = f"dt_{col}"
        combined_df[f"{col_prefix}_year"] = combined_df[col].dt.year
        combined_df[f"{col_prefix}_month"] = combined_df[col].dt.month
        combined_df[f"{col_prefix}_quarter"] = combined_df[col].dt.quarter
        combined_df[f"{col_prefix}_day"] = combined_df[col].dt.day
        combined_df[f"{col_prefix}_dayofweek"] = combined_df[col].dt.dayofweek
        combined_df[f"{col_prefix}_dayofyear"] = combined_df[col].dt.dayofyear
        combined_df[f"{col_prefix}_is_month_end"] = combined_df[
            col
        ].dt.is_month_end.astype(float)
        combined_df[f"{col_prefix}_is_year_end"] = combined_df[
            col
        ].dt.is_year_end.astype(float)
        combined_df[f"{col_prefix}_timestamp"] = (
            combined_df[col].astype("int64") // 10**9
        )
        # Trigonometric cyclical features
        combined_df[f"{col_prefix}_sin_month"] = np.sin(
            2 * np.pi * combined_df[f"{col_prefix}_month"] / 12.0
        ).astype(np.float32)
        combined_df[f"{col_prefix}_cos_month"] = np.cos(
            2 * np.pi * combined_df[f"{col_prefix}_month"] / 12.0
        ).astype(np.float32)
        combined_df[f"{col_prefix}_sin_dow"] = np.sin(
            2 * np.pi * combined_df[f"{col_prefix}_dayofweek"] / 7.0
        ).astype(np.float32)
        combined_df[f"{col_prefix}_cos_dow"] = np.cos(
            2 * np.pi * combined_df[f"{col_prefix}_dayofweek"] / 7.0
        ).astype(np.float32)
        combined_df[f"{col_prefix}_sin_doy"] = np.sin(
            2 * np.pi * combined_df[f"{col_prefix}_dayofyear"] / 365.25
        ).astype(np.float32)
        combined_df[f"{col_prefix}_cos_doy"] = np.cos(
            2 * np.pi * combined_df[f"{col_prefix}_dayofyear"] / 365.25
        ).astype(np.float32)
        combined_df.loc[combined_df[col].isna(), f"{col_prefix}_timestamp"] = np.nan

    date_diff_cols = []
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            diff_col_name = f"diff_days_{col1}_{col2}"
            combined_df[diff_col_name] = (
                combined_df[col1] - combined_df[col2]
            ).dt.total_seconds() / (24 * 3600)
            date_diff_cols.append(diff_col_name)

    # Longitudinal lifecycle features per categorical entity before dropping raw dates
    top_temporal_cats = [c for c in cat_cols if combined_df[c].nunique() < 5000][:4]
    for d_col in date_cols:
        for c_col in top_temporal_cats:
            min_date = combined_df.groupby(c_col)[d_col].transform("min")
            max_date = combined_df.groupby(c_col)[d_col].transform("max")
            combined_df[f"days_since_first_{c_col}_{d_col}"] = (
                (combined_df[d_col] - min_date).dt.total_seconds() / (24 * 3600)
            ).astype(np.float32)
            combined_df[f"days_until_last_{c_col}_{d_col}"] = (
                (max_date - combined_df[d_col]).dt.total_seconds() / (24 * 3600)
            ).astype(np.float32)
            combined_df[f"rank_time_{c_col}_{d_col}"] = (
                combined_df.groupby(c_col)[d_col].rank(pct=True).astype(np.float32)
            )

    combined_df.drop(columns=date_cols, inplace=True)

    # 1.5 Text TF-IDF + TruncatedSVD Semantic Embeddings
    text_cand_cols = [c for c in cat_cols if combined_df[c].nunique() > 5]
    if len(text_cand_cols) > 0:
        combined_df["__concat_text__"] = (
            combined_df[text_cand_cols].astype(str).agg(" ".join, axis=1)
        )
        tfidf = TfidfVectorizer(
            max_features=2500,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=3,
        )
        tfidf_mat = tfidf.fit_transform(combined_df["__concat_text__"])
        n_svd = min(16, tfidf_mat.shape[1] - 1)
        if n_svd > 1:
            svd = TruncatedSVD(n_components=n_svd, random_state=42)
            svd_mat = svd.fit_transform(tfidf_mat)
            for k in range(n_svd):
                combined_df[f"tfidf_svd_concat_{k}"] = svd_mat[:, k].astype(
                    np.float32
                )
        combined_df.drop(columns=["__concat_text__"], inplace=True)

    for c in cat_cols:
        if combined_df[c].nunique() > 30:
            c_tfidf = TfidfVectorizer(
                max_features=400,
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=5,
            )
            c_mat = c_tfidf.fit_transform(combined_df[c].astype(str))
            if c_mat.shape[1] > 4:
                c_svd = TruncatedSVD(n_components=4, random_state=42)
                c_feats = c_svd.fit_transform(c_mat)
                for k in range(4):
                    combined_df[f"svd_char_{c}_{k}"] = c_feats[:, k].astype(
                        np.float32
                    )

    # 2. Categorical Encodings
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype(str).fillna("__MISSING__")
        freq_map = combined_df[col].value_counts(normalize=True).to_dict()
        combined_df[f"freq_{col}"] = combined_df[col].map(freq_map).astype(np.float32)
        le = LabelEncoder()
        combined_df[f"le_{col}"] = le.fit_transform(combined_df[col]).astype(np.int32)

    # 3. Numeric Features & Financial Ratios
    current_num_cols = [c for c in num_cols if c in combined_df.columns]

    for col in current_num_cols:
        if combined_df[col].isna().sum() > 0:
            combined_df[f"isna_{col}"] = combined_df[col].isna().astype(np.float32)
        min_val = combined_df[col].min()
        if min_val >= 0:
            combined_df[f"log1p_{col}"] = np.log1p(
                combined_df[col].clip(lower=0).fillna(0)
            )
        else:
            combined_df[f"signed_log1p_{col}"] = np.sign(combined_df[col]) * np.log1p(
                np.abs(combined_df[col]).fillna(0)
            )

    financial_keywords = [
        "amount",
        "balance",
        "disburs",
        "commit",
        "contribut",
        "receipt",
        "transfers",
        "paid",
        "fund",
        "allocat",
    ]
    fin_cols = [
        c for c in current_num_cols if any(k in c.lower() for k in financial_keywords)
    ]
    if len(fin_cols) >= 2:
        for i in range(len(fin_cols)):
            for j in range(i + 1, min(len(fin_cols), i + 5)):
                c1, c2 = fin_cols[i], fin_cols[j]
                combined_df[f"ratio_{c1}_over_{c2}"] = (
                    combined_df[c1] / (combined_df[c2].abs() + 1e-5)
                ).astype(np.float32)
                combined_df[f"diff_{c1}_minus_{c2}"] = (
                    combined_df[c1] - combined_df[c2]
                ).astype(np.float32)
                combined_df[f"sum_{c1}_plus_{c2}"] = (
                    combined_df[c1] + combined_df[c2]
                ).astype(np.float32)
                combined_df[f"burn_share_{c1}_{c2}"] = (
                    combined_df[c1] / (combined_df[c1].abs() + combined_df[c2].abs() + 1e-5)
                ).astype(np.float32)
                combined_df[f"prod_{c1}_{c2}"] = (
                    np.sign(combined_df[c1] * combined_df[c2])
                    * np.log1p(np.abs(combined_df[c1] * combined_df[c2]))
                ).astype(np.float32)

    # Financial velocity & lifecycle momentum features
    if len(date_diff_cols) > 0 and len(fin_cols) > 0:
        for d_col in date_diff_cols[:4]:
            for f_col in fin_cols[:4]:
                combined_df[f"velocity_{f_col}_per_day_{d_col}"] = (
                    combined_df[f_col] / (combined_df[d_col].abs() + 1.0)
                ).astype(np.float32)

    # 4. Group Aggregations, Rank Statistics, and Distribution Moments
    train_mask = combined_df["is_train"] == 1
    high_importance_cats = [c for c in cat_cols if combined_df[c].nunique() < 5000][:6]
    agg_num_cols = current_num_cols[:6]

    for cat in high_importance_cats:
        for num in agg_num_cols:
            train_group = combined_df[train_mask].groupby(cat)[num]
            group_stats = train_group.agg(
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
            ).reset_index()
            group_stats["iqr"] = group_stats["q75"] - group_stats["q25"]
            group_stats["range"] = group_stats["max"] - group_stats["min"]
            group_stats.columns = [
                cat,
                f"grp_{cat}_{num}_mean",
                f"grp_{cat}_{num}_median",
                f"grp_{cat}_{num}_std",
                f"grp_{cat}_{num}_min",
                f"grp_{cat}_{num}_max",
                f"grp_{cat}_{num}_q25",
                f"grp_{cat}_{num}_q75",
                f"grp_{cat}_{num}_iqr",
                f"grp_{cat}_{num}_range",
            ]
            combined_df = combined_df.merge(group_stats, on=cat, how="left")
            combined_df[f"grp_{cat}_{num}_std"] = combined_df[
                f"grp_{cat}_{num}_std"
            ].fillna(0)
            combined_df[f"diff_from_grp_mean_{cat}_{num}"] = (
                combined_df[num] - combined_df[f"grp_{cat}_{num}_mean"]
            ).astype(np.float32)
            combined_df[f"diff_from_grp_median_{cat}_{num}"] = (
                combined_df[num] - combined_df[f"grp_{cat}_{num}_median"]
            ).astype(np.float32)
            combined_df[f"zscore_grp_{cat}_{num}"] = (
                (combined_df[num] - combined_df[f"grp_{cat}_{num}_mean"])
                / (combined_df[f"grp_{cat}_{num}_std"] + 1e-5)
            ).astype(np.float32)
            combined_df[f"ratio_grp_{cat}_{num}"] = (
                combined_df[num]
                / (combined_df[f"grp_{cat}_{num}_median"].abs() + 1e-5)
            ).astype(np.float32)
            combined_df[f"rank_pct_{cat}_{num}"] = (
                combined_df.groupby(cat)[num].rank(pct=True).astype(np.float32)
            )

    # Hierarchical compound interaction categories & Pointwise Mutual Information (PMI)
    cat_pairs = []
    if len(high_importance_cats) >= 2:
        for i in range(min(len(high_importance_cats), 5)):
            for j in range(i + 1, min(len(high_importance_cats), 5)):
                c1, c2 = high_importance_cats[i], high_importance_cats[j]
                pair_name = f"pair_{c1}_{c2}"
                combined_df[pair_name] = (
                    combined_df[c1].astype(str) + "__" + combined_df[c2].astype(str)
                )
                freq_p = combined_df[pair_name].value_counts(normalize=True).to_dict()
                combined_df[f"freq_{pair_name}"] = combined_df[pair_name].map(
                    freq_p
                ).astype(np.float32)

                p_c1 = combined_df[c1].map(combined_df[c1].value_counts(normalize=True)).astype(np.float32)
                p_c2 = combined_df[c2].map(combined_df[c2].value_counts(normalize=True)).astype(np.float32)
                p_pair = combined_df[f"freq_{pair_name}"]
                combined_df[f"pmi_{c1}_{c2}"] = np.log(
                    (p_pair + 1e-8) / (p_c1 * p_c2 + 1e-8)
                ).astype(np.float32)

                cat_pairs.append(pair_name)

    # Retain raw categorical columns temporarily for out-of-fold target encoding
    target_encoding_cats = high_importance_cats + cat_pairs

    # Split train and test
    train_processed = (
        combined_df[combined_df["is_train"] == 1].copy().reset_index(drop=True)
    )
    test_processed = (
        combined_df[combined_df["is_train"] == 0].copy().reset_index(drop=True)
    )

    train_processed.drop(columns=["is_train"], inplace=True)
    test_processed.drop(columns=["is_train", target_col], inplace=True)

    # 5. Stratified K-Fold Assignment
    n_splits = 5
    train_processed["fold"] = -1
    target_clean = train_processed[target_col].fillna(0)
    try:
        bins = pd.qcut(target_clean, q=10, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_idx, (_, val_idx) in enumerate(skf.split(train_processed, bins)):
            train_processed.loc[val_idx, "fold"] = fold_idx
    except Exception:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_idx, (_, val_idx) in enumerate(kf.split(train_processed)):
            train_processed.loc[val_idx, "fold"] = fold_idx

    # 6. Out-of-fold Multi-Statistic Smoothed Target Encoding
    global_target_mean = float(train_processed[target_col].mean())
    global_target_std = float(train_processed[target_col].std())
    global_target_median = float(train_processed[target_col].median())
    smoothing_weight = 10.0

    for cat_c in target_encoding_cats:
        if cat_c in train_processed.columns:
            for stat_type in ["mean", "std", "median"]:
                te_col_name = f"te_{stat_type}_{cat_c}"
                train_processed[te_col_name] = np.nan
                test_processed[te_col_name] = np.nan

            for f_idx in range(n_splits):
                trn_mask = train_processed["fold"] != f_idx
                val_mask = train_processed["fold"] == f_idx

                grp = train_processed[trn_mask].groupby(cat_c)[target_col]
                cnt = grp.count()
                m = grp.mean()
                s = grp.std().fillna(0)
                med = grp.median()

                smooth_mean = (cnt * m + smoothing_weight * global_target_mean) / (cnt + smoothing_weight)
                smooth_std = (cnt * s + smoothing_weight * global_target_std) / (cnt + smoothing_weight)
                smooth_median = (cnt * med + smoothing_weight * global_target_median) / (cnt + smoothing_weight)

                train_processed.loc[val_mask, f"te_mean_{cat_c}"] = (
                    train_processed.loc[val_mask, cat_c].map(smooth_mean).fillna(global_target_mean).astype(np.float32)
                )
                train_processed.loc[val_mask, f"te_std_{cat_c}"] = (
                    train_processed.loc[val_mask, cat_c].map(smooth_std).fillna(global_target_std).astype(np.float32)
                )
                train_processed.loc[val_mask, f"te_median_{cat_c}"] = (
                    train_processed.loc[val_mask, cat_c].map(smooth_median).fillna(global_target_median).astype(np.float32)
                )

            # Full training stats for test set
            full_grp = train_processed.groupby(cat_c)[target_col]
            full_cnt = full_grp.count()
            full_m = full_grp.mean()
            full_s = full_grp.std().fillna(0)
            full_med = full_grp.median()

            full_smooth_mean = (full_cnt * full_m + smoothing_weight * global_target_mean) / (full_cnt + smoothing_weight)
            full_smooth_std = (full_cnt * full_s + smoothing_weight * global_target_std) / (full_cnt + smoothing_weight)
            full_smooth_median = (full_cnt * full_med + smoothing_weight * global_target_median) / (full_cnt + smoothing_weight)

            test_processed[f"te_mean_{cat_c}"] = (
                test_processed[cat_c].map(full_smooth_mean).fillna(global_target_mean).astype(np.float32)
            )
            test_processed[f"te_std_{cat_c}"] = (
                test_processed[cat_c].map(full_smooth_std).fillna(global_target_std).astype(np.float32)
            )
            test_processed[f"te_median_{cat_c}"] = (
                test_processed[cat_c].map(full_smooth_median).fillna(global_target_median).astype(np.float32)
            )

    # Drop raw string categorical & pair columns after target encoding
    cols_to_drop_now = [c for c in cat_cols + cat_pairs if c in train_processed.columns]
    train_processed.drop(columns=cols_to_drop_now, inplace=True)
    test_processed.drop(columns=cols_to_drop_now, inplace=True)

    feature_columns = [
        c for c in train_processed.columns if c not in [target_col, "id", "fold"]
    ]
    for c in feature_columns:
        train_processed[c] = train_processed[c].replace([np.inf, -np.inf], np.nan)
        test_processed[c] = test_processed[c].replace([np.inf, -np.inf], np.nan)

    # Sanitize feature column names for LightGBM/XGBoost
    seen_names = {}
    clean_cols = []
    for c in feature_columns:
        clean_name = re.sub(r"[\[\]<>{}:,\"\s\$\(\)]+", "_", c)
        if clean_name in seen_names:
            seen_names[clean_name] += 1
            clean_name = f"{clean_name}_{seen_names[clean_name]}"
        else:
            seen_names[clean_name] = 0
        clean_cols.append(clean_name)

    rename_map = dict(zip(feature_columns, clean_cols))
    train_processed.rename(columns=rename_map, inplace=True)
    test_processed.rename(columns=rename_map, inplace=True)

    print(f"Feature engineering completed. Total features: {len(clean_cols)}")
    return train_processed, test_processed, clean_cols, target_col


def main():
    train_df, test_df, feature_cols, target_col = process_and_engineer_features()

    y_train = train_df[target_col].values.astype(np.float64)
    folds = train_df["fold"].values
    n_folds = len(np.unique(folds))

    test_ids = test_df["id"].values
    X_test = test_df[feature_cols]

    oof_lgb = np.zeros(len(train_df), dtype=np.float64)
    oof_lgb_et = np.zeros(len(train_df), dtype=np.float64)
    oof_xgb = np.zeros(len(train_df), dtype=np.float64)
    oof_cb = np.zeros(len(train_df), dtype=np.float64)
    oof_nn = np.zeros(len(train_df), dtype=np.float64)

    test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_lgb_et = np.zeros(len(test_df), dtype=np.float64)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_cb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_nn = np.zeros(len(test_df), dtype=np.float64)

    # -------------------------------------------------------------------------
    # 1. Train LightGBM (Standard GBDT)
    # -------------------------------------------------------------------------
    print("\n--- Training LightGBM GBDT ---")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    for fold in range(n_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = train_df.iloc[trn_idx][feature_cols], y_train[trn_idx]
        X_va, y_va = train_df.iloc[val_idx][feature_cols], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model_lgb = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
        oof_lgb[val_idx] = val_pred
        test_preds_lgb += (
            model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_folds
        )

        fold_rmse = compute_rmse(y_va, val_pred)
        print(f"LGB Fold {fold+1}/{n_folds} RMSE: {fold_rmse:.4f}")

    lgb_oof_rmse = compute_rmse(y_train, oof_lgb)
    print(f"Total LightGBM OOF RMSE: {lgb_oof_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 2. Train LightGBM Extra-Trees Variant
    # -------------------------------------------------------------------------
    print("\n--- Training LightGBM Extra-Trees ---")
    lgb_et_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "extra_trees": True,
        "num_leaves": 95,
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_samples": 25,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    for fold in range(n_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = train_df.iloc[trn_idx][feature_cols], y_train[trn_idx]
        X_va, y_va = train_df.iloc[val_idx][feature_cols], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model_lgb_et = lgb.train(
            lgb_et_params,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model_lgb_et.predict(X_va, num_iteration=model_lgb_et.best_iteration)
        oof_lgb_et[val_idx] = val_pred
        test_preds_lgb_et += (
            model_lgb_et.predict(X_test, num_iteration=model_lgb_et.best_iteration) / n_folds
        )

        fold_rmse = compute_rmse(y_va, val_pred)
        print(f"LGB-ET Fold {fold+1}/{n_folds} RMSE: {fold_rmse:.4f}")

    lgb_et_oof_rmse = compute_rmse(y_train, oof_lgb_et)
    print(f"Total LightGBM Extra-Trees OOF RMSE: {lgb_et_oof_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 3. Train XGBoost Across Folds (With Depth-wise and Node-wise Subsampling)
    # -------------------------------------------------------------------------
    print("\n--- Training XGBoost ---")
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 7,
        "learning_rate": 0.03,
        "min_child_weight": 4.0,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.85,
        "colsample_bynode": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
    }

    dtest = xgb.DMatrix(X_test)
    for fold in range(n_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = train_df.iloc[trn_idx][feature_cols], y_train[trn_idx]
        X_va, y_va = train_df.iloc[val_idx][feature_cols], y_train[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)

        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        val_pred = model_xgb.predict(dval)
        oof_xgb[val_idx] = val_pred
        test_preds_xgb += model_xgb.predict(dtest) / n_folds

        fold_rmse = compute_rmse(y_va, val_pred)
        print(f"XGB Fold {fold+1}/{n_folds} RMSE: {fold_rmse:.4f}")

    xgb_oof_rmse = compute_rmse(y_train, oof_xgb)
    print(f"Total XGBoost OOF RMSE: {xgb_oof_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 4. Train CatBoost Across Folds
    # -------------------------------------------------------------------------
    print("\n--- Training CatBoost ---")
    cb_params = {
        "iterations": 1500,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "subsample": 0.8,
        "border_count": 128,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    for fold in range(n_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = train_df.iloc[trn_idx][feature_cols], y_train[trn_idx]
        X_va, y_va = train_df.iloc[val_idx][feature_cols], y_train[val_idx]

        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        val_pred = model_cb.predict(X_va)
        oof_cb[val_idx] = val_pred
        test_preds_cb += model_cb.predict(X_test) / n_folds

        fold_rmse = compute_rmse(y_va, val_pred)
        print(f"CatBoost Fold {fold+1}/{n_folds} RMSE: {fold_rmse:.4f}")

    cb_oof_rmse = compute_rmse(y_train, oof_cb)
    print(f"Total CatBoost OOF RMSE: {cb_oof_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 5. Train Tabular Residual Neural Network Across Folds
    # -------------------------------------------------------------------------
    print("\n--- Training Tabular Residual Neural Network ---")
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(train_df[feature_cols].fillna(0))
    X_test_imputed = imputer.transform(X_test.fillna(0))

    for fold in range(n_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_train_imputed[trn_idx])
        X_va_norm = scaler.transform(X_train_imputed[val_idx])
        X_te_norm = scaler.transform(X_test_imputed)

        y_tr, y_va = y_train[trn_idx], y_train[val_idx]

        val_pred, test_pred = train_tabular_resnet(
            X_tr_norm=X_tr_norm,
            y_tr=y_tr,
            X_va_norm=X_va_norm,
            y_va=y_va,
            X_te_norm=X_te_norm,
            epochs=35,
            batch_size=512,
            lr=1e-3,
            weight_decay=1e-4,
            seed=42 + fold,
        )

        oof_nn[val_idx] = val_pred
        test_preds_nn += test_pred / n_folds

        fold_rmse = compute_rmse(y_va, val_pred)
        print(f"TabularResNet Fold {fold+1}/{n_folds} RMSE: {fold_rmse:.4f}")

    nn_oof_rmse = compute_rmse(y_train, oof_nn)
    print(f"Total TabularResNet OOF RMSE: {nn_oof_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 6. Multi-Level Regularized Stacking & Weight Optimization
    # -------------------------------------------------------------------------
    oof_matrix = np.column_stack([oof_lgb, oof_lgb_et, oof_xgb, oof_cb, oof_nn])
    test_matrix = np.column_stack([test_preds_lgb, test_preds_lgb_et, test_preds_xgb, test_preds_cb, test_preds_nn])
    n_models = oof_matrix.shape[1]

    def blend_objective(weights):
        pred = oof_matrix @ weights
        l2_reg = 1e-4 * np.sum(weights ** 2)
        return compute_rmse(y_train, pred) + l2_reg

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        blend_objective,
        init_weights,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    weights = opt_result.x if opt_result.success else init_weights
    weights = np.maximum(0, weights)
    weights /= np.sum(weights)

    oof_ensemble = oof_matrix @ weights
    final_test_preds = test_matrix @ weights

    if np.min(y_train) >= 0:
        oof_ensemble = np.clip(oof_ensemble, a_min=0.0, a_max=None)
        final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

    final_score = compute_rmse(y_train, oof_ensemble)

    print(
        f"\nOptimal Ensemble Weights: LGB={weights[0]:.4f}, LGB-ET={weights[1]:.4f}, XGB={weights[2]:.4f}, CB={weights[3]:.4f}, NN={weights[4]:.4f}"
    )
    print(f"Ensemble OOF RMSE: {final_score:.4f}")

    # -------------------------------------------------------------------------
    # 5. Generate and Save Submission File
    # -------------------------------------------------------------------------
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    sub_df = pd.DataFrame(
        {"id": test_ids, "Cash Disbursements (US$)": final_test_preds}
    )

    assert len(sub_df) == len(test_df), "Row count mismatch in submission."
    assert sub_df["Cash Disbursements (US$)"].notnull().all(), "Nulls in submission."
    assert list(sub_df.columns) == [
        "id",
        "Cash Disbursements (US$)",
    ], "Invalid columns."

    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to: {submission_path}")

    # Clean up memory
    del train_df, test_df, X_test
    gc.collect()

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
