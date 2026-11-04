import gc
import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =========================================================================
# 1. Evaluation Metric Utility
# =========================================================================
def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =========================================================================
# 2. Data Processing & Feature Engineering Pipeline
# =========================================================================
def process_and_engineer_features(
    train_path: str = "./input/train.parquet",
    test_path: str = "./input/test.parquet",
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete, leak-free data processing and feature engineering pipeline
    for Industry Payments Entity regression.
    """
    print("Loading raw parquet datasets...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "entityPayment"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training dataset.")

    print(f"Raw Train Shape: {train_df.shape}, Raw Test Shape: {test_df.shape}")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_train = train_df[target_col].copy()

    # Combine for unified non-target structural transformations
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    test_df[target_col] = np.nan

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 1. Identify and extract datetime features
    date_cols = []
    for col in full_df.columns:
        if col in ["is_train", target_col]:
            continue
        if pd.api.types.is_datetime64_any_dtype(full_df[col]):
            date_cols.append(col)
        elif full_df[col].dtype == object or full_df[col].dtype == "string":
            sample_non_null = full_df[col].dropna().head(100)
            if len(sample_non_null) > 0:
                try:
                    converted = pd.to_datetime(sample_non_null, errors="coerce")
                    if converted.notnull().mean() > 0.8:
                        full_df[col] = pd.to_datetime(full_df[col], errors="coerce")
                        date_cols.append(col)
                except Exception:
                    pass

    for col in date_cols:
        full_df[f"{col}_year"] = full_df[col].dt.year
        full_df[f"{col}_month"] = full_df[col].dt.month
        full_df[f"{col}_day"] = full_df[col].dt.day
        full_df[f"{col}_dayofweek"] = full_df[col].dt.dayofweek
        full_df[f"{col}_quarter"] = full_df[col].dt.quarter
        full_df[f"{col}_is_month_end"] = full_df[col].dt.is_month_end.astype(float)

        # Cyclical month and day-of-week encodings
        month_rad = 2 * np.pi * full_df[f"{col}_month"] / 12.0
        full_df[f"{col}_month_sin"] = np.sin(month_rad)
        full_df[f"{col}_month_cos"] = np.cos(month_rad)

        dow_rad = 2 * np.pi * full_df[f"{col}_dayofweek"] / 7.0
        full_df[f"{col}_dow_sin"] = np.sin(dow_rad)
        full_df[f"{col}_dow_cos"] = np.cos(dow_rad)

        full_df.drop(columns=[col], inplace=True)

    # 2. Separate numerical and categorical features
    ignore_cols = ["is_train", target_col]
    feature_cols = [c for c in full_df.columns if c not in ignore_cols]

    categorical_cols = []
    numerical_cols = []

    for col in feature_cols:
        if (
            full_df[col].dtype == object
            or full_df[col].dtype == "string"
            or isinstance(full_df[col].dtype, pd.CategoricalDtype)
        ):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(full_df[col]):
            numerical_cols.append(col)

    # 3. Categorical Cleaning, String Metrics, and TF-IDF SVD Text Representations
    train_mask = full_df["is_train"] == 1
    raw_categorical_cols = list(categorical_cols)

    for col in raw_categorical_cols:
        full_df[col] = (
            full_df[col].fillna("MISSING").astype(str).str.strip().str.lower()
        )
        full_df[f"{col}_len"] = full_df[col].str.len().astype(float)
        full_df[f"{col}_word_count"] = (
            full_df[col].apply(lambda x: len(x.split())).astype(float)
        )

        # TF-IDF with TruncatedSVD (dim=8)
        try:
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=5000,
                token_pattern=r"(?u)\b\w+\b",
            )
            train_text = full_df.loc[train_mask, col].astype(str)
            tfidf_train = tfidf.fit_transform(train_text)
            n_features = tfidf_train.shape[1]
            if n_features >= 3:
                n_svd = min(8, n_features - 1)
                svd = TruncatedSVD(n_components=n_svd, random_state=random_state)
                svd.fit(tfidf_train)
                tfidf_full = tfidf.transform(full_df[col].astype(str))
                svd_feats = svd.transform(tfidf_full)
                for d in range(n_svd):
                    full_df[f"{col}_tfidf_svd_{d}"] = svd_feats[:, d].astype(np.float32)
        except Exception:
            pass

    # 4. Skew-Adjusted Log Transforms and Full Pairwise Numerical Interactions
    raw_numerical_cols = list(numerical_cols)
    for c in raw_numerical_cols:
        min_val = full_df.loc[train_mask, c].min()
        if min_val >= 0:
            full_df[f"{c}_log1p"] = np.log1p(np.maximum(0, full_df[c]))
        else:
            full_df[f"{c}_log1p"] = np.sign(full_df[c]) * np.log1p(np.abs(full_df[c]))

    for i in range(len(raw_numerical_cols)):
        for j in range(i + 1, len(raw_numerical_cols)):
            c1, c2 = raw_numerical_cols[i], raw_numerical_cols[j]
            full_df[f"{c1}_diff_{c2}"] = full_df[c1] - full_df[c2]
            full_df[f"{c1}_ratio_{c2}"] = (full_df[c1] + 1e-5) / (full_df[c2] + 1e-5)
            full_df[f"{c2}_ratio_{c1}"] = (full_df[c2] + 1e-5) / (full_df[c1] + 1e-5)
            full_df[f"{c1}_prod_{c2}"] = full_df[c1] * full_df[c2]

    # 5. Full Pairwise Categorical Interactions across ALL raw categoricals
    if len(raw_categorical_cols) >= 2:
        for i in range(len(raw_categorical_cols)):
            for j in range(i + 1, len(raw_categorical_cols)):
                c1, c2 = raw_categorical_cols[i], raw_categorical_cols[j]
                inter_col = f"{c1}__x__{c2}"
                full_df[inter_col] = full_df[c1] + "___" + full_df[c2]
                categorical_cols.append(inter_col)

    # 6. Frequency Encodings (computed strictly on train partition)
    for col in categorical_cols:
        freq_map = full_df.loc[train_mask, col].value_counts(normalize=True).to_dict()
        count_map = full_df.loc[train_mask, col].value_counts().to_dict()
        full_df[f"{col}_freq"] = full_df[col].map(freq_map).fillna(0.0).astype(float)
        full_df[f"{col}_count"] = full_df[col].map(count_map).fillna(0.0).astype(float)

    # 7. Exhaustive Group Aggregations without column truncation
    if len(raw_numerical_cols) > 0 and len(categorical_cols) > 0:
        for c_cat in categorical_cols:
            for c_num in raw_numerical_cols:
                train_stats = (
                    full_df.loc[train_mask]
                    .groupby(c_cat)[c_num]
                    .agg(
                        mean="mean",
                        std="std",
                        min="min",
                        max="max",
                        median="median",
                        sum="sum",
                        skew="skew",
                    )
                    .reset_index()
                )
                train_stats["skew"] = train_stats["skew"].fillna(0.0)
                train_stats["std"] = train_stats["std"].fillna(0.0)
                train_stats.columns = [
                    c_cat,
                    f"{c_cat}_{c_num}_mean",
                    f"{c_cat}_{c_num}_std",
                    f"{c_cat}_{c_num}_min",
                    f"{c_cat}_{c_num}_max",
                    f"{c_cat}_{c_num}_median",
                    f"{c_cat}_{c_num}_sum",
                    f"{c_cat}_{c_num}_skew",
                ]

                full_df = full_df.merge(train_stats, on=c_cat, how="left")

                mean_col = f"{c_cat}_{c_num}_mean"
                std_col = f"{c_cat}_{c_num}_std"
                full_df[f"{c_num}_diff_{mean_col}"] = full_df[c_num] - full_df[mean_col]
                full_df[f"{c_num}_ratio_{mean_col}"] = (full_df[c_num] + 1e-5) / (
                    full_df[mean_col] + 1e-5
                )
                full_df[f"{c_num}_zscore_{c_cat}"] = (full_df[c_num] - full_df[mean_col]) / (
                    full_df[std_col] + 1e-5
                )

    # 8. Ordinal Label Encoding for categoricals
    cat_code_cols = []
    for col in categorical_cols:
        unique_vals = {val: idx for idx, val in enumerate(full_df[col].unique())}
        code_col = f"{col}_cat_code"
        full_df[code_col] = full_df[col].map(unique_vals).astype(int)
        cat_code_cols.append(code_col)
        full_df.drop(columns=[col], inplace=True)

    # 9. Impute missing numericals with median from train partition
    updated_feature_cols = [
        c for c in full_df.columns if c not in ["is_train", target_col]
    ]
    for col in updated_feature_cols:
        if full_df[col].isnull().sum() > 0:
            train_median = full_df.loc[train_mask, col].median()
            if pd.isnull(train_median):
                train_median = 0.0
            full_df[col] = full_df[col].fillna(train_median)

    # Split back into train and test
    train_processed = (
        full_df[full_df["is_train"] == 1]
        .drop(columns=["is_train"])
        .copy()
        .reset_index(drop=True)
    )
    test_processed = (
        full_df[full_df["is_train"] == 0]
        .drop(columns=["is_train", target_col])
        .copy()
        .reset_index(drop=True)
    )

    # 10. Multi-Statistic Out-Of-Fold Target Encoding (mean, std, median, min, max with smoothing)
    target_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(len(train_processed), dtype=int)

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_processed, target_bins)):
        folds[val_idx] = fold_idx

    global_mean = float(y_train.mean())
    global_median = float(y_train.median())
    global_min = float(y_train.min())
    global_max = float(y_train.max())
    smooth_weights = [10.0, 50.0]

    for te_col in cat_code_cols:
        for m in smooth_weights:
            te_name_mean = f"{te_col}_te_m{int(m)}_mean"
            train_processed[te_name_mean] = global_mean
            test_processed[te_name_mean] = global_mean

        train_processed[f"{te_col}_te_std"] = 0.0
        train_processed[f"{te_col}_te_median"] = global_median
        train_processed[f"{te_col}_te_min"] = global_min
        train_processed[f"{te_col}_te_max"] = global_max
        test_processed[f"{te_col}_te_std"] = 0.0
        test_processed[f"{te_col}_te_median"] = global_median
        test_processed[f"{te_col}_te_min"] = global_min
        test_processed[f"{te_col}_te_max"] = global_max

    for fold_idx in range(n_splits):
        trn_idx = np.where(folds != fold_idx)[0]
        val_idx = np.where(folds == fold_idx)[0]
        fold_trn_df = train_processed.iloc[trn_idx]
        fold_val_df = train_processed.iloc[val_idx]

        for te_col in cat_code_cols:
            grouped = fold_trn_df.groupby(te_col)
            counts = grouped.size()
            means = grouped[target_col].mean()
            stds = grouped[target_col].std().fillna(0.0)
            medians = grouped[target_col].median()
            mins = grouped[target_col].min()
            maxs = grouped[target_col].max()

            for m in smooth_weights:
                te_name_mean = f"{te_col}_te_m{int(m)}_mean"
                smoothed_mean = (counts * means + m * global_mean) / (counts + m)
                train_processed.loc[val_idx, te_name_mean] = (
                    fold_val_df[te_col].map(smoothed_mean).fillna(global_mean)
                )

            train_processed.loc[val_idx, f"{te_col}_te_std"] = (
                fold_val_df[te_col].map(stds).fillna(0.0)
            )
            train_processed.loc[val_idx, f"{te_col}_te_median"] = (
                fold_val_df[te_col].map(medians).fillna(global_median)
            )
            train_processed.loc[val_idx, f"{te_col}_te_min"] = (
                fold_val_df[te_col].map(mins).fillna(global_min)
            )
            train_processed.loc[val_idx, f"{te_col}_te_max"] = (
                fold_val_df[te_col].map(maxs).fillna(global_max)
            )

    # Full train target encoding for test set
    for te_col in cat_code_cols:
        grouped = train_processed.groupby(te_col)
        counts = grouped.size()
        means = grouped[target_col].mean()
        stds = grouped[target_col].std().fillna(0.0)
        medians = grouped[target_col].median()
        mins = grouped[target_col].min()
        maxs = grouped[target_col].max()

        for m in smooth_weights:
            te_name_mean = f"{te_col}_te_m{int(m)}_mean"
            smoothed_mean = (counts * means + m * global_mean) / (counts + m)
            test_processed[te_name_mean] = (
                test_processed[te_col].map(smoothed_mean).fillna(global_mean)
            )

        test_processed[f"{te_col}_te_std"] = (
            test_processed[te_col].map(stds).fillna(0.0)
        )
        test_processed[f"{te_col}_te_median"] = (
            test_processed[te_col].map(medians).fillna(global_median)
        )
        test_processed[f"{te_col}_te_min"] = (
            test_processed[te_col].map(mins).fillna(global_min)
        )
        test_processed[f"{te_col}_te_max"] = (
            test_processed[te_col].map(maxs).fillna(global_max)
        )

    train_processed["fold"] = folds

    print(
        f"Feature Engineering complete. Processed Train: {train_processed.shape}, Processed Test: {test_processed.shape}"
    )
    return train_processed, test_processed


# =========================================================================
# 3. Neural Architecture & Model Hyperparameter Specifications
# =========================================================================
class TabularLayerNormResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dropout1(self.act1(self.ln1(self.linear1(x))))
        out = self.dropout2(self.act2(self.ln2(self.linear2(out))))
        return out + res


class TabularEmbeddingResNet(nn.Module):
    def __init__(
        self,
        cat_dims: List[int],
        num_continuous_features: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for c_card in cat_dims:
            emb_dim = min(50, max(4, (c_card + 1) // 2))
            self.embeddings.append(nn.Embedding(c_card + 1, emb_dim))
            total_emb_dim += emb_dim

        in_dim = total_emb_dim + num_continuous_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                blocks.append(
                    nn.Sequential(
                        TabularLayerNormResBlock(hidden_dims[i], dropout=dropout),
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.LayerNorm(hidden_dims[i + 1]),
                        nn.SiLU(),
                    )
                )
            else:
                blocks.append(TabularLayerNormResBlock(hidden_dims[i], dropout=dropout))

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            embs.append(emb_layer(x_cat[:, i]))
        if len(embs) > 0:
            x_cat_emb = torch.cat(embs, dim=1)
            x = torch.cat([x_cat_emb, x_num], dim=1)
        else:
            x = x_num
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def get_lgb_params(seed: int = 42) -> Dict[str, Any]:
    """LightGBM hyperparameter specification calibrated for expanded feature space."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "extra_trees": True,
        "learning_rate": 0.02,
        "num_leaves": 47,
        "max_depth": 7,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 8.0,
        "min_child_samples": 20,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    """XGBoost hyperparameter specification calibrated for rich interaction space."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.80,
        "colsample_bytree": 0.65,
        "colsample_bynode": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 8.0,
        "min_child_weight": 5,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    """CatBoost hyperparameter specification optimized with regularized leaf splits."""
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.02,
        "depth": 7,
        "grow_policy": "Lossguide",
        "num_leaves": 47,
        "min_data_in_leaf": 20,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.2,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


def train_pytorch_resnet(
    X_tr_cat: np.ndarray,
    X_tr_num: np.ndarray,
    y_tr: np.ndarray,
    X_val_cat: np.ndarray,
    X_val_num: np.ndarray,
    y_val: np.ndarray,
    X_te_cat: np.ndarray,
    X_te_num: np.ndarray,
    cat_dims: List[int],
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trains TabularEmbeddingResNet using Quantile Normalization and Cosine Annealing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, max(10, len(X_tr_num))),
        random_state=seed,
        subsample=200000,
    )
    X_tr_num_scaled = np.nan_to_num(qt.fit_transform(X_tr_num), nan=0.0, posinf=0.0, neginf=0.0)
    X_val_num_scaled = np.nan_to_num(qt.transform(X_val_num), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_num_scaled = np.nan_to_num(qt.transform(X_te_num), nan=0.0, posinf=0.0, neginf=0.0)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_cat, dtype=torch.long),
        torch.tensor(X_tr_num_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_cat, dtype=torch.long),
        torch.tensor(X_val_num_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    num_cont = X_tr_num_scaled.shape[1]
    model = TabularEmbeddingResNet(
        cat_dims=cat_dims,
        num_continuous_features=num_cont,
        hidden_dims=[256, 256, 128],
        dropout=0.15,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 35
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_weights = None
    patience = 8
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_cat, batch_num)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_cat, batch_num)
                val_loss += criterion(out, batch_y).item() * len(batch_y)
                val_samples += len(batch_y)

        val_loss = val_loss / val_samples
        val_rmse = np.sqrt(val_loss)

        if val_rmse < best_loss:
            best_loss = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        val_cat_t = torch.tensor(X_val_cat, dtype=torch.long).to(device)
        val_num_t = torch.tensor(X_val_num_scaled, dtype=torch.float32).to(device)
        val_pred = model(val_cat_t, val_num_t).cpu().numpy()

        te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(device)
        te_num_t = torch.tensor(X_te_num_scaled, dtype=torch.float32).to(device)
        te_pred = model(te_cat_t, te_num_t).cpu().numpy()

    return val_pred, te_pred


# =========================================================================
# 4. Cross-Validation Training, Ensemble Blending & Submission
# =========================================================================
def main():
    random_state = 42
    seed_everything(random_state)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    n_splits = 10
    train_df, test_df = process_and_engineer_features(
        train_path="./input/train.parquet",
        test_path="./input/test.parquet",
        n_splits=n_splits,
        random_state=random_state,
    )

    target_col = "entityPayment"
    fold_col = "fold"
    feature_cols = [c for c in train_df.columns if c not in [target_col, fold_col]]
    cat_cols = [c for c in feature_cols if c.endswith("_cat_code")]
    num_cols = [c for c in feature_cols if not c.endswith("_cat_code")]
    cat_dims = [max(int(train_df[c].max()), int(test_df[c].max())) + 1 for c in cat_cols]

    y_train = train_df[target_col].values.astype(np.float64)
    folds = train_df[fold_col].values
    n_train = len(train_df)
    n_test = len(test_df)

    models_to_train = ["lightgbm", "xgboost", "tabular_resnet"]
    if HAS_CATBOOST:
        models_to_train.append("catboost")

    seeds = [42, 2024]
    oof_preds = {m: np.zeros(n_train, dtype=np.float64) for m in models_to_train}
    test_preds = {m: np.zeros(n_test, dtype=np.float64) for m in models_to_train}

    # 2. 10-Fold Multi-Seed Training Loop
    for model_name in models_to_train:
        print(f"\n--- Training Model: {model_name.upper()} ({len(seeds)} Seeds, {n_splits} Folds) ---")

        for seed in seeds:
            seed_oof = np.zeros(n_train, dtype=np.float64)
            seed_test = np.zeros(n_test, dtype=np.float64)

            for fold_idx in range(n_splits):
                trn_mask = folds != fold_idx
                val_mask = folds == fold_idx

                X_tr = train_df.loc[trn_mask, feature_cols].values
                y_tr = y_train[trn_mask]
                X_val = train_df.loc[val_mask, feature_cols].values
                y_val = y_train[val_mask]
                X_te = test_df[feature_cols].values

                if model_name == "lightgbm":
                    trn_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

                    params = get_lgb_params(seed=seed + fold_idx * 10)
                    callbacks = [
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ]

                    gbm = lgb.train(
                        params,
                        trn_data,
                        num_boost_round=2500,
                        valid_sets=[trn_data, val_data],
                        callbacks=callbacks,
                    )

                    val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
                    te_pred = gbm.predict(X_te, num_iteration=gbm.best_iteration)

                elif model_name == "xgboost":
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    dtest = xgb.DMatrix(X_te)

                    params = get_xgb_params(seed=seed + fold_idx * 10)
                    watchlist = [(dtrain, "train"), (dval, "val")]

                    model = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=2500,
                        evals=watchlist,
                        early_stopping_rounds=50,
                        verbose_eval=False,
                    )

                    val_pred = model.predict(dval)
                    te_pred = model.predict(dtest)

                elif model_name == "catboost":
                    cb_params = get_catboost_params(seed=seed + fold_idx * 10)
                    cb_params["iterations"] = 2500

                    cb_model = cb.CatBoostRegressor(**cb_params)
                    cb_model.fit(
                        X_tr,
                        y_tr,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False,
                    )

                    val_pred = cb_model.predict(X_val)
                    te_pred = cb_model.predict(X_te)

                elif model_name == "tabular_resnet":
                    X_tr_cat = train_df.loc[trn_mask, cat_cols].values.astype(np.int64)
                    X_tr_num = train_df.loc[trn_mask, num_cols].values.astype(np.float32)
                    X_val_cat = train_df.loc[val_mask, cat_cols].values.astype(np.int64)
                    X_val_num = train_df.loc[val_mask, num_cols].values.astype(np.float32)
                    X_te_cat = test_df[cat_cols].values.astype(np.int64)
                    X_te_num = test_df[num_cols].values.astype(np.float32)

                    val_pred, te_pred = train_pytorch_resnet(
                        X_tr_cat=X_tr_cat,
                        X_tr_num=X_tr_num,
                        y_tr=y_tr,
                        X_val_cat=X_val_cat,
                        X_val_num=X_val_num,
                        y_val=y_val,
                        X_te_cat=X_te_cat,
                        X_te_num=X_te_num,
                        cat_dims=cat_dims,
                        seed=seed + fold_idx * 10,
                    )

                seed_oof[val_mask] = val_pred
                seed_test += te_pred / n_splits

            oof_preds[model_name] += seed_oof / len(seeds)
            test_preds[model_name] += seed_test / len(seeds)

        overall_rmse = calculate_rmse(y_train, oof_preds[model_name])
        print(f"[{model_name.upper()}] Ensembled OOF RMSE: {overall_rmse:.4f}")

    # 3. Regularized SLSQP Ensemble Blending with Target Clipping
    oof_matrix = np.column_stack([oof_preds[m] for m in models_to_train])
    test_matrix = np.column_stack([test_preds[m] for m in models_to_train])
    n_models = len(models_to_train)

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))

    def loss_func(weights: np.ndarray) -> float:
        pred = np.dot(oof_matrix, weights)
        pred_clipped = np.clip(pred, y_min, y_max)
        reg = 1e-4 * np.sum(weights**2)
        return calculate_rmse(y_train, pred_clipped) + reg

    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        loss_func,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x if opt_res.success else initial_weights

    print("\nOptimal Regularized Blending Weights:")
    for m, w in zip(models_to_train, best_weights):
        print(f"  - {m}: {w:.4f}")

    final_oof_preds = np.clip(np.dot(oof_matrix, best_weights), y_min, y_max)
    final_test_preds = np.clip(np.dot(test_matrix, best_weights), y_min, y_max)

    final_score = calculate_rmse(y_train, final_oof_preds)

    # 4. Generate Submission File
    sub_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_preds), dtype=int),
            "entityPayment": final_test_preds,
        }
    )

    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with shape {sub_df.shape}")

    # 5. Report Final Validation Metric
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
