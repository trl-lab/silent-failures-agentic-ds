import json
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Utilities & Reproducibility
# ==============================================================================
def set_seed(seed: int = 42):
    """Sets deterministic seeds across all random generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates exact Root Mean Squared Error (official evaluation metric)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes dataframe column names for consistent access."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def find_target_column(df: pd.DataFrame) -> str:
    """Identifies the target budget column dynamically."""
    candidates = [
        "Estimated Total Budget ($)",
        "estimated_total_budget",
        "target",
        "budget",
    ]
    for c in df.columns:
        if c in candidates or "budget" in c.lower():
            return c
    raise ValueError("Target column 'Estimated Total Budget ($)' not found in dataset.")


# ==============================================================================
# 2. Feature Engineering Subroutines
# ==============================================================================
def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts rich calendar attributes and active duration spans."""
    df = df.copy()
    date_cols = []

    for col in df.columns:
        if any(
            term in col.lower()
            for term in [
                "date",
                "year",
                "fiscal",
                "time",
                "start",
                "end",
                "approval",
                "close",
            ]
        ):
            if df[col].dtype == "object" or "datetime" in str(df[col].dtype):
                date_cols.append(col)

    parsed_dates = {}
    for col in date_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0.1 * len(df):
                parsed_dates[col] = parsed
                df[f"{col}__year"] = parsed.dt.year.astype("float32")
                df[f"{col}__month"] = parsed.dt.month.astype("float32")
                df[f"{col}__quarter"] = parsed.dt.quarter.astype("float32")
                df[f"{col}__day"] = parsed.dt.day.astype("float32")
                df[f"{col}__dayofweek"] = parsed.dt.dayofweek.astype("float32")
                df[f"{col}__dayofyear"] = parsed.dt.dayofyear.astype("float32")

                df[f"{col}__month_sin"] = np.sin(
                    2 * np.pi * parsed.dt.month / 12.0
                ).astype("float32")
                df[f"{col}__month_cos"] = np.cos(
                    2 * np.pi * parsed.dt.month / 12.0
                ).astype("float32")
                df[f"{col}__dayofweek_sin"] = np.sin(
                    2 * np.pi * parsed.dt.dayofweek / 7.0
                ).astype("float32")
                df[f"{col}__dayofweek_cos"] = np.cos(
                    2 * np.pi * parsed.dt.dayofweek / 7.0
                ).astype("float32")
        except Exception:
            continue

    date_keys = list(parsed_dates.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            c1, c2 = date_keys[i], date_keys[j]
            d1, d2 = parsed_dates[c1], parsed_dates[c2]
            diff_days = (d2 - d1).dt.total_seconds() / (24 * 3600.0)
            df[f"duration_days__{c1}__to__{c2}"] = diff_days.astype("float32")
            df[f"duration_months__{c1}__to__{c2}"] = (diff_days / 30.4375).astype(
                "float32"
            )
            df[f"duration_years__{c1}__to__{c2}"] = (diff_days / 365.25).astype(
                "float32"
            )

    return df


def extract_text_meta_features(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Extracts lexical complexity and token metrics from text fields."""
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            s = df[col].astype(str).fillna("")
            df[f"{col}__char_length"] = s.apply(len).astype("float32")
            df[f"{col}__word_count"] = s.apply(lambda x: len(x.split())).astype(
                "float32"
            )
            df[f"{col}__digit_count"] = s.apply(
                lambda x: sum(c.isdigit() for c in x)
            ).astype("float32")
            df[f"{col}__uppercase_count"] = s.apply(
                lambda x: sum(c.isupper() for c in x)
            ).astype("float32")
            df[f"{col}__uppercase_ratio"] = (
                df[f"{col}__uppercase_count"] / (df[f"{col}__char_length"] + 1.0)
            ).astype("float32")
            df[f"{col}__avg_word_len"] = (
                df[f"{col}__char_length"] / (df[f"{col}__word_count"] + 1.0)
            ).astype("float32")
    return df


def create_interaction_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Generates higher-order pairwise categorical interaction features."""
    df = df.copy()
    num_cats = len(cat_cols)
    for i in range(num_cats):
        for j in range(i + 1, min(num_cats, i + 6)):
            c1, c2 = cat_cols[i], cat_cols[j]
            df[f"{c1}__X__{c2}"] = df[c1].astype(str) + "__" + df[c2].astype(str)
    return df


def fit_text_svd_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: list,
    n_components: int = 15,
):
    """Fits dual word and char TF-IDF vectorizers + SVD strictly on training set."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in text_cols:
        if col in train_df.columns:
            train_texts = train_df[col].astype(str).fillna("missing").tolist()
            test_texts = test_df[col].astype(str).fillna("missing").tolist()

            word_tfidf = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
            )
            w_train = word_tfidf.fit_transform(train_texts)
            w_test = word_tfidf.transform(test_texts)

            char_tfidf = TfidfVectorizer(
                max_features=5000, ngram_range=(3, 5), min_df=2, analyzer="char"
            )
            c_train = char_tfidf.fit_transform(train_texts)
            c_test = char_tfidf.transform(test_texts)

            combined_train = hstack([w_train, c_train]).tocsr()
            combined_test = hstack([w_test, c_test]).tocsr()

            n_comps = min(n_components, max(2, combined_train.shape[1] - 1))
            svd = TruncatedSVD(n_components=n_comps, random_state=42)
            svd_train = svd.fit_transform(combined_train)
            svd_test = svd.transform(combined_test)

            for k in range(n_comps):
                train_df[f"{col}__svd_{k}"] = svd_train[:, k].astype("float32")
                test_df[f"{col}__svd_{k}"] = svd_test[:, k].astype("float32")

    return train_df, test_df


def extract_dense_sentence_embeddings(
    texts: list,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Extracts dense semantic representations using pretrained HuggingFace MiniLM."""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [
                str(t) if pd.notna(t) and str(t).strip() != "" else "missing"
                for t in texts[i : i + batch_size]
            ]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**encoded)
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_emb = (sum_embeddings / sum_mask).cpu().numpy()
                all_embeddings.append(batch_emb)
        return np.vstack(all_embeddings)
    except Exception as e:
        print(f"Dense text embedding fallback notice: {e}")
        return None


def compute_fold_text_ridge(
    train_texts: list,
    val_texts: list,
    test_texts: list,
    y_train_fold: np.ndarray,
    train_dense: np.ndarray = None,
    val_dense: np.ndarray = None,
    test_dense: np.ndarray = None,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits TF-IDF Ridge and Dense Embedding Ridge with inner CV to produce leak-free text features."""
    word_vec = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    char_vec = TfidfVectorizer(
        max_features=8000, ngram_range=(3, 5), min_df=2, analyzer="char"
    )

    w_tr = word_vec.fit_transform(train_texts)
    w_va = word_vec.transform(val_texts)
    w_te = word_vec.transform(test_texts)

    c_tr = char_vec.fit_transform(train_texts)
    c_va = char_vec.transform(val_texts)
    c_te = char_vec.transform(test_texts)

    X_tr_sparse = hstack([w_tr, c_tr]).tocsr()
    X_va_sparse = hstack([w_va, c_va]).tocsr()
    X_te_sparse = hstack([w_te, c_te]).tocsr()

    n_train = len(train_texts)
    tfidf_oof_tr = np.zeros(n_train, dtype="float32")
    dense_oof_tr = np.zeros(n_train, dtype="float32")

    n_inner = 3
    kf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    inner_bins = pd.qcut(y_train_fold, q=n_inner, labels=False, duplicates="drop")

    for in_tr_idx, in_va_idx in kf_inner.split(np.zeros(n_train), inner_bins):
        r_tfidf = Ridge(alpha=alpha, random_state=42)
        r_tfidf.fit(X_tr_sparse[in_tr_idx], y_train_fold[in_tr_idx])
        tfidf_oof_tr[in_va_idx] = r_tfidf.predict(X_tr_sparse[in_va_idx])

        if train_dense is not None:
            r_dense = Ridge(alpha=100.0, random_state=42)
            r_dense.fit(train_dense[in_tr_idx], y_train_fold[in_tr_idx])
            dense_oof_tr[in_va_idx] = r_dense.predict(train_dense[in_va_idx])

    full_tfidf_ridge = Ridge(alpha=alpha, random_state=42)
    full_tfidf_ridge.fit(X_tr_sparse, y_train_fold)
    tfidf_val_pred = full_tfidf_ridge.predict(X_va_sparse).astype("float32")
    tfidf_te_pred = full_tfidf_ridge.predict(X_te_sparse).astype("float32")

    if train_dense is not None and val_dense is not None and test_dense is not None:
        full_dense_ridge = Ridge(alpha=100.0, random_state=42)
        full_dense_ridge.fit(train_dense, y_train_fold)
        dense_val_pred = full_dense_ridge.predict(val_dense).astype("float32")
        dense_te_pred = full_dense_ridge.predict(test_dense).astype("float32")
    else:
        dense_val_pred = np.zeros(len(val_texts), dtype="float32")
        dense_te_pred = np.zeros(len(test_texts), dtype="float32")

    return (
        tfidf_oof_tr,
        tfidf_val_pred,
        tfidf_te_pred,
        dense_oof_tr,
        dense_val_pred,
        dense_te_pred,
    )


def compute_fold_target_encodings(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    target_col: str,
    cat_cols: list,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes leak-free smoothed multi-moment target encodings (mean, std, median) per fold."""
    tr_encoded = pd.DataFrame(index=tr_df.index)
    va_encoded = pd.DataFrame(index=va_df.index)
    te_encoded = pd.DataFrame(index=te_df.index)

    global_mean = float(tr_df[target_col].mean())
    global_std = float(tr_df[target_col].std())
    global_median = float(tr_df[target_col].median())
    if pd.isna(global_std) or global_std == 0:
        global_std = 1.0

    n_inner = 3
    inner_bins = pd.qcut(
        tr_df[target_col].values, q=n_inner, labels=False, duplicates="drop"
    )
    skf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)

    for col in cat_cols:
        if col not in tr_df.columns:
            continue
        tr_mean_col = np.zeros(len(tr_df), dtype="float32")
        tr_std_col = np.zeros(len(tr_df), dtype="float32")
        tr_median_col = np.zeros(len(tr_df), dtype="float32")

        for in_tr_idx, in_va_idx in skf_inner.split(tr_df, inner_bins):
            in_tr = tr_df.iloc[in_tr_idx]
            in_va = tr_df.iloc[in_va_idx]

            stats = in_tr.groupby(col)[target_col].agg(["count", "mean", "std", "median"])
            smooth_mean = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            smooth_std = (
                stats["count"] * stats["std"].fillna(global_std)
                + smoothing * global_std
            ) / (stats["count"] + smoothing)
            smooth_median = (
                stats["count"] * stats["median"] + smoothing * global_median
            ) / (stats["count"] + smoothing)

            tr_mean_col[in_va_idx] = (
                in_va[col].map(smooth_mean).fillna(global_mean).values
            )
            tr_std_col[in_va_idx] = (
                in_va[col].map(smooth_std).fillna(global_std).values
            )
            tr_median_col[in_va_idx] = (
                in_va[col].map(smooth_median).fillna(global_median).values
            )

        tr_encoded[f"{col}__te_mean"] = tr_mean_col
        tr_encoded[f"{col}__te_std"] = tr_std_col
        tr_encoded[f"{col}__te_median"] = tr_median_col

        full_stats = tr_df.groupby(col)[target_col].agg(["count", "mean", "std", "median"])
        full_smooth_mean = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        full_smooth_std = (
            full_stats["count"] * full_stats["std"].fillna(global_std)
            + smoothing * global_std
        ) / (full_stats["count"] + smoothing)
        full_smooth_median = (
            full_stats["count"] * full_stats["median"] + smoothing * global_median
        ) / (full_stats["count"] + smoothing)

        va_encoded[f"{col}__te_mean"] = (
            va_df[col].map(full_smooth_mean).fillna(global_mean).astype("float32")
        )
        va_encoded[f"{col}__te_std"] = (
            va_df[col].map(full_smooth_std).fillna(global_std).astype("float32")
        )
        va_encoded[f"{col}__te_median"] = (
            va_df[col].map(full_smooth_median).fillna(global_median).astype("float32")
        )

        te_encoded[f"{col}__te_mean"] = (
            te_df[col].map(full_smooth_mean).fillna(global_mean).astype("float32")
        )
        te_encoded[f"{col}__te_std"] = (
            te_df[col].map(full_smooth_std).fillna(global_std).astype("float32")
        )
        te_encoded[f"{col}__te_median"] = (
            te_df[col].map(full_smooth_median).fillna(global_median).astype("float32")
        )

    return tr_encoded, va_encoded, te_encoded


# ==============================================================================
# 3. Model Definitions
# ==============================================================================
def create_lgb_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    """Instantiates LightGBM Regressor configured for RMSE minimization."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMRegressor(**params)


def create_lgb_log_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    """Instantiates LightGBM Regressor for log1p-transformed target regression."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMRegressor(**params)


def create_lgb_tweedie_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    """Instantiates LightGBM Regressor with Tweedie loss for positive skewed budget regression."""
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.25,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMRegressor(**params)


def create_ridge_cv_regressor() -> RidgeCV:
    """Instantiates RidgeCV regressor with automated regularization search."""
    return RidgeCV(alphas=np.logspace(-3, 4, 20))


def create_catboost_regressor(random_state: int = 42) -> CatBoostRegressor:
    """Instantiates CatBoost Regressor configured for RMSE minimization."""
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    return CatBoostRegressor(**params)


def create_catboost_log_regressor(random_state: int = 42) -> CatBoostRegressor:
    """Instantiates CatBoost Regressor for log1p-transformed target regression."""
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    return CatBoostRegressor(**params)


def create_et_regressor(random_state: int = 42) -> ExtraTreesRegressor:
    """Instantiates ExtraTrees Regressor for bagging variance reduction."""
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=random_state,
    )


def create_xgb_regressor(random_state: int = 42) -> xgb.XGBRegressor:
    """Instantiates XGBoost Regressor configured for RMSE minimization."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    return xgb.XGBRegressor(**params)


class ResidualBlock(nn.Module):
    """Gated Residual Block with LayerNorm, SiLU non-linearity, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        self.act = nn.SiLU()
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        val, gate = torch.chunk(out, 2, dim=-1)
        gated_out = self.act(val) * self.gate(gate)
        gated_out = self.dropout(gated_out)
        out2 = self.fc2(gated_out)
        out2 = self.ln2(out2)
        return self.act(out2 + residual)


class TabularBudgetResNet(nn.Module):
    """Deep Tabular ResNet Architecture for financial budget regression."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        out = self.head(out)
        return out.squeeze(-1)


def train_neural_resnet(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    X_val_fold: np.ndarray,
    y_val_fold: np.ndarray,
    X_test_scaled: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    patience: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Trains TabularBudgetResNet with AdamW and Cosine Annealing scheduler."""
    in_features = X_train_fold.shape[1]
    model = TabularBudgetResNet(
        in_features=in_features,
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_fold, dtype=torch.float32),
        torch.tensor(y_train_fold, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_fold, dtype=torch.float32),
        torch.tensor(y_val_fold, dtype=torch.float32),
    )
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    best_val_rmse = float("inf")
    best_model_state = None
    best_val_preds = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_pred = model(batch_x).detach().cpu().numpy()
                val_preds_list.append(batch_pred)

        val_preds = np.concatenate(val_preds_list)
        val_preds = np.maximum(0.0, val_preds)
        epoch_val_rmse = calculate_rmse(y_val_fold, val_preds)

        if epoch_val_rmse < best_val_rmse:
            best_val_rmse = epoch_val_rmse
            best_val_preds = val_preds
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(test_tensor).detach().cpu().numpy()
        test_preds = np.maximum(0.0, test_preds)

    return best_val_preds, test_preds


# ==============================================================================
# 4. Main Pipeline Execution
# ==============================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    target_col = find_target_column(train_df)
    y_train = train_df[target_col].values.astype(np.float64)

    # 2. Extract Datetime & Temporal Features
    train_df = extract_datetime_features(train_df)
    test_df = extract_datetime_features(test_df)

    # 3. Classify Text vs Categorical Columns
    candidate_text_cols = []
    candidate_cat_cols = []
    feature_cols = [c for c in train_df.columns if c != target_col]

    for c in feature_cols:
        if c in test_df.columns and train_df[c].dtype == "object":
            avg_char_len = train_df[c].astype(str).str.len().mean()
            unique_count = train_df[c].nunique()
            if avg_char_len > 40 or unique_count > 0.8 * len(train_df):
                candidate_text_cols.append(c)
            else:
                candidate_cat_cols.append(c)

    # 4. Text Meta-Features & SVD Embeddings
    if candidate_text_cols:
        train_df = extract_text_meta_features(train_df, candidate_text_cols)
        test_df = extract_text_meta_features(test_df, candidate_text_cols)
        train_df, test_df = fit_text_svd_features(
            train_df, test_df, candidate_text_cols, n_components=48
        )

    # 5. Categorical Interactions & Encodings
    if candidate_cat_cols:
        train_df = create_interaction_features(train_df, candidate_cat_cols)
        test_df = create_interaction_features(test_df, candidate_cat_cols)

        all_cat_cols = [
            c
            for c in train_df.columns
            if train_df[c].dtype == "object"
            and c not in candidate_text_cols
            and c in test_df.columns
        ]

        # Store raw categorical frames for leak-free out-of-fold target encodings
        cat_df_train = train_df[all_cat_cols + [target_col]].copy()
        cat_df_test = test_df[all_cat_cols].copy()

        # Frequency Encodings
        for col in all_cat_cols:
            freq_map = train_df[col].value_counts(normalize=True).to_dict()
            train_df[f"{col}__freq"] = (
                train_df[col].map(freq_map).fillna(0.0).astype("float32")
            )
            test_df[f"{col}__freq"] = (
                test_df[col].map(freq_map).fillna(0.0).astype("float32")
            )

        # Ordinal Categorical Codes
        for col in all_cat_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = pd.Categorical(
                test_df[col], categories=train_df[col].cat.categories
            )
            train_df[f"{col}__cat_code"] = (
                train_df[col].cat.codes.astype("float32") + 1.0
            )
            test_df[f"{col}__cat_code"] = test_df[col].cat.codes.astype("float32") + 1.0
    else:
        all_cat_cols = []
        cat_df_train = pd.DataFrame(index=train_df.index)
        cat_df_test = pd.DataFrame(index=test_df.index)

    # Prepare concatenated text representation for dense embeddings and fold-safe text Ridge regression
    train_dense = None
    test_dense = None
    if candidate_text_cols:
        train_combined_text = (
            train_df[candidate_text_cols].astype(str).agg(" ".join, axis=1).tolist()
        )
        test_combined_text = (
            test_df[candidate_text_cols].astype(str).agg(" ".join, axis=1).tolist()
        )
        all_texts = train_combined_text + test_combined_text
        dense_embs = extract_dense_sentence_embeddings(all_texts, device=device)
        if dense_embs is not None:
            train_dense = dense_embs[: len(train_df)]
            test_dense = dense_embs[len(train_df) :]
            n_dense_svd = min(48, max(2, dense_embs.shape[1] - 1))
            dense_svd = TruncatedSVD(n_components=n_dense_svd, random_state=42)
            train_dense_svd = dense_svd.fit_transform(train_dense)
            test_dense_svd = dense_svd.transform(test_dense)
            for k in range(train_dense_svd.shape[1]):
                train_df[f"dense_text__svd_{k}"] = train_dense_svd[:, k].astype(
                    "float32"
                )
                test_df[f"dense_text__svd_{k}"] = test_dense_svd[:, k].astype(
                    "float32"
                )
    else:
        train_combined_text = []
        test_combined_text = []

    # 6. Multi-statistic Group Aggregation Features
    duration_cols = [c for c in train_df.columns if "duration" in c]
    freq_cols = [c for c in train_df.columns if c.endswith("__freq")]
    num_agg_targets = duration_cols + freq_cols[:5]

    for group_col in candidate_cat_cols:
        for num_col in num_agg_targets:
            if num_col not in train_df.columns:
                continue
            group_stats = (
                train_df.groupby(group_col)[num_col]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            group_stats.columns = [
                group_col,
                f"{group_col}__{num_col}__mean",
                f"{group_col}__{num_col}__std",
                f"{group_col}__{num_col}__min",
                f"{group_col}__{num_col}__max",
            ]
            train_df = train_df.merge(group_stats, on=group_col, how="left")
            test_df = test_df.merge(group_stats, on=group_col, how="left")
            for stat in ["mean", "std", "min", "max"]:
                stat_col = f"{group_col}__{num_col}__{stat}"
                train_df[stat_col] = train_df[stat_col].fillna(0.0).astype("float32")
                test_df[stat_col] = test_df[stat_col].fillna(0.0).astype("float32")

    # 7. Select & Clean Pure Numerical Features
    exclude_cols = [target_col] + candidate_text_cols
    candidate_feature_cols = [
        c for c in train_df.columns if c not in exclude_cols and c in test_df.columns
    ]

    final_feature_cols = []
    for c in candidate_feature_cols:
        if train_df[c].dtype.name not in ["object", "category"]:
            final_feature_cols.append(c)

    for c in final_feature_cols:
        train_median = (
            train_df[c].median() if not pd.isna(train_df[c].median()) else 0.0
        )
        if train_df[c].isna().sum() > 0:
            train_df[f"{c}__isnan"] = train_df[c].isna().astype("float32")
            test_df[f"{c}__isnan"] = test_df[c].isna().astype("float32")
            if f"{c}__isnan" not in final_feature_cols:
                final_feature_cols.append(f"{c}__isnan")
        train_df[c] = train_df[c].fillna(train_median).astype("float32")
        test_df[c] = test_df[c].fillna(train_median).astype("float32")

    # Clean feature matrices
    X_train = train_df[final_feature_cols].copy().astype(np.float32)
    X_test = test_df[final_feature_cols].copy().astype(np.float32)

    n_train = len(X_train)
    n_test = len(X_test)
    n_splits = 5

    # 8. Create Stratified Validation Folds
    target_bins = pd.qcut(y_train, q=n_splits, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=int)
    for fold_num, (_, val_idx) in enumerate(skf.split(X_train, target_bins)):
        folds[val_idx] = fold_num

    # 9. Out-of-Fold Cross-Validation Modeling
    oof_lgb_raw = np.zeros(n_train, dtype=np.float64)
    oof_lgb_log = np.zeros(n_train, dtype=np.float64)
    oof_lgb_twe = np.zeros(n_train, dtype=np.float64)
    oof_cb_raw = np.zeros(n_train, dtype=np.float64)
    oof_cb_log = np.zeros(n_train, dtype=np.float64)
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    oof_et = np.zeros(n_train, dtype=np.float64)
    oof_ridge = np.zeros(n_train, dtype=np.float64)
    oof_nn = np.zeros(n_train, dtype=np.float64)

    test_lgb_raw = np.zeros(n_test, dtype=np.float64)
    test_lgb_log = np.zeros(n_test, dtype=np.float64)
    test_lgb_twe = np.zeros(n_test, dtype=np.float64)
    test_cb_raw = np.zeros(n_test, dtype=np.float64)
    test_cb_log = np.zeros(n_test, dtype=np.float64)
    test_xgb = np.zeros(n_test, dtype=np.float64)
    test_et = np.zeros(n_test, dtype=np.float64)
    test_ridge = np.zeros(n_test, dtype=np.float64)
    test_nn = np.zeros(n_test, dtype=np.float64)

    for fold in range(n_splits):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr = X_train.iloc[train_idx].copy().reset_index(drop=True)
        X_va = X_train.iloc[val_idx].copy().reset_index(drop=True)
        X_te_fold = X_test.copy().reset_index(drop=True)

        y_tr = y_train[train_idx]
        y_va = y_train[val_idx]

        y_tr_log = np.log1p(np.maximum(0.0, y_tr))
        y_va_log = np.log1p(np.maximum(0.0, y_va))

        # A. Fold-safe Dual Text TF-IDF + Ridge Regression
        if candidate_text_cols:
            tr_texts = [train_combined_text[i] for i in train_idx]
            va_texts = [train_combined_text[i] for i in val_idx]
            tr_dense_f = train_dense[train_idx] if train_dense is not None else None
            va_dense_f = train_dense[val_idx] if train_dense is not None else None
            (
                tr_tfidf_r,
                va_tfidf_r,
                te_tfidf_r,
                tr_dense_r,
                va_dense_r,
                te_dense_r,
            ) = compute_fold_text_ridge(
                tr_texts,
                va_texts,
                test_combined_text,
                y_tr,
                train_dense=tr_dense_f,
                val_dense=va_dense_f,
                test_dense=test_dense,
                alpha=10.0,
            )
            X_tr["text__tfidf_ridge_pred"] = tr_tfidf_r
            X_va["text__tfidf_ridge_pred"] = va_tfidf_r
            X_te_fold["text__tfidf_ridge_pred"] = te_tfidf_r
            X_tr["text__dense_ridge_pred"] = tr_dense_r
            X_va["text__dense_ridge_pred"] = va_dense_r
            X_te_fold["text__dense_ridge_pred"] = te_dense_r

        # B. Fold-safe Smoothed Categorical Target Encodings
        if all_cat_cols:
            tr_te, va_te, te_te = compute_fold_target_encodings(
                cat_df_train.iloc[train_idx],
                cat_df_train.iloc[val_idx],
                cat_df_test,
                target_col=target_col,
                cat_cols=all_cat_cols,
                smoothing=10.0,
            )
            for c in tr_te.columns:
                X_tr[c] = tr_te[c].values
                X_va[c] = va_te[c].values
                X_te_fold[c] = te_te[c].values

        X_tr = X_tr.fillna(0.0).astype(np.float32)
        X_va = X_va.fillna(0.0).astype(np.float32)
        X_te_fold = X_te_fold.fillna(0.0).astype(np.float32)

        # --- 1. LightGBM Regressor (Raw Target) ---
        lgb_raw_model = create_lgb_regressor(random_state=42 + fold)
        lgb_raw_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_raw_val_pred = np.maximum(0.0, lgb_raw_model.predict(X_va))
        lgb_raw_test_pred = np.maximum(0.0, lgb_raw_model.predict(X_te_fold))

        oof_lgb_raw[val_idx] = lgb_raw_val_pred
        test_lgb_raw += lgb_raw_test_pred / n_splits

        # --- 2. LightGBM Regressor (Log Target) ---
        lgb_log_model = create_lgb_log_regressor(random_state=42 + fold)
        lgb_log_model.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_log_val_pred = np.maximum(0.0, np.expm1(lgb_log_model.predict(X_va)))
        lgb_log_test_pred = np.maximum(0.0, np.expm1(lgb_log_model.predict(X_te_fold)))

        oof_lgb_log[val_idx] = lgb_log_val_pred
        test_lgb_log += lgb_log_test_pred / n_splits

        # --- 3. LightGBM Regressor (Tweedie Loss) ---
        lgb_twe_model = create_lgb_tweedie_regressor(random_state=42 + fold)
        lgb_twe_model.fit(
            X_tr,
            np.maximum(1e-4, y_tr),
            eval_set=[(X_va, np.maximum(1e-4, y_va))],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_twe_val_pred = np.maximum(0.0, lgb_twe_model.predict(X_va))
        lgb_twe_test_pred = np.maximum(0.0, lgb_twe_model.predict(X_te_fold))

        oof_lgb_twe[val_idx] = lgb_twe_val_pred
        test_lgb_twe += lgb_twe_test_pred / n_splits

        # --- 4. CatBoost Regressor (Raw Target) ---
        cb_raw_model = create_catboost_regressor(random_state=42 + fold)
        cb_raw_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_raw_val_pred = np.maximum(0.0, cb_raw_model.predict(X_va))
        cb_raw_test_pred = np.maximum(0.0, cb_raw_model.predict(X_te_fold))

        oof_cb_raw[val_idx] = cb_raw_val_pred
        test_cb_raw += cb_raw_test_pred / n_splits

        # --- 4. CatBoost Regressor (Log Target) ---
        cb_log_model = create_catboost_log_regressor(random_state=42 + fold)
        cb_log_model.fit(
            X_tr,
            y_tr_log,
            eval_set=(X_va, y_va_log),
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_log_val_pred = np.maximum(0.0, np.expm1(cb_log_model.predict(X_va)))
        cb_log_test_pred = np.maximum(0.0, np.expm1(cb_log_model.predict(X_te_fold)))

        oof_cb_log[val_idx] = cb_log_val_pred
        test_cb_log += cb_log_test_pred / n_splits

        # --- 5. XGBoost Regressor ---
        xgb_model = create_xgb_regressor(random_state=100 + fold)
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        xgb_val_pred = np.maximum(0.0, xgb_model.predict(X_va))
        xgb_test_pred = np.maximum(0.0, xgb_model.predict(X_te_fold))

        oof_xgb[val_idx] = xgb_val_pred
        test_xgb += xgb_test_pred / n_splits

        # --- 6. ExtraTrees Regressor ---
        et_model = create_et_regressor(random_state=42 + fold)
        et_model.fit(X_tr, y_tr)
        et_val_pred = np.maximum(0.0, et_model.predict(X_va))
        et_test_pred = np.maximum(0.0, et_model.predict(X_te_fold))

        oof_et[val_idx] = et_val_pred
        test_et += et_test_pred / n_splits

        # --- 7. Regularized RidgeCV Linear Baseline ---
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr.values)
        X_va_scaled = scaler.transform(X_va.values)
        X_test_scaled = scaler.transform(X_te_fold.values)

        ridge_model = create_ridge_cv_regressor()
        ridge_model.fit(X_tr_scaled, y_tr)
        ridge_val_pred = np.maximum(0.0, ridge_model.predict(X_va_scaled))
        ridge_test_pred = np.maximum(0.0, ridge_model.predict(X_test_scaled))

        oof_ridge[val_idx] = ridge_val_pred
        test_ridge += ridge_test_pred / n_splits

        # --- 8. Tabular Neural ResNet ---
        nn_val_pred, nn_test_pred = train_neural_resnet(
            X_tr_scaled,
            y_tr,
            X_va_scaled,
            y_va,
            X_test_scaled,
            device=device,
            epochs=50,
            batch_size=64,
            lr=1.5e-3,
            patience=10,
        )

        oof_nn[val_idx] = nn_val_pred
        test_nn += nn_test_pred / n_splits

        fold_mean_pred = (
            lgb_raw_val_pred
            + lgb_log_val_pred
            + lgb_twe_val_pred
            + cb_raw_val_pred
            + cb_log_val_pred
            + xgb_val_pred
            + et_val_pred
            + ridge_val_pred
            + nn_val_pred
        ) / 9.0
        fold_rmse = calculate_rmse(y_va, fold_mean_pred)
        print(f"Fold {fold + 1}/{n_splits} Ensemble RMSE: {fold_rmse:,.2f}")

    # 10. Constrained Simplex Blend Optimization
    oof_matrix = np.column_stack([
        oof_lgb_raw,
        oof_lgb_log,
        oof_lgb_twe,
        oof_cb_raw,
        oof_cb_log,
        oof_xgb,
        oof_et,
        oof_ridge,
        oof_nn,
    ])
    test_matrix = np.column_stack([
        test_lgb_raw,
        test_lgb_log,
        test_lgb_twe,
        test_cb_raw,
        test_cb_log,
        test_xgb,
        test_et,
        test_ridge,
        test_nn,
    ])

    num_models = oof_matrix.shape[1]

    def blend_objective(weights: np.ndarray) -> float:
        w_norm = weights / np.sum(weights)
        pred_blend = np.dot(oof_matrix, w_norm)
        return calculate_rmse(y_train, pred_blend) + 1e-4 * float(np.sum(w_norm**2))

    initial_weights = [1.0 / num_models] * num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x if opt_res.success else (np.ones(num_models) / num_models)
    best_weights = np.maximum(0.0, best_weights)
    best_weights = best_weights / np.sum(best_weights)

    final_oof_preds = np.maximum(0.0, np.dot(oof_matrix, best_weights))
    final_test_preds = np.maximum(0.0, np.dot(test_matrix, best_weights))

    final_validation_score = calculate_rmse(y_train, final_oof_preds)

    # 11. Generate Submission
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_preds), dtype=int),
            "Estimated Total Budget ($)": final_test_preds.astype(float),
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()