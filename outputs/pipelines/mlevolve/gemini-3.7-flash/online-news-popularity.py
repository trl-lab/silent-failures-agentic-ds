import copy
import os
import re
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Ensure necessary directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set deterministic random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================================================================
# 1. Feature Engineering & Anomaly Cleaning
# ==============================================================================
def extract_url_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal and lexical signals from the article URL."""
    urls = df["url"].astype(str)
    pattern = r"mashable\.com/(\d{4})/(\d{2})/(\d{2})/([^/]+)/?"
    extracted = urls.str.extract(pattern)

    url_year = pd.to_numeric(extracted[0], errors="coerce").fillna(-1).astype(np.int32)
    url_month = pd.to_numeric(extracted[1], errors="coerce").fillna(-1).astype(np.int32)
    url_day = pd.to_numeric(extracted[2], errors="coerce").fillna(-1).astype(np.int32)
    slug = extracted[3].fillna("")

    slug_char_len = slug.str.len()
    slug_word_count = slug.apply(lambda s: len(s.split("-")) if s else 0)
    slug_avg_word_len = slug_char_len / (slug_word_count + 1e-5)

    slug_has_number = slug.apply(lambda s: 1 if re.search(r"\d", s) else 0)
    slug_has_how = slug.apply(lambda s: 1 if "how" in s else 0)
    slug_has_why = slug.apply(lambda s: 1 if "why" in s else 0)
    slug_has_best_top = slug.apply(lambda s: 1 if ("best" in s or "top" in s) else 0)
    slug_has_video = slug.apply(lambda s: 1 if "video" in s else 0)
    slug_has_photo = slug.apply(
        lambda s: 1 if ("photo" in s or "picture" in s or "image" in s) else 0
    )

    url_feats = pd.DataFrame(
        {
            "url_year": url_year,
            "url_month": url_month,
            "url_day": url_day,
            "slug_char_len": slug_char_len,
            "slug_word_count": slug_word_count,
            "slug_avg_word_len": slug_avg_word_len,
            "slug_has_number": slug_has_number,
            "slug_has_how": slug_has_how,
            "slug_has_why": slug_has_why,
            "slug_has_best_top": slug_has_best_top,
            "slug_has_video": slug_has_video,
            "slug_has_photo": slug_has_photo,
        },
        index=df.index,
    )
    return url_feats


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data cleaning, anomaly repair, and multi-modal feature engineering."""
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    # 1. Anomaly Cleaning
    for col in ["n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens"]:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: np.nan if x > 1.0 else x)
            data[col] = data[col].fillna(
                data[col].median() if not data[col].isna().all() else 0.0
            )

    if "n_tokens_content" in data.columns:
        data["is_zero_content"] = (data["n_tokens_content"] == 0).astype(np.float32)

    # 2. Extract URL semantic features
    if "url" in data.columns:
        url_feats = extract_url_features(data)
        data = pd.concat([data, url_feats], axis=1)

    # 3. Multimedia & Content Density Features
    if "num_imgs" in data.columns and "num_videos" in data.columns:
        data["total_media"] = data["num_imgs"] + data["num_videos"]
        data["has_imgs"] = (data["num_imgs"] > 0).astype(np.float32)
        data["has_videos"] = (data["num_videos"] > 0).astype(np.float32)
        data["has_media"] = (data["total_media"] > 0).astype(np.float32)

    if "n_tokens_content" in data.columns:
        token_cnt = np.maximum(data["n_tokens_content"], 1.0)
        if "num_hrefs" in data.columns:
            data["href_density"] = data["num_hrefs"] / token_cnt
        if "num_self_hrefs" in data.columns:
            data["self_href_density"] = data["num_self_hrefs"] / token_cnt
        if "num_self_hrefs" in data.columns and "num_hrefs" in data.columns:
            data["self_href_ratio"] = data["num_self_hrefs"] / (data["num_hrefs"] + 1.0)
        if "num_imgs" in data.columns:
            data["img_density"] = data["num_imgs"] / token_cnt
        if "num_videos" in data.columns:
            data["video_density"] = data["num_videos"] / token_cnt
        if "total_media" in data.columns:
            data["media_density"] = data["total_media"] / token_cnt

    # 4. Keyword Popularity Aggregations & Dynamics
    kw_cols = [
        "kw_min_min",
        "kw_max_min",
        "kw_avg_min",
        "kw_min_max",
        "kw_max_max",
        "kw_avg_max",
        "kw_min_avg",
        "kw_max_avg",
        "kw_avg_avg",
    ]
    for c in kw_cols:
        if c in data.columns:
            data[f"{c}_clipped"] = np.maximum(data[c], 0.0)
            data[f"{c}_log1p"] = np.log1p(data[f"{c}_clipped"])

    if "kw_max_min_clipped" in data.columns and "kw_min_min_clipped" in data.columns:
        data["kw_min_spread"] = data["kw_max_min_clipped"] - data["kw_min_min_clipped"]
    if "kw_max_max_clipped" in data.columns and "kw_min_max_clipped" in data.columns:
        data["kw_max_spread"] = data["kw_max_max_clipped"] - data["kw_min_max_clipped"]
    if "kw_max_avg_clipped" in data.columns and "kw_min_avg_clipped" in data.columns:
        data["kw_avg_spread"] = data["kw_max_avg_clipped"] - data["kw_min_avg_clipped"]
    if "kw_avg_avg_clipped" in data.columns and "kw_max_avg_clipped" in data.columns:
        data["kw_avg_to_max_ratio"] = data["kw_avg_avg_clipped"] / (
            data["kw_max_avg_clipped"] + 1.0
        )
    if "kw_min_avg_clipped" in data.columns and "kw_avg_avg_clipped" in data.columns:
        data["kw_min_to_avg_ratio"] = data["kw_min_avg_clipped"] / (
            data["kw_avg_avg_clipped"] + 1.0
        )

    # 5. Self-Reference Share Signals
    self_ref_cols = [
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_sharess",
    ]
    for c in self_ref_cols:
        if c in data.columns:
            data[f"{c}_clipped"] = np.maximum(data[c], 0.0)
            data[f"{c}_log1p"] = np.log1p(data[f"{c}_clipped"])

    if "self_reference_max_shares_clipped" in data.columns and "self_reference_min_shares_clipped" in data.columns:
        data["self_ref_spread"] = (
            data["self_reference_max_shares_clipped"]
            - data["self_reference_min_shares_clipped"]
        )
    if "self_reference_max_shares_log1p" in data.columns and "self_reference_min_shares_log1p" in data.columns:
        data["self_ref_spread_log"] = (
            data["self_reference_max_shares_log1p"]
            - data["self_reference_min_shares_log1p"]
        )
    if "self_reference_avg_sharess_log1p" in data.columns and "kw_avg_avg_log1p" in data.columns:
        data["self_ref_vs_kw_avg"] = (
            data["self_reference_avg_sharess_log1p"] - data["kw_avg_avg_log1p"]
        )

    # 6. LDA Topic Dynamics & Entropy
    lda_cols = ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"]
    if all(c in data.columns for c in lda_cols):
        lda_matrix = data[lda_cols].values
        data["lda_dominant_topic"] = np.argmax(lda_matrix, axis=1).astype(np.int32)
        data["lda_max_score"] = np.max(lda_matrix, axis=1)
        data["lda_min_score"] = np.min(lda_matrix, axis=1)
        data["lda_entropy"] = -np.sum(
            lda_matrix * np.log(np.maximum(lda_matrix, 1e-9)), axis=1
        )
        data["lda_margin"] = (
            data["lda_max_score"] - np.partition(lda_matrix, -2, axis=1)[:, -2]
        )

    # 7. Sentiment and Polarity Contrasts
    if "title_sentiment_polarity" in data.columns and "global_sentiment_polarity" in data.columns:
        data["title_vs_content_polarity_diff"] = np.abs(
            data["title_sentiment_polarity"] - data["global_sentiment_polarity"]
        )
    if "title_subjectivity" in data.columns and "global_subjectivity" in data.columns:
        data["title_vs_content_subjectivity_diff"] = np.abs(
            data["title_subjectivity"] - data["global_subjectivity"]
        )
    if "global_rate_positive_words" in data.columns and "global_rate_negative_words" in data.columns:
        data["pos_neg_word_rate_ratio"] = (data["global_rate_positive_words"] + 1e-4) / (
            data["global_rate_negative_words"] + 1e-4
        )
    if "avg_positive_polarity" in data.columns and "avg_negative_polarity" in data.columns:
        data["pos_neg_polarity_ratio"] = (data["avg_positive_polarity"] + 1e-4) / (
            np.abs(data["avg_negative_polarity"]) + 1e-4
        )
    if "abs_title_sentiment_polarity" in data.columns and "title_subjectivity" in data.columns:
        data["title_emotional_intensity"] = (
            data["abs_title_sentiment_polarity"] * data["title_subjectivity"]
        )
    if "max_positive_polarity" in data.columns and "min_positive_polarity" in data.columns:
        data["positive_polarity_range"] = (
            data["max_positive_polarity"] - data["min_positive_polarity"]
        )
    if "max_negative_polarity" in data.columns and "min_negative_polarity" in data.columns:
        data["negative_polarity_range"] = (
            data["max_negative_polarity"] - data["min_negative_polarity"]
        )

    # 8. Channel and Timing Syntheses
    channel_cols = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
    ]
    present_channels = [c for c in channel_cols if c in data.columns]
    if present_channels:
        data["channel_count"] = data[present_channels].sum(axis=1)
        data["channel_is_none"] = (data["channel_count"] == 0).astype(np.float32)

    weekday_cols = [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
    ]
    present_weekdays = [c for c in weekday_cols if c in data.columns]
    if len(present_weekdays) == 7:
        day_indices = np.argmax(data[weekday_cols].values, axis=1)
        data["day_of_week"] = day_indices.astype(np.int32)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    if "timedelta" in data.columns:
        timedelta_safe = np.maximum(data["timedelta"], 1.0)
        if "kw_avg_avg_clipped" in data.columns:
            data["kw_avg_per_day"] = data["kw_avg_avg_clipped"] / timedelta_safe
        if "self_reference_avg_sharess_clipped" in data.columns:
            data["shares_per_day_self_ref"] = (
                data["self_reference_avg_sharess_clipped"] / timedelta_safe
            )

    return data


# ==============================================================================
# 2. Metric and Neural Network Model Definitions
# ==============================================================================
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


class RMSELoss(nn.Module):
    """Differentiable RMSE Loss for PyTorch."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred.view(-1), y_true.view(-1)) + self.eps)


class ResidualBlock(nn.Module):
    """Residual Dense Block with LayerNorm, SiLU, and Dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular Residual Neural Network for extreme target regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(ResidualBlock(hidden_dims[i], dropout_rate=dropout_rate))
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                )
            )
        layers.append(ResidualBlock(hidden_dims[-1], dropout_rate=dropout_rate))
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        feat = self.backbone(feat)
        out = self.head(feat)
        return out.squeeze(-1)


def get_lgbm_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "lgb_rmse": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 47,
            "max_depth": 7,
            "min_child_samples": 25,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.5,
            "reg_lambda": 3.0,
            "n_estimators": 2500,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "lgb_huber": {
            "objective": "huber",
            "metric": "rmse",
            "alpha": 0.9,
            "learning_rate": 0.035,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 2.0,
            "reg_lambda": 5.0,
            "n_estimators": 2500,
            "random_state": 101,
            "n_jobs": -1,
            "verbose": -1,
        },
        "lgb_tweedie": {
            "objective": "tweedie",
            "tweedie_variance_power": 1.15,
            "metric": "rmse",
            "learning_rate": 0.025,
            "num_leaves": 39,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.8,
            "reg_lambda": 2.5,
            "n_estimators": 2500,
            "random_state": 2024,
            "n_jobs": -1,
            "verbose": -1,
        },
    }


def get_xgb_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "xgb_rmse": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "colsample_bylevel": 0.85,
            "reg_alpha": 1.0,
            "reg_lambda": 4.0,
            "n_estimators": 2500,
            "random_state": 42,
            "n_jobs": -1,
        },
        "xgb_pseudohuber": {
            "objective": "reg:pseudohubererror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "learning_rate": 0.035,
            "max_depth": 7,
            "min_child_weight": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "colsample_bylevel": 0.8,
            "reg_alpha": 2.5,
            "reg_lambda": 6.0,
            "n_estimators": 2500,
            "random_state": 101,
            "n_jobs": -1,
        },
    }


def get_catboost_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "cat_rmse": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "subsample": 0.85,
            "iterations": 2500,
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        },
        "cat_logrmse": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "learning_rate": 0.04,
            "depth": 7,
            "l2_leaf_reg": 8.0,
            "subsample": 0.8,
            "iterations": 2500,
            "random_seed": 2024,
            "verbose": False,
            "thread_count": -1,
        },
    }


# ==============================================================================
# 3. Main Pipeline Execution
# ==============================================================================
def main():
    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")
    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    # 2. Preprocess & Feature Engineering
    processed_train = clean_and_engineer_features(train_df)
    processed_test = clean_and_engineer_features(test_df)

    # 3. Stratified Partitioning by Target Quantiles
    n_splits = 5
    target_log = np.log1p(processed_train["shares"].values)
    binned_target = pd.qcut(target_log, q=20, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = np.zeros(len(processed_train), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(processed_train, binned_target)):
        folds[val_idx] = fold_idx
    processed_train["fold"] = folds

    # Prepare features and targets
    target_col = "shares"
    fold_col = "fold"
    ignore_cols = [target_col, fold_col, "url", "id"]
    feature_cols = [
        c
        for c in processed_train.columns
        if c in processed_test.columns and c not in ignore_cols
    ]

    X_all = processed_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_all = processed_train[target_col].values.astype(np.float64)
    folds = processed_train[fold_col].values
    X_test_df = (
        processed_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    n_samples = len(processed_train)
    n_test = len(processed_test)

    # 4. Instantiate Model Suite
    lgb_configs = get_lgbm_configs()
    xgb_configs = get_xgb_configs()
    cat_configs = get_catboost_configs() if HAS_CATBOOST else {}

    model_names = (
        list(lgb_configs.keys())
        + list(xgb_configs.keys())
        + (list(cat_configs.keys()) if HAS_CATBOOST else [])
        + ["tabular_resnet"]
    )

    oof_predictions = {
        name: np.zeros(n_samples, dtype=np.float64) for name in model_names
    }
    test_predictions = {
        name: np.zeros(n_test, dtype=np.float64) for name in model_names
    }

    # 5. 5-Fold Training Loop
    for fold in range(n_splits):
        train_mask = folds != fold
        val_mask = folds == fold

        X_train_fold = X_all.iloc[train_mask]
        y_train_fold = y_all[train_mask]
        X_val_fold = X_all.iloc[val_mask]
        y_val_fold = y_all[val_mask]

        # A. LightGBM
        for name, config in lgb_configs.items():
            cfg = config.copy()
            n_est = cfg.pop("n_estimators", 2500)
            lgb_reg = lgb.LGBMRegressor(**cfg, n_estimators=n_est)
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            lgb_reg.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=callbacks,
            )

            oof_predictions[name][val_mask] = np.maximum(
                lgb_reg.predict(X_val_fold), 0.0
            )
            test_predictions[name] += (
                np.maximum(lgb_reg.predict(X_test_df), 0.0) / n_splits
            )

        # B. XGBoost
        for name, config in xgb_configs.items():
            cfg = config.copy()
            n_est = cfg.pop("n_estimators", 2500)
            xgb_reg = xgb.XGBRegressor(
                **cfg, n_estimators=n_est, early_stopping_rounds=50
            )
            xgb_reg.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            oof_predictions[name][val_mask] = np.maximum(
                xgb_reg.predict(X_val_fold), 0.0
            )
            test_predictions[name] += (
                np.maximum(xgb_reg.predict(X_test_df), 0.0) / n_splits
            )

        # C. CatBoost
        if HAS_CATBOOST:
            for name, config in cat_configs.items():
                cfg = config.copy()
                cb_reg = CatBoostRegressor(
                    **cfg, early_stopping_rounds=50
                )
                cb_reg.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=(X_val_fold, y_val_fold),
                    verbose=False,
                )

                oof_predictions[name][val_mask] = np.maximum(
                    cb_reg.predict(X_val_fold), 0.0
                )
                test_predictions[name] += (
                    np.maximum(cb_reg.predict(X_test_df), 0.0) / n_splits
                )

        # D. Deep Tabular Neural Network
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train_fold)
        X_va_scaled = scaler.transform(X_val_fold)
        X_te_scaled = scaler.transform(X_test_df)

        y_mean = np.mean(y_train_fold)
        y_std = np.std(y_train_fold) + 1e-6
        y_tr_norm = (y_train_fold - y_mean) / y_std
        y_va_norm = (y_val_fold - y_mean) / y_std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_tr_scaled.shape[1]
        nn_model = TabularResNet(
            input_dim=input_dim, hidden_dims=(256, 128, 64), dropout_rate=0.2
        ).to(device)

        nn_epochs = 20
        optimizer = torch.optim.AdamW(
            nn_model.parameters(),
            lr=2e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=nn_epochs, eta_min=1e-6
        )
        criterion = RMSELoss()

        X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_norm, dtype=torch.float32)
        X_va_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        X_te_t = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )

        best_nn_val_rmse = float("inf")
        best_nn_weights = None

        for epoch in range(nn_epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out = nn_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                va_out_norm = nn_model(X_va_t).cpu().numpy()
                va_pred_shares = np.maximum(va_out_norm * y_std + y_mean, 0.0)
                epoch_rmse = compute_rmse(y_val_fold, va_pred_shares)

                if epoch_rmse < best_nn_val_rmse:
                    best_nn_val_rmse = epoch_rmse
                    best_nn_weights = copy.deepcopy(nn_model.state_dict())

        nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()
        with torch.no_grad():
            final_va_norm = nn_model(X_va_t).cpu().numpy()
            oof_predictions["tabular_resnet"][val_mask] = np.maximum(
                final_va_norm * y_std + y_mean, 0.0
            )

            final_te_norm = nn_model(X_te_t).cpu().numpy()
            test_predictions["tabular_resnet"] += (
                np.maximum(final_te_norm * y_std + y_mean, 0.0) / n_splits
            )

        print(
            f"Fold {fold + 1}/{n_splits} complete | TabularResNet Val RMSE: {best_nn_val_rmse:.2f}"
        )

    # 6. Constrained Non-Negative Optimization Blending
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])
    test_matrix = np.column_stack([test_predictions[name] for name in model_names])

    meta_reg = LinearRegression(positive=True, fit_intercept=False)
    meta_reg.fit(oof_matrix, y_all)
    weights = meta_reg.coef_
    sum_weights = np.sum(weights)

    if sum_weights > 0:
        normalized_weights = weights / sum_weights
    else:
        normalized_weights = np.ones(len(model_names)) / len(model_names)

    final_oof = np.maximum(oof_matrix @ normalized_weights, 0.0)
    final_test_pred = np.maximum(test_matrix @ normalized_weights, 0.0)

    final_score = compute_rmse(y_all, final_oof)

    # 7. Generate Submission
    submission_df = pd.DataFrame(
        {"id": np.arange(len(final_test_pred)), "shares": final_test_pred}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
