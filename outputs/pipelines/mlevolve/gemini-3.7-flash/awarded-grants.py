import gc
import os
from typing import Any, Dict, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =========================================================================
# 1. Feature Engineering & Preprocessing
# =========================================================================
def process_awarded_grants_data():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "Financial Assistance"
    if target_col not in train_df.columns:
        possible_targets = [
            c
            for c in train_df.columns
            if "financial" in c.lower()
            or "assistance" in c.lower()
            or "award" in c.lower()
            or "amount" in c.lower()
        ]
        if possible_targets:
            target_col = possible_targets[0]

    y_train = train_df[target_col].values.astype(np.float64)
    train_df = train_df.drop(columns=[target_col])

    n_train = len(train_df)
    n_test = len(test_df)

    # 5-fold Stratified Split based on target quantiles
    n_splits = 5
    n_bins = 10
    binned_y = pd.qcut(y_train, q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_folds = np.zeros(n_train, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, binned_y)):
        oof_folds[val_idx] = fold

    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 1. Explicit Missingness Masks & Missingness Counts
    missing_dict = {}
    missing_dict["total_missing_count"] = combined_df.isna().sum(axis=1).astype(np.float32)
    for col in combined_df.columns:
        if combined_df[col].isna().sum() > 0:
            missing_dict[f"{col}_isna"] = combined_df[col].isna().astype(np.float32)

    for col in list(combined_df.columns):
        if pd.api.types.is_timedelta64_dtype(combined_df[col]):
            combined_df[f"{col}_days"] = (
                combined_df[col].dt.total_seconds() / (24 * 3600)
            ).astype(np.float32)
            combined_df = combined_df.drop(columns=[col])

    # 2. Date parsing and temporal feature extraction + duration interactions
    date_cols = []
    for col in combined_df.columns:
        col_lower = col.lower()
        if any(
            keyword in col_lower
            for keyword in ["date", "year", "start", "end", "period", "award_date"]
        ):
            date_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(combined_df[col]):
            date_cols.append(col)

    date_cols = list(set(date_cols))
    parsed_date_dict = {}
    new_date_features = {}
    for col in date_cols:
        try:
            parsed = pd.to_datetime(combined_df[col], errors="coerce")
            if parsed.notna().sum() > 0.1 * len(combined_df):
                parsed_date_dict[col] = parsed
                new_date_features[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(np.float32)
                new_date_features[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(np.float32)
                new_date_features[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(np.float32)
                new_date_features[f"{col}_quarter"] = parsed.dt.quarter.fillna(-1).astype(np.float32)
                new_date_features[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(np.float32)
                new_date_features[f"{col}_dayofyear"] = parsed.dt.dayofyear.fillna(-1).astype(np.float32)
                ts_vals = np.where(
                    parsed.notna(),
                    parsed.values.astype("datetime64[s]").astype(np.float64),
                    np.nan,
                )
                new_date_features[f"{col}_timestamp"] = pd.Series(ts_vals, index=combined_df.index, dtype=np.float32)
        except Exception:
            continue

    parsed_keys = list(parsed_date_dict.keys())
    duration_cols = []
    for i in range(len(parsed_keys)):
        for j in range(i + 1, len(parsed_keys)):
            col1 = parsed_keys[i]
            col2 = parsed_keys[j]
            diff_days = (
                (parsed_date_dict[col1] - parsed_date_dict[col2]).dt.total_seconds() / (24 * 3600)
            ).astype(np.float32)
            dname = f"diff_days_{col1}_vs_{col2}"
            new_date_features[dname] = diff_days
            new_date_features[f"{dname}_years"] = (diff_days / 365.25).astype(np.float32)
            new_date_features[f"{dname}_is_multi_year"] = (diff_days.abs() > 365.0).astype(np.float32)
            new_date_features[f"{dname}_log"] = np.log1p(np.maximum(0, diff_days)).astype(np.float32)
            duration_cols.append(dname)

    combined_df = combined_df.drop(columns=date_cols, errors="ignore")

    # 3. Dual-Level Text Features (Word + Char n-grams, TruncatedSVD & Out-of-fold Ridge)
    text_cols = []
    for col in combined_df.columns:
        if combined_df[col].dtype == object or isinstance(
            combined_df[col].dtype, pd.StringDtype
        ):
            avg_str_len = combined_df[col].dropna().astype(str).str.len().mean()
            nunique_val = combined_df[col].nunique()
            if avg_str_len > 5 or nunique_val > 15:
                text_cols.append(col)

    text_count_features = {}
    train_text_extra = {}
    test_text_extra = {}

    for col in text_cols:
        text_series = combined_df[col].fillna("").astype(str)

        text_count_features[f"{col}_char_count"] = text_series.str.len().astype(np.float32)
        text_count_features[f"{col}_word_count"] = (
            text_series.str.split().str.len().astype(np.float32)
        )
        text_count_features[f"{col}_upper_count"] = text_series.apply(
            lambda x: sum(1 for c in x if c.isupper())
        ).astype(np.float32)
        text_count_features[f"{col}_digit_count"] = text_series.apply(
            lambda x: sum(1 for c in x if c.isdigit())
        ).astype(np.float32)

        train_texts = text_series.iloc[:n_train]
        test_texts = text_series.iloc[n_train:]

        word_vec = TfidfVectorizer(
            max_features=12000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        char_vec = TfidfVectorizer(
            max_features=12000, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
        )

        train_word_mat = word_vec.fit_transform(train_texts)
        test_word_mat = word_vec.transform(test_texts)
        train_char_mat = char_vec.fit_transform(train_texts)
        test_char_mat = char_vec.transform(test_texts)

        train_tfidf_combined = sp.hstack([train_word_mat, train_char_mat]).tocsr()
        test_tfidf_combined = sp.hstack([test_word_mat, test_char_mat]).tocsr()

        n_svd = 24
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        train_svd = svd.fit_transform(train_tfidf_combined)
        test_svd = svd.transform(test_tfidf_combined)

        for k in range(n_svd):
            train_text_extra[f"{col}_svd_{k}"] = train_svd[:, k].astype(np.float32)
            test_text_extra[f"{col}_svd_{k}"] = test_svd[:, k].astype(np.float32)

        oof_ridge = np.zeros(n_train, dtype=np.float32)
        test_ridge = np.zeros(n_test, dtype=np.float32)

        for fold in range(n_splits):
            tr_idx = np.where(oof_folds != fold)[0]
            va_idx = np.where(oof_folds == fold)[0]

            ridge = Ridge(alpha=5.0, random_state=42)
            ridge.fit(train_tfidf_combined[tr_idx], y_train[tr_idx])
            oof_ridge[va_idx] = ridge.predict(train_tfidf_combined[va_idx]).astype(np.float32)
            test_ridge += ridge.predict(test_tfidf_combined).astype(np.float32) / n_splits

        train_text_extra[f"{col}_ridge_oof"] = oof_ridge
        test_text_extra[f"{col}_ridge_oof"] = test_ridge

    # 4. Categorical Encodings & Pairwise Interaction Encodings
    cat_cols = []
    for col in combined_df.columns:
        if combined_df[col].dtype == object or isinstance(
            combined_df[col].dtype, pd.CategoricalDtype
        ):
            cat_cols.append(col)
        elif combined_df[col].nunique() < 100 and not pd.api.types.is_float_dtype(
            combined_df[col]
        ):
            cat_cols.append(col)

    cat_cols = list(set(cat_cols))
    freq_features = {}

    # Construct key pairwise categorical interactions
    interaction_cols = []
    key_cat_sorted = sorted(cat_cols, key=lambda c: combined_df[c].nunique(), reverse=True)[:4]
    for i in range(len(key_cat_sorted)):
        for j in range(i + 1, len(key_cat_sorted)):
            c1, c2 = key_cat_sorted[i], key_cat_sorted[j]
            inter_name = f"inter_{c1}_{c2}"
            combined_df[inter_name] = combined_df[c1].astype(str) + "___" + combined_df[c2].astype(str)
            interaction_cols.append(inter_name)

    all_cat_cols = cat_cols + interaction_cols

    for col in all_cat_cols:
        freq = combined_df[col].value_counts(dropna=False)
        freq_features[f"{col}_freq"] = combined_df[col].map(freq).astype(np.float32)
        combined_df[col] = pd.factorize(combined_df[col].astype(str))[0].astype(np.int32)

    # 5. Group aggregations & Duration interactions
    num_cols = [
        c
        for c in combined_df.columns
        if pd.api.types.is_numeric_dtype(combined_df[c])
        and not pd.api.types.is_timedelta64_dtype(combined_df[c])
        and not pd.api.types.is_datetime64_any_dtype(combined_df[c])
        and c not in all_cat_cols
        and not c.endswith("_freq")
        and not c.endswith("_timestamp")
        and not c.endswith("_year")
    ]
    key_cat_candidates = [c for c in cat_cols if 5 <= combined_df[c].nunique() <= 500]
    key_num_candidates = [c for c in num_cols if combined_df[c].nunique() > 20]

    grp_features = {}
    if key_cat_candidates and key_num_candidates:
        selected_cat = key_cat_candidates[:4]
        selected_num = key_num_candidates[:3]
        for c_col in selected_cat:
            for n_col in selected_num:
                n_series = pd.to_numeric(combined_df[n_col], errors="coerce").fillna(0.0).astype(np.float32)
                grp_mean = pd.to_numeric(
                    combined_df.groupby(c_col)[n_col].transform("mean"),
                    errors="coerce",
                ).fillna(0.0).astype(np.float32)
                grp_std = pd.to_numeric(
                    combined_df.groupby(c_col)[n_col].transform("std"),
                    errors="coerce",
                ).fillna(0.0).astype(np.float32)
                grp_features[f"{n_col}_by_{c_col}_mean"] = grp_mean
                grp_features[f"{n_col}_by_{c_col}_std"] = grp_std
                grp_features[f"{n_col}_diff_from_{c_col}_mean"] = (n_series - grp_mean).astype(np.float32)

    if duration_cols and num_cols:
        for dur_col in duration_cols[:3]:
            dur_series = new_date_features[dur_col]
            for n_col in num_cols[:2]:
                num_s = pd.to_numeric(combined_df[n_col], errors="coerce").fillna(0.0).abs() + 1.0
                grp_features[f"{dur_col}_ratio_{n_col}"] = (dur_series / num_s).astype(np.float32)

    # 6. Adaptive Smoothed Out-Of-Fold Target Encoding (Mean, Std, Count)
    train_te_features = {}
    test_te_features = {}
    global_mean = np.mean(y_train)
    global_std = np.std(y_train) + 1e-6

    for col in all_cat_cols:
        oof_mean = np.zeros(n_train, dtype=np.float32)
        oof_std = np.zeros(n_train, dtype=np.float32)
        oof_cnt = np.zeros(n_train, dtype=np.float32)

        for fold in range(n_splits):
            train_idx = np.where(oof_folds != fold)[0]
            val_idx = np.where(oof_folds == fold)[0]

            fold_tr_cats = combined_df.iloc[:n_train].iloc[train_idx][col]
            fold_tr_y = y_train[train_idx]

            cat_stats = (
                pd.DataFrame({"cat": fold_tr_cats, "target": fold_tr_y})
                .groupby("cat")["target"]
                .agg(["mean", "std", "count"])
            )
            cat_stats["std"] = cat_stats["std"].fillna(0.0)
            smoothing = 12.0
            sm_mean = (cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean) / (cat_stats["count"] + smoothing)
            sm_std = (cat_stats["count"] * cat_stats["std"] + smoothing * global_std) / (cat_stats["count"] + smoothing)

            val_cats = combined_df.iloc[:n_train].iloc[val_idx][col]
            oof_mean[val_idx] = val_cats.map(sm_mean).fillna(global_mean).values
            oof_std[val_idx] = val_cats.map(sm_std).fillna(global_std).values
            oof_cnt[val_idx] = np.log1p(val_cats.map(cat_stats["count"]).fillna(0.0).values)

        full_cat_stats = (
            pd.DataFrame({"cat": combined_df.iloc[:n_train][col], "target": y_train})
            .groupby("cat")["target"]
            .agg(["mean", "std", "count"])
        )
        full_cat_stats["std"] = full_cat_stats["std"].fillna(0.0)
        smoothing = 12.0
        full_sm_mean = (full_cat_stats["count"] * full_cat_stats["mean"] + smoothing * global_mean) / (full_cat_stats["count"] + smoothing)
        full_sm_std = (full_cat_stats["count"] * full_cat_stats["std"] + smoothing * global_std) / (full_cat_stats["count"] + smoothing)

        test_cats = combined_df.iloc[n_train:][col]
        test_te_mean = test_cats.map(full_sm_mean).fillna(global_mean).values.astype(np.float32)
        test_te_std = test_cats.map(full_sm_std).fillna(global_std).values.astype(np.float32)
        test_te_cnt = np.log1p(test_cats.map(full_cat_stats["count"]).fillna(0.0).values).astype(np.float32)

        train_te_features[f"{col}_oof_te_mean"] = oof_mean
        train_te_features[f"{col}_oof_te_std"] = oof_std
        train_te_features[f"{col}_oof_te_cnt"] = oof_cnt
        test_te_features[f"{col}_oof_te_mean"] = test_te_mean
        test_te_features[f"{col}_oof_te_std"] = test_te_std
        test_te_features[f"{col}_oof_te_cnt"] = test_te_cnt

    # 7. Unified Assembly with single pd.concat per split
    base_parts = [combined_df]
    if missing_dict:
        base_parts.append(pd.DataFrame(missing_dict, index=combined_df.index))
    if new_date_features:
        base_parts.append(pd.DataFrame(new_date_features, index=combined_df.index))
    if text_count_features:
        base_parts.append(pd.DataFrame(text_count_features, index=combined_df.index))
    if freq_features:
        base_parts.append(pd.DataFrame(freq_features, index=combined_df.index))
    if grp_features:
        base_parts.append(pd.DataFrame(grp_features, index=combined_df.index))

    combined_assembled = pd.concat(base_parts, axis=1)

    train_base = combined_assembled.iloc[:n_train].reset_index(drop=True)
    test_base = combined_assembled.iloc[n_train:].reset_index(drop=True)

    train_all_parts = [train_base]
    test_all_parts = [test_base]

    if train_text_extra:
        train_all_parts.append(pd.DataFrame(train_text_extra))
        test_all_parts.append(pd.DataFrame(test_text_extra))

    if train_te_features:
        train_all_parts.append(pd.DataFrame(train_te_features))
        test_all_parts.append(pd.DataFrame(test_te_features))

    X_train = pd.concat(train_all_parts, axis=1)
    X_test = pd.concat(test_all_parts, axis=1)

    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        col_median = X_train[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        X_train[col] = X_train[col].fillna(col_median).astype(np.float32)
        X_test[col] = X_test[col].fillna(col_median).astype(np.float32)

    return X_train, y_train, X_test, oof_folds


# =========================================================================
# 2. PyTorch Neural Network & GBDT Architectures
# =========================================================================
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
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
        residual = x
        out = self.block(x)
        return self.activation(out + residual)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[0], dropout_rate=dropout_rate) for _ in range(2)]
        )

        intermediate_layers = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            intermediate_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.intermediate_net = nn.Sequential(*intermediate_layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.intermediate_net(x)
        out = self.head(x)
        return out.squeeze(-1)


class RMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(predictions, targets) + self.eps)


def create_neural_model(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    model = TabularResNet(input_dim=input_dim).to(device)
    criterion = RMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lgbm_params(random_state: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_lgbm_extra_params(random_state: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "extra_trees": True,
        "learning_rate": 0.035,
        "num_leaves": 95,
        "max_depth": 9,
        "min_child_samples": 15,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(random_state: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(random_state: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 3. Model Training, Ensemble Blending & Inference
# =========================================================================
def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_df, y, X_test_df, folds = process_awarded_grants_data()

    n_samples = len(X_train_df)
    n_test = len(X_test_df)
    n_splits = len(np.unique(folds))

    oof_lgb = np.zeros(n_samples, dtype=np.float64)
    oof_lgb_et = np.zeros(n_samples, dtype=np.float64)
    oof_xgb = np.zeros(n_samples, dtype=np.float64)
    oof_cb = np.zeros(n_samples, dtype=np.float64)
    oof_nn = np.zeros(n_samples, dtype=np.float64)

    test_preds_lgb = np.zeros(n_test, dtype=np.float64)
    test_preds_lgb_et = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)
    test_preds_cb = np.zeros(n_test, dtype=np.float64)
    test_preds_nn = np.zeros(n_test, dtype=np.float64)

    for fold in range(n_splits):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = X_train_df.iloc[train_idx].values, y[train_idx]
        X_va, y_va = X_train_df.iloc[val_idx].values, y[val_idx]
        X_te = X_test_df.values

        # 1. Train LightGBM Standard GBDT
        lgb_params = get_lgbm_params(random_state=seed + fold)
        model_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=3000)
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[val_idx] = model_lgb.predict(X_va)
        test_preds_lgb += model_lgb.predict(X_te) / n_splits

        # 2. Train LightGBM Extra-Trees
        lgb_et_params = get_lgbm_extra_params(random_state=seed + fold)
        model_lgb_et = lgb.LGBMRegressor(**lgb_et_params, n_estimators=3000)
        model_lgb_et.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb_et[val_idx] = model_lgb_et.predict(X_va)
        test_preds_lgb_et += model_lgb_et.predict(X_te) / n_splits

        # 3. Train XGBoost
        xgb_params = get_xgb_params(random_state=seed + fold)
        model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50, n_estimators=3000)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_va)
        test_preds_xgb += model_xgb.predict(X_te) / n_splits

        # 3. Train CatBoost
        cb_params = get_catboost_params(random_state=seed + fold)
        cb_params["early_stopping_rounds"] = 50
        model_cb = CatBoostRegressor(**cb_params, iterations=3000)
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_idx] = model_cb.predict(X_va)
        test_preds_cb += model_cb.predict(X_te) / n_splits

        # 4. Train TabularResNet
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        nn_model, criterion, optimizer, scheduler = create_neural_model(
            input_dim=X_tr.shape[1], lr=1e-3, device=str(device)
        )

        best_val_loss = float("inf")
        best_model_state = None
        patience = 8
        patience_counter = 0
        epochs = 40

        for epoch in range(epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            val_targets_list = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    preds = nn_model(batch_x)
                    val_preds_list.extend(preds.cpu().numpy().tolist())
                    val_targets_list.extend(batch_y.numpy().tolist())

            epoch_val_rmse = np.sqrt(
                mean_squared_error(val_targets_list, val_preds_list)
            )

            if epoch_val_rmse < best_val_loss:
                best_val_loss = epoch_val_rmse
                best_model_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_state is not None:
            nn_model.load_state_dict(best_model_state)
            nn_model.to(device)

        nn_model.eval()
        with torch.no_grad():
            fold_val_preds = (
                nn_model(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            fold_test_preds = (
                nn_model(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )

        oof_nn[val_idx] = fold_val_preds
        test_preds_nn += fold_test_preds / n_splits

        del (
            model_lgb,
            model_lgb_et,
            model_xgb,
            model_cb,
            nn_model,
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_tr_scaled,
            X_va_scaled,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Optimal Weighted Ensembling with SLSQP & L2 Regularization
    oof_matrix = np.column_stack([oof_lgb, oof_lgb_et, oof_xgb, oof_cb, oof_nn])
    test_matrix = np.column_stack([test_preds_lgb, test_preds_lgb_et, test_preds_xgb, test_preds_cb, test_preds_nn])

    def rmse_objective(weights):
        w_norm = weights / (np.sum(weights) + 1e-12)
        blend = oof_matrix @ w_norm
        rmse = np.sqrt(mean_squared_error(y, blend))
        reg = 1e-4 * np.sum(weights ** 2)
        return rmse + reg

    init_weights = np.array([0.25, 0.25, 0.25, 0.20, 0.05])
    bounds = [(0.0, 1.0) for _ in range(5)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    opt_result = minimize(
        rmse_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = np.maximum(0, opt_result.x)
    opt_weights = opt_weights / np.sum(opt_weights)

    oof_blend = oof_matrix @ opt_weights
    final_test_predictions = test_matrix @ opt_weights

    final_val_score = float(np.sqrt(mean_squared_error(y, oof_blend)))

    # Save submission
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_predictions), dtype=np.int64),
            "Financial Assistance": final_test_predictions,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()