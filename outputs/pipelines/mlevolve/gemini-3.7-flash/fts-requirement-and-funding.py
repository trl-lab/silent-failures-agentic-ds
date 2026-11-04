import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# 1. MODEL ARCHITECTURES & DEFINITIONS
# =============================================================================
class TargetStandardizer:
    """Zero-mean unit-variance standardizer for regression targets."""

    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) + 1e-8
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.std_ + self.mean_


class ResidualBlock(nn.Module):
    """Residual Block for tabular deep learning with LayerNorm, GELU, and Dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular Residual Neural Network for continuous regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# =============================================================================
# 2. FEATURE ENGINEERING PIPELINE
# =============================================================================
def run_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Applies leak-free feature engineering, group statistics, text SVD, and Bayesian target encodings."""
    target_col = "funding"
    test_cols = [c for c in test_df.columns]
    feature_cols = [c for c in test_cols if c != "id" and c in train_df.columns]

    num_cols = []
    cat_cols = []
    text_cols = []
    date_cols = []

    for col in feature_cols:
        col_type = train_df[col].dtype
        if np.issubdtype(col_type, np.datetime64):
            date_cols.append(col)
        elif np.issubdtype(col_type, np.number):
            num_cols.append(col)
        else:
            avg_len = (
                train_df[col].astype(str).str.len().mean() if len(train_df) > 0 else 0
            )
            n_unique = train_df[col].nunique()
            if avg_len > 40 and n_unique > 50:
                text_cols.append(col)
            else:
                cat_cols.append(col)

    train_feat = train_df.copy()
    test_feat = test_df.copy()

    # Process Date Features with Cyclical Encodings
    for col in date_cols:
        for df in [train_feat, test_feat]:
            dt = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = dt.dt.year.fillna(-1).astype(int)
            df[f"{col}_month"] = dt.dt.month.fillna(-1).astype(int)
            df[f"{col}_day"] = dt.dt.day.fillna(-1).astype(int)
            df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
            df[f"{col}_is_month_end"] = dt.dt.is_month_end.fillna(False).astype(int)

            month_norm = 2.0 * np.pi * dt.dt.month / 12.0
            dow_norm = 2.0 * np.pi * dt.dt.dayofweek / 7.0
            df[f"{col}_month_sin"] = np.sin(month_norm).fillna(0.0).astype(float)
            df[f"{col}_month_cos"] = np.cos(month_norm).fillna(0.0).astype(float)
            df[f"{col}_dow_sin"] = np.sin(dow_norm).fillna(0.0).astype(float)
            df[f"{col}_dow_cos"] = np.cos(dow_norm).fillna(0.0).astype(float)

            df.drop(columns=[col], inplace=True)
            if col in feature_cols:
                feature_cols.remove(col)
                num_cols.extend(
                    [
                        f"{col}_year",
                        f"{col}_month",
                        f"{col}_day",
                        f"{col}_dayofweek",
                        f"{col}_is_month_end",
                        f"{col}_month_sin",
                        f"{col}_month_cos",
                        f"{col}_dow_sin",
                        f"{col}_dow_cos",
                    ]
                )

    # Missing Value & Categorical Normalization
    for col in cat_cols:
        train_feat[col] = train_feat[col].fillna("MISSING").astype(str)
        test_feat[col] = test_feat[col].fillna("MISSING").astype(str)

    # Numerical Feature Engineering
    for col in num_cols:
        if train_feat[col].isna().sum() > 0 or test_feat[col].isna().sum() > 0:
            train_feat[f"{col}_isna"] = train_feat[col].isna().astype(int)
            test_feat[f"{col}_isna"] = test_feat[col].isna().astype(int)

        col_median = train_feat[col].median()
        train_feat[col] = train_feat[col].fillna(col_median)
        test_feat[col] = test_feat[col].fillna(col_median)

        if (train_feat[col] >= 0).all() and (test_feat[col] >= 0).all():
            if train_feat[col].max() > 100:
                train_feat[f"{col}_log1p"] = np.log1p(np.clip(train_feat[col], 0, None))
                test_feat[f"{col}_log1p"] = np.log1p(np.clip(test_feat[col], 0, None))

    # Numerical Ratios and Interactions
    base_num_cols = [c for c in num_cols if not c.endswith("_isna")]
    if len(base_num_cols) >= 2:
        for i in range(len(base_num_cols)):
            for j in range(i + 1, min(i + 4, len(base_num_cols))):
                c1, c2 = base_num_cols[i], base_num_cols[j]
                train_feat[f"{c1}_div_{c2}"] = train_feat[c1] / (
                    train_feat[c2].abs() + 1e-5
                )
                test_feat[f"{c1}_div_{c2}"] = test_feat[c1] / (
                    test_feat[c2].abs() + 1e-5
                )
                train_feat[f"{c1}_diff_{c2}"] = train_feat[c1] - train_feat[c2]
                test_feat[f"{c1}_diff_{c2}"] = test_feat[c1] - test_feat[c2]

    # Frequency Encoding
    for col in cat_cols:
        freq_map = train_feat[col].value_counts(normalize=True).to_dict()
        train_feat[f"{col}_freq"] = (
            train_feat[col].map(freq_map).fillna(0).astype(float)
        )
        test_feat[f"{col}_freq"] = test_feat[col].map(freq_map).fillna(0).astype(float)

        count_map = train_feat[col].value_counts().to_dict()
        train_feat[f"{col}_count"] = (
            train_feat[col].map(count_map).fillna(0).astype(float)
        )
        test_feat[f"{col}_count"] = (
            test_feat[col].map(count_map).fillna(0).astype(float)
        )

    # Group Aggregations
    if len(cat_cols) > 0 and len(base_num_cols) > 0:
        primary_cats = cat_cols[: min(3, len(cat_cols))]
        primary_nums = base_num_cols[: min(3, len(base_num_cols))]
        for cat in primary_cats:
            for num in primary_nums:
                agg_stats = (
                    train_feat.groupby(cat)[num]
                    .agg(["mean", "std", "min", "max"])
                    .reset_index()
                )
                agg_stats.columns = [
                    cat,
                    f"{num}_grp_{cat}_mean",
                    f"{num}_grp_{cat}_std",
                    f"{num}_grp_{cat}_min",
                    f"{num}_grp_{cat}_max",
                ]

                train_feat = train_feat.merge(agg_stats, on=cat, how="left")
                test_feat = test_feat.merge(agg_stats, on=cat, how="left")

                for stat in ["mean", "std", "min", "max"]:
                    stat_col = f"{num}_grp_{cat}_{stat}"
                    train_stat_median = train_feat[stat_col].median()
                    train_feat[stat_col] = train_feat[stat_col].fillna(
                        train_stat_median
                    )
                    test_feat[stat_col] = test_feat[stat_col].fillna(train_stat_median)

                mean_col = f"{num}_grp_{cat}_mean"
                train_feat[f"{num}_ratio_to_{cat}_mean"] = train_feat[num] / (
                    train_feat[mean_col].abs() + 1e-5
                )
                test_feat[f"{num}_ratio_to_{cat}_mean"] = test_feat[num] / (
                    test_feat[mean_col].abs() + 1e-5
                )

    # PCA Numerical Projections
    if len(base_num_cols) >= 3:
        pca_scaler = StandardScaler()
        train_num_scaled = pca_scaler.fit_transform(train_feat[base_num_cols].fillna(0))
        test_num_scaled = pca_scaler.transform(test_feat[base_num_cols].fillna(0))

        n_pca_comps = min(6, len(base_num_cols))
        pca = PCA(n_components=n_pca_comps, random_state=42)
        train_pca = pca.fit_transform(train_num_scaled)
        test_pca = pca.transform(test_num_scaled)

        for comp_i in range(n_pca_comps):
            train_feat[f"pca_num_{comp_i}"] = train_pca[:, comp_i]
            test_feat[f"pca_num_{comp_i}"] = test_pca[:, comp_i]

    # Multi-Resolution Text TF-IDF + SVD
    for col in text_cols:
        train_text = train_feat[col].fillna("").astype(str)
        test_text = test_feat[col].fillna("").astype(str)

        # Word-level TF-IDF
        tfidf = TfidfVectorizer(
            max_features=300, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_train = tfidf.fit_transform(train_text)
        tfidf_test = tfidf.transform(test_text)

        n_components = min(12, max(1, tfidf_train.shape[1] - 1))
        if n_components > 1:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_train = svd.fit_transform(tfidf_train)
            svd_test = svd.transform(tfidf_test)

            for comp_idx in range(n_components):
                train_feat[f"{col}_svd_{comp_idx}"] = svd_train[:, comp_idx]
                test_feat[f"{col}_svd_{comp_idx}"] = svd_test[:, comp_idx]

        # Character subword TF-IDF
        char_tfidf = TfidfVectorizer(
            max_features=200, analyzer="char_wb", ngram_range=(3, 5)
        )
        char_train = char_tfidf.fit_transform(train_text)
        char_test = char_tfidf.transform(test_text)
        n_char_comps = min(6, max(1, char_train.shape[1] - 1))
        if n_char_comps > 1:
            svd_char = TruncatedSVD(n_components=n_char_comps, random_state=42)
            svd_char_train = svd_char.fit_transform(char_train)
            svd_char_test = svd_char.transform(char_test)
            for comp_idx in range(n_char_comps):
                train_feat[f"{col}_char_svd_{comp_idx}"] = svd_char_train[:, comp_idx]
                test_feat[f"{col}_char_svd_{comp_idx}"] = svd_char_test[:, comp_idx]

        train_feat[f"{col}_len"] = train_text.str.len()
        test_feat[f"{col}_len"] = test_text.str.len()
        train_feat[f"{col}_word_count"] = train_text.str.split().str.len()
        test_feat[f"{col}_word_count"] = test_text.str.split().str.len()

        train_feat.drop(columns=[col], inplace=True)
        test_feat.drop(columns=[col], inplace=True)

    # Stratified K-Fold Setup
    n_splits = 5
    n_bins = 10
    train_feat["target_bin"] = pd.qcut(
        train_feat[target_col].rank(method="first"),
        q=n_bins,
        labels=False,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_feat["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(train_feat, train_feat["target_bin"])
    ):
        train_feat.loc[val_idx, "fold"] = fold_idx

    # Bayesian Out-Of-Fold Target Encoding
    global_target_mean = train_feat[target_col].mean()
    m_weight = 10.0

    for cat in cat_cols:
        te_col_name = f"{cat}_te"
        train_feat[te_col_name] = global_target_mean
        test_feat[te_col_name] = global_target_mean

        for fold_idx in range(n_splits):
            tr_idx = train_feat["fold"] != fold_idx
            va_idx = train_feat["fold"] == fold_idx

            fold_tr = train_feat.loc[tr_idx]
            fold_stats = fold_tr.groupby(cat)[target_col].agg(["count", "mean"])
            smoothed_val = (
                fold_stats["count"] * fold_stats["mean"] + m_weight * global_target_mean
            ) / (fold_stats["count"] + m_weight)
            smooth_dict = smoothed_val.to_dict()

            train_feat.loc[va_idx, te_col_name] = (
                train_feat.loc[va_idx, cat].map(smooth_dict).fillna(global_target_mean)
            )

        full_stats = train_feat.groupby(cat)[target_col].agg(["count", "mean"])
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + m_weight * global_target_mean
        ) / (full_stats["count"] + m_weight)
        test_feat[te_col_name] = (
            test_feat[cat].map(full_smoothed.to_dict()).fillna(global_target_mean)
        )

    # Encode Categoricals as numeric codes for tabular models
    for col in cat_cols:
        categories = list(train_feat[col].unique())
        train_feat[col] = pd.Categorical(
            train_feat[col], categories=categories
        ).codes.astype(int)
        test_feat[col] = pd.Categorical(
            test_feat[col], categories=categories
        ).codes.astype(int)

    train_feat.drop(columns=["target_bin"], inplace=True)
    return train_feat, test_feat, target_col, n_splits, cat_cols


# =============================================================================
# 3. TRAINING, EVALUATION & BLENDING
# =============================================================================
def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    train_raw = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_raw = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    train_df, test_df, target_col, n_splits, cat_cols = run_feature_engineering(
        train_raw, test_raw
    )

    feature_cols = [c for c in train_df.columns if c not in ["fold", target_col, "id"]]

    X_test_all = test_df[feature_cols].to_numpy(dtype=np.float32)
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=1e9, neginf=-1e9)

    n_train = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros(n_train, dtype=np.float64)
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    oof_cat = np.zeros(n_train, dtype=np.float64)
    oof_et = np.zeros(n_train, dtype=np.float64)
    oof_nn = np.zeros(n_train, dtype=np.float64)

    test_preds_lgb = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)
    test_preds_cat = np.zeros(n_test, dtype=np.float64)
    test_preds_et = np.zeros(n_test, dtype=np.float64)
    test_preds_nn = np.zeros(n_test, dtype=np.float64)

    y_all = train_df[target_col].to_numpy(dtype=np.float64)
    folds = train_df["fold"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        X_tr = train_df.iloc[tr_idx][feature_cols].to_numpy(dtype=np.float32)
        y_tr = y_all[tr_idx]

        X_va = train_df.iloc[va_idx][feature_cols].to_numpy(dtype=np.float32)
        y_va = y_all[va_idx]

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e9, neginf=-1e9)
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=1e9, neginf=-1e9)

        # Standardize features for Neural Network
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test_all)

        X_tr_sc = np.nan_to_num(X_tr_sc, nan=0.0, posinf=1e9, neginf=-1e9)
        X_va_sc = np.nan_to_num(X_va_sc, nan=0.0, posinf=1e9, neginf=-1e9)
        X_te_sc = np.nan_to_num(X_te_sc, nan=0.0, posinf=1e9, neginf=-1e9)

        # --- A. LightGBM ---
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l1=0.1,
            lambda_l2=1.0,
            min_child_samples=20,
            random_state=42 + fold,
            n_estimators=1200,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[va_idx] = lgb_model.predict(X_va)
        test_preds_lgb += lgb_model.predict(X_test_all) / n_splits

        # --- B. XGBoost ---
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42 + fold,
            n_estimators=1200,
            tree_method="hist",
            early_stopping_rounds=40,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] = xgb_model.predict(X_va)
        test_preds_xgb += xgb_model.predict(X_test_all) / n_splits

        # --- C. CatBoost Regressor ---
        cat_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            iterations=1200,
            verbose=False,
            early_stopping_rounds=40,
        )
        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            verbose=False,
            use_best_model=True,
        )
        oof_cat[va_idx] = cat_model.predict(X_va)
        test_preds_cat += cat_model.predict(X_test_all) / n_splits

        # --- D. ExtraTrees Regressor ---
        et_model = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.7,
            n_jobs=-1,
            random_state=42 + fold,
        )
        et_model.fit(X_tr, y_tr)
        oof_et[va_idx] = et_model.predict(X_va)
        test_preds_et += et_model.predict(X_test_all) / n_splits

        # --- E. Deep Tabular Residual Neural Network with Target Normalization ---
        target_scaler = TargetStandardizer()
        y_tr_scaled = target_scaler.fit_transform(y_tr)
        y_va_scaled = (y_va - target_scaler.mean_) / target_scaler.std_

        nn_model = TabularResNet(
            input_dim=X_tr_sc.shape[1], hidden_dim=128, num_blocks=2, dropout=0.2
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        max_epochs = 40
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )

        train_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr_scaled, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        best_val_loss = float("inf")
        best_nn_weights = None

        X_va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        y_va_tensor = torch.tensor(y_va_scaled, dtype=torch.float32).to(device)

        for epoch in range(max_epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_preds = nn_model(X_va_tensor)
                val_loss = criterion(val_preds, y_va_tensor).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_nn_weights = {
                        k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                    }

        if best_nn_weights is not None:
            nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()
        with torch.no_grad():
            scaled_va_pred = nn_model(X_va_tensor).cpu().numpy().astype(np.float64)
            oof_nn[va_idx] = target_scaler.inverse_transform(scaled_va_pred)

            X_te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            scaled_te_pred = nn_model(X_te_tensor).cpu().numpy().astype(np.float64)
            test_preds_nn += (
                target_scaler.inverse_transform(scaled_te_pred) / n_splits
            )

        fold_rmse_lgb = np.sqrt(mean_squared_error(y_va, oof_lgb[va_idx]))
        fold_rmse_xgb = np.sqrt(mean_squared_error(y_va, oof_xgb[va_idx]))
        fold_rmse_cat = np.sqrt(mean_squared_error(y_va, oof_cat[va_idx]))
        fold_rmse_et = np.sqrt(mean_squared_error(y_va, oof_et[va_idx]))
        fold_rmse_nn = np.sqrt(mean_squared_error(y_va, oof_nn[va_idx]))
        print(
            f"Fold {fold+1}/{n_splits} | LGB: {fold_rmse_lgb:.4f} | XGB: {fold_rmse_xgb:.4f} | CAT: {fold_rmse_cat:.4f} | ET: {fold_rmse_et:.4f} | NN: {fold_rmse_nn:.4f}"
        )

    # --- Out-Of-Fold Ensemble Optimization ---
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_nn]
    )

    n_models = oof_matrix.shape[1]

    def ensemble_objective(weights):
        w = np.array(weights)
        pred = np.dot(oof_matrix, w)
        rmse = np.sqrt(mean_squared_error(y_all, pred))
        l2_reg = 1e-4 * np.sum(w ** 2)
        return rmse + l2_reg

    initial_weights = [1.0 / n_models] * n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        ensemble_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = opt_res.x if opt_res.success else np.array(initial_weights)
    opt_weights = np.maximum(0.0, opt_weights)
    opt_weights = opt_weights / np.sum(opt_weights)

    final_oof_preds = np.clip(np.dot(oof_matrix, opt_weights), 0, None)
    final_test_preds = np.clip(np.dot(test_matrix, opt_weights), 0, None)

    final_rmse = np.sqrt(mean_squared_error(y_all, final_oof_preds))

    # --- Generate Submission File ---
    submission = pd.DataFrame()
    if "id" in test_raw.columns:
        submission["id"] = test_raw["id"].astype(int)
    else:
        submission["id"] = np.arange(len(test_raw), dtype=int)

    submission[target_col] = final_test_preds
    sub_path = os.path.join(submission_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
