import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        h = self.fc1(x_norm)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x_norm))
        return residual + g * h


class TabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def process_and_engineer_features():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "Recovery"
    id_col = "id"

    if id_col not in train_df.columns:
        train_df[id_col] = np.arange(len(train_df))
    if id_col not in test_df.columns:
        test_df[id_col] = np.arange(len(test_df))

    y = train_df[target_col].values
    train_ids = train_df[id_col].values
    test_ids = test_df[id_col].values

    train_feats = train_df.drop(columns=[target_col, id_col], errors="ignore").copy()
    test_feats = test_df.drop(columns=[id_col], errors="ignore").copy()

    datetime_cols = []
    text_cols = []
    categorical_cols = []
    numerical_cols = []

    for col in train_feats.columns:
        if pd.api.types.is_datetime64_any_dtype(train_feats[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(train_feats[col]):
            numerical_cols.append(col)
        else:
            sample_vals = train_feats[col].dropna().astype(str)
            avg_words = (
                sample_vals.apply(lambda x: len(x.split())).mean()
                if len(sample_vals) > 0
                else 0
            )
            if avg_words > 3:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    # 1. Datetime Features
    for col in datetime_cols:
        for df in [train_feats, test_feats]:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_dayofyear"] = df[col].dt.dayofyear
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_is_month_end"] = df[col].dt.is_month_end.astype(int)
            df.drop(columns=[col], inplace=True)

    # Group-level statistics and numeric interaction features
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            grp = train_feats.groupby(cat_col)[num_col].agg(["mean", "std", "min", "max"])
            for stat in ["mean", "std", "min", "max"]:
                stat_map = grp[stat].to_dict()
                feat_name = f"{cat_col}_{num_col}_{stat}"
                train_feats[feat_name] = train_feats[cat_col].map(stat_map)
                test_feats[feat_name] = test_feats[cat_col].map(stat_map)

            diff_feat = f"{num_col}_diff_{cat_col}_mean"
            train_feats[diff_feat] = (
                train_feats[num_col] - train_feats[f"{cat_col}_{num_col}_mean"]
            )
            test_feats[diff_feat] = (
                test_feats[num_col] - test_feats[f"{cat_col}_{num_col}_mean"]
            )
            q1, q99 = train_feats[diff_feat].quantile(0.01), train_feats[diff_feat].quantile(0.99)
            train_feats[diff_feat] = train_feats[diff_feat].clip(lower=q1, upper=q99)
            test_feats[diff_feat] = test_feats[diff_feat].clip(lower=q1, upper=q99)

            ratio_feat = f"{num_col}_ratio_{cat_col}_mean"
            train_feats[ratio_feat] = train_feats[num_col] / (
                train_feats[f"{cat_col}_{num_col}_mean"].abs() + 1e-5
            )
            test_feats[ratio_feat] = test_feats[num_col] / (
                test_feats[f"{cat_col}_{num_col}_mean"].abs() + 1e-5
            )
            q1, q99 = train_feats[ratio_feat].quantile(0.01), train_feats[ratio_feat].quantile(0.99)
            train_feats[ratio_feat] = train_feats[ratio_feat].clip(lower=q1, upper=q99)
            test_feats[ratio_feat] = test_feats[ratio_feat].clip(lower=q1, upper=q99)

    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            c1, c2 = numerical_cols[i], numerical_cols[j]
            diff_feat = f"{c1}_minus_{c2}"
            train_feats[diff_feat] = train_feats[c1] - train_feats[c2]
            test_feats[diff_feat] = test_feats[c1] - test_feats[c2]
            q1, q99 = train_feats[diff_feat].quantile(0.01), train_feats[diff_feat].quantile(0.99)
            train_feats[diff_feat] = train_feats[diff_feat].clip(lower=q1, upper=q99)
            test_feats[diff_feat] = test_feats[diff_feat].clip(lower=q1, upper=q99)

            ratio_feat = f"{c1}_ratio_{c2}"
            train_feats[ratio_feat] = train_feats[c1] / (
                train_feats[c2].abs() + 1e-5
            )
            test_feats[ratio_feat] = test_feats[c1] / (
                test_feats[c2].abs() + 1e-5
            )
            q1, q99 = train_feats[ratio_feat].quantile(0.01), train_feats[ratio_feat].quantile(0.99)
            train_feats[ratio_feat] = train_feats[ratio_feat].clip(lower=q1, upper=q99)
            test_feats[ratio_feat] = test_feats[ratio_feat].clip(lower=q1, upper=q99)

    # 2. Text Features
    for col in text_cols:
        for df in [train_feats, test_feats]:
            text_s = df[col].fillna("").astype(str)
            df[f"{col}_char_len"] = text_s.apply(len)
            df[f"{col}_word_count"] = text_s.apply(lambda x: len(x.split()))
            df[f"{col}_digit_count"] = text_s.apply(
                lambda x: sum(c.isdigit() for c in x)
            )

        tr_text = train_feats[col].fillna("").astype(str)
        te_text = test_feats[col].fillna("").astype(str)

        tfidf = TfidfVectorizer(
            max_features=3000, stop_words="english", ngram_range=(1, 2)
        )
        tr_tfidf = tfidf.fit_transform(tr_text)
        te_tfidf = tfidf.transform(te_text)

        n_comp = min(16, max(2, tr_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        tr_svd = svd.fit_transform(tr_tfidf)
        te_svd = svd.transform(te_tfidf)

        for i in range(n_comp):
            train_feats[f"{col}_svd_{i}"] = tr_svd[:, i]
            test_feats[f"{col}_svd_{i}"] = te_svd[:, i]

        train_feats.drop(columns=[col], inplace=True)
        test_feats.drop(columns=[col], inplace=True)

    # Stratified target bins for zero-inflation
    zero_mask = y == 0
    bins = np.zeros(len(y), dtype=int)
    non_zero_y = y[~zero_mask]
    if len(non_zero_y) > 0:
        quantile_bins = pd.qcut(
            non_zero_y,
            q=min(4, len(np.unique(non_zero_y))),
            labels=False,
            duplicates="drop",
        )
        bins[~zero_mask] = quantile_bins + 1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(train_feats), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_feats, bins)):
        folds[val_idx] = fold_idx

    # 3. Categorical Encodings (Frequency & Leak-Free Target Encoding)
    global_mean = float(np.mean(y))
    m = 10.0

    for col in categorical_cols:
        freq = train_feats[col].value_counts(normalize=True).to_dict()
        train_feats[f"{col}_freq"] = train_feats[col].map(freq).fillna(0)
        test_feats[f"{col}_freq"] = test_feats[col].map(freq).fillna(0)

        train_feats[f"{col}_te"] = np.nan
        for fold_idx in range(5):
            tr_idx = folds != fold_idx
            va_idx = folds == fold_idx

            grp = (
                pd.DataFrame(
                    {"cat": train_feats.iloc[tr_idx][col], "target": y[tr_idx]}
                )
                .groupby("cat")["target"]
                .agg(["mean", "count"])
            )
            smoothed = (grp["count"] * grp["mean"] + m * global_mean) / (
                grp["count"] + m
            )
            enc_map = smoothed.to_dict()

            train_feats.loc[va_idx, f"{col}_te"] = (
                train_feats.iloc[va_idx][col].map(enc_map).fillna(global_mean)
            )

        full_grp = (
            pd.DataFrame({"cat": train_feats[col], "target": y})
            .groupby("cat")["target"]
            .agg(["mean", "count"])
        )
        smoothed_full = (full_grp["count"] * full_grp["mean"] + m * global_mean) / (
            full_grp["count"] + m
        )
        test_feats[f"{col}_te"] = (
            test_feats[col].map(smoothed_full.to_dict()).fillna(global_mean)
        )

        cat_codes = pd.CategoricalDtype(categories=train_feats[col].dropna().unique())
        train_feats[col] = train_feats[col].astype(cat_codes).cat.codes
        test_feats[col] = test_feats[col].astype(cat_codes).cat.codes

    # 4. Numerical Features
    for col in numerical_cols:
        if col in train_feats.columns:
            skew_val = train_feats[col].skew()
            if skew_val > 1.5 and train_feats[col].min() >= 0:
                train_feats[f"{col}_log1p"] = np.log1p(train_feats[col])
                test_feats[f"{col}_log1p"] = np.log1p(test_feats[col])

    # Convert remaining non-numeric object columns to categorical codes
    for col in train_feats.columns:
        if train_feats[col].dtype == "object":
            train_feats[col] = train_feats[col].astype("category").cat.codes
            test_feats[col] = test_feats[col].astype("category").cat.codes

    medians = train_feats.median(numeric_only=True)
    train_feats.fillna(medians, inplace=True)
    test_feats.fillna(medians, inplace=True)

    train_feats[id_col] = train_ids
    train_feats[target_col] = y
    train_feats["fold"] = folds
    test_feats[id_col] = test_ids

    return train_feats, test_feats


def main():
    seed_everything(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_df, test_df = process_and_engineer_features()

    target_col = "Recovery"
    id_col = "id"
    fold_col = "fold"

    feature_cols = [
        c for c in train_df.columns if c not in [target_col, id_col, fold_col]
    ]

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_col].values.astype(np.float32)
    folds = train_df[fold_col].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    test_ids = test_df[id_col].values

    num_samples = len(X)
    num_test = len(X_test)
    num_folds = 5

    oof_lgb = np.zeros(num_samples)
    oof_xgb = np.zeros(num_samples)
    oof_cat = np.zeros(num_samples)
    oof_nn = np.zeros(num_samples)

    test_lgb = np.zeros(num_test)
    test_xgb = np.zeros(num_test)
    test_cat = np.zeros(num_test)
    test_nn = np.zeros(num_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(num_folds):
        tr_idx = folds != fold
        va_idx = folds == fold

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        seeds = [42 + fold * 10, 100 + fold * 10, 2024 + fold * 10]

        # LightGBM multi-seed bagging
        lgb_va_preds = np.zeros(len(X_va))
        lgb_te_preds = np.zeros(num_test)
        for s in seeds:
            model_lgb = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=s,
                n_jobs=-1,
                verbose=-1,
            )
            model_lgb.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            lgb_va_preds += model_lgb.predict(X_va) / len(seeds)
            lgb_te_preds += model_lgb.predict(X_test) / len(seeds)
        oof_lgb[va_idx] = lgb_va_preds
        test_lgb += lgb_te_preds / num_folds

        # XGBoost multi-seed bagging
        xgb_va_preds = np.zeros(len(X_va))
        xgb_te_preds = np.zeros(num_test)
        for s in seeds:
            model_xgb = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                early_stopping_rounds=50,
                random_state=s,
                n_jobs=-1,
                tree_method="hist",
            )
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            xgb_va_preds += model_xgb.predict(X_va) / len(seeds)
            xgb_te_preds += model_xgb.predict(X_test) / len(seeds)
        oof_xgb[va_idx] = xgb_va_preds
        test_xgb += xgb_te_preds / num_folds

        # CatBoost multi-seed bagging
        cat_va_preds = np.zeros(len(X_va))
        cat_te_preds = np.zeros(num_test)
        for s in seeds:
            model_cat = cb.CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=s,
                verbose=False,
            )
            model_cat.fit(
                X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
            )
            cat_va_preds += model_cat.predict(X_va) / len(seeds)
            cat_te_preds += model_cat.predict(X_test) / len(seeds)
        oof_cat[va_idx] = cat_va_preds
        test_cat += cat_te_preds / num_folds

        # PyTorch TabularResNet multi-seed bagging
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)

        nn_va_preds = np.zeros(len(X_va))
        nn_te_preds = np.zeros(num_test)

        for s in seeds:
            seed_everything(s)
            train_ds = TensorDataset(
                torch.tensor(X_tr_sc, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
            )
            val_ds = TensorDataset(
                torch.tensor(X_va_sc, dtype=torch.float32),
                torch.tensor(y_va, dtype=torch.float32),
            )

            train_loader = DataLoader(
                train_ds, batch_size=256, shuffle=True, drop_last=False
            )
            val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

            nn_model = TabularResNet(
                in_features=X.shape[1], hidden_dim=128, num_blocks=2, dropout_rate=0.15
            ).to(device)
            optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=35, eta_min=1e-5
            )
            criterion = nn.SmoothL1Loss(beta=1.0)

            best_val_loss = float("inf")
            best_nn_preds = None
            best_nn_test_preds = None

            for epoch in range(35):
                nn_model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = nn_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                nn_model.eval()
                val_preds_list = []
                with torch.no_grad():
                    for bx, _ in val_loader:
                        bx = bx.to(device)
                        val_preds_list.append(nn_model(bx).cpu().numpy())
                val_preds = np.concatenate(val_preds_list)
                val_loss = np.sqrt(mean_squared_error(y_va, val_preds))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_nn_preds = val_preds

                    with torch.no_grad():
                        te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
                        best_nn_test_preds = nn_model(te_tensor).cpu().numpy()

            nn_va_preds += best_nn_preds / len(seeds)
            nn_te_preds += best_nn_test_preds / len(seeds)

        oof_nn[va_idx] = nn_va_preds
        test_nn += nn_te_preds / num_folds

    # Lower bound clipping
    oof_lgb = np.maximum(0, oof_lgb)
    oof_xgb = np.maximum(0, oof_xgb)
    oof_cat = np.maximum(0, oof_cat)
    oof_nn = np.maximum(0, oof_nn)

    test_lgb = np.maximum(0, test_lgb)
    test_xgb = np.maximum(0, test_xgb)
    test_cat = np.maximum(0, test_cat)
    test_nn = np.maximum(0, test_nn)

    oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
    test_stack = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])

    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
    meta_model.fit(oof_stack, y)

    final_oof = meta_model.predict(oof_stack)
    final_oof = np.maximum(0, final_oof)
    final_rmse = np.sqrt(mean_squared_error(y, final_oof))

    final_test = meta_model.predict(test_stack)
    final_test = np.maximum(0, final_test)

    submission_df = pd.DataFrame({"id": test_ids.astype(int), "Recovery": final_test})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
