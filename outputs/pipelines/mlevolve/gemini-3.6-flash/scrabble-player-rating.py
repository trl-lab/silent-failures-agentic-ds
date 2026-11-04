import os
import random
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

import catboost as cb
import lightgbm as lgb
import xgboost as xgb


# ==============================================================================
# 0. REPRODUCIBILITY AND SEEDING
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)


# ==============================================================================
# 1. FEATURE ENGINEERING & DATA PREPARATION
# ==============================================================================
def load_and_preprocess_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    target_col = "rating"
    train_df["is_test"] = 0
    test_df["is_test"] = 1

    # Combine datasets for non-target feature engineering
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Bot Detection
    bot_patterns = r"(?i)(?:bot|hastybot|betterbot|steebot|bb-8|scrabblebot)"
    full_df["is_bot"] = (
        full_df["nickname"]
        .astype(str)
        .str.contains(bot_patterns, regex=True)
        .astype(int)
    )

    # Pairwise Game Features
    game_grouped = full_df.groupby("game_id")
    game_agg = game_grouped.agg(
        game_total_score=("score", "sum"),
        game_max_score=("score", "max"),
        game_min_score=("score", "min"),
        game_player_count=("nickname", "count"),
    ).reset_index()

    full_df = full_df.merge(game_agg, on="game_id", how="left")

    full_df["opp_score"] = full_df["game_total_score"] - full_df["score"]
    full_df["score_diff"] = full_df["score"] - full_df["opp_score"]
    full_df["score_ratio"] = (full_df["score"] + 1.0) / (full_df["opp_score"] + 1.0)
    full_df["is_win"] = (full_df["score"] > full_df["opp_score"]).astype(int)
    full_df["is_draw"] = (full_df["score"] == full_df["opp_score"]).astype(int)
    full_df["score_pct_of_game"] = (full_df["score"] + 1.0) / (
        full_df["game_total_score"] + 2.0
    )

    # Opponent nickname lookup
    game_players = full_df[["game_id", "nickname", "score", "is_bot"]].copy()
    game_players_opp = game_players.merge(
        game_players, on="game_id", suffixes=("", "_opp")
    )
    game_players_opp = game_players_opp[
        game_players_opp["nickname"] != game_players_opp["nickname_opp"]
    ]
    game_players_opp = game_players_opp.drop_duplicates(subset=["game_id", "nickname"])
    game_players_opp = game_players_opp[
        ["game_id", "nickname", "nickname_opp", "is_bot_opp"]
    ]

    full_df = full_df.merge(game_players_opp, on=["game_id", "nickname"], how="left")
    full_df["nickname_opp"] = full_df["nickname_opp"].fillna("Unknown")
    full_df["is_bot_opp"] = full_df["is_bot_opp"].fillna(0).astype(int)

    # Player Historical Aggregations (Non-target features & score quantiles)
    player_stats = (
        full_df.groupby("nickname")
        .agg(
            player_game_count=("game_id", "count"),
            player_mean_score=("score", "mean"),
            player_std_score=("score", "std"),
            player_min_score=("score", "min"),
            player_max_score=("score", "max"),
            player_median_score=("score", "median"),
            player_q25_score=("score", lambda x: np.percentile(x, 25)),
            player_q75_score=("score", lambda x: np.percentile(x, 75)),
            player_win_rate=("is_win", "mean"),
        )
        .reset_index()
    )
    player_stats["player_std_score"] = player_stats["player_std_score"].fillna(0)
    player_stats["player_iqr_score"] = (
        player_stats["player_q75_score"] - player_stats["player_q25_score"]
    )

    full_df = full_df.merge(player_stats, on="nickname", how="left")

    opp_player_stats = player_stats.rename(
        columns=lambda x: f"opp_{x}" if x != "nickname" else "nickname_opp"
    )
    full_df = full_df.merge(opp_player_stats, on="nickname_opp", how="left")

    opp_stat_cols = [c for c in full_df.columns if c.startswith("opp_player_")]
    for c in opp_stat_cols:
        full_df[c] = full_df[c].fillna(0)

    full_df["score_vs_player_mean"] = full_df["score"] - full_df["player_mean_score"]
    full_df["score_vs_player_std"] = full_df["score_vs_player_mean"] / (
        full_df["player_std_score"] + 1e-5
    )
    full_df["score_vs_opp_mean"] = full_df["score"] - full_df["opp_player_mean_score"]

    # Strength of Schedule (SOS) Features
    player_sos = (
        full_df.groupby("nickname")
        .agg(
            player_opp_mean_score=("opp_player_mean_score", "mean"),
            player_opp_win_rate=("opp_player_win_rate", "mean"),
        )
        .reset_index()
    )
    full_df = full_df.merge(player_sos, on="nickname", how="left")

    opp_player_sos = player_sos.rename(
        columns={
            "nickname": "nickname_opp",
            "player_opp_mean_score": "opp_player_opp_mean_score",
            "player_opp_win_rate": "opp_player_opp_win_rate",
        }
    )
    full_df = full_df.merge(opp_player_sos, on="nickname_opp", how="left")
    full_df["opp_player_opp_mean_score"] = full_df["opp_player_opp_mean_score"].fillna(0)
    full_df["opp_player_opp_win_rate"] = full_df["opp_player_opp_win_rate"].fillna(0)

    full_df["player_opp_mean_score_diff"] = (
        full_df["player_opp_mean_score"] - full_df["opp_player_opp_mean_score"]
    )
    full_df["player_opp_win_rate_diff"] = (
        full_df["player_opp_win_rate"] - full_df["opp_player_opp_win_rate"]
    )

    # Head-to-Head (H2H) Features
    h2h_stats = (
        full_df.groupby(["nickname", "nickname_opp"])
        .agg(
            h2h_count=("game_id", "count"),
            h2h_player_win_rate=("is_win", "mean"),
            h2h_mean_score_diff=("score_diff", "mean"),
        )
        .reset_index()
    )
    full_df = full_df.merge(h2h_stats, on=["nickname", "nickname_opp"], how="left")
    full_df["h2h_count"] = full_df["h2h_count"].fillna(0)
    full_df["h2h_player_win_rate"] = full_df["h2h_player_win_rate"].fillna(0.5)
    full_df["h2h_mean_score_diff"] = full_df["h2h_mean_score_diff"].fillna(0)

    # Relative player vs opponent stat differentials
    full_df["expected_score_diff"] = full_df["player_mean_score"] - full_df["opp_player_mean_score"]
    full_df["player_win_rate_diff"] = full_df["player_win_rate"] - full_df["opp_player_win_rate"]
    full_df["score_ratio_rel_exp"] = (full_df["score"] + 1.0) / (full_df["opp_player_mean_score"] + 1.0)
    full_df["score_diff_vs_exp_diff"] = full_df["score_diff"] - full_df["expected_score_diff"]
    full_df["player_game_count_diff"] = full_df["player_game_count"] - full_df["opp_player_game_count"]
    full_df["player_max_score_diff"] = full_df["player_max_score"] - full_df["opp_player_max_score"]
    full_df["player_min_score_diff"] = full_df["player_min_score"] - full_df["opp_player_min_score"]

    # Categorical Frequency Encodings
    for cat_col in ["nickname", "nickname_opp"]:
        freq = full_df[cat_col].value_counts().to_dict()
        full_df[f"{cat_col}_freq"] = full_df[cat_col].map(freq).fillna(0)

    # Split back into Train and Test
    train_processed = full_df[full_df["is_test"] == 0].copy().reset_index(drop=True)
    test_processed = full_df[full_df["is_test"] == 1].copy().reset_index(drop=True)

    train_processed = train_processed.drop(columns=["is_test"])
    test_processed = test_processed.drop(columns=["is_test", target_col])

    return train_processed, test_processed


# ==============================================================================
# 2. PYTORCH DATASET AND MODEL ARCHITECTURE
# ==============================================================================
class TabularDataset(Dataset):
    def __init__(self, num_feats, cat_feats, targets=None):
        self.num_feats = torch.tensor(num_feats, dtype=torch.float32)
        self.cat_feats = torch.tensor(cat_feats, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.num_feats)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.num_feats[idx], self.cat_feats[idx], self.targets[idx]
        return self.num_feats[idx], self.cat_feats[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout_rate=0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.dropout1(self.act1(self.bn1(self.fc1(x))))
        out = self.dropout2(self.act2(self.bn2(self.fc2(out))))
        return out + residual


class ScrabbleSkillMLP(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        num_categories=3000,
        embedding_dim=32,
        hidden_dim=256,
        num_blocks=2,
        dropout_rate=0.15,
    ):
        super().__init__()
        self.has_embeddings = num_categories is not None and num_categories > 0
        if self.has_embeddings:
            self.nickname_embed = nn.Embedding(
                num_categories, embedding_dim, padding_idx=0
            )
            self.opp_embed = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
            total_in_dim = num_numerical_features + 2 * embedding_dim
        else:
            total_in_dim = num_numerical_features

        self.input_layer = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, num_x, cat_x=None):
        if self.has_embeddings and cat_x is not None:
            nick_emb = self.nickname_embed(cat_x[:, 0])
            opp_emb = self.opp_embed(cat_x[:, 1])
            x = torch.cat([num_x, nick_emb, opp_emb], dim=1)
        else:
            x = num_x

        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)

        out = self.head(x)
        return out.squeeze(-1)


# ==============================================================================
# 3. MAIN TRAINING, CROSS-VALIDATION & ENSEMBLE PIPELINE
# ==============================================================================
def main():
    train_df, test_df = load_and_preprocess_data()

    target_col = "rating"
    group_col = "game_id"

    # Categorical Nickname Indexing for Neural Network Entity Embeddings
    all_nicknames = (
        set(train_df["nickname"].unique())
        .union(set(train_df["nickname_opp"].unique()))
        .union(set(test_df["nickname"].unique()))
        .union(set(test_df["nickname_opp"].unique()))
    )
    nick2idx = {nick: idx + 1 for idx, nick in enumerate(sorted(list(all_nicknames)))}
    num_categories = len(nick2idx) + 2

    train_cat_feats = np.column_stack(
        [
            train_df["nickname"].map(nick2idx).fillna(0).astype(int).values,
            train_df["nickname_opp"].map(nick2idx).fillna(0).astype(int).values,
        ]
    )
    test_cat_feats = np.column_stack(
        [
            test_df["nickname"].map(nick2idx).fillna(0).astype(int).values,
            test_df["nickname_opp"].map(nick2idx).fillna(0).astype(int).values,
        ]
    )

    # Numerical Base Feature Selection
    exclude_cols = [target_col, group_col, "id", "nickname", "nickname_opp"]
    base_num_cols = [c for c in train_df.columns if c not in exclude_cols]

    for c in base_num_cols:
        med_val = train_df[c].median()
        train_df[c] = train_df[c].fillna(med_val)
        test_df[c] = test_df[c].fillna(med_val)

    y_train = train_df[target_col].values
    groups = train_df[group_col].values

    n_samples = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)
    oof_nn = np.zeros(n_samples)

    preds_lgb = np.zeros(n_test)
    preds_xgb = np.zeros(n_test)
    preds_cat = np.zeros(n_test)
    preds_nn = np.zeros(n_test)

    gkf = GroupKFold(n_splits=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./working", exist_ok=True)

    # Prepare Sparse Indicator Matrices for Ridge Regression Latent Skill
    all_nicks_sorted = sorted(list(all_nicknames))
    nick2sparse = {nick: i for i, nick in enumerate(all_nicks_sorted)}
    K_sparse = len(nick2sparse)

    def make_ridge_matrix(df):
        n = len(df)
        row_idx = np.arange(n)
        nick_idx = df["nickname"].map(nick2sparse).fillna(0).astype(int).values
        opp_idx = df["nickname_opp"].map(nick2sparse).fillna(0).astype(int).values
        data = np.ones(n, dtype=np.float32)
        X_nick = csr_matrix((data, (row_idx, nick_idx)), shape=(n, K_sparse))
        X_opp = csr_matrix((data, (row_idx, opp_idx)), shape=(n, K_sparse))
        return hstack([X_nick, X_opp]).tocsr()

    X_train_ridge_full = make_ridge_matrix(train_df)
    X_test_ridge_full = make_ridge_matrix(test_df)

    tr_nick_indices = train_df["nickname"].map(nick2sparse).fillna(0).astype(int).values
    tr_opp_indices = train_df["nickname_opp"].map(nick2sparse).fillna(0).astype(int).values
    te_nick_indices = test_df["nickname"].map(nick2sparse).fillna(0).astype(int).values
    te_opp_indices = test_df["nickname_opp"].map(nick2sparse).fillna(0).astype(int).values

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(train_df, y_train, groups=groups)
    ):
        # 1. Out-of-fold Ridge Regression Latent Skill Estimation
        X_tr_ridge = X_train_ridge_full[tr_idx]
        y_tr_fold = y_train[tr_idx]

        ridge = Ridge(alpha=10.0, random_state=42 + fold)
        ridge.fit(X_tr_ridge, y_tr_fold)

        ridge_coef = ridge.coef_
        ridge_intercept = ridge.intercept_

        p_ridge_tr = ridge_intercept + ridge_coef[tr_nick_indices]
        opp_ridge_tr = ridge_intercept + ridge_coef[K_sparse + tr_opp_indices]
        ridge_pred_tr = ridge.predict(X_train_ridge_full)

        p_ridge_te = ridge_intercept + ridge_coef[te_nick_indices]
        opp_ridge_te = ridge_intercept + ridge_coef[K_sparse + te_opp_indices]
        ridge_pred_te = ridge.predict(X_test_ridge_full)

        # 2. Bayesian Smoothed Target Statistics (m-estimate)
        tr_df_fold = train_df.iloc[tr_idx]
        g_mean = tr_df_fold[target_col].mean()
        g_min = tr_df_fold[target_col].min()
        g_max = tr_df_fold[target_col].max()
        g_std = tr_df_fold[target_col].std()
        if np.isnan(g_std):
            g_std = 0.0

        stats = tr_df_fold.groupby("nickname")[target_col].agg(
            n="count", mean="mean", min="min", max="max", std="std"
        )
        stats["std"] = stats["std"].fillna(g_std)

        m_param = 10.0
        stats["s_mean"] = (stats["n"] * stats["mean"] + m_param * g_mean) / (stats["n"] + m_param)
        stats["s_min"] = (stats["n"] * stats["min"] + m_param * g_min) / (stats["n"] + m_param)
        stats["s_max"] = (stats["n"] * stats["max"] + m_param * g_max) / (stats["n"] + m_param)
        stats["s_std"] = (stats["n"] * stats["std"] + m_param * g_std) / (stats["n"] + m_param)

        d_mean = stats["s_mean"].to_dict()
        d_min = stats["s_min"].to_dict()
        d_max = stats["s_max"].to_dict()
        d_std = stats["s_std"].to_dict()

        p_bm_tr = train_df["nickname"].map(d_mean).fillna(g_mean).values
        p_bmin_tr = train_df["nickname"].map(d_min).fillna(g_min).values
        p_bmax_tr = train_df["nickname"].map(d_max).fillna(g_max).values
        p_bstd_tr = train_df["nickname"].map(d_std).fillna(g_std).values

        opp_bm_tr = train_df["nickname_opp"].map(d_mean).fillna(g_mean).values
        opp_bmin_tr = train_df["nickname_opp"].map(d_min).fillna(g_min).values
        opp_bmax_tr = train_df["nickname_opp"].map(d_max).fillna(g_max).values
        opp_bstd_tr = train_df["nickname_opp"].map(d_std).fillna(g_std).values

        p_bm_te = test_df["nickname"].map(d_mean).fillna(g_mean).values
        p_bmin_te = test_df["nickname"].map(d_min).fillna(g_min).values
        p_bmax_te = test_df["nickname"].map(d_max).fillna(g_max).values
        p_bstd_te = test_df["nickname"].map(d_std).fillna(g_std).values

        opp_bm_te = test_df["nickname_opp"].map(d_mean).fillna(g_mean).values
        opp_bmin_te = test_df["nickname_opp"].map(d_min).fillna(g_min).values
        opp_bmax_te = test_df["nickname_opp"].map(d_max).fillna(g_max).values
        opp_bstd_te = test_df["nickname_opp"].map(d_std).fillna(g_std).values

        target_feats_tr = np.column_stack([
            p_ridge_tr, opp_ridge_tr, ridge_pred_tr,
            p_bm_tr, p_bmin_tr, p_bmax_tr, p_bstd_tr,
            opp_bm_tr, opp_bmin_tr, opp_bmax_tr, opp_bstd_tr
        ])

        target_feats_te = np.column_stack([
            p_ridge_te, opp_ridge_te, ridge_pred_te,
            p_bm_te, p_bmin_te, p_bmax_te, p_bstd_te,
            opp_bm_te, opp_bmin_te, opp_bmax_te, opp_bstd_te
        ])

        X_num_all = np.column_stack([train_df[base_num_cols].values, target_feats_tr])
        X_num_test_fold = np.column_stack([test_df[base_num_cols].values, target_feats_te])

        X_tr, y_tr = X_num_all[tr_idx], y_train[tr_idx]
        X_va, y_va = X_num_all[val_idx], y_train[val_idx]

        cat_tr = train_cat_feats[tr_idx]
        cat_va = train_cat_feats[val_idx]

        # LightGBM Regressor (L1 / MAE Objective)
        lgb_model = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va)
        preds_lgb += lgb_model.predict(X_num_test_fold) / 5.0

        # XGBoost Regressor (Absolute Error Objective)
        xgb_model = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
            tree_method="hist",
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_va)
        preds_xgb += xgb_model.predict(X_num_test_fold) / 5.0

        # CatBoost Regressor (MAE Objective)
        cb_model = cb.CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = cb_model.predict(X_va)
        preds_cat += cb_model.predict(X_num_test_fold) / 5.0

        # PyTorch Neural Network (ScrabbleSkillMLP)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_num_test_fold)

        train_ds = TabularDataset(X_tr_sc, cat_tr, y_tr)
        val_ds = TabularDataset(X_va_sc, cat_va, y_va)
        test_ds = TabularDataset(X_te_sc, test_cat_feats)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

        nn_model = ScrabbleSkillMLP(
            num_numerical_features=X_tr.shape[1],
            num_categories=num_categories,
            embedding_dim=32,
            hidden_dim=256,
            num_blocks=2,
            dropout_rate=0.15,
        ).to(device)

        criterion = nn.L1Loss()
        optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_mae = float("inf")
        best_val_preds = None

        for epoch in range(12):
            nn_model.train()
            for num_x_b, cat_x_b, y_b in train_loader:
                num_x_b, cat_x_b, y_b = (
                    num_x_b.to(device),
                    cat_x_b.to(device),
                    y_b.to(device),
                )
                optimizer.zero_grad()
                out = nn_model(num_x_b, cat_x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for num_x_b, cat_x_b, _ in val_loader:
                    num_x_b, cat_x_b = num_x_b.to(device), cat_x_b.to(device)
                    val_preds_list.extend(nn_model(num_x_b, cat_x_b).cpu().numpy())

            val_preds_arr = np.array(val_preds_list)
            val_mae_ep = mean_absolute_error(y_va, val_preds_arr)

            if val_mae_ep < best_val_mae:
                best_val_mae = val_mae_ep
                best_val_preds = val_preds_arr
                torch.save(nn_model.state_dict(), f"./working/nn_best_fold_{fold}.pt")

        oof_nn[val_idx] = best_val_preds

        nn_model.load_state_dict(torch.load(f"./working/nn_best_fold_{fold}.pt"))
        nn_model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for num_x_b, cat_x_b in test_loader:
                num_x_b, cat_x_b = num_x_b.to(device), cat_x_b.to(device)
                fold_test_preds.extend(nn_model(num_x_b, cat_x_b).cpu().numpy())
        preds_nn += np.array(fold_test_preds) / 5.0

    # L1 Ensemble Blending Optimization
    oofs_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
    preds_matrix = np.column_stack([preds_lgb, preds_xgb, preds_cat, preds_nn])

    def ensemble_loss(weights):
        weights = weights / np.sum(weights)
        blend_oof = np.dot(oofs_matrix, weights)
        return mean_absolute_error(y_train, blend_oof)

    init_weights = np.ones(4) / 4.0
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    opt_res = minimize(
        ensemble_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_weights = opt_res.x / np.sum(opt_res.x)

    final_oof_preds = np.dot(oofs_matrix, opt_weights)
    final_cv_mae = mean_absolute_error(y_train, final_oof_preds)

    final_test_preds = np.dot(preds_matrix, opt_weights)

    # Save Submission File
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame({"id": test_df["id"].astype(int), "rating": final_test_preds})
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_cv_mae:.5f}")


if __name__ == "__main__":
    main()
