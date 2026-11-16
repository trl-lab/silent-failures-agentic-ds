import os
import re
import warnings
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_and_preprocess_features():
    input_dir = "./input"
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Train or test parquet files not found in ./input directory."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "funding"
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' missing from train data.")

    # Preserve or assign row index / id
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    # Extract target and target transformation (log1p for right-skewed funding)
    y_raw = np.maximum(0, train_df[target_col].values)
    y_log = np.log1p(y_raw)

    # 5-Fold split based on target log-quantiles
    n_splits = 5
    num_bins = min(10, max(2, len(y_log) // 100))
    bins = pd.qcut(y_log, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, bins)):
        fold_assignments[val_idx] = fold

    train_df["fold"] = fold_assignments

    # Combine for unified non-leaking transformations
    df_all = pd.concat(
        [
            train_df.assign(is_train=1),
            test_df.assign(is_train=0, fold=-1, funding=np.nan),
        ],
        ignore_index=True,
    )

    # Automatic Column Classification
    date_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    exclude_cols = {"id", target_col, "fold", "is_train"}

    for col in df_all.columns:
        if col in exclude_cols:
            continue
        col_dtype = df_all[col].dtype
        col_sample = df_all[col].dropna().astype(str)

        if pd.api.types.is_datetime64_any_dtype(col_dtype) or (
            col_sample.str.contains(r"^\d{4}-\d{2}-\d{2}", regex=True).mean() > 0.5
            if len(col_sample) > 0
            else False
        ):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(col_dtype):
            num_cols.append(col)
        else:
            avg_words = (
                col_sample.str.split().str.len().mean() if len(col_sample) > 0 else 0
            )
            if avg_words > 3:
                text_cols.append(col)
            else:
                cat_cols.append(col)

    # Temporal Feature Engineering
    date_parsed_cols = []
    for dcol in date_cols:
        parsed_series = pd.to_datetime(df_all[dcol], errors="coerce")
        df_all[f"{dcol}_year"] = parsed_series.dt.year
        df_all[f"{dcol}_month"] = parsed_series.dt.month
        df_all[f"{dcol}_quarter"] = parsed_series.dt.quarter
        df_all[f"{dcol}_dayofweek"] = parsed_series.dt.dayofweek
        df_all[f"{dcol}_dayofyear"] = parsed_series.dt.dayofyear
        date_parsed_cols.append(parsed_series)

    # Extract temporal intervals / gaps between date columns
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            diff_days = (
                date_parsed_cols[j] - date_parsed_cols[i]
            ).dt.total_seconds() / 86400.0
            df_all[f"diff_days_{col2}_minus_{col1}"] = diff_days

    # Categorical Feature Engineering & Out-of-Fold Target Encoding
    train_mask = df_all["is_train"] == 1
    test_mask = df_all["is_train"] == 0

    for ccol in cat_cols:
        df_all[ccol] = df_all[ccol].fillna("Missing").astype(str)
        # Frequency Encoding (computed strictly on train set)
        freq_map = df_all.loc[train_mask, ccol].value_counts(normalize=True).to_dict()
        df_all[f"{ccol}_freq"] = df_all[ccol].map(freq_map).fillna(0)

        # Categorical Label Encoding
        categories = {k: i for i, k in enumerate(df_all[ccol].unique())}
        df_all[f"{ccol}_code"] = df_all[ccol].map(categories).astype(int)

        # Out-of-fold Target Encoding on Log Target with additive smoothing
        smooth_weight = 10.0
        global_log_mean = y_log.mean()

        te_col_name = f"{ccol}_te_log"
        df_all[te_col_name] = global_log_mean

        # Assign target log to df_all for group aggregation
        df_all.loc[train_mask, "_target_log"] = y_log

        # Train OOF computation
        for fold in range(n_splits):
            fold_tr_mask = (df_all["is_train"] == 1) & (df_all["fold"] != fold)
            fold_val_mask = (df_all["is_train"] == 1) & (df_all["fold"] == fold)

            grp_stats = df_all.loc[fold_tr_mask].groupby(ccol)["_target_log"].agg(["count", "mean"])
            te_map = (
                (grp_stats["count"] * grp_stats["mean"] + smooth_weight * global_log_mean)
                / (grp_stats["count"] + smooth_weight)
            ).to_dict()
            df_all.loc[fold_val_mask, te_col_name] = (
                df_all.loc[fold_val_mask, ccol].map(te_map).fillna(global_log_mean)
            )

        # Test set TE mapping using full train set statistics
        full_grp_stats = df_all.loc[train_mask].groupby(ccol)["_target_log"].agg(["count", "mean"])
        full_train_stats = (
            (full_grp_stats["count"] * full_grp_stats["mean"] + smooth_weight * global_log_mean)
            / (full_grp_stats["count"] + smooth_weight)
        ).to_dict()
        df_all.drop(columns=["_target_log"], inplace=True, errors="ignore")
        df_all.loc[test_mask, te_col_name] = (
            df_all.loc[test_mask, ccol].map(full_train_stats).fillna(global_log_mean)
        )

    # Group Aggregation Statistics (fit strictly on train set to avoid leakage)
    agg_num_cols = [c for c in num_cols if c in df_all.columns]
    for ccol in cat_cols:
        for ncol in agg_num_cols[:10]:
            stats = df_all.loc[train_mask].groupby(ccol)[ncol].agg(["mean", "std", "min", "max"])
            df_all[f"{ccol}_{ncol}_mean"] = df_all[ccol].map(stats["mean"])
            df_all[f"{ccol}_{ncol}_std"] = df_all[ccol].map(stats["std"]).fillna(0)
            df_all[f"{ccol}_{ncol}_min"] = df_all[ccol].map(stats["min"])
            df_all[f"{ccol}_{ncol}_max"] = df_all[ccol].map(stats["max"])

    # Text Feature Engineering (Statistics + TF-IDF + Transformer Embeddings + Truncated SVD)
    if text_cols:
        text_combined = df_all[text_cols].astype(str).agg(" ".join, axis=1)
        emb_matrix = None
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            emb_matrix = st_model.encode(text_combined.tolist(), batch_size=64, show_progress_bar=False)
        except Exception:
            try:
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                model.eval()
                embeddings_list = []
                batch_size = 64
                text_list = text_combined.tolist()
                for i in range(0, len(text_list), batch_size):
                    batch_texts = text_list[i : i + batch_size]
                    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        embeddings_list.append(batch_emb)
                emb_matrix = np.vstack(embeddings_list)
            except Exception:
                emb_matrix = None

        if emb_matrix is not None:
            svd_emb = TruncatedSVD(n_components=32, random_state=42)
            svd_emb.fit(emb_matrix[train_mask])
            reduced_emb = svd_emb.transform(emb_matrix)
            for comp in range(reduced_emb.shape[1]):
                df_all[f"text_transformer_emb_{comp}"] = reduced_emb[:, comp]

    for tcol in text_cols:
        df_all[tcol] = df_all[tcol].fillna("").astype(str)
        df_all[f"{tcol}_len"] = df_all[tcol].str.len()
        df_all[f"{tcol}_word_cnt"] = df_all[tcol].str.split().str.len()
        df_all[f"{tcol}_digit_cnt"] = df_all[tcol].apply(
            lambda s: sum(c.isdigit() for c in s)
        )

        tfidf = TfidfVectorizer(
            max_features=2500, stop_words="english", ngram_range=(1, 2)
        )
        svd = TruncatedSVD(n_components=8, random_state=42)

        train_text = df_all.loc[train_mask, tcol]
        all_text = df_all[tcol]

        try:
            tfidf.fit(train_text)
            tfidf_all = tfidf.transform(all_text)
            svd.fit(tfidf.transform(train_text))
            svd_feats = svd.transform(tfidf_all)

            for comp_idx in range(svd_feats.shape[1]):
                col_name = f"{tcol}_svd_{comp_idx}"
                df_all[col_name] = svd_feats[:, comp_idx]
        except Exception:
            pass

    # Numerical Feature Processing & Transformations
    for ncol in num_cols:
        med_val = df_all.loc[train_mask, ncol].median()
        if pd.isna(med_val):
            med_val = 0.0

        df_all[f"{ncol}_isna"] = df_all[ncol].isna().astype(int)
        df_all[ncol] = df_all[ncol].fillna(med_val)

        if (df_all[ncol] >= 0).all():
            skew = df_all.loc[train_mask, ncol].skew()
            if skew > 1.5:
                df_all[f"{ncol}_log1p"] = np.log1p(df_all[ncol])

    # Feature Interactions (products, ratios, differences) for top numerical features
    eps = 1e-5
    if len(num_cols) >= 2:
        top_nums = num_cols[:5]
        for i in range(len(top_nums)):
            for j in range(i + 1, len(top_nums)):
                c1, c2 = top_nums[i], top_nums[j]
                df_all[f"prod_{c1}_{c2}"] = df_all[c1] * df_all[c2]
                df_all[f"diff_{c1}_{c2}"] = df_all[c1] - df_all[c2]
                df_all[f"ratio_{c1}_{c2}"] = df_all[c1] / (df_all[c2].abs() + eps)
                df_all[f"ratio_{c2}_{c1}"] = df_all[c2] / (df_all[c1].abs() + eps)

    # Cross-categorical Frequency Encodings
    if len(cat_cols) >= 2:
        top_cats = cat_cols[:4]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                c1, c2 = top_cats[i], top_cats[j]
                combo = df_all[c1].astype(str) + "_" + df_all[c2].astype(str)
                freq_map = combo[train_mask].value_counts(normalize=True).to_dict()
                df_all[f"freq_{c1}_{c2}"] = combo.map(freq_map).fillna(0)

    # Drop raw date strings, unstructured text, and categorical raw strings
    cols_to_drop = date_cols + text_cols + cat_cols
    df_all_processed = df_all.drop(
        columns=[c for c in cols_to_drop if c in df_all.columns]
    )

    train_processed = (
        df_all_processed[df_all_processed["is_train"] == 1]
        .drop(columns=["is_train"])
        .reset_index(drop=True)
    )
    test_processed = (
        df_all_processed[df_all_processed["is_train"] == 0]
        .drop(columns=["is_train", "funding", "fold"])
        .reset_index(drop=True)
    )

    # Ensure numeric types across all features
    for c in train_processed.columns:
        if train_processed[c].dtype == "object":
            train_processed[c] = pd.to_numeric(
                train_processed[c], errors="coerce"
            ).fillna(0)
            test_processed[c] = pd.to_numeric(
                test_processed[c], errors="coerce"
            ).fillna(0)

    train_processed.to_parquet(
        os.path.join(working_dir, "train_processed.parquet"), index=False
    )
    test_processed.to_parquet(
        os.path.join(working_dir, "test_processed.parquet"), index=False
    )

    return train_processed, test_processed


def run_pipeline():
    train_df, test_df = build_and_preprocess_features()

    target_col = "funding"
    ignore_cols = {"id", target_col, "fold"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X = train_df[feature_cols].copy()
    y_raw = np.maximum(0, train_df[target_col].values)
    y_log = np.log1p(y_raw)
    folds = (
        train_df["fold"].values
        if "fold" in train_df.columns
        else np.zeros(len(train_df))
    )

    X_test = test_df[feature_cols].copy()

    for col in feature_cols:
        fill_val = X[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        X[col] = X[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)

    n_splits = int(np.max(folds)) + 1 if len(np.unique(folds)) > 1 else 5

    oof_lgb_log = np.zeros(len(train_df))
    oof_lgb_raw = np.zeros(len(train_df))
    oof_lgb_tweedie = np.zeros(len(train_df))
    oof_xgb_log = np.zeros(len(train_df))
    oof_xgb_raw = np.zeros(len(train_df))
    oof_cat_log = np.zeros(len(train_df))
    oof_cat_raw = np.zeros(len(train_df))
    oof_ridge_log = np.zeros(len(train_df))
    oof_mlp_log = np.zeros(len(train_df))

    test_lgb_log = np.zeros(len(test_df))
    test_lgb_raw = np.zeros(len(test_df))
    test_lgb_tweedie = np.zeros(len(test_df))
    test_xgb_log = np.zeros(len(test_df))
    test_xgb_raw = np.zeros(len(test_df))
    test_cat_log = np.zeros(len(test_df))
    test_cat_raw = np.zeros(len(test_df))
    test_ridge_log = np.zeros(len(test_df))
    test_mlp_log = np.zeros(len(test_df))

    for fold in range(n_splits):
        val_idx = folds == fold
        tr_idx = ~val_idx

        X_tr, y_tr_raw, y_tr_log = X.iloc[tr_idx], y_raw[tr_idx], y_log[tr_idx]
        X_val, y_val_raw, y_val_log = X.iloc[val_idx], y_raw[val_idx], y_log[val_idx]

        # 1. LightGBM Log-scale Regressor
        model_lgb_log = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        model_lgb_log.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_val_lgb_log = model_lgb_log.predict(X_val)
        pred_test_lgb_log = model_lgb_log.predict(X_test)
        oof_lgb_log[val_idx] = np.expm1(np.maximum(0, pred_val_lgb_log))
        test_lgb_log += np.expm1(np.maximum(0, pred_test_lgb_log)) / n_splits

        # 1b. LightGBM Direct Raw-Scale MSE Regressor
        model_lgb_raw = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        model_lgb_raw.fit(
            X_tr,
            y_tr_raw,
            eval_set=[(X_val, y_val_raw)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb_raw[val_idx] = np.maximum(0, model_lgb_raw.predict(X_val))
        test_lgb_raw += np.maximum(0, model_lgb_raw.predict(X_test)) / n_splits

        # 2. LightGBM Tweedie Raw-scale Regressor
        model_lgb_tweedie = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        model_lgb_tweedie.fit(
            X_tr,
            y_tr_raw,
            eval_set=[(X_val, y_val_raw)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb_tweedie[val_idx] = np.maximum(0, model_lgb_tweedie.predict(X_val))
        test_lgb_tweedie += np.maximum(0, model_lgb_tweedie.predict(X_test)) / n_splits

        # 3. XGBoost Log-scale Regressor
        model_xgb_log = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model_xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
        pred_val_xgb_log = model_xgb_log.predict(X_val)
        pred_test_xgb_log = model_xgb_log.predict(X_test)
        oof_xgb_log[val_idx] = np.expm1(np.maximum(0, pred_val_xgb_log))
        test_xgb_log += np.expm1(np.maximum(0, pred_test_xgb_log)) / n_splits

        # 3b. XGBoost Direct Raw-scale MSE Regressor
        model_xgb_raw = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model_xgb_raw.fit(X_tr, y_tr_raw, eval_set=[(X_val, y_val_raw)], verbose=False)
        oof_xgb_raw[val_idx] = np.maximum(0, model_xgb_raw.predict(X_val))
        test_xgb_raw += np.maximum(0, model_xgb_raw.predict(X_test)) / n_splits

        # 4. CatBoost Log-scale Regressor
        model_cat_log = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
        )
        model_cat_log.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log))
        pred_val_cat_log = model_cat_log.predict(X_val)
        pred_test_cat_log = model_cat_log.predict(X_test)
        oof_cat_log[val_idx] = np.expm1(np.maximum(0, pred_val_cat_log))
        test_cat_log += np.expm1(np.maximum(0, pred_test_cat_log)) / n_splits

        # 4b. CatBoost Direct Raw-scale RMSE Regressor
        model_cat_raw = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
        )
        model_cat_raw.fit(X_tr, y_tr_raw, eval_set=(X_val, y_val_raw))
        oof_cat_raw[val_idx] = np.maximum(0, model_cat_raw.predict(X_val))
        test_cat_raw += np.maximum(0, model_cat_raw.predict(X_test)) / n_splits

        # 5. Ridge Log-scale Regressor (strictly scaled per fold)
        scaler_ridge = StandardScaler()
        X_tr_ridge = scaler_ridge.fit_transform(X_tr)
        X_val_ridge = scaler_ridge.transform(X_val)
        X_test_ridge = scaler_ridge.transform(X_test)

        model_ridge = Ridge(alpha=10.0, random_state=42 + fold)
        model_ridge.fit(X_tr_ridge, y_tr_log)
        pred_val_ridge_log = model_ridge.predict(X_val_ridge)
        pred_test_ridge_log = model_ridge.predict(X_test_ridge)
        oof_ridge_log[val_idx] = np.expm1(np.maximum(0, pred_val_ridge_log))
        test_ridge_log += np.expm1(np.maximum(0, pred_test_ridge_log)) / n_splits

        # 6. PyTorch Tabular Neural Network (MLP)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        tr_ds = TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr_log, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_log, dtype=torch.float32))

        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        mlp_model = TabularMLP(input_dim=X_tr.shape[1])
        optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 80
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_model_weights = None
        patience, patience_counter = 15, 0

        for epoch in range(epochs):
            mlp_model.train()
            for bx, by in tr_loader:
                optimizer.zero_grad()
                out = mlp_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = mlp_model(bx)
                    val_loss += criterion(out, by).item() * len(by)
            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(mlp_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_weights is not None:
            mlp_model.load_state_dict(best_model_weights)
        mlp_model.eval()

        with torch.no_grad():
            mlp_val_pred = mlp_model(torch.tensor(X_val_scaled, dtype=torch.float32)).numpy()
            mlp_test_pred = mlp_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

        oof_mlp_log[val_idx] = np.expm1(np.maximum(0, mlp_val_pred))
        test_mlp_log += np.expm1(np.maximum(0, mlp_test_pred)) / n_splits

    oof_matrix = np.column_stack(
        [
            oof_lgb_log,
            oof_lgb_raw,
            oof_lgb_tweedie,
            oof_xgb_log,
            oof_xgb_raw,
            oof_cat_log,
            oof_cat_raw,
            oof_ridge_log,
            oof_mlp_log,
        ]
    )

    test_matrix = np.column_stack(
        [
            test_lgb_log,
            test_lgb_raw,
            test_lgb_tweedie,
            test_xgb_log,
            test_xgb_raw,
            test_cat_log,
            test_cat_raw,
            test_ridge_log,
            test_mlp_log,
        ]
    )

    # SLSQP Constrained Optimization on Raw Target RMSE with Flexible Scale Learning
    n_models = oof_matrix.shape[1]

    def objective(params):
        weights = params[:n_models]
        scale = params[n_models]
        power_scale = params[n_models + 1]
        pred_blend = np.dot(oof_matrix, weights)
        pred_calibrated = np.maximum(0, pred_blend * scale) ** power_scale
        return np.sqrt(mean_squared_error(y_raw, pred_calibrated))

    init_weights = [1.0 / n_models] * n_models
    init_params = np.array(init_weights + [1.0, 1.0])
    bounds = [(0, 1)] * n_models + [(0.1, 5.0), (0.5, 1.5)]
    constraints = {"type": "eq", "fun": lambda p: np.sum(p[:n_models]) - 1.0}

    opt_res = minimize(
        objective, init_params, method="SLSQP", bounds=bounds, constraints=constraints
    )

    opt_weights = opt_res.x[:n_models]
    opt_scale = opt_res.x[n_models]
    opt_power = opt_res.x[n_models + 1]

    oof_final = np.maximum(0, np.dot(oof_matrix, opt_weights) * opt_scale) ** opt_power
    final_score = np.sqrt(mean_squared_error(y_raw, oof_final))

    # Compute and format submission
    test_final = np.maximum(0, np.dot(test_matrix, opt_weights) * opt_scale) ** opt_power
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")

    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )
    sub_df = pd.DataFrame({"id": test_ids, "funding": test_final})
    sub_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    run_pipeline()