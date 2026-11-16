import os
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.act(x + self.block(x)))


class TabularResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_blocks=2, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)


def run_pipeline():
    # 1. Setup directories and file paths
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print("Loading raw parquet data...")
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    target_col = "TARGET"
    id_col = "ID" if "ID" in train_df.columns else None

    y_train = train_df[target_col].values
    cols_to_drop = [c for c in [target_col, id_col] if c in train_df.columns]

    X_train = train_df.drop(columns=cols_to_drop).copy()
    X_test = test_df.drop(
        columns=[id_col] if id_col and id_col in test_df.columns else []
    ).copy()

    # Align columns between train and test
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # 2. Remove constant features (variance == 0 on train) and quasi-constant features (>99.9% single value)
    constant_features = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    if constant_features:
        X_train.drop(columns=constant_features, inplace=True)
        X_test.drop(columns=constant_features, inplace=True)
        print(f"Removed {len(constant_features)} constant features.")

    quasi_constant_features = []
    for col in X_train.columns:
        top_freq = X_train[col].value_counts(normalize=True).iloc[0]
        if top_freq >= 0.999:
            quasi_constant_features.append(col)
    if quasi_constant_features:
        X_train.drop(columns=quasi_constant_features, inplace=True)
        X_test.drop(columns=quasi_constant_features, inplace=True)
        print(f"Removed {len(quasi_constant_features)} quasi-constant features.")

    # 3. Remove duplicate features based on train set
    train_t = X_train.T
    duplicate_features = train_t[train_t.duplicated()].index.tolist()
    if duplicate_features:
        X_train.drop(columns=duplicate_features, inplace=True)
        X_test.drop(columns=duplicate_features, inplace=True)
        print(f"Removed {len(duplicate_features)} duplicate features.")

    feature_cols = list(X_train.columns)

    # 4. Feature Engineering
    print("Engineering domain, aggregation, lifecycle, and sparsity features...")
    for df in [X_train, X_test]:
        # Domain fix for var3 (nationality missing code -999999 -> 2)
        if "var3" in df.columns:
            df["var3_is_missing"] = (df["var3"] == -999999).astype(np.int8)
            df["var3"] = df["var3"].replace(-999999, 2)

        # Domain fix for var38 (mortgage default value spike at 117310.979016494)
        if "var38" in df.columns:
            df["var38_is_spike"] = (
                np.abs(df["var38"] - 117310.979016494) < 1e-4
            ).astype(np.int8)
            df["var38_log"] = np.log1p(np.maximum(df["var38"], 0))

        # Row-level sparsity and value counts across all original feature columns
        df["count_zeros"] = (df[feature_cols] == 0).sum(axis=1)
        df["count_positive"] = (df[feature_cols] > 0).sum(axis=1)
        df["count_negative"] = (df[feature_cols] < 0).sum(axis=1)
        df["zero_ratio"] = df["count_zeros"] / len(feature_cols)

        # Family aggregations: saldo (balances)
        saldo_cols = [
            c for c in feature_cols if c.startswith("saldo_") and c in df.columns
        ]
        if saldo_cols:
            df["saldo_sum"] = df[saldo_cols].sum(axis=1)
            df["saldo_mean"] = df[saldo_cols].mean(axis=1)
            df["saldo_std"] = df[saldo_cols].std(axis=1).fillna(0)
            df["saldo_max"] = df[saldo_cols].max(axis=1)
            df["saldo_min"] = df[saldo_cols].min(axis=1)
            df["saldo_nonzeros"] = (df[saldo_cols] != 0).sum(axis=1)
            df["saldo_zeros"] = (df[saldo_cols] == 0).sum(axis=1)

        # Family aggregations: imp (monetary transaction amounts)
        imp_cols = [c for c in feature_cols if c.startswith("imp_") and c in df.columns]
        if imp_cols:
            df["imp_sum"] = df[imp_cols].sum(axis=1)
            df["imp_mean"] = df[imp_cols].mean(axis=1)
            df["imp_max"] = df[imp_cols].max(axis=1)
            df["imp_nonzeros"] = (df[imp_cols] != 0).sum(axis=1)
            df["imp_zeros"] = (df[imp_cols] == 0).sum(axis=1)

        # Family aggregations: num (counts of products/transactions)
        num_cols = [c for c in feature_cols if c.startswith("num_") and c in df.columns]
        if num_cols:
            df["num_sum"] = df[num_cols].sum(axis=1)
            df["num_mean"] = df[num_cols].mean(axis=1)
            df["num_max"] = df[num_cols].max(axis=1)
            df["num_zeros"] = (df[num_cols] == 0).sum(axis=1)

        # Family aggregations: ind (binary indicator flags for products)
        ind_cols = [c for c in feature_cols if c.startswith("ind_") and c in df.columns]
        if ind_cols:
            df["ind_sum"] = df[ind_cols].sum(axis=1)
            df["ind_zeros"] = (df[ind_cols] == 0).sum(axis=1)

        # Balance velocity deltas and ratios
        if "saldo_medio_var5_ult3" in df.columns and "saldo_medio_var5_hace2" in df.columns:
            df["delta_saldo_medio_var5_ult3_hace2"] = df["saldo_medio_var5_ult3"] - df["saldo_medio_var5_hace2"]
        if "saldo_medio_var5_ult1" in df.columns and "saldo_medio_var5_ult3" in df.columns:
            df["delta_saldo_medio_var5_ult1_ult3"] = df["saldo_medio_var5_ult1"] - df["saldo_medio_var5_ult3"]
            df["ratio_saldo_medio_var5_ult1_ult3"] = df["saldo_medio_var5_ult1"] / (np.abs(df["saldo_medio_var5_ult3"]) + 1.0)
        if "saldo_var30" in df.columns and "saldo_var5" in df.columns:
            df["delta_saldo_var30_var5"] = df["saldo_var30"] - df["saldo_var5"]
            df["ratio_saldo_var30_var5"] = df["saldo_var30"] / (np.abs(df["saldo_var5"]) + 1.0)
        if "imp_op_var41_comer_ult3" in df.columns and "imp_op_var41_comer_ult1" in df.columns:
            df["delta_imp_op_var41_comer_ult3_ult1"] = df["imp_op_var41_comer_ult3"] - df["imp_op_var41_comer_ult1"]
        if "imp_op_var39_comer_ult3" in df.columns and "imp_op_var39_comer_ult1" in df.columns:
            df["delta_imp_op_var39_comer_ult3_ult1"] = df["imp_op_var39_comer_ult3"] - df["imp_op_var39_comer_ult1"]

        # Granular non-linear age lifecycle features and interactions
        if "var15" in df.columns:
            age = np.maximum(df["var15"], 1)
            df["age_lt_23"] = (df["var15"] < 23).astype(np.int8)
            df["age_23_30"] = ((df["var15"] >= 23) & (df["var15"] <= 30)).astype(np.int8)
            df["age_30_45"] = ((df["var15"] > 30) & (df["var15"] <= 45)).astype(np.int8)
            df["age_45_60"] = ((df["var15"] > 45) & (df["var15"] <= 60)).astype(np.int8)
            df["age_gt_60"] = (df["var15"] > 60).astype(np.int8)
            df["age_sq"] = df["var15"] ** 2
            df["age_log"] = np.log1p(df["var15"])

            if "saldo_sum" in df.columns:
                df["saldo_to_age_ratio"] = df["saldo_sum"] / age
            if "saldo_var30" in df.columns:
                df["saldo_var30_to_age"] = df["saldo_var30"] / age
            if "saldo_var5" in df.columns:
                df["saldo_var5_to_age"] = df["saldo_var5"] / age
            if "imp_sum" in df.columns:
                df["imp_to_age_ratio"] = df["imp_sum"] / age
            if "ind_sum" in df.columns:
                df["products_to_age_ratio"] = df["ind_sum"] / age
            if "num_var4" in df.columns:
                df["num_var4_to_age"] = df["num_var4"] / age
            if "var38" in df.columns:
                df["wealth_to_age_ratio"] = df["var38"] / (age + 1.0)
        if "saldo_sum" in df.columns and "ind_sum" in df.columns:
            df["balance_to_products_ratio"] = df["saldo_sum"] / (df["ind_sum"] + 1.0)

        # Product-to-balance and product-to-activity interactions
        if "saldo_var30" in df.columns and "num_var4" in df.columns:
            df["saldo_var30_to_num_var4"] = df["saldo_var30"] / (df["num_var4"] + 1.0)
        if "saldo_var5" in df.columns and "num_var4" in df.columns:
            df["saldo_var5_to_num_var4"] = df["saldo_var5"] / (df["num_var4"] + 1.0)
        if "imp_sum" in df.columns and "num_var4" in df.columns:
            df["imp_sum_to_num_var4"] = df["imp_sum"] / (df["num_var4"] + 1.0)
        if "saldo_sum" in df.columns and "num_var4" in df.columns:
            df["saldo_sum_to_num_var4"] = df["saldo_sum"] / (df["num_var4"] + 1.0)

    # Frequency encoding for key discrete categorical/pseudo-categorical features
    freq_cols = [c for c in ["var36", "var15", "var3"] if c in X_train.columns]
    for col in freq_cols:
        freq_map = X_train[col].value_counts(normalize=True).to_dict()
        X_train[f"{col}_freq"] = X_train[col].map(freq_map).fillna(0)
        X_test[f"{col}_freq"] = X_test[col].map(freq_map).fillna(0)

    all_features = list(X_train.columns)
    n_samples = len(X_train)
    n_test = len(X_test)
    n_folds = 10

    print(
        f"Engineered features count: {len(all_features)} | Train: {n_samples} | Test: {n_test}"
    )

    # 5. Model Configurations
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 16,
        "max_depth": 5,
        "min_child_samples": 30,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb2_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.015,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.50,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
        "n_estimators": 2000,
        "random_state": 100,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_child_weight": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "iterations": 2000,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }

    # 6. Stratified Cross-Validation Setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_lgb1 = np.zeros(n_samples, dtype=np.float64)
    oof_lgb2 = np.zeros(n_samples, dtype=np.float64)
    oof_xgb = np.zeros(n_samples, dtype=np.float64)
    oof_cb = np.zeros(n_samples, dtype=np.float64)
    oof_nn = np.zeros(n_samples, dtype=np.float64)

    oof_lgb1_rank = np.zeros(n_samples, dtype=np.float64)
    oof_lgb2_rank = np.zeros(n_samples, dtype=np.float64)
    oof_xgb_rank = np.zeros(n_samples, dtype=np.float64)
    oof_cb_rank = np.zeros(n_samples, dtype=np.float64)
    oof_nn_rank = np.zeros(n_samples, dtype=np.float64)

    test_lgb1_preds = np.zeros(n_test, dtype=np.float64)
    test_lgb2_preds = np.zeros(n_test, dtype=np.float64)
    test_xgb_preds = np.zeros(n_test, dtype=np.float64)
    test_cb_preds = np.zeros(n_test, dtype=np.float64)
    test_nn_preds = np.zeros(n_test, dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx].copy()
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx].copy()
        y_va = y_train[val_idx]
        X_te = X_test.copy()

        # TruncatedSVD latent customer financial representations fitted strictly on train fold
        svd = TruncatedSVD(n_components=5, random_state=42)
        scaler_svd = StandardScaler()
        X_tr_scaled = scaler_svd.fit_transform(X_tr.fillna(0))
        X_va_scaled = scaler_svd.transform(X_va.fillna(0))
        X_te_scaled = scaler_svd.transform(X_te.fillna(0))
        svd_tr = svd.fit_transform(X_tr_scaled)
        svd_va = svd.transform(X_va_scaled)
        svd_te = svd.transform(X_te_scaled)
        for i in range(5):
            X_tr[f"svd_comp_{i}"] = svd_tr[:, i]
            X_va[f"svd_comp_{i}"] = svd_va[:, i]
            X_te[f"svd_comp_{i}"] = svd_te[:, i]

        # Out-of-fold Bayesian smoothed target encodings for key discrete features
        te_cols = [c for c in ["var36", "var3"] if c in X_tr.columns]
        for col in te_cols:
            smooth_weight = 10.0
            global_mean = y_tr.mean()
            stats = (
                pd.DataFrame({"y": y_tr, "cat": X_tr[col]})
                .groupby("cat")["y"]
                .agg(["count", "mean"])
            )
            smooth_enc = (
                stats["count"] * stats["mean"] + smooth_weight * global_mean
            ) / (stats["count"] + smooth_weight)
            enc_dict = smooth_enc.to_dict()
            X_tr[f"{col}_te"] = X_tr[col].map(enc_dict).fillna(global_mean)
            X_va[f"{col}_te"] = X_va[col].map(enc_dict).fillna(global_mean)
            X_te[f"{col}_te"] = X_te[col].map(enc_dict).fillna(global_mean)

        # 1. Train LightGBM Model 1 (Shallow Tuned)
        lgb_model1 = lgb.LGBMClassifier(**lgb_params)
        lgb_model1.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        lgb1_val_prob = lgb_model1.predict_proba(X_va)[:, 1]
        oof_lgb1[val_idx] = lgb1_val_prob
        test_lgb1_preds += lgb_model1.predict_proba(X_te)[:, 1] / n_folds

        # 2. Train LightGBM Model 2 (Deep Complementary)
        lgb_model2 = lgb.LGBMClassifier(**lgb2_params)
        lgb_model2.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        lgb2_val_prob = lgb_model2.predict_proba(X_va)[:, 1]
        oof_lgb2[val_idx] = lgb2_val_prob
        test_lgb2_preds += lgb_model2.predict_proba(X_te)[:, 1] / n_folds

        # 3. Train XGBoost Model
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_val_prob = xgb_model.predict_proba(X_va)[:, 1]
        oof_xgb[val_idx] = xgb_val_prob
        test_xgb_preds += xgb_model.predict_proba(X_te)[:, 1] / n_folds

        # 4. Train CatBoost Model
        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_val_prob = cb_model.predict_proba(X_va)[:, 1]
        oof_cb[val_idx] = cb_val_prob
        test_cb_preds += cb_model.predict_proba(X_te)[:, 1] / n_folds

        # 5. Train PyTorch Tabular ResNet
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_scaler = StandardScaler()
        X_tr_nn = nn_scaler.fit_transform(X_tr.fillna(0).values)
        X_va_nn = nn_scaler.transform(X_va.fillna(0).values)
        X_te_nn = nn_scaler.transform(X_te.fillna(0).values)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_nn, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )

        nn_model = TabularResNet(
            in_features=X_tr_nn.shape[1],
            hidden_dim=128,
            num_blocks=2,
            dropout_rate=0.2,
        ).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            nn_model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        epochs = 20
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_auc = 0.0
        best_nn_val_preds = np.zeros(len(val_idx))
        best_nn_test_preds = np.zeros(n_test)

        X_va_tensor = torch.tensor(X_va_nn, dtype=torch.float32).to(device)
        X_te_tensor = torch.tensor(X_te_nn, dtype=torch.float32).to(device)

        for epoch in range(epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = nn_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_preds_ep = nn_model(X_va_tensor).cpu().numpy()
                ep_auc = roc_auc_score(y_va, val_preds_ep)
                if ep_auc > best_val_auc:
                    best_val_auc = ep_auc
                    best_nn_val_preds = val_preds_ep
                    best_nn_test_preds = nn_model(X_te_tensor).cpu().numpy()

        oof_nn[val_idx] = best_nn_val_preds
        test_nn_preds += best_nn_test_preds / n_folds

        fold_lgb1_auc = roc_auc_score(y_va, lgb1_val_prob)
        fold_lgb2_auc = roc_auc_score(y_va, lgb2_val_prob)
        fold_xgb_auc = roc_auc_score(y_va, xgb_val_prob)
        fold_cb_auc = roc_auc_score(y_va, cb_val_prob)
        fold_nn_auc = roc_auc_score(y_va, best_nn_val_preds)
        print(
            f"Fold {fold+1:02d}/{n_folds} - LGB1: {fold_lgb1_auc:.5f} | LGB2: {fold_lgb2_auc:.5f} | XGB: {fold_xgb_auc:.5f} | CB: {fold_cb_auc:.5f} | NN: {fold_nn_auc:.5f}"
        )

    # 7. Model Evaluation & ROC-AUC Metric Optimization
    lgb1_overall_auc = roc_auc_score(y_train, oof_lgb1)
    lgb2_overall_auc = roc_auc_score(y_train, oof_lgb2)
    xgb_overall_auc = roc_auc_score(y_train, oof_xgb)
    cb_overall_auc = roc_auc_score(y_train, oof_cb)
    nn_overall_auc = roc_auc_score(y_train, oof_nn)

    print(f"Overall LGBM-1   OOF ROC-AUC: {lgb1_overall_auc:.5f}")
    print(f"Overall LGBM-2   OOF ROC-AUC: {lgb2_overall_auc:.5f}")
    print(f"Overall XGBoost  OOF ROC-AUC: {xgb_overall_auc:.5f}")
    print(f"Overall CatBoost OOF ROC-AUC: {cb_overall_auc:.5f}")
    print(f"Overall TabResNet OOF ROC-AUC: {nn_overall_auc:.5f}")

    oof_models = [
        rankdata(oof_lgb1) / n_samples,
        rankdata(oof_lgb2) / n_samples,
        rankdata(oof_xgb) / n_samples,
        rankdata(oof_cb) / n_samples,
        rankdata(oof_nn) / n_samples,
    ]
    test_models = [
        rankdata(test_lgb1_preds) / n_test,
        rankdata(test_lgb2_preds) / n_test,
        rankdata(test_xgb_preds) / n_test,
        rankdata(test_cb_preds) / n_test,
        rankdata(test_nn_preds) / n_test,
    ]
    n_models = len(oof_models)

    # Find optimal blend weights directly maximizing ROC-AUC
    def neg_auc_objective(raw_w):
        w = np.exp(raw_w) / np.sum(np.exp(raw_w))
        blend_rank = sum(w[i] * oof_models[i] for i in range(n_models))
        return -roc_auc_score(y_train, blend_rank)

    opt_result = minimize(
        neg_auc_objective,
        x0=np.zeros(n_models),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-5},
    )

    opt_weights = np.exp(opt_result.x) / np.sum(np.exp(opt_result.x))
    print(f"Optimized Blend Weights: {[round(float(w), 4) for w in opt_weights]}")

    oof_blend = sum(opt_weights[i] * oof_models[i] for i in range(n_models))
    final_score = roc_auc_score(y_train, oof_blend)
    print(f"Overall Optimized Ensembled OOF ROC-AUC: {final_score:.5f}")

    # 8. Generate Submission File
    test_final_preds = sum(opt_weights[i] * test_models[i] for i in range(n_models))
    submission_df = pd.DataFrame(
        {"id": np.arange(n_test, dtype=np.int64), "TARGET": test_final_preds}
    )

    sub_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(
        f"Submission successfully saved to {sub_path} with shape {submission_df.shape}"
    )

    # 9. Required Final Validation Score Output
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    run_pipeline()