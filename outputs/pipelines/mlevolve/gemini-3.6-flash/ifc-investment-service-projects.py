import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. PyTorch Neural Architecture Design
# ==========================================
class SwiGLU(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x_proj, gate = self.fc(x).chunk(2, dim=-1)
        return x_proj * torch.sigmoid(gate)


class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.ln1(x)
        out = self.swiglu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return torch.nn.functional.silu(out + residual)


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out


class RobustFinancialLoss(nn.Module):

    def __init__(self, alpha=0.6):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=1.0)
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_huber = self.huber(pred, target)
        pred_log = torch.log1p(torch.clamp(pred, min=0))
        target_log = torch.log1p(torch.clamp(target, min=0))
        loss_log = torch.sqrt(self.mse(pred_log, target_log) + 1e-8)
        return self.alpha * loss_huber + (1.0 - self.alpha) * loss_log


# ==========================================
# 2. Main Execution Flow
# ==========================================
def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # Load Datasets
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "Total IFC investment as approved by Board(Million - USD)"

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    y_train = train_df[target_col].values

    id_col = None
    for col in ["id", "ID", "Id", "Project ID", "project_id"]:
        if col in test_df.columns:
            id_col = col
            break

    if id_col is not None:
        test_ids = test_df[id_col].values
    else:
        test_ids = np.arange(len(test_df))

    feature_cols = [c for c in train_df.columns if c != target_col and c != id_col]

    n_train = len(train_df)
    n_test = len(test_df)

    combined = pd.concat(
        [train_df[feature_cols], test_df[feature_cols]], axis=0, ignore_index=True
    )

    # 1. Temporal Feature Extraction
    date_cols = []
    for col in combined.columns:
        if (
            combined[col].dtype == "object"
            or "date" in col.lower()
            or "year" in col.lower()
        ):
            sample_nonnull = combined[col].dropna().astype(str)
            if len(sample_nonnull) > 0 and (
                "-" in sample_nonnull.iloc[0]
                or "/" in sample_nonnull.iloc[0]
                or "date" in col.lower()
            ):
                try:
                    parsed = pd.to_datetime(combined[col], errors="coerce")
                    if parsed.notna().sum() > 0.3 * len(combined):
                        combined[col + "_year"] = parsed.dt.year
                        combined[col + "_month"] = parsed.dt.month
                        combined[col + "_quarter"] = parsed.dt.quarter
                        combined[col + "_dayofweek"] = parsed.dt.dayofweek
                        date_cols.append(col)
                except Exception:
                    pass

    # 2. Unstructured Text Decomposition (Fitted on Train to Avoid Leakage)
    text_cols = []
    for col in combined.columns:
        if col not in date_cols and combined[col].dtype == "object":
            avg_len = combined[col].dropna().astype(str).str.len().mean()
            if avg_len > 30 or combined[col].nunique() > 1000:
                text_cols.append(col)

    train_sub = combined.iloc[:n_train]
    for col in text_cols:
        col_text = combined[col].fillna("").astype(str)
        col_text_tr = train_sub[col].fillna("").astype(str)
        combined[f"{col}_len"] = col_text.str.len()
        combined[f"{col}_word_cnt"] = col_text.apply(lambda x: len(x.split()))

        tfidf_w = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        mat_w_tr = tfidf_w.fit_transform(col_text_tr)
        mat_w_all = tfidf_w.transform(col_text)
        if mat_w_tr.shape[1] > 1:
            n_comp_w = min(12, mat_w_tr.shape[1] - 1)
            svd_w = TruncatedSVD(n_components=n_comp_w, random_state=42)
            svd_w.fit(mat_w_tr)
            feats_w = svd_w.transform(mat_w_all)
            for i in range(feats_w.shape[1]):
                combined[f"{col}_svd_word_{i}"] = feats_w[:, i]

        tfidf_c = TfidfVectorizer(
            max_features=5000, analyzer="char", ngram_range=(2, 4)
        )
        mat_c_tr = tfidf_c.fit_transform(col_text_tr)
        mat_c_all = tfidf_c.transform(col_text)
        if mat_c_tr.shape[1] > 1:
            n_comp_c = min(12, mat_c_tr.shape[1] - 1)
            svd_c = TruncatedSVD(n_components=n_comp_c, random_state=42)
            svd_c.fit(mat_c_tr)
            feats_c = svd_c.transform(mat_c_all)
            for i in range(feats_c.shape[1]):
                combined[f"{col}_svd_char_{i}"] = feats_c[:, i]

    # 3. Categorical Processing & Frequency Encoding (Fitted on Train)
    cat_cols = [
        c
        for c in combined.columns
        if combined[c].dtype == "object" and c not in text_cols and c not in date_cols
    ]
    train_sub = combined.iloc[:n_train]
    for c in cat_cols:
        freq_map = train_sub[c].value_counts(normalize=True).to_dict()
        combined[c + "_freq"] = combined[c].map(freq_map).fillna(0)
        combined[c] = pd.factorize(combined[c])[0]

    drop_cols = date_cols + text_cols
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

    # 4. Numerical Missingness Indicators & Imputation (Median from Train)
    num_cols = [
        c
        for c in combined.columns
        if pd.api.types.is_numeric_dtype(combined[c]) and c not in cat_cols
    ]
    train_sub = combined.iloc[:n_train]
    for c in num_cols:
        if combined[c].isna().sum() > 0:
            combined[c + "_isna"] = combined[c].isna().astype(int)
            med_val = train_sub[c].median()
            combined[c] = combined[c].fillna(med_val)

    # 5. Domain Interaction Aggregations (Computed on Train)
    top_cats = cat_cols[:3]
    top_nums = [
        c
        for c in num_cols
        if "svd_" not in c and "_freq" not in c and "_len" not in c and "_word_cnt" not in c
    ][:4]
    train_sub = combined.iloc[:n_train]
    for cat in top_cats:
        for num in top_nums:
            grp_tr = train_sub.groupby(cat)[num]
            mean_map = grp_tr.mean().to_dict()
            std_map = grp_tr.std().fillna(0).to_dict()
            med_map = grp_tr.median().to_dict()
            sum_map = grp_tr.sum().to_dict()
            min_map = grp_tr.min().to_dict()
            max_map = grp_tr.max().to_dict()
            glob_mean = train_sub[num].mean()

            mean_val = combined[cat].map(mean_map).fillna(glob_mean)
            combined[f"{num}_mean_by_{cat}"] = mean_val
            combined[f"{num}_std_by_{cat}"] = combined[cat].map(std_map).fillna(0)
            combined[f"{num}_median_by_{cat}"] = combined[cat].map(med_map).fillna(train_sub[num].median())
            combined[f"{num}_sum_by_{cat}"] = combined[cat].map(sum_map).fillna(0)
            combined[f"{num}_min_by_{cat}"] = combined[cat].map(min_map).fillna(train_sub[num].min())
            combined[f"{num}_max_by_{cat}"] = combined[cat].map(max_map).fillna(train_sub[num].max())
            combined[f"{num}_ratio_by_{cat}"] = combined[num] / (mean_val.abs() + 1e-5)

    X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

    # Quantile-Stratified Multi-Seed 5-Fold Cross Validation
    y_bins = pd.qcut(
        y_train,
        q=min(10, len(np.unique(y_train))),
        labels=False,
        duplicates="drop",
    )

    # Prediction Arrays across Model Suite
    oof_m1 = np.zeros(n_train)
    oof_m2 = np.zeros(n_train)
    oof_m3 = np.zeros(n_train)
    oof_m4 = np.zeros(n_train)
    oof_m5 = np.zeros(n_train)
    oof_m6 = np.zeros(n_train)

    test_m1 = np.zeros(n_test)
    test_m2 = np.zeros(n_test)
    test_m3 = np.zeros(n_test)
    test_m4 = np.zeros(n_test)
    test_m5 = np.zeros(n_test)
    test_m6 = np.zeros(n_test)

    seeds = [42, 2024]
    n_seeds = len(seeds)

    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_bins)):
            X_tr = X_train.iloc[train_idx].copy().reset_index(drop=True)
            y_tr = y_train[train_idx]
            X_va = X_train.iloc[val_idx].copy().reset_index(drop=True)
            y_va = y_train[val_idx]
            X_te = X_test.copy().reset_index(drop=True)

            # Dynamic Nested In-Fold Multi-Statistic Target Encoding (Mean & Std)
            for c in cat_cols:
                if c in X_tr.columns:
                    global_mean = float(np.mean(y_tr))
                    global_std = float(np.std(y_tr))
                    X_tr[c + "_te_mean"] = global_mean
                    X_tr[c + "_te_std"] = global_std
                    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    for in_tr_idx, in_va_idx in inner_kf.split(X_tr):
                        sub_tr = X_tr.iloc[in_tr_idx].assign(target=y_tr[in_tr_idx])
                        in_map_mean = sub_tr.groupby(c)["target"].mean().to_dict()
                        in_map_std = sub_tr.groupby(c)["target"].std().fillna(0).to_dict()
                        X_tr.loc[X_tr.index[in_va_idx], c + "_te_mean"] = (
                            X_tr.iloc[in_va_idx][c].map(in_map_mean).fillna(global_mean)
                        )
                        X_tr.loc[X_tr.index[in_va_idx], c + "_te_std"] = (
                            X_tr.iloc[in_va_idx][c].map(in_map_std).fillna(global_std)
                        )
                    full_tr = X_tr.assign(target=y_tr)
                    full_map_mean = full_tr.groupby(c)["target"].mean().to_dict()
                    full_map_std = full_tr.groupby(c)["target"].std().fillna(0).to_dict()
                    X_va[c + "_te_mean"] = X_va[c].map(full_map_mean).fillna(global_mean)
                    X_va[c + "_te_std"] = X_va[c].map(full_map_std).fillna(global_std)
                    X_te[c + "_te_mean"] = X_te[c].map(full_map_mean).fillna(global_mean)
                    X_te[c + "_te_std"] = X_te[c].map(full_map_std).fillna(global_std)

            # Model 1: LightGBM Tweedie Regressor
            model_tweedie = lgb.LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                n_jobs=-1,
                verbose=-1,
            )
            model_tweedie.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            oof_m1[val_idx] += np.maximum(0, model_tweedie.predict(X_va)) / n_seeds
            test_m1 += np.maximum(0, model_tweedie.predict(X_te)) / (5.0 * n_seeds)

            # Model 2: LightGBM Log1p Regressor
            y_tr_log = np.log1p(np.maximum(0, y_tr))
            y_va_log = np.log1p(np.maximum(0, y_va))
            model_log = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=100 + fold,
                n_jobs=-1,
                verbose=-1,
            )
            model_log.fit(
                X_tr,
                y_tr_log,
                eval_set=[(X_va, y_va_log)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            oof_m2[val_idx] += np.expm1(np.maximum(0, model_log.predict(X_va))) / n_seeds
            test_m2 += np.expm1(np.maximum(0, model_log.predict(X_te))) / (5.0 * n_seeds)

            # Model 3: XGBoost Regressor
            model_xgb = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=200 + fold,
                n_jobs=-1,
                early_stopping_rounds=50,
            )
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_m3[val_idx] += np.maximum(0, model_xgb.predict(X_va)) / n_seeds
            test_m3 += np.maximum(0, model_xgb.predict(X_te)) / (5.0 * n_seeds)

            # Model 4: CatBoost Regressor (Raw RMSE)
            model_cb_raw = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                subsample=0.8,
                random_seed=300 + fold,
                verbose=0,
                early_stopping_rounds=50,
            )
            model_cb_raw.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            oof_m4[val_idx] += np.maximum(0, model_cb_raw.predict(X_va)) / n_seeds
            test_m4 += np.maximum(0, model_cb_raw.predict(X_te)) / (5.0 * n_seeds)

            # Model 5: CatBoost Regressor (Log1p)
            model_cb_log = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                subsample=0.8,
                random_seed=400 + fold,
                verbose=0,
                early_stopping_rounds=50,
            )
            model_cb_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), use_best_model=True)
            oof_m5[val_idx] += np.expm1(np.maximum(0, model_cb_log.predict(X_va))) / n_seeds
            test_m5 += np.expm1(np.maximum(0, model_cb_log.predict(X_te))) / (5.0 * n_seeds)

            # Model 6: PyTorch Tabular ResNet
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)
            X_te_scaled = scaler.transform(X_te)

            tr_ds = TensorDataset(
                torch.tensor(X_tr_scaled, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
            )
            va_ds = TensorDataset(
                torch.tensor(X_va_scaled, dtype=torch.float32),
                torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
            )

            tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

            nn_model = TabularResNet(
                input_dim=X_tr.shape[1], hidden_dim=128, num_blocks=2, dropout=0.2
            ).to(device)
            criterion = RobustFinancialLoss(alpha=0.6)
            optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=35, eta_min=1e-5
            )

            best_val_loss = float("inf")
            best_weights = None

            for epoch in range(35):
                nn_model.train()
                for bx, by in tr_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = nn_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                nn_model.eval()
                val_preds_epoch = []
                with torch.no_grad():
                    for bx, _ in va_loader:
                        bx = bx.to(device)
                        val_preds_epoch.append(nn_model(bx).cpu().numpy())
                val_preds_arr = np.maximum(0, np.vstack(val_preds_epoch).flatten())
                val_score = np.sqrt(mean_squared_error(y_va, val_preds_arr))

                if val_score < best_val_loss:
                    best_val_loss = val_score
                    best_weights = {
                        k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                    }

            if best_weights is not None:
                nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

            nn_model.eval()
            with torch.no_grad():
                va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
                te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
                oof_m6[val_idx] += np.maximum(0, nn_model(va_tensor).cpu().numpy().flatten()) / n_seeds
                test_m6 += np.maximum(0, nn_model(te_tensor).cpu().numpy().flatten()) / (5.0 * n_seeds)

    # SLSQP Constrained Optimization for Ensembling Weights
    def loss_func(weights):
        w1, w2, w3, w4, w5, w6 = weights
        pred = (
            w1 * oof_m1
            + w2 * oof_m2
            + w3 * oof_m3
            + w4 * oof_m4
            + w5 * oof_m5
            + w6 * oof_m6
        )
        return np.sqrt(mean_squared_error(y_train, pred))

    init_weights = [1.0 / 6] * 6
    bounds = [(0, 1)] * 6
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}
    res = minimize(
        loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    w1, w2, w3, w4, w5, w6 = res.x
    oof_blend = np.maximum(
        0,
        w1 * oof_m1
        + w2 * oof_m2
        + w3 * oof_m3
        + w4 * oof_m4
        + w5 * oof_m5
        + w6 * oof_m6,
    )
    test_blend = np.maximum(
        0,
        w1 * test_m1
        + w2 * test_m2
        + w3 * test_m3
        + w4 * test_m4
        + w5 * test_m5
        + w6 * test_m6,
    )

    final_score = np.sqrt(mean_squared_error(y_train, oof_blend))

    # Save Test Submission
    submission_df = pd.DataFrame({"id": test_ids, target_col: test_blend})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()