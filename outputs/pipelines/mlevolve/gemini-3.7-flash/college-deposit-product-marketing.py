import json
import os
from typing import Any, Dict, List, Optional

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =========================================================================
# 1. Tabular Neural Network Architecture (PyTorch)
# =========================================================================
class TabularResidualBlock(nn.Module):
    """Residual Block with LayerNorm, GELU activation, and Dropout for tabular data."""

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet architecture for continuous regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        layers = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(TabularResidualBlock(in_dim, dropout_rate))
            if in_dim != out_dim:
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(in_dim),
                        nn.Linear(in_dim, out_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                    )
                )
            layers.append(TabularResidualBlock(out_dim, dropout_rate))

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        feat = self.backbone(feat)
        out = self.head(feat)
        return out.squeeze(-1)


# =========================================================================
# 2. Main End-to-End Pipeline
# =========================================================================
def main():
    os.makedirs("./working/models", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Set random seeds for exact reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Load Raw Parquet Data ---
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path).copy()
    test_df = pd.read_parquet(test_path).copy()

    target_col = "TOTAL_ACTIVE_ACCOUNTS_"
    if target_col not in train_df.columns:
        matches = [c for c in train_df.columns if c.upper() == target_col.upper()]
        if matches:
            target_col = matches[0]
        else:
            raise KeyError(f"Target column {target_col} not found in train data.")

    # Preserve test index/id
    if "id" not in test_df.columns and "ID" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    elif "ID" in test_df.columns and "id" not in test_df.columns:
        test_df["id"] = test_df["ID"]

    feature_cols = [c for c in train_df.columns if c != target_col]
    cat_cols = []
    num_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # --- Step 2: Data Cleaning & Imputation ---
    for c in cat_cols:
        train_df[c] = train_df[c].fillna("MISSING").astype(str).str.strip().str.upper()
        test_df[c] = test_df[c].fillna("MISSING").astype(str).str.strip().str.upper()

    for c in num_cols:
        med_val = train_df[c].median()
        if pd.isna(med_val):
            med_val = 0.0
        if train_df[c].isna().sum() > 0 or test_df[c].isna().sum() > 0:
            train_df[f"{c}_isnan"] = train_df[c].isna().astype(float)
            test_df[f"{c}_isnan"] = test_df[c].isna().astype(float)
        train_df[c] = train_df[c].fillna(med_val).astype(float)
        test_df[c] = test_df[c].fillna(med_val).astype(float)

    # Setup K-Fold split (10-fold for optimal sample efficiency on small tabular data)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_df)):
        train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    # --- Step 3: Feature Engineering ---
    # (a) Frequency Encoding
    for c in cat_cols:
        freq = train_df[c].value_counts(normalize=True).to_dict()
        train_df[f"{c}_freq"] = train_df[c].map(freq).fillna(0.0).astype(float)
        test_df[f"{c}_freq"] = test_df[c].map(freq).fillna(0.0).astype(float)

    # (b) Non-linear Transformations and Quantile Ranks for Numerical Features
    for c in num_cols:
        train_df[f"{c}_qrank"] = train_df[c].rank(pct=True).astype(float)
        test_df[f"{c}_qrank"] = test_df[c].rank(pct=True).astype(float)
        if (train_df[c] >= 0).all() and (test_df[c] >= 0).all():
            train_df[f"{c}_log1p"] = np.log1p(train_df[c])
            test_df[f"{c}_log1p"] = np.log1p(test_df[c])
            train_df[f"{c}_sqrt"] = np.sqrt(train_df[c])
            test_df[f"{c}_sqrt"] = np.sqrt(test_df[c])

    # (c) Pairwise Numerical Ratios, Differences, and Cross-Products across all continuous signals
    for i in range(len(num_cols)):
        for j in range(i + 1, min(i + 8, len(num_cols))):
            c1, c2 = num_cols[i], num_cols[j]
            train_df[f"{c1}_minus_{c2}"] = train_df[c1] - train_df[c2]
            test_df[f"{c1}_minus_{c2}"] = test_df[c1] - test_df[c2]
            train_df[f"{c1}_ratio_{c2}"] = train_df[c1] / (train_df[c2].abs() + 1.0)
            test_df[f"{c1}_ratio_{c2}"] = test_df[c1] / (test_df[c2].abs() + 1.0)
            train_df[f"{c1}_prod_{c2}"] = train_df[c1] * train_df[c2]
            test_df[f"{c1}_prod_{c2}"] = test_df[c1] * test_df[c2]

    # (d) Pairwise Categorical Interactions
    top_cats = cat_cols[:6]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            pair_name = f"{c1}_X_{c2}"
            train_df[pair_name] = train_df[c1] + "__" + train_df[c2]
            test_df[pair_name] = test_df[c1] + "__" + test_df[c2]

            pair_freq = train_df[pair_name].value_counts(normalize=True).to_dict()
            train_df[f"{pair_name}_freq"] = (
                train_df[pair_name].map(pair_freq).fillna(0.0).astype(float)
            )
            test_df[f"{pair_name}_freq"] = (
                test_df[pair_name].map(pair_freq).fillna(0.0).astype(float)
            )

            train_df.drop(columns=[pair_name], inplace=True)
            test_df.drop(columns=[pair_name], inplace=True)

    # (e) Multi-Scale Out-of-Fold Target Encoding (Direct & Log1p Mean + Std, Strictly Leak-Free)
    target_enc_cols = cat_cols[:8]
    smoothing_grid = [5.0, 20.0, 50.0]
    global_mean = train_df[target_col].mean()
    global_std = train_df[target_col].std()
    log_target = np.log1p(np.clip(train_df[target_col], 0, None))
    global_log_mean = log_target.mean()
    train_df["_target_log_tmp"] = log_target

    for c in target_enc_cols:
        for sm in smoothing_grid:
            te_mean_col = f"{c}_te_mean_sm{int(sm)}"
            te_log_mean_col = f"{c}_te_log_mean_sm{int(sm)}"
            train_df[te_mean_col] = np.nan
            train_df[te_log_mean_col] = np.nan
            test_te_mean_acc = np.zeros(len(test_df))
            test_te_log_acc = np.zeros(len(test_df))

            for fold in range(n_splits):
                tr_idx = train_df["fold"] != fold
                va_idx = train_df["fold"] == fold

                counts = train_df.loc[tr_idx, c].value_counts()
                sums = train_df.loc[tr_idx].groupby(c)[target_col].sum()
                fold_te_mean = ((sums + global_mean * sm) / (counts + sm)).to_dict()

                sums_log = train_df.loc[tr_idx].groupby(c)["_target_log_tmp"].sum()
                fold_te_log = ((sums_log + global_log_mean * sm) / (counts + sm)).to_dict()

                train_df.loc[va_idx, te_mean_col] = (
                    train_df.loc[va_idx, c].map(fold_te_mean).fillna(global_mean)
                )
                test_te_mean_acc += (
                    test_df[c].map(fold_te_mean).fillna(global_mean).values / n_splits
                )

                train_df.loc[va_idx, te_log_mean_col] = (
                    train_df.loc[va_idx, c].map(fold_te_log).fillna(global_log_mean)
                )
                test_te_log_acc += (
                    test_df[c].map(fold_te_log).fillna(global_log_mean).values / n_splits
                )

            test_df[te_mean_col] = test_te_mean_acc
            test_df[te_log_mean_col] = test_te_log_acc

        # Out-of-fold target standard deviation encoding
        te_std_col = f"{c}_te_std"
        train_df[te_std_col] = np.nan
        test_te_std_acc = np.zeros(len(test_df))
        for fold in range(n_splits):
            tr_idx = train_df["fold"] != fold
            va_idx = train_df["fold"] == fold

            stds = train_df.loc[tr_idx].groupby(c)[target_col].std().fillna(global_std).to_dict()
            train_df.loc[va_idx, te_std_col] = (
                train_df.loc[va_idx, c].map(stds).fillna(global_std)
            )
            test_te_std_acc += test_df[c].map(stds).fillna(global_std).values / n_splits
        test_df[te_std_col] = test_te_std_acc

    train_df.drop(columns=["_target_log_tmp"], inplace=True)

    # (f) Hierarchical Group-by Aggregations with Relative Deviations
    top_group_cats = cat_cols[:6]
    top_agg_nums = num_cols[:8]
    for gc in top_group_cats:
        for nc in top_agg_nums:
            grp_stats = (
                train_df.groupby(gc)[nc]
                .agg(["mean", "std", "median", "min", "max", "skew", "count"])
                .reset_index()
            )
            grp_stats.columns = [
                gc,
                f"{gc}_{nc}_mean",
                f"{gc}_{nc}_std",
                f"{gc}_{nc}_median",
                f"{gc}_{nc}_min",
                f"{gc}_{nc}_max",
                f"{gc}_{nc}_skew",
                f"{gc}_{nc}_count",
            ]
            grp_stats[f"{gc}_{nc}_std"] = grp_stats[f"{gc}_{nc}_std"].fillna(0.0)
            grp_stats[f"{gc}_{nc}_skew"] = grp_stats[f"{gc}_{nc}_skew"].fillna(0.0)

            train_df = train_df.merge(grp_stats, on=gc, how="left")
            test_df = test_df.merge(grp_stats, on=gc, how="left")

            for stat in ["mean", "std", "median", "min", "max", "skew", "count"]:
                stat_col = f"{gc}_{nc}_{stat}"
                default_val = train_df[stat_col].median()
                train_df[stat_col] = (
                    train_df[stat_col].fillna(default_val).astype(float)
                )
                test_df[stat_col] = test_df[stat_col].fillna(default_val).astype(float)

            # Relative deviation features
            mean_col = f"{gc}_{nc}_mean"
            std_col = f"{gc}_{nc}_std"
            train_df[f"{nc}_diff_from_{gc}_mean"] = train_df[nc] - train_df[mean_col]
            test_df[f"{nc}_diff_from_{gc}_mean"] = test_df[nc] - test_df[mean_col]
            train_df[f"{nc}_ratio_{gc}_mean"] = train_df[nc] / (train_df[mean_col].abs() + 1e-5)
            test_df[f"{nc}_ratio_{gc}_mean"] = test_df[nc] / (test_df[mean_col].abs() + 1e-5)
            train_df[f"{nc}_zscore_{gc}"] = (train_df[nc] - train_df[mean_col]) / (train_df[std_col] + 1e-5)
            test_df[f"{nc}_zscore_{gc}"] = (test_df[nc] - test_df[mean_col]) / (test_df[std_col] + 1e-5)

    # (g) Ordinal Encoding for remaining raw categoricals
    for c in cat_cols:
        cat_map = {val: idx for idx, val in enumerate(train_df[c].unique())}
        train_df[c] = train_df[c].map(cat_map).fillna(-1).astype(int)
        test_df[c] = test_df[c].map(cat_map).fillna(-1).astype(int)

    # Prepare final feature lists
    ignore_cols = {target_col, "fold", "id", "ID"}
    final_feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    for c in final_feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0

    train_df[final_feature_cols] = train_df[final_feature_cols].fillna(0.0)
    test_df[final_feature_cols] = test_df[final_feature_cols].fillna(0.0)

    target_is_non_negative = (train_df[target_col] >= 0).all()

    # --- Step 4: 10-Fold Multi-Family Ensemble Training ---
    oof_preds = {
        "lgb": np.zeros(len(train_df)),
        "xgb": np.zeros(len(train_df)),
        "cat": np.zeros(len(train_df)),
        "et": np.zeros(len(train_df)),
        "rf": np.zeros(len(train_df)),
        "ridge": np.zeros(len(train_df)),
        "nn": np.zeros(len(train_df)),
    }
    test_preds = {
        "lgb": np.zeros(len(test_df)),
        "xgb": np.zeros(len(test_df)),
        "cat": np.zeros(len(test_df)),
        "et": np.zeros(len(test_df)),
        "rf": np.zeros(len(test_df)),
        "ridge": np.zeros(len(test_df)),
        "nn": np.zeros(len(test_df)),
    }

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        va_idx = train_df["fold"] == fold

        X_tr = train_df.loc[tr_idx, final_feature_cols].values
        y_tr = train_df.loc[tr_idx, target_col].values
        X_va = train_df.loc[va_idx, final_feature_cols].values
        y_va = train_df.loc[va_idx, target_col].values
        X_te = test_df[final_feature_cols].values

        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=2000,
            random_state=seed + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        val_p_lgb = model_lgb.predict(X_va)
        oof_preds["lgb"][va_idx] = val_p_lgb
        test_preds["lgb"] += model_lgb.predict(X_te) / n_splits

        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=2000,
            random_state=seed + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_p_xgb = model_xgb.predict(X_va)
        oof_preds["xgb"][va_idx] = val_p_xgb
        test_preds["xgb"] += model_xgb.predict(X_te) / n_splits

        # 3. CatBoost
        model_cat = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            iterations=2000,
            random_seed=seed + fold,
            verbose=False,
            early_stopping_rounds=50,
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        val_p_cat = model_cat.predict(X_va)
        oof_preds["cat"][va_idx] = val_p_cat
        test_preds["cat"] += model_cat.predict(X_te) / n_splits

        # 4. ExtraTrees Regressor (Bagging Model)
        model_et = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed + fold,
            n_jobs=-1,
        )
        model_et.fit(X_tr, y_tr)
        val_p_et = model_et.predict(X_va)
        oof_preds["et"][va_idx] = val_p_et
        test_preds["et"] += model_et.predict(X_te) / n_splits

        # 5. RandomForest Regressor (Bagging Model)
        model_rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed + fold,
            n_jobs=-1,
        )
        model_rf.fit(X_tr, y_tr)
        val_p_rf = model_rf.predict(X_va)
        oof_preds["rf"][va_idx] = val_p_rf
        test_preds["rf"] += model_rf.predict(X_te) / n_splits

        # Standard Scaler for Linear and Deep Learning Models
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        # 6. Regularized Ridge Linear Model
        model_ridge = Ridge(alpha=10.0)
        model_ridge.fit(X_tr_s, y_tr)
        val_p_ridge = model_ridge.predict(X_va_s)
        oof_preds["ridge"][va_idx] = val_p_ridge
        test_preds["ridge"] += model_ridge.predict(X_te_s) / n_splits

        # 7. PyTorch TabularResNet

        train_dataset = TensorDataset(
            torch.tensor(X_tr_s, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_s, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )
        test_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        nn_model = TabularResNet(
            input_dim=X_tr.shape[1],
            hidden_dims=[256, 128, 64],
            dropout_rate=0.15,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_nn_val_preds = None
        best_weights_path = f"./working/models/best_nn_fold{fold}.pt"

        for epoch in range(40):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = nn_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    preds = nn_model(batch_x)
                    val_preds_list.append(preds.cpu().numpy())
            val_preds_arr = np.concatenate(val_preds_list)
            val_loss = np.sqrt(mean_squared_error(y_va, val_preds_arr))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_val_preds = val_preds_arr
                torch.save(nn_model.state_dict(), best_weights_path)

        nn_model.load_state_dict(torch.load(best_weights_path))
        nn_model.eval()
        with torch.no_grad():
            test_nn_preds = nn_model(test_tensor).cpu().numpy()

        oof_preds["nn"][va_idx] = best_nn_val_preds
        test_preds["nn"] += test_nn_preds / n_splits

    # --- Step 5: Optimize Ensemble Blending Weights ---
    y_true = train_df[target_col].values
    model_names = ["lgb", "xgb", "cat", "et", "rf", "ridge", "nn"]
    oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    def blend_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.dot(oof_matrix, w)
        if target_is_non_negative:
            pred = np.clip(pred, 0, None)
        # Regularized blending with L2 weight shrinkage to avoid meta-overfitting
        l2_penalty = 1e-4 * np.sum((w - 1.0 / len(model_names)) ** 2)
        return np.sqrt(mean_squared_error(y_true, pred)) + l2_penalty

    init_weights = [1.0 / len(model_names)] * len(model_names)
    bounds = [(0.0, 1.0)] * len(model_names)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x / np.sum(opt_res.x)

    final_oof_pred = np.dot(oof_matrix, best_weights)
    if target_is_non_negative:
        final_oof_pred = np.clip(final_oof_pred, 0, None)

    final_score = np.sqrt(mean_squared_error(y_true, final_oof_pred))

    # --- Step 6: Generate Final Submission ---
    final_test_pred = np.dot(test_matrix, best_weights)
    if target_is_non_negative:
        final_test_pred = np.clip(final_test_pred, 0, None)

    submission_df = pd.DataFrame(
        {"id": test_df["id"].astype(int), target_col: final_test_pred}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
