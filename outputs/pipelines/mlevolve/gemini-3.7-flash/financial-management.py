import json
import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings("ignore")

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# =========================================================================
# Seed Setting for Reproducibility
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# =========================================================================
# 1. Feature Engineering & Preprocessing Pipeline
# =========================================================================
def create_feature_pipeline():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw parquet files...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "State Share"
    if target_col not in train_df.columns:
        for c in train_df.columns:
            if c.lower() == "state share" or (
                "state" in c.lower() and "share" in c.lower()
            ):
                target_col = c
                break

    y_train = train_df[target_col].copy().reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col]).copy()
    test_features = test_df.copy()

    if "id" in test_features.columns:
        test_ids = test_features["id"].values
    else:
        test_ids = np.arange(len(test_features), dtype=np.int64)

    train_len = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    cat_cols = []
    num_cols = []
    datetime_cols = []

    for col in combined.columns:
        if col.lower() == "id":
            continue
        if pd.api.types.is_datetime64_any_dtype(combined[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(combined[col]):
            num_cols.append(col)
        else:
            sample_non_null = combined[col].dropna()
            if len(sample_non_null) > 0:
                try:
                    pd.to_numeric(sample_non_null.iloc[:100])
                    combined[col] = pd.to_numeric(combined[col], errors="coerce")
                    num_cols.append(col)
                except Exception:
                    cat_cols.append(col)
            else:
                cat_cols.append(col)

    for col in datetime_cols:
        combined[f"{col}_year"] = combined[col].dt.year
        combined[f"{col}_quarter"] = combined[col].dt.quarter
        combined[f"{col}_month"] = combined[col].dt.month
        num_cols.extend([f"{col}_year", f"{col}_quarter", f"{col}_month"])

    total_comp_col = None
    fed_share_col = None
    for col in num_cols:
        col_clean = col.lower().replace(" ", "").replace("_", "")
        if (
            "totalcomputable" in col_clean
            or "totalexpenditure" in col_clean
            or col_clean == "total"
        ):
            total_comp_col = col
        elif "federalshare" in col_clean or "fedshare" in col_clean:
            fed_share_col = col

    engineered_num_cols = []
    eps = 1e-5

    if total_comp_col is not None and fed_share_col is not None:
        combined["diff_total_fed"] = combined[total_comp_col] - combined[fed_share_col]
        combined["ratio_fed_to_total"] = combined[fed_share_col] / (
            combined[total_comp_col].abs() + eps
        )
        combined["ratio_diff_to_total"] = combined["diff_total_fed"] / (
            combined[total_comp_col].abs() + eps
        )
        engineered_num_cols.extend(
            ["diff_total_fed", "ratio_fed_to_total", "ratio_diff_to_total"]
        )

    for col in num_cols:
        if col not in ["id", f"{col}_year", f"{col}_quarter", f"{col}_month"]:
            log_col_name = f"{col}_signed_log1p"
            combined[log_col_name] = np.sign(combined[col].fillna(0)) * np.log1p(
                np.abs(combined[col].fillna(0))
            )
            engineered_num_cols.append(log_col_name)

    base_nums = [c for c in num_cols if c not in ["id"]]
    if len(base_nums) >= 2:
        for i in range(min(len(base_nums), 4)):
            for j in range(i + 1, min(len(base_nums), 4)):
                c1, c2 = base_nums[i], base_nums[j]
                diff_name = f"{c1}_minus_{c2}"
                ratio_name = f"{c1}_div_{c2}"
                combined[diff_name] = combined[c1] - combined[c2]
                combined[ratio_name] = combined[c1] / (np.abs(combined[c2]) + eps)
                engineered_num_cols.extend([diff_name, ratio_name])

    for col in cat_cols:
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq).fillna(0).astype(np.float32)
        engineered_num_cols.append(f"{col}_freq")

    inter_cat_cols = []
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                col1, col2 = cat_cols[i], cat_cols[j]
                inter_col = f"{col1}_x_{col2}"
                inter_series = (
                    combined[col1].astype(str) + "___" + combined[col2].astype(str)
                )
                inter_freq = inter_series.value_counts(dropna=False)
                combined[f"{inter_col}_freq"] = (
                    inter_series.map(inter_freq).fillna(0).astype(np.float32)
                )
                engineered_num_cols.append(f"{inter_col}_freq")
                combined[inter_col] = (
                    inter_series.astype("category").cat.codes.astype(np.int32)
                )
                inter_cat_cols.append(inter_col)

    # Hierarchical group aggregations (mean, std, min, max, median) and deviations
    agg_cats = list(set(cat_cols + inter_cat_cols[:3]))
    agg_nums = [c for c in num_cols + (["diff_total_fed"] if "diff_total_fed" in combined.columns else []) if c != "id"][:4]

    for cat_c in agg_cats:
        for num_c in agg_nums:
            grouped = combined.groupby(cat_c)[num_c]
            grp_mean = grouped.transform("mean").astype(np.float32)
            grp_std = grouped.transform("std").fillna(0).astype(np.float32)
            grp_min = grouped.transform("min").astype(np.float32)
            grp_max = grouped.transform("max").astype(np.float32)
            grp_median = grouped.transform("median").astype(np.float32)

            combined[f"{cat_c}_{num_c}_mean"] = grp_mean
            combined[f"{cat_c}_{num_c}_std"] = grp_std
            combined[f"{cat_c}_{num_c}_min"] = grp_min
            combined[f"{cat_c}_{num_c}_max"] = grp_max
            combined[f"{cat_c}_{num_c}_median"] = grp_median
            combined[f"{cat_c}_{num_c}_diff_mean"] = (combined[num_c] - grp_mean).astype(np.float32)
            combined[f"{cat_c}_{num_c}_ratio_mean"] = (combined[num_c] / (np.abs(grp_mean) + eps)).astype(np.float32)

            engineered_num_cols.extend([
                f"{cat_c}_{num_c}_mean",
                f"{cat_c}_{num_c}_std",
                f"{cat_c}_{num_c}_min",
                f"{cat_c}_{num_c}_max",
                f"{cat_c}_{num_c}_median",
                f"{cat_c}_{num_c}_diff_mean",
                f"{cat_c}_{num_c}_ratio_mean",
            ])

    for col in cat_cols:
        combined[col] = combined[col].astype("category").cat.codes.astype(np.int32)

    for col in datetime_cols:
        if col in combined.columns:
            combined = combined.drop(columns=[col])

    for col in combined.columns:
        if col.lower() != "id" and combined[col].dtype == "object":
            combined[col] = combined[col].astype("category").cat.codes.astype(np.int32)

    X_train = combined.iloc[:train_len].copy().reset_index(drop=True)
    X_test = combined.iloc[train_len:].copy().reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(train_len, dtype=np.int32)

    target_enc_cols = [c for c in cat_cols + inter_cat_cols if combined[c].nunique() > 1]

    oof_target_means = {
        col: np.zeros(train_len, dtype=np.float32) for col in target_enc_cols
    }
    test_target_means = {
        col: np.zeros(len(X_test), dtype=np.float32) for col in target_enc_cols
    }
    oof_target_stds = {
        col: np.zeros(train_len, dtype=np.float32) for col in target_enc_cols
    }
    test_target_stds = {
        col: np.zeros(len(X_test), dtype=np.float32) for col in target_enc_cols
    }

    global_target_mean = float(y_train.mean())
    global_target_std = float(y_train.std())
    smoothing = 10.0

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        folds[val_idx] = fold
        y_trn_fold = y_train.iloc[trn_idx]
        X_trn_fold = X_train.iloc[trn_idx]
        X_val_fold = X_train.iloc[val_idx]

        for col in target_enc_cols:
            stats = y_trn_fold.groupby(X_trn_fold[col]).agg(["count", "mean", "std"])
            smooth_mean = (
                stats["count"] * stats["mean"] + smoothing * global_target_mean
            ) / (stats["count"] + smoothing)
            smooth_std = (
                stats["count"] * stats["std"].fillna(0) + smoothing * global_target_std
            ) / (stats["count"] + smoothing)

            oof_target_means[col][val_idx] = (
                X_val_fold[col].map(smooth_mean).fillna(global_target_mean).values
            )
            test_target_means[col] += (
                X_test[col].map(smooth_mean).fillna(global_target_mean).values
                / kf.n_splits
            )

            oof_target_stds[col][val_idx] = (
                X_val_fold[col].map(smooth_std).fillna(global_target_std).values
            )
            test_target_stds[col] += (
                X_test[col].map(smooth_std).fillna(global_target_std).values
                / kf.n_splits
            )

    for col in target_enc_cols:
        X_train[f"{col}_target_mean"] = oof_target_means[col]
        X_test[f"{col}_target_mean"] = test_target_means[col]
        X_train[f"{col}_target_std"] = oof_target_stds[col]
        X_test[f"{col}_target_std"] = test_target_stds[col]

    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(
                    df[col].median() if not pd.isna(df[col].median()) else 0
                )

    feature_names = [c for c in X_train.columns if c.lower() != "id"]

    X_train["fold"] = folds
    X_train["target"] = y_train.values

    X_train.to_parquet("./working/train_processed.parquet", index=False)
    X_test.to_parquet("./working/test_processed.parquet", index=False)

    meta = {
        "feature_names": feature_names,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols + engineered_num_cols,
        "target_col": target_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
    }
    with open("./working/feature_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}, Features count: {len(feature_names)}"
    )


# =========================================================================
# 2. PyTorch Tabular Architecture and Optimizers
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.norm1(self.fc1(x))))
        out = self.drop2(self.act2(self.norm2(self.fc2(out))))
        return residual + out


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            in_d = hidden_dims[i]
            out_d = hidden_dims[i + 1]
            if in_d == out_d:
                blocks.append(ResidualBlock(dim=in_d, dropout_rate=dropout_rate))
            else:
                blocks.append(
                    nn.Sequential(
                        nn.Linear(in_d, out_d),
                        nn.LayerNorm(out_d),
                        nn.SiLU(),
                        nn.Dropout(dropout_rate),
                    )
                )
        self.residual_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        out = self.head(x)
        return out.squeeze(-1)


def get_neural_criterion() -> nn.Module:
    return nn.MSELoss()


def get_neural_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )
    return optimizer, scheduler


# =========================================================================
# 3. Gradient Boosted Decision Tree Model Factories
# =========================================================================
def get_lgbm_regressor(
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return params


def get_xgb_regressor(
    custom_params: Optional[Dict[str, Any]] = None
) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBRegressor(**params)


def get_catboost_regressor(custom_params: Optional[Dict[str, Any]] = None) -> Any:
    if not HAS_CATBOOST:
        return None
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        params.update(custom_params)
    return cb.CatBoostRegressor(**params)


# =========================================================================
# 4. Training, Out-of-Fold Evaluation, and Blending Pipeline
# =========================================================================
def train_and_evaluate():
    seed_everything(42)

    print("Loading processed datasets...")
    train_df = pd.read_parquet("./working/train_processed.parquet")
    test_df = pd.read_parquet("./working/test_processed.parquet")

    y = train_df["target"].values.astype(np.float64)
    folds = train_df["fold"].values
    n_folds = len(np.unique(folds))

    drop_cols = ["id", "fold", "target"]
    feature_cols = [
        c for c in train_df.columns if c not in drop_cols and c in test_df.columns
    ]

    X = train_df[feature_cols].copy().astype(np.float32)
    X_test = test_df[feature_cols].copy().astype(np.float32)

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df), dtype=np.int64)

    n_train, n_features = X.shape
    n_test = len(X_test)
    print(
        f"Training data: {n_train} samples, {n_features} features across {n_folds} folds. Test data: {n_test} samples."
    )

    models_to_train = ["lgb", "xgb", "resnet"]
    if HAS_CATBOOST:
        models_to_train.append("catboost")

    oof_preds = {m: np.zeros(n_train, dtype=np.float64) for m in models_to_train}
    test_preds = {m: np.zeros(n_test, dtype=np.float64) for m in models_to_train}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(n_folds):
        trn_mask = folds != fold
        val_mask = folds == fold

        X_trn, y_trn = X[trn_mask].copy(), y[trn_mask]
        X_val, y_val = X[val_mask].copy(), y[val_mask]

        # -------------------------------------------------------------
        # A. LightGBM Regressor
        # -------------------------------------------------------------
        lgb_params = get_lgbm_regressor(
            {"random_state": 42 + fold, "n_estimators": 3000}
        )
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_trn,
            y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=60, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        lgb_val_pred = model_lgb.predict(X_val)
        lgb_test_pred = model_lgb.predict(X_test)
        oof_preds["lgb"][val_mask] = lgb_val_pred
        test_preds["lgb"] += lgb_test_pred / n_folds

        # -------------------------------------------------------------
        # B. XGBoost Regressor
        # -------------------------------------------------------------
        model_xgb = get_xgb_regressor(
            {
                "random_state": 42 + fold,
                "n_estimators": 3000,
                "early_stopping_rounds": 60,
            }
        )
        model_xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)

        xgb_val_pred = model_xgb.predict(X_val)
        xgb_test_pred = model_xgb.predict(X_test)
        oof_preds["xgb"][val_mask] = xgb_val_pred
        test_preds["xgb"] += xgb_test_pred / n_folds

        # -------------------------------------------------------------
        # C. CatBoost Regressor
        # -------------------------------------------------------------
        if HAS_CATBOOST:
            model_cb = get_catboost_regressor(
                {
                    "random_seed": 42 + fold,
                    "iterations": 3000,
                    "early_stopping_rounds": 60,
                }
            )
            model_cb.fit(
                X_trn,
                y_trn,
                eval_set=(X_val, y_val),
                verbose=False,
                use_best_model=True,
            )

            cb_val_pred = model_cb.predict(X_val)
            cb_test_pred = model_cb.predict(X_test)
            oof_preds["catboost"][val_mask] = cb_val_pred
            test_preds["catboost"] += cb_test_pred / n_folds

        # -------------------------------------------------------------
        # D. PyTorch Tabular Residual Network
        # -------------------------------------------------------------
        scaler_X = StandardScaler()
        X_trn_scaled = scaler_X.fit_transform(X_trn)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_trn_scaled = scaler_y.fit_transform(y_trn.reshape(-1, 1)).squeeze(-1)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).squeeze(-1)

        batch_size = min(128, len(X_trn))
        trn_dataset = TensorDataset(
            torch.tensor(X_trn_scaled, dtype=torch.float32),
            torch.tensor(y_trn_scaled, dtype=torch.float32),
        )
        trn_loader = DataLoader(
            trn_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(trn_dataset) > batch_size),
        )

        nn_model = TabularResNet(
            input_dim=n_features,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.15,
        ).to(device)

        nn_optimizer, nn_scheduler = get_neural_optimizer_and_scheduler(
            nn_model, lr=2e-3, weight_decay=1e-4, epochs=80
        )
        nn_criterion = get_neural_criterion()

        best_val_loss = float("inf")
        best_nn_val_preds = None
        best_nn_test_preds = None
        patience = 20
        patience_counter = 0

        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

        for epoch in range(80):
            nn_model.train()
            for bx, by in trn_loader:
                bx, by = bx.to(device), by.to(device)
                nn_optimizer.zero_grad()
                out = nn_model(bx)
                loss = nn_criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                nn_optimizer.step()
            nn_scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_out = nn_model(X_val_t).cpu().numpy()
                val_loss = np.mean((val_out - y_val_scaled) ** 2)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_nn_val_preds = scaler_y.inverse_transform(
                        val_out.reshape(-1, 1)
                    ).squeeze(-1)
                    test_out = nn_model(X_test_t).cpu().numpy()
                    best_nn_test_preds = scaler_y.inverse_transform(
                        test_out.reshape(-1, 1)
                    ).squeeze(-1)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        oof_preds["resnet"][val_mask] = best_nn_val_preds
        test_preds["resnet"] += best_nn_test_preds / n_folds

        f_lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
        f_xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
        f_nn_rmse = np.sqrt(mean_squared_error(y_val, best_nn_val_preds))
        print(
            f"Fold {fold + 1}/{n_folds} - LGB RMSE: {f_lgb_rmse:.4f} | XGB RMSE: {f_xgb_rmse:.4f} | NN RMSE: {f_nn_rmse:.4f}"
        )

    print("\n--- Model Out-Of-Fold Summary ---")
    for m in models_to_train:
        m_rmse = np.sqrt(mean_squared_error(y, oof_preds[m]))
        print(f"Model [{m.upper()}] Full OOF RMSE: {m_rmse:.6f}")

    M = np.column_stack([oof_preds[m] for m in models_to_train])
    M_test = np.column_stack([test_preds[m] for m in models_to_train])

    def loss_func(weights):
        pred = M @ weights
        return np.sqrt(np.mean((y - pred) ** 2))

    init_weights = np.ones(len(models_to_train)) / len(models_to_train)
    bounds = [(0.0, 1.0) for _ in range(len(models_to_train))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x if opt_res.success else init_weights

    print(
        f"\nOptimal Ensemble Weights: "
        + ", ".join([f"{m}: {w:.4f}" for m, w in zip(models_to_train, best_weights)])
    )

    final_oof_preds = M @ best_weights
    final_test_preds = M_test @ best_weights

    final_validation_score = float(np.sqrt(np.mean((y - final_oof_preds) ** 2)))
    print(f"Ensemble Out-of-Fold RMSE: {final_validation_score:.6f}")

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame({"id": test_ids, "State Share": final_test_preds})

    assert (
        len(sub_df) == n_test
    ), f"Mismatch in submission rows: {len(sub_df)} vs {n_test}"
    assert (
        not sub_df["State Share"].isnull().any()
    ), "Submission contains null predictions!"
    assert not np.isinf(
        sub_df["State Share"]
    ).any(), "Submission contains infinite values!"

    sub_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with shape {sub_df.shape}.")

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    create_feature_pipeline()
    train_and_evaluate()