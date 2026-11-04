import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error


# =========================================================================
# 1. Model Definitions & Architectures
# =========================================================================
def get_lgb_model_params(random_state: int = 42) -> Dict[str, Any]:
    """Returns optimal LightGBM hyperparameters for tabular RMSE regression."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 2500,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(random_state: int = 42) -> Dict[str, Any]:
    """Returns optimal XGBoost hyperparameters for tabular RMSE regression."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 2500,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def build_lgb_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    """Instantiates a LightGBM Regressor."""
    params = get_lgb_model_params(random_state=random_state)
    return lgb.LGBMRegressor(**params)


def build_xgb_regressor(random_state: int = 42) -> xgb.XGBRegressor:
    """Instantiates an XGBoost Regressor."""
    params = get_xgb_model_params(random_state=random_state)
    return xgb.XGBRegressor(**params)


class TabularNeuralNetwork(nn.Module):
    """Deep Tabular Neural Network with Entity Embeddings, LayerNorm,
    Dropout, and Residual Projections for regression.
    """

    def __init__(
        self,
        embedding_sizes: List[Tuple[int, int]],
        num_continuous_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        super(TabularNeuralNetwork, self).__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, emb_dim)
                for num_classes, emb_dim in embedding_sizes
            ]
        )
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_sizes)

        self.num_bn = (
            nn.BatchNorm1d(num_continuous_features)
            if num_continuous_features > 0
            else None
        )

        in_dim = total_emb_dim + num_continuous_features
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor = None) -> torch.Tensor:
        features = []
        if len(self.embeddings) > 0 and x_cat is not None and x_cat.shape[1] > 0:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            features.append(torch.cat(emb_outs, dim=1))

        if x_cont is not None and x_cont.shape[1] > 0:
            if self.num_bn is not None:
                x_cont = self.num_bn(x_cont)
            features.append(x_cont)

        x = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        hidden = self.mlp(x)
        out = self.head(hidden).squeeze(-1)
        return out


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss faithful to competition metric."""

    def __init__(self, eps: float = 1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def get_torch_criterion_and_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Defines loss function, AdamW optimizer, and Cosine Annealing scheduler."""
    criterion = RMSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    return criterion, optimizer, scheduler


# =========================================================================
# 2. Main Processing, Training, and Evaluation Pipeline
# =========================================================================
def main():
    input_dir = "./input"
    working_dir = "./working"
    sub_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "usdprice"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    # Identify and parse date/temporal columns
    date_cols = [
        col
        for col in train_df.columns
        if "date" in col.lower() or "time" in col.lower() or "year" in col.lower()
    ]
    for col in date_cols:
        if train_df[col].dtype == "object" or np.issubdtype(
            train_df[col].dtype, np.datetime64
        ):
            try:
                train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
                test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
            except Exception:
                pass

    # Create validation split (temporal if valid datetime exists, otherwise random)
    primary_date_col = None
    for col in date_cols:
        if np.issubdtype(train_df[col].dtype, np.datetime64) and train_df[
            col
        ].notnull().sum() > 0.8 * len(train_df):
            primary_date_col = col
            break

    if primary_date_col is not None:
        train_df = train_df.sort_values(by=primary_date_col).reset_index(drop=True)
        split_idx = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_idx].copy().reset_index(drop=True)
        val_split = train_df.iloc[split_idx:].copy().reset_index(drop=True)
    else:
        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=42
        )
        train_split = train_split.reset_index(drop=True)
        val_split = val_split.reset_index(drop=True)

    y_train = train_split[target_col].copy()
    y_val = val_split[target_col].copy()
    X_train_raw = train_split.drop(columns=[target_col]).copy()
    X_val_raw = val_split.drop(columns=[target_col]).copy()
    X_test_raw = test_df.copy()
    if target_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[target_col])

    # Extract Datetime Features
    def extract_datetime_features(df):
        df_out = df.copy()
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                dt = df[col].dt
                df_out[f"{col}_year"] = dt.year.astype("float32")
                df_out[f"{col}_month"] = dt.month.astype("float32")
                df_out[f"{col}_day"] = dt.day.astype("float32")
                df_out[f"{col}_dayofweek"] = dt.dayofweek.astype("float32")
                df_out[f"{col}_quarter"] = dt.quarter.astype("float32")
                df_out[f"{col}_is_month_end"] = dt.is_month_end.astype("float32")
                df_out[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.month / 12.0).astype(
                    "float32"
                )
                df_out[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.month / 12.0).astype(
                    "float32"
                )
                df_out = df_out.drop(columns=[col])
        return df_out

    X_train = extract_datetime_features(X_train_raw)
    X_val = extract_datetime_features(X_val_raw)
    X_test = extract_datetime_features(X_test_raw)

    # Clean string/categorical columns & extract categorical pairs
    cat_cols = [
        col
        for col in X_train.columns
        if X_train[col].dtype == "object" or X_train[col].dtype.name == "category"
    ]

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).str.strip().str.lower()
        X_val[c] = X_val[c].astype(str).str.strip().str.lower()
        X_test[c] = X_test[c].astype(str).str.strip().str.lower()

    if len(cat_cols) >= 2:
        top_cats = cat_cols[: min(4, len(cat_cols))]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                col_a, col_b = top_cats[i], top_cats[j]
                inter_name = f"{col_a}_x_{col_b}"
                X_train[inter_name] = X_train[col_a] + "__" + X_train[col_b]
                X_val[inter_name] = X_val[col_a] + "__" + X_val[col_b]
                X_test[inter_name] = X_test[col_a] + "__" + X_test[col_b]

    cat_cols = [
        col
        for col in X_train.columns
        if X_train[col].dtype == "object" or X_train[col].dtype.name == "category"
    ]

    # Frequency Encodings (fit strictly on train)
    for c in cat_cols:
        freq_map = X_train[c].value_counts(normalize=True).to_dict()
        X_train[f"{c}_freq"] = X_train[c].map(freq_map).fillna(0).astype("float32")
        X_val[f"{c}_freq"] = X_val[c].map(freq_map).fillna(0).astype("float32")
        X_test[f"{c}_freq"] = X_test[c].map(freq_map).fillna(0).astype("float32")

    # Out-of-fold Smoothed Target Encoding (fit strictly on train)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = float(y_train.mean())
    smooth_weight = 10.0

    for c in cat_cols:
        te_col_name = f"{c}_target_enc"
        X_train[te_col_name] = np.nan

        for tr_idx, v_idx in kf.split(X_train):
            fold_df = pd.DataFrame(
                {"cat": X_train.iloc[tr_idx][c], "target": y_train.iloc[tr_idx]}
            )
            stats = fold_df.groupby("cat")["target"].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"] + smooth_weight * global_mean
            ) / (stats["count"] + smooth_weight)
            X_train.iloc[v_idx, X_train.columns.get_loc(te_col_name)] = (
                X_train.iloc[v_idx][c]
                .map(smoothed)
                .fillna(global_mean)
                .astype("float32")
            )

        full_df = pd.DataFrame({"cat": X_train[c], "target": y_train})
        stats_full = full_df.groupby("cat")["target"].agg(["count", "mean"])
        smoothed_full = (
            stats_full["count"] * stats_full["mean"] + smooth_weight * global_mean
        ) / (stats_full["count"] + smooth_weight)
        X_val[te_col_name] = (
            X_val[c].map(smoothed_full).fillna(global_mean).astype("float32")
        )
        X_test[te_col_name] = (
            X_test[c].map(smoothed_full).fillna(global_mean).astype("float32")
        )
        X_train[te_col_name] = (
            X_train[te_col_name].fillna(global_mean).astype("float32")
        )

    # Numeric Group Aggregations (fit strictly on train)
    num_cols = [
        col
        for col in X_train.columns
        if np.issubdtype(X_train[col].dtype, np.number)
        and not col.endswith(("_freq", "_target_enc"))
    ]
    if len(cat_cols) > 0 and len(num_cols) > 0:
        primary_cats = cat_cols[: min(3, len(cat_cols))]
        primary_nums = num_cols[: min(3, len(num_cols))]
        for c in primary_cats:
            for num in primary_nums:
                agg_stats = X_train.groupby(c)[num].agg(["mean", "std"]).reset_index()
                agg_stats.columns = [c, f"{c}_{num}_mean", f"{c}_{num}_std"]

                X_train = X_train.merge(agg_stats, on=c, how="left")
                X_val = X_val.merge(agg_stats, on=c, how="left")
                X_test = X_test.merge(agg_stats, on=c, how="left")

                mean_val = float(X_train[f"{c}_{num}_mean"].mean())
                X_train[f"{c}_{num}_mean"] = (
                    X_train[f"{c}_{num}_mean"].fillna(mean_val).astype("float32")
                )
                X_val[f"{c}_{num}_mean"] = (
                    X_val[f"{c}_{num}_mean"].fillna(mean_val).astype("float32")
                )
                X_test[f"{c}_{num}_mean"] = (
                    X_test[f"{c}_{num}_mean"].fillna(mean_val).astype("float32")
                )

                X_train[f"{c}_{num}_std"] = (
                    X_train[f"{c}_{num}_std"].fillna(0).astype("float32")
                )
                X_val[f"{c}_{num}_std"] = (
                    X_val[f"{c}_{num}_std"].fillna(0).astype("float32")
                )
                X_test[f"{c}_{num}_std"] = (
                    X_test[f"{c}_{num}_std"].fillna(0).astype("float32")
                )

    # Convert categorical columns to category dtypes for LightGBM
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = pd.Categorical(X_val[c], categories=X_train[c].cat.categories)
        X_test[c] = pd.Categorical(X_test[c], categories=X_train[c].cat.categories)

    # Multi-seed LightGBM Model Training and Ensembling
    seeds = [42, 2023, 777]
    val_preds_list = []
    test_preds_list = []
    models = []

    y_train_arr = y_train.values
    y_val_arr = y_val.values

    for seed in seeds:
        model = build_lgb_regressor(random_state=seed)
        model.fit(
            X_train,
            y_train_arr,
            eval_set=[(X_val, y_val_arr)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_preds_list.append(val_pred)
        test_preds_list.append(test_pred)
        models.append(model)

    # Ensemble predictions and enforce physical non-negative constraint
    val_preds_ensemble = np.mean(val_preds_list, axis=0)
    test_preds_ensemble = np.mean(test_preds_list, axis=0)

    val_preds_final = np.clip(val_preds_ensemble, a_min=0.0, a_max=None)
    test_preds_final = np.clip(test_preds_ensemble, a_min=0.0, a_max=None)

    # Compute official evaluation metric (RMSE)
    val_rmse = float(np.sqrt(mean_squared_error(y_val_arr, val_preds_final)))

    # Save artifacts and output submission
    joblib.dump(models[0], os.path.join(working_dir, "lgb_best_model.joblib"))

    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_preds_final), dtype=int),
            "usdprice": test_preds_final,
        }
    )
    submission_path = os.path.join(sub_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {val_rmse}")


if __name__ == "__main__":
    main()
