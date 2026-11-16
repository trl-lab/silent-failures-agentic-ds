import json
import os
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

# =========================================================================
# 1. Feature Engineering & Preprocessing
# =========================================================================


def load_and_preprocess_data():
    """
    Data loading, anomaly cleaning, feature engineering, and leak-free split generation
    for Metro Interstate Traffic Volume Prediction.
    """
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load raw datasets
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "traffic_volume"

    # 2. Data Cleaning & Anomaly Handling
    for df in [train_df, test_df]:
        df.loc[df["temp"] < 200.0, "temp"] = np.nan
        df.loc[df["rain_1h"] > 100.0, "rain_1h"] = 100.0
        df["snow_1h"] = df["snow_1h"].clip(lower=0.0, upper=50.0)
        df["clouds_all"] = df["clouds_all"].clip(lower=0, upper=100)

    # Impute missing/cleaned temp with median computed strictly on training data
    train_temp_median = train_df["temp"].median()
    train_df["temp"] = train_df["temp"].fillna(train_temp_median)
    test_df["temp"] = test_df["temp"].fillna(train_temp_median)

    # 3. Comprehensive Feature Engineering
    def extract_features(df):
        data = df.copy()
        dt = pd.to_datetime(data["date_time"])

        data["year"] = dt.dt.year.astype(np.int32)
        data["month"] = dt.dt.month.astype(np.int32)
        data["day"] = dt.dt.day.astype(np.int32)
        data["hour"] = dt.dt.hour.astype(np.int32)
        data["dayofweek"] = dt.dt.dayofweek.astype(np.int32)
        data["dayofyear"] = dt.dt.dayofyear.astype(np.int32)
        data["quarter"] = dt.dt.quarter.astype(np.int32)
        data["weekofyear"] = dt.dt.isocalendar().week.astype(np.int32)

        # Cyclical temporal representations
        data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0).astype(np.float32)
        data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0).astype(np.float32)
        data["sin_dayofweek"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0).astype(
            np.float32
        )
        data["cos_dayofweek"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0).astype(
            np.float32
        )
        data["sin_month"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0).astype(
            np.float32
        )
        data["cos_month"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0).astype(
            np.float32
        )
        data["sin_dayofyear"] = np.sin(
            2 * np.pi * (data["dayofyear"] - 1) / 365.25
        ).astype(np.float32)
        data["cos_dayofyear"] = np.cos(
            2 * np.pi * (data["dayofyear"] - 1) / 365.25
        ).astype(np.float32)

        # 168-hour composite weekly hour
        data["hour_of_week"] = (data["dayofweek"] * 24 + data["hour"]).astype(np.int32)

        data["is_weekend"] = (data["dayofweek"] >= 5).astype(np.int32)
        data["is_month_start"] = dt.dt.is_month_start.astype(np.int32)
        data["is_month_end"] = dt.dt.is_month_end.astype(np.int32)

        # Peak hours
        data["is_morning_rush"] = (
            (data["hour"] >= 6) & (data["hour"] <= 9) & (data["is_weekend"] == 0)
        ).astype(np.int32)
        data["is_evening_rush"] = (
            (data["hour"] >= 15) & (data["hour"] <= 18) & (data["is_weekend"] == 0)
        ).astype(np.int32)
        data["is_rush_hour"] = (
            data["is_morning_rush"] | data["is_evening_rush"]
        ).astype(np.int32)
        data["is_daytime"] = ((data["hour"] >= 6) & (data["hour"] <= 20)).astype(
            np.int32
        )
        data["is_night"] = ((data["hour"] < 6) | (data["hour"] >= 22)).astype(np.int32)

        # Holiday features
        holiday_str = data["holiday"].astype(str).fillna("None").replace("nan", "None")
        data["is_holiday"] = (holiday_str != "None").astype(np.int32)
        data["holiday_clean"] = holiday_str

        # Weather derivations
        data["temp_celsius"] = (data["temp"] - 273.15).astype(np.float32)
        data["is_freezing"] = (data["temp_celsius"] < 0.0).astype(np.int32)
        data["is_raining"] = (data["rain_1h"] > 0.0).astype(np.int32)
        data["is_snowing"] = (data["snow_1h"] > 0.0).astype(np.int32)
        data["is_precipitating"] = (
            (data["rain_1h"] > 0.0) | (data["snow_1h"] > 0.0)
        ).astype(np.int32)

        data["log_rain_1h"] = np.log1p(data["rain_1h"]).astype(np.float32)
        data["log_snow_1h"] = np.log1p(data["snow_1h"]).astype(np.float32)

        data["weather_severity_index"] = (
            (data["rain_1h"] * 1.5)
            + (data["snow_1h"] * 3.0)
            + (data["clouds_all"] / 100.0)
            + (data["is_freezing"] * 1.0)
        ).astype(np.float32)

        data["rush_adverse_weather"] = (
            data["is_rush_hour"] * data["is_precipitating"]
        ).astype(np.int32)
        data["rush_freezing"] = (data["is_rush_hour"] * data["is_freezing"]).astype(
            np.int32
        )
        data["rush_weather_severity"] = (
            data["is_rush_hour"] * data["weather_severity_index"]
        ).astype(np.float32)

        data["weather_main"] = (
            data["weather_main"].astype(str).fillna("Unknown").replace("nan", "Unknown")
        )
        data["weather_description"] = (
            data["weather_description"]
            .astype(str)
            .fillna("Unknown")
            .replace("nan", "Unknown")
        )

        # Compound interaction representations
        data["hour_dayofweek"] = (
            data["hour"].astype(str) + "_" + data["dayofweek"].astype(str)
        )
        data["hour_holiday"] = (
            data["hour"].astype(str) + "_" + data["is_holiday"].astype(str)
        )
        data["month_hour"] = (
            data["month"].astype(str) + "_" + data["hour"].astype(str)
        )
        data["weather_hour"] = (
            data["weather_main"].astype(str) + "_" + data["hour"].astype(str)
        )

        return data

    train_fe = extract_features(train_df)
    test_fe = extract_features(test_df)

    # 4. Chronological Holdout Validation Split (last 20% for temporal evaluation)
    train_fe_sorted = train_fe.sort_values("date_time").reset_index(drop=True)
    val_split_idx = int(len(train_fe_sorted) * 0.8)

    train_split = train_fe_sorted.iloc[:val_split_idx].copy().reset_index(drop=True)
    val_split = train_fe_sorted.iloc[val_split_idx:].copy().reset_index(drop=True)

    # 5. Categorical Encodings (Fitted on train_split only)
    cat_cols = ["holiday_clean", "weather_main", "weather_description"]

    for col in cat_cols:
        freq_map = train_split[col].value_counts(normalize=True).to_dict()
        train_split[f"{col}_freq"] = (
            train_split[col].map(freq_map).fillna(0.0).astype(np.float32)
        )
        val_split[f"{col}_freq"] = (
            val_split[col].map(freq_map).fillna(0.0).astype(np.float32)
        )
        test_fe[f"{col}_freq"] = (
            test_fe[col].map(freq_map).fillna(0.0).astype(np.float32)
        )

        unique_cats = sorted(train_split[col].unique().tolist())
        cat2idx = {cat: idx for idx, cat in enumerate(unique_cats)}
        train_split[f"{col}_idx"] = (
            train_split[col].map(lambda x: cat2idx.get(x, len(cat2idx))).astype(np.int32)
        )
        val_split[f"{col}_idx"] = (
            val_split[col].map(lambda x: cat2idx.get(x, len(cat2idx))).astype(np.int32)
        )
        test_fe[f"{col}_idx"] = (
            test_fe[col].map(lambda x: cat2idx.get(x, len(cat2idx))).astype(np.int32)
        )

    # 6. Out-Of-Fold Target Encodings (Fitted on train_split only)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_split["fold"] = -1
    for fold, (t_idx, v_idx) in enumerate(kf.split(train_split)):
        train_split.loc[v_idx, "fold"] = fold

    te_cols = [
        "hour_of_week",
        "hour_dayofweek",
        "hour_holiday",
        "month_hour",
        "weather_hour",
        "weather_main",
        "weather_description",
        "holiday_clean",
    ]
    global_target_mean = train_split[target_col].mean()
    m = 15.0

    for col in te_cols:
        te_col_name = f"{col}_te_mean"
        train_split[te_col_name] = np.nan

        for fold in range(5):
            fold_train = train_split[train_split["fold"] != fold]
            fold_val = train_split[train_split["fold"] == fold]

            stats = fold_train.groupby(col)[target_col].agg(["count", "sum"])
            smoothed_map = (
                (stats["sum"] + m * global_target_mean) / (stats["count"] + m)
            ).to_dict()
            train_split.loc[fold_val.index, te_col_name] = (
                fold_val[col].map(smoothed_map).fillna(global_target_mean)
            )

        train_split[te_col_name] = train_split[te_col_name].astype(np.float32)

        stats_full = train_split.groupby(col)[target_col].agg(["count", "sum"])
        smoothed_full_map = (
            (stats_full["sum"] + m * global_target_mean) / (stats_full["count"] + m)
        ).to_dict()
        val_split[te_col_name] = (
            val_split[col].map(smoothed_full_map).fillna(global_target_mean).astype(np.float32)
        )
        test_fe[te_col_name] = (
            test_fe[col].map(smoothed_full_map).fillna(global_target_mean).astype(np.float32)
        )

    ignore_cols = [
        "date_time",
        "holiday",
        "holiday_clean",
        "weather_main",
        "weather_description",
        "hour_dayofweek",
        "hour_holiday",
        "month_hour",
        "weather_hour",
    ]
    feature_cols = [
        c
        for c in train_fe.columns
        if c not in ignore_cols and c != target_col and c != "fold"
    ]
    categorical_feature_cols = [f"{c}_idx" for c in cat_cols]
    numerical_feature_cols = [c for c in feature_cols if not c.endswith("_idx")]

    return {
        "train_split": train_split,
        "val_split": val_split,
        "test_df": test_fe,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "num_cols": numerical_feature_cols,
        "cat_cols": categorical_feature_cols,
    }


# =========================================================================
# 2. Model Architectures & Configurations
# =========================================================================


def get_lightgbm_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": 0,
    }


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TrafficTabularResNet(nn.Module):

    def __init__(
        self,
        num_numerical_features: int,
        cat_cardinalities: List[int],
        cat_embedding_dims: Optional[List[int]] = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        for idx, card in enumerate(cat_cardinalities):
            if cat_embedding_dims is not None and idx < len(cat_embedding_dims):
                emb_dim = cat_embedding_dims[idx]
            else:
                emb_dim = max(4, min(50, (card + 1) // 2))
            self.embeddings.append(
                nn.Sequential(
                    nn.Embedding(card + 2, emb_dim),
                    nn.LayerNorm(emb_dim),
                )
            )
            total_emb_dim += emb_dim

        in_dim = num_numerical_features + total_emb_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = [x_num]
        if x_cat is not None and len(self.embeddings) > 0:
            for i, emb_layer in enumerate(self.embeddings):
                features.append(emb_layer(x_cat[:, i]))

        x = torch.cat(features, dim=-1)
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


class TrafficDataset(Dataset):

    def __init__(
        self,
        num_features: np.ndarray,
        cat_features: np.ndarray,
        targets: np.ndarray = None,
    ):
        self.num_features = torch.tensor(num_features, dtype=torch.float32)
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self) -> int:
        return len(self.num_features)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return (
                self.num_features[idx],
                self.cat_features[idx],
                self.targets[idx],
            )
        return self.num_features[idx], self.cat_features[idx]


# =========================================================================
# 3. Training & Evaluation Pipeline
# =========================================================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_and_preprocess_data()

    train_df = data["train_split"]
    val_df = data["val_split"]
    test_df = data["test_df"]
    target_col = data["target_col"]
    feature_cols = data["feature_cols"]
    num_cols = data["num_cols"]
    cat_cols = data["cat_cols"]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values.astype(np.float32)

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values.astype(np.float32)

    X_test = test_df[feature_cols].copy()

    # 1. Train LightGBM
    lgb_params = get_lightgbm_params(seed=42)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    lgb_val_preds = lgb_model.predict(X_val)
    lgb_test_preds = lgb_model.predict(X_test)

    # 2. Train XGBoost
    xgb_params = get_xgboost_params(seed=42)
    xgb_params["early_stopping_rounds"] = 50
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_val_preds = xgb_model.predict(X_val)
    xgb_test_preds = xgb_model.predict(X_test)

    # 3. Train CatBoost
    cat_params = get_catboost_params(seed=42)
    cat_params["early_stopping_rounds"] = 50
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    cat_val_preds = cat_model.predict(X_val)
    cat_test_preds = cat_model.predict(X_test)

    # 4. Train Tabular ResNet
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(
        train_df[num_cols].fillna(0).values
    ).astype(np.float32)
    X_val_num_scaled = scaler.transform(val_df[num_cols].fillna(0).values).astype(
        np.float32
    )
    X_test_num_scaled = scaler.transform(test_df[num_cols].fillna(0).values).astype(
        np.float32
    )

    X_train_cat = train_df[cat_cols].values.astype(np.int64)
    X_val_cat = val_df[cat_cols].values.astype(np.int64)
    X_test_cat = test_df[cat_cols].values.astype(np.int64)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std())
    y_train_nn = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_nn = ((y_val - target_mean) / target_std).astype(np.float32)

    cat_cardinalities = [
        int(max(train_df[c].max(), val_df[c].max(), test_df[c].max()) + 1)
        for c in cat_cols
    ]

    train_dataset = TrafficDataset(X_train_num_scaled, X_train_cat, y_train_nn)
    val_dataset = TrafficDataset(X_val_num_scaled, X_val_cat, y_val_nn)
    test_dataset = TrafficDataset(X_test_num_scaled, X_test_cat)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False, drop_last=False
    )

    max_epochs = 30
    nn_model = TrafficTabularResNet(
        num_numerical_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
        output_dim=1,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_nn_val_rmse = float("inf")
    best_weights_path = "./working/best_tabular_resnet.pt"
    patience = 8
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        nn_model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_num, batch_cat, batch_y in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = nn_model(batch_num, batch_cat)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_y)
            train_count += len(batch_y)

        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_num, batch_cat, _ in val_loader:
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)
                preds = nn_model(batch_num, batch_cat)
                val_preds_list.append(preds.cpu().numpy())

        current_val_preds = (
            np.concatenate(val_preds_list, axis=0) * target_std + target_mean
        )
        current_val_rmse = root_mean_squared_error(y_val, current_val_preds)
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} - Val RMSE: {current_val_rmse:.2f}"
        )

        if current_val_rmse < best_nn_val_rmse:
            best_nn_val_rmse = current_val_rmse
            torch.save(nn_model.state_dict(), best_weights_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(
        torch.load(best_weights_path, map_location=device, weights_only=True)
    )
    nn_model.eval()

    val_preds_list = []
    with torch.no_grad():
        for batch_num, batch_cat, _ in val_loader:
            preds = nn_model(batch_num.to(device), batch_cat.to(device))
            val_preds_list.append(preds.cpu().numpy())
    nn_val_preds = (
        np.concatenate(val_preds_list, axis=0) * target_std + target_mean
    )

    test_preds_list = []
    with torch.no_grad():
        for batch_num, batch_cat in test_loader:
            preds = nn_model(batch_num.to(device), batch_cat.to(device))
            test_preds_list.append(preds.cpu().numpy())
    nn_test_preds = (
        np.concatenate(test_preds_list, axis=0) * target_std + target_mean
    )

    # 5. Blending Ensemble Optimization
    val_matrix = np.column_stack(
        [lgb_val_preds, xgb_val_preds, cat_val_preds, nn_val_preds]
    )
    test_matrix = np.column_stack(
        [lgb_test_preds, xgb_test_preds, cat_test_preds, nn_test_preds]
    )

    def blend_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.clip(val_matrix @ w, 0.0, None)
        return root_mean_squared_error(y_val, pred)

    initial_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = res.x / np.sum(res.x)

    final_val_preds = np.clip(val_matrix @ best_weights, 0.0, None)
    final_val_score = root_mean_squared_error(y_val, final_val_preds)
    final_test_preds = np.clip(test_matrix @ best_weights, 0.0, None)

    # 6. Output Submission File
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_preds), dtype=np.int64),
            "traffic_volume": final_test_preds.astype(np.float32),
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()