import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def compute_psychrometrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical thermodynamic variables for temperature/humidity pairs."""
    for i in range(1, 10):
        t_col = f"T{i}"
        rh_col = f"RH_{i}"
        if t_col in df.columns and rh_col in df.columns:
            es = 0.61078 * np.exp((17.27 * df[t_col]) / (df[t_col] + 237.3))
            ea = es * (df[rh_col] / 100.0)
            ah = (216.7 * ea) / (df[t_col] + 273.15)
            df[f"AH_{i}"] = ah
            df[f"Ea_{i}"] = ea

    if "T_out" in df.columns and "RH_out" in df.columns:
        es_out = 0.61078 * np.exp((17.27 * df["T_out"]) / (df["T_out"] + 237.3))
        ea_out = es_out * (df["RH_out"] / 100.0)
        ah_out = (216.7 * ea_out) / (df["T_out"] + 273.15)
        df["AH_out"] = ah_out
        df["Ea_out"] = ea_out

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering function applied cleanly per dataset."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        dates = df["date"].astype(str).str.replace(r"(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})", r"\1 \2", regex=True)
        df["date"] = pd.to_datetime(dates, format="mixed")

    minute_of_day = df["date"].dt.hour * 60 + df["date"].dt.minute
    df["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    day_of_week = df["date"].dt.dayofweek
    df["sin_day"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["cos_day"] = np.cos(2 * np.pi * day_of_week / 7.0)

    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df = compute_psychrometrics(df)

    temp_cols = [f"T{i}" for i in range(1, 10) if f"T{i}" in df.columns]
    rh_cols = [f"RH_{i}" for i in range(1, 10) if f"RH_{i}" in df.columns]
    ah_cols = [f"AH_{i}" for i in range(1, 10) if f"AH_{i}" in df.columns]

    df["T_mean"] = df[temp_cols].mean(axis=1)
    df["T_std"] = df[temp_cols].std(axis=1)
    df["T_min"] = df[temp_cols].min(axis=1)
    df["T_max"] = df[temp_cols].max(axis=1)
    df["T_range"] = df["T_max"] - df["T_min"]

    df["RH_mean"] = df[rh_cols].mean(axis=1)
    df["RH_std"] = df[rh_cols].std(axis=1)
    df["RH_min"] = df[rh_cols].min(axis=1)
    df["RH_max"] = df[rh_cols].max(axis=1)
    df["RH_range"] = df["RH_max"] - df["RH_min"]

    df["AH_mean"] = df[ah_cols].mean(axis=1)

    if "T_out" in df.columns:
        df["T_diff_out"] = df["T_mean"] - df["T_out"]
    if "RH_out" in df.columns:
        df["RH_diff_out"] = df["RH_mean"] - df["RH_out"]
    if "Tdewpoint" in df.columns and "T_out" in df.columns:
        df["Dew_point_gap"] = df["T_out"] - df["Tdewpoint"]

    noise_cols = ["rv1", "rv2"]
    df = df.drop(columns=[c for c in noise_cols if c in df.columns])

    return df


def add_sequence_lags(df_all: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal deltas and lag features across continuous timeline without leakage."""
    lag_cols = [
        "T1",
        "T_out",
        "RH_1",
        "RH_out",
        "lights",
        "Press_mm_hg",
        "Windspeed",
        "T_mean",
        "RH_mean",
    ]
    lag_cols = [c for c in lag_cols if c in df_all.columns]

    for col in lag_cols:
        df_all[f"{col}_lag1"] = df_all[col].shift(1)
        df_all[f"{col}_lag2"] = df_all[col].shift(2)
        df_all[f"{col}_diff1"] = df_all[col] - df_all[f"{col}_lag1"]

        df_all[f"{col}_roll_mean3"] = (
            df_all[col].rolling(window=3, min_periods=1).mean()
        )
        df_all[f"{col}_roll_std3"] = (
            df_all[col].rolling(window=3, min_periods=1).std().fillna(0)
        )

    df_all = df_all.bfill().ffill()
    return df_all


class TabularResBlock(nn.Module):
    """Residual block for tabular data with LayerNorm, GELU, and Dropout regularization."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular ResNet with skip connections for learning smooth continuous features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_nn_model_and_optimizer(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.15
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def build_gbm_models(seed: int = 42):
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    cat_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": seed,
        "verbose": False,
    }

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    xgb_model = xgb.XGBRegressor(**xgb_params)
    cat_model = cb.CatBoostRegressor(**cat_params)

    return lgb_model, xgb_model, cat_model


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))

    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)

    split_train_len = len(train_feat)
    target_col = "Appliances"
    y_full = train_feat[target_col].values

    feature_cols = [
        c
        for c in train_feat.columns
        if c not in [target_col, "date", "id"] and c in test_feat.columns
    ]

    train_feat = add_sequence_lags(train_feat)
    test_feat = add_sequence_lags(test_feat)

    feature_cols = [
        c
        for c in train_feat.columns
        if c not in [target_col, "date", "id"] and c in test_feat.columns
    ]

    X_train_full = train_feat[feature_cols].copy()
    X_test = test_feat[feature_cols].copy()

    val_size = int(len(X_train_full) * 0.2)
    train_size = len(X_train_full) - val_size

    X_train = X_train_full.iloc[:train_size].reset_index(drop=True)
    y_train = y_full[:train_size]

    X_val = X_train_full.iloc[train_size:].reset_index(drop=True)
    y_val = y_full[train_size:]

    X_test = X_test.reset_index(drop=True)

    lgb_model, xgb_model, cat_model = build_gbm_models(seed=42)

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_preds_lgb = lgb_model.predict(X_val)
    test_preds_lgb = lgb_model.predict(X_test)

    xgb_model.set_params(early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds_xgb = xgb_model.predict(X_val)
    test_preds_xgb = xgb_model.predict(X_test)

    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_preds_cat = cat_model.predict(X_val)
    test_preds_cat = cat_model.predict(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    nn_model, criterion, optimizer = build_nn_model_and_optimizer(
        input_dim=X_train.shape[1], lr=1e-3, weight_decay=1e-4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model.to(device)

    best_val_loss = float("inf")
    best_nn_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, 101):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
        val_loss /= len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_state = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_nn_state)
    nn_model.to(device)
    nn_model.eval()

    with torch.no_grad():
        val_preds_nn = (
            nn_model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
        )
        test_preds_nn = (
            nn_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
        )

    val_preds_matrix = np.column_stack(
        [val_preds_lgb, val_preds_xgb, val_preds_cat, val_preds_nn]
    )
    test_preds_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
    )

    def objective(weights):
        blend = val_preds_matrix @ weights
        return np.sqrt(mean_squared_error(y_val, blend))

    init_weights = np.ones(4) / 4.0
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = opt.minimize(
        objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )
    best_weights = opt_res.x / np.sum(opt_res.x)

    val_blend = np.clip(val_preds_matrix @ best_weights, a_min=0, a_max=None)
    test_blend = np.clip(test_preds_matrix @ best_weights, a_min=0, a_max=None)

    final_val_rmse = np.sqrt(mean_squared_error(y_val, val_blend))

    submission_df = pd.DataFrame({"id": test_ids, "Appliances": test_blend})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_rmse}")


if __name__ == "__main__":
    main()