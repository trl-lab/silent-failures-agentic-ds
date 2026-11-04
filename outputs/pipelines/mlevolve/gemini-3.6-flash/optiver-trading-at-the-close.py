import gc
import os
import random
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------
def compute_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes microstructural order book ratios, spreads, and imbalances."""
    df = df.copy()

    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    df["ask_bid_ratio"] = df["ask_price"] / (df["bid_price"] + 1e-8)
    df["bid_ask_size_diff"] = df["bid_size"] - df["ask_size"]
    df["bid_ask_size_ratio"] = df["bid_size"] / (df["ask_size"] + 1e-8)
    df["price_spread_ref_wap"] = df["reference_price"] - df["wap"]

    df["signed_imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    df["imbalance_ratio"] = df["imbalance_size"] / (df["matched_size"] + 1e-8)
    df["liquidity_imbalance"] = df["bid_ask_size_diff"] / (
        df["bid_size"] + df["ask_size"] + 1e-8
    )
    df["matched_imbalance_total"] = df["imbalance_size"] + df["matched_size"]
    df["volume_imbalance_ratio"] = df["bid_ask_size_diff"] / (
        df["matched_imbalance_total"] + 1e-8
    )

    df["far_price_wap_diff"] = df["far_price"] - df["wap"]
    df["near_price_wap_diff"] = df["near_price"] - df["wap"]

    return df


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes cross-sectional market aggregation features per (date_id, seconds_in_bucket)."""
    time_group = df.groupby(["date_id", "seconds_in_bucket"])

    df["market_wap_mean"] = time_group["wap"].transform("mean")
    df["market_wap_std"] = time_group["wap"].transform("std")
    df["market_imbalance_sum"] = time_group["signed_imbalance"].transform("sum")
    df["market_spread_mean"] = time_group["bid_ask_spread"].transform("mean")
    df["market_matched_sum"] = time_group["matched_size"].transform("sum")

    df["wap_market_diff"] = df["wap"] - df["market_wap_mean"]
    df["spread_market_diff"] = df["bid_ask_spread"] - df["market_spread_mean"]
    df["imbalance_market_share"] = df["signed_imbalance"] / (
        df["market_imbalance_sum"].abs() + 1e-8
    )

    df["stock_market_wap_ratio"] = df["wap"] / (df["market_wap_mean"] + 1e-8)
    df["stock_market_spread_ratio"] = df["bid_ask_spread"] / (
        df["market_spread_mean"] + 1e-8
    )
    df["stock_market_imbalance_ratio"] = df["signed_imbalance"] / (
        df["market_imbalance_sum"].abs() + 1e-8
    )
    df["stock_market_pressure_ratio"] = (
        df["signed_imbalance"] / (df["matched_size"] + 1e-8)
    ) / (
        (df["market_imbalance_sum"] / (df["market_matched_sum"] + 1e-8)).abs()
        + 1e-8
    )

    return df


def compute_temporal_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes intra-day sequential lag and momentum features per (stock_id, date_id)."""
    df = df.sort_values(by=["stock_id", "date_id", "seconds_in_bucket"]).reset_index(
        drop=True
    )
    stock_day_group = df.groupby(["stock_id", "date_id"])

    for lag in [1, 2, 3, 6]:
        df[f"wap_lag_{lag}"] = stock_day_group["wap"].shift(lag)
        df[f"spread_lag_{lag}"] = stock_day_group["bid_ask_spread"].shift(lag)
        df[f"imbalance_lag_{lag}"] = stock_day_group["signed_imbalance"].shift(lag)

        df[f"wap_return_{lag}"] = (df["wap"] - df[f"wap_lag_{lag}"]) / (
            df[f"wap_lag_{lag}"] + 1e-8
        )
        df[f"spread_diff_{lag}"] = df["bid_ask_spread"] - df[f"spread_lag_{lag}"]

    df["wap_rolling_mean_3"] = stock_day_group["wap"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["wap_rolling_std_3"] = (
        stock_day_group["wap"]
        .transform(lambda x: x.rolling(3, min_periods=1).std())
        .fillna(0)
    )
    df["spread_rolling_mean_3"] = stock_day_group["bid_ask_spread"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    for window in [6, 12]:
        for col in ["wap", "bid_ask_spread", "signed_imbalance"]:
            df[f"{col}_roll_mean_{window}"] = stock_day_group[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{window}"] = stock_day_group[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).fillna(0)
            df[f"{col}_roll_min_{window}"] = stock_day_group[col].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f"{col}_roll_max_{window}"] = stock_day_group[col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

    return df


# -----------------------------------------------------------------------------
# PyTorch Model Components
# -----------------------------------------------------------------------------
class TabularResBlock(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_features),
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FinancialTabularResNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(
                    in_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        return self.head(out)


class PyTorchTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = (
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# -----------------------------------------------------------------------------
# GBDT Model Constructors
# -----------------------------------------------------------------------------
def get_lightgbm_params():
    return {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def build_xgboost_model():
    return xgb.XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_estimators=1500,
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )


def build_catboost_model():
    return cb.CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        iterations=1500,
        early_stopping_rounds=50,
        random_seed=SEED,
        verbose=False,
    )


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Clean missing target rows
    train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)

    # Feature Engineering
    train_df = compute_order_book_features(train_df)
    train_df = compute_cross_sectional_features(train_df)
    train_df = compute_temporal_lag_features(train_df)

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    test_df = compute_order_book_features(test_df)
    test_df = compute_cross_sectional_features(test_df)
    test_df = compute_temporal_lag_features(test_df)

    ignore_cols = ["id", "row_id", "target", "time_id"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    # Split train and validation by time (date_id)
    train_mask = train_df["date_id"] <= 420
    val_mask = train_df["date_id"] > 420

    train_split = train_df[train_mask].reset_index(drop=True)
    val_split = train_df[val_mask].reset_index(drop=True)

    # Impute missing feature values using train split medians
    train_medians = train_split[feature_cols].median()

    X_train = train_split[feature_cols].fillna(train_medians).values.astype(np.float32)
    y_train = train_split["target"].values.astype(np.float32)

    X_val = val_split[feature_cols].fillna(train_medians).values.astype(np.float32)
    y_val = val_split["target"].values.astype(np.float32)

    X_test = test_df[feature_cols].fillna(train_medians).values.astype(np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    del train_df, train_split, val_split
    gc.collect()

    # 1. Train LightGBM
    lgb_params = get_lightgbm_params()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_preds_lgb = lgb_model.predict(X_val)
    test_preds_lgb = lgb_model.predict(X_test)
    mae_lgb = mean_absolute_error(y_val, val_preds_lgb)

    # 2. Train XGBoost
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds_xgb = xgb_model.predict(X_val)
    test_preds_xgb = xgb_model.predict(X_test)
    mae_xgb = mean_absolute_error(y_val, val_preds_xgb)

    # 3. Train CatBoost
    cb_model = build_catboost_model()
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    val_preds_cb = cb_model.predict(X_val)
    test_preds_cb = cb_model.predict(X_test)
    mae_cb = mean_absolute_error(y_val, val_preds_cb)

    # 4. Train PyTorch Tabular ResNet
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_ds = PyTorchTabularDataset(X_train_scaled, y_train)
    val_ds = PyTorchTabularDataset(X_val_scaled, y_val)
    test_ds = PyTorchTabularDataset(X_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, drop_last=False)

    nn_model = FinancialTabularResNet(
        num_features=X_train.shape[1],
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_nn_val_mae = float("inf")
    os.makedirs("./working", exist_ok=True)
    best_nn_model_path = "./working/best_nn_model.pth"

    for epoch in range(1, 11):
        nn_model.train()
        total_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(bx)

        train_loss = total_train_loss / len(train_ds)

        nn_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                total_val_loss += criterion(out, by).item() * len(bx)

        val_loss = total_val_loss / len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_nn_val_mae:
            best_nn_val_mae = val_loss
            torch.save(nn_model.state_dict(), best_nn_model_path)

        print(
            f"Epoch {epoch:02d} | Train MAE: {train_loss:.5f} | Val MAE:"
            f" {val_loss:.5f}"
        )

    nn_model.load_state_dict(torch.load(best_nn_model_path))
    nn_model.eval()

    val_preds_nn_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            out = nn_model(bx)
            val_preds_nn_list.append(out.cpu().numpy().flatten())
    val_preds_nn = np.concatenate(val_preds_nn_list)

    test_preds_nn_list = []
    with torch.no_grad():
        for bx in test_loader:
            bx = bx.to(device)
            out = nn_model(bx)
            test_preds_nn_list.append(out.cpu().numpy().flatten())
    test_preds_nn = np.concatenate(test_preds_nn_list)

    mae_nn = mean_absolute_error(y_val, val_preds_nn)

    # Optimal MAE Weighting Ensemble via scipy.optimize.minimize
    val_preds_matrix = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_cb, val_preds_nn])
    test_preds_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_nn])

    def ensemble_mae_objective(weights):
        w = weights / np.sum(weights)
        pred = val_preds_matrix @ w
        return mean_absolute_error(y_val, pred)

    init_weights = np.ones(4) / 4.0
    bounds = [(0, 1) for _ in range(4)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    opt_res = minimize(
        ensemble_mae_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )

    weights = opt_res.x / np.sum(opt_res.x)
    print(f"Optimized Ensemble Weights: {weights}")

    val_preds_ensemble = val_preds_matrix @ weights
    test_preds_ensemble = test_preds_matrix @ weights

    final_score = mean_absolute_error(y_val, val_preds_ensemble)

    # Save Submission File
    os.makedirs("./submission", exist_ok=True)
    if "id" in test_df.columns:
        sub_ids = test_df["id"]
    else:
        sub_ids = np.arange(len(test_df))

    sub_df = pd.DataFrame({"id": sub_ids, "target": test_preds_ensemble})
    sub_df = sub_df.sort_values("id").reset_index(drop=True)
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()