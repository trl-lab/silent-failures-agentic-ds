import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for non-linear feature selection and noise suppression."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualBlock(nn.Module):
    """Residual block with gated activation, LayerNorm, and Dropout."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.glu = GatedLinearUnit(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.glu(x)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out


class GatedResidualNet(nn.Module):
    """Deep Gated Residual Network tailored for financial tabular regression."""

    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def main():
    seed_everything(42)

    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Datasets
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train_df = (
        train_raw.reset_index() if "date" not in train_raw.columns else train_raw.copy()
    )
    test_df = (
        test_raw.reset_index() if "date" not in test_raw.columns else test_raw.copy()
    )

    train_df.columns = [str(col).strip() for col in train_df.columns]
    test_df.columns = [str(col).strip() for col in test_df.columns]

    # Deduplicate column names if necessary
    def rename_duplicates(df):
        cols = []
        count = {}
        for col in df.columns:
            if col in count:
                count[col] += 1
                cols.append(f"{col}_{count[col]}")
            else:
                count[col] = 0
                cols.append(col)
        df.columns = cols
        return df

    train_df = rename_duplicates(train_df)
    test_df = rename_duplicates(test_df)

    # Identify target column
    target_col = "ISE" if "ISE" in train_df.columns else train_df.columns[1]

    # Convert date to datetime and sort chronologically
    train_df["date"] = pd.to_datetime(train_df["date"], format="mixed")
    test_df["date"] = pd.to_datetime(test_df["date"], format="mixed")

    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )
    train_len = len(train_df)

    y_train_full = train_df[target_col].values

    # Combine for unified feature engineering
    train_features = train_df.drop(
        columns=[c for c in train_df.columns if target_col in c or c == "id"],
        errors="ignore",
    )
    test_features = test_df.drop(
        columns=[c for c in test_df.columns if target_col in c or c == "id"],
        errors="ignore",
    )

    df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

    index_cols = [c for c in df_all.columns if c not in ["date", "id"]]

    new_features = {}

    # Calendar Features
    new_features["dayofweek"] = df_all["date"].dt.dayofweek
    new_features["month"] = df_all["date"].dt.month
    new_features["quarter"] = df_all["date"].dt.quarter
    new_features["dayofyear"] = df_all["date"].dt.dayofyear
    new_features["is_month_start"] = df_all["date"].dt.is_month_start.astype(int)
    new_features["is_month_end"] = df_all["date"].dt.is_month_end.astype(int)

    new_features["sin_dayofweek"] = np.sin(2 * np.pi * new_features["dayofweek"] / 7.0)
    new_features["cos_dayofweek"] = np.cos(2 * np.pi * new_features["dayofweek"] / 7.0)
    new_features["sin_month"] = np.sin(2 * np.pi * new_features["month"] / 12.0)
    new_features["cos_month"] = np.cos(2 * np.pi * new_features["month"] / 12.0)

    # Cross-Market Aggregates
    global_mean = df_all[index_cols].mean(axis=1)
    global_std = df_all[index_cols].std(axis=1)
    global_max = df_all[index_cols].max(axis=1)
    global_min = df_all[index_cols].min(axis=1)

    new_features["global_market_mean"] = global_mean
    new_features["global_market_std"] = global_std
    new_features["global_market_max"] = global_max
    new_features["global_market_min"] = global_min
    new_features["global_market_spread"] = global_max - global_min

    eu_cols = [c for c in ["DAX", "FTSE", "EU"] if c in index_cols]
    am_cols = [c for c in ["SP", "BOVESPA"] if c in index_cols]

    if eu_cols:
        new_features["eu_mean"] = df_all[eu_cols].mean(axis=1)
    if am_cols:
        new_features["am_mean"] = df_all[am_cols].mean(axis=1)
    if "NIKKEI" in index_cols:
        new_features["asia_return"] = df_all["NIKKEI"]
    if "EM" in index_cols:
        new_features["em_return"] = df_all["EM"]

    if "eu_mean" in new_features and "am_mean" in new_features:
        new_features["eu_am_spread"] = new_features["eu_mean"] - new_features["am_mean"]
    if "em_return" in new_features and "eu_mean" in new_features:
        new_features["em_eu_spread"] = new_features["em_return"] - new_features["eu_mean"]

    # Cross-Asset Relative Ratios & Spreads
    for col in index_cols:
        new_features[f"{col}_rel_global"] = df_all[col] - global_mean

    # Global Macro PCA Risk Factors
    pca = PCA(n_components=min(3, len(index_cols)), random_state=42)
    pca.fit(df_all.iloc[:train_len][index_cols].fillna(0))
    pca_features = pca.transform(df_all[index_cols].fillna(0))
    for i in range(pca_features.shape[1]):
        new_features[f"macro_pca_{i+1}"] = pca_features[:, i]

    # Lags, Rolling Volatility Z-Scores, and EWMA MACD Spreads
    for col in index_cols:
        for lag in [1, 2, 3, 5]:
            new_features[f"{col}_lag_{lag}"] = df_all[col].shift(lag)

        for window in [3, 5, 10, 20]:
            r_mean = df_all[col].shift(1).rolling(window=window, min_periods=1).mean()
            r_std = df_all[col].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
            new_features[f"{col}_roll_mean_{window}"] = r_mean
            new_features[f"{col}_roll_std_{window}"] = r_std
            new_features[f"{col}_zscore_{window}"] = (df_all[col].shift(1) - r_mean) / (r_std + 1e-8)

        ewma_3 = df_all[col].shift(1).ewm(span=3, adjust=False).mean()
        ewma_5 = df_all[col].shift(1).ewm(span=5, adjust=False).mean()
        ewma_10 = df_all[col].shift(1).ewm(span=10, adjust=False).mean()
        ewma_20 = df_all[col].shift(1).ewm(span=20, adjust=False).mean()

        new_features[f"{col}_ewma_3"] = ewma_3
        new_features[f"{col}_ewma_5"] = ewma_5
        new_features[f"{col}_ewma_10"] = ewma_10
        new_features[f"{col}_ewma_20"] = ewma_20

        # MACD Spreads (short minus long EWMA)
        new_features[f"{col}_macd_3_10"] = ewma_3 - ewma_10
        new_features[f"{col}_macd_5_20"] = ewma_5 - ewma_20

    # Non-fragmented DataFrame Construction
    df_features = pd.DataFrame(new_features, index=df_all.index)
    df_all = pd.concat([df_all, df_features], axis=1)
    df_all = df_all.bfill().ffill().fillna(0)

    feature_cols = [c for c in df_all.columns if c not in ["date", "id"]]

    X_full = df_all.iloc[:train_len][feature_cols].copy()
    X_test_raw = df_all.iloc[train_len:][feature_cols].copy()

    val_size = int(train_len * 0.2)
    split_idx = train_len - val_size

    X_train_raw, X_val_raw = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

    # Scale strictly on Training Set
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_cols)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=feature_cols)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    # 2. Model 1: LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 16,
        "max_depth": 4,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42,
    }

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_val_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    # 3. Model 2: XGBoost
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.75,
        colsample_bytree=0.75,
        random_state=42,
        eval_metric="rmse",
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_preds = xgb_model.predict(X_val)
    xgb_test_preds = xgb_model.predict(X_test)

    # 4. Model 3: CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.02,
        depth=4,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    cb_val_preds = cb_model.predict(X_val)
    cb_test_preds = cb_model.predict(X_test)

    # 5. Model 4: Ridge Regression
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    ridge_val_preds = ridge_model.predict(X_val)
    ridge_test_preds = ridge_model.predict(X_test)

    # 6. Model 5: PyTorch Deep Gated Residual Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    nn_model = GatedResidualNet(
        input_dim=X_train.shape[1], hidden_dim=128, num_blocks=3, dropout=0.2
    ).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.01)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(120):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            v_preds = (
                nn_model(torch.tensor(X_val.values, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            v_loss = np.sqrt(mean_squared_error(y_val, v_preds))

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_nn_weights)
    nn_model.eval()
    with torch.no_grad():
        nn_val_preds = (
            nn_model(torch.tensor(X_val.values, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        nn_test_preds = (
            nn_model(torch.tensor(X_test.values, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    # 7. SLSQP Meta-Ensemble Optimization
    val_preds_matrix = np.column_stack(
        [
            lgb_val_preds,
            xgb_val_preds,
            cb_val_preds,
            ridge_val_preds,
            nn_val_preds,
        ]
    )
    test_preds_matrix = np.column_stack(
        [
            lgb_test_preds,
            xgb_test_preds,
            cb_test_preds,
            ridge_test_preds,
            nn_test_preds,
        ]
    )

    num_models = val_preds_matrix.shape[1]

    def ensemble_loss(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = val_preds_matrix @ w
        return np.sqrt(mean_squared_error(y_val, pred))

    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        ensemble_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = res.x / np.sum(res.x)

    final_val_preds = val_preds_matrix @ best_weights
    final_test_preds = test_preds_matrix @ best_weights
    final_val_score = np.sqrt(mean_squared_error(y_val, final_val_preds))

    # Save Submission File
    submission = pd.DataFrame({"id": test_ids, "ISE": final_test_preds})
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()