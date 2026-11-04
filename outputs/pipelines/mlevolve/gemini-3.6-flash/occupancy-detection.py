import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# ==========================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# ==========================================


def load_raw_data(train_path: str, test_path: str):
    """Load raw train and test parquet datasets."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    # Ensure all sensor and target columns are converted to numeric data types
    sensor_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    for col in sensor_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(np.float64)
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float64)

    if "Occupancy" in train_df.columns:
        train_df["Occupancy"] = pd.to_numeric(train_df["Occupancy"], errors="coerce").astype(np.float32)

    # Convert date to datetime and sort sequentially for accurate temporal feature extraction
    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    return train_df, test_df


def engineer_features(
    df: pd.DataFrame, is_train: bool = True, hourly_stats: pd.DataFrame = None
):
    """
    Perform leak-free domain feature engineering including thermodynamic proxies,
    temporal derivatives, exponential moving averages (EMA), backward rolling statistics,
    and cyclic temporal features.
    """
    df = df.copy()
    new_cols = {}

    # 1. Calendar & Cyclic Time Features
    hour = df["date"].dt.hour
    minute = df["date"].dt.minute
    dayofweek = df["date"].dt.dayofweek

    new_cols["hour"] = hour
    new_cols["minute"] = minute
    new_cols["dayofweek"] = dayofweek
    new_cols["is_weekend"] = (dayofweek >= 5).astype(np.int32)

    min_of_day = hour * 60 + minute
    new_cols["sin_min_of_day"] = np.sin(2 * np.pi * min_of_day / 1440.0).astype(np.float32)
    new_cols["cos_min_of_day"] = np.cos(2 * np.pi * min_of_day / 1440.0).astype(np.float32)

    # 2. Thermodynamic & Physical Sensor Cross-Interactions
    new_cols["dew_point_approx"] = (
        df["Temperature"] - ((100.0 - df["Humidity"]) / 5.0)
    ).astype(np.float32)
    new_cols["temp_humidity_prod"] = (df["Temperature"] * df["Humidity"]).astype(np.float32)
    new_cols["temp_hum_ratio_prod"] = (df["Temperature"] * df["HumidityRatio"]).astype(
        np.float32
    )
    new_cols["co2_to_humidity"] = (df["CO2"] / (df["Humidity"] + 1e-5)).astype(np.float32)
    new_cols["light_co2_prod"] = (
        np.log1p(np.maximum(0, df["Light"])) * np.log1p(np.maximum(0, df["CO2"]))
    ).astype(np.float32)
    new_cols["light_is_zero"] = (df["Light"] <= 0).astype(np.int32)
    new_cols["light_is_active"] = (df["Light"] > 10.0).astype(np.int32)

    # 3. Dynamic Temporal Differences (Velocities & Rates of Change)
    sensor_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    for col in sensor_cols:
        new_cols[f"{col}_diff1"] = df[col].diff(1).fillna(0).astype(np.float32)
        new_cols[f"{col}_diff3"] = df[col].diff(3).fillna(0).astype(np.float32)
        new_cols[f"{col}_diff5"] = df[col].diff(5).fillna(0).astype(np.float32)
        new_cols[f"{col}_diff15"] = df[col].diff(15).fillna(0).astype(np.float32)

    # 4. Second-Order Dynamic Differences (Accelerations)
    for col in ["CO2", "Light"]:
        new_cols[f"{col}_accel1"] = pd.Series(new_cols[f"{col}_diff1"], index=df.index).diff(1).fillna(0).astype(np.float32)

    # 5. Exponential Moving Averages (EMA) & Instantaneous Residual Deviations
    ema_spans = [5, 15, 30]
    for col in ["CO2", "Temperature", "Light"]:
        for span in ema_spans:
            ema_series = df[col].ewm(span=span, adjust=False).mean().astype(np.float32)
            new_cols[f"{col}_ema{span}"] = ema_series
            new_cols[f"{col}_ema{span}_res"] = (df[col] - ema_series).astype(np.float32)

    # 6. Backward-Shifted Rolling Window Statistics (Strictly No Leakage)
    windows = [5, 15, 30]
    for col in ["CO2", "Light", "Humidity", "Temperature"]:
        shifted_series = df[col].shift(1)
        for w in windows:
            roll = shifted_series.rolling(window=w, min_periods=1)
            mean_val = roll.mean().fillna(df[col]).astype(np.float32)
            std_val = roll.std().fillna(0).astype(np.float32)
            max_val = roll.max().fillna(df[col]).astype(np.float32)
            min_val = roll.min().fillna(df[col]).astype(np.float32)

            new_cols[f"{col}_roll{w}_mean"] = mean_val
            new_cols[f"{col}_roll{w}_std"] = std_val
            new_cols[f"{col}_roll{w}_max"] = max_val
            new_cols[f"{col}_roll{w}_min"] = min_val
            new_cols[f"{col}_roll{w}_range"] = (max_val - min_val).astype(np.float32)

    # 7. Dynamic Deviation from Short-Term Rolling Mean
    new_cols["CO2_dev_roll15"] = (df["CO2"] - new_cols["CO2_roll15_mean"]).astype(np.float32)
    new_cols["Light_dev_roll15"] = (df["Light"] - new_cols["Light_roll15_mean"]).astype(np.float32)

    # Concatenate new features at once to prevent DataFrame fragmentation
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    # 8. Hourly Context Aggregation (Fit on train set only)
    if is_train:
        hourly_stats = df.groupby("hour")[
            ["CO2", "Light", "Temperature", "Humidity"]
        ].agg(["mean", "std"])
        hourly_stats.columns = [f"{c[0]}_hourly_{c[1]}" for c in hourly_stats.columns]
        hourly_stats = hourly_stats.reset_index()

    if hourly_stats is not None:
        df = df.merge(hourly_stats, on="hour", how="left")
    df = df.fillna(0)

    return df, hourly_stats


# ==========================================
# 2. MODEL ARCHITECTURES & DEFINITIONS
# ==========================================


class TabularResBlock(nn.Module):
    """Residual block for dynamic tabular feature representations."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    """
    Deep Tabular Neural Network with Residual Blocks, Batch Normalization,
    and Dropout for continuous non-linear occupancy estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out


def get_pytorch_components(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    return criterion, optimizer, scheduler


def get_lightgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(random_state: int = 42) -> cb.CatBoostClassifier:
    return cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=0,
        eval_metric="AUC",
    )


# ==========================================
# 3. TRAINING AND EVALUATION PIPELINE
# ==========================================


def train_and_evaluate():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    print("Loading raw datasets...")
    train_df, test_df = load_raw_data(train_path, test_path)

    print("Engineering features (base temporal & sensor features)...")
    train_df, hourly_stats = engineer_features(train_df, is_train=True)
    test_df, _ = engineer_features(test_df, is_train=False, hourly_stats=hourly_stats)

    # Re-sort to original ID order to guarantee target alignment
    train_df = train_df.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    target_col = "Occupancy"
    id_col = "id"
    drop_cols = [id_col, "date", target_col]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    test_ids = test_df[id_col].values

    print(
        f"Data preprocessed: Train={len(X)}, Test={len(X_test)}, Features={len(feature_cols)}"
    )

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(X), dtype=np.float32)
    oof_xgb = np.zeros(len(X), dtype=np.float32)
    oof_cat = np.zeros(len(X), dtype=np.float32)
    oof_nn = np.zeros(len(X), dtype=np.float32)

    test_preds_lgb = np.zeros(len(X_test), dtype=np.float32)
    test_preds_xgb = np.zeros(len(X_test), dtype=np.float32)
    test_preds_cat = np.zeros(len(X_test), dtype=np.float32)
    test_preds_nn = np.zeros(len(X_test), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # LightGBM
        lgb_model = get_lightgbm_model(random_state=42 + fold)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

        # XGBoost
        xgb_model = get_xgboost_model(random_state=42 + fold)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_splits

        # CatBoost
        cat_model = get_catboost_model(random_state=42 + fold)
        cat_model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0
        )
        oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits

        # PyTorch TabularResNet
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_test_scaled = scaler.transform(X_test)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=512, shuffle=False, num_workers=2
        )

        nn_model = TabularResNet(
            input_dim=X.shape[1], hidden_dim=128, num_blocks=2, dropout=0.2
        ).to(device)
        criterion, optimizer, scheduler = get_pytorch_components(
            nn_model, lr=1e-3, weight_decay=1e-4
        )

        best_val_auc = -1.0
        best_nn_preds = np.zeros(len(val_idx), dtype=np.float32)
        fold_test_nn = np.zeros(len(X_test), dtype=np.float32)

        for epoch in range(25):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    logits = nn_model(bx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.append(probs)
            val_preds_fold = np.vstack(val_preds_list).ravel()
            val_auc = roc_auc_score(y_va, val_preds_fold)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_nn_preds = val_preds_fold

                test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(
                    device
                )
                with torch.no_grad():
                    test_logits = nn_model(test_tensor)
                    fold_test_nn = torch.sigmoid(test_logits).cpu().numpy().ravel()

        oof_nn[val_idx] = best_nn_preds
        test_preds_nn += fold_test_nn / n_splits

        fold_blend = (
            0.35 * oof_lgb[val_idx]
            + 0.35 * oof_xgb[val_idx]
            + 0.20 * oof_cat[val_idx]
            + 0.10 * oof_nn[val_idx]
        )
        fold_auc = roc_auc_score(y_va, fold_blend)
        print(f"Fold {fold + 1}/{n_splits} Blend ROC-AUC: {fold_auc:.6f}")

    # Out-of-Fold Evaluation
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    auc_cat = roc_auc_score(y, oof_cat)
    auc_nn = roc_auc_score(y, oof_nn)

    print(
        f"OOF Scores -> LightGBM: {auc_lgb:.6f} | XGBoost: {auc_xgb:.6f} | CatBoost: {auc_cat:.6f} | PyTorch NN: {auc_nn:.6f}"
    )

    # SciPy SLSQP Constrained Optimization for Ensemble Weights
    def loss_func(weights):
        weights = np.array(weights)
        weights = np.maximum(0, weights)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        blend = (
            weights[0] * oof_lgb
            + weights[1] * oof_xgb
            + weights[2] * oof_cat
            + weights[3] * oof_nn
        )
        return -roc_auc_score(y, blend)

    init_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0.0, 1.0) for _ in range(4)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})

    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_w = np.maximum(0, res.x)
    opt_w = opt_w / opt_w.sum()

    print(f"Optimized Weights -> LGB: {opt_w[0]:.4f} | XGB: {opt_w[1]:.4f} | CAT: {opt_w[2]:.4f} | NN: {opt_w[3]:.4f}")

    final_oof_blend = (
        opt_w[0] * oof_lgb
        + opt_w[1] * oof_xgb
        + opt_w[2] * oof_cat
        + opt_w[3] * oof_nn
    )
    final_score = roc_auc_score(y, final_oof_blend)

    # Test Prediction & Submission Output
    final_test_blend = (
        opt_w[0] * test_preds_lgb
        + opt_w[1] * test_preds_xgb
        + opt_w[2] * test_preds_cat
        + opt_w[3] * test_preds_nn
    )

    submission_df = pd.DataFrame(
        {"id": test_ids.astype(int), "Occupancy": final_test_blend}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    train_and_evaluate()