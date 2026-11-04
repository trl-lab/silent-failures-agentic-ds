import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


def sanitize_column_names(df):
    """Standardize column names to remove spaces, standardize target, and ensure numeric types."""
    rename_dict = {
        "Whole weight": "Whole_weight",
        "Shucked weight": "Shucked_weight",
        "Viscera weight": "Viscera_weight",
        "Shell weight": "Shell_weight",
        "Class_number_of_rings": "Class_number_of_rings",
        "Class_Rings": "Class_number_of_rings",
        "Rings": "Class_number_of_rings",
    }
    df = df.rename(columns=rename_dict)
    df.columns = [col.replace(" ", "_") for col in df.columns]
    for col in df.columns:
        if col not in ["Sex", "id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_anomalies(df):
    """Fix physical anomalies such as 0 height in abalone measurements."""
    df = df.copy()
    zero_height_mask = df["Height"] <= 0
    if zero_height_mask.sum() > 0:
        valid_mask = df["Height"] > 0
        median_ratio = (
            df.loc[valid_mask, "Height"]
            / (df.loc[valid_mask, "Length"] * df.loc[valid_mask, "Diameter"]) ** 0.5
        ).median()
        df.loc[zero_height_mask, "Height"] = (
            median_ratio
            * (
                df.loc[zero_height_mask, "Length"]
                * df.loc[zero_height_mask, "Diameter"]
            )
            ** 0.5
        )
    return df


def engineer_features(df):
    """Create biological and physical interaction features tailored to abalone growth."""
    df = df.copy()
    eps = 1e-6

    # 1. Geometric & Volume Proxies
    df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["Area"] = df["Length"] * df["Diameter"]
    df["Length_to_Diameter"] = df["Length"] / (df["Diameter"] + eps)
    df["Length_to_Height"] = df["Length"] / (df["Height"] + eps)
    df["Diameter_to_Height"] = df["Diameter"] / (df["Height"] + eps)

    df["Ellipsoid_Volume"] = (
        (4.0 / 3.0)
        * np.pi
        * (df["Length"] / 2.0)
        * (df["Diameter"] / 2.0)
        * (df["Height"] / 2.0)
    )

    # 2. Mass Conservation & Ratios
    df["Non_Shell_Weight"] = df["Shucked_weight"] + df["Viscera_weight"]
    df["Water_Weight"] = df["Whole_weight"] - (
        df["Shucked_weight"] + df["Viscera_weight"] + df["Shell_weight"]
    )

    df["Shell_Weight_Ratio"] = df["Shell_weight"] / (df["Whole_weight"] + eps)
    df["Shucked_Weight_Ratio"] = df["Shucked_weight"] / (df["Whole_weight"] + eps)
    df["Viscera_Weight_Ratio"] = df["Viscera_weight"] / (df["Whole_weight"] + eps)
    df["Water_Weight_Ratio"] = df["Water_Weight"] / (df["Whole_weight"] + eps)
    df["Meat_to_Shell_Ratio"] = df["Shucked_weight"] / (df["Shell_weight"] + eps)

    # 3. Volumetric Densities
    df["Whole_Density"] = df["Whole_weight"] / (df["Volume"] + eps)
    df["Shell_Density"] = df["Shell_weight"] / (df["Volume"] + eps)
    df["Shucked_Density"] = df["Shucked_weight"] / (df["Volume"] + eps)

    # 4. Isometric Scaling Laws (Mass vs Linear Dimensions)
    df["Whole_Weight_CubeRoot"] = np.cbrt(np.maximum(0, df["Whole_weight"]))
    df["Shell_Weight_CubeRoot"] = np.cbrt(np.maximum(0, df["Shell_weight"]))
    df["Shucked_Weight_CubeRoot"] = np.cbrt(np.maximum(0, df["Shucked_weight"]))

    df["Length_vs_WeightCbrt"] = df["Length"] / (df["Whole_Weight_CubeRoot"] + eps)

    # 5. Non-linear Log Transforms
    df["Log_Volume"] = np.log1p(np.maximum(0, df["Volume"]))
    df["Log_Whole_Weight"] = np.log1p(np.maximum(0, df["Whole_weight"]))
    df["Log_Shell_Weight"] = np.log1p(np.maximum(0, df["Shell_weight"]))
    df["Log_Shucked_Weight"] = np.log1p(np.maximum(0, df["Shucked_weight"]))

    # 6. Sex Categorical Encoding & Interactions
    if "Sex" in df.columns:
        df["Sex_I"] = (df["Sex"] == "I").astype(np.float32)
        df["Sex_M"] = (df["Sex"] == "M").astype(np.float32)
        df["Sex_F"] = (df["Sex"] == "F").astype(np.float32)

        df["Is_Infant_Volume"] = df["Sex_I"] * df["Volume"]
        df["Is_Infant_Shell_Ratio"] = df["Sex_I"] * df["Shell_Weight_Ratio"]
        df["Is_Infant_Length"] = df["Sex_I"] * df["Length"]

    return df


class AbaloneTabularResNet(nn.Module):
    """Residual Neural Network with Layer Normalization, GELU, and Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super(AbaloneTabularResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.block1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_norm1 = nn.LayerNorm(hidden_dim)
        self.block1_act = nn.GELU()
        self.block1_drop = nn.Dropout(dropout)
        self.block1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_norm2 = nn.LayerNorm(hidden_dim)

        self.block2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.block2_norm1 = nn.LayerNorm(hidden_dim)
        self.block2_act = nn.GELU()
        self.block2_drop = nn.Dropout(dropout)
        self.block2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.block2_norm2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)

        out1 = self.block1_drop(self.block1_act(self.block1_norm1(self.block1_fc1(x))))
        out1 = self.block1_norm2(self.block1_fc2(out1))
        x = x + out1

        out2 = self.block2_drop(self.block2_act(self.block2_norm1(self.block2_fc1(x))))
        out2 = self.block2_norm2(self.block2_fc2(out2))
        x = x + out2

        output = self.head(x)
        return output.squeeze(-1)


def get_lgbm_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(random_state: int = 42) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=50,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def train_neural_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    device: str = "cpu",
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = AbaloneTabularResNet(
        input_dim=X_train.shape[1], hidden_dim=128, dropout=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_count += len(by)

        val_loss /= val_count
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        val_preds = model(val_tensor).cpu().numpy()

        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds = model(test_tensor).cpu().numpy()

    return val_preds, test_preds


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train_df = sanitize_column_names(train_raw)
    test_df = sanitize_column_names(test_raw)

    if "id" in test_df.columns:
        test_ids = test_df["id"]
    else:
        test_ids = pd.Series(test_df.index, name="id")

    # 2. Handle Anomalies & Engineer Features
    train_df = handle_anomalies(train_df)
    test_df = handle_anomalies(test_df)

    train_processed = engineer_features(train_df)
    test_processed = engineer_features(test_df)

    target_col = "Class_number_of_rings"
    if target_col not in train_processed.columns:
        for candidate in ["Class_Rings", "Rings", "target"]:
            if candidate in train_processed.columns:
                target_col = candidate
                break

    # 3. Setup Stratified Cross-Validation
    target_bins = pd.qcut(
        train_processed[target_col], q=10, labels=False, duplicates="drop"
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_processed["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_processed, target_bins)
    ):
        train_processed.loc[val_idx, "fold"] = fold

    if "Sex" in train_processed.columns:
        train_processed = train_processed.drop(columns=["Sex"])
    if "Sex" in test_processed.columns:
        test_processed = test_processed.drop(columns=["Sex"])

    drop_cols = [target_col, "fold", "id"]
    feature_cols = [c for c in train_processed.columns if c not in drop_cols]

    X = train_processed[feature_cols].values.astype(np.float32)
    y = train_processed[target_col].values.astype(np.float32)
    folds = train_processed["fold"].values

    X_test = test_processed[feature_cols].values.astype(np.float32)

    n_samples = len(train_processed)
    n_folds = int(folds.max() + 1)

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cb = np.zeros(n_samples)
    oof_nn = np.zeros(n_samples)

    test_preds_lgb = np.zeros((len(test_processed), n_folds))
    test_preds_xgb = np.zeros((len(test_processed), n_folds))
    test_preds_cb = np.zeros((len(test_processed), n_folds))
    test_preds_nn = np.zeros((len(test_processed), n_folds))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Train Models across CV Folds
    for fold in range(n_folds):
        val_mask = folds == fold
        train_mask = ~val_mask

        X_train_f, y_train_f = X[train_mask], y[train_mask]
        X_val_f, y_val_f = X[val_mask], y[val_mask]

        # LightGBM
        lgb_model = get_lgbm_model(random_state=42 + fold)
        lgb_model.fit(
            X_train_f,
            y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        val_lgb = lgb_model.predict(X_val_f)
        oof_lgb[val_mask] = val_lgb
        test_preds_lgb[:, fold] = lgb_model.predict(X_test)

        # XGBoost
        xgb_model = get_xgboost_model(random_state=42 + fold)
        xgb_model.fit(
            X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False
        )
        val_xgb = xgb_model.predict(X_val_f)
        oof_xgb[val_mask] = val_xgb
        test_preds_xgb[:, fold] = xgb_model.predict(X_test)

        # CatBoost
        cb_model = get_catboost_model(random_state=42 + fold)
        cb_model.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)
        val_cb = cb_model.predict(X_val_f)
        oof_cb[val_mask] = val_cb
        test_preds_cb[:, fold] = cb_model.predict(X_test)

        # PyTorch Neural Network
        val_nn, test_nn = train_neural_net(
            X_train_f,
            y_train_f,
            X_val_f,
            y_val_f,
            X_test,
            epochs=80,
            batch_size=64,
            device=device,
        )
        oof_nn[val_mask] = val_nn
        test_preds_nn[:, fold] = test_nn

    mean_test_lgb = test_preds_lgb.mean(axis=1)
    mean_test_xgb = test_preds_xgb.mean(axis=1)
    mean_test_cb = test_preds_cb.mean(axis=1)
    mean_test_nn = test_preds_nn.mean(axis=1)

    # 5. Optimize Ensemble Weights via SLSQP
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
    test_matrix = np.column_stack(
        [mean_test_lgb, mean_test_xgb, mean_test_cb, mean_test_nn]
    )

    def loss_func(weights):
        weights = np.array(weights)
        pred = np.dot(oof_matrix, weights)
        return np.sqrt(mean_squared_error(y, pred))

    init_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = res.x

    final_oof_preds = np.dot(oof_matrix, best_weights)
    final_oof_preds = np.clip(final_oof_preds, a_min=1.0, a_max=None)
    final_rmse = np.sqrt(mean_squared_error(y, final_oof_preds))

    # 6. Generate Submission File
    final_test_preds = np.dot(test_matrix, best_weights)
    final_test_preds = np.clip(final_test_preds, a_min=1.0, a_max=None)

    sub = pd.DataFrame({"id": test_ids, "Class_number_of_rings": final_test_preds})
    sub.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
