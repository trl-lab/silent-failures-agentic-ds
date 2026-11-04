import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Step 1: Feature Engineering & Preprocessing
# ---------------------------------------------------------
def process_and_engineer_features():
    os.makedirs("./working", exist_ok=True)

    # 1. Load Datasets
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "NET_GEN"
    assert (
        target_col in train_df.columns
    ), f"Target column {target_col} missing in train data."

    y_train_raw = train_df[target_col].values.astype(np.float32)
    train_df = train_df.drop(columns=[target_col])

    id_col = None
    for col in ["id", "ID", "Id"]:
        if col in train_df.columns:
            id_col = col
            break

    train_ids = train_df[id_col].values if id_col else np.arange(len(train_df))
    test_ids = test_df[id_col].values if id_col else np.arange(len(test_df))

    if id_col:
        train_df = train_df.drop(columns=[id_col])
        test_df = test_df.drop(columns=[id_col])

    n_train = len(train_df)
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 2. Schema Discovery
    numeric_cols = []
    categorical_cols = []
    lat_col, lon_col = None, None
    capacity_cols = []

    for col in full_df.columns:
        col_lower = col.lower()
        if "lat" in col_lower:
            lat_col = col
        elif "lon" in col_lower or "lng" in col_lower:
            lon_col = col
        elif any(
            kw in col_lower for kw in ["cap", "mw", "power", "gen_cap", "nameplate"]
        ):
            capacity_cols.append(col)

        if pd.api.types.is_numeric_dtype(full_df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # 3. Categorical Processing & Missing Values
    train_feats = full_df.iloc[:n_train].copy()
    test_feats = full_df.iloc[n_train:].copy()

    for col in categorical_cols:
        train_feats[col] = train_feats[col].fillna("Missing").astype(str)
        test_feats[col] = test_feats[col].fillna("Missing").astype(str)

    for col in numeric_cols:
        median_val = train_feats[col].median()
        train_feats[col] = train_feats[col].fillna(median_val)
        test_feats[col] = test_feats[col].fillna(median_val)

    # Frequency Encoding
    for col in categorical_cols:
        freq_map = train_feats[col].value_counts(normalize=True).to_dict()
        train_feats[f"{col}_freq"] = train_feats[col].map(freq_map).fillna(0)
        test_feats[f"{col}_freq"] = test_feats[col].map(freq_map).fillna(0)

    # Ordinal Encoding
    if len(categorical_cols) > 0:
        ord_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        train_feats[categorical_cols] = ord_encoder.fit_transform(
            train_feats[categorical_cols]
        )
        test_feats[categorical_cols] = ord_encoder.transform(
            test_feats[categorical_cols]
        )

    # 4. Physical & Domain Feature Engineering
    for cap_col in capacity_cols:
        if cap_col in numeric_cols:
            train_feats[f"{cap_col}_log1p"] = np.log1p(
                np.maximum(0, train_feats[cap_col])
            )
            test_feats[f"{cap_col}_log1p"] = np.log1p(
                np.maximum(0, test_feats[cap_col])
            )

            train_feats[f"{cap_col}_max_annual_mwh"] = train_feats[cap_col] * 8760.0
            test_feats[f"{cap_col}_max_annual_mwh"] = test_feats[cap_col] * 8760.0

    # Check for age / year features
    for col in full_df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["year", "age", "built", "online", "commission", "oper"]):
            if pd.api.types.is_numeric_dtype(full_df[col]):
                current_year = 2024
                mean_val = full_df[col].dropna().mean() if len(full_df[col].dropna()) > 0 else 0
                if mean_val > 1800:
                    train_feats[f"{col}_age"] = current_year - train_feats[col]
                    test_feats[f"{col}_age"] = current_year - test_feats[col]

    for cap_col in capacity_cols:
        for group_cat in categorical_cols:
            group_stats = (
                train_feats.groupby(group_cat)[cap_col]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            group_stats.columns = [
                group_cat,
                f"{cap_col}_{group_cat}_grp_mean",
                f"{cap_col}_{group_cat}_grp_std",
                f"{cap_col}_{group_cat}_grp_min",
                f"{cap_col}_{group_cat}_grp_max",
            ]

            train_feats = train_feats.merge(group_stats, on=group_cat, how="left")
            test_feats = test_feats.merge(group_stats, on=group_cat, how="left")

            train_feats[f"{cap_col}_{group_cat}_grp_std"] = train_feats[
                f"{cap_col}_{group_cat}_grp_std"
            ].fillna(1.0)
            test_feats[f"{cap_col}_{group_cat}_grp_std"] = test_feats[
                f"{cap_col}_{group_cat}_grp_std"
            ].fillna(1.0)

            train_feats[f"{cap_col}_{group_cat}_ratio"] = train_feats[cap_col] / (
                train_feats[f"{cap_col}_{group_cat}_grp_mean"] + 1e-5
            )
            test_feats[f"{cap_col}_{group_cat}_ratio"] = test_feats[cap_col] / (
                test_feats[f"{cap_col}_{group_cat}_grp_mean"] + 1e-5
            )

    # 5. Spatial Feature Engineering
    if lat_col and lon_col:
        for df in [train_feats, test_feats]:
            for angle in [30, 45, 60]:
                rad = np.radians(angle)
                df[f"rot_{angle}_x"] = df[lat_col] * np.cos(rad) - df[lon_col] * np.sin(
                    rad
                )
                df[f"rot_{angle}_y"] = df[lat_col] * np.sin(rad) + df[lon_col] * np.cos(
                    rad
                )

            mean_lat, mean_lon = (
                train_feats[lat_col].mean(),
                train_feats[lon_col].mean(),
            )
            d_lat = df[lat_col] - mean_lat
            d_lon = df[lon_col] - mean_lon
            df["dist_to_center"] = np.sqrt(d_lat**2 + d_lon**2)
            df["angle_to_center"] = np.arctan2(d_lat, d_lon)

        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        train_coords = train_feats[[lat_col, lon_col]].values
        test_coords = test_feats[[lat_col, lon_col]].values

        train_feats["geo_cluster"] = kmeans.fit_predict(train_coords)
        test_feats["geo_cluster"] = kmeans.predict(test_coords)

    # 6. Target Transformation & CV Splits
    y_train_trans = np.sign(y_train_raw) * np.log1p(np.abs(y_train_raw))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kf.split(train_feats))

    # Out-of-fold Target Encoding & Domain Interactions
    y_series = pd.Series(y_train_trans, index=train_feats.index)
    for col in categorical_cols:
        te_train = np.zeros(n_train, dtype=np.float32)
        global_mean = y_train_trans.mean()

        for tr_idx, va_idx in cv_splits:
            tr_cat = train_feats.iloc[tr_idx][col]
            tr_y = y_series.iloc[tr_idx]
            cat_means = tr_y.groupby(tr_cat).mean().to_dict()
            te_train[va_idx] = (
                train_feats.iloc[va_idx][col].map(cat_means).fillna(global_mean).values
            )

        full_cat_means = y_series.groupby(train_feats[col]).mean().to_dict()
        te_test = test_feats[col].map(full_cat_means).fillna(global_mean).values

        train_feats[f"{col}_te"] = te_train
        test_feats[f"{col}_te"] = te_test

        for cap_col in capacity_cols:
            if cap_col in train_feats.columns:
                train_feats[f"{col}_{cap_col}_te_inter"] = (
                    train_feats[f"{col}_te"] * train_feats[cap_col]
                )
                test_feats[f"{col}_{cap_col}_te_inter"] = (
                    test_feats[f"{col}_te"] * test_feats[cap_col]
                )

    train_feats = train_feats.fillna(0.0)
    test_feats = test_feats.fillna(0.0)

    return {
        "X_train": train_feats.values.astype(np.float32),
        "y_train": y_train_trans.astype(np.float32),
        "y_train_raw": y_train_raw,
        "X_test": test_feats.values.astype(np.float32),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "cv_splits": cv_splits,
    }


# ---------------------------------------------------------
# Step 2: Model Definition (TabAttentionNet)
# ---------------------------------------------------------
class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        out = self.norm(residual + g * h)
        return out


class TabAttentionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.grn_blocks = nn.ModuleList(
            [GatedResidualBlock(embed_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.self_attention(h_seq, h_seq, h_seq)
        h = self.attn_norm(h + attn_out.squeeze(1))
        for block in self.grn_blocks:
            h = block(h)
        out = self.head(h).squeeze(-1)
        return out


# ---------------------------------------------------------
# Step 3: Training, Evaluation & Submission Assembly
# ---------------------------------------------------------
def invert_target_transformation(transformed_preds):
    return np.sign(transformed_preds) * np.expm1(np.abs(transformed_preds))


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    data = process_and_engineer_features()

    X_train = data["X_train"]
    y_train = data["y_train"]
    y_train_raw = data["y_train_raw"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    cv_splits = data["cv_splits"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 35
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-3
    patience = 8

    oof_nn_raw = np.zeros(len(X_train), dtype=np.float32)
    oof_lgb_raw = np.zeros(len(X_train), dtype=np.float32)
    oof_xgb_raw = np.zeros(len(X_train), dtype=np.float32)
    oof_cb_raw = np.zeros(len(X_train), dtype=np.float32)

    test_nn_folds = np.zeros((len(cv_splits), len(X_test)), dtype=np.float32)
    test_lgb_folds = np.zeros((len(cv_splits), len(X_test)), dtype=np.float32)
    test_xgb_folds = np.zeros((len(cv_splits), len(X_test)), dtype=np.float32)
    test_cb_folds = np.zeros((len(cv_splits), len(X_test)), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[train_idx])
        X_va = scaler.transform(X_train[val_idx])
        X_te = scaler.transform(X_test)
        X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_tr, y_va = y_train[train_idx], y_train[val_idx]
        y_tr_raw, y_va_raw = y_train_raw[train_idx], y_train_raw[val_idx]

        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        model = TabAttentionNet(
            input_dim=X_train.shape[1], embed_dim=128, num_heads=4
        ).to(device)
        criterion = nn.SmoothL1Loss(beta=1.0)
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_rmse = float("inf")
        best_model_path = f"./working/best_model_fold_{fold}.pt"
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(bx)

            scheduler.step()
            train_loss /= len(train_ds)

            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    out = model(bx)
                    val_preds_list.append(out.cpu().numpy())

            val_preds_trans = np.concatenate(val_preds_list)
            val_preds_raw = invert_target_transformation(val_preds_trans)
            val_rmse = calculate_rmse(y_va_raw, val_preds_raw)

            print(
                f"Fold {fold+1} | Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.2f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        with torch.no_grad():
            val_preds_list = []
            for bx, _ in val_loader:
                bx = bx.to(device)
                val_preds_list.append(model(bx).cpu().numpy())
            fold_oof_trans = np.concatenate(val_preds_list)
            oof_nn_raw[val_idx] = invert_target_transformation(fold_oof_trans)

            test_preds_trans = model(X_te_tensor).cpu().numpy()
            test_nn_folds[fold] = invert_target_transformation(test_preds_trans)

        # Train LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model_lgb.fit(
            X_tr,
            y_tr_raw,
            eval_set=[(X_va, y_va_raw)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_lgb_raw[val_idx] = model_lgb.predict(X_va)
        test_lgb_folds[fold] = model_lgb.predict(X_te)

        # Train XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
        )
        model_xgb.fit(
            X_tr,
            y_tr_raw,
            eval_set=[(X_va, y_va_raw)],
            verbose=False,
        )
        oof_xgb_raw[val_idx] = model_xgb.predict(X_va)
        test_xgb_folds[fold] = model_xgb.predict(X_te)

        # Train CatBoost
        model_cb = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
        )
        model_cb.fit(
            X_tr,
            y_tr_raw,
            eval_set=(X_va, y_va_raw),
            use_best_model=True,
        )
        oof_cb_raw[val_idx] = model_cb.predict(X_va)
        test_cb_folds[fold] = model_cb.predict(X_te)

    rmse_nn = calculate_rmse(y_train_raw, oof_nn_raw)
    rmse_lgb = calculate_rmse(y_train_raw, oof_lgb_raw)
    rmse_xgb = calculate_rmse(y_train_raw, oof_xgb_raw)
    rmse_cb = calculate_rmse(y_train_raw, oof_cb_raw)

    print(f"OOF RMSE - Neural Net: {rmse_nn:.4f}")
    print(f"OOF RMSE - LightGBM:   {rmse_lgb:.4f}")
    print(f"OOF RMSE - XGBoost:    {rmse_xgb:.4f}")
    print(f"OOF RMSE - CatBoost:   {rmse_cb:.4f}")

    def blend_objective(weights):
        w = np.maximum(0, weights)
        s = np.sum(w)
        if s == 0:
            return 1e9
        w = w / s
        blend_oof = (
            w[0] * oof_nn_raw
            + w[1] * oof_lgb_raw
            + w[2] * oof_xgb_raw
            + w[3] * oof_cb_raw
        )
        return calculate_rmse(y_train_raw, blend_oof)

    np.random.seed(42)
    initial_points = [
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.05, 0.45, 0.25, 0.25]),
        np.array([0.0, 0.5, 0.25, 0.25]),
        np.array([0.0, 0.4, 0.3, 0.3]),
    ] + [np.random.dirichlet(np.ones(4)) for _ in range(15)]

    best_score = float("inf")
    best_weights = None

    for init_w in initial_points:
        res = minimize(
            blend_objective,
            init_w,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5},
        )
        if res.fun < best_score:
            best_score = res.fun
            w_norm = np.maximum(0, res.x)
            best_weights = w_norm / np.sum(w_norm)

    print(f"Optimized Blend Weights (NN, LGB, XGB, CB): {best_weights}")

    oof_blend_raw = (
        best_weights[0] * oof_nn_raw
        + best_weights[1] * oof_lgb_raw
        + best_weights[2] * oof_xgb_raw
        + best_weights[3] * oof_cb_raw
    )
    overall_oof_rmse = calculate_rmse(y_train_raw, oof_blend_raw)

    test_nn_raw = np.mean(test_nn_folds, axis=0)
    test_lgb_raw = np.mean(test_lgb_folds, axis=0)
    test_xgb_raw = np.mean(test_xgb_folds, axis=0)
    test_cb_raw = np.mean(test_cb_folds, axis=0)

    final_test_preds = (
        best_weights[0] * test_nn_raw
        + best_weights[1] * test_lgb_raw
        + best_weights[2] * test_xgb_raw
        + best_weights[3] * test_cb_raw
    )
    final_test_preds = np.maximum(0.0, final_test_preds)

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame({"id": test_ids, "NET_GEN": final_test_preds})
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {overall_oof_rmse:.4f}")


if __name__ == "__main__":
    main()