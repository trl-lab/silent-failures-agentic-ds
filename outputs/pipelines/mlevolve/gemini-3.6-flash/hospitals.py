import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==========================================
# 1. PyTorch Tabular ResNet Neural Network
# ==========================================
class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ==========================================
# 2. Data Processing & Feature Engineering
# ==========================================
def process_and_engineer_features():
    print("Starting data processing and feature engineering...")
    os.makedirs("./working", exist_ok=True)

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "BEDS"
    if target_col in test_df.columns:
        test_df = test_df.drop(columns=[target_col])

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    train_df[target_col] = train_df[target_col].clip(lower=0)

    train_cols = [c for c in train_df.columns if c not in ["id", target_col]]
    test_cols = [c for c in test_df.columns if c != "id"]
    feature_cols = [c for c in train_cols if c in test_cols]

    df_train_feat = train_df[feature_cols].copy()
    df_test_feat = test_df[feature_cols].copy()

    num_cols = df_train_feat.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train_feat.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for c in cat_cols:
        df_train_feat[c] = (
            df_train_feat[c].astype(str).str.upper().str.strip().fillna("MISSING")
        )
        df_test_feat[c] = (
            df_test_feat[c].astype(str).str.upper().str.strip().fillna("MISSING")
        )

    cat_list = cat_cols.copy()
    if len(cat_list) >= 2:
        for i in range(len(cat_list)):
            for j in range(i + 1, min(i + 3, len(cat_list))):
                c1, c2 = cat_list[i], cat_list[j]
                inter_col = f"{c1}_{c2}_inter"
                df_train_feat[inter_col] = (
                    df_train_feat[c1].astype(str) + "_" + df_train_feat[c2].astype(str)
                )
                df_test_feat[inter_col] = (
                    df_test_feat[c1].astype(str) + "_" + df_test_feat[c2].astype(str)
                )

    lat_col = next(
        (c for c in num_cols if "LAT" in c.upper() or c.upper() in ["Y", "LATITUDE"]),
        None,
    )
    lon_col = next(
        (c for c in num_cols if "LON" in c.upper() or c.upper() in ["X", "LONGITUDE"]),
        None,
    )

    if lat_col and lon_col:
        print(f"Creating geospatial features using {lat_col} and {lon_col}...")
        for df in [df_train_feat, df_test_feat]:
            rad45 = np.radians(45)
            rad30 = np.radians(30)
            rad60 = np.radians(60)

            df["rot_45_x"] = df[lon_col] * np.cos(rad45) - df[lat_col] * np.sin(rad45)
            df["rot_45_y"] = df[lon_col] * np.sin(rad45) + df[lat_col] * np.cos(rad45)
            df["rot_30_x"] = df[lon_col] * np.cos(rad30) - df[lat_col] * np.sin(rad30)
            df["rot_30_y"] = df[lon_col] * np.sin(rad30) + df[lat_col] * np.cos(rad30)
            df["rot_60_x"] = df[lon_col] * np.cos(rad60) - df[lat_col] * np.sin(rad60)
            df["rot_60_y"] = df[lon_col] * np.sin(rad60) + df[lat_col] * np.cos(rad60)

            lat_rad = np.radians(df[lat_col].fillna(df[lat_col].median()))
            lon_rad = np.radians(df[lon_col].fillna(df[lon_col].median()))
            df["sphere_x"] = np.cos(lat_rad) * np.cos(lon_rad)
            df["sphere_y"] = np.cos(lat_rad) * np.sin(lon_rad)
            df["sphere_z"] = np.sin(lat_rad)

        train_coords = df_train_feat[[lat_col, lon_col]].fillna(
            df_train_feat[[lat_col, lon_col]].median()
        )
        test_coords = df_test_feat[[lat_col, lon_col]].fillna(
            df_train_feat[[lat_col, lon_col]].median()
        )

        kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
        kmeans.fit(train_coords)
        df_train_feat["geo_cluster"] = kmeans.predict(train_coords)
        df_test_feat["geo_cluster"] = kmeans.predict(test_coords)

        knn = NearestNeighbors(n_neighbors=10, algorithm="kd_tree")
        knn.fit(train_coords)

        train_dists, _ = knn.kneighbors(train_coords)
        test_dists, _ = knn.kneighbors(test_coords)

        df_train_feat["knn_dist_3"] = train_dists[:, 2]
        df_train_feat["knn_dist_5"] = train_dists[:, 4]
        df_train_feat["knn_dist_10"] = train_dists[:, 9]
        df_train_feat["knn_mean_dist"] = train_dists.mean(axis=1)

        df_test_feat["knn_dist_3"] = test_dists[:, 2]
        df_test_feat["knn_dist_5"] = test_dists[:, 4]
        df_test_feat["knn_dist_10"] = test_dists[:, 9]
        df_test_feat["knn_mean_dist"] = test_dists.mean(axis=1)

    cat_cols = df_train_feat.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols_agg = df_train_feat.select_dtypes(include=[np.number]).columns.tolist()

    for col in cat_cols:
        freq = df_train_feat[col].value_counts(normalize=True).to_dict()
        df_train_feat[f"{col}_freq"] = df_train_feat[col].map(freq).fillna(0)
        df_test_feat[f"{col}_freq"] = df_test_feat[col].map(freq).fillna(0)

        if df_train_feat[col].nunique() <= 300:
            for num_col in num_cols_agg:
                for stat in ["mean", "median", "std", "min", "max", "count"]:
                    stat_series = df_train_feat.groupby(col)[num_col].agg(stat)
                    feat_name = f"{col}_{num_col}_{stat}"
                    df_train_feat[feat_name] = df_train_feat[col].map(stat_series)
                    df_test_feat[feat_name] = df_test_feat[col].map(stat_series)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for col in cat_cols:
        target_enc_train = pd.Series(0.0, index=df_train_feat.index)
        global_mean = train_df[target_col].mean()

        for tr_idx, val_idx in kf.split(df_train_feat):
            fold_tr_cat = df_train_feat.iloc[tr_idx][col]
            fold_tr_target = train_df.iloc[tr_idx][target_col]
            means = fold_tr_target.groupby(fold_tr_cat).mean()
            target_enc_train.iloc[val_idx] = (
                df_train_feat.iloc[val_idx][col].map(means).fillna(global_mean)
            )

        df_train_feat[f"{col}_te"] = target_enc_train
        overall_means = train_df[target_col].groupby(df_train_feat[col]).mean()
        df_test_feat[f"{col}_te"] = (
            df_test_feat[col].map(overall_means).fillna(global_mean)
        )

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train_feat[col], df_test_feat[col]]).astype(str)
        le.fit(combined)
        df_train_feat[col] = le.transform(df_train_feat[col].astype(str))
        df_test_feat[col] = le.transform(df_test_feat[col].astype(str))

    df_train_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_train_raw = df_train_feat.copy()
    df_test_raw = df_test_feat.copy()
    df_train_raw["id"] = train_df["id"].values
    df_train_raw[target_col] = train_df[target_col].values
    df_test_raw["id"] = test_df["id"].values

    all_feature_cols = df_train_feat.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    df_train_imp = pd.DataFrame(
        imputer.fit_transform(df_train_feat), columns=all_feature_cols
    )
    df_test_imp = pd.DataFrame(
        imputer.transform(df_test_feat), columns=all_feature_cols
    )

    df_train_imp["id"] = train_df["id"].values
    df_train_imp[target_col] = train_df[target_col].values
    df_test_imp["id"] = test_df["id"].values

    return df_train_raw, df_test_raw, df_train_imp, df_test_imp


# ==========================================
# 3. Model Training & Validation Pipeline
# ==========================================
def train_and_evaluate():
    train_raw, test_raw, train_imp, test_imp = process_and_engineer_features()

    target_col = "BEDS"
    id_col = "id"
    feature_cols = [c for c in train_raw.columns if c not in [target_col, id_col]]

    X_tree = train_raw[feature_cols].values
    X_test_tree = test_raw[feature_cols].values

    X_nn = train_imp[feature_cols].values
    X_test_nn = test_imp[feature_cols].values

    y_raw = train_raw[target_col].values
    y_log = np.log1p(np.clip(y_raw, 0, None))

    n_samples = len(train_raw)
    n_test = len(test_raw)
    n_splits = 10
    seeds = [42, 100]

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cb = np.zeros(n_samples)
    oof_nn = np.zeros(n_samples)

    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)
    test_cb = np.zeros(n_test)
    test_nn = np.zeros(n_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training ensemble models using {n_splits}-fold CV with multi-seed ensembling...")

    for seed_idx, seed in enumerate(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_lgb_seed = np.zeros(n_samples)
        oof_xgb_seed = np.zeros(n_samples)
        oof_cb_seed = np.zeros(n_samples)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_tree, y_log)):
            X_tr_tree, y_tr = X_tree[train_idx], y_log[train_idx]
            X_va_tree, y_va = X_tree[val_idx], y_log[val_idx]
            X_tr_nn = X_nn[train_idx]
            X_va_nn = X_nn[val_idx]

            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbosity=-1,
                random_state=seed + fold * 10,
                n_jobs=-1,
            )
            lgb_model.fit(
                X_tr_tree,
                y_tr,
                eval_set=[(X_va_tree, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof_lgb_seed[val_idx] = lgb_model.predict(X_va_tree)
            test_lgb += lgb_model.predict(X_test_tree) / (n_splits * len(seeds))

            # XGBoost
            xgb_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method="hist",
                verbosity=0,
                early_stopping_rounds=50,
                random_state=seed + fold * 10,
                n_jobs=-1,
            )
            xgb_model.fit(
                X_tr_tree,
                y_tr,
                eval_set=[(X_va_tree, y_va)],
                verbose=False,
            )
            oof_xgb_seed[val_idx] = xgb_model.predict(X_va_tree)
            test_xgb += xgb_model.predict(X_test_tree) / (n_splits * len(seeds))

            # CatBoost
            cb_model = cb.CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=1200,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=seed + fold * 10,
                early_stopping_rounds=50,
                verbose=0,
            )
            cb_model.fit(X_tr_tree, y_tr, eval_set=(X_va_tree, y_va), verbose=False)
            oof_cb_seed[val_idx] = cb_model.predict(X_va_tree)
            test_cb += cb_model.predict(X_test_tree) / (n_splits * len(seeds))

            # PyTorch TabularResNet (run on seed 0)
            if seed_idx == 0:
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr_nn)
                X_va_sc = scaler.transform(X_va_nn)
                X_te_sc = scaler.transform(X_test_nn)

                train_ds = TensorDataset(
                    torch.tensor(X_tr_sc, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
                )
                val_ds = TensorDataset(
                    torch.tensor(X_va_sc, dtype=torch.float32),
                    torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
                )

                train_loader = DataLoader(
                    train_ds, batch_size=128, shuffle=True, drop_last=False
                )
                val_loader = DataLoader(
                    val_ds, batch_size=256, shuffle=False, drop_last=False
                )

                nn_model = TabularResNet(input_dim=len(feature_cols)).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(
                    nn_model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3
                )

                best_val_loss = float("inf")
                best_state = None
                patience, patience_counter = 15, 0

                for epoch in range(100):
                    nn_model.train()
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        out = nn_model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()

                    nn_model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx, by = bx.to(device), by.to(device)
                            val_loss += criterion(nn_model(bx), by).item() * bx.size(0)
                    val_loss /= len(val_ds)
                    val_rmse = np.sqrt(val_loss)

                    scheduler.step(val_rmse)

                    if val_rmse < best_val_loss:
                        best_val_loss = val_rmse
                        best_state = {
                            k: v.cpu().clone()
                            for k, v in nn_model.state_dict().items()
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break

                nn_model.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()}
                )
                nn_model.eval()
                with torch.no_grad():
                    va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(
                        device
                    )
                    te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(
                        device
                    )
                    oof_nn[val_idx] = nn_model(va_tensor).cpu().numpy().flatten()
                    test_nn += nn_model(te_tensor).cpu().numpy().flatten() / n_splits

            fold_rmse = np.sqrt(mean_squared_error(y_va, oof_lgb_seed[val_idx]))
            print(
                f"Seed {seed} Fold {fold+1}/{n_splits} complete - LightGBM Fold RMSE: {fold_rmse:.4f}"
            )

        oof_lgb += oof_lgb_seed / len(seeds)
        oof_xgb += oof_xgb_seed / len(seeds)
        oof_cb += oof_cb_seed / len(seeds)

    oof_lgb_exp = np.expm1(np.clip(oof_lgb, 0, None))
    oof_xgb_exp = np.expm1(np.clip(oof_xgb, 0, None))
    oof_cb_exp = np.expm1(np.clip(oof_cb, 0, None))
    oof_nn_exp = np.expm1(np.clip(oof_nn, 0, None))

    test_lgb_exp = np.expm1(np.clip(test_lgb, 0, None))
    test_xgb_exp = np.expm1(np.clip(test_xgb, 0, None))
    test_cb_exp = np.expm1(np.clip(test_cb, 0, None))
    test_nn_exp = np.expm1(np.clip(test_nn, 0, None))

    # 5-fold Cross-Validated Ridge Stacking Meta-Learner
    X_meta = np.column_stack([oof_lgb_exp, oof_xgb_exp, oof_cb_exp, oof_nn_exp])
    X_meta_test = np.column_stack([test_lgb_exp, test_xgb_exp, test_cb_exp, test_nn_exp])

    meta_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros(n_samples)
    test_meta = np.zeros(n_test)

    for m_tr_idx, m_va_idx in meta_kf.split(X_meta, y_raw):
        meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        meta_model.fit(X_meta[m_tr_idx], y_raw[m_tr_idx])
        oof_meta[m_va_idx] = meta_model.predict(X_meta[m_va_idx])
        test_meta += meta_model.predict(X_meta_test) / 5.0

    oof_final = np.clip(oof_meta, 0, None)
    test_final = np.clip(test_meta, 0, None)
    final_score = np.sqrt(mean_squared_error(y_raw, oof_final))

    # Export Submission
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": test_raw[id_col].astype(int),
            "BEDS": test_final,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    train_and_evaluate()
