import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Setup Directories and Random Seeds
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Raw Input Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "heating_load"
y_train = train_df[target_col].values.astype(np.float32)

feature_cols = [
    col for col in train_df.columns if col not in [target_col, "cooling_load", "id"]
]

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()


# 3. Physics-Based Feature Engineering
def engineer_features(data_df):
    df = data_df.copy()

    rc = df["relative_compactness"]
    sa = df["surface_area"]
    wa = df["wall_area"]
    ra = df["roof_area"]
    h = df["overall_height"]
    orient = df["orientation"]
    ga = df["glazing_area"]
    gad = df["glazing_area_distribution"]

    # Exact Building Thermodynamic Features
    floor_area = sa - wa - ra
    window_area = wa * ga
    net_wall_area = wa * (1.0 - ga)
    exact_volume = floor_area * h

    df["floor_area"] = floor_area
    df["window_area"] = window_area
    df["net_wall_area"] = net_wall_area
    df["exact_volume"] = exact_volume
    df["window_to_floor_ratio"] = window_area / (floor_area + 1e-6)

    # Window-to-Wall Ratio & Envelope Exposure Indices
    df["wwr"] = window_area / (wa + 1e-6)
    df["envelope_exposure"] = sa / (exact_volume + 1e-6)
    df["wall_exposure"] = wa / (exact_volume + 1e-6)
    df["roof_exposure"] = ra / (exact_volume + 1e-6)
    df["window_exposure"] = window_area / (exact_volume + 1e-6)

    # Geometric & Physical Volume Calculations
    vol = ra * h
    df["calc_volume"] = vol
    df["surface_to_volume"] = sa / (vol + 1e-6)
    df["volume_to_surface"] = vol / (sa + 1e-6)

    # Envelope Surface Area Proportions
    df["wall_to_surface_ratio"] = wa / (sa + 1e-6)
    df["roof_to_surface_ratio"] = ra / (sa + 1e-6)
    df["wall_to_roof_ratio"] = wa / (ra + 1e-6)

    # Window (Glazing) Heat Loss & Gain Proxies
    df["effective_glazing_area"] = ga * wa
    df["glazing_to_surface_ratio"] = (ga * wa) / (sa + 1e-6)

    # Architectural Shape & Heat Capacity Indicators
    df["height_to_base"] = h / (np.sqrt(ra) + 1e-6)
    df["shape_factor"] = rc * sa
    df["thermal_capacity_proxy"] = vol * h

    # Non-linear & Interaction Terms
    df["height_x_wall"] = h * wa
    df["compactness_x_height"] = rc * h
    df["compactness_x_roof"] = rc * ra
    df["log_volume"] = np.log1p(vol)
    df["sqrt_surface"] = np.sqrt(sa)

    # Cyclic Orientation Encoding & Directional Exposure Vectors
    orient_rad = (orient - 2) * (2 * np.pi / 4)
    df["orient_sin"] = np.sin(orient_rad)
    df["orient_cos"] = np.cos(orient_rad)
    df["solar_exposure_sin"] = df["orient_sin"] * window_area
    df["solar_exposure_cos"] = df["orient_cos"] * window_area

    # Categorical One-Hot Encodings
    for c in ["orientation", "glazing_area_distribution"]:
        dummies = pd.get_dummies(df[c], prefix=c, dtype=float)
        df = pd.concat([df, dummies], axis=1)

    return df


n_train = len(X_train_raw)
combined_raw = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
combined_processed = engineer_features(combined_raw)

X_train_df = combined_processed.iloc[:n_train].copy()
X_test_df = combined_processed.iloc[n_train:].copy()
X_train_df, X_test_df = X_train_df.align(X_test_df, join="left", axis=1, fill_value=0)


# 4. PyTorch Residual Tabular Neural Network Architecture
class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.act(self.block(x) + residual)
        return self.head(x).squeeze(-1)


# 5. Multi-Seed Repeated 10-Fold Cross Validation Setup & Containers
SEEDS = [42, 100, 2024]
n_test = len(X_test_df)
num_models = 7  # LGBM, XGB, CatBoost, ExtraTrees, PyTorch ResNet, SVR, KernelRidge

oof_all_seeds = np.zeros((len(SEEDS), n_train, num_models), dtype=np.float32)
test_preds_all = np.zeros((len(SEEDS), 10, n_test, num_models), dtype=np.float32)

# 6. Model Training Across Multi-Seed Repeated 10 Folds (3 seeds, 30 folds)
for s_idx, seed in enumerate(SEEDS):
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
        X_tr_df, y_tr = X_train_df.iloc[train_idx].copy(), y_train[train_idx]
        X_va_df, y_va = X_train_df.iloc[val_idx].copy(), y_train[val_idx]
        X_te_df = X_test_df.copy()

        # Fold-isolated target encoding for Ecotect DOE shape parameters
        tr_group_df = pd.DataFrame({
            "rc": X_tr_df["relative_compactness"],
            "h": X_tr_df["overall_height"],
            "target": y_tr,
        })
        group_stats = (
            tr_group_df.groupby(["rc", "h"])["target"]
            .agg(["mean", "std"])
            .reset_index()
        )
        global_mean = float(y_tr.mean())
        global_std = float(y_tr.std())
        stats_dict = {}
        for _, row in group_stats.iterrows():
            m = row["mean"]
            s = global_std if pd.isna(row["std"]) else row["std"]
            stats_dict[(row["rc"], row["h"])] = (m, s)

        def apply_shape_stats(df):
            keys = list(zip(df["relative_compactness"], df["overall_height"]))
            means = [stats_dict[k][0] if k in stats_dict else global_mean for k in keys]
            stds = [stats_dict[k][1] if k in stats_dict else global_std for k in keys]
            df["shape_group_mean"] = np.array(means, dtype=np.float32)
            df["shape_group_std"] = np.array(stds, dtype=np.float32)
            return df

        X_tr_df = apply_shape_stats(X_tr_df)
        X_va_df = apply_shape_stats(X_va_df)
        X_te_df = apply_shape_stats(X_te_df)

        # Fit scaler strictly on training split
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_df).astype(np.float32)
        X_va_sc = scaler.transform(X_va_df).astype(np.float32)
        X_te_sc = scaler.transform(X_te_df).astype(np.float32)

        # --- LightGBM Regressor ---
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=1500,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=seed + fold,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=[(X_va_df, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        oof_all_seeds[s_idx, val_idx, 0] = lgb_model.predict(X_va_df)
        test_preds_all[s_idx, fold, :, 0] = lgb_model.predict(X_te_df)

        # --- XGBoost Regressor ---
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=seed + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)

        oof_all_seeds[s_idx, val_idx, 1] = xgb_model.predict(X_va_df)
        test_preds_all[s_idx, fold, :, 1] = xgb_model.predict(X_te_df)

        # --- CatBoost Regressor ---
        cat_model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.02,
            depth=6,
            random_seed=seed + fold,
            verbose=0,
        )
        cat_model.fit(
            X_tr_df,
            y_tr,
            eval_set=(X_va_df, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        oof_all_seeds[s_idx, val_idx, 2] = cat_model.predict(X_va_df)
        test_preds_all[s_idx, fold, :, 2] = cat_model.predict(X_te_df)

        # --- Extra Trees Regressor ---
        et_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=12,
            random_state=seed + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr_df, y_tr)

        oof_all_seeds[s_idx, val_idx, 3] = et_model.predict(X_va_df)
        test_preds_all[s_idx, fold, :, 3] = et_model.predict(X_te_df)

        # --- PyTorch Residual MLP ---
        torch.manual_seed(seed + fold)
        train_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_sc, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        nn_model = TabularResNet(
            input_dim=X_tr_sc.shape[1], hidden_dim=128, dropout=0.1
        ).to(device)
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=160, eta_min=1e-5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_nn_weights = None

        for epoch in range(160):
            nn_model.train()
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(b_x)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(device), b_y.to(device)
                    preds = nn_model(b_x)
                    val_loss += criterion(preds, b_y).item() * len(b_y)
            val_loss /= len(X_va_sc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            test_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            oof_all_seeds[s_idx, val_idx, 4] = nn_model(val_tensor).cpu().numpy()
            test_preds_all[s_idx, fold, :, 4] = nn_model(test_tensor).cpu().numpy()

        # --- Support Vector Regressor (SVR) ---
        svr_model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
        svr_model.fit(X_tr_sc, y_tr)

        oof_all_seeds[s_idx, val_idx, 5] = svr_model.predict(X_va_sc)
        test_preds_all[s_idx, fold, :, 5] = svr_model.predict(X_te_sc)

        # --- Kernel Ridge Regressor ---
        krr_model = KernelRidge(alpha=0.1, kernel="rbf")
        krr_model.fit(X_tr_sc, y_tr)

        oof_all_seeds[s_idx, val_idx, 6] = krr_model.predict(X_va_sc)
        test_preds_all[s_idx, fold, :, 6] = krr_model.predict(X_te_sc)

# 7. RidgeCV Meta-Stacking on Out-of-Fold Predictions
oof_avg = oof_all_seeds.mean(axis=0)  # Shape: (n_train, 5)
test_avg = test_preds_all.mean(axis=(0, 1))  # Shape: (n_test, 5)

meta_model = RidgeCV(alphas=np.logspace(-3, 3, 100))
meta_model.fit(oof_avg, y_train)

oof_final = meta_model.predict(oof_avg)
test_final = meta_model.predict(test_avg)

final_score = np.sqrt(np.mean((y_train - oof_final) ** 2))

# 8. Export Submission File
sub_df = pd.DataFrame({"id": np.arange(len(test_final)), "heating_load": test_final})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score:.4f}")