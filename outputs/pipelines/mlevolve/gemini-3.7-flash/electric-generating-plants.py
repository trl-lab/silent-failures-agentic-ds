import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 0. Global Seed & Directory Setup
# ---------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)

input_dir = "./input"
submission_dir = "./submission"
working_dir = "./working"
os.makedirs(submission_dir, exist_ok=True)
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_df_raw = pd.read_parquet(os.path.join(input_dir, "train.parquet")).copy()
test_df_raw = pd.read_parquet(os.path.join(input_dir, "test.parquet")).copy()

target_col = "NET_GEN"
target = train_df_raw[target_col].copy()
train_features = train_df_raw.drop(columns=[target_col]).copy()
test_features = test_df_raw.copy()


# ---------------------------------------------------------
# 2. Physics-Informed & Domain Feature Engineering
# ---------------------------------------------------------
def engineer_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # Decompose and transform datetime columns into numerical features
    datetime_cols = df_tr.select_dtypes(include=["datetime", "datetime64", "datetimetz"]).columns.tolist()
    for col in datetime_cols:
        for d in [df_tr, df_te]:
            if col in d.columns:
                d[f"{col}_YEAR"] = d[col].dt.year.astype(np.float32)
                d[f"{col}_MONTH"] = d[col].dt.month.astype(np.float32)
                d[f"{col}_DAY"] = d[col].dt.day.astype(np.float32)
                d[f"{col}_DOY"] = d[col].dt.dayofyear.astype(np.float32)
                d[f"{col}_TIMESTAMP"] = (pd.to_numeric(d[col]) / 1e9).astype(np.float32)
                d.drop(columns=[col], inplace=True)

    # Identify capacity columns
    cap_cols = [
        c
        for c in ["NAMEPLATE", "SUMMER_CAP", "WINTER_CAP", "TOTAL_CAP", "CAP_MW"]
        if c in df_tr.columns
    ]

    for c in cap_cols:
        df_tr[f"{c}_log1p"] = np.log1p(np.maximum(0, df_tr[c].fillna(0)))
        df_te[f"{c}_log1p"] = np.log1p(np.maximum(0, df_te[c].fillna(0)))

        # Theoretical 8760-hour maximum generation envelope (MWh)
        df_tr[f"{c}_THEO_MAX_GEN"] = df_tr[c].fillna(0) * 8760.0
        df_te[f"{c}_THEO_MAX_GEN"] = df_te[c].fillna(0) * 8760.0

    # Capacity differentials and seasonal variance
    if "SUMMER_CAP" in df_tr.columns and "WINTER_CAP" in df_tr.columns:
        df_tr["CAP_SEASON_DIFF"] = df_tr["WINTER_CAP"] - df_tr["SUMMER_CAP"]
        df_te["CAP_SEASON_DIFF"] = df_te["WINTER_CAP"] - df_te["SUMMER_CAP"]

        df_tr["CAP_SEASON_RATIO"] = df_tr["SUMMER_CAP"] / (df_tr["WINTER_CAP"] + 1e-4)
        df_te["CAP_SEASON_RATIO"] = df_te["SUMMER_CAP"] / (df_te["WINTER_CAP"] + 1e-4)

        df_tr["CAP_SEASON_MEAN"] = (df_tr["SUMMER_CAP"] + df_tr["WINTER_CAP"]) / 2.0
        df_te["CAP_SEASON_MEAN"] = (df_te["SUMMER_CAP"] + df_te["WINTER_CAP"]) / 2.0

    if "NAMEPLATE" in df_tr.columns and "SUMMER_CAP" in df_tr.columns:
        df_tr["SUMMER_TO_NAMEPLATE_RATIO"] = df_tr["SUMMER_CAP"] / (
            df_tr["NAMEPLATE"] + 1e-4
        )
        df_te["SUMMER_TO_NAMEPLATE_RATIO"] = df_te["SUMMER_CAP"] / (
            df_te["NAMEPLATE"] + 1e-4
        )

    # Geospatial Coordinates & Projections
    lat_col = [c for c in ["LATITUDE", "LAT", "Y"] if c in df_tr.columns]
    lon_col = [c for c in ["LONGITUDE", "LON", "X"] if c in df_tr.columns]

    if lat_col and lon_col:
        lat_c, lon_c = lat_col[0], lon_col[0]
        lat_median = df_tr[lat_c].median()
        lon_median = df_tr[lon_c].median()

        for d in [df_tr, df_te]:
            lat_rad = np.radians(d[lat_c].fillna(lat_median))
            lon_rad = np.radians(d[lon_c].fillna(lon_median))
            d["COORD_X"] = np.cos(lat_rad) * np.cos(lon_rad)
            d["COORD_Y"] = np.cos(lat_rad) * np.sin(lon_rad)
            d["COORD_Z"] = np.sin(lat_rad)
            d["LAT_LON_RATIO"] = d[lat_c].fillna(lat_median) / (
                np.abs(d[lon_c].fillna(lon_median)) + 1e-4
            )

        # Spatial K-Means Clustering (fitted strictly on training data)
        train_coords = (
            df_tr[[lat_c, lon_c]].fillna({lat_c: lat_median, lon_c: lon_median}).values
        )
        kmeans = KMeans(n_clusters=12, random_state=42, n_init=10).fit(train_coords)
        df_tr["SPATIAL_CLUSTER"] = kmeans.predict(train_coords)
        test_coords = (
            df_te[[lat_c, lon_c]].fillna({lat_c: lat_median, lon_c: lon_median}).values
        )
        df_te["SPATIAL_CLUSTER"] = kmeans.predict(test_coords)

    # Plant Age & Vintage
    year_cols = [
        c
        for c in ["OPER_YEAR", "PLAN_YEAR", "INITIAL_OP_YR", "YEAR"]
        if c in df_tr.columns
    ]
    for yc in year_cols:
        med_year = df_tr[yc].median()
        df_tr[f"{yc}_AGE"] = 2024.0 - df_tr[yc].fillna(med_year)
        df_te[f"{yc}_AGE"] = 2024.0 - df_te[yc].fillna(med_year)

    # High-Order Categorical Interactions
    source_cols = [
        c
        for c in ["PRIM_SOURC", "SOURCE", "FUEL_TYPE", "ENERGY_SOURCE"]
        if c in df_tr.columns
    ]
    mover_cols = [
        c for c in ["PRIM_MVR", "PRIME_MOVER", "TECH_DESC"] if c in df_tr.columns
    ]

    if source_cols and mover_cols:
        sc, mc = source_cols[0], mover_cols[0]
        df_tr["SOURCE_MOVER_INTERACTION"] = (
            df_tr[sc].astype(str) + "_" + df_tr[mc].astype(str)
        )
        df_te["SOURCE_MOVER_INTERACTION"] = (
            df_te[sc].astype(str) + "_" + df_te[mc].astype(str)
        )

    # Categorical Frequency and Ordinal Encoding
    cat_cols = df_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq_map = df_tr[col].value_counts(normalize=True).to_dict()
        df_tr[f"{col}_FREQ"] = df_tr[col].map(freq_map).fillna(0.0).astype(np.float32)
        df_te[f"{col}_FREQ"] = df_te[col].map(freq_map).fillna(0.0).astype(np.float32)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_tr[col] = enc.fit_transform(df_tr[[col]].astype(str)).astype(np.float32)
        df_te[col] = enc.transform(df_te[[col]].astype(str)).astype(np.float32)

    # Numerical missingness indicators and imputation
    num_cols = df_tr.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df_tr[col].isnull().sum() > 0 or df_te[col].isnull().sum() > 0:
            df_tr[f"{col}_IS_NAN"] = df_tr[col].isnull().astype(np.float32)
            df_te[f"{col}_IS_NAN"] = df_te[col].isnull().astype(np.float32)
            med_val = float(df_tr[col].median()) if not np.isnan(df_tr[col].median()) else 0.0
            df_tr[col] = df_tr[col].fillna(med_val).astype(np.float32)
            df_te[col] = df_te[col].fillna(med_val).astype(np.float32)

    # Ensure all remaining columns are strictly numeric float32
    for col in df_tr.columns:
        df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").fillna(0.0).astype(np.float32)
        df_te[col] = pd.to_numeric(df_te[col], errors="coerce").fillna(0.0).astype(np.float32)

    return df_tr, df_te


X_train, X_test = engineer_features(train_features, test_features)

# Leak-Free Stratified 5-Fold Split
n_splits = 5
n_bins = 10
strat_bins = pd.qcut(target.rank(method="first"), q=n_bins, labels=False)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(X_train), dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, strat_bins)):
    fold_assignments[val_idx] = fold_idx

X_train["fold"] = fold_assignments
X_train[target_col] = target.values

feature_cols = [c for c in X_train.columns if c not in [target_col, "fold"]]
num_train = len(X_train)
num_test = len(X_test)
y_all = X_train[target_col].values.astype(np.float32)
min_target_val = float(np.min(y_all))


# ---------------------------------------------------------
# 3. Model Architectures & Configurations
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


def get_lgbm_config(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2500,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_config(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2500,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "n_jobs": -1,
    }


def get_catboost_config(seed: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "iterations": 2500,
        "verbose": False,
    }


# ---------------------------------------------------------
# 4. Training & Validation Loop
# ---------------------------------------------------------
models_to_train = ["lightgbm", "xgboost", "catboost", "tabular_resnet"]
oof_preds = {m: np.zeros(num_train, dtype=np.float32) for m in models_to_train}
test_preds = {m: np.zeros(num_test, dtype=np.float32) for m in models_to_train}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(n_splits):
    train_mask = X_train["fold"] != fold
    val_mask = X_train["fold"] == fold

    X_tr = X_train.loc[train_mask, feature_cols].copy()
    y_tr = X_train.loc[train_mask, target_col].values.astype(np.float32)

    X_va = X_train.loc[val_mask, feature_cols].copy()
    y_va = X_train.loc[val_mask, target_col].values.astype(np.float32)

    X_te = X_test[feature_cols].copy()
    val_indices = X_train.index[val_mask].values

    # 4.1 LightGBM
    lgb_model = lgb.LGBMRegressor(**get_lgbm_config(seed=42 + fold))
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb = lgb_model.predict(X_va)
    oof_preds["lightgbm"][val_indices] = oof_lgb
    test_preds["lightgbm"] += lgb_model.predict(X_te) / n_splits

    # 4.2 XGBoost
    xgb_model = xgb.XGBRegressor(**get_xgb_config(seed=42 + fold))
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb = xgb_model.predict(X_va)
    oof_preds["xgboost"][val_indices] = oof_xgb
    test_preds["xgboost"] += xgb_model.predict(X_te) / n_splits

    # 4.3 CatBoost
    cb_model = cb.CatBoostRegressor(**get_catboost_config(seed=42 + fold))
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb = cb_model.predict(X_va)
    oof_preds["catboost"][val_indices] = oof_cb
    test_preds["catboost"] += cb_model.predict(X_te) / n_splits

    # 4.4 Tabular ResNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr.fillna(0.0)).astype(np.float32)
    X_va_scaled = scaler.transform(X_va.fillna(0.0)).astype(np.float32)
    X_te_scaled = scaler.transform(X_te.fillna(0.0)).astype(np.float32)

    nn_model = TabularResNet(
        input_dim=len(feature_cols), hidden_dim=256, num_blocks=3, dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 45
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_dataset = TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_va_scaled), torch.from_numpy(y_va))

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)

    best_val_rmse = float("inf")
    best_weights = None

    for epoch in range(epochs):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = nn_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                val_preds_list.append(nn_model(bx).cpu().numpy())
        val_epoch_preds = np.concatenate(val_preds_list)
        val_epoch_rmse = root_mean_squared_error(y_va, val_epoch_preds)

        if val_epoch_rmse < best_val_rmse:
            best_val_rmse = val_epoch_rmse
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_va_scaled).to(device)
        oof_nn = nn_model(val_tensor).cpu().numpy()

        test_tensor = torch.from_numpy(X_te_scaled).to(device)
        test_nn = nn_model(test_tensor).cpu().numpy()

    oof_preds["tabular_resnet"][val_indices] = oof_nn
    test_preds["tabular_resnet"] += test_nn / n_splits

    fold_lgb_rmse = root_mean_squared_error(y_va, oof_lgb)
    fold_xgb_rmse = root_mean_squared_error(y_va, oof_xgb)
    fold_cb_rmse = root_mean_squared_error(y_va, oof_cb)
    fold_nn_rmse = root_mean_squared_error(y_va, oof_nn)
    print(
        f"Fold {fold+1}/{n_splits} - LGB: {fold_lgb_rmse:.4f} | XGB: {fold_xgb_rmse:.4f} | CB: {fold_cb_rmse:.4f} | ResNet: {fold_nn_rmse:.4f}"
    )

# ---------------------------------------------------------
# 5. Constrained Ensemble Optimization
# ---------------------------------------------------------
for m in models_to_train:
    model_score = root_mean_squared_error(y_all, oof_preds[m])
    print(f"Overall OOF RMSE for {m}: {model_score:.4f}")

oof_matrix = np.column_stack([oof_preds[m] for m in models_to_train])
test_matrix = np.column_stack([test_preds[m] for m in models_to_train])


def ensemble_loss(weights):
    weights = weights / np.sum(weights)
    pred_ensemble = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y_all, pred_ensemble)


initial_weights = np.ones(len(models_to_train)) / len(models_to_train)
bounds = [(0.0, 1.0) for _ in range(len(models_to_train))]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = (
    opt_res.x
    if opt_res.success
    else np.ones(len(models_to_train)) / len(models_to_train)
)
best_weights = best_weights / np.sum(best_weights)

print(
    "Optimal Ensemble Weights: "
    + ", ".join(f"{m}: {w:.4f}" for m, w in zip(models_to_train, best_weights))
)

final_oof_preds = np.dot(oof_matrix, best_weights)
final_validation_score = float(root_mean_squared_error(y_all, final_oof_preds))

# ---------------------------------------------------------
# 6. Generate Test Predictions & Save Submission
# ---------------------------------------------------------
final_test_preds = np.dot(test_matrix, best_weights)
final_test_preds = np.maximum(final_test_preds, min_target_val)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), target_col: final_test_preds}
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert (
    len(submission_df) == num_test
), f"Expected {num_test} rows, found {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    "NET_GEN",
], f"Invalid columns: {submission_df.columns}"
assert submission_df["NET_GEN"].isna().sum() == 0, "Submission contains NaN values!"

print(f"Final Validation Score: {final_validation_score}")