import os
import random
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 0. Reproducibility Seed Setter
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Feature Engineering & Data Processing Pipeline
# =========================================================================
def engineer_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "youSaveSpend",
    n_splits: int = 10,
    seed: int = 42,
):
    """
    Complete data processing and feature engineering pipeline for vehicle fuel economy.
    Applies reciprocal consumption rates, engine interactions, vintage metrics, frequency encoding,
    and strictly isolated out-of-fold target encoding.
    """
    os.makedirs("./working", exist_ok=True)

    y_train = df_train[target_col].copy() if target_col in df_train.columns else None
    X_train = df_train.drop(columns=[target_col], errors="ignore").copy()
    X_test = df_test.drop(columns=[target_col], errors="ignore").copy()

    n_train = len(X_train)

    # 1.1 Domain transforms & physical fuel economy features
    def apply_domain_transforms(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()

        # Reciprocal Fuel Consumption (Gallons per Mile = 1 / MPG)
        for mpg_col in [
            "comb08",
            "city08",
            "highway08",
            "combA08",
            "cityA08",
            "highwayA08",
            "UCity",
            "UHighway",
        ]:
            if mpg_col in res.columns:
                safe_mpg = res[mpg_col].replace(0, np.nan)
                res[f"{mpg_col}_gpm"] = 1.0 / safe_mpg
                res[f"{mpg_col}_gpm_100m"] = 100.0 / safe_mpg

        # MPG Ratios & Differentials
        if "city08" in res.columns and "highway08" in res.columns:
            res["city_hwy_ratio"] = res["city08"] / res["highway08"].replace(0, np.nan)
            res["hwy_city_diff"] = res["highway08"] - res["city08"]
            res["city_hwy_harmonic_mean"] = 2.0 / (
                (1.0 / res["city08"].replace(0, np.nan))
                + (1.0 / res["highway08"].replace(0, np.nan))
            )

        # Engine Mechanics
        if "displ" in res.columns and "cylinders" in res.columns:
            res["displ_per_cyl"] = res["displ"] / res["cylinders"].replace(0, np.nan)
            res["cyl_x_displ"] = res["cylinders"] * res["displ"]

        # Fuel Cost Interactions & EPA 5-Year Physical Metrics
        if "fuelCost08" in res.columns:
            res["fuelCost08_5yr"] = 5.0 * res["fuelCost08"]
            if "comb08" in res.columns:
                safe_comb = res["comb08"].replace(0, np.nan)
                res["cost_per_comb_mpg"] = res["fuelCost08"] * (1.0 / safe_comb)
                res["annual_gallons_comb"] = 15000.0 / safe_comb
                res["five_year_gallons_comb"] = 75000.0 / safe_comb
                res["implied_fuel_price"] = (
                    res["fuelCost08"] * safe_comb
                ) / 15000.0

        if "fuelCostA08" in res.columns:
            res["fuelCostA08_5yr"] = 5.0 * res["fuelCostA08"].fillna(0)
            if "combA08" in res.columns:
                safe_combA = res["combA08"].replace(0, np.nan)
                res["annual_gallons_combA"] = 15000.0 / safe_combA
                res["five_year_gallons_combA"] = 75000.0 / safe_combA

        if "fuelCost08" in res.columns and "fuelCostA08" in res.columns:
            res["fuelCost_total"] = res["fuelCost08"] + res["fuelCostA08"].fillna(0)
            res["fuelCost_total_5yr"] = 5.0 * res["fuelCost_total"]
            res["fuelCost_diff"] = res["fuelCost08"] - res["fuelCostA08"].fillna(0)
            res["is_dual_fuel"] = ((res["fuelCostA08"].fillna(0) > 0)).astype(int)
            res["fuelCost_ratio"] = res["fuelCost08"] / (
                res["fuelCostA08"].fillna(0) + 1.0
            )

        # Emissions efficiency
        if "co2TailpipeGpm" in res.columns and "comb08" in res.columns:
            res["co2_per_comb_gpm"] = res["co2TailpipeGpm"] / res["comb08"].replace(
                0, np.nan
            )
        if "co2" in res.columns and "displ" in res.columns:
            res["co2_per_displ"] = res["co2"] / res["displ"].replace(0, np.nan)

        # Vehicle Vintage
        if "year" in res.columns:
            res["vehicle_age"] = 2025 - res["year"]
            res["is_modern_era"] = (res["year"] >= 2010).astype(int)
            res["is_ev_era"] = (res["year"] >= 2018).astype(int)

        # Volume aggregates
        vol_cols = [
            c for c in ["hlv", "hpv", "lv2", "lv4", "pv2", "pv4"] if c in res.columns
        ]
        if len(vol_cols) > 0:
            res["total_volume"] = res[vol_cols].sum(axis=1)

        return res

    X_train_eng = apply_domain_transforms(X_train)
    X_test_eng = apply_domain_transforms(X_test)

    # Compute year-based and class-based benchmark differentials
    all_features = pd.concat([X_train_eng, X_test_eng], axis=0)
    if "year" in all_features.columns and "fuelCost_total_5yr" in all_features.columns:
        year_bench_5yr = (
            all_features.groupby("year")["fuelCost_total_5yr"].mean().to_dict()
        )
        X_train_eng["year_bench_fuelCost_5yr"] = X_train_eng["year"].map(
            year_bench_5yr
        )
        X_test_eng["year_bench_fuelCost_5yr"] = X_test_eng["year"].map(
            year_bench_5yr
        )
        X_train_eng["diff_year_bench_5yr"] = (
            X_train_eng["fuelCost_total_5yr"] - X_train_eng["year_bench_fuelCost_5yr"]
        )
        X_test_eng["diff_year_bench_5yr"] = (
            X_test_eng["fuelCost_total_5yr"] - X_test_eng["year_bench_fuelCost_5yr"]
        )

        year_bench_1yr = (
            all_features.groupby("year")["fuelCost08"].mean().to_dict()
        )
        X_train_eng["year_bench_fuelCost08"] = X_train_eng["year"].map(
            year_bench_1yr
        )
        X_test_eng["year_bench_fuelCost08"] = X_test_eng["year"].map(
            year_bench_1yr
        )
        X_train_eng["diff_year_bench_1yr"] = (
            X_train_eng["fuelCost08"] - X_train_eng["year_bench_fuelCost08"]
        )
        X_test_eng["diff_year_bench_1yr"] = (
            X_test_eng["fuelCost08"] - X_test_eng["year_bench_fuelCost08"]
        )

    if "VClass" in all_features.columns and "fuelCost_total_5yr" in all_features.columns:
        vclass_bench_5yr = (
            all_features.groupby("VClass")["fuelCost_total_5yr"].mean().to_dict()
        )
        X_train_eng["vclass_bench_fuelCost_5yr"] = X_train_eng["VClass"].map(
            vclass_bench_5yr
        )
        X_test_eng["vclass_bench_fuelCost_5yr"] = X_test_eng["VClass"].map(
            vclass_bench_5yr
        )
        X_train_eng["diff_vclass_bench_5yr"] = (
            X_train_eng["fuelCost_total_5yr"]
            - X_train_eng["vclass_bench_fuelCost_5yr"]
        )
        X_test_eng["diff_vclass_bench_5yr"] = (
            X_test_eng["fuelCost_total_5yr"]
            - X_test_eng["vclass_bench_fuelCost_5yr"]
        )

    # 1.2 Identify Column Modalities
    numeric_cols = []
    categorical_cols = []

    for col in X_train_eng.columns:
        if col in ["id"]:
            continue
        if pd.api.types.is_numeric_dtype(X_train_eng[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # 1.3 Categorical Processing: Frequency Encodings & Label Encodings
    for col in categorical_cols:
        X_train_eng[col] = X_train_eng[col].astype(str).fillna("Missing")
        X_test_eng[col] = X_test_eng[col].astype(str).fillna("Missing")

        freq_map = X_train_eng[col].value_counts(normalize=True).to_dict()
        X_train_eng[f"{col}_freq"] = X_train_eng[col].map(freq_map).fillna(0.0)
        X_test_eng[f"{col}_freq"] = X_test_eng[col].map(freq_map).fillna(0.0)
        numeric_cols.append(f"{col}_freq")

        categories = sorted(
            list(set(X_train_eng[col].unique()).union(set(X_test_eng[col].unique())))
        )
        cat_map = {cat: idx for idx, cat in enumerate(categories)}
        X_train_eng[f"{col}_cat"] = X_train_eng[col].map(cat_map).astype(int)
        X_test_eng[f"{col}_cat"] = X_test_eng[col].map(cat_map).astype(int)
        numeric_cols.append(f"{col}_cat")

    X_train_eng = X_train_eng.drop(columns=categorical_cols, errors="ignore")
    X_test_eng = X_test_eng.drop(columns=categorical_cols, errors="ignore")

    # 1.4 Isolated Cross-Validation Splitting & Out-of-Fold Target Encoding & Benchmark References
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_folds = np.zeros(n_train, dtype=int)

    te_cols = [
        f"{c}_cat" for c in categorical_cols if f"{c}_cat" in X_train_eng.columns
    ]

    fuel_cost_col = "fuelCost08" if "fuelCost08" in X_train_eng.columns else None
    if fuel_cost_col is not None and "year" in X_train_eng.columns:
        X_train_eng["epa_c_year_oof"] = np.nan
        train_ref_costs = y_train + 5.0 * X_train_eng[fuel_cost_col]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_eng, y_train)):
        train_folds[val_idx] = fold

        for te_col in te_cols:
            te_name = f"{te_col}_target_enc"
            if te_name not in X_train_eng.columns:
                X_train_eng[te_name] = np.nan

            fold_train_target = y_train.iloc[train_idx]
            fold_train_feats = X_train_eng[te_col].iloc[train_idx]
            global_mean = fold_train_target.mean()

            stats = fold_train_target.groupby(fold_train_feats).agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + 10.0 * global_mean) / (
                stats["count"] + 10.0
            )
            te_dict = smoothed.to_dict()

            X_train_eng.loc[X_train_eng.index[val_idx], te_name] = (
                X_train_eng[te_col].iloc[val_idx].map(te_dict).fillna(global_mean)
            )

        if fuel_cost_col is not None and "year" in X_train_eng.columns:
            fold_train_ref = train_ref_costs.iloc[train_idx]
            fold_train_years = X_train_eng["year"].iloc[train_idx]
            global_ref_mean = fold_train_ref.mean()

            year_ref_stats = fold_train_ref.groupby(fold_train_years).agg(["count", "mean"])
            smoothed_year_ref = (
                year_ref_stats["count"] * year_ref_stats["mean"] + 5.0 * global_ref_mean
            ) / (year_ref_stats["count"] + 5.0)
            year_ref_dict = smoothed_year_ref.to_dict()

            X_train_eng.loc[X_train_eng.index[val_idx], "epa_c_year_oof"] = (
                X_train_eng["year"].iloc[val_idx].map(year_ref_dict).fillna(global_ref_mean)
            )

    # Full train target encoding & reference benchmark mapping to test
    global_mean_full = y_train.mean()
    for te_col in te_cols:
        te_name = f"{te_col}_target_enc"
        stats = y_train.groupby(X_train_eng[te_col]).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + 10.0 * global_mean_full) / (
            stats["count"] + 10.0
        )
        te_dict = smoothed.to_dict()
        X_test_eng[te_name] = X_test_eng[te_col].map(te_dict).fillna(global_mean_full)

    if fuel_cost_col is not None and "year" in X_train_eng.columns:
        global_ref_full = (y_train + 5.0 * X_train_eng[fuel_cost_col]).mean()
        full_ref_stats = (
            (y_train + 5.0 * X_train_eng[fuel_cost_col])
            .groupby(X_train_eng["year"])
            .agg(["count", "mean"])
        )
        smoothed_full_ref = (
            full_ref_stats["count"] * full_ref_stats["mean"] + 5.0 * global_ref_full
        ) / (full_ref_stats["count"] + 5.0)
        full_ref_dict = smoothed_full_ref.to_dict()

        X_test_eng["epa_c_year_oof"] = (
            X_test_eng["year"].map(full_ref_dict).fillna(global_ref_full)
        )
        X_train_eng["epa_physical_base_ref"] = (
            X_train_eng["epa_c_year_oof"] - 5.0 * X_train_eng[fuel_cost_col]
        )
        X_test_eng["epa_physical_base_ref"] = (
            X_test_eng["epa_c_year_oof"] - 5.0 * X_test_eng[fuel_cost_col]
        )

    # 1.5 Missing Value & Infinite Value Imputation
    X_train_eng = X_train_eng.replace([np.inf, -np.inf], np.nan)
    X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan)

    train_medians = X_train_eng.median()
    X_train_eng = X_train_eng.fillna(train_medians).fillna(0.0)
    X_test_eng = X_test_eng.fillna(train_medians).fillna(0.0)

    final_cols = [c for c in X_train_eng.columns if c not in ["id"]]
    X_train_final = X_train_eng[final_cols].copy()
    X_test_final = X_test_eng[final_cols].copy()

    X_train_final["fold"] = train_folds
    X_train_final[target_col] = y_train.values

    X_train_final.to_parquet("./working/train_engineered.parquet", index=False)
    X_test_final.to_parquet("./working/test_engineered.parquet", index=False)

    return X_train_final, X_test_final


# =========================================================================
# 1.6 Physical EPA Reference Baseline Estimator
# =========================================================================
class PhysicalBaselineEstimator:
    """
    Computes continuous out-of-fold EPA baseline references C_year = y + 5 * fuelCost08
    with Bayesian shrinkage and linear interpolation across vehicle model years.
    """

    def __init__(self):
        self.year_baseline_map = {}
        self.global_baseline = 0.0
        self.known_years = np.array([], dtype=np.float64)
        self.known_values = np.array([], dtype=np.float64)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        fuel_cost = (
            X["fuelCost08"].values
            if "fuelCost08" in X.columns
            else np.zeros_like(y, dtype=np.float64)
        )
        c_samples = y.astype(np.float64) + 5.0 * fuel_cost.astype(np.float64)
        years = (
            X["year"].values
            if "year" in X.columns
            else np.zeros_like(y, dtype=np.float64)
        )
        self.global_baseline = float(np.median(c_samples))

        df_c = pd.DataFrame({"year": years, "c": c_samples})
        stats = df_c.groupby("year")["c"].agg(["count", "median"])

        self.year_baseline_map = {}
        for yr, row in stats.iterrows():
            cnt = row["count"]
            med = row["median"]
            smoothed_c = (cnt * med + 5.0 * self.global_baseline) / (cnt + 5.0)
            self.year_baseline_map[yr] = smoothed_c

        self.known_years = np.array(
            sorted(self.year_baseline_map.keys()), dtype=np.float64
        )
        self.known_values = np.array(
            [self.year_baseline_map[y] for y in self.known_years], dtype=np.float64
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fuel_cost = (
            X["fuelCost08"].values
            if "fuelCost08" in X.columns
            else np.zeros(len(X), dtype=np.float64)
        )
        years = (
            X["year"].values
            if "year" in X.columns
            else np.zeros(len(X), dtype=np.float64)
        )

        if len(self.known_years) > 1:
            c_pred = np.interp(years, self.known_years, self.known_values)
        elif len(self.known_years) == 1:
            c_pred = np.full(len(X), self.known_values[0], dtype=np.float64)
        else:
            c_pred = np.full(len(X), self.global_baseline, dtype=np.float64)

        return c_pred - 5.0 * fuel_cost.astype(np.float64)


# =========================================================================
# 2. PyTorch Tabular Neural Network Architecture (Tabular ResNet)
# =========================================================================
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
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = self.activation(out + residual)
        return out


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


def get_torch_model_and_optimizer(
    input_dim: int,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
    t_max: int = 60,
    eta_min: float = 1e-6,
    device: str = "cpu",
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        output_dim=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    return model, criterion, optimizer, scheduler


# =========================================================================
# 3. Gradient Boosted Tree Model Configurations
# =========================================================================
def get_lgbm_model(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(seed: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(seed: int = 42) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        subsample=0.85,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


# =========================================================================
# 4. Main Training, Cross-Validation, Blending & Inference Pipeline
# =========================================================================
def main():
    # Load input data
    train_df_raw = pd.read_parquet("./input/train.parquet")
    test_df_raw = pd.read_parquet("./input/test.parquet")

    target_col = "youSaveSpend"
    train_df, test_df = engineer_features(
        train_df_raw, test_df_raw, target_col=target_col, n_splits=10, seed=42
    )

    feature_cols = [c for c in train_df.columns if c not in ["fold", target_col]]
    n_splits = train_df["fold"].nunique()
    n_train = len(train_df)
    n_test = len(test_df)

    X_test_raw = test_df[feature_cols].values
    y_train_all = train_df[target_col].values

    model_names = ["lgb", "xgb", "cb", "nn", "ridge"]
    oof_preds = {m: np.zeros(n_train, dtype=np.float64) for m in model_names}
    test_preds = {m: np.zeros(n_test, dtype=np.float64) for m in model_names}

    # Cross-Validation Training Loop with Physics-Informed Residual Learning
    for fold in range(n_splits):
        train_idx = (train_df["fold"] != fold).values
        val_idx = (train_df["fold"] == fold).values

        X_tr_df = train_df.loc[train_idx, feature_cols]
        y_tr = train_df.loc[train_idx, target_col].values
        X_val_df = train_df.loc[val_idx, feature_cols]
        y_val = train_df.loc[val_idx, target_col].values
        X_test_df = test_df[feature_cols]

        # 4.0 Physical Reference Baseline Estimation
        base_estimator = PhysicalBaselineEstimator()
        base_estimator.fit(X_tr_df, y_tr)

        base_pred_tr = base_estimator.predict(X_tr_df)
        base_pred_val = base_estimator.predict(X_val_df)
        base_pred_test = base_estimator.predict(X_test_df)

        # Compute Target Residuals
        r_tr = y_tr - base_pred_tr
        r_val = y_val - base_pred_val

        # 4.1 LightGBM Residual Fitting
        lgb_model = get_lgbm_model(seed=42 + fold)
        lgb_model.fit(
            X_tr_df,
            r_tr,
            eval_set=[(X_val_df, r_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_preds["lgb"][val_idx] = base_pred_val + lgb_model.predict(X_val_df)
        test_preds["lgb"] += (base_pred_test + lgb_model.predict(X_test_df)) / n_splits

        # 4.2 XGBoost Residual Fitting
        xgb_model = get_xgb_model(seed=42 + fold)
        xgb_model.fit(X_tr_df, r_tr, eval_set=[(X_val_df, r_val)], verbose=False)
        oof_preds["xgb"][val_idx] = base_pred_val + xgb_model.predict(X_val_df)
        test_preds["xgb"] += (base_pred_test + xgb_model.predict(X_test_df)) / n_splits

        # 4.3 CatBoost Residual Fitting
        cb_model = get_catboost_model(seed=42 + fold)
        cb_model.fit(
            X_tr_df,
            r_tr,
            eval_set=(X_val_df, r_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_preds["cb"][val_idx] = base_pred_val + cb_model.predict(X_val_df)
        test_preds["cb"] += (base_pred_test + cb_model.predict(X_test_df)) / n_splits

        # 4.4 PyTorch TabularResNet Residual Fitting
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_df.values)
        X_val_scaled = scaler.transform(X_val_df.values)
        X_test_scaled = scaler.transform(X_test_df.values)

        train_tensor_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(r_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_tensor_dataset, batch_size=128, shuffle=True, drop_last=False
        )

        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

        nn_model, criterion, optimizer, scheduler = get_torch_model_and_optimizer(
            input_dim=len(feature_cols),
            lr=3e-3,
            weight_decay=1e-4,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.15,
            t_max=60,
            device=str(device),
        )

        best_val_loss = float("inf")
        best_nn_weights = None
        epochs = 60

        for epoch in range(epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_out = nn_model(X_val_t)
                val_loss = criterion(
                    val_out, torch.tensor(r_val, dtype=torch.float32).to(device)
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
        nn_model.eval()
        with torch.no_grad():
            nn_val_res = nn_model(X_val_t).cpu().numpy().flatten()
            nn_test_res = nn_model(X_test_t).cpu().numpy().flatten()
            oof_preds["nn"][val_idx] = base_pred_val + nn_val_res
            test_preds["nn"] += (
                (base_pred_test + nn_test_res) / n_splits
            ).astype(np.float64)

        # 4.5 Regularized Ridge Regression Residual Fitting
        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X_tr_scaled, r_tr)
        oof_preds["ridge"][val_idx] = base_pred_val + ridge_model.predict(X_val_scaled)
        test_preds["ridge"] += (base_pred_test + ridge_model.predict(X_test_scaled)) / n_splits

    # 4.6 Optimal Ensemble Blending via SLSQP
    def ensemble_loss(weights):
        weights = np.array(weights)
        weights = np.maximum(0, weights)
        weights = weights / (np.sum(weights) + 1e-8)
        blended = sum(weights[i] * oof_preds[m] for i, m in enumerate(model_names))
        return root_mean_squared_error(y_train_all, blended)

    initial_weights = [1.0 / len(model_names)] * len(model_names)
    bounds = [(0.0, 1.0) for _ in model_names]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        ensemble_loss,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = opt_res.x / np.sum(opt_res.x)

    # 4.7 Final OOF Prediction & Metric Evaluation
    final_oof_pred = sum(
        optimal_weights[i] * oof_preds[m] for i, m in enumerate(model_names)
    )
    final_score = root_mean_squared_error(y_train_all, final_oof_pred)

    # 4.8 Generate Test Predictions & Save Submission
    final_test_pred = sum(
        optimal_weights[i] * test_preds[m] for i, m in enumerate(model_names)
    )

    os.makedirs("./submission", exist_ok=True)
    if "id" in test_df_raw.columns:
        sub_df = pd.DataFrame({"id": test_df_raw["id"].values, target_col: final_test_pred})
    else:
        sub_df = pd.DataFrame({"id": np.arange(n_test), target_col: final_test_pred})
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
