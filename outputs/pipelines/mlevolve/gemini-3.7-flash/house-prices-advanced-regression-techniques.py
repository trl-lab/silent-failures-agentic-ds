import json
import math
import os
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Environment & Path Setup
# =========================================================================
WORKING_DIR = Path("./working")
SUBMISSION_DIR = Path("./submission")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = Path("./input/train.parquet")
TEST_PATH = Path("./input/test.parquet")

# =========================================================================
# 2. Metric Definition
# =========================================================================


def compute_rmsle(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """Computes Root Mean Squared Logarithmic Error (RMSLE)."""
    y_true_clipped = np.clip(y_true_raw, 0, None)
    y_pred_clipped = np.clip(y_pred_raw, 0, None)
    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


# =========================================================================
# 3. Data Processing & Feature Engineering
# =========================================================================
print("Loading raw datasets...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "SalePrice"
y_raw = train_df[target_col].copy().values
y_log = np.log1p(y_raw)
n_train = len(train_df)
n_test = len(test_df)

combined_df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Physical absence categoricals
none_cat_cols = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]
for col in none_cat_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna("None").astype(str)

# Numerical amenity absence zeros
zero_num_cols = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]
for col in zero_num_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna(0.0)

# Impute LotFrontage strictly using train statistics
train_frontage = (
    combined_df.iloc[:n_train].groupby("Neighborhood")["LotFrontage"].median()
)
global_frontage_med = combined_df.iloc[:n_train]["LotFrontage"].median()
combined_df["LotFrontage"] = combined_df.apply(
    lambda row: (
        train_frontage.get(row["Neighborhood"], global_frontage_med)
        if pd.isna(row["LotFrontage"])
        else row["LotFrontage"]
    ),
    axis=1,
)

# Impute remaining categoricals using train modes
mode_cat_cols = [
    "MSZoning",
    "Utilities",
    "Exterior1st",
    "Exterior2nd",
    "Electrical",
    "KitchenQual",
    "Functional",
    "SaleType",
]
for col in mode_cat_cols:
    if col in combined_df.columns:
        train_mode = combined_df.iloc[:n_train][col].mode()
        fill_val = train_mode.iloc[0] if len(train_mode) > 0 else "None"
        combined_df[col] = combined_df[col].fillna(fill_val).astype(str)

# Full Ordinal Appraisal & Condition Mappings
ordinal_qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
qual_cols_to_map = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]
for col in qual_cols_to_map:
    if col in combined_df.columns:
        combined_df[f"{col}_score"] = (
            combined_df[col].map(ordinal_qual_map).fillna(0).astype(float)
        )

bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
if "BsmtExposure" in combined_df.columns:
    combined_df["BsmtExposure_score"] = (
        combined_df["BsmtExposure"].map(bsmt_exposure_map).fillna(0).astype(float)
    )

bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
for col in ["BsmtFinType1", "BsmtFinType2"]:
    if col in combined_df.columns:
        combined_df[f"{col}_score"] = (
            combined_df[col].map(bsmt_fin_map).fillna(0).astype(float)
        )

garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
if "GarageFinish" in combined_df.columns:
    combined_df["GarageFinish_score"] = (
        combined_df["GarageFinish"].map(garage_finish_map).fillna(0).astype(float)
    )

landslope_map = {"Gtl": 1, "Mod": 2, "Sev": 3, "None": 0}
if "LandSlope" in combined_df.columns:
    combined_df["LandSlope_score"] = (
        combined_df["LandSlope"].map(landslope_map).fillna(0).astype(float)
    )

utilities_map = {"AllPub": 4, "NoSewr": 3, "NoSeWa": 2, "ELO": 1, "None": 0}
if "Utilities" in combined_df.columns:
    combined_df["Utilities_score"] = (
        combined_df["Utilities"].map(utilities_map).fillna(0).astype(float)
    )

centralair_map = {"Y": 1, "N": 0, "None": 0}
if "CentralAir" in combined_df.columns:
    combined_df["CentralAir_score"] = (
        combined_df["CentralAir"].map(centralair_map).fillna(0).astype(float)
    )

paveddrive_map = {"Y": 3, "P": 2, "N": 1, "None": 0}
if "PavedDrive" in combined_df.columns:
    combined_df["PavedDrive_score"] = (
        combined_df["PavedDrive"].map(paveddrive_map).fillna(0).astype(float)
    )

functional_map = {
    "Typ": 8,
    "Min1": 7,
    "Min2": 6,
    "Mod": 5,
    "Maj1": 4,
    "Maj2": 3,
    "Sev": 2,
    "Sal": 1,
    "None": 0,
}
if "Functional" in combined_df.columns:
    combined_df["Functional_score"] = (
        combined_df["Functional"].map(functional_map).fillna(0).astype(float)
    )

lotshape_map = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1, "None": 0}
if "LotShape" in combined_df.columns:
    combined_df["LotShape_score"] = (
        combined_df["LotShape"].map(lotshape_map).fillna(0).astype(float)
    )

fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
if "Fence" in combined_df.columns:
    combined_df["Fence_score"] = (
        combined_df["Fence"].map(fence_map).fillna(0).astype(float)
    )

electrical_map = {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP": 2, "Mix": 1, "None": 0}
if "Electrical" in combined_df.columns:
    combined_df["Electrical_score"] = (
        combined_df["Electrical"].map(electrical_map).fillna(0).astype(float)
    )

# Domain Feature Engineering
combined_df["TotalSF"] = (
    combined_df["TotalBsmtSF"] + combined_df["1stFlrSF"] + combined_df["2ndFlrSF"]
)
combined_df["TotalLivingSF"] = combined_df["GrLivArea"] + combined_df["TotalBsmtSF"]
combined_df["TotalPorchSF"] = (
    combined_df["OpenPorchSF"]
    + combined_df["EnclosedPorch"]
    + combined_df["3SsnPorch"]
    + combined_df["ScreenPorch"]
    + combined_df["WoodDeckSF"]
)

combined_df["TotalBath"] = (
    combined_df["FullBath"]
    + 0.5 * combined_df["HalfBath"]
    + combined_df["BsmtFullBath"]
    + 0.5 * combined_df["BsmtHalfBath"]
)

combined_df["HouseAge"] = np.maximum(
    0, combined_df["YrSold"] - combined_df["YearBuilt"]
)
combined_df["RemodelAge"] = np.maximum(
    0, combined_df["YrSold"] - combined_df["YearRemodAdd"]
)
combined_df["EffectiveAge"] = (
    0.65 * combined_df["RemodelAge"] + 0.35 * combined_df["HouseAge"]
)
combined_df["GarageAge"] = np.where(
    combined_df["GarageYrBlt"] > 0,
    np.maximum(0, combined_df["YrSold"] - combined_df["GarageYrBlt"]),
    combined_df["HouseAge"],
)
combined_df["IsRemodeled"] = (
    combined_df["YearRemodAdd"] != combined_df["YearBuilt"]
).astype(int)
combined_df["IsNew"] = (combined_df["YrSold"] == combined_df["YearBuilt"]).astype(int)

# High Quality Finished Square Footage
is_high_qual_bsmt = (combined_df["BsmtFinType1_score"] >= 5).astype(float)
combined_df["HighQualFinSF"] = (
    combined_df["1stFlrSF"]
    + combined_df["2ndFlrSF"]
    + combined_df["BsmtFinSF1"] * is_high_qual_bsmt
)

# Composite Quality and Finished Space Metrics
combined_df["TotalQuality"] = (
    combined_df["OverallQual"]
    + combined_df["ExterQual_score"]
    + combined_df["BsmtQual_score"]
    + combined_df["KitchenQual_score"]
    + combined_df["HeatingQC_score"]
    + combined_df["GarageQual_score"]
    + combined_df["FireplaceQu_score"] * 0.5
)
combined_df["TotalFinishedSF"] = (
    combined_df["BsmtFinSF1"]
    + combined_df["BsmtFinSF2"]
    + combined_df["1stFlrSF"]
    + combined_df["2ndFlrSF"]
)

# Total Amenities Score
combined_df["TotalAmenitiesScore"] = (
    combined_df["Fireplaces"] * (combined_df["FireplaceQu_score"] + 1.0)
    + combined_df["TotalBath"] * 2.0
    + combined_df["GarageCars"] * 1.5
    + (combined_df["PoolArea"] > 0).astype(float) * 3.0
    + (combined_df["TotalPorchSF"] > 0).astype(float) * 1.0
    + combined_df["CentralAir_score"] * 2.0
    + combined_df["PavedDrive_score"] * 0.5
)

# Multipliers & Interactions
combined_df["OverallQual_sq"] = combined_df["OverallQual"] ** 2
combined_df["OverallGrade"] = combined_df["OverallQual"] * combined_df["OverallCond"]
combined_df["Qual_SF_Interaction"] = combined_df["OverallQual"] * combined_df["TotalSF"]
combined_df["HighQual_SF_Interaction"] = (
    combined_df["OverallQual"] * combined_df["HighQualFinSF"]
)
combined_df["TotalSF_TotalQuality"] = combined_df["TotalSF"] * combined_df["TotalQuality"]
combined_df["Living_Qual_Score"] = combined_df["OverallQual"] * combined_df["GrLivArea"]
combined_df["Kitchen_Score"] = (
    combined_df["KitchenQual_score"] * combined_df["KitchenAbvGr"]
)
combined_df["Exter_Score"] = (
    combined_df["ExterQual_score"] * combined_df["ExterCond_score"]
)
combined_df["Bsmt_Score"] = (
    combined_df["BsmtQual_score"] * combined_df["BsmtCond_score"]
)
combined_df["Garage_Score"] = (
    combined_df["GarageQual_score"] * combined_df["GarageCond_score"]
)

# Extended Domain Features & Interactions
combined_df["TotalRealEstateSF"] = (
    combined_df["TotalSF"] + combined_df["GarageArea"] + combined_df["TotalPorchSF"]
)
combined_df["BathToBedRatio"] = combined_df["TotalBath"] / (
    combined_df["BedroomAbvGr"] + 1.0
)
combined_df["PorchRatio"] = combined_df["TotalPorchSF"] / (
    combined_df["GrLivArea"] + 1.0
)
combined_df["QualityAgeRatio"] = combined_df["TotalQuality"] / (
    combined_df["EffectiveAge"] + 1.0
)
combined_df["OverallQual_Age"] = combined_df["OverallQual"] * (
    100.0 / (combined_df["EffectiveAge"] + 10.0)
)

# Frequency Encodings across Nominal Categories
freq_cols = [
    "Neighborhood",
    "MSSubClass",
    "MSZoning",
    "Exterior1st",
    "Exterior2nd",
    "Foundation",
    "BldgType",
    "HouseStyle",
    "SaleType",
    "SaleCondition",
]
for col in freq_cols:
    if col in combined_df.columns:
        freq = combined_df[col].value_counts(normalize=True).to_dict()
        combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0.0).astype(float)

# Cyclical trigonometric features for MoSold
mosold_numeric = pd.to_numeric(combined_df["MoSold"], errors="coerce").fillna(1).values
combined_df["MoSold_sin"] = np.sin(2.0 * np.pi * mosold_numeric / 12.0)
combined_df["MoSold_cos"] = np.cos(2.0 * np.pi * mosold_numeric / 12.0)

# Indicators & Ratios
combined_df["HasPool"] = (combined_df["PoolArea"] > 0).astype(int)
combined_df["Has2ndFlr"] = (combined_df["2ndFlrSF"] > 0).astype(int)
combined_df["HasGarage"] = (combined_df["GarageArea"] > 0).astype(int)
combined_df["HasBasement"] = (combined_df["TotalBsmtSF"] > 0).astype(int)
combined_df["HasFireplace"] = (combined_df["Fireplaces"] > 0).astype(int)
combined_df["HasPorch"] = (combined_df["TotalPorchSF"] > 0).astype(int)

combined_df["LotDepthRatio"] = combined_df["LotArea"] / (
    combined_df["LotFrontage"] + 1.0
)
combined_df["LivingAreaRatio"] = combined_df["GrLivArea"] / (
    combined_df["LotArea"] + 1.0
)
combined_df["Bsmt_TotalSF_ratio"] = combined_df["TotalBsmtSF"] / (
    combined_df["TotalSF"] + 1.0
)
combined_df["Garage_Cars_ratio"] = combined_df["GarageArea"] / (
    combined_df["GarageCars"] + 1.0
)
combined_df["BsmtFinRatio"] = (
    combined_df["BsmtFinSF1"] + combined_df["BsmtFinSF2"]
) / (combined_df["TotalBsmtSF"] + 1.0)
combined_df["AvgRoomSF"] = combined_df["GrLivArea"] / (
    combined_df["TotRmsAbvGrd"] + 1e-5
)
combined_df["BedroomRatio"] = combined_df["BedroomAbvGr"] / (
    combined_df["TotRmsAbvGrd"] + 1e-5
)
combined_df["BathPerRoom"] = combined_df["TotalBath"] / (
    combined_df["TotRmsAbvGrd"] + 1e-5
)
combined_df["BsmtUnfRatio"] = combined_df["BsmtUnfSF"] / (
    combined_df["TotalBsmtSF"] + 1.0
)

# Log1p transformation on skewed continuous variables
skewed_cols = [
    "LotArea",
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "TotalSF",
    "TotalFinishedSF",
    "TotalLivingSF",
    "TotalRealEstateSF",
    "BsmtFinSF1",
    "BsmtUnfSF",
    "GarageArea",
    "OpenPorchSF",
    "WoodDeckSF",
    "LotFrontage",
    "MasVnrArea",
]
for col in skewed_cols:
    if col in combined_df.columns:
        combined_df[f"{col}_log"] = np.log1p(np.maximum(0, combined_df[col].values))

combined_df["MSSubClass"] = combined_df["MSSubClass"].astype(str)
combined_df["MoSold"] = combined_df["MoSold"].astype(str)
combined_df["YrSold_cat"] = combined_df["YrSold"].astype(str)

# Separate Train & Test
processed_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
processed_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

processed_train["SalePrice"] = y_raw
processed_train["SalePrice_log"] = y_log

# 10-Fold Stratified-by-target Split
target_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
processed_train["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(kf.split(processed_train, target_bins)):
    processed_train.loc[val_idx, "fold"] = fold_idx

all_features = [
    col
    for col in processed_train.columns
    if col not in ["SalePrice", "SalePrice_log", "fold", "Id", "id"]
]
cat_features = [
    col
    for col in all_features
    if processed_train[col].dtype == "object"
    or processed_train[col].dtype.name == "category"
]
num_features = [col for col in all_features if col not in cat_features]

# =========================================================================
# 4. Neural Network Architecture & Factories
# =========================================================================


class LogSpaceHuberLoss(nn.Module):

    def __init__(self, delta: float = 0.08):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=delta)

    def forward(
        self, y_pred_log: torch.Tensor, y_true_log: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(y_pred_log.squeeze(-1), y_true_log.squeeze(-1))


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.input_layer(x)
        for block in self.res_blocks:
            features = block(features)
        return self.head(features).squeeze(-1)


def create_neural_model_bundle(
    input_dim: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    max_epochs: int = 90,
    device: str = "cpu",
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=192, num_blocks=3, dropout_rate=0.25
    ).to(device)
    criterion = LogSpaceHuberLoss(delta=0.08)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )
    return model, criterion, optimizer, scheduler


def get_lgbm_regressor(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2500,
        learning_rate=0.02,
        num_leaves=20,
        max_depth=5,
        min_child_samples=12,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.20,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_regressor(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=2500,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.55,
        colsample_bylevel=0.75,
        reg_alpha=0.20,
        reg_lambda=3.0,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def get_catboost_regressor(seed: int = 42):
    if CATBOOST_AVAILABLE:
        return CatBoostRegressor(
            iterations=2500,
            learning_rate=0.02,
            depth=5,
            l2_leaf_reg=6.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed,
            verbose=False,
            thread_count=-1,
        )
    return None


# =========================================================================
# 5. Cross-Validation Training & Out-of-Fold Evaluation
# =========================================================================
n_splits = 10
folds = processed_train["fold"].values

models_dict = {
    "lgbm": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "xgb": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "extratrees": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "ridge": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "elasticnet": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "bayesian_ridge": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "svr": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
    "nn": {"oof": np.zeros(n_train), "test": np.zeros(n_test)},
}

if CATBOOST_AVAILABLE:
    models_dict["catboost"] = {"oof": np.zeros(n_train), "test": np.zeros(n_test)}

for fold in range(n_splits):
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    y_tr_log, y_va_log = y_log[train_idx], y_log[val_idx]
    y_va_raw = y_raw[val_idx]

    # Tree Features base copies
    X_tr_tree = processed_train.iloc[train_idx][all_features].copy()
    X_va_tree = processed_train.iloc[val_idx][all_features].copy()
    X_te_tree = processed_test[all_features].copy()

    # Fold-isolated Neighborhood Aggregate Statistics
    neigh_grliv = (
        processed_train.iloc[train_idx].groupby("Neighborhood")["GrLivArea"].median()
    )
    neigh_totalsf = (
        processed_train.iloc[train_idx].groupby("Neighborhood")["TotalSF"].median()
    )
    neigh_yrblt = (
        processed_train.iloc[train_idx].groupby("Neighborhood")["YearBuilt"].median()
    )

    glob_grliv = float(processed_train.iloc[train_idx]["GrLivArea"].median())
    glob_totalsf = float(processed_train.iloc[train_idx]["TotalSF"].median())
    glob_yrblt = float(processed_train.iloc[train_idx]["YearBuilt"].median())

    for df_split, df_src in [
        (X_tr_tree, processed_train.iloc[train_idx]),
        (X_va_tree, processed_train.iloc[val_idx]),
        (X_te_tree, processed_test),
    ]:
        df_split["Neigh_Med_GrLivArea"] = (
            df_src["Neighborhood"].map(neigh_grliv).fillna(glob_grliv).astype(float)
        )
        df_split["Neigh_Med_TotalSF"] = (
            df_src["Neighborhood"].map(neigh_totalsf).fillna(glob_totalsf).astype(float)
        )
        df_split["Neigh_Med_YearBuilt"] = (
            df_src["Neighborhood"].map(neigh_yrblt).fillna(glob_yrblt).astype(float)
        )
        df_split["GrLivArea_to_Neigh"] = df_split["GrLivArea"] / (
            df_split["Neigh_Med_GrLivArea"] + 1.0
        )
        df_split["TotalSF_to_Neigh"] = df_split["TotalSF"] / (
            df_split["Neigh_Med_TotalSF"] + 1.0
        )
        df_split["YearBuilt_diff_Neigh"] = (
            df_split["YearBuilt"] - df_split["Neigh_Med_YearBuilt"]
        )

    neigh_new_cols = [
        "Neigh_Med_GrLivArea",
        "Neigh_Med_TotalSF",
        "Neigh_Med_YearBuilt",
        "GrLivArea_to_Neigh",
        "TotalSF_to_Neigh",
        "YearBuilt_diff_Neigh",
    ]

    # Out-of-Fold Smoothed Target Encodings across all key nominal features
    te_cols = [
        "Neighborhood",
        "MSSubClass",
        "MSZoning",
        "Exterior1st",
        "Exterior2nd",
        "Foundation",
        "BldgType",
        "HouseStyle",
        "SaleType",
        "SaleCondition",
    ]
    te_features = []
    global_mean = float(np.mean(y_tr_log))
    global_median = float(np.median(y_tr_log))
    smoothing = 10.0

    for col in te_cols:
        if col in processed_train.columns:
            group = pd.DataFrame(
                {"cat": processed_train.iloc[train_idx][col].astype(str), "target": y_tr_log}
            ).groupby("cat")["target"]
            stats = group.agg(["count", "mean", "median"])
            smooth_mean = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            smooth_med = (stats["count"] * stats["median"] + smoothing * global_median) / (
                stats["count"] + smoothing
            )

            mean_col = f"{col}_te_mean"
            med_col = f"{col}_te_med"
            te_features.extend([mean_col, med_col])

            X_tr_tree[mean_col] = (
                processed_train.iloc[train_idx][col].astype(str).map(smooth_mean).fillna(global_mean).values
            )
            X_tr_tree[med_col] = (
                processed_train.iloc[train_idx][col].astype(str).map(smooth_med).fillna(global_median).values
            )
            X_va_tree[mean_col] = (
                processed_train.iloc[val_idx][col].astype(str).map(smooth_mean).fillna(global_mean).values
            )
            X_va_tree[med_col] = (
                processed_train.iloc[val_idx][col].astype(str).map(smooth_med).fillna(global_median).values
            )
            X_te_tree[mean_col] = (
                processed_test[col].astype(str).map(smooth_mean).fillna(global_mean).values
            )
            X_te_tree[med_col] = (
                processed_test[col].astype(str).map(smooth_med).fillna(global_median).values
            )

    ord_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32
    )
    if cat_features:
        X_tr_tree[cat_features] = ord_enc.fit_transform(
            X_tr_tree[cat_features].astype(str)
        )
        X_va_tree[cat_features] = ord_enc.transform(X_va_tree[cat_features].astype(str))
        X_te_tree[cat_features] = ord_enc.transform(X_te_tree[cat_features].astype(str))

    # Dense Linear / NN Features (StandardScaler + OneHotEncoder fitted strictly on train_idx)
    scaler = StandardScaler()
    num_and_te = num_features + neigh_new_cols + te_features
    X_tr_num_scaled = scaler.fit_transform(X_tr_tree[num_and_te].fillna(0))
    X_va_num_scaled = scaler.transform(X_va_tree[num_and_te].fillna(0))
    X_te_num_scaled = scaler.transform(X_te_tree[num_and_te].fillna(0))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    X_tr_cat_ohe = ohe.fit_transform(
        processed_train.iloc[train_idx][cat_features].astype(str)
    )
    X_va_cat_ohe = ohe.transform(
        processed_train.iloc[val_idx][cat_features].astype(str)
    )
    X_te_cat_ohe = ohe.transform(processed_test[cat_features].astype(str))

    X_tr_dense = np.hstack([X_tr_num_scaled, X_tr_cat_ohe]).astype(np.float32)
    X_va_dense = np.hstack([X_va_num_scaled, X_va_cat_ohe]).astype(np.float32)
    X_te_dense = np.hstack([X_te_num_scaled, X_te_cat_ohe]).astype(np.float32)

    # 1. LightGBM
    lgb_model = get_lgbm_regressor(seed=42 + fold)
    lgb_model.fit(
        X_tr_tree,
        y_tr_log,
        eval_set=[(X_va_tree, y_va_log)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    lgb_val_pred = lgb_model.predict(X_va_tree)
    lgb_test_pred = lgb_model.predict(X_te_tree)
    models_dict["lgbm"]["oof"][val_idx] = lgb_val_pred
    models_dict["lgbm"]["test"] += lgb_test_pred / n_splits

    # 2. XGBoost
    xgb_model = get_xgb_regressor(seed=42 + fold)
    xgb_model.fit(X_tr_tree, y_tr_log, eval_set=[(X_va_tree, y_va_log)], verbose=False)
    xgb_val_pred = xgb_model.predict(X_va_tree)
    xgb_test_pred = xgb_model.predict(X_te_tree)
    models_dict["xgb"]["oof"][val_idx] = xgb_val_pred
    models_dict["xgb"]["test"] += xgb_test_pred / n_splits

    # 3. ExtraTrees Regressor
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.65,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr_tree, y_tr_log)
    models_dict["extratrees"]["oof"][val_idx] = et_model.predict(X_va_tree)
    models_dict["extratrees"]["test"] += et_model.predict(X_te_tree) / n_splits

    # 4. CatBoost (if installed)
    if CATBOOST_AVAILABLE:
        cb_model = get_catboost_regressor(seed=42 + fold)
        cb_model.fit(
            X_tr_tree,
            y_tr_log,
            eval_set=(X_va_tree, y_va_log),
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_val_pred = cb_model.predict(X_va_tree)
        cb_test_pred = cb_model.predict(X_te_tree)
        models_dict["catboost"]["oof"][val_idx] = cb_val_pred
        models_dict["catboost"]["test"] += cb_test_pred / n_splits

    # 5. Linear Models (Ridge, ElasticNet, BayesianRidge) & Kernel SVR
    ridge = Ridge(alpha=12.0, random_state=42 + fold)
    ridge.fit(X_tr_dense, y_tr_log)
    models_dict["ridge"]["oof"][val_idx] = ridge.predict(X_va_dense)
    models_dict["ridge"]["test"] += ridge.predict(X_te_dense) / n_splits

    elastic = ElasticNet(alpha=0.002, l1_ratio=0.4, max_iter=3000, random_state=42 + fold)
    elastic.fit(X_tr_dense, y_tr_log)
    models_dict["elasticnet"]["oof"][val_idx] = elastic.predict(X_va_dense)
    models_dict["elasticnet"]["test"] += elastic.predict(X_te_dense) / n_splits

    bridge = BayesianRidge(max_iter=500)
    bridge.fit(X_tr_dense, y_tr_log)
    models_dict["bayesian_ridge"]["oof"][val_idx] = bridge.predict(X_va_dense)
    models_dict["bayesian_ridge"]["test"] += bridge.predict(X_te_dense) / n_splits

    svr = SVR(C=12.0, epsilon=0.04, kernel="rbf")
    svr.fit(X_tr_dense, y_tr_log)
    models_dict["svr"]["oof"][val_idx] = svr.predict(X_va_dense)
    models_dict["svr"]["test"] += svr.predict(X_te_dense) / n_splits

    # 5. Tabular Residual Neural Network
    nn_model, nn_criterion, nn_optimizer, nn_scheduler = create_neural_model_bundle(
        input_dim=X_tr_dense.shape[1],
        lr=2e-3,
        weight_decay=1e-3,
        max_epochs=90,
        device=device,
    )

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_dense, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    tr_loader = DataLoader(
        tr_dataset, batch_size=48, shuffle=True, drop_last=False, num_workers=2
    )

    va_tensor_x = torch.tensor(X_va_dense, dtype=torch.float32).to(device)
    va_tensor_y = torch.tensor(y_va_log, dtype=torch.float32).to(device)
    te_tensor_x = torch.tensor(X_te_dense, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(90):
        nn_model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            nn_optimizer.zero_grad()
            pred_y = nn_model(batch_x)
            loss = nn_criterion(pred_y, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.5)
            nn_optimizer.step()
        nn_scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_preds_epoch = nn_model(va_tensor_x)
            val_loss = nn_criterion(val_preds_epoch, va_tensor_y).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()
    with torch.no_grad():
        nn_val_pred = nn_model(va_tensor_x).cpu().numpy()
        nn_test_pred = nn_model(te_tensor_x).cpu().numpy()

    models_dict["nn"]["oof"][val_idx] = nn_val_pred
    models_dict["nn"]["test"] += nn_test_pred / n_splits

    fold_lgb_rmsle = compute_rmsle(y_va_raw, np.expm1(lgb_val_pred))
    fold_xgb_rmsle = compute_rmsle(y_va_raw, np.expm1(xgb_val_pred))
    fold_svr_rmsle = compute_rmsle(y_va_raw, np.expm1(models_dict["svr"]["oof"][val_idx]))
    fold_nn_rmsle = compute_rmsle(y_va_raw, np.expm1(nn_val_pred))
    print(
        f"Fold {fold+1}/{n_splits} | LGBM RMSLE: {fold_lgb_rmsle:.5f} | XGB RMSLE: {fold_xgb_rmsle:.5f} | SVR RMSLE: {fold_svr_rmsle:.5f} | NN RMSLE: {fold_nn_rmsle:.5f}"
    )

# =========================================================================
# 6. SLSQP Blending Optimization & Validation Evaluation
# =========================================================================
model_names = list(models_dict.keys())
oof_matrix = np.column_stack([models_dict[name]["oof"] for name in model_names])
test_matrix = np.column_stack([models_dict[name]["test"] for name in model_names])

for name in model_names:
    score = compute_rmsle(y_raw, np.expm1(models_dict[name]["oof"]))
    print(f"Model [{name.upper():<10}] OOF RMSLE: {score:.5f}")


def blend_objective(weights: np.ndarray) -> float:
    weights_norm = weights / (np.sum(weights) + 1e-8)
    blend_log = np.dot(oof_matrix, weights_norm)
    loss = compute_rmsle(y_raw, np.expm1(blend_log))
    reg = 1e-5 * np.sum(weights_norm ** 2)
    return loss + reg


init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0.0, 1.0) for _ in range(len(model_names))]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-7},
)
best_weights = opt_res.x / np.sum(opt_res.x)

oof_ensemble_log = np.dot(oof_matrix, best_weights)
oof_ensemble_raw = np.expm1(oof_ensemble_log)
final_val_rmsle = compute_rmsle(y_raw, oof_ensemble_raw)

# =========================================================================
# 7. Test Inference & Submission Generation
# =========================================================================
final_test_pred_log = np.dot(test_matrix, best_weights)
final_test_pred_raw = np.expm1(final_test_pred_log)
final_test_pred_raw = np.clip(final_test_pred_raw, 10000.0, None)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_pred_raw), dtype=int),
        "SalePrice": final_test_pred_raw,
    }
)

sub_path = SUBMISSION_DIR / "submission.csv"
submission_df.to_csv(sub_path, index=False)

print(f"Submission generated successfully at: {sub_path}")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_rmsle:.6f}")