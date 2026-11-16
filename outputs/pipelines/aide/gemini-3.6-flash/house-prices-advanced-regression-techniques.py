import os
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Remove known extreme outliers in training set
if "GrLivArea" in train_df.columns and "SalePrice" in train_df.columns:
    train_df = train_df[
        ~((train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000))
    ].reset_index(drop=True)

y_train = np.log1p(train_df["SalePrice"].values)
train_df = train_df.drop(columns=["SalePrice"], errors="ignore")

n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)


# 2. Comprehensive Feature Engineering
def engineer_features(df_in):
    df = df_in.copy()

    # Ordinal mappings
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0}
    exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0}
    fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "NA": 0}
    shape_map = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
    util_map = {"AllPub": 4, "NoSewr": 3, "NoSeWa": 2, "Elo": 1}
    slope_map = {"Gtl": 3, "Mod": 2, "Sev": 1}
    air_map = {"Y": 1, "N": 0}
    paved_map = {"Y": 3, "P": 2, "N": 1}
    func_map = {
        "Typ": 8,
        "Min1": 7,
        "Min2": 6,
        "Mod": 5,
        "Maj1": 4,
        "Maj2": 3,
        "Sev": 2,
        "Sal": 1,
    }

    ordinal_cols = [
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
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].fillna("NA").map(qual_map).fillna(0)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].fillna("NA").map(exposure_map).fillna(0)

    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col] = df[col].fillna("NA").map(bsmt_fin_map).fillna(0)

    if "Fence" in df.columns:
        df["Fence"] = df["Fence"].fillna("NA").map(fence_map).fillna(0)

    if "LotShape" in df.columns:
        df["LotShape"] = df["LotShape"].map(shape_map).fillna(0)
    if "Utilities" in df.columns:
        df["Utilities"] = df["Utilities"].map(util_map).fillna(0)
    if "LandSlope" in df.columns:
        df["LandSlope"] = df["LandSlope"].map(slope_map).fillna(0)
    if "CentralAir" in df.columns:
        df["CentralAir"] = df["CentralAir"].map(air_map).fillna(0)
    if "PavedDrive" in df.columns:
        df["PavedDrive"] = df["PavedDrive"].map(paved_map).fillna(0)
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].map(func_map).fillna(0)

    # Impute missing numerical values
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(
            df["YearBuilt"] if "YearBuilt" in df.columns else 1970
        )
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # Spatial and Aggregate Living Area Features
    gr_sf = df["GrLivArea"] if "GrLivArea" in df.columns else 0
    bsmt_sf = df["TotalBsmtSF"] if "TotalBsmtSF" in df.columns else 0
    flr1_sf = df["1stFlrSF"] if "1stFlrSF" in df.columns else 0
    flr2_sf = df["2ndFlrSF"] if "2ndFlrSF" in df.columns else 0

    df["TotalSF"] = gr_sf + bsmt_sf
    df["TotalLivingSF"] = flr1_sf + flr2_sf + bsmt_sf

    # Bathroom Aggregations
    full_bath = df["FullBath"] if "FullBath" in df.columns else 0
    half_bath = df["HalfBath"] if "HalfBath" in df.columns else 0
    bsmt_full = df["BsmtFullBath"] if "BsmtFullBath" in df.columns else 0
    bsmt_half = df["BsmtHalfBath"] if "BsmtHalfBath" in df.columns else 0
    df["TotalBath"] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

    # Porch Aggregations
    open_p = df["OpenPorchSF"] if "OpenPorchSF" in df.columns else 0
    enc_p = df["EnclosedPorch"] if "EnclosedPorch" in df.columns else 0
    three_p = df["3SsnPorch"] if "3SsnPorch" in df.columns else 0
    screen_p = df["ScreenPorch"] if "ScreenPorch" in df.columns else 0
    wood_d = df["WoodDeckSF"] if "WoodDeckSF" in df.columns else 0
    df["TotalPorchSF"] = open_p + enc_p + three_p + screen_p + wood_d

    # Quality and Condition Interactions
    qual = df["OverallQual"] if "OverallQual" in df.columns else 5
    cond = df["OverallCond"] if "OverallCond" in df.columns else 5
    df["QualCond_Prod"] = qual * cond
    df["Qual_Sq"] = qual**2
    df["Qual_per_Room"] = (
        qual / (df["TotRmsAbvGrd"] + 1e-5) if "TotRmsAbvGrd" in df.columns else qual
    )

    # Age & Remodel Features
    yr_sold = df["YrSold"] if "YrSold" in df.columns else 2010
    yr_built = df["YearBuilt"] if "YearBuilt" in df.columns else 1970
    yr_remod = df["YearRemodAdd"] if "YearRemodAdd" in df.columns else yr_built

    df["HouseAge"] = (yr_sold - yr_built).clip(lower=0)
    df["RemodAge"] = (yr_sold - yr_remod).clip(lower=0)
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["IsNew"] = (df["YearBuilt"] == df["YrSold"]).astype(int)
    df["Qual_Age_Ratio"] = qual / (df["HouseAge"] + 1.0)

    # Garage Area Features
    gar_built = df["GarageYrBlt"] if "GarageYrBlt" in df.columns else yr_built
    df["GarageAge"] = np.where(gar_built > 0, (yr_sold - gar_built).clip(lower=0), 0)
    gar_cars = df["GarageCars"] if "GarageCars" in df.columns else 0
    gar_area = df["GarageArea"] if "GarageArea" in df.columns else 0
    df["GarageArea_per_Car"] = gar_area / (gar_cars + 1e-5)

    # Binary Feature Indicators
    df["HasPool"] = (df["PoolArea"] > 0).astype(int) if "PoolArea" in df.columns else 0
    df["Has2ndFloor"] = (
        (df["2ndFlrSF"] > 0).astype(int) if "2ndFlrSF" in df.columns else 0
    )
    df["HasGarage"] = (gar_area > 0).astype(int)
    df["HasBsmt"] = (bsmt_sf > 0).astype(int)
    df["HasFireplace"] = (
        (df["Fireplaces"] > 0).astype(int) if "Fireplaces" in df.columns else 0
    )

    # Additional Structural Ratios
    df["TotalSF_per_Room"] = df["TotalSF"] / (
        df["TotRmsAbvGrd"] + 1e-5 if "TotRmsAbvGrd" in df.columns else 1
    )
    df["LivArea_Ratio"] = gr_sf / (df["TotalSF"] + 1e-5)
    df["TotalSF_Qual"] = df["TotalSF"] * qual
    df["GrLivArea_Qual"] = gr_sf * qual

    # Group-based Multi-granularity Standardization (Neighborhood & MSSubClass)
    group_cols = [
        "TotalSF",
        "GrLivArea",
        "OverallQual",
        "YearBuilt",
        "GarageArea",
        "TotalBath",
    ]
    for grp_col in ["Neighborhood", "MSSubClass"]:
        if grp_col in df.columns:
            for target_c in group_cols:
                if target_c in df.columns:
                    grp_mean = df.groupby(grp_col)[target_c].transform("mean")
                    grp_std = df.groupby(grp_col)[target_c].transform("std").fillna(1.0)
                    grp_median = df.groupby(grp_col)[target_c].transform("median")

                    df[f"{target_c}_{grp_col}_zscore"] = (df[target_c] - grp_mean) / (
                        grp_std + 1e-5
                    )
                    df[f"{target_c}_{grp_col}_ratio"] = df[target_c] / (
                        grp_median + 1e-5
                    )

    # Log-transforms for skewed numerical features
    skewed_cols = [
        "LotArea",
        "GrLivArea",
        "TotalSF",
        "TotalLivingSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "LotFrontage",
        "MasVnrArea",
    ]
    for c in skewed_cols:
        if c in df.columns:
            df[c + "_log"] = np.log1p(df[c].clip(lower=0))

    return df


df_engineered = engineer_features(combined)

# Separate categorical and numerical columns
cat_cols = df_engineered.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

# Preprocess categorical features using Frequency Encoding for tree models
for col in cat_cols:
    freq = df_engineered[col].value_counts()
    df_engineered[col + "_freq"] = df_engineered[col].map(freq)
    df_engineered[col] = df_engineered[col].astype("category")

num_cols_updated = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

X_all = df_engineered.iloc[:n_train].copy()
X_test_all = df_engineered.iloc[n_train:].copy()

# Preprocess linear models (standard scale numericals + one-hot encode categoricals)
preprocessor_linear = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

X_train_lin = preprocessor_linear.fit_transform(X_all)
X_test_lin = preprocessor_linear.transform(X_test_all)

# Tree feature matrix with numeric types (fillna for missing values)
X_train_tree = X_all[num_cols_updated].fillna(0)
X_test_tree = X_test_all[num_cols_updated].fillna(0)

# 3. Model Definition and Cross-Validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

models_tree = {
    "lgb": lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        num_leaves=20,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=42,
        verbose=-1,
    ),
    "xgb": xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=42,
        verbosity=0,
    ),
    "hgb": HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.02,
        max_depth=5,
        l2_regularization=1.0,
        random_state=42,
    ),
    "gbr": GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.7,
        max_features=0.6,
        random_state=42,
    ),
    "et": ExtraTreesRegressor(
        n_estimators=300, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
    ),
}

if HAS_CATBOOST:
    models_tree["cb"] = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
    )

models_lin = {
    "ridge": Ridge(alpha=15.0, random_state=42),
    "elastic": ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=2000, random_state=42),
}

model_names = list(models_tree.keys()) + list(models_lin.keys())
oof_predictions = {m: np.zeros(n_train) for m in model_names}
test_predictions = {m: np.zeros(len(X_test_all)) for m in model_names}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tree, y_train)):
    y_tr, y_va = y_train[train_idx], y_train[val_idx]

    # Tree Models Training
    for name, model in models_tree.items():
        X_tr = X_train_tree.iloc[train_idx]
        X_va = X_train_tree.iloc[val_idx]

        if name == "lgb":
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif name == "xgb":
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        oof_predictions[name][val_idx] = val_pred
        test_predictions[name] += model.predict(X_test_tree) / n_splits

    # Linear Models Training
    for name, model in models_lin.items():
        X_tr_lin, X_va_lin = X_train_lin[train_idx], X_train_lin[val_idx]
        model.fit(X_tr_lin, y_tr)

        val_pred = model.predict(X_va_lin)
        oof_predictions[name][val_idx] = val_pred
        test_predictions[name] += model.predict(X_test_lin) / n_splits

# 4. Ensembling via Constrained Optimization (SLSQP)
oof_matrix = np.column_stack([oof_predictions[m] for m in model_names])
test_matrix = np.column_stack([test_predictions[m] for m in model_names])


def loss_func(weights):
    weights = np.array(weights)
    blend_pred = np.dot(oof_matrix, weights)
    return np.sqrt(mean_squared_error(y_train, blend_pred))


init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1) for _ in model_names]
constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
opt_weights = res.x

oof_blend = np.dot(oof_matrix, opt_weights)
cv_rmsle = np.sqrt(mean_squared_error(y_train, oof_blend))

# Print cross-validation performance
print(f"Validation RMSLE: {cv_rmsle:.5f}")

# 5. Generate Submission File
test_blend_log = np.dot(test_matrix, opt_weights)
final_test_preds = np.expm1(test_blend_log)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "SalePrice": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
