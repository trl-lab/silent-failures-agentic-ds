import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb


# =========================================================================
# 0. Reproducibility & Directory Setup
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Domain Feature Engineering
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Filter out 2 known extreme Ames training outliers (GrLivArea > 4000 with SalePrice < 300,000)
if "GrLivArea" in train_df.columns and "SalePrice" in train_df.columns:
    outlier_mask = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)
    train_df = train_df[~outlier_mask].reset_index(drop=True)

target = train_df["SalePrice"].values
target_log = np.log1p(target)

# Compute neighborhood price aggregations strictly from training partition
train_pps = train_df["SalePrice"] / (train_df["GrLivArea"] + 1e-5)
neigh_pps_median = train_pps.groupby(train_df["Neighborhood"]).median()
global_pps_median = train_pps.median()

neigh_price_median = train_df.groupby("Neighborhood")["SalePrice"].median()
global_price_median = train_df["SalePrice"].median()

train_features = (
    train_df.drop(columns=["SalePrice"])
    if "SalePrice" in train_df.columns
    else train_df.copy()
)
test_features = test_df.copy()

train_features["_is_train"] = 1
test_features["_is_train"] = 0

combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# Map neighborhood price aggregations
combined["Neigh_Price_Per_SqFt"] = (
    combined["Neighborhood"].map(neigh_pps_median).fillna(global_pps_median)
)
combined["Neigh_Median_Price"] = (
    combined["Neighborhood"].map(neigh_price_median).fillna(global_price_median)
)

# Domain Imputation: Categorical features where NA indicates absence
none_cols = [
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
for col in none_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna("None").astype(str)

# Numerical features where NA indicates zero physical dimension
zero_cols = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]
for col in zero_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna(0.0)

# Impute LotFrontage by Neighborhood median strictly from training partition
lot_frontage_medians = (
    combined[combined["_is_train"] == 1].groupby("Neighborhood")["LotFrontage"].median()
)
global_lot_median = combined[combined["_is_train"] == 1]["LotFrontage"].median()
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(lot_frontage_medians.get(x.name, global_lot_median))
)
combined["LotFrontage"] = combined["LotFrontage"].fillna(global_lot_median)

# Ordinal Categorical Encodings
qual_dict = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for col in [
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
]:
    if col in combined.columns:
        combined[f"{col}_Num"] = combined[col].map(qual_dict).fillna(0).astype(float)

bsmt_expo_dict = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
if "BsmtExposure" in combined.columns:
    combined["BsmtExposure_Num"] = (
        combined["BsmtExposure"].map(bsmt_expo_dict).fillna(0).astype(float)
    )

bsmt_fin_dict = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
for col in ["BsmtFinType1", "BsmtFinType2"]:
    if col in combined.columns:
        combined[f"{col}_Num"] = (
            combined[col].map(bsmt_fin_dict).fillna(0).astype(float)
        )

functional_dict = {
    "None": 0,
    "Sal": 1,
    "Sev": 2,
    "Maj2": 3,
    "Maj1": 4,
    "Mod": 5,
    "Min2": 6,
    "Min1": 7,
    "Typ": 8,
}
if "Functional" in combined.columns:
    combined["Functional_Num"] = (
        combined["Functional"].map(functional_dict).fillna(0).astype(float)
    )

garage_finish_dict = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
if "GarageFinish" in combined.columns:
    combined["GarageFinish_Num"] = (
        combined["GarageFinish"].map(garage_finish_dict).fillna(0).astype(float)
    )

paved_drive_dict = {"N": 0, "P": 1, "Y": 2}
if "PavedDrive" in combined.columns:
    combined["PavedDrive_Num"] = (
        combined["PavedDrive"].map(paved_drive_dict).fillna(0).astype(float)
    )

# Domain Feature Construction
combined["TotalSF"] = (
    combined["TotalBsmtSF"] + combined["1stFlrSF"] + combined["2ndFlrSF"]
)
combined["TotalLivingSF"] = combined["GrLivArea"] + combined["TotalBsmtSF"]
combined["TotalPorchSF"] = (
    combined["OpenPorchSF"]
    + combined["EnclosedPorch"]
    + combined["3SsnPorch"]
    + combined["ScreenPorch"]
    + combined["WoodDeckSF"]
)
combined["TotalBathrooms"] = (
    combined["FullBath"]
    + 0.5 * combined["HalfBath"]
    + combined["BsmtFullBath"]
    + 0.5 * combined["BsmtHalfBath"]
)
combined["TotalBathroomsAbvGrd"] = combined["FullBath"] + 0.5 * combined["HalfBath"]

combined["HouseAge"] = (combined["YrSold"] - combined["YearBuilt"]).clip(lower=0)
combined["RemodAge"] = (combined["YrSold"] - combined["YearRemodAdd"]).clip(lower=0)
combined["GarageAge"] = np.where(
    combined["GarageYrBlt"] > 0,
    (combined["YrSold"] - combined["GarageYrBlt"]).clip(lower=0),
    0,
)
combined["IsRemodeled"] = (combined["YearRemodAdd"] != combined["YearBuilt"]).astype(
    int
)
combined["IsNewHouse"] = (combined["YrSold"] == combined["YearBuilt"]).astype(int)

def get_col_val(df, candidates, default=0.0):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return default

combined["OverallGrade"] = combined["OverallQual"] * combined["OverallCond"]
combined["Qual_TotalSF"] = combined["OverallQual"] * combined["TotalSF"]
combined["Qual_GrLivArea"] = combined["OverallQual"] * combined["GrLivArea"]
combined["ExterQualCond"] = combined["ExterQual_Num"] * combined["ExterCond_Num"]
combined["KitchenScore"] = combined["KitchenQual_Num"] * get_col_val(combined, ["KitchenAbvGr", "KitchenAbvGrd", "Kitchen"])
combined["GarageScore"] = combined["GarageQual_Num"] * get_col_val(combined, ["GarageCars"])
combined["FireplaceScore"] = combined["FireplaceQu_Num"] * get_col_val(combined, ["Fireplaces"])
combined["PoolScore"] = combined["PoolQC_Num"] * get_col_val(combined, ["PoolArea"])

combined["HasBasement"] = (combined["TotalBsmtSF"] > 0).astype(int)
combined["HasGarage"] = (combined["GarageArea"] > 0).astype(int)
combined["Has2ndFloor"] = (combined["2ndFlrSF"] > 0).astype(int)
combined["HasFireplace"] = (combined["Fireplaces"] > 0).astype(int)
combined["HasPool"] = (combined["PoolArea"] > 0).astype(int)
combined["HasPorch"] = (combined["TotalPorchSF"] > 0).astype(int)
combined["HasWoodDeck"] = (combined["WoodDeckSF"] > 0).astype(int)

combined["GrLivArea_per_Room"] = combined["GrLivArea"] / (
    get_col_val(combined, ["TotRmsAbvGrd", "TotRmsAbvGr", "TotRoomsAbvGrd"], default=1.0) + 1e-5
)
combined["LotArea_per_GrLivArea"] = combined["LotArea"] / (combined["GrLivArea"] + 1e-5)
combined["BsmtFinRatio"] = combined["BsmtFinSF1"] / (combined["TotalBsmtSF"] + 1e-5)

# Non-linear interaction terms and composite features
combined["TotalSF_Sq"] = combined["TotalSF"] ** 2
combined["OverallQual_Sq"] = combined["OverallQual"] ** 2
combined["RemodAge_x_OverallQual"] = combined["RemodAge"] * combined["OverallQual"]
combined["TotalSF_x_OverallGrade"] = combined["TotalSF"] * combined["OverallGrade"]
combined["GrLivArea_x_OverallQual"] = combined["GrLivArea"] * combined["OverallQual"]

# Convert discrete identifier columns to string categorical features
for col in ["MSSubClass", "MoSold", "YrSold"]:
    if col in combined.columns:
        combined[col] = combined[col].astype(str)

# Remaining categorical imputations (train mode)
cat_cols = combined.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    if col in combined.columns and combined[col].isnull().sum() > 0:
        mode_val = combined[combined["_is_train"] == 1][col].mode()[0]
        combined[col] = combined[col].fillna(mode_val)

# Remaining numeric imputations (train median)
num_cols = combined.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if col not in ["_is_train", "Id", "id"] and combined[col].isnull().sum() > 0:
        median_val = combined[combined["_is_train"] == 1][col].median()
        combined[col] = combined[col].fillna(median_val)

# Apply log1p transformation to highly skewed continuous numerical features (skew > 0.75)
numeric_cols_for_skew = combined.select_dtypes(include=[np.number]).columns.tolist()
exclude_skew = ["_is_train", "Id", "id", "fold", "SalePrice", "SalePrice_Log"]
candidate_skew_cols = [
    c
    for c in numeric_cols_for_skew
    if c not in exclude_skew
    and not c.startswith("Has")
    and not c.startswith("Is")
    and not c.endswith("_Num")
]

train_mask = combined["_is_train"] == 1
skewness_series = combined.loc[train_mask, candidate_skew_cols].apply(
    lambda x: skew(x.dropna())
)
high_skew_cols = skewness_series[skewness_series > 0.75].index.tolist()

for col in high_skew_cols:
    combined[col] = np.log1p(np.maximum(0, combined[col]))

nominal_cols = combined.select_dtypes(include=["object"]).columns.tolist()
for col in nominal_cols:
    combined[col] = combined[col].astype(str).astype("category")

train_processed = (
    combined[combined["_is_train"] == 1].drop(columns=["_is_train"]).copy()
)
test_processed = combined[combined["_is_train"] == 0].drop(columns=["_is_train"]).copy()

train_processed["SalePrice"] = target
train_processed["SalePrice_Log"] = target_log

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(train_processed), dtype=int)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed, target)):
    fold_assignments[val_idx] = fold

train_processed["fold"] = fold_assignments

train_processed.to_parquet("./working/train_processed.parquet", index=False)
test_processed.to_parquet("./working/test_processed.parquet", index=False)


# =========================================================================
# 2. Model Architectures & Builders
# =========================================================================
def build_lgb_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 15,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMRegressor(**params)


def build_xgb_regressor(random_state: int = 42) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": 3000,
        "learning_rate": 0.015,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "random_state": random_state,
        "n_jobs": -1,
    }
    return xgb.XGBRegressor(**params)


def build_et_regressor(random_state: int = 42) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=random_state,
        n_jobs=-1,
    )


def build_ridge_regressor() -> Pipeline:
    alphas = np.logspace(-2, 3, 100)
    return Pipeline(
        [
            ("variance_thresh", VarianceThreshold(threshold=1e-4)),
            ("scaler", RobustScaler()),
            ("model", RidgeCV(alphas=alphas, cv=5)),
        ]
    )


# =========================================================================
# 3. Training & Validation Execution
# =========================================================================
y_true = train_processed["SalePrice"].values
y_true_log = train_processed["SalePrice_Log"].values
folds = train_processed["fold"].values
n_folds = len(np.unique(folds))

ignore_cols = ["Id", "id", "SalePrice", "SalePrice_Log", "fold", "_is_train"]
feature_cols = [c for c in train_processed.columns if c not in ignore_cols]

X_tree = train_processed[feature_cols].copy()
X_test_tree = test_processed[feature_cols].copy()

# One-hot encoded feature matrices for dense linear and extra-trees models
combined_feats = pd.concat([X_tree, X_test_tree], axis=0, ignore_index=True)
combined_dense = pd.get_dummies(combined_feats, drop_first=True, dtype=np.float32)

X_dense = combined_dense.iloc[: len(train_processed)].values.astype(np.float32)
X_test_dense = combined_dense.iloc[len(train_processed) :].values.astype(np.float32)

X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
X_test_dense = np.nan_to_num(X_test_dense, nan=0.0, posinf=0.0, neginf=0.0)

oof_lgb = np.zeros(len(train_processed))
oof_xgb = np.zeros(len(train_processed))
oof_et = np.zeros(len(train_processed))
oof_ridge = np.zeros(len(train_processed))

test_preds_lgb = np.zeros(len(test_processed))
test_preds_xgb = np.zeros(len(test_processed))
test_preds_et = np.zeros(len(test_processed))
test_preds_ridge = np.zeros(len(test_processed))

for fold in range(n_folds):
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    y_tr_log = y_true_log[train_idx]
    y_va_log = y_true_log[val_idx]

    # Model A: LightGBM Regressor
    lgb_model = build_lgb_regressor(random_state=42 + fold)
    lgb_model.fit(
        X_tree.iloc[train_idx],
        y_tr_log,
        eval_set=[(X_tree.iloc[val_idx], y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_tree.iloc[val_idx])
    test_preds_lgb += lgb_model.predict(X_test_tree) / n_folds

    # Model B: XGBoost Regressor
    xgb_model = build_xgb_regressor(random_state=42 + fold)
    xgb_model.fit(
        X_tree.iloc[train_idx],
        y_tr_log,
        eval_set=[(X_tree.iloc[val_idx], y_va_log)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_tree.iloc[val_idx])
    test_preds_xgb += xgb_model.predict(X_test_tree) / n_folds

    # Model C: ExtraTrees Regressor
    et_model = build_et_regressor(random_state=42 + fold)
    et_model.fit(X_dense[train_idx], y_tr_log)
    oof_et[val_idx] = et_model.predict(X_dense[val_idx])
    test_preds_et += et_model.predict(X_test_dense) / n_folds

    # Model D: RidgeCV Regressor
    ridge_model = build_ridge_regressor()
    ridge_model.fit(X_dense[train_idx], y_tr_log)
    oof_ridge[val_idx] = ridge_model.predict(X_dense[val_idx])
    test_preds_ridge += ridge_model.predict(X_test_dense) / n_folds

# =========================================================================
# 4. Non-Negative Blending Optimization
# =========================================================================
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_et, oof_ridge])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_et, test_preds_ridge]
)


def blend_objective(weights):
    w = np.maximum(0, weights)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 1e10
    w = w / w_sum
    blended_log = oof_matrix @ w
    blended_price = np.expm1(blended_log)
    return root_mean_squared_error(y_true, blended_price)


initial_weights = np.ones(4) / 4.0
bounds = [(0, 1) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-9},
)

optimal_weights = opt_res.x / np.sum(opt_res.x)

# =========================================================================
# 5. Validation Scoring & Submission Generation
# =========================================================================
final_oof_log = oof_matrix @ optimal_weights
final_oof_preds = np.expm1(final_oof_log)
final_cv_rmse = root_mean_squared_error(y_true, final_oof_preds)

final_test_log = test_matrix @ optimal_weights
final_test_preds = np.expm1(final_test_log)
final_test_preds = np.clip(final_test_preds, a_min=10000.0, a_max=None)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "SalePrice": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created."
assert len(submission_df) == len(
    test_df
), f"Expected {len(test_df)} rows, found {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    "SalePrice",
], f"Incorrect columns: {submission_df.columns}"
assert (
    submission_df["SalePrice"].isnull().sum() == 0
), "Submission contains missing values."

print(f"Final Validation Score: {final_cv_rmse}")
