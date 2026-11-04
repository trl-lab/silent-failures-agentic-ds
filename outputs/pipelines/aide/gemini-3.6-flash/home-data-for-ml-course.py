import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Remove extreme area-price outliers
if "SalePrice" in train_df.columns:
    train_df = train_df[
        ~((train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000))
    ].reset_index(drop=True)

y_train = train_df["SalePrice"].values
y_train_log = np.log1p(y_train)
train_df = train_df.drop(columns=["SalePrice"])

# Combine train and test for consistent preprocessing
df_all = pd.concat([train_df, test_df], ignore_index=True)

# Categorical and numerical missing value handling
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
]
for c in none_cols:
    if c in df_all.columns:
        df_all[c] = df_all[c].fillna("None")

zero_cols = [
    "GarageYrBlt",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]
for c in zero_cols:
    if c in df_all.columns:
        df_all[c] = df_all[c].fillna(0)

if "LotFrontage" in df_all.columns:
    df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    df_all["LotFrontage"] = df_all["LotFrontage"].fillna(df_all["LotFrontage"].median())

# Ordinal quality mappings
qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
bsmt_fin_map = {
    "GLQ": 6,
    "ALQ": 5,
    "BLQ": 4,
    "Rec": 3,
    "LwQ": 2,
    "Unf": 1,
    "None": 0,
}
func_map = {
    "Typ": 7,
    "Min1": 6,
    "Min2": 5,
    "Mod": 4,
    "Maj1": 3,
    "Maj2": 2,
    "Sev": 1,
    "Sal": 0,
}
garage_fin_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
paved_map = {"Y": 3, "P": 2, "N": 1}

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
    if col in df_all.columns:
        df_all[col + "_num"] = df_all[col].map(qual_map).fillna(0)

if "BsmtExposure" in df_all.columns:
    df_all["BsmtExposure_num"] = df_all["BsmtExposure"].map(exposure_map).fillna(0)
if "BsmtFinType1" in df_all.columns:
    df_all["BsmtFinType1_num"] = df_all["BsmtFinType1"].map(bsmt_fin_map).fillna(0)
if "BsmtFinType2" in df_all.columns:
    df_all["BsmtFinType2_num"] = df_all["BsmtFinType2"].map(bsmt_fin_map).fillna(0)
if "Functional" in df_all.columns:
    df_all["Functional_num"] = df_all["Functional"].map(func_map).fillna(0)
if "GarageFinish" in df_all.columns:
    df_all["GarageFinish_num"] = df_all["GarageFinish"].map(garage_fin_map).fillna(0)
if "PavedDrive" in df_all.columns:
    df_all["PavedDrive_num"] = df_all["PavedDrive"].map(paved_map).fillna(0)

# Composite Quality Index
df_all["TotalQualIndex"] = (
    df_all["OverallQual"]
    + df_all["OverallCond"]
    + df_all.get("ExterQual_num", 0)
    + df_all.get("BsmtQual_num", 0)
    + df_all.get("KitchenQual_num", 0)
    + df_all.get("GarageQual_num", 0)
    + df_all.get("FireplaceQu_num", 0)
    + df_all.get("HeatingQC_num", 0)
)

# Area and Bath Aggregations
df_all["TotalSF"] = (
    df_all["TotalBsmtSF"].fillna(0)
    + df_all["1stFlrSF"].fillna(0)
    + df_all["2ndFlrSF"].fillna(0)
)
df_all["Total_Bath"] = (
    df_all["FullBath"].fillna(0)
    + 0.5 * df_all["HalfBath"].fillna(0)
    + df_all["BsmtFullBath"].fillna(0)
    + 0.5 * df_all["BsmtHalfBath"].fillna(0)
)
df_all["Total_Porch"] = (
    df_all["OpenPorchSF"].fillna(0)
    + df_all["3SsnPorch"].fillna(0)
    + df_all["EnclosedPorch"].fillna(0)
    + df_all["ScreenPorch"].fillna(0)
    + df_all["WoodDeckSF"].fillna(0)
)

# Temporal & Interaction features
df_all["HouseAge"] = df_all["YrSold"] - df_all["YearBuilt"]
df_all["RemodAge"] = df_all["YrSold"] - df_all["YearRemodAdd"]
df_all["IsRemodeled"] = (df_all["YearRemodAdd"] != df_all["YearBuilt"]).astype(int)
df_all["IsNew"] = (df_all["YrSold"] == df_all["YearBuilt"]).astype(int)
df_all["GarageAge"] = np.where(
    df_all["GarageYrBlt"] > 0, df_all["YrSold"] - df_all["GarageYrBlt"], 0
)

df_all["TotalQual_TotalSF"] = df_all["TotalQualIndex"] * df_all["TotalSF"]
df_all["OverallQual_TotalSF"] = df_all["OverallQual"] * df_all["TotalSF"]
df_all["OverallQual_GrLivArea"] = df_all["OverallQual"] * df_all["GrLivArea"]
df_all["SF_Per_Room"] = df_all["TotalSF"] / (df_all["TotRmsAbvGrd"] + 1)
df_all["SF_Per_Qual"] = df_all["TotalSF"] / (df_all["TotalQualIndex"] + 1)

# Group ratios
neigh_grliv = df_all.groupby("Neighborhood")["GrLivArea"].transform("mean")
df_all["Neigh_GrLivArea_Ratio"] = df_all["GrLivArea"] / (neigh_grliv + 1e-5)
neigh_lot = df_all.groupby("Neighborhood")["LotArea"].transform("mean")
df_all["Neigh_LotArea_Ratio"] = df_all["LotArea"] / (neigh_lot + 1e-5)

# Binary flags
df_all["HasPool"] = (df_all["PoolArea"] > 0).astype(int)
df_all["HasGarage"] = (df_all["GarageArea"] > 0).astype(int)
df_all["HasBsmt"] = (df_all["TotalBsmtSF"] > 0).astype(int)
df_all["HasFireplace"] = (df_all["Fireplaces"] > 0).astype(int)

# Skewness transformation for continuous features
num_cols = df_all.select_dtypes(include=[np.number]).columns
skewed_cols = [c for c in num_cols if c not in ["YrSold", "YearBuilt", "YearRemodAdd"]]
for col in skewed_cols:
    skew_val = df_all[col].skew()
    if skew_val > 0.75 and (df_all[col] >= 0).all():
        df_all[col] = np.log1p(df_all[col])

# Fill categoricals as string
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    df_all[col] = df_all[col].fillna("Missing").astype(str)

# Create two feature sets:
# 1) Tree models: label encoding
# 2) Linear models: One-Hot Encoding
df_all_tree = df_all.copy()
for col in cat_cols:
    df_all_tree[col] = df_all_tree[col].astype("category").cat.codes

df_all_linear = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)

# Split back to train and test
X_train_tree = df_all_tree.iloc[: len(train_df)].copy()
X_test_tree = df_all_tree.iloc[len(train_df) :].copy()

X_train_linear = df_all_linear.iloc[: len(train_df)].copy()
X_test_linear = df_all_linear.iloc[len(train_df) :].copy()

# Target Encoding inside CV
te_cols = [
    "Neighborhood",
    "MSSubClass",
    "Exterior1st",
    "Exterior2nd",
    "Condition1",
    "Zoning",
    "SaleCondition",
    "HouseStyle",
]
te_cols = [c for c in te_cols if c in df_all.columns]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in te_cols:
    X_train_tree[col + "_TE"] = np.nan
    X_test_tree[col + "_TE"] = np.nan
    X_train_linear[col + "_TE"] = np.nan
    X_test_linear[col + "_TE"] = np.nan
    global_mean = y_train_log.mean()
    m = 10.0

    for train_idx, val_idx in kf.split(X_train_tree):
        tr_cat = df_all.iloc[: len(train_df)].iloc[train_idx][col]
        tr_target = y_train_log[train_idx]
        val_cat = df_all.iloc[: len(train_df)].iloc[val_idx][col]

        stats = (
            pd.DataFrame({col: tr_cat, "target": tr_target})
            .groupby(col)["target"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + m * global_mean) / (
            stats["count"] + m
        )

        te_val = val_cat.map(smooth).fillna(global_mean).values
        X_train_tree.loc[X_train_tree.index[val_idx], col + "_TE"] = te_val
        X_train_linear.loc[X_train_linear.index[val_idx], col + "_TE"] = te_val

    stats_full = (
        pd.DataFrame({col: df_all.iloc[: len(train_df)][col], "target": y_train_log})
        .groupby(col)["target"]
        .agg(["count", "mean"])
    )
    smooth_full = (stats_full["count"] * stats_full["mean"] + m * global_mean) / (
        stats_full["count"] + m
    )
    te_test = (
        df_all.iloc[len(train_df) :][col].map(smooth_full).fillna(global_mean).values
    )
    X_test_tree[col + "_TE"] = te_test
    X_test_linear[col + "_TE"] = te_test

# Missing value final imputation
X_train_tree_clean = X_train_tree.fillna(0)
X_test_tree_clean = X_test_tree.fillna(0)

X_train_lin_clean = X_train_linear.fillna(0)
X_test_lin_clean = X_test_linear.fillna(0)

# Model initialization & setup
models = ["CatBoost", "LightGBM", "XGBoost", "Ridge", "Huber"]
oof_preds = {m: np.zeros(len(train_df)) for m in models}
test_preds = {m: np.zeros(len(test_df)) for m in models}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tree_clean)):
    # Tree features split
    X_tr_t, y_tr = X_train_tree_clean.iloc[train_idx], y_train_log[train_idx]
    X_val_t, y_val = X_train_tree_clean.iloc[val_idx], y_train_log[val_idx]
    X_te_t = X_test_tree_clean.copy()

    # Linear features split
    X_tr_l = X_train_lin_clean.iloc[train_idx]
    X_val_l = X_train_lin_clean.iloc[val_idx]
    X_te_l = X_test_lin_clean.copy()

    # 1. CatBoost Regressor
    model_cb = cb.CatBoostRegressor(
        iterations=3500,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=4,
        subsample=0.8,
        loss_function="RMSE",
        verbose=0,
        random_seed=42 + fold,
    )
    model_cb.fit(
        X_tr_t,
        y_tr,
        eval_set=(X_val_t, y_val),
        early_stopping_rounds=100,
        verbose=False,
    )
    oof_preds["CatBoost"][val_idx] = model_cb.predict(X_val_t)
    test_preds["CatBoost"] += model_cb.predict(X_te_t) / kf.n_splits

    # 2. LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=3500,
        learning_rate=0.02,
        num_leaves=24,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=15,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr_t,
        y_tr,
        eval_set=[(X_val_t, y_val)],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    oof_preds["LightGBM"][val_idx] = model_lgb.predict(X_val_t)
    test_preds["LightGBM"] += model_lgb.predict(X_te_t) / kf.n_splits

    # 3. XGBoost Regressor
    model_xgb = xgb.XGBRegressor(
        n_estimators=3500,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=80,
    )
    model_xgb.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)], verbose=False)
    oof_preds["XGBoost"][val_idx] = model_xgb.predict(X_val_t)
    test_preds["XGBoost"] += model_xgb.predict(X_te_t) / kf.n_splits

    # 4. Scaled Linear Models (Ridge & Huber) on One-Hot Encoded features
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr_l)
    X_val_sc = scaler.transform(X_val_l)
    X_te_sc = scaler.transform(X_te_l)

    model_ridge = Ridge(alpha=25.0)
    model_ridge.fit(X_tr_sc, y_tr)
    oof_preds["Ridge"][val_idx] = model_ridge.predict(X_val_sc)
    test_preds["Ridge"] += model_ridge.predict(X_te_sc) / kf.n_splits

    model_huber = HuberRegressor(alpha=15.0, max_iter=1000)
    model_huber.fit(X_tr_sc, y_tr)
    oof_preds["Huber"][val_idx] = model_huber.predict(X_val_sc)
    test_preds["Huber"] += model_huber.predict(X_te_sc) / kf.n_splits

# SLSQP Optimization on original dollar RMSE
oof_mat = np.column_stack([oof_preds[m] for m in models])
test_mat = np.column_stack([test_preds[m] for m in models])


def loss_fn(weights):
    w = weights / np.sum(weights)
    blend_log = np.dot(oof_mat, w)
    blend_dollar = np.expm1(blend_log)
    return np.sqrt(np.mean((y_train - blend_dollar) ** 2))


init_weights = [1.0 / len(models)] * len(models)
bounds = [(0, 1)] * len(models)
res = minimize(loss_fn, init_weights, method="SLSQP", bounds=bounds)
opt_w = res.x / np.sum(res.x)

oof_final_log = np.dot(oof_mat, opt_w)
oof_final_dollar = np.expm1(oof_final_log)

test_final_log = np.dot(test_mat, opt_w)
test_final_dollar = np.expm1(test_final_log)

# Print final metric
val_rmse = np.sqrt(np.mean((y_train - oof_final_dollar) ** 2))
print(f"Validation RMSE: {val_rmse:.2f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {"id": np.arange(len(test_final_dollar)), "SalePrice": test_final_dollar}
)
sub.to_csv("./working/submission.csv", index=False)
