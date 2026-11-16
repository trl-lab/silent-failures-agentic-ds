import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SalePrice"
y = train_df[target_col].values
y_log = np.log1p(y)

train_len = len(train_df)
test_len = len(test_df)
combined = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], axis=0
).reset_index(drop=True)

if "Id" in combined.columns:
    combined = combined.drop(columns=["Id"])

# Domain Feature Engineering - Quality mappings
qual_map = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
    np.nan: 0,
    "NA": 0,
}
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
        combined[col + "_Num"] = combined[col].map(qual_map).fillna(0).astype(float)

# Numeric imputations and additive features
num_cols = [
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "WoodDeckSF",
    "LotFrontage",
    "MasVnrArea",
    "GarageArea",
    "GarageCars",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
]
for col in num_cols:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

combined["TotalSF"] = (
    combined["TotalBsmtSF"] + combined["1stFlrSF"] + combined["2ndFlrSF"]
)
combined["TotalBath"] = (
    combined["FullBath"]
    + 0.5 * combined["HalfBath"]
    + combined["BsmtFullBath"]
    + 0.5 * combined["BsmtHalfBath"]
)
combined["TotalPorchSF"] = (
    combined["OpenPorchSF"]
    + combined["EnclosedPorch"]
    + combined["3SsnPorch"]
    + combined["ScreenPorch"]
    + combined["WoodDeckSF"]
)

combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
combined["RemodAge"] = combined["YrSold"] - combined["YearRemodAdd"]
garage_yr = pd.to_numeric(
    combined.get("GarageYrBlt", combined["YearBuilt"]), errors="coerce"
).fillna(combined["YearBuilt"])
combined["GarageAge"] = combined["YrSold"] - garage_yr

combined["Qual_TotalSF"] = combined["OverallQual"] * combined["TotalSF"]
combined["Qual_GrLivArea"] = combined["OverallQual"] * combined["GrLivArea"]
combined["OverallGrade"] = combined["OverallQual"] * combined["OverallCond"]

combined["HasGarage"] = (combined["GarageArea"] > 0).astype(int)
combined["HasBasement"] = (combined["TotalBsmtSF"] > 0).astype(int)
combined["HasFireplace"] = (
    pd.to_numeric(combined.get("Fireplaces", 0), errors="coerce").fillna(0) > 0
).astype(int)
combined["HasPool"] = (
    pd.to_numeric(combined.get("PoolArea", 0), errors="coerce").fillna(0) > 0
).astype(int)

# Identify categorical features and prepare representations
raw_cat_cols = []
for col in combined.columns:
    if (
        combined[col].dtype == "object"
        or combined[col].dtype.name == "category"
        or col == "MSSubClass"
    ):
        combined[col] = combined[col].astype(str).fillna("Missing")
        raw_cat_cols.append(col)
    else:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(0 if np.isnan(median_val) else median_val)

# Dataset for CatBoost
df_cb = combined.copy()

# Dataset for LightGBM
df_lgb = combined.copy()
for col in raw_cat_cols:
    df_lgb[col] = df_lgb[col].astype("category")

# Dataset for XGBoost (Label Encoded)
df_xgb = combined.copy()
for col in raw_cat_cols:
    le = LabelEncoder()
    df_xgb[col] = le.fit_transform(df_xgb[col].astype(str))

# Dataset for Ridge (One-Hot Encoded & Scaled)
df_ridge = pd.get_dummies(combined, columns=raw_cat_cols, drop_first=True)
scaler = StandardScaler()
df_ridge_scaled = pd.DataFrame(scaler.fit_transform(df_ridge), columns=df_ridge.columns)

# Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(train_len)
test_cb = np.zeros(test_len)

oof_lgb = np.zeros(train_len)
test_lgb = np.zeros(test_len)

oof_xgb = np.zeros(train_len)
test_xgb = np.zeros(test_len)

oof_ridge = np.zeros(train_len)
test_ridge = np.zeros(test_len)

for fold, (train_idx, val_idx) in enumerate(kf.split(combined.iloc[:train_len])):
    y_tr, y_va = y_log[train_idx], y_log[val_idx]

    # 1. CatBoost
    X_tr_cb, X_va_cb = (
        df_cb.iloc[train_idx].copy(),
        df_cb.iloc[val_idx].copy(),
    )
    X_te_cb = df_cb.iloc[train_len:].copy()
    cb_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=100,
    )
    cb_model.fit(
        X_tr_cb,
        y_tr,
        cat_features=raw_cat_cols,
        eval_set=(X_va_cb, y_va),
        use_best_model=True,
    )
    oof_cb[val_idx] = cb_model.predict(X_va_cb)
    test_cb += cb_model.predict(X_te_cb) / n_splits

    # 2. LightGBM
    X_tr_lgb, X_va_lgb = (
        df_lgb.iloc[train_idx].copy(),
        df_lgb.iloc[val_idx].copy(),
    )
    X_te_lgb = df_lgb.iloc[train_len:].copy()
    lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr)
    lgb_val = lgb.Dataset(X_va_lgb, label=y_va, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42 + fold,
        "verbose": -1,
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(
        X_va_lgb, num_iteration=lgb_model.best_iteration
    )
    test_lgb += (
        lgb_model.predict(X_te_lgb, num_iteration=lgb_model.best_iteration) / n_splits
    )

    # 3. XGBoost
    X_tr_xgb, X_va_xgb = (
        df_xgb.iloc[train_idx].copy(),
        df_xgb.iloc[val_idx].copy(),
    )
    X_te_xgb = df_xgb.iloc[train_len:].copy()
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.75,
        colsample_bytree=0.7,
        random_state=42 + fold,
        eval_metric="rmse",
        early_stopping_rounds=100,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va_xgb)
    test_xgb += xgb_model.predict(X_te_xgb) / n_splits

    # 4. Ridge Regression
    X_tr_r, X_va_r = (
        df_ridge_scaled.iloc[train_idx].copy(),
        df_ridge_scaled.iloc[val_idx].copy(),
    )
    X_te_r = df_ridge_scaled.iloc[train_len:].copy()
    ridge_model = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge_model.fit(X_tr_r, y_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_va_r)
    test_ridge += ridge_model.predict(X_te_r) / n_splits

# Individual OOF RMSLE scores
print(f"CatBoost OOF RMSLE: {root_mean_squared_error(y_log, oof_cb):.5f}")
print(f"LightGBM OOF RMSLE: {root_mean_squared_error(y_log, oof_lgb):.5f}")
print(f"XGBoost OOF RMSLE:  {root_mean_squared_error(y_log, oof_xgb):.5f}")
print(f"Ridge OOF RMSLE:    {root_mean_squared_error(y_log, oof_ridge):.5f}")

# Finding optimal ensemble weights
oof_matrix = np.column_stack([oof_cb, oof_lgb, oof_xgb, oof_ridge])
test_matrix = np.column_stack([test_cb, test_lgb, test_xgb, test_ridge])


def loss_func(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    return root_mean_squared_error(y_log, pred)


init_weights = np.ones(4) / 4
bounds = [(0, 1) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)

opt_weights = opt_res.x / np.sum(opt_res.x)
print(f"Optimal Ensemble Weights (CB, LGB, XGB, Ridge): {np.round(opt_weights, 4)}")

oof_blend = np.dot(oof_matrix, opt_weights)
cv_rmsle = root_mean_squared_error(y_log, oof_blend)
print(f"Ensemble OOF RMSLE: {cv_rmsle:.5f}")

# Generate test predictions
test_blend_log = np.dot(test_matrix, opt_weights)
test_preds = np.expm1(test_blend_log)

# Create submission dataframe
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "SalePrice": test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print(f"Saved submission to ./working/submission.csv with shape {submission.shape}")
