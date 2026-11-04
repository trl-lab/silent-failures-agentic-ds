import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Load Data
os.makedirs("./working", exist_ok=True)
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SalePrice"
y_raw = train_df[target_col].values.astype(np.float64)
y_log = np.log1p(y_raw)

train_features = train_df.drop(columns=[target_col], errors="ignore")
test_features = test_df.copy()

drop_cols = [c for c in ["Id", "id"] if c in train_features.columns]
if drop_cols:
    train_features = train_features.drop(columns=drop_cols)
drop_cols_test = [c for c in ["Id", "id"] if c in test_features.columns]
if drop_cols_test:
    test_features = test_features.drop(columns=drop_cols_test)

n_train = len(train_features)
n_test = len(test_features)
combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)


# 2. Feature Engineering
def engineer_features(df):
    data = df.copy()

    # Treat MSSubClass as categorical
    if "MSSubClass" in data.columns:
        data["MSSubClass"] = data["MSSubClass"].astype(str)

    # Ordinal mappings for quality-related features
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0, "Missing": 0}
    qual_cols = [
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
    for col in qual_cols:
        if col in data.columns:
            data[col] = data[col].fillna("Missing").map(lambda x: qual_map.get(x, 0))

    # Impute numeric columns with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    # Structural / Area aggregations
    data["TotalSF"] = (
        data.get("TotalBsmtSF", 0) + data.get("1stFlrSF", 0) + data.get("2ndFlrSF", 0)
    )
    data["FinishedBsmtSF"] = np.maximum(
        0, data.get("TotalBsmtSF", 0) - data.get("BsmtUnfSF", 0)
    )
    data["FinishedTotalSF"] = data["FinishedBsmtSF"] + data.get("GrLivArea", 0)
    data["TotalBath"] = (
        data.get("FullBath", 0)
        + 0.5 * data.get("HalfBath", 0)
        + data.get("BsmtFullBath", 0)
        + 0.5 * data.get("BsmtHalfBath", 0)
    )
    data["TotalPorchSF"] = (
        data.get("OpenPorchSF", 0)
        + data.get("EnclosedPorch", 0)
        + data.get("3SsnPorch", 0)
        + data.get("ScreenPorch", 0)
    )

    # Indicator features
    data["HasPool"] = (data.get("PoolArea", 0) > 0).astype(int)
    data["Has2ndFloor"] = (data.get("2ndFlrSF", 0) > 0).astype(int)
    data["HasGarage"] = (data.get("GarageArea", 0) > 0).astype(int)
    data["HasBsmt"] = (data.get("TotalBsmtSF", 0) > 0).astype(int)
    data["HasFireplace"] = (data.get("Fireplaces", 0) > 0).astype(int)

    # Age features
    data["HouseAge"] = data.get("YrSold", 2010) - data.get("YearBuilt", 1970)
    data["RemodAge"] = data.get("YrSold", 2010) - data.get("YearRemodAdd", 1970)
    data["GarageAge"] = data.get("YrSold", 2010) - data.get("GarageYrBlt", 1970)

    # Key interaction & ratio features
    data["OverallGrade"] = data.get("OverallQual", 5) * data.get("OverallCond", 5)
    data["TotalSF_OverallQual"] = data["TotalSF"] * data.get("OverallQual", 5)
    data["GrLivArea_OverallQual"] = data.get("GrLivArea", 0) * data.get(
        "OverallQual", 5
    )
    data["RoomSize_Avg"] = data.get("GrLivArea", 0) / (
        data.get("TotRmsAbvGrd", 1) + 1e-5
    )
    data["QualPerAge"] = data.get("OverallQual", 5) / (data["HouseAge"] + 1.0)

    # Neighborhood-relative features
    neigh_liv_mean = data.groupby("Neighborhood")["GrLivArea"].transform("mean")
    neigh_sf_mean = data.groupby("Neighborhood")["TotalSF"].transform("mean")
    neigh_qual_mean = data.groupby("Neighborhood")["OverallQual"].transform("mean")

    data["GrLivArea_rel_Neigh"] = data["GrLivArea"] / (neigh_liv_mean + 1e-5)
    data["TotalSF_rel_Neigh"] = data["TotalSF"] / (neigh_sf_mean + 1e-5)
    data["OverallQual_diff_Neigh"] = data["OverallQual"] - neigh_qual_mean

    # Convert remaining non-numeric columns to category
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        data[col] = data[col].fillna("Missing").astype(str).astype("category")

    return data


processed = engineer_features(combined)

X_train = processed.iloc[:n_train].copy()
X_test = processed.iloc[n_train:].copy()

cat_features = list(X_train.select_dtypes(include=["category"]).columns)
num_features = list(X_train.select_dtypes(exclude=["category"]).columns)

# Preprocessing pipeline for Ridge
ridge_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), num_features),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            cat_features,
        ),
    ]
)

# 3. Model Training & 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(n_test)

oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(n_test)

oof_cat = np.zeros(n_train)
test_cat = np.zeros(n_test)

oof_ridge = np.zeros(n_train)
test_ridge = np.zeros(n_test)

# Convert category columns to string for CatBoost
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
for col in cat_features:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_log[val_idx]

    # --- LightGBM ---
    model_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        n_estimators=2000,
        random_state=42 + fold,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / n_splits

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=42 + fold,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / n_splits

    # --- CatBoost ---
    X_tr_c, X_va_c = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    model_cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_features,
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr_c, y_tr, eval_set=(X_va_c, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_va_c)
    test_cat += model_cat.predict(X_test_cat) / n_splits

    # --- Ridge Regression ---
    ridge_pipe = Pipeline([("prep", ridge_preprocessor), ("ridge", Ridge(alpha=15.0))])
    ridge_pipe.fit(X_tr, y_tr)
    oof_ridge[val_idx] = ridge_pipe.predict(X_va)
    test_ridge += ridge_pipe.predict(X_test) / n_splits

# 4. Ensemble Weight Optimization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_ridge])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_ridge])


def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    pred_log = oof_matrix @ weights
    pred_raw = np.expm1(pred_log)
    return root_mean_squared_error(y_raw, pred_raw)


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
opt_weights = res.x / np.sum(res.x)

oof_pred_raw = np.expm1(oof_matrix @ opt_weights)
test_pred_raw = np.expm1(test_matrix @ opt_weights)

overall_rmse = root_mean_squared_error(y_raw, oof_pred_raw)
print(f"Optimal Ensemble Weights (LGB, XGB, CAT, Ridge): {np.round(opt_weights, 4)}")
print(f"Overall OOF Validation RMSE: {overall_rmse:.4f}")

# 5. Export Submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_raw)), "SalePrice": test_pred_raw}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
