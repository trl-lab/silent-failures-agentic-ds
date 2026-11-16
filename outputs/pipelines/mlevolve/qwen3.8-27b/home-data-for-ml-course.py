import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Separate target
target = train_df["SalePrice"].copy()
train_df.drop("SalePrice", axis=1, inplace=True)

# Keep test IDs for submission
test_ids = test_df.index.copy()


# ============================================================
# 2. FEATURE ENGINEERING (leakage-safe: fit on train only)
# ============================================================
def engineer_features(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()

    # --- Missing value indicators ---
    missing_cols = [
        "PoolQC",
        "Fence",
        "Alley",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "LotFrontage",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]
    for col in missing_cols:
        if col in train.columns:
            train[f"{col}_Missing"] = (train[col].isna()).astype(np.int8)
        if col in test.columns:
            test[f"{col}_Missing"] = (test[col].isna()).astype(np.int8)

    # LotFrontage: fill with train median
    if "LotFrontage" in train.columns:
        lf_median = train["LotFrontage"].median()
        train["LotFrontage"] = train["LotFrontage"].fillna(lf_median)
        test["LotFrontage"] = test["LotFrontage"].fillna(lf_median)

    # --- Fill missing values ---
    for col in train.select_dtypes(include=["object"]).columns:
        train[col] = train[col].fillna("None")
    for col in test.select_dtypes(include=["object"]).columns:
        test[col] = test[col].fillna("None")

    for col in train.select_dtypes(include=[np.number]).columns:
        if train[col].isna().any():
            med = train[col].median()
            train[col] = train[col].fillna(med)
            if col in test.columns:
                test[col] = test[col].fillna(med)

    # --- Ordinal encoding (domain knowledge) ---
    ordinal_maps = {
        "Street": {"Gravel": 0, "Paved": 1},
        "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
        "LandContour": {"HLS": 0, "Low": 1, "Lvl": 2},
        "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
        "Condition1": {
            "Artery": 0,
            "RRNn": 1,
            "PosN": 2,
            "RRNe": 3,
            "PosA": 4,
            "Norm": 5,
            "RRAn": 6,
            "RRAe": 7,
        },
        "Condition2": {
            "Artery": 0,
            "RRNn": 1,
            "PosN": 2,
            "RRNe": 3,
            "PosA": 4,
            "Norm": 5,
            "RRAn": 6,
            "RRAe": 7,
        },
        "BldgType": {
            "1Story": 0,
            "2Story": 1,
            "2Inc": 2,
            "1Fam": 3,
            "TwnhsE": 4,
            "Twnhs": 5,
        },
        "HouseStyle": {
            "1Story": 0,
            "1.5Fin": 1,
            "1.5Unf": 2,
            "2Story": 3,
            "2StoryNew": 4,
            "2.5Fin": 5,
            "2.5Unf": 6,
            "SFoyer": 7,
            "SLvl": 8,
        },
        "RoofClass": {
            "Flat": 0,
            "Gambrel": 1,
            "Gable": 2,
            "Hip": 3,
            "Mansard": 4,
            "Shed": 5,
        },
        "RoofMatl": {
            "ClyTile": 0,
            "Tymetal": 1,
            "Membrn": 2,
            "Precast": 3,
            "WdShake": 4,
            "WdShngl": 5,
            "Tar&Grv": 6,
        },
        "Exterior1st": {
            "AsbShng": 0,
            "BrkDwn": 1,
            "BrkTile": 2,
            "CBlock": 3,
            "CemntBd": 4,
            "CommStone": 5,
            "Concrete": 6,
            "HdBoard": 7,
            "MetalSd": 8,
            "Plywood": 9,
            "PreCast": 10,
            "Stone": 11,
            "Stucco": 12,
            "VinylSd": 13,
            "Wd Sdng": 14,
            "WdShing": 15,
        },
        "Exterior2nd": {
            "AsbShng": 0,
            "BrkDwn": 1,
            "BrkTile": 2,
            "CBlock": 3,
            "CemntBd": 4,
            "CommStone": 5,
            "Concrete": 6,
            "HdBoard": 7,
            "MetalSd": 8,
            "Other": 9,
            "Plywood": 10,
            "PreCast": 11,
            "Stone": 12,
            "Stucco": 13,
            "VinylSd": 14,
            "Wd Sdng": 15,
            "WdShing": 16,
            "None": 17,
        },
        "MasVnrType": {"BrkCmn": 0, "BrkTile": 1, "None": 2, "Stone": 3},
        "Foundation": {
            "BrkTil": 0,
            "CBlock": 1,
            "PConc": 2,
            "Slab": 3,
            "Stone": 4,
            "Wood": 5,
        },
        "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4},
        "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
        "BsmtFinType1": {"Unf": 0, "LwN": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
        "BsmtFinType2": {"Unf": 0, "LwN": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
        "Heating": {
            "Floor": 0,
            "Gravity": 1,
            "OthW": 2,
            "Wall": 3,
            "GasA": 4,
            "GasW": 5,
        },
        "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
        "CentralAir": {"N": 0, "Y": 1},
        "Electrical": {"SBrkr": 1, "FuseA": 0, "FuseF": 0, "FuseS": 0},
        "PavedDrive": {"N": 0, "P": 1, "Y": 2},
        "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3},
        "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
        "GarageType": {
            "2Car": 0,
            "Attchd": 1,
            "Basment": 2,
            "BuiltIn": 3,
            "CarPort": 4,
            "Detchd": 5,
            "None": 6,
        },
        "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
        "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "SaleType": {
            "COD": 0,
            "CWD": 1,
            "Con": 2,
            "ConLD": 3,
            "ConLI": 4,
            "ConLw": 5,
            "New": 6,
            "Oth": 7,
            "WD": 8,
        },
        "SaleCondition": {
            "Abnorml": 0,
            "AdjLand": 1,
            "Family": 2,
            "Normal": 3,
            "Partial": 4,
        },
        "LotConfig": {"Corner": 0, "CulDSac": 1, "FR2": 2, "FR3": 3, "Inside": 4},
        "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    }

    for col, mapping in ordinal_maps.items():
        if col in train.columns:
            train[col] = train[col].map(mapping).fillna(-1).astype(np.int8)
        if col in test.columns:
            test[col] = test[col].map(mapping).fillna(-1).astype(np.int8)

    # --- Numerical feature engineering ---
    area_cols = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea"]
    existing_area_cols = [c for c in area_cols if c in train.columns]
    if len(existing_area_cols) >= 2:
        train["TotalArea"] = train[existing_area_cols].sum(axis=1)
        test["TotalArea"] = test[existing_area_cols].sum(axis=1)

    porch_cols = [
        "OpenPorchSF",
        "ClosedPorchSF",
        "ThreeSeasonPorch",
        "EnclosedPorch",
        "ScreenPorch",
    ]
    existing_porch_cols = [c for c in porch_cols if c in train.columns]
    if len(existing_porch_cols) >= 2:
        train["TotalPorchArea"] = train[existing_porch_cols].sum(axis=1)
        test["TotalPorchArea"] = test[existing_porch_cols].sum(axis=1)

    bsmt_fin_cols = ["BsmtFinSF1", "BsmtFinSF2"]
    existing_bsmt_cols = [c for c in bsmt_fin_cols if c in train.columns]
    if len(existing_bsmt_cols) >= 1:
        train["TotalBsmtFinArea"] = train[existing_bsmt_cols].sum(axis=1)
        test["TotalBsmtFinArea"] = test[existing_bsmt_cols].sum(axis=1)

    if "YearBuilt" in train.columns:
        train["HouseAge"] = 2010 - train["YearBuilt"]
        test["HouseAge"] = 2010 - test["YearBuilt"]
    if "YearRemodAdd" in train.columns:
        train["RemodAge"] = 2010 - train["YearRemodAdd"]
        test["RemodAge"] = 2010 - test["YearRemodAdd"]
        if "YearBuilt" in train.columns:
            train["WasRemodeled"] = (
                train["YearRemodAdd"] != train["YearBuilt"]
            ).astype(np.int8)
            test["WasRemodeled"] = (test["YearRemodAdd"] != test["YearBuilt"]).astype(
                np.int8
            )

    if "OverallQual" in train.columns and "TotalArea" in train.columns:
        train["Qual_x_Area"] = train["OverallQual"] * train["TotalArea"]
        test["Qual_x_Area"] = test["OverallQual"] * test["TotalArea"]
    if "OverallQual" in train.columns and "HouseAge" in train.columns:
        train["Qual_x_Age"] = train["OverallQual"] * train["HouseAge"]
        test["Qual_x_Age"] = test["OverallQual"] * test["HouseAge"]

    # --- Label encode remaining categoricals (train categories only) ---
    remaining_cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    for col in remaining_cat_cols:
        categories = sorted(train[col].unique())
        cat_map = {c: i for i, c in enumerate(categories)}
        train[col] = train[col].map(cat_map).fillna(-1).astype(np.int8)
        test[col] = test[col].map(cat_map).fillna(-1).astype(np.int8)

    return train, test


train_processed, test_processed = engineer_features(train_df, test_df)

# ============================================================
# 3. LOG TRANSFORM TARGET & SPLIT
# ============================================================
target_log = np.log1p(target)

target_bins = pd.qcut(target_log, q=10, labels=False, duplicates="drop")
train_idx, val_idx = train_test_split(
    train_processed.index, test_size=0.15, random_state=42, stratify=target_bins
)

X_train = train_processed.loc[train_idx].reset_index(drop=True)
X_val = train_processed.loc[val_idx].reset_index(drop=True)
y_train = target_log.loc[train_idx].reset_index(drop=True)
y_val = target_log.loc[val_idx].reset_index(drop=True)

# --- Neighborhood target encoding (fit on train split only to prevent leakage) ---
_nbhd_temp = pd.DataFrame({
    "Neighborhood": X_train["Neighborhood"].values,
    "target": y_train.values,
})
_nbhd_mean = _nbhd_temp.groupby("Neighborhood")["target"].mean()
_nbhd_median = _nbhd_temp.groupby("Neighborhood")["target"].median()

X_train["Nbhd_Mean"] = X_train["Neighborhood"].map(_nbhd_mean)
X_val["Nbhd_Mean"] = X_val["Neighborhood"].map(_nbhd_mean)
test_processed["Nbhd_Mean"] = test_processed["Neighborhood"].map(_nbhd_mean)

X_train["Nbhd_Median"] = X_train["Neighborhood"].map(_nbhd_median)
X_val["Nbhd_Median"] = X_val["Neighborhood"].map(_nbhd_median)
test_processed["Nbhd_Median"] = test_processed["Neighborhood"].map(_nbhd_median)

# Neighborhood x OverallQual interaction
X_train["Nbhd_x_Qual"] = X_train["Neighborhood"] * X_train["OverallQual"]
X_val["Nbhd_x_Qual"] = X_val["Neighborhood"] * X_val["OverallQual"]
test_processed["Nbhd_x_Qual"] = test_processed["Neighborhood"] * test_processed["OverallQual"]

# Ensure test has same columns as train
test_processed = test_processed[X_train.columns]

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {test_processed.shape}")
print(
    f"NaN in X_train: {X_train.isna().sum().sum()}, X_val: {X_val.isna().sum().sum()}, X_test: {test_processed.isna().sum().sum()}"
)

# ============================================================
# 4. MODEL PARAMETERS
# ============================================================
model_params = {
    "objective": "huber",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# ============================================================
# 5. TRAIN ENSEMBLE (5 seeds) WITH EARLY STOPPING
# ============================================================
seeds = [42, 123, 456, 789, 1024]
val_preds_list = []
best_iters = []

for seed in seeds:
    params = {**model_params, "random_state": seed}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    best_iters.append(model.best_iteration_)
    val_pred = model.predict(X_val)
    val_preds_list.append(val_pred)
    val_pred_orig = np.expm1(val_pred)
    y_val_orig = np.expm1(y_val)
    model_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
    print(f"  Seed {seed}: best_iter={best_iters[-1]}, val_RMSE={model_rmse:.2f}")

# --- XGBoost ensemble (3 seeds) for model diversity ---
xgb_seeds = [42, 123, 456]
xgb_best_iters = []

for seed in xgb_seeds:
    xgb_model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=5000,
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_best_iters.append(xgb_model.best_iteration)
    xgb_val_pred = xgb_model.predict(X_val)
    val_preds_list.append(xgb_val_pred)
    xgb_val_pred_orig = np.expm1(xgb_val_pred)
    y_val_orig = np.expm1(y_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val_orig, xgb_val_pred_orig))
    print(f"  XGB Seed {seed}: best_iter={xgb_model.best_iteration}, val_RMSE={xgb_rmse:.2f}")

# Ensemble validation (5 LGB + 3 XGB = 8 models)
val_preds_avg = np.mean(val_preds_list, axis=0)
val_preds_avg_orig = np.expm1(val_preds_avg)
y_val_orig = np.expm1(y_val)
final_val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_preds_avg_orig))
print(f"\nEnsemble Validation RMSE (original scale): {final_val_rmse:.4f}")

# ============================================================
# 6. TRAIN FINAL MODELS ON FULL DATA & PREDICT TEST
# ============================================================
X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = target_log.reset_index(drop=True)

median_best_iter = int(np.median(best_iters))
print(f"Median best iterations: {median_best_iter}")

test_preds_list = []
for seed in seeds:
    params = {**model_params, "random_state": seed, "n_estimators": median_best_iter}
    model_final = lgb.LGBMRegressor(**params)
    model_final.fit(X_full, y_full)
    test_pred = model_final.predict(test_processed)
    test_preds_list.append(test_pred)

# Final XGBoost models on full data
median_best_iter_xgb = int(np.median(xgb_best_iters))
print(f"Median XGB best iterations: {median_best_iter_xgb}")

for seed in xgb_seeds:
    xgb_final = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=median_best_iter_xgb,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_final.fit(X_full, y_full)
    xgb_test_pred = xgb_final.predict(test_processed)
    test_preds_list.append(xgb_test_pred)

test_preds_avg = np.mean(test_preds_list, axis=0)
test_preds_avg_orig = np.expm1(test_preds_avg)

# ============================================================
# 7. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "SalePrice": test_preds_avg_orig}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved: {submission.shape[0]} rows")
print(
    f"Predicted range: [{test_preds_avg_orig.min():.2f}, {test_preds_avg_orig.max():.2f}], mean: {test_preds_avg_orig.mean():.2f}"
)

# ============================================================
# 8. FINAL VALIDATION SCORE (OFFICIAL METRIC: RMSE)
# ============================================================
score = final_val_rmse
print(f"Final Validation Score: {score}")
