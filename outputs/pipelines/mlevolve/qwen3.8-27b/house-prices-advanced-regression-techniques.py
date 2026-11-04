import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import lightgbm as lgb
from catboost import CatBoostRegressor
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "SalePrice"
id_col = "Id"

test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))

y = train[target_col].values
train = train.drop(columns=[target_col])
if id_col in train.columns:
    train_ids = train[id_col].values
    train = train.drop(columns=[id_col])
if id_col in test.columns:
    test = test.drop(columns=[id_col])

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT (BEFORE any processing)
# ============================================================
train_indices, val_indices = train_test_split(
    np.arange(len(train)), test_size=0.2, random_state=42
)

train_split = train.iloc[train_indices].reset_index(drop=True)
val_split = train.iloc[val_indices].reset_index(drop=True)
y_train = y[train_indices]
y_val = y[val_indices]

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ============================================================
# 3. IDENTIFY COLUMN TYPES
# ============================================================
all_data = pd.concat([train_split, val_split, test], axis=0)

numeric_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = all_data.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_cols = [c for c in numeric_cols if c not in [id_col, target_col]]
categorical_cols = [c for c in categorical_cols if c not in [id_col, target_col]]

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# ============================================================
# 4. DOMAIN-INFORMED ORDINAL ENCODING
# ============================================================
ordinal_mappings = {
    "Street": {"Grvl": 0, "Pave": 1},
    "Alley": {"nan": 0, "Grvl": 1, "Pave": 2},
    "LotShape": {"HV": 0, "IR1": 1, "IR2": 2, "IR3": 3, "Reg": 4},
    "Utilities": {"ELO": 0, "NoSewr": 1, "NoSvr": 2, "AllPub": 3},
    "LotConfig": ("Corner", "CulDSac", "EndSaw", "Inside", "FR2", "FR3"),
    "LandSlope": ("Lvl", "Gtl", "Mod", "Steep"),
    "Condition1": ("Artery", "RRAe", "PosN", "PosA", "Norm", "RRNn"),
    "Condition2": ("Artery", "RRAe", "PosN", "PosA", "Norm", "RRNn"),
    "BldgType": ("1StPr", "TwnhsE", "Twnhs", "Duplx", "2Con"),
    "HouseStyle": (
        "1Story",
        "1.5Fin",
        "1.5Unf",
        "2Story",
        "2.5Fin",
        "2.5Unf",
        "SFoyer",
        "SLvl",
    ),
    "ExterQual": ("Po", "Fa", "TA", "Gd", "Ex"),
    "ExterCond": ("Po", "Fa", "TA", "Gd", "Ex"),
    "MasVnrType": ("None", "BrkN", "BrkC", "AtchSt", "Stone"),
    "Foundation": ("BrkTil", "Buffalo", "CBlock", "PConc", "Slab", "Wood"),
    "BsmtQual": ("Po", "Fa", "TA", "Gd", "Ex", "No"),
    "BsmtCond": ("Po", "Fa", "TA", "Gd", "Ex", "No"),
    "BsmtExposure": ("No", "Mn", "Av", "Gd", "No"),
    "BsmtFinType1": ("Unf", "LwA", "Rec", "BLQ", "ALQ", "GLQ", "No"),
    "BsmtFinType2": ("Unf", "LwA", "Rec", "BLQ", "ALQ", "GLQ", "No"),
    "Heating": ("Floor", "Gravity", "OthW", "Wall", "GasS", "OthH", "OthW"),
    "CentralAir": ("N", "Y"),
    "Electrical": ("Sbrkr", "FusA", "FusF", "FusP", "Fuse"),
    "Functional": ("Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"),
    "FireplaceQu": ("Po", "Fa", "TA", "Gd", "Ex", "No"),
    "GarageType": ("Detchd", "BuiltIn", "CarPort", "2Types", "Basment", "Attchd", "No"),
    "GarageFinish": ("Unf", "RFn", "Fin", "No"),
    "GarageQual": ("Po", "Fa", "TA", "Gd", "Ex", "No"),
    "GarageCond": ("Po", "Fa", "TA", "Gd", "Ex", "No"),
    "PavedDrive": ("N", "Y", "D"),
    "Fence": ("MnPrv", "MnVw", "GdPrv", "GdVw", "No"),
    "SaleType": ("Wd", "CWD", "New", "COD", "ConLD", "ConLw", "ConLI", "Oth", "Con"),
    "SaleCondition": ("Abnorm", "AdjLand", "Family", "Normal", "Partial"),
}

none_indicators = {
    "BsmtQual": "No",
    "BsmtCond": "No",
    "BsmtExposure": "No",
    "BsmtFinType1": "No",
    "BsmtFinType2": "No",
    "GarageType": "No",
    "GarageFinish": "No",
    "GarageQual": "No",
    "GarageCond": "No",
    "FireplaceQu": "No",
    "MasVnrType": "None",
    "PoolQC": "No",
    "Misc": "0",
}

target_encode_cols = [
    "Neighborhood",
    "MSSubClass",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "CentralAir",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "SaleCondition",
]

ordinal_cols = [
    c
    for c in categorical_cols
    if c in ordinal_mappings and ordinal_mappings[c] is not None
]
remaining_cat_cols = [
    c for c in categorical_cols if c not in ordinal_cols and c not in target_encode_cols
]

# ============================================================
# 5. MISSING VALUE HANDLING
# ============================================================
missing_strategies = {}
for col, none_val in none_indicators.items():
    if col in train_split.columns:
        missing_strategies[col] = none_val

for col in numeric_cols:
    if col in train_split.columns and train_split[col].isna().any():
        if col in [
            "GarageYrBuilt",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal",
        ]:
            missing_strategies[col] = 0
        else:
            missing_strategies[col] = "median"

cat_missing_map = {
    "Alley": "None",
    "Street": "Pave",
    "Utilities": "AllPub",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0,
    "BsmtQual": "No",
    "BsmtCond": "No",
    "BsmtExposure": "No",
    "BsmtFinType1": "No",
    "BsmtFinType2": "No",
    "BsmtFinSF1": 0,
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "BsmtFullBath": 0,
    "BsmtHalfBath": 0,
    "GarageType": "No",
    "GarageYrBuilt": 0,
    "GarageFinish": "No",
    "GarageCars": 0,
    "GarageArea": 0,
    "FireplaceQu": "No",
    "PoolQC": "No",
    "Misc": "0",
    "Fence": "No",
    "MSZoning": "RL",
    "ExterQual": "TA",
    "ExterCond": "TA",
}


def impute_missing(df, is_fitting=False, medians=None):
    df = df.copy()
    if medians is None:
        medians = {}
    for col in df.columns:
        if df[col].isna().any():
            if col in missing_strategies:
                val = missing_strategies[col]
                if val == "median":
                    if is_fitting:
                        medians[col] = df[col].median()
                    df[col] = df[col].fillna(medians.get(col, 0))
                else:
                    df[col] = df[col].fillna(val)
            elif col in cat_missing_map:
                df[col] = df[col].fillna(cat_missing_map[col])
            elif df[col].dtype in ["object", "category"]:
                if is_fitting:
                    mode_val = df[col].mode()
                    medians[col] = mode_val[0] if len(mode_val) > 0 else "Unknown"
                df[col] = df[col].fillna(medians.get(col, "Unknown"))
            else:
                if is_fitting:
                    medians[col] = df[col].median()
                df[col] = df[col].fillna(medians.get(col, 0))
    return df, medians


train_split, medians = impute_missing(train_split, is_fitting=True)
val_split, _ = impute_missing(val_split, is_fitting=False, medians=medians)
test, _ = impute_missing(test, is_fitting=False, medians=medians)

# ============================================================
# 6. FIT TARGET ENCODING ON TRAIN ONLY
# ============================================================
te_mappings = {}
global_mean_te = y_train_log.mean()
for col in target_encode_cols:
    if col in train_split.columns:
        group_means = (
            train_split.groupby(col).apply(lambda g: g["__target__"].mean())
            if "__target__" in train_split.columns
            else None
        )
        # Add target temporarily
        train_split_temp = train_split.copy()
        train_split_temp["__target__"] = y_train_log
        group_means = train_split_temp.groupby(col)["__target__"].mean()
        group_counts = train_split_temp.groupby(col)["__target__"].count()
        smooth = 10
        for cat in train_split_temp[col].unique():
            if cat in group_means.index:
                gm = group_means[cat]
                gc = group_counts[cat]
                te_mappings[col + "_" + str(cat)] = (
                    gc * gm + smooth * global_mean_te
                ) / (gc + smooth)
            else:
                te_mappings[col + "_" + str(cat)] = global_mean_te

freq_mappings = {}
for col in remaining_cat_cols:
    if col in train_split.columns:
        freq_mappings[col] = train_split[col].value_counts(normalize=True)

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================


def apply_ordinal_encoding(df):
    df = df.copy()
    for col, mapping in ordinal_mappings.items():
        if col not in df.columns:
            continue
        if isinstance(mapping, dict):
            df[col + "_ord"] = df[col].map(mapping)
        elif isinstance(mapping, tuple):
            order = {v: i for i, v in enumerate(mapping)}
            df[col + "_ord"] = df[col].map(order)
        if col in df.columns and df[col].dtype in ["object", "category"]:
            df = df.drop(columns=[col])
    return df


def apply_target_encoding(df):
    df = df.copy()
    for col in target_encode_cols:
        if col not in df.columns:
            continue
        df[col + "_te"] = df[col].map(
            lambda x: te_mappings.get(col + "_" + str(x), global_mean_te)
        )
        if col in df.columns and df[col].dtype in ["object", "category"]:
            df = df.drop(columns=[col])
    return df


def apply_frequency_encoding(df):
    df = df.copy()
    for col in remaining_cat_cols:
        if col not in df.columns:
            continue
        if col in freq_mappings:
            df[col + "_freq"] = df[col].map(freq_mappings[col]).fillna(0)
        else:
            df[col + "_freq"] = 0.0
        if col in df.columns and df[col].dtype in ["object", "category"]:
            df = df.drop(columns=[col])
    return df


def engineer_features(df):
    df = df.copy()
    df = apply_ordinal_encoding(df)
    df = apply_target_encoding(df)
    df = apply_frequency_encoding(df)

    if "GrLivArea" in df.columns:
        df["TotalSF"] = df.get("GrLivArea", 0) + df.get("TotalBsmtSF", 0)
    if "GrLivArea" in df.columns and "1stFlrSF" in df.columns:
        df["2ndFlrSF"] = df["GrLivArea"] - df["1stFlrSF"]

    if "GrLivArea" in df.columns:
        porch_cols = ["OpenPorchSF", "EnclosedPorch", "ScreenPorch"]
        porch_sum = sum(df.get(c, 0) for c in porch_cols if c in df.columns)
        df["PorchRatio"] = porch_sum / (df["GrLivArea"] + 1)

    if "GrLivArea" in df.columns and "TotalBsmtSF" in df.columns:
        df["BsmtRatio"] = df["TotalBsmtSF"] / (df["GrLivArea"] + 1)

    if "OverallQual" in df.columns and "GrLivArea" in df.columns:
        df["Qual_Area"] = df["OverallQual"] * df["GrLivArea"]
    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        df["Qual_Cond"] = df["OverallQual"] * df["OverallCond"]
    if "OverallQual" in df.columns and "YearBuilt" in df.columns:
        df["Qual_Year"] = df["OverallQual"] * df["YearBuilt"]
    if "OverallQual" in df.columns and "GarageArea" in df.columns:
        df["Qual_Garage"] = df["OverallQual"] * df["GarageArea"]
    if "OverallQual" in df.columns and "TotalBsmtSF" in df.columns:
        df["Qual_Bsmt"] = df["OverallQual"] * df["TotalBsmtSF"]
    if "OverallQual" in df.columns and "TotalSF" in df.columns:
        df["Qual_TotalSF"] = df["OverallQual"] * df["TotalSF"]

    if "GarageArea" in df.columns and "GarageCars" in df.columns:
        df["GarageAreaPerCar"] = df["GarageArea"] / (df["GarageCars"] + 1)

    if "YearBuilt" in df.columns and "YrSold" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodAge"] = df["YearRemodAdd"] - df["YearBuilt"]
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    if "GrLivArea" in df.columns:
        if "TotRmsAbvGrd" in df.columns:
            df["RoomsPerArea"] = df["TotRmsAbvGrd"] / (df["GrLivArea"] + 1)
        if "FullBath" in df.columns:
            df["BathPerArea"] = (df["FullBath"] + 0.5 * df.get("HalfBath", 0)) / (
                df["GrLivArea"] + 1
            )

    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "2ndFlrSF" in df.columns:
        df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    if "FireplaceQu_ord" in df.columns:
        df["HasFireplace"] = (df["FireplaceQu_ord"] > 0).astype(int)
    if "BsmtFinSF1" in df.columns:
        df["HasBsmtFin"] = ((df["BsmtFinSF1"] + df.get("BsmtFinSF2", 0)) > 0).astype(
            int
        )
    if "WoodDeckSF" in df.columns:
        df["HasDeck"] = (df["WoodDeckSF"] > 0).astype(int)
    if "OpenPorchSF" in df.columns:
        df["HasPorch"] = (df["OpenPorchSF"] > 0).astype(int)

    if "ExterQual_ord" in df.columns and "ExterCond_ord" in df.columns:
        df["ExterScore"] = df["ExterQual_ord"].fillna(0) + df["ExterCond_ord"].fillna(0)
    if "BsmtQual_ord" in df.columns and "BsmtCond_ord" in df.columns:
        df["BsmtScore"] = df["BsmtQual_ord"].fillna(0) + df["BsmtCond_ord"].fillna(0)
    if "GarageQual_ord" in df.columns and "GarageCond_ord" in df.columns:
        df["GarageScore"] = df["GarageQual_ord"].fillna(0) + df[
            "GarageCond_ord"
        ].fillna(0)

    if "LotArea" in df.columns and "LotFrontage" in df.columns:
        df["LotFrontagePerArea"] = df["LotFrontage"].fillna(0) / (df["LotArea"] + 1)
    if "LotArea" in df.columns and "GrLivArea" in df.columns:
        df["LotAreaPerGrLiv"] = df["LotArea"] / (df["GrLivArea"] + 1)

    if "FullBath" in df.columns:
        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df.get("HalfBath", 0)
            + df.get("BsmtFullBath", 0)
            + 0.5 * df.get("BsmtHalfBath", 0)
        )

    if "GrLivArea" in df.columns and "TotRmsAbvGrd" in df.columns:
        df["AreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

    if "FireplaceQu_ord" in df.columns and "GrLivArea" in df.columns:
        df["Fireplace_Area"] = df["FireplaceQu_ord"].fillna(0) * df["GrLivArea"]

    return df


train_split = engineer_features(train_split)
val_split = engineer_features(val_split)
test = engineer_features(test)

# ============================================================
# 8. FINAL FEATURE SELECTION & CLEANUP
# ============================================================
final_numeric_cols = train_split.select_dtypes(include=[np.number]).columns.tolist()
all_features = [
    c for c in final_numeric_cols if c in val_split.columns and c in test.columns
]

train_split = train_split[all_features].fillna(0).replace([np.inf, -np.inf], 0)
val_split = val_split[all_features].fillna(0).replace([np.inf, -np.inf], 0)
test = test[all_features].fillna(0).replace([np.inf, -np.inf], 0)

print(f"\nFinal feature count: {len(all_features)}")
print(f"Train: {train_split.shape}, Val: {val_split.shape}, Test: {test.shape}")

# ============================================================
# 9. MODEL ARCHITECTURE: CatBoost Regressor
# ============================================================

catboost_params = {
    "learning_rate": 0.03,
    "iterations": 5000,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_strength": 2,
    "early_stopping_rounds": 150,
    "verbose": 0,
    "random_seed": 42,
}


# ============================================================
# 10. TRAINING CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)

N_FOLDS = 5

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 20,
    "max_depth": -1,
    "min_child_samples": 15,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
}

# ============================================================
# 11. PREPARE FEATURES
# ============================================================
X_train = train_split[all_features].values.astype(np.float32)
X_val = val_split[all_features].values.astype(np.float32)
X_test = test[all_features].values.astype(np.float32)

n_features = len(all_features)

# ============================================================
# 12. K-FOLD OOF TRAINING (on train_split only)
# ============================================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cb_oof = np.zeros(len(X_train))
lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))
cb_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/{N_FOLDS}")

    X_tr = X_train[tr_idx]
    X_va = X_train[va_idx]
    X_te = X_test
    y_tr, y_va = y_train_log[tr_idx], y_train_log[va_idx]

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    lgb_oof[va_idx] = lgb_model.predict(X_va)
    lgb_test_preds += lgb_model.predict(X_te) / N_FOLDS

    # --- CatBoost ---
    cb_model = CatBoostRegressor(**catboost_params)
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=0,
    )
    cb_oof[va_idx] = cb_model.predict(X_va)
    cb_test_preds += cb_model.predict(X_te) / N_FOLDS

# ============================================================
# 13. HOLD-OUT VALIDATION PREDICTIONS (independent val set)
# ============================================================
# Train full models on entire train_split, predict on val_split
lgb_full = lgb.LGBMRegressor(**lgb_params)
lgb_full.fit(X_train, y_train_log)
lgb_val_pred = lgb_full.predict(X_val)

cb_full = CatBoostRegressor(**catboost_params)
cb_full.fit(X_train, y_train_log, verbose=0)
cb_val_pred = cb_full.predict(X_val)

# ============================================================
# 14. ENSEMBLE WEIGHT OPTIMIZATION (on held-out val set)
# ============================================================


def rmsle_blend_val(w):
    pred_log = w * cb_val_pred + (1 - w) * lgb_val_pred
    pred_original = np.expm1(pred_log)
    return np.sqrt(np.mean((np.log1p(pred_original) - y_val_log) ** 2))


result = minimize_scalar(rmsle_blend_val, bounds=(0, 1), method="bounded")
opt_w = result.x
cb_rmsle_val = rmsle_blend_val(1.0)
lgb_rmsle_val = rmsle_blend_val(0.0)
blend_rmsle_val = rmsle_blend_val(opt_w)
print(
    f"Val RMSLE -> CatBoost: {cb_rmsle_val:.6f}, LGB: {lgb_rmsle_val:.6f}, Blend: {blend_rmsle_val:.6f}"
)
print(f"Optimal CatBoost weight: {opt_w:.4f}")

# ============================================================
# 15. FINAL ENSEMBLE PREDICTIONS (using K-fold test predictions)
# ============================================================
test_pred_log = opt_w * cb_test_preds + (1 - opt_w) * lgb_test_preds

# ============================================================
# 16. VALIDATION METRIC (held-out val set, official RMSLE)
# ============================================================
rmsle_score = blend_rmsle_val

print(f"\n=== FINAL RESULTS ===")
print(f"CatBoost Val RMSLE: {cb_rmsle_val:.6f}")
print(f"LGB Val RMSLE: {lgb_rmsle_val:.6f}")
print(f"Ensemble Val RMSLE: {rmsle_score:.6f}")
print(f"Blend weight (CatBoost): {opt_w:.4f}")

# ============================================================
# 17. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 100000, None)

submission = pd.DataFrame({"id": test_ids.astype(int), "SalePrice": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Prediction range: [{test_pred.min():.0f}, {test_pred.max():.0f}]")

print(f"Final Validation Score: {rmsle_score:.6f}")