"""Ames Housing Price Prediction - Complete Training and Inference Pipeline."""

import os
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# Set random seed for deterministic reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =============================================================================
# 1. Load Data
# =============================================================================
print("Loading raw datasets...")
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "Sale_Price"
assert (
    target_col in train_raw.columns
), f"Target '{target_col}' must be present in train data."

y_train_all = train_raw[target_col].values
X_train_raw = train_raw.drop(columns=[target_col]).copy()
X_test_raw = test_raw.copy()

print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")


# =============================================================================
# 2. Domain-Specific Feature Engineering & Ordinal Mappings
# =============================================================================
def find_col(df, candidates):
    cols_lower = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        cand_clean = cand.lower().replace("_", "").replace(" ", "")
        if cand_clean in cols_lower:
            return cols_lower[cand_clean]
    return None


def get_col_val(df, candidates, default=0.0):
    col = find_col(df, candidates)
    if col is not None:
        return pd.to_numeric(df[col].astype(object), errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def get_col_str(df, candidates, default="None"):
    col = find_col(df, candidates)
    if col is not None:
        return df[col].astype(object).fillna(default).astype(str).str.strip()
    return pd.Series(default, index=df.index)


# Explicit monotonic quality and condition mappings
QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0, "NA": 0, "Missing": 0, "": 0}
BSMT_EXP_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0, "NA": 0, "Missing": 0, "": 0}
BSMT_FIN_MAP = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0, "NA": 0, "Missing": 0, "": 0}
GARAGE_FIN_MAP = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0, "NA": 0, "Missing": 0, "": 0}
FUNCTIONAL_MAP = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0, "None": 0, "NA": 0, "Missing": 0, "": 0}
LAND_SLOPE_MAP = {"Gtl": 3, "Mod": 2, "Sev": 1, "None": 0, "NA": 0, "Missing": 0, "": 0}
PAVED_DRIVE_MAP = {"Y": 3, "P": 2, "N": 1, "None": 0, "NA": 0, "Missing": 0, "": 0}
CENTRAL_AIR_MAP = {"Y": 1, "N": 0, "None": 0, "NA": 0, "Missing": 0, "": 0}


def engineer_features(df):
    df_feat = df.copy()

    # Explicit Ordinal Feature Mappings
    df_feat["ExterQual_Ord"] = get_col_str(df, ["Exter_Qual", "ExterQual"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["ExterCond_Ord"] = get_col_str(df, ["Exter_Cond", "ExterCond"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["BsmtQual_Ord"] = get_col_str(df, ["Bsmt_Qual", "BsmtQual"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["BsmtCond_Ord"] = get_col_str(df, ["Bsmt_Cond", "BsmtCond"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["HeatingQC_Ord"] = get_col_str(df, ["Heating_QC", "HeatingQC"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["KitchenQual_Ord"] = get_col_str(df, ["Kitchen_Qual", "KitchenQual"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["FireplaceQu_Ord"] = get_col_str(df, ["Fireplace_Qu", "FireplaceQu"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["GarageQual_Ord"] = get_col_str(df, ["Garage_Qual", "GarageQual"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["GarageCond_Ord"] = get_col_str(df, ["Garage_Cond", "GarageCond"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["PoolQC_Ord"] = get_col_str(df, ["Pool_QC", "PoolQC"]).map(QUAL_MAP).fillna(0.0).astype(float)
    df_feat["BsmtExposure_Ord"] = get_col_str(df, ["Bsmt_Exposure", "BsmtExposure"]).map(BSMT_EXP_MAP).fillna(0.0).astype(float)
    df_feat["BsmtFinType1_Ord"] = get_col_str(df, ["BsmtFin_Type_1", "BsmtFinType1"]).map(BSMT_FIN_MAP).fillna(0.0).astype(float)
    df_feat["BsmtFinType2_Ord"] = get_col_str(df, ["BsmtFin_Type_2", "BsmtFinType2"]).map(BSMT_FIN_MAP).fillna(0.0).astype(float)
    df_feat["GarageFinish_Ord"] = get_col_str(df, ["Garage_Finish", "GarageFinish"]).map(GARAGE_FIN_MAP).fillna(0.0).astype(float)
    df_feat["Functional_Ord"] = get_col_str(df, ["Functional"]).map(FUNCTIONAL_MAP).fillna(0.0).astype(float)
    df_feat["LandSlope_Ord"] = get_col_str(df, ["Land_Slope", "LandSlope"]).map(LAND_SLOPE_MAP).fillna(0.0).astype(float)
    df_feat["PavedDrive_Ord"] = get_col_str(df, ["Paved_Drive", "PavedDrive"]).map(PAVED_DRIVE_MAP).fillna(0.0).astype(float)
    df_feat["CentralAir_Ord"] = get_col_str(df, ["Central_Air", "CentralAir"]).map(CENTRAL_AIR_MAP).fillna(0.0).astype(float)

    # Square footage features
    gr_liv_area = get_col_val(df, ["Gr_Liv_Area", "GrLivArea", "Gr_Liv_SF"])
    first_flr_sf = get_col_val(df, ["First_Flr_SF", "1st_Flr_SF", "1stFlrSF"])
    second_flr_sf = get_col_val(df, ["Second_Flr_SF", "2nd_Flr_SF", "2ndFlrSF"])
    total_bsmt_sf = get_col_val(df, ["Total_Bsmt_SF", "TotalBsmtSF"])
    bsmt_fin_sf1 = get_col_val(df, ["BsmtFin_SF_1", "BsmtFinSF1"])
    bsmt_fin_sf2 = get_col_val(df, ["BsmtFin_SF_2", "BsmtFinSF2"])
    bsmt_unf_sf = get_col_val(df, ["Bsmt_Unf_SF", "BsmtUnfSF"])
    garage_area = get_col_val(df, ["Garage_Area", "GarageArea"])
    wood_deck_sf = get_col_val(df, ["Wood_Deck_SF", "WoodDeckSF"])
    open_porch_sf = get_col_val(df, ["Open_Porch_SF", "OpenPorchSF"])
    enclosed_porch = get_col_val(df, ["Enclosed_Porch", "EnclosedPorch"])
    three_ssn_porch = get_col_val(df, ["Three_season_porch", "3Ssn_Porch", "3SsnPorch"])
    screen_porch = get_col_val(df, ["Screen_Porch", "ScreenPorch"])
    pool_area = get_col_val(df, ["Pool_Area", "PoolArea"])
    lot_area = get_col_val(df, ["Lot_Area", "LotArea"])

    # Combined Area metrics
    df_feat["Total_SF"] = gr_liv_area + total_bsmt_sf
    df_feat["Total_Finished_SF"] = gr_liv_area + bsmt_fin_sf1 + bsmt_fin_sf2
    df_feat["Total_Porch_SF"] = (
        wood_deck_sf + open_porch_sf + enclosed_porch + three_ssn_porch + screen_porch
    )
    df_feat["Total_Indoor_SF"] = df_feat["Total_SF"] + garage_area
    df_feat["Lot_to_Living_Ratio"] = lot_area / (gr_liv_area + 1.0)
    df_feat["Bsmt_Fin_Ratio"] = (bsmt_fin_sf1 + bsmt_fin_sf2) / (total_bsmt_sf + 1.0)

    # Bathrooms
    full_bath = get_col_val(df, ["Full_Bath", "FullBath"])
    half_bath = get_col_val(df, ["Half_Bath", "HalfBath"])
    bsmt_full_bath = get_col_val(df, ["Bsmt_Full_Bath", "BsmtFullBath"])
    bsmt_half_bath = get_col_val(df, ["Bsmt_Half_Bath", "BsmtHalfBath"])
    df_feat["Total_Baths"] = (
        full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
    )
    df_feat["Total_Above_Baths"] = full_bath + 0.5 * half_bath

    # Quality and Condition composites
    overall_qual = get_col_val(df, ["Overall_Qual", "OverallQual"], default=5.0)
    overall_cond = get_col_val(df, ["Overall_Cond", "OverallCond"], default=5.0)
    df_feat["Overall_Score"] = overall_qual * overall_cond
    df_feat["Total_Qual_Score"] = (
        overall_qual * 2.0
        + df_feat["ExterQual_Ord"] * 1.5
        + df_feat["KitchenQual_Ord"] * 1.5
        + df_feat["BsmtQual_Ord"]
        + df_feat["GarageQual_Ord"]
        + df_feat["FireplaceQu_Ord"]
        + df_feat["HeatingQC_Ord"]
    )
    df_feat["Total_Cond_Score"] = (
        overall_cond * 2.0
        + df_feat["ExterCond_Ord"]
        + df_feat["BsmtCond_Ord"]
        + df_feat["GarageCond_Ord"]
    )
    df_feat["Overall_Index"] = df_feat["Total_Qual_Score"] * df_feat["Total_Cond_Score"]
    df_feat["Qual_x_LivingArea"] = df_feat["Total_Qual_Score"] * gr_liv_area
    df_feat["Qual_x_TotalSF"] = df_feat["Total_Qual_Score"] * df_feat["Total_SF"]
    df_feat["Fin_Bsmt_Score"] = (
        df_feat["BsmtFinType1_Ord"] * bsmt_fin_sf1 + df_feat["BsmtFinType2_Ord"] * bsmt_fin_sf2
    )

    # Age & Remodel features
    yr_sold = get_col_val(df, ["Yr_Sold", "YrSold", "Year_Sold"], default=2008.0)
    yr_built = get_col_val(df, ["Year_Built", "YearBuilt", "Yr_Built"], default=1970.0)
    yr_remod = get_col_val(
        df, ["Year_Remod_Add", "YearRemodAdd", "Yr_Remod"], default=1970.0
    )
    garage_yr_blt = get_col_val(df, ["Garage_Yr_Blt", "GarageYrBlt"], default=yr_built)

    df_feat["House_Age"] = (yr_sold - yr_built).clip(lower=0)
    df_feat["Remod_Age"] = (yr_sold - yr_remod).clip(lower=0)
    df_feat["Is_Remodeled"] = (yr_remod > yr_built).astype(int)
    df_feat["Is_New_House"] = (yr_sold == yr_built).astype(int)
    df_feat["Garage_Age"] = (yr_sold - garage_yr_blt).clip(lower=0)

    # Rooms breakdown
    tot_rms = get_col_val(df, ["TotRms_AbvGrd", "TotRmsAbvGrd"], default=6.0)
    bedrooms = get_col_val(df, ["Bedroom_AbvGr", "BedroomAbvGr"], default=3.0)
    kitchens = get_col_val(df, ["Kitchen_AbvGr", "KitchenAbvGr"], default=1.0)
    df_feat["Other_Rooms"] = (tot_rms - bedrooms - kitchens).clip(lower=0)
    df_feat["Avg_Room_Size"] = gr_liv_area / (tot_rms + 1e-5)

    # Binary / Presence indicators
    df_feat["Has_Garage"] = (garage_area > 0).astype(int)
    df_feat["Has_Bsmt"] = (total_bsmt_sf > 0).astype(int)
    df_feat["Has_Fireplace"] = (get_col_val(df, ["Fireplaces"]) > 0).astype(int)
    df_feat["Has_Pool"] = (pool_area > 0).astype(int)
    df_feat["Has_2nd_Floor"] = (second_flr_sf > 0).astype(int)
    df_feat["Has_Wood_Deck"] = (wood_deck_sf > 0).astype(int)
    df_feat["Has_Open_Porch"] = (open_porch_sf > 0).astype(int)
    df_feat["Has_Enclosed_Porch"] = (enclosed_porch > 0).astype(int)

    return df_feat


print("Engineering domain features and monotonic ordinal mappings...")
X_train_eng = engineer_features(X_train_raw)
X_test_eng = engineer_features(X_test_raw)

# Encode all remaining categorical/nominal features consistently
cat_cols = list(
    set(X_train_eng.select_dtypes(include=["object", "category"]).columns)
    | set(X_test_eng.select_dtypes(include=["object", "category"]).columns)
)

for col in cat_cols:
    s_tr = (
        X_train_eng[col].astype(object).fillna("Missing").astype(str)
        if col in X_train_eng.columns
        else pd.Series("Missing", index=X_train_eng.index)
    )
    s_te = (
        X_test_eng[col].astype(object).fillna("Missing").astype(str)
        if col in X_test_eng.columns
        else pd.Series("Missing", index=X_test_eng.index)
    )

    all_cats = sorted(list(set(s_tr.unique()) | set(s_te.unique())))
    cat_map = {val: idx for idx, val in enumerate(all_cats)}
    freq_map = s_tr.value_counts(normalize=True).to_dict()

    X_train_eng[f"{col}_Code"] = s_tr.map(cat_map).fillna(-1).astype(float)
    X_test_eng[f"{col}_Code"] = s_te.map(cat_map).fillna(-1).astype(float)

    X_train_eng[f"{col}_Freq"] = s_tr.map(freq_map).fillna(0.0).astype(float)
    X_test_eng[f"{col}_Freq"] = s_te.map(freq_map).fillna(0.0).astype(float)

n_train = len(X_train_eng)
n_test = len(X_test_eng)


# =============================================================================
# 3. Validation Strategy Setup
# =============================================================================
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
y_bins = pd.qcut(y_train_all, q=10, labels=False, duplicates="drop")
fold_indices = np.zeros(n_train, dtype=int)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_eng, y_bins)):
    fold_indices[val_idx] = fold

# =============================================================================
# 4. Multi-Architecture Model Training and Cross-Validation
# =============================================================================
model_names = ["lgb", "xgb", "ridge"]
if HAS_CATBOOST:
    model_names.append("catboost")

oof_preds_dict = {name: np.zeros(n_train, dtype=np.float64) for name in model_names}
test_preds_dict = {name: np.zeros(n_test, dtype=np.float64) for name in model_names}

print("\nStarting Cross-Validation Training Loop...")
for fold in range(n_splits):
    val_mask = fold_indices == fold
    trn_mask = ~val_mask

    num_cols = X_train_eng.select_dtypes(include=[np.number]).columns
    X_tr = X_train_eng.iloc[trn_mask][num_cols].fillna(0.0).values
    y_tr = y_train_all[trn_mask]
    X_va = X_train_eng.iloc[val_mask][num_cols].fillna(0.0).values
    y_va = y_train_all[val_mask]
    X_te = X_test_eng[num_cols].fillna(0.0).values

    # Model 1: LightGBM
    lgb_clf = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
        importance_type="gain",
    )
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=60, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=lgb_callbacks,
    )
    oof_preds_dict["lgb"][val_mask] = lgb_clf.predict(X_va)
    test_preds_dict["lgb"] += lgb_clf.predict(X_te) / n_splits

    # Model 2: XGBoost
    xgb_clf = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=60,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_dict["xgb"][val_mask] = xgb_clf.predict(X_va)
    test_preds_dict["xgb"] += xgb_clf.predict(X_te) / n_splits

    # Model 3: CatBoost (if installed)
    if HAS_CATBOOST:
        cb_clf = cb.CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            l2_leaf_reg=3.0,
            early_stopping_rounds=60,
            random_seed=RANDOM_SEED + fold,
            verbose=False,
            thread_count=-1,
        )
        cb_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds_dict["catboost"][val_mask] = cb_clf.predict(X_va)
        test_preds_dict["catboost"] += cb_clf.predict(X_te) / n_splits

    # Model 4: Ridge Linear Regression
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    ridge_clf = Ridge(alpha=25.0, random_state=RANDOM_SEED + fold)
    ridge_clf.fit(X_tr_sc, y_tr)
    oof_preds_dict["ridge"][val_mask] = ridge_clf.predict(X_va_sc)
    test_preds_dict["ridge"] += ridge_clf.predict(X_te_sc) / n_splits

    lgb_rmse = compute_rmse(y_va, oof_preds_dict["lgb"][val_mask])
    xgb_rmse = compute_rmse(y_va, oof_preds_dict["xgb"][val_mask])
    print(
        f"Fold {fold + 1}/{n_splits} | LGBM RMSE: {lgb_rmse:.2f} | XGB RMSE: {xgb_rmse:.2f}"
    )

# Individual model scores
print("\n--- Out-of-Fold Performance per Architecture ---")
for name in model_names:
    score = compute_rmse(y_train_all, oof_preds_dict[name])
    print(f"Model [{name.upper():8s}] Overall OOF RMSE: {score:.4f}")

# =============================================================================
# 5. Constrained SLSQP Ensemble Blending
# =============================================================================
oof_matrix = np.column_stack([oof_preds_dict[name] for name in model_names])
test_matrix = np.column_stack([test_preds_dict[name] for name in model_names])
n_models = len(model_names)


def ensemble_loss(weights: np.ndarray) -> float:
    w = np.array(weights)
    pred = oof_matrix @ w
    return compute_rmse(y_train_all, pred)


initial_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-9},
)

optimal_weights = opt_res.x / np.sum(opt_res.x)
print("\n--- Optimal Ensemble Weights ---")
for name, weight in zip(model_names, optimal_weights):
    print(f"{name.upper():8s} Weight: {weight:.4f}")

oof_ensemble = oof_matrix @ optimal_weights
test_ensemble = test_matrix @ optimal_weights

# Lower bound sanity clipping
min_positive_price = y_train_all.min() * 0.5
oof_ensemble = np.clip(oof_ensemble, a_min=min_positive_price, a_max=None)
test_ensemble = np.clip(test_ensemble, a_min=min_positive_price, a_max=None)

final_validation_score = compute_rmse(y_train_all, oof_ensemble)

# =============================================================================
# 6. Generate Final Submission
# =============================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_ensemble), dtype=int), "Sale_Price": test_ensemble}
)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"\nSubmission generated successfully at '{submission_path}' with shape {submission_df.shape}."
)

# Mandatory final score output
print(f"Final Validation Score: {final_validation_score}")