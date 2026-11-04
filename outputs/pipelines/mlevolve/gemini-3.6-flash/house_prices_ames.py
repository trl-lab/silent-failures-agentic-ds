import os
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# Ensure required output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)


def get_col(df, candidates, default=None):
    """Helper to retrieve column present in dataframe regardless of casing/formatting."""
    for c in candidates:
        if c in df.columns:
            return df[c]
        c_norm = c.lower().replace("_", "").replace(" ", "")
        for col in df.columns:
            if col.lower().replace("_", "").replace(" ", "") == c_norm:
                return df[col]
    if default is not None:
        return pd.Series(default, index=df.index)
    return pd.Series(0, index=df.index)


def get_col_name(df, candidates):
    """Utility function to identify column name in DataFrame regardless of casing/formatting."""
    for c in candidates:
        if c in df.columns:
            return c
        c_norm = c.lower().replace("_", "").replace(" ", "")
        for col in df.columns:
            if col.lower().replace("_", "").replace(" ", "") == c_norm:
                return col
    return None


def engineer_base_features(df):
    """Extract domain specific numerical, ordinal, and structural features."""
    df_feat = pd.DataFrame(index=df.index)

    def get_num(candidates, default=0.0):
        s = get_col(df, candidates)
        return pd.to_numeric(s.astype(str), errors="coerce").fillna(default)

    # Fetch core numericals safely
    gr_liv_area = get_num(["Gr_Liv_Area", "GrLivArea", "Gr_Liv_SF"], 0)
    total_bsmt_sf = get_num(
        ["Total_Bsmt_SF", "TotalBsmtSF", "Total_Bsmt_Area"], 0
    )
    flr1_sf = get_num(
        ["1st_Flr_SF", "First_Flr_SF", "1stFlrSF", "Flr_SF_1"], 0
    )
    flr2_sf = get_num(
        ["2nd_Flr_SF", "Second_Flr_SF", "2ndFlrSF", "Flr_SF_2"], 0
    )
    low_qual_sf = get_num(["Low_Qual_Fin_SF", "LowQualFinSF"], 0)

    full_bath = get_num(["Full_Bath", "FullBath"], 0)
    half_bath = get_num(["Half_Bath", "HalfBath"], 0)
    bsmt_full_bath = get_num(["Bsmt_Full_Bath", "BsmtFullBath"], 0)
    bsmt_half_bath = get_num(["Bsmt_Half_Bath", "BsmtHalfBath"], 0)

    yr_built = get_num(["Year_Built", "Yr_Built", "YearBuilt", "YrBuilt"], 1970)
    yr_remod = get_num(
        ["Year_Remod_Add", "Yr_Remod_Add", "YearRemodAdd", "YrRemodAdd"], 1970
    )
    yr_sold = get_num(["Yr_Sold", "Year_Sold", "YrSold", "YearSold"], 2008)
    mo_sold = get_num(["Mo_Sold", "Month_Sold", "MoSold"], 6)

    overall_qual = get_num(["Overall_Qual", "OverallQual"], 5)
    overall_cond = get_num(["Overall_Cond", "OverallCond"], 5)

    garage_cars = get_num(["Garage_Cars", "GarageCars"], 0)
    garage_area = get_num(["Garage_Area", "GarageArea"], 0)
    garage_yr = get_num(["Garage_Yr_Blt", "GarageYrBlt"], 1970)

    lot_area = get_num(["Lot_Area", "LotArea"], 0)
    lot_frontage = get_num(["Lot_Frontage", "LotFrontage"], 0)

    op_porch = get_num(["Open_Porch_SF", "OpenPorchSF"], 0)
    enc_porch = get_num(["Enclosed_Porch", "EnclosedPorch"], 0)
    ssn_porch = get_num(["3Ssn_Porch", "Three_Ssn_Porch", "3SsnPorch"], 0)
    scr_porch = get_num(["Screen_Porch", "ScreenPorch"], 0)
    wood_deck = get_num(["Wood_Deck_SF", "WoodDeckSF"], 0)
    pool_area = get_num(["Pool_Area", "PoolArea"], 0)

    bedrooms = get_num(["Bedroom_AbvGr", "Bedroom"], 0)
    kitchens = get_num(["Kitchen_AbvGr", "Kitchen"], 0)
    tot_rooms = get_num(["TotRms_AbvGrd", "TotRmsAbvGrd"], 0)
    fireplaces = get_num(["Fireplaces"], 0)
    bsmt_unf_sf = get_num(["Bsmt_Unf_SF", "BsmtUnfSF"], 0)
    mas_vnr_area = get_num(["Mas_Vnr_Area", "MasVnrArea"], 0)

    # Composite Square Footage Features
    df_feat["Total_SF"] = gr_liv_area + total_bsmt_sf
    df_feat["Total_Above_SF"] = flr1_sf + flr2_sf + low_qual_sf
    df_feat["Total_Porch_SF"] = op_porch + enc_porch + ssn_porch + scr_porch + wood_deck
    df_feat["Total_Bathrooms"] = (
        full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
    )

    # Raw numerical copies
    df_feat["Gr_Liv_Area"] = gr_liv_area
    df_feat["Total_Bsmt_SF"] = total_bsmt_sf
    df_feat["1st_Flr_SF"] = flr1_sf
    df_feat["2nd_Flr_SF"] = flr2_sf
    df_feat["Lot_Area"] = lot_area
    df_feat["Lot_Frontage"] = lot_frontage
    df_feat["Garage_Area"] = garage_area
    df_feat["Garage_Cars"] = garage_cars
    df_feat["Overall_Qual"] = overall_qual
    df_feat["Overall_Cond"] = overall_cond
    df_feat["TotRms_AbvGrd"] = tot_rooms
    df_feat["Bedroom_AbvGr"] = bedrooms
    df_feat["Kitchen_AbvGr"] = kitchens
    df_feat["Fireplaces"] = fireplaces
    df_feat["Mas_Vnr_Area"] = mas_vnr_area

    # Age & Time Features
    df_feat["House_Age"] = (yr_sold - yr_built).clip(lower=0)
    df_feat["Remod_Age"] = (yr_sold - yr_remod).clip(lower=0)
    df_feat["Is_Remodeled"] = (yr_remod > yr_built).astype(int)
    df_feat["Garage_Age"] = (yr_sold - garage_yr).clip(lower=0)
    df_feat["Mo_Sold"] = mo_sold
    df_feat["Yr_Sold"] = yr_sold
    df_feat["Sin_Mo_Sold"] = np.sin(2 * np.pi * mo_sold / 12)
    df_feat["Cos_Mo_Sold"] = np.cos(2 * np.pi * mo_sold / 12)

    # Interaction Features
    df_feat["Qual_x_TotalSF"] = overall_qual * df_feat["Total_SF"]
    df_feat["Qual_x_GrLiv"] = overall_qual * gr_liv_area
    df_feat["Qual_x_1stFlr"] = overall_qual * flr1_sf
    df_feat["Qual_x_Bsmt"] = overall_qual * total_bsmt_sf
    df_feat["Qual_x_Garage"] = overall_qual * garage_area
    df_feat["Cond_x_TotalSF"] = overall_cond * df_feat["Total_SF"]
    df_feat["Qual_x_Cond"] = overall_qual * overall_cond
    df_feat["Bath_per_Room"] = df_feat["Total_Bathrooms"] / (tot_rooms + 1.0)
    df_feat["LivSF_per_Room"] = gr_liv_area / (tot_rooms + 1.0)
    df_feat["Bed_per_Room"] = bedrooms / (tot_rooms + 1.0)
    df_feat["Bsmt_Unf_Ratio"] = bsmt_unf_sf / (total_bsmt_sf + 1.0)

    # Neighborhood micro-market relative features
    neigh_s = get_col(df, ["Neighborhood", "Neighbourhood"], default="NA").astype(str)
    for col_name, ser in [("Gr_Liv_Area", gr_liv_area), ("Overall_Qual", overall_qual), ("Total_SF", df_feat["Total_SF"])]:
        med = ser.groupby(neigh_s).transform("median")
        glob_med = ser.median()
        med = med.fillna(glob_med).replace(0, glob_med)
        df_feat[f"{col_name}_Neigh_Ratio"] = ser / med

    # Presence Binary Flags
    df_feat["Has_Bsmt"] = (total_bsmt_sf > 0).astype(int)
    df_feat["Has_Garage"] = (garage_area > 0).astype(int)
    df_feat["Has_Fireplace"] = (fireplaces > 0).astype(int)
    df_feat["Has_Pool"] = (pool_area > 0).astype(int)
    df_feat["Has_Porch"] = (df_feat["Total_Porch_SF"] > 0).astype(int)
    df_feat["Has_2ndFlr"] = (flr2_sf > 0).astype(int)
    df_feat["Has_MasVnr"] = (mas_vnr_area > 0).astype(int)

    # Ordinal Mappings for Quality Ratings
    qual_dict = {
        "Ex": 5,
        "Gd": 4,
        "TA": 3,
        "Fa": 2,
        "Po": 1,
        "None": 0,
        "NA": 0,
        np.nan: 0,
    }
    bsmt_exp_dict = {
        "Gd": 4,
        "Av": 3,
        "Mn": 2,
        "No": 1,
        "None": 0,
        "NA": 0,
        np.nan: 0,
    }
    bsmt_fin_dict = {
        "GLQ": 6,
        "ALQ": 5,
        "BLQ": 4,
        "Rec": 3,
        "LwQ": 2,
        "Unf": 1,
        "None": 0,
        "NA": 0,
        np.nan: 0,
    }

    for col_alias, map_dict in [
        (["Exter_Qual", "ExterQual"], qual_dict),
        (["Exter_Cond", "ExterCond"], qual_dict),
        (["Bsmt_Qual", "BsmtQual"], qual_dict),
        (["Bsmt_Cond", "BsmtCond"], qual_dict),
        (["Heating_QC", "HeatingQC"], qual_dict),
        (["Kitchen_Qual", "KitchenQual"], qual_dict),
        (["Fireplace_Qu", "FireplaceQu"], qual_dict),
        (["Garage_Qual", "GarageQual"], qual_dict),
        (["Garage_Cond", "GarageCond"], qual_dict),
        (["Pool_QC", "PoolQC"], qual_dict),
        (["Bsmt_Exposure", "BsmtExposure"], bsmt_exp_dict),
        (["BsmtFin_Type_1", "BsmtFinType1"], bsmt_fin_dict),
    ]:
        s = get_col(df, col_alias, default="None").astype(str)
        df_feat[col_alias[0] + "_ord"] = s.map(map_dict).fillna(0)

    return df_feat


def run_pipeline():
    # Load Raw Datasets
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    target_col = get_col_name(train_raw, ["Sale_Price", "SalePrice", "target"])
    if target_col is None:
        raise ValueError("Target column not found in training data.")

    # Remove extreme outliers based on standard domain literature
    gr_liv_col = get_col_name(train_raw, ["Gr_Liv_Area", "GrLivArea", "Gr_Liv_SF"])
    if gr_liv_col is not None:
        gr_liv_vals = pd.to_numeric(train_raw[gr_liv_col].astype(str), errors="coerce").fillna(0)
        target_vals = pd.to_numeric(train_raw[target_col].astype(str), errors="coerce").fillna(0)
        mask = ~((gr_liv_vals > 4000) & (target_vals < 300000))
        train_raw = train_raw[mask].reset_index(drop=True)

    y_train_raw = pd.to_numeric(train_raw[target_col].astype(str), errors="coerce").values
    y_train_log = np.log1p(y_train_raw)

    # Domain Feature Extraction
    X_train_df = engineer_base_features(train_raw)
    X_test_df = engineer_base_features(test_raw)

    # Categorical Frequency Encodings
    cat_cols = [
        col
        for col in train_raw.columns
        if col != target_col
        and (
            train_raw[col].dtype == "object" or str(train_raw[col].dtype) == "category"
        )
    ]
    for col in cat_cols:
        s_tr = train_raw[col].astype(str)
        s_te = test_raw[col].astype(str)
        freq_map = s_tr.value_counts(normalize=True).to_dict()
        X_train_df[f"{col}_freq"] = s_tr.map(freq_map).fillna(0.0).astype(np.float32)
        X_test_df[f"{col}_freq"] = s_te.map(freq_map).fillna(0.0).astype(np.float32)

    # Clean String Encodings across Train & Test
    for col in X_train_df.columns:
        if (
            X_train_df[col].dtype == "object"
            or str(X_train_df[col].dtype) == "category"
        ):
            s_tr = X_train_df[col].astype(str)
            s_te = X_test_df[col].astype(str)
            cat_map = {val: idx for idx, val in enumerate(s_tr.unique())}
            X_train_df[col] = s_tr.map(cat_map).fillna(-1).astype(int)
            X_test_df[col] = s_te.map(cat_map).fillna(-1).astype(int)

    # Cross-Validation Configuration
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_train = len(X_train_df)
    n_test = len(X_test_df)

    num_models = 5
    oof_preds_matrix = np.zeros((n_train, num_models))
    test_preds_matrix = np.zeros((n_test, num_models))

    te_cols = [col for col in cat_cols if col in train_raw.columns]

    # 5-Fold Ensemble Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
        X_tr = X_train_df.iloc[train_idx].copy()
        y_tr_log = y_train_log[train_idx]
        X_va = X_train_df.iloc[val_idx].copy()
        X_te = X_test_df.copy()

        # In-Fold Target Encoding
        global_mean = np.mean(y_tr_log)
        for col in te_cols:
            tr_cat = train_raw.iloc[train_idx][col].astype(str)
            va_cat = train_raw.iloc[val_idx][col].astype(str)
            te_cat = test_raw[col].astype(str) if col in test_raw.columns else pd.Series("NA", index=test_raw.index).astype(str)

            te_df = pd.DataFrame({"cat": tr_cat, "target": y_tr_log})
            te_map = te_df.groupby("cat", observed=False)["target"].agg(
                ["count", "mean"]
            )
            smooth = 10.0
            te_smooth = (te_map["count"] * te_map["mean"] + smooth * global_mean) / (
                te_map["count"] + smooth
            )
            te_dict = te_smooth.to_dict()

            X_tr[f"{col}_te"] = tr_cat.map(te_dict).fillna(global_mean).values
            X_va[f"{col}_te"] = va_cat.map(te_dict).fillna(global_mean).values
            X_te[f"{col}_te"] = te_cat.map(te_dict).fillna(global_mean).values

        X_tr_num = X_tr.values.astype(np.float32)
        X_va_num = X_va.values.astype(np.float32)
        X_te_num = X_te.values.astype(np.float32)

        # Model 1: LightGBM Regressor (L2 Loss)
        m1 = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1,
        )
        m1.fit(
            X_tr_num,
            y_tr_log,
            eval_set=[(X_va_num, y_train_log[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_preds_matrix[val_idx, 0] = m1.predict(X_va_num)
        test_preds_matrix[:, 0] += m1.predict(X_te_num) / n_splits
        joblib.dump(m1, f"./working/models/lgbm_l2_fold{fold}.joblib")

        # Model 2: LightGBM Regressor (Huber Loss)
        m2 = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.9,
            n_estimators=1500,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=15,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=142 + fold,
            verbosity=-1,
            n_jobs=-1,
        )
        m2.fit(
            X_tr_num,
            y_tr_log,
            eval_set=[(X_va_num, y_train_log[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_preds_matrix[val_idx, 1] = m2.predict(X_va_num)
        test_preds_matrix[:, 1] += m2.predict(X_te_num) / n_splits
        joblib.dump(m2, f"./working/models/lgbm_huber_fold{fold}.joblib")

        # Model 3: XGBoost Regressor
        m3 = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=242 + fold,
            early_stopping_rounds=50,
            eval_metric="rmse",
            n_jobs=-1,
        )
        m3.fit(
            X_tr_num,
            y_tr_log,
            eval_set=[(X_va_num, y_train_log[val_idx])],
            verbose=False,
        )
        oof_preds_matrix[val_idx, 2] = m3.predict(X_va_num)
        test_preds_matrix[:, 2] += m3.predict(X_te_num) / n_splits
        joblib.dump(m3, f"./working/models/xgboost_fold{fold}.joblib")

        # Model 4: CatBoost or ExtraTrees Regressor
        if HAS_CATBOOST:
            m4 = CatBoostRegressor(
                iterations=1500,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=342 + fold,
                early_stopping_rounds=50,
                verbose=0,
            )
            m4.fit(
                X_tr_num,
                y_tr_log,
                eval_set=(X_va_num, y_train_log[val_idx]),
                verbose=False,
            )
            oof_preds_matrix[val_idx, 3] = m4.predict(X_va_num)
            test_preds_matrix[:, 3] += m4.predict(X_te_num) / n_splits
            joblib.dump(m4, f"./working/models/catboost_fold{fold}.joblib")
        else:
            m4 = make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=4,
                    random_state=342 + fold,
                    n_jobs=-1,
                ),
            )
            m4.fit(X_tr_num, y_tr_log)
            oof_preds_matrix[val_idx, 3] = m4.predict(X_va_num)
            test_preds_matrix[:, 3] += m4.predict(X_te_num) / n_splits
            joblib.dump(m4, f"./working/models/extratrees_fold{fold}.joblib")

        # Model 5: Robust Scaled Regularized Ridge Pipeline
        m5 = make_pipeline(
            SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=20.0)
        )
        m5.fit(X_tr_num, y_tr_log)
        oof_preds_matrix[val_idx, 4] = m5.predict(X_va_num)
        test_preds_matrix[:, 4] += m5.predict(X_te_num) / n_splits
        joblib.dump(m5, f"./working/models/ridge_fold{fold}.joblib")

    # Log-Normal Variance Bias Correction and Post-Processing Scale Optimization on OOF Predictions
    oof_raw_matrix = np.zeros_like(oof_preds_matrix)
    test_raw_matrix = np.zeros_like(test_preds_matrix)

    for m in range(num_models):
        resids = y_train_log - oof_preds_matrix[:, m]
        sigma2 = np.var(resids)
        oof_raw_m = np.expm1(oof_preds_matrix[:, m] + 0.5 * sigma2)
        test_raw_m = np.expm1(test_preds_matrix[:, m] + 0.5 * sigma2)

        # Scale optimization
        scale = np.sum(y_train_raw * oof_raw_m) / (np.sum(oof_raw_m ** 2) + 1e-8)
        oof_raw_matrix[:, m] = oof_raw_m * scale
        test_raw_matrix[:, m] = test_raw_m * scale

    # SLSQP Non-Negative Weight Optimization directly targeting Raw RMSE
    def raw_rmse_objective(weights):
        weights = np.maximum(0, weights)
        s = np.sum(weights)
        if s > 0:
            weights = weights / s
        pred_raw = oof_raw_matrix @ weights
        return np.sqrt(mean_squared_error(y_train_raw, pred_raw))

    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        raw_rmse_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_weights = np.maximum(0, res.x)
    opt_weights /= np.sum(opt_weights)

    # Out-Of-Fold Final Predictions & Performance
    oof_final_raw = oof_raw_matrix @ opt_weights
    final_val_score = np.sqrt(mean_squared_error(y_train_raw, oof_final_raw))

    # Generate Test Predictions conforming to specification
    test_final_raw = test_raw_matrix @ opt_weights
    test_final_raw = np.clip(test_final_raw, a_min=10000, a_max=None)

    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_final_raw)), "Sale_Price": test_final_raw}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    return final_val_score


if __name__ == "__main__":
    final_score = run_pipeline()
    print(f"Final Validation Score: {final_score}")
