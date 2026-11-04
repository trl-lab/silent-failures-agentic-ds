import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Sale_Price"
y = train_df[target_col].values

# Extract IDs
if "id" in train_df.columns:
    train_df = train_df.drop(columns=["id"])
if "id" in test_df.columns:
    test_id = test_df["id"].values
    test_df = test_df.drop(columns=["id"])
else:
    test_id = np.arange(len(test_df))

X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()


# Enhanced Domain Feature Engineering
def safe_num(df, col_name):
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index)


def add_domain_features(df):
    df = df.copy()

    # Square Footage Metrics
    gr_liv = safe_num(df, "Gr_Liv_Area")
    total_bsmt = safe_num(df, "Total_Bsmt_SF")
    bsmt_fin1 = safe_num(df, "BsmtFin_SF_1")
    bsmt_fin2 = safe_num(df, "BsmtFin_SF_2")
    garage_area = safe_num(df, "Garage_Area")

    df["Total_SF"] = gr_liv + total_bsmt
    df["Bsmt_Finished_SF"] = bsmt_fin1 + bsmt_fin2
    df["Total_Finished_SF"] = gr_liv + df["Bsmt_Finished_SF"]
    df["Total_Space_SF"] = df["Total_SF"] + garage_area

    # Bathroom Count
    df["Total_Baths"] = (
        safe_num(df, "Full_Bath")
        + 0.5 * safe_num(df, "Half_Bath")
        + safe_num(df, "Bsmt_Full_Bath")
        + 0.5 * safe_num(df, "Bsmt_Half_Bath")
    )

    # Age and Remodel Metrics
    year_sold = (
        safe_num(df, "Year_Sold")
        if "Year_Sold" in df.columns
        else safe_num(df, "Yr_Sold")
    )
    year_built = safe_num(df, "Year_Built")
    year_remod = safe_num(df, "Year_Remod_Add")

    df["House_Age"] = year_sold - year_built
    df["Remod_Age"] = year_sold - year_remod
    df["Is_Remodeled"] = (year_remod != year_built).astype(int)

    # Aggregate Porch Area
    df["Total_Porch_SF"] = (
        safe_num(df, "Wood_Deck_SF")
        + safe_num(df, "Open_Porch_SF")
        + safe_num(df, "Enclosed_Porch")
        + safe_num(df, "3Ssn_Porch")
        + safe_num(df, "Screen_Porch")
    )

    # Expanded Ordinal Quality Mappings
    qual_cols = [
        "Exter_Qual",
        "Exter_Cond",
        "Bsmt_Qual",
        "Bsmt_Cond",
        "Heating_QC",
        "Kitchen_Qual",
        "Fireplace_Qu",
        "Garage_Qual",
        "Garage_Cond",
        "Pool_QC",
    ]
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    for col in qual_cols:
        if col in df.columns:
            df[col + "_num"] = df[col].astype(str).map(qual_map).fillna(0).astype(float)

    if "Bsmt_Exposure" in df.columns:
        bsmt_exp_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
        df["Bsmt_Exposure_num"] = (
            df["Bsmt_Exposure"].astype(str).map(bsmt_exp_map).fillna(0).astype(float)
        )

    # Additional Ordinals
    fin_map = {
        "GLQ": 6,
        "ALQ": 5,
        "BLQ": 4,
        "Rec": 3,
        "LwQ": 2,
        "Unf": 1,
        "None": 0,
        "NA": 0,
    }
    if "BsmtFin_Type_1" in df.columns:
        df["BsmtFin_Type_1_num"] = (
            df["BsmtFin_Type_1"].astype(str).map(fin_map).fillna(0).astype(float)
        )

    gar_fin_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0, "NA": 0}
    if "Garage_Finish" in df.columns:
        df["Garage_Finish_num"] = (
            df["Garage_Finish"].astype(str).map(gar_fin_map).fillna(0).astype(float)
        )

    funct_map = {
        "Typ": 8,
        "Min1": 7,
        "Min2": 6,
        "Mod": 5,
        "Maj1": 4,
        "Maj2": 3,
        "Sev": 2,
        "Sal": 1,
    }
    if "Functional" in df.columns:
        df["Functional_num"] = (
            df["Functional"].astype(str).map(funct_map).fillna(7).astype(float)
        )

    # Overall Quality interactions
    qual_str_map = {
        "Very_Excellent": 10,
        "Excellent": 9,
        "Very_Good": 8,
        "Good": 7,
        "Above_Average": 6,
        "Average": 5,
        "Below_Average": 4,
        "Fair": 3,
        "Poor": 2,
        "Very_Poor": 1,
    }
    if "Overall_Qual" in df.columns:
        overall_q = pd.to_numeric(df["Overall_Qual"], errors="coerce")
        if overall_q.isna().sum() > len(df) * 0.5:
            overall_q = df["Overall_Qual"].astype(str).map(qual_str_map).fillna(5)
        df["Overall_Qual_Num"] = overall_q.fillna(5)
        df["Qual_SF"] = df["Overall_Qual_Num"] * df["Total_SF"]
        df["Qual_Finished_SF"] = df["Overall_Qual_Num"] * df["Total_Finished_SF"]

    return df


X = add_domain_features(X)
X_test = add_domain_features(X_test)


# Add Neighborhood-Relative Benchmark Features
def add_neighborhood_relative_features(df_tr, df_te):
    combined = pd.concat([df_tr, df_te], axis=0, ignore_index=True)

    if "Neighborhood" in combined.columns:
        neigh = combined["Neighborhood"].astype(str)

        gr_liv = safe_num(combined, "Gr_Liv_Area")
        total_sf = safe_num(combined, "Total_SF")
        total_fin = safe_num(combined, "Total_Finished_SF")
        overall_qual = safe_num(combined, "Overall_Qual_Num")

        neigh_gr_liv_mean = combined.groupby(neigh)["Gr_Liv_Area"].transform("mean")
        neigh_total_sf_mean = combined.groupby(neigh)["Total_SF"].transform("mean")
        neigh_total_fin_mean = combined.groupby(neigh)["Total_Finished_SF"].transform(
            "mean"
        )
        neigh_qual_mean = combined.groupby(neigh)["Overall_Qual_Num"].transform("mean")

        combined["GrLivArea_to_Neigh_Mean"] = gr_liv / (neigh_gr_liv_mean + 1e-5)
        combined["TotalSF_to_Neigh_Mean"] = total_sf / (neigh_total_sf_mean + 1e-5)
        combined["TotalFinSF_to_Neigh_Mean"] = total_fin / (neigh_total_fin_mean + 1e-5)
        combined["Qual_to_Neigh_Mean"] = overall_qual / (neigh_qual_mean + 1e-5)

        combined["GrLivArea_diff_Neigh_Mean"] = gr_liv - neigh_gr_liv_mean
        combined["TotalSF_diff_Neigh_Mean"] = total_sf - neigh_total_sf_mean

    n_train = len(df_tr)
    return (
        combined.iloc[:n_train].reset_index(drop=True),
        combined.iloc[n_train:].reset_index(drop=True),
    )


X, X_test = add_neighborhood_relative_features(X, X_test)

# Categorical handling setup
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Prepare datasets for categorical splits
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    categories = pd.concat([X[c], X_test[c]], axis=0).astype(str).unique()
    X_lgb[c] = pd.Categorical(X_lgb[c].astype(str), categories=categories)
    X_test_lgb[c] = pd.Categorical(X_test_lgb[c].astype(str), categories=categories)

X_cb = X.copy()
X_test_cb = X_test.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str).fillna("Missing")
    X_test_cb[c] = X_test_cb[c].astype(str).fillna("Missing")

# Target Encoding columns setup
te_candidate_cols = [
    "Neighborhood",
    "MSSubClass",
    "MS_Zoning",
    "Bldg_Type",
    "House_Style",
    "Exterior_1st",
]
target_enc_cols = [c for c in te_candidate_cols if c in X.columns]

# 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb_preds = np.zeros(len(test_df))
test_cb_preds = np.zeros(len(test_df))
test_xgb_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    y_tr, y_va = y[train_idx], y[val_idx]

    # Out-Of-Fold Target Encoding
    X_tr_lgb, X_va_lgb = (
        X_lgb.iloc[train_idx].copy(),
        X_lgb.iloc[val_idx].copy(),
    )
    X_te_lgb_fold = X_test_lgb.copy()

    X_tr_cb, X_va_cb = X_cb.iloc[train_idx].copy(), X_cb.iloc[val_idx].copy()
    X_te_cb_fold = X_test_cb.copy()

    for col in target_enc_cols:
        tr_cat = X.iloc[train_idx][col].astype(str)
        va_cat = X.iloc[val_idx][col].astype(str)
        te_cat = X_test[col].astype(str)

        global_mean = y_tr.mean()
        stats = (
            pd.DataFrame({"cat": tr_cat, "y": y_tr})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        m = 10.0
        smooth_map = (
            (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
        ).to_dict()

        te_feature_name = col + "_TE"
        X_tr_lgb[te_feature_name] = (
            tr_cat.map(smooth_map).fillna(global_mean).astype(float)
        )
        X_va_lgb[te_feature_name] = (
            va_cat.map(smooth_map).fillna(global_mean).astype(float)
        )
        X_te_lgb_fold[te_feature_name] = (
            te_cat.map(smooth_map).fillna(global_mean).astype(float)
        )

        X_tr_cb[te_feature_name] = X_tr_lgb[te_feature_name]
        X_va_cb[te_feature_name] = X_va_lgb[te_feature_name]
        X_te_cb_fold[te_feature_name] = X_te_lgb_fold[te_feature_name]

    # --- 1. LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=15,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_lgb)
    test_lgb_preds += lgb_model.predict(X_te_lgb_fold) / kf.n_splits

    # --- 2. CatBoost ---
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        subsample=0.8,
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = cb_model.predict(X_va_cb)
    test_cb_preds += cb_model.predict(X_te_cb_fold) / kf.n_splits

    # --- 3. XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_va_lgb)
    test_xgb_preds += xgb_model.predict(X_te_lgb_fold) / kf.n_splits


# Optimize Blending Weights on OOF predictions
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
    return np.sqrt(mean_squared_error(y, pred))


res = minimize(
    loss_func,
    x0=[0.4, 0.4, 0.2],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_lgb, w_cb, w_xgb = res.x
print(
    f"Optimal Weights -> LightGBM: {w_lgb:.4f}, CatBoost: {w_cb:.4f}, XGBoost: {w_xgb:.4f}"
)

# Compute final Out-Of-Fold RMSE
oof_preds = w_lgb * oof_lgb + w_cb * oof_cb + w_xgb * oof_xgb
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation OOF RMSE: {oof_rmse:.4f}")

# Compute final Test Predictions and Save
test_preds = w_lgb * test_lgb_preds + w_cb * test_cb_preds + w_xgb * test_xgb_preds

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_id, "Sale_Price": test_preds})
submission.to_csv("./working/submission.csv", index=False)
