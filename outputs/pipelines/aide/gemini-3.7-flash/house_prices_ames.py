import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Sale_Price"
y = train_df[target_col].values.astype(np.float64)
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Quality mapping for ordinal categories in make_ames
qual_map = {
    "Very_Poor": 1,
    "Poor": 2,
    "Fair": 3,
    "Below_Average": 4,
    "Average": 5,
    "Above_Average": 6,
    "Good": 7,
    "Very_Good": 8,
    "Excellent": 9,
    "Very_Excellent": 10,
    "None": 0,
    "No_Basement": 0,
    "No_Garage": 0,
    "No_Fireplace": 0,
    "Typical": 5,
    "Ex": 9,
    "Gd": 7,
    "TA": 5,
    "Fa": 3,
    "Po": 1,
}


def map_ordinal(series):
    if series.dtype == "object" or series.dtype.name == "category":
        mapped = series.astype(str).map(qual_map)
        numeric = pd.to_numeric(series, errors="coerce")
        return mapped.combine_first(numeric).fillna(5.0)
    return pd.to_numeric(series, errors="coerce").fillna(5.0)


def engineer_features(df):
    data = df.copy()

    # Living and basement square footage
    gr_liv = pd.to_numeric(data.get("Gr_Liv_Area", 0), errors="coerce").fillna(0)
    total_bsmt = pd.to_numeric(data.get("Total_Bsmt_SF", 0), errors="coerce").fillna(0)
    first_flr = pd.to_numeric(
        data.get(
            "First_Flr_SF",
            data.get("1st_Flr_SF", 0),
        ),
        errors="coerce",
    ).fillna(0)
    second_flr = pd.to_numeric(
        data.get(
            "Second_Flr_SF",
            data.get("2nd_Flr_SF", 0),
        ),
        errors="coerce",
    ).fillna(0)
    garage_area = pd.to_numeric(data.get("Garage_Area", 0), errors="coerce").fillna(0)

    data["Total_SF"] = gr_liv + total_bsmt
    data["Total_Living_SF"] = first_flr + second_flr + total_bsmt

    # Age features
    yr_sold = pd.to_numeric(
        data.get("Yr_Sold", data.get("Year_Sold", 2010)), errors="coerce"
    ).fillna(2010)
    yr_built = pd.to_numeric(data.get("Year_Built", 1970), errors="coerce").fillna(1970)
    yr_remod = pd.to_numeric(
        data.get("Year_Remod_Add", yr_built), errors="coerce"
    ).fillna(yr_built)

    data["House_Age"] = (yr_sold - yr_built).clip(lower=0)
    data["Remodel_Age"] = (yr_sold - yr_remod).clip(lower=0)
    data["Is_Remodeled"] = (yr_remod != yr_built).astype(int)

    # Bathrooms
    full_bath = pd.to_numeric(data.get("Full_Bath", 0), errors="coerce").fillna(0)
    half_bath = pd.to_numeric(data.get("Half_Bath", 0), errors="coerce").fillna(0)
    bsmt_full = pd.to_numeric(data.get("Bsmt_Full_Bath", 0), errors="coerce").fillna(0)
    bsmt_half = pd.to_numeric(data.get("Bsmt_Half_Bath", 0), errors="coerce").fillna(0)
    data["Total_Bathrooms"] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

    # Porch areas
    wood_deck = pd.to_numeric(data.get("Wood_Deck_SF", 0), errors="coerce").fillna(0)
    open_porch = pd.to_numeric(data.get("Open_Porch_SF", 0), errors="coerce").fillna(0)
    enc_porch = pd.to_numeric(data.get("Enclosed_Porch", 0), errors="coerce").fillna(0)
    screen_porch = pd.to_numeric(data.get("Screen_Porch", 0), errors="coerce").fillna(0)
    data["Total_Porch_SF"] = wood_deck + open_porch + enc_porch + screen_porch

    # Quality and interactions
    if "Overall_Qual" in data.columns:
        overall_qual = map_ordinal(data["Overall_Qual"])
    else:
        overall_qual = 5.0

    if "Overall_Cond" in data.columns:
        overall_cond = map_ordinal(data["Overall_Cond"])
    else:
        overall_cond = 5.0

    data["Overall_Qual_Num"] = overall_qual
    data["Overall_Cond_Num"] = overall_cond
    data["Qual_Cond_Score"] = overall_qual * overall_cond
    data["Qual_Total_SF"] = overall_qual * data["Total_SF"]
    data["Qual_Gr_Liv"] = overall_qual * gr_liv

    # Amenity flags
    data["Has_Bsmt"] = (total_bsmt > 0).astype(int)
    data["Has_Garage"] = (garage_area > 0).astype(int)
    if "Fireplaces" in data.columns:
        fireplaces = pd.to_numeric(data["Fireplaces"], errors="coerce").fillna(0)
        data["Has_Fireplace"] = (fireplaces > 0).astype(int)
    if "Pool_Area" in data.columns:
        pool_area = pd.to_numeric(data["Pool_Area"], errors="coerce").fillna(0)
        data["Has_Pool"] = (pool_area > 0).astype(int)

    return data


X_eng = engineer_features(X)
X_test_eng = engineer_features(X_test)

# Categorical columns processing
cat_cols = X_eng.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    X_eng[col] = X_eng[col].astype(str).fillna("Missing")
    X_test_eng[col] = X_test_eng[col].astype(str).fillna("Missing")

# Format categorical columns for LightGBM & XGBoost
X_lgb = X_eng.copy()
X_test_lgb = X_test_eng.copy()
for col in cat_cols:
    categories = sorted(
        list(set(X_eng[col].unique()).union(set(X_test_eng[col].unique())))
    )
    X_lgb[col] = pd.Categorical(X_lgb[col], categories=categories)
    X_test_lgb[col] = pd.Categorical(X_test_lgb[col], categories=categories)

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_eng, y)):
    # LightGBM
    X_tr_lgb, y_tr = X_lgb.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_lgb.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    val_data = lgb.Dataset(X_va_lgb, label=y_va, reference=trn_data)

    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(
        X_va_lgb, num_iteration=lgb_model.best_iteration
    )
    test_lgb += (
        lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)
        / kf.n_splits
    )

    # CatBoost
    X_tr_cb = X_eng.iloc[train_idx]
    X_va_cb = X_eng.iloc[val_idx]

    cb_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), verbose=False)
    oof_cat[val_idx] = cb_model.predict(X_va_cb)
    test_cat += cb_model.predict(X_test_eng) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        tree_method="hist",
        enable_categorical=True,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va_lgb)
    test_xgb += xgb_model.predict(X_test_lgb) / kf.n_splits

# Optimal ensemble weights via optimization
oof_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
test_matrix = np.column_stack([test_lgb, test_cat, test_xgb])


def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = oof_matrix @ w
    return root_mean_squared_error(y, pred)


res = minimize(
    loss_func,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
)
best_weights = res.x / np.sum(res.x)

oof_ensemble = oof_matrix @ best_weights
test_ensemble = test_matrix @ best_weights

cv_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"LGBM RMSE: {root_mean_squared_error(y, oof_lgb):.4f}")
print(f"CatBoost RMSE: {root_mean_squared_error(y, oof_cat):.4f}")
print(f"XGBoost RMSE: {root_mean_squared_error(y, oof_xgb):.4f}")
print(f"Optimal Ensemble Weights (LGB, Cat, XGB): {best_weights.round(4)}")
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Sale_Price": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
