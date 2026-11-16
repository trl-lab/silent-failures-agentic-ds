import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "youSaveSpend"
y = train[target_col].values

# Separate features from target
X = train.drop(columns=[target_col]).copy()
X_test = test.copy()


# Enhanced domain feature engineering with explicit EPA 5-year savings estimators
def create_domain_features(df_tr, df_te):
    combined = pd.concat([df_tr, df_te], axis=0).reset_index(drop=True)

    # Basic 5-year fuel cost features
    if "fuelCost08" in combined.columns:
        combined["fuelCost08_5yr"] = combined["fuelCost08"] * 5
    if "fuelCostA08" in combined.columns:
        combined["fuelCostA08_5yr"] = combined["fuelCostA08"] * 5

    # Combined total annual fuel cost for dual fuel / plug-in vehicles
    if "fuelCost08" in combined.columns and "fuelCostA08" in combined.columns:
        combined["fuelCost_total"] = combined["fuelCost08"] + combined["fuelCostA08"]
        combined["fuelCost_total_5yr"] = combined["fuelCost_total"] * 5

    if "city08" in combined.columns and "highway08" in combined.columns:
        combined["mpg_diff"] = combined["highway08"] - combined["city08"]
        combined["mpg_ratio"] = combined["city08"] / (combined["highway08"] + 1e-5)
    if "displ" in combined.columns and "cylinders" in combined.columns:
        combined["displ_per_cyl"] = combined["displ"] / (combined["cylinders"] + 1e-5)
    if "co2" in combined.columns and "comb08" in combined.columns:
        combined["co2_per_comb"] = combined["co2"] / (combined["comb08"] + 1e-5)
    if "barrels08" in combined.columns and "comb08" in combined.columns:
        combined["barrels_per_comb"] = combined["barrels08"] / (
            combined["comb08"] + 1e-5
        )

    if "year" in combined.columns:
        combined["vehicle_age"] = 2025 - combined["year"]

        # Year benchmark relative features & direct EPA 5-year savings proxy
        if "fuelCost08" in combined.columns:
            year_fc_mean = combined.groupby("year")["fuelCost08"].transform("mean")
            year_fc_median = combined.groupby("year")["fuelCost08"].transform("median")
            year_fc_std = (
                combined.groupby("year")["fuelCost08"].transform("std").fillna(0)
            )

            # Direct EPA 5-year savings formula: 5 * (year_baseline - vehicle_annual_cost)
            combined["epa_savings_est_08_mean"] = 5.0 * (
                year_fc_mean - combined["fuelCost08"]
            )
            combined["epa_savings_est_08_median"] = 5.0 * (
                year_fc_median - combined["fuelCost08"]
            )
            combined["fuelCost08_year_zscore"] = (
                combined["fuelCost08"] - year_fc_mean
            ) / (year_fc_std + 1e-5)
            combined["fuelCost08_year_diff"] = combined["fuelCost08"] - year_fc_mean
            combined["fuelCost08_year_ratio"] = combined["fuelCost08"] / (
                year_fc_mean + 1e-5
            )

        if "fuelCost_total" in combined.columns:
            year_fct_mean = combined.groupby("year")["fuelCost_total"].transform("mean")
            combined["epa_savings_est_total"] = 5.0 * (
                year_fct_mean - combined["fuelCost_total"]
            )

        if "comb08" in combined.columns:
            year_comb_mean = combined.groupby("year")["comb08"].transform("mean")
            combined["comb08_year_diff"] = combined["comb08"] - year_comb_mean
            combined["comb08_year_ratio"] = combined["comb08"] / (year_comb_mean + 1e-5)

    # Subgroup benchmark relative features: year x VClass
    if (
        "year" in combined.columns
        and "VClass" in combined.columns
        and "fuelCost08" in combined.columns
    ):
        y_vc_fc = combined.groupby(["year", "VClass"])["fuelCost08"].transform("mean")
        combined["fuelCost08_y_vc_diff"] = combined["fuelCost08"] - y_vc_fc
        combined["epa_savings_est_y_vc"] = 5.0 * (y_vc_fc - combined["fuelCost08"])

    # Vehicle Class benchmark relative features
    if "VClass" in combined.columns:
        if "fuelCost08" in combined.columns:
            vclass_fc = combined.groupby("VClass")["fuelCost08"].transform("mean")
            combined["fuelCost08_vclass_diff"] = combined["fuelCost08"] - vclass_fc
            combined["fuelCost08_vclass_ratio"] = combined["fuelCost08"] / (
                vclass_fc + 1e-5
            )
        if "comb08" in combined.columns:
            vclass_comb = combined.groupby("VClass")["comb08"].transform("mean")
            combined["comb08_vclass_diff"] = combined["comb08"] - vclass_comb
            combined["comb08_vclass_ratio"] = combined["comb08"] / (vclass_comb + 1e-5)

    # Fuel Type benchmark relative features
    fuel_type_col = (
        "fuelType1"
        if "fuelType1" in combined.columns
        else ("fuelType" if "fuelType" in combined.columns else None)
    )
    if fuel_type_col is not None:
        if "fuelCost08" in combined.columns:
            ft_fc = combined.groupby(fuel_type_col)["fuelCost08"].transform("mean")
            combined["fuelCost08_ft_diff"] = combined["fuelCost08"] - ft_fc
            combined["fuelCost08_ft_ratio"] = combined["fuelCost08"] / (ft_fc + 1e-5)
        if "comb08" in combined.columns:
            ft_comb = combined.groupby(fuel_type_col)["comb08"].transform("mean")
            combined["comb08_ft_diff"] = combined["comb08"] - ft_comb
            combined["comb08_ft_ratio"] = combined["comb08"] / (ft_comb + 1e-5)

    X_tr_out = combined.iloc[: len(df_tr)].copy().reset_index(drop=True)
    X_te_out = combined.iloc[len(df_tr) :].copy().reset_index(drop=True)
    return X_tr_out, X_te_out


X, X_test = create_domain_features(X, X_test)

# Categorical processing
cat_cols = [
    c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"
]

for col in cat_cols:
    X[col] = X[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

    # Frequency encoding
    freq = X[col].value_counts().to_dict()
    X[f"{col}_freq"] = X[col].map(freq).fillna(0)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)

if cat_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Ensure all remaining object dtypes are numeric
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

# Identify features for target encoding
te_cols = [c for c in cat_cols if X[c].nunique() > 5]

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

m_smooth = 10.0  # Smoothing weight for target encoding

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[train_idx].copy(), y[train_idx]
    X_va, y_va = X.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test.copy()

    # Smoothed out-of-fold target encoding
    global_mean = y_tr.mean()
    for col in te_cols:
        grp = pd.Series(y_tr).groupby(X_tr[col])
        counts = grp.count()
        means = grp.mean()
        smoothed_means = (counts * means + m_smooth * global_mean) / (counts + m_smooth)

        X_tr[f"{col}_te"] = X_tr[col].map(smoothed_means).fillna(global_mean)
        X_va[f"{col}_te"] = X_va[col].map(smoothed_means).fillna(global_mean)
        X_te[f"{col}_te"] = X_te[col].map(smoothed_means).fillna(global_mean)

    # Train XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof = xgb_model.predict(X_va)
    xgb_test = xgb_model.predict(X_te)

    # Train LightGBM Regressor
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.025,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42 + fold,
        "verbose": -1,
    }
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_oof = lgb_model.predict(X_va)
    lgb_test = lgb_model.predict(X_te)

    # Ensemble predictions (50/50 blend)
    oof_preds[val_idx] = 0.5 * xgb_oof + 0.5 * lgb_oof
    test_preds += (0.5 * xgb_test + 0.5 * lgb_test) / kf.n_splits

# Compute evaluation metric
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation Metric: {oof_rmse:.4f}")

# Save final predictions
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "youSaveSpend": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
