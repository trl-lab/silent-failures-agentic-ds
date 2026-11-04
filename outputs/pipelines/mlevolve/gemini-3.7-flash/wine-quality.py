import os
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Loading & Domain-Specific Feature Engineering
# =========================================================================

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df["is_test"] = 0
test_df["is_test"] = 1
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Conditionally encode categorical wine color if present in the dataset
if "color" in combined_df.columns:
    combined_df["color_code"] = (
        combined_df["color"].astype(str).str.lower() == "white"
    ).astype(int)


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    f_df = df.copy()
    eps = 1e-6

    # Sulfur dioxide chemistry & Molecular SO2 equilibrium
    f_df["bound_sulfur_dioxide"] = np.maximum(
        0, f_df["total_sulfur_dioxide"] - f_df["free_sulfur_dioxide"]
    )
    f_df["free_to_total_so2_ratio"] = f_df["free_sulfur_dioxide"] / (
        f_df["total_sulfur_dioxide"] + eps
    )
    f_df["bound_to_total_so2_ratio"] = f_df["bound_sulfur_dioxide"] / (
        f_df["total_sulfur_dioxide"] + eps
    )
    # Fundamental enological formula: Molecular SO2 = Free SO2 / (1 + 10^(pH - 1.81))
    f_df["molecular_so2"] = f_df["free_sulfur_dioxide"] / (
        1.0 + 10.0 ** (f_df["pH"] - 1.81)
    )

    # Acidity chemistry & Spoilage indices
    f_df["total_acidity"] = (
        f_df["fixed_acidity"] + f_df["volatile_acidity"] + f_df["citric_acid"]
    )
    f_df["volatile_to_fixed_acidity"] = f_df["volatile_acidity"] / (
        f_df["fixed_acidity"] + eps
    )
    f_df["citric_to_fixed_acidity"] = f_df["citric_acid"] / (
        f_df["fixed_acidity"] + eps
    )
    f_df["citric_to_volatile_acidity"] = f_df["citric_acid"] / (
        f_df["volatile_acidity"] + eps
    )
    f_df["pH_to_total_acidity"] = f_df["pH"] / (f_df["total_acidity"] + eps)
    f_df["pH_times_total_acidity"] = f_df["pH"] * f_df["total_acidity"]

    # Sugar, Alcohol, and Density balance
    f_df["alcohol_to_sugar"] = f_df["alcohol"] / (f_df["residual_sugar"] + eps)
    f_df["sugar_to_density"] = f_df["residual_sugar"] / (f_df["density"] + eps)
    f_df["alcohol_to_density"] = f_df["alcohol"] / (f_df["density"] + eps)
    f_df["alcohol_times_density"] = f_df["alcohol"] * f_df["density"]

    # True apparent dry extract balance in enology: 1000*(density - 1) + 2*alcohol
    f_df["dry_extract_balance"] = 1000.0 * (f_df["density"] - 1.0) + 2.0 * f_df["alcohol"]
    f_df["estimated_dry_extract"] = (f_df["density"] - 0.990) * 1000 - (
        f_df["alcohol"] * 0.794
    )

    # Mineral, preservative efficacy, and interaction terms
    f_df["chlorides_to_sulphates"] = f_df["chlorides"] / (f_df["sulphates"] + eps)
    f_df["sulphates_to_total_so2"] = f_df["sulphates"] / (
        f_df["total_sulfur_dioxide"] + eps
    )
    f_df["chlorides_times_density"] = f_df["chlorides"] * f_df["density"]
    f_df["volatile_acidity_times_alcohol"] = f_df["volatile_acidity"] * f_df["alcohol"]
    f_df["preservative_efficacy"] = f_df["free_sulfur_dioxide"] * f_df["pH"]

    # Palate sensory balance indicators & buffer dynamics
    f_df["sweetness_to_acid_ratio"] = f_df["residual_sugar"] / (f_df["total_acidity"] + eps)
    f_df["non_sugar_dry_extract"] = f_df["dry_extract_balance"] - f_df["residual_sugar"]
    f_df["buffer_capacity_index"] = f_df["fixed_acidity"] / (f_df["pH"] + eps)
    f_df["alcohol_to_total_acidity"] = f_df["alcohol"] / (f_df["total_acidity"] + eps)
    f_df["alcohol_to_fixed_acidity"] = f_df["alcohol"] / (f_df["fixed_acidity"] + eps)
    f_df["sulphur_efficacy_ratio"] = f_df["molecular_so2"] / (f_df["pH"] + eps)
    f_df["free_so2_to_total_acidity"] = f_df["free_sulfur_dioxide"] / (f_df["total_acidity"] + eps)
    f_df["sulphates_times_free_so2"] = f_df["sulphates"] * f_df["free_sulfur_dioxide"]

    # Log transformations for heavily skewed continuous variables
    skewed_cols = [
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "sulphates",
        "molecular_so2",
    ]
    for col in skewed_cols:
        f_df[f"log_{col}"] = np.log1p(np.maximum(0, f_df[col]))

    return f_df


processed_combined = engineer_domain_features(combined_df)

# Color-conditioned aggregations if color feature is present in data
if "color_code" in processed_combined.columns:
    train_mask = processed_combined["is_test"] == 0
    base_numeric_cols = [
        "alcohol",
        "volatile_acidity",
        "residual_sugar",
        "total_sulfur_dioxide",
        "density",
        "total_acidity",
        "chlorides",
        "sulphates",
        "pH",
    ]

    for col in base_numeric_cols:
        group_stats = (
            processed_combined[train_mask]
            .groupby("color_code")[col]
            .agg(["mean", "std"])
            .reset_index()
        )
        group_stats.columns = ["color_code", f"{col}_color_mean", f"{col}_color_std"]

        processed_combined = processed_combined.merge(
            group_stats, on="color_code", how="left"
        )
        processed_combined[f"{col}_color_diff"] = (
            processed_combined[col] - processed_combined[f"{col}_color_mean"]
        )
        processed_combined[f"{col}_color_zscore"] = (
            processed_combined[col] - processed_combined[f"{col}_color_mean"]
        ) / (processed_combined[f"{col}_color_std"] + 1e-6)

# Separate back into train and test sets safely
cols_to_drop_train = [c for c in ["is_test", "color"] if c in processed_combined.columns]
cols_to_drop_test = [
    c for c in ["is_test", "color", "quality"] if c in processed_combined.columns
]
train_processed = (
    processed_combined[processed_combined["is_test"] == 0]
    .drop(columns=cols_to_drop_train)
    .reset_index(drop=True)
)
test_processed = (
    processed_combined[processed_combined["is_test"] == 1]
    .drop(columns=cols_to_drop_test)
    .reset_index(drop=True)
)

# Stratified 10-Fold Split based on binned quality target for balanced fold distribution
strat_target = pd.cut(
    train_processed["quality"],
    bins=[-np.inf, 4.5, 5.5, 6.5, np.inf],
    labels=False,
)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_processed["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, strat_target)
):
    train_processed.loc[val_idx, "fold"] = fold

# =========================================================================
# 2. Model Architectures and Model Factories
# =========================================================================


def get_lgbm_model(seed: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=50,
    )


def get_catboost_model(seed: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=50,
    )


def get_rf_model(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def get_extratrees_model(seed: int = 42) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


# =========================================================================
# 3. Cross-Validation Training & Inference Loop
# =========================================================================

feature_cols = [c for c in train_processed.columns if c not in ["quality", "fold"]]
X_train_full = train_processed[feature_cols]
y_train_full = train_processed["quality"].values
X_test = test_processed[feature_cols]

n_splits = train_processed["fold"].nunique()
n_train = len(train_processed)
n_test = len(test_processed)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_et = np.zeros(n_train)
oof_rf = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_et = np.zeros(n_test)
test_preds_rf = np.zeros(n_test)

for fold in range(n_splits):
    train_mask = train_processed["fold"] != fold
    val_mask = train_processed["fold"] == fold

    X_tr_df, y_tr = (
        train_processed.loc[train_mask, feature_cols],
        y_train_full[train_mask],
    )
    X_va_df, y_va = train_processed.loc[val_mask, feature_cols], y_train_full[val_mask]

    # --- LightGBM ---
    lgb_model = get_lgbm_model(seed=42 + fold)
    lgb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_mask] = lgb_model.predict(X_va_df)
    test_preds_lgb += lgb_model.predict(X_test) / n_splits

    # --- XGBoost ---
    xgb_model = get_xgb_model(seed=42 + fold)
    xgb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        verbose=False,
    )
    oof_xgb[val_mask] = xgb_model.predict(X_va_df)
    test_preds_xgb += xgb_model.predict(X_test) / n_splits

    # --- CatBoost ---
    cat_model = get_catboost_model(seed=42 + fold)
    cat_model.fit(
        X_tr_df,
        y_tr,
        eval_set=(X_va_df, y_va),
        verbose=False,
    )
    oof_cat[val_mask] = cat_model.predict(X_va_df)
    test_preds_cat += cat_model.predict(X_test) / n_splits

    # --- ExtraTrees ---
    et_model = get_extratrees_model(seed=42 + fold)
    et_model.fit(X_tr_df, y_tr)
    oof_et[val_mask] = et_model.predict(X_va_df)
    test_preds_et += et_model.predict(X_test) / n_splits

    # --- Random Forest ---
    rf_model = get_rf_model(seed=42 + fold)
    rf_model.fit(X_tr_df, y_tr)
    oof_rf[val_mask] = rf_model.predict(X_va_df)
    test_preds_rf += rf_model.predict(X_test) / n_splits

    fold_lgb_rmse = np.sqrt(mean_squared_error(y_va, oof_lgb[val_mask]))
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_va, oof_xgb[val_mask]))
    fold_cat_rmse = np.sqrt(mean_squared_error(y_va, oof_cat[val_mask]))
    fold_et_rmse = np.sqrt(mean_squared_error(y_va, oof_et[val_mask]))
    fold_rf_rmse = np.sqrt(mean_squared_error(y_va, oof_rf[val_mask]))
    print(
        f"Fold {fold+1} RMSE | LGB: {fold_lgb_rmse:.4f}, XGB: {fold_xgb_rmse:.4f}, CAT: {fold_cat_rmse:.4f}, ET: {fold_et_rmse:.4f}, RF: {fold_rf_rmse:.4f}"
    )

# =========================================================================
# 4. Ensemble Weight Optimization & Hold-out Evaluation
# =========================================================================

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_rf]
)


def objective(weights):
    weights = np.array(weights)
    blended = np.dot(oof_matrix, weights)
    return np.sqrt(mean_squared_error(y_train_full, blended))


init_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
bounds = [(0, 1) for _ in range(5)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x
print(
    f"Optimized Ensemble Weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CAT={best_weights[2]:.3f}, ET={best_weights[3]:.3f}, RF={best_weights[4]:.3f}"
)

final_oof_predictions = np.dot(oof_matrix, best_weights)
final_test_predictions = np.dot(test_matrix, best_weights)

# Constrain predictions to the valid domain score range
final_oof_predictions = np.clip(final_oof_predictions, 0.0, 10.0)
final_test_predictions = np.clip(final_test_predictions, 0.0, 10.0)

final_score = np.sqrt(mean_squared_error(y_train_full, final_oof_predictions))

# Save submission file
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_predictions)), "quality": final_test_predictions}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
