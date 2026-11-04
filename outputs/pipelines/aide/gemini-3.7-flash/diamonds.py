import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Set random seed
SEED = 42
np.random.seed(SEED)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y = train_df["price"].values
n_train = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=["price"]), test_df], axis=0
).reset_index(drop=True)

# Ordinal mappings
cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_map = {
    "I1": 1,
    "SI2": 2,
    "SI1": 3,
    "VS2": 4,
    "VS1": 5,
    "VVS2": 6,
    "VVS1": 7,
    "IF": 8,
}

# Handle physical zero dimensions: replace 0 with median of similar carat
for col in ["x", "y", "z"]:
    combined_df[col] = combined_df[col].replace(0, np.nan)
    combined_df[col] = combined_df[col].fillna(
        combined_df.groupby("carat")[col].transform("median")
    )
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Map ordinals
combined_df["cut_num"] = combined_df["cut"].map(cut_map).astype(float)
combined_df["color_num"] = combined_df["color"].map(color_map).astype(float)
combined_df["clarity_num"] = combined_df["clarity"].map(clarity_map).astype(float)

# Domain feature engineering
combined_df["volume"] = combined_df["x"] * combined_df["y"] * combined_df["z"]
combined_df["density"] = combined_df["carat"] / (combined_df["volume"] + 1e-5)
combined_df["depth_calc"] = (
    2 * combined_df["z"] / (combined_df["x"] + combined_df["y"] + 1e-5) * 100
)
combined_df["depth_diff"] = combined_df["depth"] - combined_df["depth_calc"]
combined_df["surface_area"] = 2 * (
    combined_df["x"] * combined_df["y"]
    + combined_df["y"] * combined_df["z"]
    + combined_df["x"] * combined_df["z"]
)
combined_df["table_area"] = (
    combined_df["x"] * combined_df["y"] * (combined_df["table"] / 100.0)
)

# Ratios
combined_df["ratio_xy"] = combined_df["x"] / (combined_df["y"] + 1e-5)
combined_df["ratio_xz"] = combined_df["x"] / (combined_df["z"] + 1e-5)
combined_df["ratio_yz"] = combined_df["y"] / (combined_df["z"] + 1e-5)
combined_df["carat_per_xyz"] = combined_df["carat"] / (
    combined_df["x"] + combined_df["y"] + combined_df["z"] + 1e-5
)

# Non-linear & interactions
combined_df["carat_squared"] = combined_df["carat"] ** 2
combined_df["carat_sqrt"] = np.sqrt(combined_df["carat"])
combined_df["overall_quality"] = (
    combined_df["cut_num"] * combined_df["color_num"] * combined_df["clarity_num"]
)
combined_df["carat_x_quality"] = combined_df["carat"] * combined_df["overall_quality"]
combined_df["carat_x_cut"] = combined_df["carat"] * combined_df["cut_num"]
combined_df["carat_x_color"] = combined_df["carat"] * combined_df["color_num"]
combined_df["carat_x_clarity"] = combined_df["carat"] * combined_df["clarity_num"]

# Drop raw object columns
features_to_drop = ["cut", "color", "clarity"]
combined_df = combined_df.drop(columns=features_to_drop)

# Split back
X_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

feature_cols = list(X_train.columns)

# 5-fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

preds_lgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # 1. LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": SEED + fold,
        "n_jobs": -1,
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2500,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    preds_lgb += (
        lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
    )

    # 2. CatBoost
    cat_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.04,
        depth=7,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED + fold,
        verbose=0,
    )
    cat_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    oof_cat[val_idx] = cat_model.predict(X_va)
    preds_cat += cat_model.predict(X_test) / n_splits

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    preds_xgb += xgb_model.predict(X_test) / n_splits

# Individual OOF RMSE
rmse_lgb = root_mean_squared_error(y, oof_lgb)
rmse_cat = root_mean_squared_error(y, oof_cat)
rmse_xgb = root_mean_squared_error(y, oof_xgb)

# Optimal blending weights
oof_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
test_matrix = np.column_stack([preds_lgb, preds_cat, preds_xgb])


def blend_loss(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blended_oof = oof_matrix @ weights
    return root_mean_squared_error(y, blended_oof)


res = minimize(
    blend_loss, [1 / 3, 1 / 3, 1 / 3], bounds=[(0, 1), (0, 1), (0, 1)], method="SLSQP"
)
optimal_weights = res.x / np.sum(res.x)

oof_ensemble = oof_matrix @ optimal_weights
final_rmse = root_mean_squared_error(y, oof_ensemble)

# Make test predictions
test_predictions = np.clip(test_matrix @ optimal_weights, a_min=0, a_max=None)

# Create submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), "price": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"LGBM OOF RMSE: {rmse_lgb:.4f}")
print(f"CatBoost OOF RMSE: {rmse_cat:.4f}")
print(f"XGBoost OOF RMSE: {rmse_xgb:.4f}")
print(f"Ensemble Blend Weights: {optimal_weights}")
print(f"Final Ensemble OOF RMSE: {final_rmse:.4f}")
