import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and exclude non-feature columns
target_col = "heating_load"
ignore_cols = [c for c in ["cooling_load", target_col] if c in train_df.columns]

y_train = train_df[target_col].values
X_train_raw = train_df.drop(columns=ignore_cols, errors="ignore")
X_test_raw = test_df.drop(columns=ignore_cols, errors="ignore")


def engineer_features(df):
    data = df.copy()

    # Domain physical & architectural interactions
    data["floor_area"] = data["roof_area"]
    data["volume"] = data["floor_area"] * data["overall_height"]
    data["surface_to_volume"] = data["surface_area"] / (data["volume"] + 1e-6)
    data["wall_to_height"] = data["wall_area"] / (data["overall_height"] + 1e-6)
    data["wall_to_surface"] = data["wall_area"] / (data["surface_area"] + 1e-6)
    data["roof_to_surface"] = data["roof_area"] / (data["surface_area"] + 1e-6)

    # Glazing features
    data["glazing_area_abs_wall"] = data["glazing_area"] * data["wall_area"]
    data["glazing_area_abs_floor"] = data["glazing_area"] * data["floor_area"]
    data["net_wall_area"] = data["wall_area"] - data["glazing_area_abs_wall"]

    # Thermal loss proxy
    data["thermal_loss_proxy"] = (
        data["net_wall_area"] * 0.5
        + data["roof_area"] * 0.3
        + data["glazing_area_abs_wall"] * 2.5
    )

    # Proportions & ratios
    data["compactness_volume"] = data["relative_compactness"] * data["volume"]
    data["glazing_height_ratio"] = data["glazing_area"] * data["overall_height"]
    data["height_compactness_ratio"] = data["overall_height"] / (
        data["relative_compactness"] + 1e-6
    )

    # Categorical columns handling
    cat_cols = ["orientation", "glazing_area_distribution"]
    for c in cat_cols:
        if c in data.columns:
            data[c] = data[c].astype("category")

    return data


X_train = engineer_features(X_train_raw)
X_test = engineer_features(X_test_raw)

cat_cols = ["orientation", "glazing_area_distribution"]

seeds = [42, 101, 2023, 777, 999, 1234, 4321, 555, 888, 314]
n_seeds = len(seeds)

oof_xgb_accum = np.zeros(len(X_train))
oof_lgb_accum = np.zeros(len(X_train))
oof_cat_accum = np.zeros(len(X_train))

test_xgb_accum = np.zeros(len(X_test))
test_lgb_accum = np.zeros(len(X_test))
test_cat_accum = np.zeros(len(X_test))

for seed_idx, seed in enumerate(seeds):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof_xgb_seed = np.zeros(len(X_train))
    oof_lgb_seed = np.zeros(len(X_train))
    oof_cat_seed = np.zeros(len(X_train))

    xgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "enable_categorical": True,
        "early_stopping_rounds": 100,
        "random_state": seed,
        "eval_metric": "rmse",
    }

    lgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
    }

    cat_params = {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": seed,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 100,
        "verbose": False,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        # XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb_seed[val_idx] = model_xgb.predict(X_va)
        test_xgb_accum += model_xgb.predict(X_test) / (kf.n_splits * n_seeds)

        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        callbacks = [lgb.early_stopping(100, verbose=False)]
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
        oof_lgb_seed[val_idx] = model_lgb.predict(X_va)
        test_lgb_accum += model_lgb.predict(X_test) / (kf.n_splits * n_seeds)

        # CatBoost
        cat_tr = X_tr.copy()
        cat_va = X_va.copy()
        cat_te = X_test.copy()
        for c in cat_cols:
            cat_tr[c] = cat_tr[c].astype(int)
            cat_va[c] = cat_va[c].astype(int)
            cat_te[c] = cat_te[c].astype(int)

        model_cat = CatBoostRegressor(**cat_params, cat_features=cat_cols)
        model_cat.fit(cat_tr, y_tr, eval_set=(cat_va, y_va), verbose=False)
        oof_cat_seed[val_idx] = model_cat.predict(cat_va)
        test_cat_accum += model_cat.predict(cat_te) / (kf.n_splits * n_seeds)

    oof_xgb_accum += oof_xgb_seed / n_seeds
    oof_lgb_accum += oof_lgb_seed / n_seeds
    oof_cat_accum += oof_cat_seed / n_seeds


# Optimize blend weights on multi-seed averaged OOF predictions
def objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_pred = (
        weights[0] * oof_xgb_accum
        + weights[1] * oof_lgb_accum
        + weights[2] * oof_cat_accum
    )
    return np.sqrt(mean_squared_error(y_train, blend_pred))


res = minimize(
    objective, [1 / 3, 1 / 3, 1 / 3], method="SLSQP", bounds=[(0, 1), (0, 1), (0, 1)]
)
best_w = res.x / np.sum(res.x)

oof_blend = (
    best_w[0] * oof_xgb_accum + best_w[1] * oof_lgb_accum + best_w[2] * oof_cat_accum
)
cv_rmse = np.sqrt(mean_squared_error(y_train, oof_blend))
print(f"Multi-Seed Ensemble 5-Fold Cross-Validation OOF RMSE: {cv_rmse:.4f}")

test_predictions = (
    best_w[0] * test_xgb_accum + best_w[1] * test_lgb_accum + best_w[2] * test_cat_accum
)

# Export predictions
submission = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), "heating_load": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
