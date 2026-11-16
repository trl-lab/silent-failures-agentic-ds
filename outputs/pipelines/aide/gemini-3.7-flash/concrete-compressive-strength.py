import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_path = "./working/submission.csv"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


# 2. Enhanced Feature Engineering with Concrete Physics & Hydration Kinetics
def engineer_features(df):
    data = df.copy()

    # Total binder and EN-206 equivalent active binder
    binder = data["cement"] + data["blast_furnace_slag"] + data["fly_ash"]
    data["total_binder"] = binder
    # SCM efficiency (k-value concept: slag ~0.7-0.8, fly ash ~0.4)
    data["effective_binder"] = (
        data["cement"] + 0.75 * data["blast_furnace_slag"] + 0.40 * data["fly_ash"]
    )

    # Water-to-binder and Bolomey cement-to-water ratios (Abrams' law)
    data["water_binder_ratio"] = data["water"] / (binder + 1e-6)
    data["water_cement_ratio"] = data["water"] / (data["cement"] + 1e-6)
    data["binder_water_ratio"] = binder / (data["water"] + 1e-6)
    data["cement_water_ratio"] = data["cement"] / (data["water"] + 1e-6)
    data["eff_binder_water_ratio"] = data["effective_binder"] / (data["water"] + 1e-6)

    # Superplasticizer ratios
    data["sp_binder_ratio"] = data["superplasticizer"] / (binder + 1e-6)
    data["sp_cement_ratio"] = data["superplasticizer"] / (data["cement"] + 1e-6)

    # Aggregate & Paste characteristics
    total_agg = data["coarse_aggregate"] + data["fine_aggregate"]
    data["total_aggregate"] = total_agg
    data["sand_aggregate_ratio"] = data["fine_aggregate"] / (total_agg + 1e-6)
    data["fine_coarse_ratio"] = data["fine_aggregate"] / (
        data["coarse_aggregate"] + 1e-6
    )
    data["binder_agg_ratio"] = binder / (total_agg + 1e-6)

    paste_vol = binder + data["water"] + data["superplasticizer"]
    data["paste_volume"] = paste_vol
    data["paste_agg_ratio"] = paste_vol / (total_agg + 1e-6)

    # Mixture mass fractions
    total_mass = paste_vol + total_agg
    data["total_mass"] = total_mass
    data["cement_mass_fraction"] = data["cement"] / (total_mass + 1e-6)
    data["water_mass_fraction"] = data["water"] / (total_mass + 1e-6)

    # Age non-linear representations (concrete hydration kinetics)
    data["log_age"] = np.log1p(data["age"])
    data["sqrt_age"] = np.sqrt(data["age"])

    # Pozzolanic and latent hydraulic hydration kinetics over time
    data["binder_log_age"] = binder * data["log_age"]
    data["cement_log_age"] = data["cement"] * data["log_age"]
    data["slag_log_age"] = data["blast_furnace_slag"] * data["log_age"]
    data["flyash_log_age"] = data["fly_ash"] * data["log_age"]
    data["water_binder_age_interaction"] = data["water_binder_ratio"] / (
        data["log_age"] + 1e-6
    )

    return data


train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

target_col = "strength"
feature_cols = [col for col in train_feat.columns if col != target_col]

X = train_feat[feature_cols]
y = train_feat[target_col].values
X_test = test_feat[feature_cols]

# 3. Model Definition and Multi-Model 5-Fold Cross-Validation
models = {
    "xgb": lambda: xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        learning_rate=0.03,
        n_estimators=1200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    ),
    "lgb": lambda: lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "cb": lambda: CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    ),
    "et": lambda: ExtraTreesRegressor(
        n_estimators=600,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "hgb": lambda: HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=1000,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    ),
}

model_names = list(models.keys())
n_models = len(model_names)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_dict = {
    name: np.zeros(len(train_feat), dtype=np.float64) for name in model_names
}
test_preds_dict = {
    name: np.zeros(len(test_feat), dtype=np.float64) for name in model_names
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    for name in model_names:
        clf = models[name]()
        clf.fit(X_train, y_train)

        val_pred = clf.predict(X_val)
        oof_preds_dict[name][val_idx] = val_pred
        test_preds_dict[name] += clf.predict(X_test) / n_splits

for name in model_names:
    score = root_mean_squared_error(y, oof_preds_dict[name])
    print(f"Model {name} OOF RMSE: {score:.4f}")

# 4. Out-of-Fold Ensemble Weight Optimization
oof_matrix = np.column_stack([oof_preds_dict[name] for name in model_names])
test_matrix = np.column_stack([test_preds_dict[name] for name in model_names])


def loss_func(weights):
    weights = np.array(weights)
    pred = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y, pred)


init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = opt_res.x
print(
    "Optimal Ensemble Weights:",
    dict(zip(model_names, [round(w, 4) for w in optimal_weights])),
)

# 5. Final Out-of-fold and Test Predictions
ensemble_oof = np.dot(oof_matrix, optimal_weights)
cv_rmse = root_mean_squared_error(y, ensemble_oof)
print(f"Ensemble 5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

ensemble_test = np.dot(test_matrix, optimal_weights)

# 6. Save Submission
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(ensemble_test)), "strength": ensemble_test})
sub.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} with shape {sub.shape}")
