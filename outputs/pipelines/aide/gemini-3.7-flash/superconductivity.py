import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set random seed for reproducibility
RANDOM_STATE = 42
N_SPLITS = 5

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "critical_temp"


def extract_features(df):
    """
    Engineer domain-specific physical descriptors capturing differences
    and ratios between weighted (stoichiometric) and unweighted statistics.
    """
    df_feat = df.copy()
    eps = 1e-6

    properties = [
        "atomic_mass",
        "fie",
        "atomic_radius",
        "Density",
        "ElectronAffinity",
        "FusionHeat",
        "ThermalConductivity",
        "Valence",
    ]

    new_cols = {}
    for prop in properties:
        mean_col = f"mean_{prop}"
        wtd_mean_col = f"wtd_mean_{prop}"
        gmean_col = f"gmean_{prop}"
        wtd_gmean_col = f"wtd_gmean_{prop}"
        entropy_col = f"entropy_{prop}"
        wtd_entropy_col = f"wtd_entropy_{prop}"
        std_col = f"std_{prop}"
        wtd_std_col = f"wtd_std_{prop}"
        range_col = f"range_{prop}"
        wtd_range_col = f"wtd_range_{prop}"

        # Mean differences and ratios
        new_cols[f"diff_mean_{prop}"] = df_feat[wtd_mean_col] - df_feat[mean_col]
        new_cols[f"ratio_mean_{prop}"] = df_feat[wtd_mean_col] / (
            df_feat[mean_col].abs() + eps
        )

        # Geometric mean differences
        new_cols[f"diff_gmean_{prop}"] = df_feat[wtd_gmean_col] - df_feat[gmean_col]

        # Entropy differences (dispersion across elements vs stoichiometries)
        new_cols[f"diff_entropy_{prop}"] = (
            df_feat[wtd_entropy_col] - df_feat[entropy_col]
        )

        # Standard deviation differences
        new_cols[f"diff_std_{prop}"] = df_feat[wtd_std_col] - df_feat[std_col]

        # Coefficient of variation (relative dispersion)
        new_cols[f"cv_{prop}"] = df_feat[std_col] / (df_feat[mean_col].abs() + eps)
        new_cols[f"wtd_cv_{prop}"] = df_feat[wtd_std_col] / (
            df_feat[wtd_mean_col].abs() + eps
        )

        # Range differences and relative range
        new_cols[f"diff_range_{prop}"] = df_feat[wtd_range_col] - df_feat[range_col]
        new_cols[f"range_to_mean_{prop}"] = df_feat[range_col] / (
            df_feat[mean_col].abs() + eps
        )

    new_features_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df_feat, new_features_df], axis=1)


# Apply feature engineering
train_enriched = extract_features(train_df)
test_enriched = extract_features(test_df)

feature_cols = [col for col in train_enriched.columns if col != target_col]

X = train_enriched[feature_cols].copy()
y = train_enriched[target_col].copy()
X_test = test_enriched[feature_cols].copy()

# Setup 5-fold cross-validation
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

# Hyperparameters tuned for enriched feature space
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 95,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "n_estimators": 3500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 5,
    "n_estimators": 3500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Training Fold {fold + 1}/{N_SPLITS} ---")
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

    # 1. LightGBM
    trn_lgb = lgb.Dataset(X_train_f, label=y_train_f)
    val_lgb = lgb.Dataset(X_val_f, label=y_val_f, reference=trn_lgb)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model_lgb = lgb.train(
        params=lgb_params,
        train_set=trn_lgb,
        valid_sets=[trn_lgb, val_lgb],
        callbacks=lgb_callbacks,
    )
    oof_lgb[val_idx] = model_lgb.predict(
        X_val_f, num_iteration=model_lgb.best_iteration
    )
    test_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / N_SPLITS
    )

    # 2. XGBoost
    model_xgb = xgb.XGBRegressor(
        **xgb_params,
        early_stopping_rounds=100,
    )
    model_xgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val_f)
    test_xgb += model_xgb.predict(X_test) / N_SPLITS

    # 3. CatBoost
    model_cat = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.04,
        depth=7,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=100,
        verbose=False,
        thread_count=-1,
    )
    model_cat.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        use_best_model=True,
    )
    oof_cat[val_idx] = model_cat.predict(X_val_f)
    test_cat += model_cat.predict(X_test) / N_SPLITS

    f_lgb_rmse = root_mean_squared_error(y_val_f, np.clip(oof_lgb[val_idx], 0, None))
    f_xgb_rmse = root_mean_squared_error(y_val_f, np.clip(oof_xgb[val_idx], 0, None))
    f_cat_rmse = root_mean_squared_error(y_val_f, np.clip(oof_cat[val_idx], 0, None))
    print(
        f"Fold {fold + 1} RMSE -> LGB: {f_lgb_rmse:.4f}, XGB: {f_xgb_rmse:.4f}, CAT: {f_cat_rmse:.4f}"
    )

print("\n--- Individual Model Overall OOF RMSE ---")
print(f"LightGBM OOF RMSE: {root_mean_squared_error(y, np.clip(oof_lgb, 0, None)):.4f}")
print(f"XGBoost  OOF RMSE: {root_mean_squared_error(y, np.clip(oof_xgb, 0, None)):.4f}")
print(f"CatBoost OOF RMSE: {root_mean_squared_error(y, np.clip(oof_cat, 0, None)):.4f}")

# Optimal ensemble weights via constrained optimization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat])


def objective(weights):
    w = weights / np.sum(weights)
    pred = np.clip(oof_matrix @ w, 0, None)
    return root_mean_squared_error(y, pred)


init_weights = np.ones(3) / 3
bounds = [(0, 1)] * 3
constraints = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}
opt_res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
weights = opt_res.x / np.sum(opt_res.x)

print(
    f"\nOptimal Ensemble Weights -> LGB: {weights[0]:.4f}, XGB: {weights[1]:.4f}, CAT: {weights[2]:.4f}"
)

oof_ensemble = np.clip(oof_matrix @ weights, 0, None)
final_oof_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"Validation Metric (Ensemble OOF RMSE): {final_oof_rmse:.4f}")

# Generate final test predictions
final_test_preds = np.clip(test_matrix @ weights, 0, None)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "critical_temp": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
