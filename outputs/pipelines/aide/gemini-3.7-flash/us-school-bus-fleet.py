import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Bus age"
drop_cols = [target_col] if target_col in train_df.columns else []
raw_features = [c for c in train_df.columns if c not in drop_cols]

# Identify column types
cat_cols = []
num_cols = []

for c in raw_features:
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Preprocessing: Ordinal encode and frequency encode categorical features
for col in cat_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[col] = oe.fit_transform(train_df[[col]])
    test_df[col] = oe.transform(test_df[[col]])

    # Frequency encoding
    freq_map = train_df[col].value_counts(normalize=True).to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
    test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

# Fill numeric NaNs
for col in num_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

feature_cols = [c for c in train_df.columns if c not in drop_cols]

X = train_df[feature_cols].copy()
y = train_df[target_col].values
X_test = test_df[feature_cols].copy()

# Setup 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

# Model configurations
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 15,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "random_state": 42,
    "n_estimators": 2000,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # 1. LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / kf.n_splits
    )

    # 2. XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / kf.n_splits

    # 3. CatBoost
    model_cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0,
    )
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test) / kf.n_splits

print(f"LGB CV RMSE: {np.sqrt(mean_squared_error(y, oof_lgb)):.5f}")
print(f"XGB CV RMSE: {np.sqrt(mean_squared_error(y, oof_xgb)):.5f}")
print(f"CAT CV RMSE: {np.sqrt(mean_squared_error(y, oof_cat)):.5f}")


# Find optimal ensemble weights
def ensemble_loss(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    return np.sqrt(mean_squared_error(y, pred))


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    ensemble_loss, init_weights, bounds=bounds, constraints=constraints, method="SLSQP"
)
best_w = res.x
print(
    f"Optimal Ensemble Weights (LGB, XGB, CAT): {best_w[0]:.4f}, {best_w[1]:.4f}, {best_w[2]:.4f}"
)

oof_ensemble = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cat
test_ensemble = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat

# Post-processing: Bus age is non-negative
if np.all(y >= 0):
    oof_ensemble = np.clip(oof_ensemble, 0, None)
    test_ensemble = np.clip(test_ensemble, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Prepare submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Bus age": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
