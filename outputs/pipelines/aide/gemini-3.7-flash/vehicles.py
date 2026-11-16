import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "youSaveSpend"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Feature Engineering
if "comb08" in combined_df.columns:
    combined_df["gpm_comb"] = 1.0 / (combined_df["comb08"].replace(0, np.nan))
if "city08" in combined_df.columns:
    combined_df["gpm_city"] = 1.0 / (combined_df["city08"].replace(0, np.nan))
if "highway08" in combined_df.columns:
    combined_df["gpm_highway"] = 1.0 / (combined_df["highway08"].replace(0, np.nan))

if "displ" in combined_df.columns and "cylinders" in combined_df.columns:
    combined_df["displ_per_cyl"] = combined_df["displ"] / (
        combined_df["cylinders"].replace(0, np.nan)
    )

if "year" in combined_df.columns and "fuelCost08" in combined_df.columns:
    year_fuel_mean = combined_df.groupby("year")["fuelCost08"].transform("mean")
    combined_df["fuelCost08_diff_year_mean"] = (
        combined_df["fuelCost08"] - year_fuel_mean
    )

# Identify categorical and numerical columns
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if (
        combined_df[col].dtype == "object"
        or combined_df[col].dtype.name == "category"
        or combined_df[col].dtype == "bool"
    ):
        cat_cols.append(col)
        combined_df[col] = combined_df[col].astype(str).fillna("missing")
        # Add frequency encoding
        freq = combined_df[col].value_counts()
        combined_df[f"{col}_freq"] = combined_df[col].map(freq)
    else:
        num_cols.append(col)

# Encode categoricals for GBDTs
if cat_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_df[cat_cols] = encoder.fit_transform(combined_df[cat_cols])

X = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# 3. Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)

test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))
test_preds_cb = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 45,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# 4. Train Models Across Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # --- LightGBM ---
    dtrain_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval_lgb = lgb.Dataset(
        X_va, label=y_va, reference=dtrain_lgb, categorical_feature=cat_cols
    )

    model_lgb = lgb.train(
        lgb_params,
        dtrain_lgb,
        num_boost_round=1500,
        valid_sets=[dtrain_lgb, dval_lgb],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # --- XGBoost ---
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
    dval_xgb = xgb.DMatrix(X_va, label=y_va)
    dtest_xgb = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=1500,
        evals=[(dtrain_xgb, "train"), (dval_xgb, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(dval_xgb)
    test_preds_xgb += model_xgb.predict(dtest_xgb) / n_splits

    # --- CatBoost ---
    cb_model = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = cb_model.predict(X_va)
    test_preds_cb += cb_model.predict(X_test) / n_splits


# 5. Optimize Blending Weights
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_cb
    return np.sqrt(mean_squared_error(y, blend))


res = minimize(
    loss_func, [1 / 3, 1 / 3, 1 / 3], bounds=[(0, 1), (0, 1), (0, 1)], method="SLSQP"
)
weights = res.x / np.sum(res.x)
print(f"Optimal Ensemble Weights (LGB, XGB, CB): {weights.round(4)}")

oof_blend = weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_cb
final_cv_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"5-Fold CV RMSE (LightGBM): {np.sqrt(mean_squared_error(y, oof_lgb)):.4f}")
print(f"5-Fold CV RMSE (XGBoost):  {np.sqrt(mean_squared_error(y, oof_xgb)):.4f}")
print(f"5-Fold CV RMSE (CatBoost): {np.sqrt(mean_squared_error(y, oof_cb)):.4f}")
print(f"5-Fold CV RMSE (Ensemble): {final_cv_rmse:.4f}")

# 6. Generate and Save Submission
final_test_preds = (
    weights[0] * test_preds_lgb
    + weights[1] * test_preds_xgb
    + weights[2] * test_preds_cb
)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "youSaveSpend": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
