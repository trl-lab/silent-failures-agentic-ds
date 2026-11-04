import os
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
SEED = 42

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "fatalities"
y_train = train_df[target_col].values

# Combine data for unified feature processing
df_train_no_target = train_df.drop(columns=[target_col], errors="ignore")
df_all = pd.concat([df_train_no_target, test_df], ignore_index=True)

# Parse datetime columns if present and derive temporal features
datetime_cols = []
for col in df_all.columns:
    if df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col])
            df_all[f"{col}_year"] = parsed.dt.year
            df_all[f"{col}_month"] = parsed.dt.month
            df_all[f"{col}_day"] = parsed.dt.day
            df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df_all[f"{col}_dayofyear"] = parsed.dt.dayofyear
            df_all[f"{col}_quarter"] = parsed.dt.quarter
            df_all.drop(columns=[col], inplace=True)
            datetime_cols.append(col)
        except Exception:
            pass

# Identify categorical features
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()

# Create pairwise interaction features between categorical variables
cat_interaction_cols = []
if len(cat_cols) >= 2:
    for c1, c2 in itertools.combinations(cat_cols[:5], 2):
        col_name = f"{c1}_x_{c2}"
        df_all[col_name] = df_all[c1].astype(str) + "_" + df_all[c2].astype(str)
        cat_interaction_cols.append(col_name)

all_cats = cat_cols + cat_interaction_cols

# Frequency and Label Encoding
for col in all_cats:
    df_all[col] = df_all[col].astype(str).fillna("missing")
    # Frequency encoding
    freq_map = df_all[col].value_counts(normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq_map).astype(np.float32)
    # Label encoding
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])

# Group aggregate features on numeric variables
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
num_cols_for_agg = [c for c in num_cols if c != "id" and not c.endswith("_freq")]

for cat in cat_cols[:4]:
    group_stats = df_all.groupby(cat)[num_cols_for_agg].transform("mean")
    group_stats.columns = [f"{c}_mean_by_{cat}" for c in num_cols_for_agg]
    group_std = df_all.groupby(cat)[num_cols_for_agg].transform("std")
    group_std.columns = [f"{c}_std_by_{cat}" for c in num_cols_for_agg]
    df_all = pd.concat([df_all, group_stats, group_std], axis=1)

# Impute missing values for all numeric features
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    median_val = df_all[col].median()
    df_all[col] = df_all[col].fillna(median_val if not pd.isna(median_val) else 0)

# Drop non-predictive id column from training features
feature_cols = [c for c in df_all.columns if c != "id"]

# Split back to train and test sets
X_train = df_all.iloc[: len(train_df)][feature_cols].copy().reset_index(drop=True)
X_test = df_all.iloc[len(train_df) :][feature_cols].copy().reset_index(drop=True)

# 5-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1000,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.05,
    "depth": 6,
    "iterations": 1000,
    "random_seed": SEED,
    "verbose": False,
}

xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # Train LightGBM model
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Train CatBoost model
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )

    # Train XGBoost model
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    # Predictions for validation fold
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    oof_cb[val_idx] = cb_model.predict(X_va)
    oof_xgb[val_idx] = xgb_model.predict(X_va)

    # Predictions for test set
    test_lgb += lgb_model.predict(X_test) / kf.n_splits
    test_cb += cb_model.predict(X_test) / kf.n_splits
    test_xgb += xgb_model.predict(X_test) / kf.n_splits


# Optimize ensemble weights on OOF predictions
def ensemble_loss(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = w[0] * oof_lgb + w[1] * oof_cb + w[2] * oof_xgb
    clipped_oof = np.clip(blend_oof, 0, None)
    return np.sqrt(mean_squared_error(y_train, clipped_oof))


res = minimize(
    ensemble_loss,
    x0=[1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    method="SLSQP",
)

best_weights = res.x / np.sum(res.x)

# Evaluate final out-of-fold RMSE
oof_preds = np.clip(
    best_weights[0] * oof_lgb + best_weights[1] * oof_cb + best_weights[2] * oof_xgb,
    0,
    None,
)
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Validation RMSE: {oof_rmse:.5f}")

# Compute final test predictions
test_preds = np.clip(
    best_weights[0] * test_lgb + best_weights[1] * test_cb + best_weights[2] * test_xgb,
    0,
    None,
)

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame()
if "id" in test_df.columns:
    submission["id"] = test_df["id"]
else:
    submission["id"] = np.arange(len(test_df))
submission["fatalities"] = test_preds

submission.to_csv("./working/submission.csv", index=False)
