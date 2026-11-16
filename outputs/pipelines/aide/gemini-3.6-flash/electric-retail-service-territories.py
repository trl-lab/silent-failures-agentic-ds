import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Set paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_PATH = "./working/submission.csv"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "TOTAL_MWH"
y = train_df[target_col].values
y_log = np.log1p(np.maximum(0, y))

# Combine datasets for consistent feature engineering
train_df["is_train"] = 1
test_df["is_train"] = 0
test_df[target_col] = np.nan

df = pd.concat([train_df, test_df], ignore_index=True)

# Identify feature types
ignore_cols = [target_col, "is_train", "id"]
feature_cols = [c for c in df.columns if c not in ignore_cols]

cat_cols = [
    c for c in feature_cols if df[c].dtype == "object" or df[c].dtype.name == "category"
]
num_cols = [c for c in feature_cols if c not in cat_cols]

# 1. Log-transform numerical features to reduce skewness
log_num_cols = []
for col in num_cols:
    log_col = f"{col}_log"
    df[log_col] = np.log1p(np.maximum(0, df[col].fillna(0)))
    log_num_cols.append(log_col)

# 2. Construct numerical pairwise ratio and interaction features among primary numericals
for i in range(len(num_cols)):
    for j in range(i + 1, min(i + 4, len(num_cols))):
        c1, c2 = num_cols[i], num_cols[j]
        df[f"{c1}_ratio_{c2}"] = (df[c1].fillna(0) + 1e-5) / (df[c2].fillna(0) + 1e-5)

# 3. Frequency encoding for categoricals
for col in cat_cols:
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)

# 4. Group summary statistics across key categorical groupings using log-transformed numericals
group_cols = [c for c in ["STATE", "TYPE", "NERC"] if c in cat_cols]
for g_col in group_cols:
    for n_col in log_num_cols[:8]:
        grouped = df.groupby(g_col)[n_col].agg(["mean", "std"]).reset_index()
        grouped.columns = [g_col, f"{g_col}_{n_col}_mean", f"{g_col}_{n_col}_std"]
        df = df.merge(grouped, on=g_col, how="left")

# Convert categoricals to category dtype for LightGBM, CatBoost, and XGBoost
for col in cat_cols:
    df[col] = df[col].astype(str).fillna("missing")
    df[col] = df[col].astype("category")

# Separate train and test
train_clean = df[df["is_train"] == 1].drop(columns=["is_train", target_col])
test_clean = df[df["is_train"] == 0].drop(columns=["is_train", target_col])

# Features for modeling
train_features = [c for c in train_clean.columns if c != "id"]

X = train_clean[train_features]
X_test = test_clean[train_features]

# Prepare CatBoost categorical indices
cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(train_clean))
oof_preds_cb = np.zeros(len(train_clean))
oof_preds_xgb = np.zeros(len(train_clean))

test_preds_lgb = np.zeros(len(test_clean))
test_preds_cb = np.zeros(len(test_clean))
test_preds_xgb = np.zeros(len(test_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
    X_tr, y_tr_log = X.iloc[train_idx], y_log[train_idx]
    X_va, y_va_log = X.iloc[val_idx], y_log[val_idx]

    # LightGBM Model
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model_lgb.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred_lgb_log = model_lgb.predict(X_va)
    oof_preds_lgb[val_idx] = np.maximum(0, np.expm1(val_pred_lgb_log))
    test_preds_lgb += np.maximum(0, np.expm1(model_lgb.predict(X_test))) / kf.n_splits

    # CatBoost Model
    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
        cat_features=cat_indices,
    )

    model_cb.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log))

    val_pred_cb_log = model_cb.predict(X_va)
    oof_preds_cb[val_idx] = np.maximum(0, np.expm1(val_pred_cb_log))
    test_preds_cb += np.maximum(0, np.expm1(model_cb.predict(X_test))) / kf.n_splits

    # XGBoost Model
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=50,
        n_jobs=-1,
    )

    model_xgb.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        verbose=False,
    )

    val_pred_xgb_log = model_xgb.predict(X_va)
    oof_preds_xgb[val_idx] = np.maximum(0, np.expm1(val_pred_xgb_log))
    test_preds_xgb += np.maximum(0, np.expm1(model_xgb.predict(X_test))) / kf.n_splits


# Optimize ensemble weights using SciPy minimize
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_preds_lgb + w2 * oof_preds_cb + w3 * oof_preds_xgb
    return np.sqrt(mean_squared_error(y, pred))


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x

oof_blend = (
    best_weights[0] * oof_preds_lgb
    + best_weights[1] * oof_preds_cb
    + best_weights[2] * oof_preds_xgb
)
overall_rmse = np.sqrt(mean_squared_error(y, oof_blend))

print(
    f"Optimal weights: LGB={best_weights[0]:.4f}, CB={best_weights[1]:.4f}, XGB={best_weights[2]:.4f}"
)
print(f"Validation Metric (CV RMSE): {overall_rmse:.4f}")

# Final Test Predictions
final_test_preds = (
    best_weights[0] * test_preds_lgb
    + best_weights[1] * test_preds_cb
    + best_weights[2] * test_preds_xgb
)
final_test_preds = np.maximum(0, final_test_preds)

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "TOTAL_MWH": final_test_preds}
)
submission.to_csv(SUBMISSION_PATH, index=False)
