import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "TOTAL_ACTIVE_ACCOUNTS_"
assert target_col in train_df.columns, f"Target column {target_col} not in train_df"

y = train_df[target_col].values.astype(float)
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Separate column types
cat_cols = [
    col
    for col in combined.columns
    if combined[col].dtype == "object" or combined[col].dtype.name == "category"
]
num_cols = [col for col in combined.columns if col not in cat_cols]

# Frequency Encoding for categorical variables
for c in cat_cols:
    freq = combined[c].astype(str).value_counts()
    combined[f"{c}_freq"] = combined[c].astype(str).map(freq).fillna(0).astype(float)

# Aggregations on numerical features grouped by categoricals
group_cols = [
    c for c in cat_cols if combined[c].nunique() > 1 and combined[c].nunique() < 200
][:3]
for g_col in group_cols:
    for n_col in num_cols[:5]:
        agg = combined.groupby(g_col, observed=False)[n_col].transform("mean")
        combined[f"{g_col}_{n_col}_mean"] = agg
        agg_std = (
            combined.groupby(g_col, observed=False)[n_col].transform("std").fillna(0)
        )
        combined[f"{g_col}_{n_col}_std"] = agg_std

# Pairwise numerical interactions
for i in range(len(num_cols)):
    for j in range(i + 1, min(i + 4, len(num_cols))):
        c1, c2 = num_cols[i], num_cols[j]
        combined[f"{c1}_ratio_{c2}"] = combined[c1] / (combined[c2].abs() + 1e-5)

# Ensure string category columns are category type for tree models
for c in cat_cols:
    combined[c] = combined[c].astype(str).astype("category")

X_all = combined.iloc[:n_train].copy()
X_test_all = combined.iloc[n_train:].copy()

# Setup Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

test_preds_lgb = np.zeros(len(test_df))
test_preds_cb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
    X_tr, y_tr = X_all.iloc[train_idx].copy(), y[train_idx]
    X_va, y_va = X_all.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test_all.copy()

    global_mean = float(np.mean(y_tr))

    # Safely compute out-of-fold target encoding without dtype mismatch bugs
    for c in cat_cols:
        te_map = pd.Series(y_tr).groupby(X_tr[c].astype(str).values).mean().to_dict()
        X_tr[f"{c}_te"] = (
            X_tr[c].astype(str).map(te_map).astype(float).fillna(global_mean)
        )
        X_va[f"{c}_te"] = (
            X_va[c].astype(str).map(te_map).astype(float).fillna(global_mean)
        )
        X_te[f"{c}_te"] = (
            X_te[c].astype(str).map(te_map).astype(float).fillna(global_mean)
        )

    # Convert categoricals to ordinal codes for XGBoost
    X_tr_xgb = X_tr.copy()
    X_va_xgb = X_va.copy()
    X_te_xgb = X_te.copy()
    for c in cat_cols:
        X_tr_xgb[c] = X_tr_xgb[c].cat.codes
        X_va_xgb[c] = X_va_xgb[c].cat.codes
        X_te_xgb[c] = X_te_xgb[c].cat.codes

    # 1. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = np.clip(lgb_model.predict(X_va), 0, None)
    test_preds_lgb += np.clip(lgb_model.predict(X_te), 0, None) / kf.n_splits

    # 2. CatBoost
    cat_cols_list = list(cat_cols)
    cb_model = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6, random_seed=42 + fold, verbose=0
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_cols_list,
        early_stopping_rounds=50,
    )
    oof_cb[val_idx] = np.clip(cb_model.predict(X_va), 0, None)
    test_preds_cb += np.clip(cb_model.predict(X_te), 0, None) / kf.n_splits

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[val_idx] = np.clip(xgb_model.predict(X_va_xgb), 0, None)
    test_preds_xgb += np.clip(xgb_model.predict(X_te_xgb), 0, None) / kf.n_splits


# Find optimal weights for ensembling
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
    return root_mean_squared_error(y, pred)


res = minimize(
    loss_func,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)
best_weights = res.x
best_rmse = res.fun

print(
    f"Optimal Blend Weights (LGB, CB, XGB): {best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}"
)
print(f"Validation RMSE: {best_rmse:.5f}")

# Final predictions
w1, w2, w3 = best_weights
final_test_preds = np.clip(
    w1 * test_preds_lgb + w2 * test_preds_cb + w3 * test_preds_xgb, 0, None
)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), target_col: final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
