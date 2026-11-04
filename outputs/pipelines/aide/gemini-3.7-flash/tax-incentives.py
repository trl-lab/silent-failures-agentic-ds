import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "credit_amount"
features = [col for col in train_df.columns if col != target_col]

n_train = len(train_df)
combined = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# 2. Preprocessing & Feature Engineering
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()

# Group aggregations (mean, std, min, max) of numerical features grouped by categorical features
for cat_col in cat_cols:
    if combined[cat_col].nunique() <= 100:  # Avoid excessive cardinality explosion
        for num_col in num_cols:
            grouped = combined.groupby(cat_col)[num_col].agg(
                ["mean", "std", "min", "max"]
            )
            grouped.columns = [
                f"{cat_col}_{num_col}_{stat}" for stat in ["mean", "std", "min", "max"]
            ]
            combined = combined.merge(grouped, on=cat_col, how="left")
            # Difference and ratio from group mean
            combined[f"{num_col}_diff_{cat_col}_mean"] = (
                combined[num_col] - combined[f"{cat_col}_{num_col}_mean"]
            )

# Frequency encoding
for col in cat_cols:
    freq = combined[col].value_counts(dropna=False)
    combined[f"{col}_freq"] = combined[col].map(freq)

# Numerical interactions: pairwise multiplication and division ratios
if len(num_cols) >= 2:
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            combined[f"{c1}_mult_{c2}"] = combined[c1] * combined[c2]
            combined[f"{c1}_div_{c2}"] = combined[c1] / (combined[c2].abs() + 1e-5)
            combined[f"{c2}_div_{c1}"] = combined[c2] / (combined[c1].abs() + 1e-5)

# Ordinal encoding for categoricals
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined[cat_cols] = oe.fit_transform(combined[cat_cols].astype(str))

# Clean column names for LightGBM
combined.columns = [
    str(c).replace(":", "_").replace(" ", "_").replace(",", "_")
    for c in combined.columns
]

# Split back to train and test
X = combined.iloc[:n_train].copy()
y = train_df[target_col].values
X_test = combined.iloc[n_train:].copy()

# 3. Cross-Validation & Multi-Model Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
oof_hgb = np.zeros(n_train)

test_xgb = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))
test_hgb = np.zeros(len(test_df))

xgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "rmse",
}

lgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]

    # Model 1: XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / n_splits

    # Model 2: LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / n_splits

    # Model 3: HistGradientBoosting
    model_hgb = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=40,
        l2_regularization=0.5,
        random_state=42 + fold,
        early_stopping=True,
    )
    model_hgb.fit(X_tr, y_tr)
    oof_hgb[val_idx] = model_hgb.predict(X_va)
    test_hgb += model_hgb.predict(X_test) / n_splits


# 4. Find optimal blending weights
def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = w[0] * oof_xgb + w[1] * oof_lgb + w[2] * oof_hgb
    pred = np.clip(pred, 0, None)
    return root_mean_squared_error(y, pred)


res = minimize(loss_func, [0.5, 0.4, 0.1], bounds=[(0, 1), (0, 1), (0, 1)])
w_opt = res.x / np.sum(res.x)

oof_ensemble = np.clip(
    w_opt[0] * oof_xgb + w_opt[1] * oof_lgb + w_opt[2] * oof_hgb, 0, None
)
test_preds = np.clip(
    w_opt[0] * test_xgb + w_opt[1] * test_lgb + w_opt[2] * test_hgb, 0, None
)

cv_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"Optimal weights [XGB, LGB, HGB]: {np.round(w_opt, 4)}")
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# 5. Save Predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "credit_amount": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
