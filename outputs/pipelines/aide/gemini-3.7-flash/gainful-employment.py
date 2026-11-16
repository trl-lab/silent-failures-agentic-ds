import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb
import xgboost as xgb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Debt-to-Earnings Annual Rate"

y = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

common_cols = [c for c in X_train_raw.columns if c in X_test_raw.columns]
X_train = X_train_raw[common_cols].copy()
X_test = X_test_raw[common_cols].copy()

# 2. Identify column types
cat_cols = []
num_cols = []
for col in common_cols:
    if (
        X_train[col].dtype == "object"
        or pd.api.types.is_string_dtype(X_train[col])
        or pd.api.types.is_categorical_dtype(X_train[col])
    ):
        cat_cols.append(col)
    else:
        cat_cols_candidate = False
        try:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            num_cols.append(col)
        except Exception:
            cat_cols.append(col)

# 3. Feature Engineering
# (a) Missing value indicators / counts
X_train["missing_count"] = X_train.isnull().sum(axis=1)
X_test["missing_count"] = X_test.isnull().sum(axis=1)

# (b) Frequency encoding for categoricals
for col in cat_cols:
    freq = (
        pd.concat([X_train[col], X_test[col]], axis=0)
        .value_counts(normalize=True, dropna=False)
        .to_dict()
    )
    X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0).astype(float)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0).astype(float)

# (c) Group-level statistical aggregations on top categorical columns
key_cats = [c for c in cat_cols if 2 <= X_train[c].nunique(dropna=False) <= 500][:3]
top_nums = num_cols[:4] if len(num_cols) >= 4 else num_cols

for cat in key_cats:
    for num in top_nums:
        combined = pd.concat([X_train[[cat, num]], X_test[[cat, num]]], axis=0)
        group_mean = combined.groupby(cat, observed=False)[num].mean().to_dict()
        group_std = combined.groupby(cat, observed=False)[num].std().to_dict()

        X_train[f"{cat}_{num}_mean"] = X_train[cat].map(group_mean).astype(float)
        X_test[f"{cat}_{num}_mean"] = X_test[cat].map(group_mean).astype(float)
        X_train[f"{cat}_{num}_std"] = X_train[cat].map(group_std).astype(float)
        X_test[f"{cat}_{num}_std"] = X_test[cat].map(group_std).astype(float)

# (d) Out-of-fold Target Encoding for categoricals
if len(cat_cols) > 0:
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).fillna("__missing__")
        X_test[col] = X_test[col].astype(str).fillna("__missing__")

    target_enc = TargetEncoder(smooth="auto", cv=5, random_state=42)
    te_train = target_enc.fit_transform(X_train[cat_cols], y)
    te_test = target_enc.transform(X_test[cat_cols])

    for idx, col in enumerate(cat_cols):
        X_train[f"{col}_te"] = te_train[:, idx]
        X_test[f"{col}_te"] = te_test[:, idx]

# (e) Categorical formatting for LightGBM and XGBoost
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 4. 5-Fold Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_train_f, y_train_f = X_train.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X_train.iloc[val_idx], y.iloc[val_idx]

    # --- LightGBM ---
    train_data = lgb.Dataset(X_train_f, label=y_train_f, categorical_feature=cat_cols)
    val_data = lgb.Dataset(
        X_val_f, label=y_val_f, reference=train_data, categorical_feature=cat_cols
    )

    model_lgb = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_lgb[val_idx] = model_lgb.predict(
        X_val_f, num_iteration=model_lgb.best_iteration
    )
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )

    model_xgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )

    oof_xgb[val_idx] = model_xgb.predict(X_val_f)
    test_preds_xgb += model_xgb.predict(X_test) / n_splits

# 5. Model Ensembling & Evaluation
oof_ensemble = 0.5 * oof_lgb + 0.5 * oof_xgb
final_test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb

cv_lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
cv_xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
cv_ensemble_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))

print(f"5-Fold CV LightGBM RMSE: {cv_lgb_rmse:.5f}")
print(f"5-Fold CV XGBoost RMSE:  {cv_xgb_rmse:.5f}")
print(f"5-Fold CV Ensemble RMSE: {cv_ensemble_rmse:.5f}")

# 6. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), target_col: final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
