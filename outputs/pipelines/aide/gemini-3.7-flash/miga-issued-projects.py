import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Gross Issued Amount (US$)"
y = train_df[target_col].values
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()

# Combine for consistent feature engineering
df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
n_train = len(train_features)

# Identify datetime features
date_cols = []
for col in df_all.columns:
    if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
        try:
            converted = pd.to_datetime(df_all[col], errors="coerce")
            if converted.notnull().sum() > 0.3 * len(df_all):
                df_all[col] = converted
                date_cols.append(col)
        except Exception:
            pass

# Pairwise date differences
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        col1, col2 = date_cols[i], date_cols[j]
        diff_col = f"diff_days_{col1}_{col2}"
        df_all[diff_col] = (df_all[col1] - df_all[col2]).dt.total_seconds() / 86400.0

# Extract date components
for col in date_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all.drop(columns=[col], inplace=True)

# Separate categorical and numeric columns
cat_cols = []
num_cols = []

for col in df_all.columns:
    if (
        df_all[col].dtype == "object"
        or df_all[col].dtype.name == "category"
        or df_all[col].dtype == "bool"
    ):
        cat_cols.append(col)
        df_all[col] = df_all[col].fillna("Missing").astype(str)
    else:
        num_cols.append(col)
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Frequency encoding for categoricals
for col in cat_cols:
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Prepare CatBoost dataset
X_cb = df_all.copy()
cb_cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]
X_cb_train = X_cb.iloc[:n_train].reset_index(drop=True)
X_cb_test = X_cb.iloc[n_train:].reset_index(drop=True)

# Prepare Ordinal Encoded dataset for LightGBM and XGBoost
X_num = df_all.copy()
for col in cat_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_num[col] = encoder.fit_transform(X_num[[col]])
    X_num[col] = X_num[col].astype(int)

X_num_train = X_num.iloc[:n_train].reset_index(drop=True)
X_num_test = X_num.iloc[n_train:].reset_index(drop=True)

# K-Fold validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

test_cb = np.zeros(len(X_cb_test))
test_lgb = np.zeros(len(X_num_test))
test_xgb = np.zeros(len(X_num_test))

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.04,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1500,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_train, y)):
    # 1. CatBoost
    X_tr_cb, y_tr = X_cb_train.iloc[train_idx], y[train_idx]
    X_va_cb, y_va = X_cb_train.iloc[val_idx], y[val_idx]

    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(
        X_tr_cb,
        y_tr,
        cat_features=cb_cat_indices,
        eval_set=(X_va_cb, y_va),
        use_best_model=True,
    )
    oof_cb[val_idx] = np.clip(model_cb.predict(X_va_cb), 0, None)
    test_cb += np.clip(model_cb.predict(X_cb_test), 0, None) / n_splits

    # 2. LightGBM
    X_tr_num, X_va_num = X_num_train.iloc[train_idx], X_num_train.iloc[val_idx]
    lgb_train = lgb.Dataset(X_tr_num, label=y_tr, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(
        X_va_num, label=y_va, reference=lgb_train, categorical_feature=cat_cols
    )

    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = np.clip(model_lgb.predict(X_va_num), 0, None)
    test_lgb += np.clip(model_lgb.predict(X_num_test), 0, None) / n_splits

    # 3. XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr_num,
        y_tr,
        eval_set=[(X_va_num, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = np.clip(model_xgb.predict(X_va_num), 0, None)
    test_xgb += np.clip(model_xgb.predict(X_num_test), 0, None) / n_splits

print(f"CatBoost OOF RMSE: {root_mean_squared_error(y, oof_cb):.4f}")
print(f"LightGBM OOF RMSE: {root_mean_squared_error(y, oof_lgb):.4f}")
print(f"XGBoost OOF RMSE:  {root_mean_squared_error(y, oof_xgb):.4f}")


# Optimize blend weights to minimize RMSE
def objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    pred = weights[0] * oof_cb + weights[1] * oof_lgb + weights[2] * oof_xgb
    return root_mean_squared_error(y, pred)


res = minimize(
    objective,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    method="SLSQP",
)
w = res.x / np.sum(res.x)
print(
    f"Optimized Blend Weights (CatBoost, LightGBM, XGBoost): [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]"
)

oof_ensemble = np.clip(w[0] * oof_cb + w[1] * oof_lgb + w[2] * oof_xgb, 0, None)
final_oof_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"Overall Ensemble OOF Validation RMSE: {final_oof_rmse:.4f}")

# Generate test predictions
test_ensemble = np.clip(w[0] * test_cb + w[1] * test_lgb + w[2] * test_xgb, 0, None)

# Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_ensemble)), target_col: test_ensemble}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
