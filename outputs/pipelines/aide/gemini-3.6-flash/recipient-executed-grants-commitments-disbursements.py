import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Disbursements USD (Trust funds)"
y = train[target_col].values
X_train = train.drop(columns=[target_col])
X_test = test.copy()

n_train = len(train)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# 2. Feature Engineering & Preprocessing
datetime_cols = []
numeric_cols = []
categorical_cols = []

for col in df_all.columns:
    if col.lower() in ["id", "index"]:
        continue
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        datetime_cols.append(col)
    elif df_all[col].dtype == "object":
        if any(kw in col.lower() for kw in ["date", "time", "year", "as of"]):
            try:
                df_all[col] = pd.to_datetime(df_all[col])
                datetime_cols.append(col)
                continue
            except Exception:
                pass
        categorical_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_all[col]):
        if "year" in col.lower() and df_all[col].nunique() < 100:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

# Process Datetime Features
for col in datetime_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_timestamp"] = df_all[col].astype("int64") // 10**9
    df_all.drop(columns=[col], inplace=True)

# Process Categorical Features
for col in categorical_cols:
    df_all[col] = df_all[col].astype(str)
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    df_all[col] = df_all[col].astype("category").cat.codes

# Process Numeric Features
num_feats = [c for c in numeric_cols if c.lower() not in ["id", "index"]]
for col in num_feats:
    if df_all[col].isnull().sum() > 0:
        df_all[f"{col}_isna"] = df_all[col].isnull().astype(int)
    min_val = df_all[col].min()
    if min_val >= 0:
        df_all[f"{col}_log1p"] = np.log1p(df_all[col].fillna(0))

# Group Summary Aggregations
if len(categorical_cols) > 0 and len(num_feats) > 0:
    for cat_col in categorical_cols[:3]:
        for num_col in num_feats[:5]:
            grouped = (
                df_all.groupby(cat_col)[num_col].agg(["mean", "std"]).reset_index()
            )
            grouped.columns = [
                cat_col,
                f"{num_col}_by_{cat_col}_mean",
                f"{num_col}_by_{cat_col}_std",
            ]
            df_all = df_all.merge(grouped, on=cat_col, how="left")

# Prepare final feature matrices
feature_cols = [
    c
    for c in df_all.columns
    if c.lower() not in ["id", "index"] and pd.api.types.is_numeric_dtype(df_all[c])
]
X_all = df_all[feature_cols].copy().fillna(-999)

X_train_proc = X_all.iloc[:n_train].reset_index(drop=True)
X_test_proc = X_all.iloc[n_train:].reset_index(drop=True)

# 3. Validation & Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb_raw = np.zeros(n_train)
oof_lgb_log = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

test_preds_lgb_raw = np.zeros(len(X_test_proc))
test_preds_lgb_log = np.zeros(len(X_test_proc))
test_preds_xgb = np.zeros(len(X_test_proc))

log_y = np.log1p(np.clip(y, 0, None))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]
    log_y_tr, log_y_va = log_y[train_idx], log_y[val_idx]

    # Model 1: LightGBM (Raw Target)
    model_lgb1 = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_lgb1.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb_raw[val_idx] = model_lgb1.predict(X_va)
    test_preds_lgb_raw += model_lgb1.predict(X_test_proc) / kf.n_splits

    # Model 2: LightGBM (Log Target)
    model_lgb2 = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=142 + fold,
        n_jobs=-1,
    )
    model_lgb2.fit(
        X_tr,
        log_y_tr,
        eval_set=[(X_va, log_y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb_log[val_idx] = np.expm1(model_lgb2.predict(X_va))
    test_preds_lgb_log += np.expm1(model_lgb2.predict(X_test_proc)) / kf.n_splits

    # Model 3: XGBoost (Raw Target)
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=242 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test_proc) / kf.n_splits

# Enforce target lower bound if non-negative
if np.min(y) >= 0:
    oof_lgb_raw = np.clip(oof_lgb_raw, 0, None)
    oof_lgb_log = np.clip(oof_lgb_log, 0, None)
    oof_xgb = np.clip(oof_xgb, 0, None)
    test_preds_lgb_raw = np.clip(test_preds_lgb_raw, 0, None)
    test_preds_lgb_log = np.clip(test_preds_lgb_log, 0, None)
    test_preds_xgb = np.clip(test_preds_xgb, 0, None)

# Optimize Ensemble Weights
best_rmse = float("inf")
best_weights = (0.33, 0.33, 0.34)

for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, 21):
        w3 = max(0.0, round(1.0 - w1 - w2, 4))
        blend_oof = w1 * oof_lgb_raw + w2 * oof_lgb_log + w3 * oof_xgb
        score = np.sqrt(mean_squared_error(y, blend_oof))
        if score < best_rmse:
            best_rmse = score
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
final_oof = w1 * oof_lgb_raw + w2 * oof_lgb_log + w3 * oof_xgb
final_test_preds = (
    w1 * test_preds_lgb_raw + w2 * test_preds_lgb_log + w3 * test_preds_xgb
)

val_rmse = np.sqrt(mean_squared_error(y, final_oof))
print(f"Validation RMSE: {val_rmse:.4f}")

# 4. Generate Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test)), target_col: final_test_preds})
submission.to_csv("./working/submission.csv", index=False)
