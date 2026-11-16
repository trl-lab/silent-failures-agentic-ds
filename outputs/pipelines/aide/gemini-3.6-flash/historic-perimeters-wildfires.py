import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "gisacres"

y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

n_train = len(train_df)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Feature Engineering: Extract datetime features and frequency encodings
for col in list(df_all.columns):
    if df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col])
            if parsed.notnull().sum() > 0.5 * len(df_all):
                df_all[col + "_year"] = parsed.dt.year
                df_all[col + "_month"] = parsed.dt.month
                df_all[col + "_day"] = parsed.dt.day
                df_all[col + "_dayofweek"] = parsed.dt.dayofweek
                df_all[col + "_dayofyear"] = parsed.dt.dayofyear
        except Exception:
            pass

cat_cols = []
for col in list(df_all.columns):
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        cat_cols.append(col)
        freq = df_all[col].value_counts()
        df_all[col + "_freq"] = df_all[col].map(freq)

for col in cat_cols:
    df_all[col] = df_all[col].astype(str)
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_all[col] = oe.fit_transform(df_all[[col]])

X_train_proc = df_all.iloc[:n_train].copy()
X_test_proc = df_all.iloc[n_train:].copy()

y_train_log = np.log1p(np.maximum(0, y_train))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(n_train)
oof_preds_cat = np.zeros(n_train)
oof_preds_xgb = np.zeros(n_train)

test_preds_lgb = np.zeros(len(test_df))
test_preds_cat = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train_log)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train_log[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train_log[val_idx]

    # 1. LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test_proc) / 5.0

    # 2. CatBoost Regressor
    model_cat = CatBoostRegressor(
        iterations=1000, learning_rate=0.04, depth=6, random_seed=42 + fold, verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    oof_preds_cat[val_idx] = model_cat.predict(X_va)
    test_preds_cat += model_cat.predict(X_test_proc) / 5.0

    # 3. XGBoost Regressor
    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test_proc) / 5.0

# Blend models
oof_preds_log = (oof_preds_lgb + oof_preds_cat + oof_preds_xgb) / 3.0
test_preds_log = (test_preds_lgb + test_preds_cat + test_preds_xgb) / 3.0

# Inverse transform predictions
oof_preds = np.expm1(np.maximum(0, oof_preds_log))
test_preds = np.expm1(np.maximum(0, test_preds_log))

rmse_log = root_mean_squared_error(y_train_log, oof_preds_log)
print(f"Validation RMSE (log scale): {rmse_log:.4f}")

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "gisacres": test_preds})
submission.to_csv("./working/submission.csv", index=False)
