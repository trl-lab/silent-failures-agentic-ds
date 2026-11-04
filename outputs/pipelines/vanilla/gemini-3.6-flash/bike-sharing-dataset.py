import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y_train = train_df["cnt"].values
y_log = np.log1p(y_train)

def process_features(df):
    df = df.copy()
    drop_cols = ["cnt", "casual", "registered", "instant"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    if "dteday" in df.columns:
        dteday = pd.to_datetime(df["dteday"])
        df["dt_year"] = dteday.dt.year
        df["dt_month"] = dteday.dt.month
        df["dt_day"] = dteday.dt.day
        df["dt_dayofweek"] = dteday.dt.dayofweek
        df["dt_dayofyear"] = dteday.dt.dayofyear
        df["dt_quarter"] = dteday.dt.quarter
        df["dt_is_weekend"] = dteday.dt.dayofweek.isin([5, 6]).astype(int)
        df = df.drop(columns=["dteday"])
        
    if "hr" in df.columns:
        df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24.0)
        df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24.0)
        
    if "mnth" in df.columns:
        df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12.0)
        df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12.0)
        
    if "weekday" in df.columns:
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
        
    if "temp" in df.columns and "atemp" in df.columns:
        df["temp_atemp_diff"] = df["atemp"] - df["temp"]
        
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    return df

X_train = process_features(train_df)
X_test = process_features(test_df)

X_train, X_test = X_train.align(X_test, join="left", axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_log[train_idx]
    
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb.fit(X_tr, y_tr)
    preds_lgb += lgb.predict(X_test) / kf.n_splits
    
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)
    preds_xgb += xgb.predict(X_test) / kf.n_splits
    
    cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=7,
        random_seed=42,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    preds_cat += cat.predict(X_test) / kf.n_splits

final_log_preds = 0.4 * preds_lgb + 0.3 * preds_xgb + 0.3 * preds_cat
final_preds = np.expm1(final_log_preds)
final_preds = np.clip(final_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "cnt": final_preds
})
submission.to_csv("predictions.csv", index=False)