import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Score"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

n_train = len(X_train_raw)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

for col in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        df_all[f"{col}_year"] = df_all[col].dt.year
        df_all[f"{col}_month"] = df_all[col].dt.month
        df_all[f"{col}_day"] = df_all[col].dt.day
        df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
        df_all.drop(columns=[col], inplace=True)

cat_cols = []
for col in list(df_all.columns):
    if not pd.api.types.is_numeric_dtype(df_all[col]):
        cat_cols.append(col)

for col in cat_cols:
    df_all[col] = df_all[col].astype(str)
    freq = df_all[col].value_counts(normalize=True)
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    cat_type = pd.CategoricalDtype(categories=df_all[col].unique(), ordered=False)
    df_all[col] = df_all[col].astype(cat_type).cat.codes

clean_cols = [re.sub(r'[^\w]', '_', str(c)) for c in df_all.columns]
df_all.columns = clean_cols

X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / n_splits

    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    cat_preds += model_cat.predict(X_test) / n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='rmse',
        tree_method='hist'
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds += model_xgb.predict(X_test) / n_splits

final_preds = 0.4 * lgb_preds + 0.3 * cat_preds + 0.3 * xgb_preds

output_df = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "Score": final_preds
})

output_df.to_csv("predictions.csv", index=False)