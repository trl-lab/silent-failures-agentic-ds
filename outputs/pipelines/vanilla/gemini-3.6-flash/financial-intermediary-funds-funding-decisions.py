```python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Amount in USD"
    y_train = train_df[target_col].values.astype(np.float64)
    X_train_df = train_df.drop(columns=[target_col])
    X_test_df = test_df.copy()

    n_train = len(train_df)
    n_test = len(test_df)

    df_all = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)

    date_cols = []
    cat_cols = []
    num_cols = []

    for col in df_all.columns:
        if pd.api.types.is_datetime64_any_dtype(df_all[col]):
            date_cols.append(col)
        elif df_all[col].dtype == 'object' or isinstance(df_all[col].dtype, pd.CategoricalDtype):
            converted = pd.to_datetime(df_all[col], errors='coerce')
            if converted.notna().mean() > 0.5:
                df_all[col] = converted
                date_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            num_cols.append(col)

    # Extract date features
    for col in date_cols:
        df_all[col + '_year'] = df_all[col].dt.year
        df_all[col + '_month'] = df_all[col].dt.month
        df_all[col + '_day'] = df_all[col].dt.day
        df_all[col + '_dayofweek'] = df_all[col].dt.dayofweek
        df_all[col + '_quarter'] = df_all[col].dt.quarter
        valid_ts = df_all[col].astype(np.int64) // 10**9
        df_all[col + '_ts'] = np.where(df_all[col].isna(), np.nan, valid_ts)
        df_all.drop(columns=[col], inplace=True)

    # Frequency encoding for categoricals
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        freq = df_all[col].value_counts(normalize=True).to_dict()
        df_all[col + '_freq'] = df_all[col].map(freq).astype(np.float32)

    X_train = df_all.iloc[:n_train].copy()
    X_test = df_all.iloc[n_train:].copy()

    # Out-of-fold target encoding
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = np.mean(y_train)
    smooth = 10.0

    for col in cat_cols:
        te_train = np.zeros(n_train, dtype=np.float32)
        for tr_idx, val_idx in kf.split(X_train, y_train):
            sub_cat = X_train.iloc[tr_idx][col]
            sub_y = y_train[tr_idx]
            val_cat = X_train.iloc[val_idx][col]

            stats = pd.DataFrame({'cat': sub_cat, 'y': sub_y}).groupby('cat')['y'].agg(['count', 'mean'])
            te