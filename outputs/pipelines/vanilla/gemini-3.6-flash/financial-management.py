import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def parse_formatted_series(series):
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).str.strip()
        if len(sample) == 0:
            return None
        is_pct = sample.str.endswith('%')
        cleaned = sample.str.replace('%', '', regex=False)\
                        .str.replace('$', '', regex=False)\
                        .str.replace(',', '', regex=False)\
                        .str.replace('(', '-', regex=False)\
                        .str.replace(')', '', regex=False)\
                        .str.strip()
        parsed = pd.to_numeric(cleaned, errors='coerce')
        if parsed.notna().sum() / max(len(sample), 1) > 0.7:
            all_cleaned = series.astype(str).str.replace('%', '', regex=False)\
                                            .str.replace('$', '', regex=False)\
                                            .str.replace(',', '', regex=False)\
                                            .str.replace('(', '-', regex=False)\
                                            .str.replace(')', '', regex=False)\
                                            .str.strip()
            num_res = pd.to_numeric(all_cleaned, errors='coerce')
            if is_pct.mean() > 0.5:
                num_res = num_res / 100.0
            return num_res
    return None

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "State Share"

if train_df[target_col].dtype == 'object' or pd.api.types.is_string_dtype(train_df[target_col]):
    parsed_y = parse_formatted_series(train_df[target_col])
    if parsed_y is not None:
        y_all = parsed_y.values
    else:
        y_all = pd.to_numeric(train_df[target_col], errors='coerce').values
else:
    y_all = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

valid_mask = ~np.isnan(y_all)
if not valid_mask.all():
    y = y_all[valid_mask]
    X_train = X_train.iloc[valid_mask].reset_index(drop=True)
else:
    y = y_all

for col in ['id', 'Id', 'ID', 'Index', 'index']:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

for col in list(df_all.columns):
    parsed_num = parse_formatted_series(df_all[col])
    if parsed_num is not None:
        df_all[col] = parsed_num

for col in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        dt_col = df_all[col]
        df_all[f'{col}_year'] = dt_col.dt.year
        df_all[f'{col}_month'] = dt_col.dt.month
        df_all[f'{col}_quarter'] = dt_col.dt.quarter
        df_all[f'{col}_day'] = dt_col.dt.day
        df_all.drop(columns=[col], inplace=True)
    elif 'date' in col.lower() or 'year' in col.lower() or 'quarter' in col.lower():
        try:
            dt_col = pd.to_datetime(df_all[col], errors='coerce')
            if dt_col.notna().sum() > 0.5 * len(df_all):
                df_all[f'{col}_year'] = dt_col.dt.year
                df_all[f'{col}_month'] = dt_col.dt.month
                df_all[f'{col}_quarter'] = dt_col.dt.quarter
                df_all[f'{col}_day'] = dt_col.dt.day
        except Exception:
            pass

num_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
cat_cols = [c for c in df_all.columns if c not in num_cols]

for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna('missing')

num_cols_orig = list(num_cols)
if 2 <= len(num_cols_orig) <= 40:
    for i in range(len(num_cols_orig)):
        for j in range(i + 1, len(num_cols_orig)):
            c1, c2 = num_cols_orig[i], num_cols_orig[j]
            df_all[f'{c1}_minus_{c2}'] = df_all[c1] - df_all[c2]
            denom = df_all[c2].replace(0, np.nan)
            df_all[f'{c1}_ratio_{c2}'] = (df_all[c1] / (denom.abs() + 1e-5)).fillna(0)

num_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]

for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True).to_dict()
    df_all[f'{col}_freq'] = df_all[col].map(freq).astype(np.float32)

constant_cols = [col for col in df_all.columns if df_all[col].nunique(dropna=False) <= 1]
if constant_cols:
    df_all.drop(columns=constant_cols, inplace=True)

cat_cols = [c for c in cat_cols if c in df_all.columns]
num_cols = [