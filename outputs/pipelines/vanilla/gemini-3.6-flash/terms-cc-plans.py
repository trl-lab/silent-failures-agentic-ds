import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "APR"

y_raw = train_df[target_col].astype(str).str.replace("%", "", regex=False).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
y_train = pd.to_numeric(y_raw, errors="coerce").values

valid_mask = ~np.isnan(y_train)
if not np.all(valid_mask):
    train_df = train_df.iloc[valid_mask].reset_index(drop=True)
    y_train = y_train[valid_mask]

train_df = train_df.drop(columns=[target_col], errors="ignore")
test_df = test_df.drop(columns=[target_col], errors="ignore")

n_train = len(train_df)
n_test = len(test_df)

df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

drop_cols = [c for c in df_all.columns if c.lower() in ["id", "index", "unnamed: 0"]]
if drop_cols:
    df_all = df_all.drop(columns=drop_cols)

datetime_cols = []
numeric_cols = []
cat_cols = []
text_cols = []

for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        datetime_cols.append(col)
        continue

    if df_all[col].dtype == "object" or df_all[col].dtype.name in ["category", "string"]:
        sample_s = df_all[col].dropna().astype(str)
        if len(sample_s) > 0:
            cleaned = sample_s.str.replace("$", "", regex=False).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
            num_converted = pd.to_numeric(cleaned, errors="coerce")
            if num_converted.notna().mean() > 0.8:
                df_all[col] = pd.to_numeric(df_all[col].astype(str).str.replace("$", "", regex=False).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip(), errors="coerce")
                numeric_cols.append(col)
                continue

    if pd.api.types.is_numeric_dtype(df_all[col]):
        numeric_cols.append(col)
    else:
        sample_s = df_all[col].fillna("").astype(str)
        avg_len = sample_s.str.len().mean()
        n_unique = sample_s.nunique()

        if avg_len > 100 or n_unique > 2000:
            text_cols.append(col)
        else:
            cat_cols.append(col)

for col in datetime_cols:
    dt = pd.to_datetime(df_all[col], errors="coerce")
    df_all[f"{col}_year"] = dt.dt.year
    df_all[f"{col}_month"] = dt.dt.month
    df_all[f"{col}_day"] = dt.dt