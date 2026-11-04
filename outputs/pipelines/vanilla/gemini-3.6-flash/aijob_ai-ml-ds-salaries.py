```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "salary_in_usd"
    y_train = train_df[target_col].values.astype(np.float64)

    n_train = len(train_df)
    n_test = len(test_df)

    # Combine datasets for consistent preprocessing
    drop_cols = {target_col, "id", "ID", "index", "Index", "row_id", "Unnamed: 0"}
    common_cols = [c for c in train_df.columns if c in test_df.columns and c not in drop_cols]
    
    df = pd.concat([train_df[common_cols], test_df[common_cols]], ignore_index=True)

    # Domain-specific ordinal & flag features
    if 'experience_level' in df.columns:
        exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
        df['exp_level_num'] = df['experience_level'].map(exp_map).fillna(0).astype(int)

    if 'company_size' in df.columns:
        size_map = {'S': 1, 'M': 2, 'L': 3}
        df['company_size_num'] = df['company_size'].map(size_map).fillna(0).astype(int)

    if 'employee_residence' in df.columns and 'company_location' in df.columns:
        df['is_same_country'] = (df['employee_residence'] == df['company_location']).astype(int)
        df['is_us_company'] = (df['company_location'] == 'US').astype(int)
        df['is_us_residence'] = (df['employee_residence'] == 'US').astype(int)

    if 'job_title' in df.columns:
        jt = df['job_title'].astype(str).str.lower()
        df['jt_len'] = jt.str.len()
        df['jt_words'] = jt.str.split().str.len()
        df['jt_is_manager'] = jt.str.contains('manager|head|director|lead|chief|vp', regex=True).astype(int)
        df['jt_is_senior'] = jt.str.contains('senior|sr|principal', regex=True).astype(int)
        df['jt_is_ml'] = jt.str.contains('machine learning|ml|ai|vision|nlp|deep learning', regex=True).astype(int)
        df['jt_is_ds'] = jt.str.contains('data scien', regex=True).astype(int)
        df['jt_is_de'] = jt.str.contains('data engin', regex=True).astype(int)
        df['jt_is_analyst'] = jt.str.contains('analyst|analytics', regex=True).astype(int)
        df['jt_is_architect'] = jt.str.contains('architect', regex=True).astype(int)
        df['jt_is_research'] = jt.str.contains('research', regex=True).astype(int)

    # Identify categorical vs numerical columns
    cat_cols = []
    num_cols = []
    for