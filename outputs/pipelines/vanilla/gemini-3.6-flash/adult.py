import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

# 1. Load Data
train_path = "/path/to/user/workspace/ds_failures/datasets/uci/adult/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/adult/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "income"

# Save original target classes and create mapping
orig_classes = list(train[target_col].unique())
pos_candidates = [c for c in orig_classes if ">50K" in str(c) or ">50" in str(c)]
if pos_candidates:
    pos_label = pos_candidates[0]
    neg_label = [c for c in orig_classes if c != pos_label][0]
else:
    neg_label, pos_label = orig_classes[0], orig_classes[1]

label_map = {neg_label: 0, pos_label: 1}
inv_label_map = {0: neg_label, 1: pos_label}

y = train[target_col].map(label_map).values

# 2. Preprocess Features
X_train = train.drop(columns=[target_col]).copy()
X_test = test.drop(columns=[target_col]).copy() if target_col in test.columns else test.copy()

if "id" in X_train.columns:
    X_train = X_train.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

cat_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
num_cols = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]

# Clean categorical columns
for col in cat_cols:
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(str).str.strip()
        df_all[col] = df_all[col].replace({'?': 'Missing', 'nan': 'Missing', 'None': 'Missing', '': 'Missing'})

# Clean numeric columns
for col in num_cols:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Feature Engineering
df_all['capital_diff'] = df_all['capital-gain']