```python
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = None
for col in ['item_cnt_day', 'item_cnt_month', 'target']:
    if col in train.columns:
        target_col = col
        break

if target_col is None:
    candidate_cols = [c for c in train.columns if c not in test.columns and c.lower() not in ['id', 'index']]
    target_col = candidate_cols[0] if candidate_cols else train.columns[-1]

common_features = [c for c in train.columns if c in test.columns and c.lower() not in ['id', 'index', target_col]]

is_minimal_kaggle = (
    'shop_id' in train.columns and 'item_id' in train.columns and
    'shop_id' in test.columns and 'item_id' in