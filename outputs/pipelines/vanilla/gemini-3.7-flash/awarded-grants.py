import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Financial Assistance'
y = train_df[target_col].values.astype(np.float32)
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Datetime processing
date_cols = ['Project Period Start Date', 'Grant Project Period End Date', 'Data Warehouse Record Create Date']
for col in date_cols:
    if col in df_all.columns:
        dt_series = pd.to_datetime(df_all[col], errors='coerce')
        df_all[f'{col}_year'] = dt_series.dt.year.astype('float32')
        df_all[f'{col}_month'] = dt_series.dt.month.astype('float32')
        df_all[f'{col}_day'] = dt_series.dt.day.astype('float32')
        df_all[f'{col}_dayofweek'] = dt_series.dt.dayofweek.astype('float32')
        df_all[f'{col}_ts'] = dt_series.astype('int64') // 10**9
        df_all.loc[dt_series.isna(), f'{col}_ts'] = np.nan
        df_all.drop(columns=[col], inplace=True)

if 'Project Period Start Date_ts' in df_all.columns and 'Grant Project Period End Date_ts' in df_all.columns:
    df_all['project_duration_days'] = (df_all['Grant Project Period End Date_ts'] - df_all['Project Period Start Date_ts']) / 86400.0

# Drop redundant raw date text columns
drop_cols = [
    'Project Period Start Date Text String',
    'Grant Project Period End Date Text',
    'Data Warehouse Record Create Date Text'
]
for col in drop_cols:
    if col in df_all.columns:
        df_all.drop(columns=[col], inplace=True)

# Handle text / categorical columns
cat_cols = []
for col in df_all.columns:
    if df_all[col].dtype == 'object' or isinstance(df_all[col].dtype, pd.CategoricalDtype):
        cat_cols.append(col)
        # Add character length feature for text columns
        if col == 'Uniform Data System Grant Program Description' or 'Description' in col:
            df_all[f'{col}_len'] = df_all[col].astype(str).str.len()
        
        # Ordinal encode
        df_all[col] = df_all[col].astype(str).fillna('__MISSING__')
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_all[col] = oe.fit_transform(df_all[[col]]).astype(np.int32)

# Split back to train and test
X_train_clean = df_all.iloc[:n_train].copy()
X_test_clean = df_all.iloc[n_train:].copy()

# 5-fold cross validation with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_clean), dtype=np.float64)

for train_idx, val_idx in kf.split(X_train_clean, y):
    X_tr, y_tr = X_train_clean.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_clean.iloc[val_idx], y[val_idx]

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=cat_cols
    )
    
    test_preds += model.predict(X_test_clean) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Financial Assistance': test_preds
})
output_df.to_csv('predictions.csv', index=False)