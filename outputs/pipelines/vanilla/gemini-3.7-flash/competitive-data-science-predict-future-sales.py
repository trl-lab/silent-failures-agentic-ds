import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'item_cnt_day'
y_train = train_df[target_col].values

# Feature engineering function
def preprocess_df(df):
    df = df.copy()
    if 'date' in df.columns:
        date_series = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        df['year'] = date_series.dt.year.fillna(-1).astype(int)
        df['month'] = date_series.dt.month.fillna(-1).astype(int)
        df['day'] = date_series.dt.day.fillna(-1).astype(int)
        df['dayofweek'] = date_series.dt.dayofweek.fillna(-1).astype(int)
        df.drop(columns=['date'], inplace=True)
    return df

X_train_full = preprocess_df(train_df.drop(columns=[target_col]))
X_test_full = preprocess_df(test_df)

# Use common features available in both train and test
ignore_cols = {'id', 'ID', target_col}
feature_cols = [col for col in X_train_full.columns if col in X_test_full.columns and col not in ignore_cols]

X_train = X_train_full[feature_cols].copy()
X_test = X_test_full[feature_cols].copy()

# Categorical column handling
for col in feature_cols:
    if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col]):
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# K-Fold Cross Validation and Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'item_cnt_day': test_preds
})
output_df.to_csv('predictions.csv', index=False)