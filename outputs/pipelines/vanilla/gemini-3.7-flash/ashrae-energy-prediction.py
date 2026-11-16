import pandas as pd
import numpy as np
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target
y_train = np.log1p(np.clip(train_df['meter_reading'].values, 0, None))

# Feature engineering
def preprocess(df):
    df = df.copy()
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'])
        df['hour'] = dt.dt.hour.astype(np.int8)
        df['day'] = dt.dt.day.astype(np.int8)
        df['dayofweek'] = dt.dt.dayofweek.astype(np.int8)
        df['month'] = dt.dt.month.astype(np.int8)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(np.int8)
        df = df.drop(columns=['timestamp'])
    return df

X_train = preprocess(train_df.drop(columns=['meter_reading']))
X_test = preprocess(test_df)

# Align columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Identify categorical features
cat_cols = ['building_id', 'meter', 'dayofweek', 'month', 'hour', 'is_weekend', 'primary_use', 'site_id']
cat_cols = [c for c in cat_cols if c in common_cols]
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Train LightGBM model
model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=300,
    learning_rate=0.08,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict and transform back from log space
preds_log = model.predict(X_test)
preds = np.clip(np.expm1(preds_log), 0, None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(preds)),
    'meter_reading': preds
})
output_df.to_csv('predictions.csv', index=False)