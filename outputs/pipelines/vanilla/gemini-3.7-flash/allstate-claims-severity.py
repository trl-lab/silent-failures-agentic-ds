import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Prepare target and features
target = train_df['loss']
shift = 200.0
y_train = np.log(target + shift)

# Identify feature columns
drop_cols = ['id', 'loss']
features = [col for col in train_df.columns if col not in drop_cols]
cat_cols = [col for col in features if train_df[col].dtype == 'object' or str(train_df[col].dtype) == 'category']

X_train = train_df[features].copy()
X_test = test_df[features].copy()

# Convert categorical columns to category dtype
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 20,
    'n_estimators': 1500,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    pred_log = model.predict(X_test)
    test_preds += (np.exp(pred_log) - shift) / n_splits

# Ensure non-negative predictions
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'loss': test_preds
})
output_df.to_csv('predictions.csv', index=False)