import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Extract target and separate features
y_train = train_df['isFraud'].values
drop_cols = ['TransactionID', 'isFraud']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols].copy()
X_test = test_df[[c for c in feature_cols if c in test_df.columns]].copy()

# Feature engineering on TransactionDT
for df in [X_train, X_test]:
    if 'TransactionDT' in df.columns:
        df['Transaction_hour'] = (df['TransactionDT'] / 3600) % 24
        df['Transaction_day'] = (df['TransactionDT'] / (3600 * 24)) % 7

# TransactionAmt decimal part
for df in [X_train, X_test]:
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - np.floor(df['TransactionAmt'])) * 1000).round()

# Handle categorical columns
cat_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category':
        cat_cols.append(col)
        
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    le = LabelEncoder()
    le.fit(list(X_train[col].unique()) + list(X_test[col].unique()))
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Model training with Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'n_estimators': 600,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'isFraud': test_preds
})
output_df.to_csv('predictions.csv', index=False)