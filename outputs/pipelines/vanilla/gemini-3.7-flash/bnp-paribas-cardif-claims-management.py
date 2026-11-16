import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y_train = train_df['target'].values
drop_cols = ['ID', 'target'] if 'target' in train_df.columns else ['ID']
X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test = test_df.drop(columns=['ID'], errors='ignore')

# 2. Feature Engineering & Preprocessing
# Add count of missing values per row (known strong signal for this dataset)
X_train['nan_count'] = X_train.isnull().sum(axis=1)
X_test['nan_count'] = X_test.isnull().sum(axis=1)

# Handle categorical features
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    
    # Factorize categorical variables consistently
    all_cats = pd.concat([X_train[col], X_test[col]], axis=0).unique()
    cat_mapping = {val: idx for idx, val in enumerate(all_cats)}
    
    X_train[col] = X_train[col].map(cat_mapping).astype('category')
    X_test[col] = X_test[col].map(cat_mapping).astype('category')

# 3. Model Training with K-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    model = LGBMClassifier(**lgb_params)
    
    # Fit with early stopping
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[]
    )
    
    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred
    
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

# 4. Save Predictions
test_preds_clipped = np.clip(test_preds, 1e-15, 1 - 1e-15)

submission = pd.DataFrame({
    'id': np.arange(len(test_preds_clipped)),
    'target': test_preds_clipped
})

submission.to_csv('predictions.csv', index=False)