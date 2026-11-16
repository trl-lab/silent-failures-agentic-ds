import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'credit_amount'
y = train_df[target_col].values.astype(float)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Date feature extraction
date_cols = ['createdon', 'enddate', 'modifiedon', 'startdate']
for col in date_cols:
    if col in X_train.columns:
        for df in [X_train, X_test]:
            dt = pd.to_datetime(df[col], errors='coerce', utc=True)
            df[f'{col}_year'] = dt.dt.year.fillna(-1).astype(float)
            df[f'{col}_month'] = dt.dt.month.fillna(-1).astype(float)
            df[f'{col}_day'] = dt.dt.day.fillna(-1).astype(float)
            df[f'{col}_isnull'] = df[col].isnull().astype(int)
            df.drop(columns=[col], inplace=True)

# Process remaining columns (numeric vs categorical)
for col in list(X_train.columns):
    if col.endswith('_year') or col.endswith('_month') or col.endswith('_day') or col.endswith('_isnull'):
        continue
    s_train = pd.to_numeric(X_train[col], errors='coerce')
    s_test = pd.to_numeric(X_test[col], errors='coerce')
    
    if s_train.notnull().mean() > 0.5:
        X_train[col] = s_train.fillna(-999.0).astype(float)
        X_test[col] = s_test.fillna(-999.0).astype(float)
    else:
        X_train[col] = X_train[col].fillna('__MISSING__').astype(str)
        X_test[col] = X_test[col].fillna('__MISSING__').astype(str)

cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Train CatBoost models with K-Fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=0
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'credit_amount': test_preds
})
output_df.to_csv('predictions.csv', index=False)