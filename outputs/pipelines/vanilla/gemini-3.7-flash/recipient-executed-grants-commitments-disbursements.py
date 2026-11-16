import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Disbursements USD (Trust funds)'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

def preprocess_features(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    # Process dates
    for df in [df_train, df_test]:
        dt = pd.to_datetime(df['Grant Agreement Date'], errors='coerce')
        df['agreement_year'] = dt.dt.year
        df['agreement_month'] = dt.dt.month
        df['agreement_day'] = dt.dt.day
        df['agreement_dayofweek'] = dt.dt.dayofweek
        df['agreement_dayofyear'] = dt.dt.dayofyear
        df['fiscal_agreement_year_diff'] = df['Fiscal Year'] - df['agreement_year']
        
        # Numeric log transforms
        commitments = np.maximum(0, df['Grant Commitments USD'].fillna(0))
        df['log_commitments'] = np.log1p(commitments)
        df.drop(columns=['Grant Agreement Date'], inplace=True)
    
    cat_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Frequency encoding and categorical casting
    for col in cat_cols:
        freq = df_train[col].value_counts()
        df_train[f'{col}_freq'] = df_train[col].map(freq).fillna(0)
        df_test[f'{col}_freq'] = df_test[col].map(freq).fillna(0)
        
        df_train[col] = df_train[col].astype(str).astype('category')
        df_test[col] = df_test[col].astype(str).astype('category')
        
    return df_train, df_test

X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_proc))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_preds += model.predict(X_test_proc) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Disbursements USD (Trust funds)': test_preds
})
output_df.to_csv('predictions.csv', index=False)