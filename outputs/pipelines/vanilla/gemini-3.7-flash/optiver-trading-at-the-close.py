import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/optiver-trading-at-the-close/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/optiver-trading-at-the-close/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df = train_df.dropna(subset=['target']).reset_index(drop=True)

def feature_engineering(df):
    df = df.copy()
    
    df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    df['price_spread'] = df['ask_price'] - df['bid_price']
    df['spread_wap_ratio'] = df['price_spread'] / (df['wap'] + 1e-8)
    df['reference_wap_diff'] = df['reference_price'] - df['wap']
    df['imbalance_matched_ratio'] = df['imbalance_size'] / (df['matched_size'] + 1e-8)
    df['total_volume'] = df['bid_size'] + df['ask_size']
    df['bid_ask_size_ratio'] = df['bid_size'] / (df['ask_size'] + 1e-8)
    df['far_near_diff'] = df['far_price'] - df['near_price']
    
    return df

train_features = feature_engineering(train_df)
test_features = feature_engineering(test_df)

drop_cols = ['target', 'row_id', 'date_id', 'time_id']
feature_cols = [c for c in train_features.columns if c not in drop_cols]

X = train_features[feature_cols]
y = train_features['target']
X_test = test_features[feature_cols]

for c in feature_cols:
    if X[c].dtype == 'object':
        X[c] = X[c].astype('category')
        X_test[c] = X_test[c].astype('category')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(test_features))

lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 600,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_predictions += model.predict(X_test) / kf.n_splits

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'target': test_predictions
})

sub.to_csv('predictions.csv', index=False)