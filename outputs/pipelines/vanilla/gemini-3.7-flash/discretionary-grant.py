import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'AwardTotal'
y_train = train_df[target_col].values

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature Engineering
def preprocess_features(df):
    df = df.copy()
    
    # Process AwardDate
    if 'AwardDate' in df.columns:
        dates = pd.to_datetime(df['AwardDate'], errors='coerce')
        df['AwardDate_Year'] = dates.dt.year
        df['AwardDate_Month'] = dates.dt.month
        df['AwardDate_Day'] = dates.dt.day
        df['AwardDate_DayOfWeek'] = dates.dt.dayofweek
        df = df.drop(columns=['AwardDate'])
        
    # Text length features
    text_cols = ['Institution', 'ProjectTitle', 'Description', 'OrgType', 'Program', 'ProjectType']
    for col in text_cols:
        if col in df.columns:
            df[f'{col}_len'] = df[col].astype(str).str.len()
            df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
            
    return df

X_train = preprocess_features(X_train)
X_test = preprocess_features(X_test)

# Identify column types
cat_cols = []
num_cols = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col]):
        cat_cols.append(col)
        X_train[col] = X_train[col].astype(str).fillna('missing')
        X_test[col] = X_test[col].astype(str).fillna('missing')
    else:
        num_cols.append(col)
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Convert categoricals to category dtype for LightGBM
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1500
}

# 1. Train LightGBM
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_lgb, y_train)):
    X_tr, y_tr = X_train_lgb.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_lgb.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    lgb_preds += model_lgb.predict(X_test_lgb) / kf.n_splits

# 2. Train CatBoost
cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0,
        cat_features=cat_indices
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    
    cb_preds += cb_model.predict(X_test) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions to CSV
submission = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'AwardTotal': final_predictions
})
submission.to_csv('predictions.csv', index=False)