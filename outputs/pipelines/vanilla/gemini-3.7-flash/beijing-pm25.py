import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop missing targets if any
train_df = train_df.dropna(subset=['pm2.5']).reset_index(drop=True)

target_col = 'pm2.5'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.copy()

# Feature Engineering
def engineer_features(df):
    df = df.copy()
    # Datetime features
    dt = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['dayofweek'] = dt.dt.dayofweek
    df['dayofyear'] = dt.dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical encodings
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Interaction features
    df['temp_dewp_diff'] = df['TEMP'] - df['DEWP']
    
    return df

X_train_fe = engineer_features(X_train)
X_test_fe = engineer_features(X_test)

# Categorical column handling
cat_cols = ['cbwd']
for col in cat_cols:
    X_train_fe[col] = X_train_fe[col].astype('category')
    X_test_fe[col] = X_test_fe[col].astype('category')

# K-Fold Cross-Validation & Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_fe))
cb_preds = np.zeros(len(X_test_fe))

# Prepare for CatBoost (convert category to string / handle NaNs)
X_train_cb = X_train_fe.copy()
X_test_cb = X_test_fe.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

for train_idx, val_idx in kf.split(X_train_fe, y_train):
    # LightGBM
    X_tr_lgb, y_tr = X_train_fe.iloc[train_idx], y_train.iloc[train_idx]
    X_va_lgb, y_va = X_train_fe.iloc[val_idx], y_train.iloc[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_fe) / kf.n_splits

    # CatBoost
    X_tr_cb = X_train_cb.iloc[train_idx]
    X_va_cb = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        cat_features=cat_cols,
        random_seed=42,
        verbose=False
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False)
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Blend predictions and ensure non-negative PM2.5
final_preds = np.clip(0.5 * lgb_preds + 0.5 * cb_preds, 0, None)

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'pm2.5': final_preds
})
submission.to_csv('predictions.csv', index=False)