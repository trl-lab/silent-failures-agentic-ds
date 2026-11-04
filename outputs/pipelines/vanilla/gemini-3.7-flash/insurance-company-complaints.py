import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Recovery'
y = train_df[target_col].values

# 2. Feature Engineering
def preprocess_features(df):
    df_feat = df.copy()
    
    # Date processing
    for col in ['Opened', 'Closed']:
        if col in df_feat.columns:
            df_feat[col + '_dt'] = pd.to_datetime(df_feat[col], errors='coerce')
            df_feat[col + '_year'] = df_feat[col + '_dt'].dt.year
            df_feat[col + '_month'] = df_feat[col + '_dt'].dt.month
            df_feat[col + '_day'] = df_feat[col + '_dt'].dt.day
            df_feat[col + '_dayofweek'] = df_feat[col + '_dt'].dt.dayofweek
    
    if 'Opened_dt' in df_feat.columns and 'Closed_dt' in df_feat.columns:
        df_feat['duration_days'] = (df_feat['Closed_dt'] - df_feat['Opened_dt']).dt.total_seconds() / 86400.0
        df_feat.drop(columns=['Opened_dt', 'Closed_dt'], inplace=True)
    
    drop_cols = ['Opened', 'Closed']
    if target_col in df_feat.columns:
        drop_cols.append(target_col)
    df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], inplace=True)
    
    return df_feat

X = preprocess_features(train_df)
X_test = preprocess_features(test_df)

categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']

# Frequency encoding & Ordinal encoding for categoricals
for col in categorical_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)
    
    # Frequency encoding
    freq = X[col].value_counts()
    X[col + '_freq'] = X[col].map(freq).fillna(0)
    X_test[col + '_freq'] = X_test[col].map(freq).fillna(0)

# Ordinal encode categoricals
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

# 3. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]
    
    # LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost Regressor
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test) / kf.n_splits

# 4. Ensemble and Post-process
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds
final_predictions = np.clip(final_predictions, 0, None)

# 5. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Recovery': final_predictions
})
submission_df.to_csv('predictions.csv', index=False)