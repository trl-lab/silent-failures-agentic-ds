import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Handle target missing values if any (-200 is missing value in UCI Air Quality)
target_col = 'CO(GT)'
train_df = train_df[train_df[target_col] >= 0].reset_index(drop=True)

def preprocess_features(df):
    df_feat = df.copy()
    
    # Parse Date
    if 'Date' in df_feat.columns:
        date_series = pd.to_datetime(df_feat['Date'], errors='coerce')
        df_feat['Month'] = date_series.dt.month
        df_feat['Day'] = date_series.dt.day
        df_feat['DayOfWeek'] = date_series.dt.dayofweek
        df_feat['Year'] = date_series.dt.year
        df_feat.drop(columns=['Date'], inplace=True)
        
    # Parse Time
    if 'Time' in df_feat.columns:
        time_series = pd.to_datetime(df_feat['Time'], format='%H:%M:%S', errors='coerce')
        df_feat['Hour'] = time_series.dt.hour
        df_feat.drop(columns=['Time'], inplace=True)
        
    # Replace -200 (UCI missing value representation) with NaN
    for col in df_feat.columns:
        if col != target_col:
            df_feat[col] = df_feat[col].replace(-200, np.nan)
            df_feat[col] = df_feat[col].replace(-200.0, np.nan)
            
    return df_feat

train_processed = preprocess_features(train_df)
test_processed = preprocess_features(test_df)

X_train = train_processed.drop(columns=[target_col])
y_train = train_processed[target_col]
X_test = test_processed.copy()

# Ensure all columns are numeric
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Model training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train, y_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(n_estimators=500, learning_rate=0.03, random_state=42, verbose=-1)
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost
    cat_model = CatBoostRegressor(iterations=600, learning_rate=0.04, random_seed=42, verbose=0)
    cat_model.fit(X_tr, y_tr)
    cat_preds += cat_model.predict(X_test) / kf.n_splits
    
    # XGBoost
    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.03, random_state=42, max_depth=6)
    xgb_model.fit(X_tr, y_tr)
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

# Ensemble predictions
final_predictions = (lgb_preds * 0.4 + cat_preds * 0.35 + xgb_preds * 0.25)
final_predictions = np.clip(final_predictions, 0, None)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'CO(GT)': final_predictions
})
output_df.to_csv('predictions.csv', index=False)