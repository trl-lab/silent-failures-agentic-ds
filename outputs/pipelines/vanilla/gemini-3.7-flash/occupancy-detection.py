import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

def preprocess(df):
    data = df.copy()
    
    # Process datetime
    if 'date' in data.columns:
        dt = pd.to_datetime(data['date'], errors='coerce')
        data['hour'] = dt.dt.hour
        data['minute'] = dt.dt.minute
        data['dayofweek'] = dt.dt.dayofweek
        data['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        data['time_of_day'] = dt.dt.hour + dt.dt.minute / 60.0
        data.drop(columns=['date'], inplace=True)
    
    # Convert numerical columns
    numeric_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Feature engineering
    if 'Light' in data.columns:
        data['is_light_on'] = (data['Light'] > 0).astype(int)
    
    return data

# Prepare feature sets
X_train = preprocess(train_df.drop(columns=['Occupancy']))
y_train = train_df['Occupancy'].astype(int)
X_test = preprocess(test_df)

# Impute missing values with median
for col in X_train.columns:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# 5-fold Stratified Cross-Validation with Ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # LightGBM
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=42 + fold, verbose=-1)
    lgb.fit(X_tr, y_tr)
    
    # CatBoost
    cat = CatBoostClassifier(iterations=300, learning_rate=0.03, verbose=0, random_seed=42 + fold)
    cat.fit(X_tr, y_tr)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, random_state=42 + fold, eval_metric='logloss')
    xgb.fit(X_tr, y_tr)
    
    fold_preds = (
        lgb.predict_proba(X_test)[:, 1] * 0.4 +
        cat.predict_proba(X_test)[:, 1] * 0.3 +
        xgb.predict_proba(X_test)[:, 1] * 0.3
    )
    
    test_preds += fold_preds / skf.n_splits

# Save predictions
output = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Occupancy': test_preds
})
output.to_csv('predictions.csv', index=False)