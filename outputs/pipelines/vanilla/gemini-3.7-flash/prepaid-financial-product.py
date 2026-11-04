import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'current_status'
target_le = LabelEncoder()
y_train = target_le.fit_transform(train_df[target_col].astype(str))

# Feature engineering
def preprocess(df):
    data = df.copy()
    if target_col in data.columns:
        data = data.drop(columns=[target_col])
        
    # Date parsing
    eff_date = pd.to_datetime(data['agreement_effective_date'], errors='coerce')
    created_date = pd.to_datetime(data['created_date'], errors='coerce')
    
    data['eff_year'] = eff_date.dt.year
    data['eff_month'] = eff_date.dt.month
    data['eff_day'] = eff_date.dt.day
    data['eff_dayofweek'] = eff_date.dt.dayofweek
    
    data['created_year'] = created_date.dt.year
    data['created_month'] = created_date.dt.month
    data['created_day'] = created_date.dt.day
    data['created_hour'] = created_date.dt.hour
    data['created_dayofweek'] = created_date.dt.dayofweek
    
    data['date_diff_days'] = (created_date - eff_date).dt.total_seconds() / 86400.0
    
    # Drop raw date and complex text path columns
    cols_to_drop = ['agreement_effective_date', 'created_date', 'path', 'direct_download']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # Convert categorical/object columns to category dtype
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('Missing').astype(str).astype('category')
        elif str(data[col].dtype) == 'category':
            data[col] = data[col].astype(str).astype('category')
            
    return data

X_train = preprocess(train_df)
X_test = preprocess(test_df)

# Align categorical columns
for col in X_train.select_dtypes(include=['category']).columns:
    all_cats = pd.concat([X_train[col].astype(str), X_test[col].astype(str)]).unique()
    X_train[col] = pd.Categorical(X_train[col].astype(str), categories=all_cats)
    X_test[col] = pd.Categorical(X_test[col].astype(str), categories=all_cats)

# K-Fold Cross Validation & Prediction
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

# Save predictions
predictions = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'current_status': test_preds
})

predictions.to_csv('predictions.csv', index=False)