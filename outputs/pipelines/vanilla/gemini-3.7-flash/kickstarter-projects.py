import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/kickstarter-projects/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/kickstarter-projects/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'status'
y_raw = train_df[target_col]

# Identify target labels and encode (successful=1, failed=0 or alphabetical)
unique_targets = sorted(y_raw.dropna().unique())
target_mapping = {val: idx for idx, val in enumerate(unique_targets)}
if 'successful' in target_mapping and 'failed' in target_mapping:
    target_mapping = {'failed': 0, 'successful': 1}

y = y_raw.map(target_mapping).values

def preprocess_features(df):
    data = df.copy()
    
    # Parse dates
    for col in ['launched_at', 'deadline']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            prefix = col.split('_')[0]
            data[f'{prefix}_year'] = data[col].dt.year
            data[f'{prefix}_month'] = data[col].dt.month
            data[f'{prefix}_day'] = data[col].dt.day
            data[f'{prefix}_dayofweek'] = data[col].dt.dayofweek
            data[f'{prefix}_hour'] = data[col].dt.hour
            data.drop(columns=[col], inplace=True)
            
    # Numerical transformations
    if 'goal_usd' in data.columns:
        data['log_goal_usd'] = np.log1p(np.maximum(0, data['goal_usd'].fillna(0)))
        
    # Drop identifier and target if present
    drop_cols = ['id', 'status', 'name']
    for c in drop_cols:
        if c in data.columns:
            data.drop(columns=[c], inplace=True)
            
    # Convert object columns to category for LightGBM
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        data[col] = data[col].astype('category')
        
    return data

X_train = preprocess_features(train_df)
X_test = preprocess_features(test_df)

# Align categories
for col in X_train.select_dtypes(include=['category']).columns:
    train_cats = X_train[col].cat.categories
    test_cats = X_test[col].cat.categories
    all_cats = list(set(train_cats).union(set(test_cats)))
    X_train[col] = pd.Categorical(X_train[col], categories=all_cats)
    X_test[col] = pd.Categorical(X_test[col], categories=all_cats)

# K-Fold Training and Inference
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(test_df))
oof_preds = np.zeros(len(train_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    
    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'status': test_preds
})

sub.to_csv('predictions.csv', index=False)