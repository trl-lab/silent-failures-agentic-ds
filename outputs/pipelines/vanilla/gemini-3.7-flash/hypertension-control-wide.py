import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Estimated Percentage of Patients with Controlled Blood Pressure'

y_train = train_df[target_col].astype(int)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Feature Engineering & Date Processing
def preprocess_features(df):
    df = df.copy()
    
    # Process date columns
    date_cols = ['Project Period Start Date', 'Grant Project Period End Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    
    if 'Project Period Start Date' in df.columns and 'Grant Project Period End Date' in df.columns:
        df['Project_Duration_Days'] = (df['Grant Project Period End Date'] - df['Project Period Start Date']).dt.days
    
    # Drop original datetime objects and redundant raw date string columns
    drop_cols = [
        'Project Period Start Date', 
        'Grant Project Period End Date',
        'Project Period Start Date Text String',
        'Grant Project Period End Date Text'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    # Convert object columns to category
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).fillna('missing').astype('category')
            
    return df

X_train_proc = preprocess_features(X_train)
X_test_proc = preprocess_features(X_test)

# Align categories between train and test
for col in X_train_proc.select_dtypes(include=['category']).columns:
    categories = sorted(list(set(X_train_proc[col].astype(str)).union(set(X_test_proc[col].astype(str)))))
    X_train_proc[col] = pd.Categorical(X_train_proc[col].astype(str), categories=categories)
    X_test_proc[col] = pd.Categorical(X_test_proc[col].astype(str), categories=categories)

# 3. Model Training with K-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test_proc))
cat_test_preds = np.zeros(len(X_test_proc))

cat_cols = list(X_train_proc.select_dtypes(include=['category']).columns)

# Prepare string representation for CatBoost
X_train_cb = X_train_proc.copy()
X_test_cb = X_test_proc.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

for train_idx, val_idx in skf.split(X_train_proc, y_train):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]
    
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_test_preds += lgb_model.predict_proba(X_test_proc)[:, 1] / n_splits
    
    # CatBoost
    X_tr_cb, X_va_cb = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False)
    cat_test_preds += cb_model.predict_proba(X_test_cb)[:, 1] / n_splits

# 4. Ensemble Predictions
final_preds = 0.5 * lgb_test_preds + 0.5 * cat_test_preds

# 5. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})

submission_df.to_csv('predictions.csv', index=False)