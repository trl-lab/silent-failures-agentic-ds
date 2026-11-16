import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

# File paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target variable
y_train = train_df['deal_probability'].values

# Feature engineering function
def extract_features(df):
    feats = pd.DataFrame(index=df.index)
    
    # Categorical columns
    cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 
                'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
    for col in cat_cols:
        feats[col] = df[col].astype('category')
    
    # Numeric features
    feats['price'] = np.log1p(df['price'].fillna(-1).clip(lower=-1))
    feats['item_seq_number'] = np.log1p(df['item_seq_number'].fillna(0))
    feats['has_image'] = df['image'].notnull().astype(int)
    
    # Text length features
    feats['title_len'] = df['title'].fillna('').astype(str).str.len()
    feats['title_words'] = df['title'].fillna('').astype(str).str.split().str.len()
    feats['desc_len'] = df['description'].fillna('').astype(str).str.len()
    feats['desc_words'] = df['description'].fillna('').astype(str).str.split().str.len()
    
    # Date features
    dates = pd.to_datetime(df['activation_date'], errors='coerce')
    feats['dayofweek'] = dates.dt.dayofweek.fillna(-1).astype(int)
    feats['day'] = dates.dt.day.fillna(-1).astype(int)
    
    return feats

X_train = extract_features(train_df)
X_test = extract_features(test_df)

# Align categories between train and test
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
for col in cat_cols:
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype('category')
    X_train[col] = pd.Categorical(X_train[col], categories=combined.cat.categories)
    X_test[col] = pd.Categorical(X_test[col], categories=combined.cat.categories)

# LightGBM Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 127,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_data_in_leaf': 50,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

# Post-processing: ensure predictions are within [0, 1]
test_preds = np.clip(test_preds, 0.0, 1.0)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'deal_probability': test_preds
})
output_df.to_csv('predictions.csv', index=False)