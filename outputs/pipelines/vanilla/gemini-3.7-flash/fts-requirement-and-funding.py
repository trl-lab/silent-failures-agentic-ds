import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/fts-requirement-and-funding/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/fts-requirement-and-funding/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. Preprocess & Feature Engineering
def preprocess_features(df):
    df = df.copy()
    
    # Process dates
    start_dt = pd.to_datetime(df['startDate'], errors='coerce')
    end_dt = pd.to_datetime(df['endDate'], errors='coerce')
    
    df['start_year'] = start_dt.dt.year
    df['start_month'] = start_dt.dt.month
    df['start_day'] = start_dt.dt.day
    df['end_year'] = end_dt.dt.year
    df['end_month'] = end_dt.dt.month
    df['end_day'] = end_dt.dt.day
    df['duration_days'] = (end_dt - start_dt).dt.days
    
    # Log transform of requirements
    df['log_requirements'] = np.log1p(np.maximum(df['requirements'].fillna(0), 0))
    
    # Drop raw date columns
    df = df.drop(columns=['startDate', 'endDate'], errors='ignore')
    
    # Convert object columns to category
    cat_cols = ['countryCode', 'name', 'code', 'cluster']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing').astype('category')
            
    return df

target_col = 'funding'
y_train = train_df[target_col].values

X_train = preprocess_features(train_df.drop(columns=[target_col]))
X_test = preprocess_features(test_df)

# Align categories
cat_cols = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
for col in cat_cols:
    train_cats = set(X_train[col].cat.categories)
    test_cats = set(X_test[col].cat.categories)
    all_cats = list(train_cats.union(test_cats))
    X_train[col] = pd.Categorical(X_train[col], categories=all_cats)
    X_test[col] = pd.Categorical(X_test[col], categories=all_cats)

# 3. Model Training with Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_cols]

# LightGBM
for train_idx, val_idx in kf.split(X_train, y_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_tr, y_tr)
    lgb_preds += model.predict(X_test) / kf.n_splits

# CatBoost
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for train_idx, val_idx in kf.split(X_train_cb, y_train):
    X_tr, y_tr = X_train_cb.iloc[train_idx], y_train[train_idx]
    
    cb_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_indices,
        verbose=0,
        random_state=42
    )
    cb_model.fit(X_tr, y_tr)
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Blend predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# 4. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'funding': final_predictions
})
submission_df.to_csv('predictions.csv', index=False)