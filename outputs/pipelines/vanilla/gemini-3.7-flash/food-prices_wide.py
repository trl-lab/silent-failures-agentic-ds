import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'usdprice'
features = [col for col in train_df.columns if col != target_col]

X_train = train_df[features].copy()
y_train = train_df[target_col].copy()
X_test = test_df[features].copy()

# Identify categorical and numeric columns
numeric_cols = ['latitude', 'longitude']
categorical_cols = [c for c in features if c not in numeric_cols]

# Handle categorical variables consistently across train and test
for col in categorical_cols:
    combined_cats = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0).astype('category').cat.categories
    X_train[col] = pd.Categorical(X_train[col].astype(str), categories=combined_cats)
    X_test[col] = pd.Categorical(X_test[col].astype(str), categories=combined_cats)

# Convert numeric columns to float64
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# 5-fold cross-validation with LightGBM
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test), dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
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
    )

    test_preds += model.predict(X_test) / n_splits

# Prepare submission DataFrame
submission_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'usdprice': test_preds
})

# Save predictions
submission_df.to_csv('predictions.csv', index=False)