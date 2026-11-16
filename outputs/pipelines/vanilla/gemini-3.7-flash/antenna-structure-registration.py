import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Strucht'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(X_train)
combined_df = pd.concat([X_train, X_test], ignore_index=True)

# Drop high cardinality unique identifiers or URL if present, but extract features if useful
ignore_cols = ['url']
for col in ignore_cols:
    if col in combined_df.columns:
        combined_df.drop(columns=[col], inplace=True)

# Process categorical and numeric columns
categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()

for col in categorical_cols:
    combined_df[col] = combined_df[col].astype(str).fillna('missing')
    freq = combined_df[col].value_counts()
    combined_df[col + '_freq'] = combined_df[col].map(freq)
    combined_df[col] = combined_df[col].astype('category')

for col in numeric_cols:
    combined_df[col] = combined_df[col].astype(float)

X_train_clean = combined_df.iloc[:n_train].copy()
X_test_clean = combined_df.iloc[n_train:].copy()

# 2. Model Training with K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_clean))
cb_preds = np.zeros(len(X_test_clean))

cat_feature_indices = [X_train_clean.columns.get_loc(c) for c in categorical_cols if c in X_train_clean.columns]

# CatBoost data copy (categorical features as strings)
X_train_cb = X_train_clean.copy()
X_test_cb = X_test_clean.copy()
for col in categorical_cols:
    if col in X_train_cb.columns:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_clean, y_train)):
    # LightGBM
    X_tr, y_tr = X_train_clean.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_clean.iloc[val_idx], y_train[val_idx]

    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_clean) / n_splits

    # CatBoost
    X_tr_cb, y_tr_cb = X_train_cb.iloc[train_idx], y_train[train_idx]
    X_va_cb, y_va_cb = X_train_cb.iloc[val_idx], y_train[val_idx]

    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_feature_indices,
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(
        X_tr_cb, y_tr_cb,
        eval_set=(X_va_cb, y_va_cb),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / n_splits

# Blend predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# 3. Save predictions
submission = pd.DataFrame({
    'id': np.arange(len(final_predictions), dtype=int),
    'Strucht': final_predictions
})
submission.to_csv('predictions.csv', index=False)