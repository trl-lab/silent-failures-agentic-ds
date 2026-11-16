import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load dataset
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Res_capacityM3'
y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature preprocessing
n_train = len(X)
combined = pd.concat([X, X_test], ignore_index=True)

# Attempt to convert object columns to numeric if applicable
for col in combined.columns:
    if combined[col].dtype == 'object':
        converted = pd.to_numeric(combined[col], errors='coerce')
        if converted.notna().mean() > 0.5:
            combined[col] = converted

# Identify numeric and categorical columns
cat_cols = []
num_cols = []
for col in combined.columns:
    if combined[col].dtype == 'object' or pd.api.types.is_categorical_dtype(combined[col]):
        cat_cols.append(col)
        combined[col] = combined[col].astype(str).fillna('missing')
    else:
        num_cols.append(col)
        combined[col] = combined[col].astype(np.float32)

# Separate train and test
X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# LightGBM categorical handling
for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Prepare for CatBoost with string categoricals
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_cb.iloc[train_idx], y_train_fold,
        eval_set=(X_cb.iloc[val_idx], y_val_fold),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Res_capacityM3': final_preds
})
submission.to_csv('predictions.csv', index=False)