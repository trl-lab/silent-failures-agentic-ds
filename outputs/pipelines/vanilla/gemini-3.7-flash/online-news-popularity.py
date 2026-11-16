import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Strip column names of any leading/trailing whitespace
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

target_col = 'shares'
y_train = train_df[target_col].values

# Drop target and non-predictive columns like 'url' if present
drop_cols = [c for c in ['url', target_col] if c in train_df.columns]
X_train = train_df.drop(columns=drop_cols)

test_drop_cols = [c for c in ['url', target_col] if c in test_df.columns]
X_test = test_df.drop(columns=test_drop_cols)

# Align columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Handle categorical or object columns if any
for col in common_cols:
    if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col]):
        X_train[col] = pd.factorize(X_train[col])[0]
        X_test[col] = pd.factorize(X_test[col])[0]

# Fill missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# K-Fold ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))
cat_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    lgb_test_preds += lgb.predict(X_test) / n_splits

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)
    xgb_test_preds += xgb.predict(X_test) / n_splits

    # CatBoost
    cat = CatBoostRegressor(
        iterations=500,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    cat_test_preds += cat.predict(X_test) / n_splits

# Ensemble average
final_predictions = 0.4 * lgb_test_preds + 0.3 * xgb_test_preds + 0.3 * cat_test_preds

# Clip negative values if any since shares >= 0
final_predictions = np.clip(final_predictions, 0, None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'shares': final_predictions
})
output_df.to_csv('predictions.csv', index=False)