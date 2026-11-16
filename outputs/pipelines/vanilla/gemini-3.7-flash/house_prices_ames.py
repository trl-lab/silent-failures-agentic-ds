import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Sale_Price'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Identify categorical and numerical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype.name == 'category' or X_train[col].dtype == 'object']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

# Standardize categorical columns
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    categories = sorted(list(set(X_train[col].unique()).union(set(X_test[col].unique()))))
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    X_train[col] = X_train[col].astype(cat_type)
    X_test[col] = X_test[col].astype(cat_type)

for col in numerical_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Model training with 5-fold CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test))
cat_test_preds = np.zeros(len(X_test))

# Convert categoricals to integers for LightGBM
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in categorical_cols:
    X_train_lgb[col] = X_train_lgb[col].cat.codes
    X_test_lgb[col] = X_test_lgb[col].cat.codes

for train_idx, val_idx in kf.split(X_train, y_train):
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train_lgb.iloc[train_idx], y_train[train_idx],
        eval_set=[(X_train_lgb.iloc[val_idx], y_train[val_idx])],
        callbacks=[]
    )
    lgb_test_preds += lgb_model.predict(X_test_lgb) / n_splits

    # CatBoost
    cat_train_x = X_train.copy()
    cat_test_x = X_test.copy()
    for col in categorical_cols:
        cat_train_x[col] = cat_train_x[col].astype(str)
        cat_test_x[col] = cat_test_x[col].astype(str)

    cat_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        cat_features=categorical_cols,
        verbose=0,
        random_seed=42
    )
    cat_model.fit(
        cat_train_x.iloc[train_idx], y_train[train_idx],
        eval_set=(cat_train_x.iloc[val_idx], y_train[val_idx]),
        early_stopping_rounds=100,
        verbose=False
    )
    cat_test_preds += cat_model.predict(cat_test_x) / n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_test_preds + 0.5 * cat_test_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Sale_Price': final_predictions
})
output_df.to_csv('predictions.csv', index=False)