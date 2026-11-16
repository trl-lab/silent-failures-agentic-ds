import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/global-power-plant/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/global-power-plant/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'capacity_mw'
y_train = train_df[target_col].values

# Drop target and identifier/url columns that don't generalize
drop_cols = [target_col, 'gppd_idnr', 'name', 'url', 'wepp_id']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Combine for uniform processing
combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0).reset_index(drop=True)

# Feature engineering
combined['has_other_fuel1'] = combined['other_fuel1'].notna().astype(int)
combined['has_other_fuel2'] = combined['other_fuel2'].notna().astype(int)
combined['has_other_fuel3'] = combined['other_fuel3'].notna().astype(int)
combined['num_fuels'] = 1 + combined['has_other_fuel1'] + combined['has_other_fuel2'] + combined['has_other_fuel3']

# Age features
current_year = 2023.0
combined['plant_age'] = current_year - combined['commissioning_year']
combined['years_since_capacity_data'] = current_year - combined['year_of_capacity_data']

# Identify categorical and numeric columns
cat_cols = ['country', 'country_long', 'primary_fuel', 'other_fuel1', 'other_fuel2', 
            'other_fuel3', 'owner', 'source', 'geolocation_source', 'generation_data_source']

for col in cat_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna('Missing').astype(str)
        # Frequency encoding
        freq = combined[col].value_counts(normalize=True)
        combined[f'{col}_freq'] = combined[col].map(freq).astype(float)
        combined[col] = combined[col].astype('category')

num_cols = [c for c in combined.columns if c not in cat_cols]
for col in num_cols:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')

n_train = len(train_df)
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# Setup K-Fold cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(test_df))
cat_test_preds = np.zeros(len(test_df))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 20,
    'random_state': 42,
    'verbosity': -1,
    'n_estimators': 1500
}

# CatBoost processing: convert category types to string
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    if col in X_train_cb.columns:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

for train_idx, val_idx in kf.split(X_train):
    # LightGBM
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_test_preds += model_lgb.predict(X_test) / n_splits
    
    # CatBoost
    X_tr_cb, y_tr_cb = X_train_cb.iloc[train_idx], y_train[train_idx]
    X_va_cb, y_va_cb = X_train_cb.iloc[val_idx], y_train[val_idx]
    
    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=False,
        cat_features=cat_cols
    )
    model_cb.fit(
        X_tr_cb, y_tr_cb,
        eval_set=(X_va_cb, y_va_cb),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_test_preds += model_cb.predict(X_test_cb) / n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_test_preds + 0.5 * cat_test_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'capacity_mw': final_predictions
})
output_df.to_csv('predictions.csv', index=False)