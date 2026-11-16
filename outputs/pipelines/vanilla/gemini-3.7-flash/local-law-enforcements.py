import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-law-enforcements/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-law-enforcements/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'number_of_officers'
y = train_df[target_col].values

n_train = len(train_df)
combined = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Datetime feature engineering
for date_col in ['SOURCEDATE', 'VAL_DATE']:
    if date_col in combined.columns:
        combined[date_col] = pd.to_datetime(combined[date_col])
        combined[f'{date_col}_year'] = combined[date_col].dt.year
        combined[f'{date_col}_month'] = combined[date_col].dt.month
        combined[f'{date_col}_day'] = combined[date_col].dt.day
        combined[f'{date_col}_dayofweek'] = combined[date_col].dt.dayofweek
        combined[f'{date_col}_ts'] = combined[date_col].astype('int64') // 10**9
        combined.drop(columns=[date_col], inplace=True)

# Text length & keyword features
if 'NAME' in combined.columns:
    combined['NAME_len'] = combined['NAME'].astype(str).str.len()
    combined['NAME_num_words'] = combined['NAME'].astype(str).str.split().apply(len)
    combined['IS_POLICE'] = combined['NAME'].astype(str).str.upper().str.contains('POLICE').astype(float)
    combined['IS_SHERIFF'] = combined['NAME'].astype(str).str.upper().str.contains('SHERIFF').astype(float)
    combined['IS_DEPT'] = combined['NAME'].astype(str).str.upper().str.contains('DEPARTMENT').astype(float)
    combined['IS_UNIV'] = combined['NAME'].astype(str).str.upper().str.contains('UNIVERSITY|COLLEGE').astype(float)

# Ratio / interaction features
if 'FTCIV' in combined.columns and 'PTCIV' in combined.columns:
    combined['CIV_TOTAL'] = combined['FTCIV'].fillna(0) + combined['PTCIV'].fillna(0)
    combined['FT_CIV_RATIO'] = (combined['FTCIV'].fillna(0) + 1) / (combined['CIV_TOTAL'] + 1)

# Categorical column handling
cat_cols = []
for col in combined.columns:
    if combined[col].dtype == 'object' or isinstance(combined[col].dtype, pd.CategoricalDtype):
        combined[col] = combined[col].astype('category')
        cat_cols.append(col)

# Separate back into train and test sets
X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 5-Fold Cross-Validation Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

cat_feature_names = cat_cols

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    X_tr_cat = X_tr.copy()
    X_va_cat = X_va.copy()
    X_test_cat = X_test.copy()
    for col in cat_cols:
        X_tr_cat[col] = X_tr_cat[col].astype(str).fillna('missing')
        X_va_cat[col] = X_va_cat[col].astype(str).fillna('missing')
        X_test_cat[col] = X_test_cat[col].astype(str).fillna('missing')

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(
        X_tr_cat, y_tr,
        eval_set=(X_va_cat, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test_cat) / kf.n_splits

# Ensemble average
final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'number_of_officers': final_predictions
})
output_df.to_csv('predictions.csv', index=False)