import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Size of Purchase (MW)'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Drop columns with urls or high unique free-form text if not useful
drop_cols = ['Ref. No.', 'Source 1', 'Source 2', 'Source 3', 'Source 4', 'Source 5', 'Notes']
drop_cols = [col for col in drop_cols if col in X_train.columns]
X_train.drop(columns=drop_cols, inplace=True)
X_test.drop(columns=drop_cols, inplace=True)

# Combine for consistent feature preprocessing
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Attempt numeric conversions for object columns that are mostly numeric
for col in df_all.columns:
    if df_all[col].dtype == 'object':
        numeric_series = pd.to_numeric(df_all[col], errors='coerce')
        if numeric_series.notna().sum() > 0.4 * len(df_all):
            df_all[col] = numeric_series

# Identify numeric and categorical columns
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df_all.columns if c not in numeric_cols]

for c in cat_cols:
    df_all[c] = df_all[c].astype(str).fillna('missing').astype('category')

X_train_processed = df_all.iloc[:len(train_df)].copy()
X_test_processed = df_all.iloc[len(train_df):].copy()

# Model training with K-Fold cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_processed))
cb_preds = np.zeros(len(X_test_processed))

# Prepare for CatBoost
cat_features_idx = [X_train_processed.columns.get_loc(c) for c in cat_cols]
X_train_cb = X_train_processed.copy()
X_test_cb = X_test_processed.copy()
for c in cat_cols:
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

for train_idx, val_idx in kf.split(X_train_processed):
    # LightGBM
    X_tr_lgb, y_tr = X_train_processed.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train_processed.iloc[val_idx], y_train[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test_processed) / n_splits

    # CatBoost
    X_tr_cb = X_train_cb.iloc[train_idx]
    X_va_cb = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=False
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Size of Purchase (MW)': final_predictions
})
output_df.to_csv('predictions.csv', index=False)