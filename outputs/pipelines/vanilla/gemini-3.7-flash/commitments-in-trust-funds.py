import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'New Commitments (US$)'
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature Engineering
all_data = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

categorical_cols = [
    'Trust Fund', 'Trust Fund name', 'Trust Fund status', 'Execution Type',
    'Trustee', 'Trustee Status', 'Trustee Name', 'Fund Classification', 'Program group'
]

for col in categorical_cols:
    all_data[col] = all_data[col].fillna('Missing').astype(str)
    # Frequency encoding
    freq = all_data[col].value_counts()
    all_data[f'{col}_freq'] = all_data[col].map(freq)

# Text length features
all_data['name_len'] = all_data['Trust Fund name'].str.len()
all_data['trustee_name_len'] = all_data['Trustee Name'].str.len()
all_data['name_word_count'] = all_data['Trust Fund name'].str.split().str.len()

# Convert categorical columns to category dtype for LightGBM
for col in categorical_cols:
    all_data[col] = all_data[col].astype('category')

X_train_processed = all_data.iloc[:len(train_df)].copy()
X_test_processed = all_data.iloc[len(train_df):].copy()

# Cross-validation and Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(test_df))
cat_preds = np.zeros(len(test_df))

# Prepare for CatBoost (convert category to str)
X_train_cat = X_train_processed.copy()
X_test_cat = X_test_processed.copy()
for col in categorical_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

for train_idx, val_idx in kf.split(X_train_processed):
    # LightGBM
    X_tr, y_tr = X_train_processed.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_processed.iloc[val_idx], y[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_processed) / kf.n_splits

    # CatBoost
    X_tr_c, y_tr_c = X_train_cat.iloc[train_idx], y[train_idx]
    X_va_c, y_va_c = X_train_cat.iloc[val_idx], y[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=categorical_cols,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(
        X_tr_c, y_tr_c,
        eval_set=(X_va_c, y_va_c),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cb_model.predict(X_test_cat) / kf.n_splits

# Blend predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    target_col: final_predictions
})
output_df.to_csv('predictions.csv', index=False)