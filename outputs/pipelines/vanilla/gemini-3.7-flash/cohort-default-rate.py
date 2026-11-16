import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Average or Greater than 30'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test = test_df.copy()

# 2. Preprocessing
cat_cols = []
num_cols = []

for col in X.columns:
    if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Fill missing values
for col in cat_cols:
    X[col] = X[col].fillna('__MISSING__').astype(str)
    X_test[col] = X_test[col].fillna('__MISSING__').astype(str)

for col in num_cols:
    fill_val = X[col].median() if not np.isnan(X[col].median()) else 0
    X[col] = X[col].fillna(fill_val)
    X_test[col] = X_test[col].fillna(fill_val)

# Ordinal encode for LightGBM
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = X.copy()
X_test_encoded = X_test.copy()
if len(cat_cols) > 0:
    X_encoded[cat_cols] = encoder.fit_transform(X[cat_cols])
    X_test_encoded[cat_cols] = encoder.transform(X_test[cat_cols])

# 3. Model training with Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_test_preds = np.zeros(len(X_test))
lgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # CatBoost
    X_tr_cb, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va_cb, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    cb_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False)
    cb_test_preds += cb_model.predict_proba(X_test)[:, 1] / n_splits
    
    # LightGBM
    X_tr_lgb = X_encoded.iloc[train_idx]
    X_va_lgb = X_encoded.iloc[val_idx]
    
    lgb_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        eval_metric='auc',
        callbacks=[]
    )
    lgb_test_preds += lgb_model.predict_proba(X_test_encoded)[:, 1] / n_splits

# Ensemble predictions
final_preds = 0.6 * cb_test_preds + 0.4 * lgb_test_preds

# 4. Save predictions
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})
submission.to_csv('predictions.csv', index=False)