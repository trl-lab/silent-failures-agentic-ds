import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Total Contribution (USD)'
features = [c for c in train_df.columns if c != target_col]

X = train_df[features].copy()
y = train_df[target_col].values
X_test = test_df[features].copy()

# Identify categorical and numerical columns
cat_cols = ['Fund', 'Trustee Fund Number', 'Trustee Fund Name', 'Donor Code', 'Donor Name', 'Donor Agency', 'Status']
cat_cols = [c for c in cat_cols if c in features]
num_cols = [c for c in features if c not in cat_cols]

# Preprocess categorical features
for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

# LightGBM requires 'category' dtype
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross-Validation Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds_cb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # CatBoost
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        verbose=0,
        random_seed=42 + fold
    )
    cb_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, verbose=False)
    test_preds_cb += cb_model.predict(X_test) / kf.n_splits
    
    # LightGBM
    X_train_lgb, y_train_lgb = X_lgb.iloc[train_idx], y[train_idx]
    X_val_lgb, y_val_lgb = X_lgb.iloc[val_idx], y[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train_lgb)
    test_preds_lgb += lgb_model.predict(X_test_lgb) / kf.n_splits

final_predictions = 0.5 * test_preds_cb + 0.5 * test_preds_lgb

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    target_col: final_predictions
})
output_df.to_csv('predictions.csv', index=False)