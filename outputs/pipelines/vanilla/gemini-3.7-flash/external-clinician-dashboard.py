import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Score'
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Identify categorical features
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

# Prepare datasets for LightGBM and CatBoost
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test))
cat_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # LightGBM
    X_train_lgb, y_train = X_lgb.iloc[train_idx], y[train_idx]
    X_val_lgb, y_val = X_lgb.iloc[val_idx], y[val_idx]

    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train_lgb, y_train,
        eval_set=[(X_val_lgb, y_val)],
        eval_metric='rmse'
    )
    lgb_test_preds += lgb_model.predict(X_test_lgb) / n_splits

    # CatBoost
    X_train_cat = X.iloc[train_idx]
    X_val_cat = X.iloc[val_idx]

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    cat_model.fit(
        X_train_cat, y_train,
        eval_set=(X_val_cat, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_test_preds += cat_model.predict(X_test) / n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_test_preds + 0.5 * cat_test_preds

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Score': final_predictions
})
submission.to_csv('predictions.csv', index=False)