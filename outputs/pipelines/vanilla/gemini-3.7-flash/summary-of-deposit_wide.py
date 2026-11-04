import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'DEPSUM'
y_train = train_df[target_col].values.astype(np.float64)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature engineering: date handling
for df in [X_train, X_test]:
    if 'SIMS_ESTABLISHED_DATE' in df.columns:
        dt = pd.to_datetime(df['SIMS_ESTABLISHED_DATE'], errors='coerce')
        df['EST_YEAR'] = dt.dt.year
        df['EST_MONTH'] = dt.dt.month
        df['EST_DAY'] = dt.dt.day
        df['EST_AGE_DAYS'] = (pd.to_datetime('2024-01-01') - dt).dt.days

drop_cols = ['SIMS_ESTABLISHED_DATE']
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Encode categorical features
cat_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col]):
        cat_cols.append(col)
        X_train[col] = X_train[col].astype(str).astype('category')
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_train[col].cat.categories)

# 5-fold cross-validation with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test), dtype=np.float64)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'n_estimators': 2000,
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_f_tr, y_f_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_f_val, y_f_val = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_f_tr, y_f_tr,
        eval_set=[(X_f_val, y_f_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    test_preds += model.predict(X_test) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'DEPSUM': test_preds
})
output_df.to_csv('predictions.csv', index=False)