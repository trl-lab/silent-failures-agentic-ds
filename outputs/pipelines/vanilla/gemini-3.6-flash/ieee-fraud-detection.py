import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = train['isFraud'].values
train = train.drop(columns=['isFraud'])

df_all = pd.concat([train, test], ignore_index=True)
del train, test
gc.collect()

if 'TransactionDT' in df_all.columns:
    df_all['dt_hour'] = (df_all['TransactionDT'] // 3600) % 24
    df_all['dt_day'] = (df_all['TransactionDT'] // (3600 * 24)) % 7

if 'TransactionID' in df_all.columns:
    df_all = df_all.drop(columns=['TransactionID'])

cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).astype('category')

num_cols = df_all.select_dtypes(include=['number']).columns.tolist()
for col in num_cols:
    if df_all[col].dtype == 'float64':
        df_all[col] = df_all[col].astype(np.float32)
    elif df_all[col].dtype == 'int64':
        df_all[col] = df_all[col].astype(np.int32)

n_train = len(y)
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

del df_all
gc.collect()

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    gc.collect()

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'isFraud': test_preds
})
sub.to_csv('predictions.csv', index=False)