import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Supplier Contract Amount (USD)'
y_train = train_df[target_col].values.astype(np.float32)
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

n_train = len(X_train)
df = pd.concat([X_train, X_test], ignore_index=True)

# Date processing
for date_col in ['Contract Signing Date', 'As of Date']:
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors='coerce')
        df[f'{date_col}_year'] = dt.dt.year.fillna(-1).astype(np.int32)
        df[f'{date_col}_month'] = dt.dt.month.fillna(-1).astype(np.int32)
        df[f'{date_col}_day'] = dt.dt.day.fillna(-1).astype(np.int32)
        df[f'{date_col}_dayofweek'] = dt.dt.dayofweek.fillna(-1).astype(np.int32)
        df.drop(columns=[date_col], inplace=True)

# Text length features
text_cols = ['Contract Description', 'Project Name']
for col in text_cols:
    if col in df.columns:
        df[f'{col}_len'] = df[col].astype(str).str.len()
        df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()

# Categorical column handling
cat_cols = []
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        cat_cols.append(col)
        df[col] = df[col].astype('category')

X_train_proc = df.iloc[:n_train].copy()
X_test_proc = df.iloc[n_train:].copy()

# K-Fold LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_proc), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_proc)):
    X_tr, y_tr = X_train_proc.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    test_preds += model.predict(X_test_proc) / kf.n_splits

output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Supplier Contract Amount (USD)': test_preds
})
output_df.to_csv('predictions.csv', index=False)