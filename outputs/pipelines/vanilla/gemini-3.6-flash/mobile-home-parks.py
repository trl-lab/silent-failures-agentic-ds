import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/mobile-home-parks/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/mobile-home-parks/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'SIZE'

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col])
num_classes = len(le.classes_)

X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

drop_cols = []
for c in df_all.columns:
    if df_all[c].nunique(dropna=False) <= 1:
        drop_cols.append(c)

df_all = df_all.drop(columns=drop_cols)

for col in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        df_all[f'{col}_year'] = df_all[col].dt.year
        df_all[f'{col}_month'] = df_all[col].dt.month
        df_all[f'{col}_day'] = df_all[col].dt.day
        df_all[f'{col}_dayofweek'] = df_all[col].dt.dayofweek
        df_all = df_all.drop(columns=[col])
    elif df_all[col].dtype == 'object' or isinstance(df_all[col].dtype, pd.CategoricalDtype):
        freq = df_all[col].value_counts(dropna=False)
        df_all[f'{col}_freq'] = df_all[col].map(freq)
        df_all[col] = LabelEncoder().fit_transform(df_all[col].astype(str))

df_all = df_all.fillna(-999)

X_train_proc = df_all.iloc[:len(train_df)].copy()
X_test_proc = df_all.iloc[len(train_df):].copy()

min_class_cnt = pd.Series(y_train).value_counts().min()
n_splits = min(5, max(2, min_class_cnt))

if min_class_cnt < n_splits:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
else:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((len(X_test_proc), num_classes))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    cb_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric='mlogloss' if num_classes > 2 else 'logloss',
        early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    p_cb = cb_model.predict_proba(X_test_proc)
    p_lgb = lgb_model.predict_proba(X_test_proc)
    p_xgb = xgb_model.predict_proba(X_test_proc)
    
    fold_probs = (p_cb + p_lgb + p_xgb) / 3.0
    test_preds += fold_probs / n_splits

final_class_indices = np.argmax(test_preds, axis=1)
final_preds = le.inverse_transform(final_class_indices)

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'SIZE': final_preds
})
submission.to_csv('predictions.csv', index=False)