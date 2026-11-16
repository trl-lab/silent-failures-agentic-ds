import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'number_health_problems'

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col])

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

for col in list(X_train_raw.columns):
    if col.lower() in ['id', 'index', 'unnamed: 0']:
        X_train_raw = X_train_raw.drop(columns=[col])
        if col in X_test_raw.columns:
            X_test_raw = X_test_raw.drop(columns=[col])

df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

for c in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype(df_all[c]):
        df_all[c + '_year'] = df_all[c].dt.year
        df_all[c + '_month'] = df_all[c].dt.month
        df_all[c + '_day'] = df_all[c].dt.day
        df_all[c + '_dayofweek'] = df_all[c].dt.dayofweek
        df_all = df_all.drop(columns=[c])

cat_cols = []
num_cols = []

for c in df_all.columns:
    if df_all[c].dtype == 'object' or df_all[c].dtype.name == 'category' or df_all[c].dtype == 'bool':
        cat_cols.append(c)
    else:
        num_cols.append(c)

for c in cat_cols:
    df_all[c] = df_all[c].astype(str)
    le = LabelEncoder()
    df_all[c] = le.fit_transform(df_all[c])

for c in num_cols:
    med = df_all[c].median()
    if pd.isna(med):
        med = 0
    df_all[c] = df_all[c].fillna(med)

X_train = df_all.iloc[:len(train_df)].copy()
X_test = df_all.iloc[len(train_df):].copy()

num_classes = len(np.unique(y))

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_lgb = np.zeros((len(X_test), num_classes))
test_preds_cb = np.zeros((len(X_test), num_classes))

lgb_params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000
}

cb_params = {
    'loss_function': 'MultiClass',
    'iterations': 1000,
    'learning_rate': 0.05,
    'random_seed': 42,
    'verbose': 0
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict_proba(X_test) / n_splits
    
    model_cb = cb.CatBoostClassifier(**cb_params)
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    test_preds_cb += model_cb.predict_proba(X_test) / n_splits

blend_test_preds = test_preds_lgb * 0.5 + test_preds_cb * 0.5
final_pred_indices = np.argmax(blend_test_preds, axis=1)
final_preds = le_target.inverse_transform(final_pred_indices)

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'number_health_problems': final_preds
})

output_df.to_csv('predictions.csv', index=False)