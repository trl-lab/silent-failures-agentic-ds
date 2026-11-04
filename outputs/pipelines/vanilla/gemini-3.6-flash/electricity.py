import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'

le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col])

X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col]) if target_col in test_df.columns else test_df.copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

for c in df_all.columns:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

features_to_roll = ['NSWprice', 'NSWdemand', 'VICprice', 'VICdemand', 'transfer']

new_features = {}

for col in features_to_roll:
    if col in df_all.columns:
        for w in [3, 6, 12, 24, 48]:
            rmean = df_all[col].rolling(window=w, min_periods=1).mean()
            new_features[f'{col}_roll_mean_{w}'] = rmean
            new_features[f'{col}_diff_mean_{w}'] = df_all[col] - rmean
        
        for w in [12, 48]:
            new_features[f'{col}_roll_std_{w}'] = df_all[col].rolling(window=w, min_periods=1).std().fillna(0)
            new_features[f'{col}_roll_min_{w}'] = df_all[col].rolling(window=w, min_periods=1).min()
            new_features[f'{col}_roll_max_{w}'] = df_all[col].rolling(window=w, min_periods=1).max()
        
        for lag in [1, 2, 48]:
            new_features[f'{col}_lag_{lag}'] = df_all[col].shift(lag).bfill()
            new_features[f'{col}_diff_{lag}'] = df_all[col] - df_all[col].shift(lag).bfill()

if 'NSWdemand' in df_all.columns and 'VICdemand' in df_all.columns:
    new_features['demand_diff'] = df_all['NSWdemand'] - df_all['VICdemand']
if 'NSWprice' in df_all.columns and 'VICprice' in df_all.columns:
    new_features['price_diff'] = df_all['NSWprice'] - df_all['VICprice']

if 'Period' in df_all.columns:
    p = df_all['Period']
    new_features['period_sin'] = np.sin(2 * np.pi * p / 48.0)
    new_features['period_cos'] = np.cos(2 * np.pi * p / 48.0)

if 'Day' in df_all.columns:
    d = df_all['Day']
    new_features['day_sin'] = np.sin(2 * np.pi * d / 7.0)
    new_features['day_cos'] = np.cos(2 * np.pi * d / 7.0)

df_features = pd.DataFrame(new_features, index=df_all.index)
df_full = pd.concat([df_all, df_features], axis=1)

X_train_feat = df_full.iloc[:n_train].copy().fillna(0)
X_test_feat = df_full.iloc[n_train:].copy().fillna(0)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

num_classes = len(np.unique(y_train))
lgb_preds = np.zeros((len(X_test_feat), num_classes))
cb_preds = np.zeros((len(X_test_feat), num_classes))
xgb_preds = np.zeros((len(X_test_feat), num_classes))

for train_idx, val_idx in skf.split(X_train_feat, y_train):
    X_tr, y_tr = X_train_feat.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_feat.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test_feat) / skf.n_splits

    model_cb = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cb_preds += model_cb.predict_proba(X_test_feat) / skf.n_splits

    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test_feat) / skf.n_splits

final_probs = (lgb_preds + cb_preds + xgb_preds) / 3.0
pred_indices = np.argmax(final_probs, axis=1)
pred_labels = le.inverse_transform(pred_indices)

output_df = pd.DataFrame({
    'id': np.arange(len(pred_labels)),
    'class': pred_labels
})
output_df.to_csv('predictions.csv', index=False)