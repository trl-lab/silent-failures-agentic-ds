import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Female (%)'
y_train = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or X_train[c].dtype.name == 'category']
num_cols = [c for c in X_train.columns if c not in cat_cols]

for col in cat_cols:
    X_train[col] = X_train[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_cat_encoded = oe.fit_transform(X_train[cat_cols])
X_test_cat_encoded = oe.transform(X_test[cat_cols])

X_train_num = X_train[num_cols].values
X_test_num = X_test[num_cols].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_target_enc_occ = np.zeros(len(X_train))
test_target_enc_occ = np.zeros(len(X_test))

global_mean = np.mean(y_train)
occ_col = 'Occupation Code' if 'Occupation Code' in X_train.columns else cat_cols[0]

for train_idx, val_idx in kf.split(X_train):
    tr_df = pd.DataFrame({'occ': X_train.iloc[train_idx][occ_col], 'target': y_train[train_idx]})
    means = tr_df.groupby('occ')['target'].agg(['mean', 'count'])
    smooth = 10
    smoothed_means = (means['count'] * means['mean'] + smooth * global_mean) / (means['count'] + smooth)
    
    val_occ = X_train.iloc[val_idx][occ_col]
    oof_target_enc_occ[val_idx] = val_occ.map(smoothed_means).fillna(global_mean).values

full_tr_df = pd.DataFrame({'occ': X_train[occ_col], 'target': y_train})
full_means = full_tr_df.groupby('occ')['target'].agg(['mean', 'count'])
smooth = 10
full_smoothed = (full_means['count'] * full_means['mean'] + smooth * global_mean) / (full_means['count'] + smooth)
test_target_enc_occ = X_test[occ_col].map(full_smoothed).fillna(global_mean).values

X_tr_all = np.hstack([X_train_num, X_train_cat_encoded, oof_target_enc_occ.reshape(-1, 1)])
X_te_all = np.hstack([X_test_num, X_test_cat_encoded, test_target_enc_occ.reshape(-1, 1)])

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_tr_all):
    X_tr_fold, y_tr_fold = X_tr_all[train_idx], y_train[train_idx]
    X_va_fold, y_va_fold = X_tr_all[val_idx], y_train[val_idx]
    
    lgb_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    lgb_model.fit(X_tr_fold, y_tr_fold)
    lgb_preds += lgb_model.predict(X_te_all) / 5.0
    
    cb_model = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    cb_model.fit(X_tr_fold, y_tr_fold)
    cb_preds += cb_model.predict(X_te_all) / 5.0
    
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0)
    xgb_model.fit(X_tr_fold, y_tr_fold)
    xgb_preds += xgb_model.predict(X_te_all) / 5.0

final_preds = (lgb_preds + cb_preds + xgb_preds) / 3.0
final_preds = np.clip(final_preds, 0.0, 100.0)

submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Female (%)': final_preds
})

submission_df.to_csv('predictions.csv', index=False)