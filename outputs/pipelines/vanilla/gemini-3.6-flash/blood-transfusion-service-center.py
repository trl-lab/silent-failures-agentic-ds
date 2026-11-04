import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Class'
y_raw = train_df[target_col]
unique_targets = sorted(y_raw.unique())
target_map = {val: i for i, val in enumerate(unique_targets)}
y = y_raw.map(target_map).values

drop_cols = [target_col]
if 'id' in train_df.columns:
    drop_cols.append('id')

feature_cols = [c for c in train_df.columns if c not in drop_cols]

def add_features(df, cols):
    X = df[cols].copy()
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) >= 4:
        r, f, m, t = num_cols[0], num_cols[1], num_cols[2], num_cols[3]
        X['rf_ratio'] = X[r] / (X[f] + 1e-5)
        X['rt_ratio'] = X[r] / (X[t] + 1e-5)
        X['ft_ratio'] = X[f] / (X[t] + 1e-5)
        X['active_time'] = X[t] - X[r]
        X['avg_interval'] = (X[t] - X[r]) / np.maximum(X[f] - 1, 1)
        X['is_first_donation'] = (X[f] == 1).astype(float)
        X['recency_vs_interval'] = X[r] - X['avg_interval']
    return X

X_train = add_features(train_df, feature_cols)
X_test = add_features(test_df, feature_cols)

for c in X_train.columns:
    if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'category':
        X_train[c] = X_train[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
lr_preds = np.zeros(len(X_test))
et_preds = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(X_train_imp, y):
    X_tr, y_tr = X_train_imp[train_idx], y[train_idx]
    X_tr_sc = X_train_scaled[train_idx]
    
    model_lgb = lgb.LGBMClassifier(
        max_depth=3, num_leaves=7, learning_rate=0.03, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        random_state=42, verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_preds += model_lgb.predict_proba(X_test_imp)[:, 1] / skf.n_splits

    model_xgb = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.03, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42
    )
    model_xgb.fit(X_tr, y_tr)
    xgb_preds += model_xgb.predict_proba(X_test_imp)[:, 1] / skf.n_splits

    model_cat = CatBoostClassifier(
        depth=4, learning_rate=0.03, iterations=200,
        l2_leaf_reg=3.0, random_seed=42, verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    cat_preds += model_cat.predict_proba(X_test_imp)[:, 1] / skf.n_splits

    model_lr = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    model_lr.fit(X_tr_sc, y_tr)
    lr_preds += model_lr.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

    model_et = ExtraTreesClassifier(
        n_estimators=150, max_depth=5, min_samples_split=5, random_state=42
    )
    model_et.fit(X_tr, y_tr)
    et_preds += model_et.predict_proba(X_test_imp)[:, 1] / skf.n_splits

final_preds = 0.25 * lgb_preds + 0.25 * xgb_preds + 0.25 * cat_preds + 0.15 * lr_preds + 0.10 * et_preds

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Class': final_preds
})
sub.to_csv('predictions.csv', index=False)