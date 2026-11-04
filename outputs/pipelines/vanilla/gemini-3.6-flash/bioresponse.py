import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = train['Activity'].values
drop_cols = [c for c in ['Activity', 'id'] if c in train.columns]
feature_cols = [c for c in train.columns if c not in drop_cols]

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

cat_cols = X.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    X = pd.get_dummies(X, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

X = X.fillna(0)
X_test = X_test.fillna(0)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

models_def = {
    'lgb1': lambda: LGBMClassifier(n_estimators=700, learning_rate=0.02, num_leaves=31, random_state=42, verbose=-1, colsample_bytree=0.7, subsample=0.8, n_jobs=-1),
    'lgb2': lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, random_state=43, verbose=-1, colsample_bytree=0.5, subsample=0.7, n_jobs=-1),
    'xgb1': lambda: XGBClassifier(n_estimators=700, learning_rate=0.02, max_depth=6, random_state=42, eval_metric='logloss', colsample_bytree=0.7, subsample=0.8, n_jobs=-1),
    'xgb2': lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=8, random_state=43, eval_metric='logloss', colsample_bytree=0.5, subsample=0.7, n_jobs=-1),
    'cat1': lambda: CatBoostClassifier(iterations=700, learning_rate=0.03, depth=6, random_state=42, verbose=0, thread_count=-1),
    'et1': lambda: ExtraTreesClassifier(n_estimators=500, max_features='sqrt', min_samples_split=4, random_state=42, n_jobs=-1),
    'rf1': lambda: RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_split=4, random_state=42, n_jobs=-1),
}

oof_preds = {name: np.zeros(len(train)) for name in models_def}
test_preds = {name: np.zeros(len(test)) for name in models_def}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    for name, model_fn in models_def.items():
        model = model_fn()
        model.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds[name] += model.predict_proba(X_test)[:, 1] / n_splits

oof_matrix = np.column_stack([oof_preds[name] for name in models_def])
test_matrix = np.column_stack([test_preds[name] for name in models_def])

def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = np.dot(oof_matrix, w)
    blend_oof = np.clip(blend_oof, 1e-15, 1 - 1e-15)
    return log_loss(y, blend_oof)

init_weights = [1.0 / len(models_def)] * len(models_def)
bounds = [(0, 1)] * len(models_def)
res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds)

best_weights = res.x / np.sum(res.x)
final_test_preds = np.dot(test_matrix, best_weights)
final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)

submission = pd.DataFrame({
    'id': np.arange(len(final_test_preds)),
    'Activity': final_test_preds
})
submission.to_csv('predictions.csv', index=False)