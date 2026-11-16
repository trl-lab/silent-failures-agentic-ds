import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "ViolentCrimesPerPop"
y_train = train_df[target_col].values

drop_cols = [target_col, "fold", "communityname"]
drop_cols = [c for c in drop_cols if c in train_df.columns]

X_train = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=drop_cols)

for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

valid_cols = [col for col in X_train.columns if X_train[col].notna().any()]
X_train = X_train[valid_cols]
X_test = X_test[valid_cols]

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_imp = scaler.transform(imputer.transform(X_test))

models = {
    "ridge": (RidgeCV(alphas=np.logspace(-2, 4, 100)), True),
    "elasticnet": (ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=5, random_state=42, max_iter=3000), True),
    "bayesian_ridge": (BayesianRidge(), True),
    "catboost": (CatBoostRegressor(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=5, random_seed=42, verbose=0), False),
    "lgb": (lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, num_leaves=7, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1), False),
    "xgb": (xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbosity=0), False),
    "hgb": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, l2_regularization=1.0, random_state=42), False),
    "et": (ExtraTreesRegressor(n_estimators=300, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42), True),
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_preds = {name: np.zeros(len(y_train)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

for name, (model_inst, uses_imp) in models.items():
    X_tr_data = X_train_imp if uses_imp else X_train
    X_te_data = X_test_imp if uses_imp else X_test
    
    for train_idx, val_idx in kf.split(X_tr_data):
        if uses_imp:
            X_tr, y_tr = X_tr_data[train_idx], y_train[train_idx]
            X_va = X_tr_data[val_idx]
        else:
            X_tr, y_tr = X_tr_data.iloc[train_idx], y_train[train_idx]
            X_va = X_tr_data.iloc[val_idx]
            
        model = clone(model_inst)
        model.fit(X_tr, y_tr)
        
        oof_preds[name][val_idx] = model.predict(X_va)
        test_preds[name] += model.predict(X_te_data) / kf.n_splits

names = list(models.keys())
OOF_matrix = np.column_stack([oof_preds[name] for name in names])
Test_matrix = np.column_stack([test_preds[name] for name in names])

def loss_func(weights):
    pred = OOF_matrix @ weights
    return np.sqrt(np.mean((y_train - pred) ** 2))

n_models = len(names)
init_weights = np.ones(n_models) / n_models
bounds = [(0, 1) for _ in range(n_models)]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

final_test_preds = Test_matrix @ best_weights

min_val = y_train.min()
max_val = y_train.max()
final_test_preds = np.clip(final_test_preds, min_val, max_val)

submission = pd.DataFrame({
    'id': np.arange(len(final_test_preds)),
    'ViolentCrimesPerPop': final_test_preds
})
submission.to_csv('predictions.csv', index=False)