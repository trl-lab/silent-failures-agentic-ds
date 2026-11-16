import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class"
features = [c for c in train_df.columns if c != target_col]

X = train_df[features].copy()
y = train_df[target_col].copy()
X_test = test_df[features].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

medians = X.median()
X = X.fillna(medians)
X_test = X_test.fillna(medians)

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

def get_models(seed=42):
    return {
        'cb1': CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=3, random_seed=seed, verbose=0),
        'cb2': CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_seed=seed, verbose=0),
        'lgb1': LGBMClassifier(n_estimators=350, learning_rate=0.03, max_depth=4, num_leaves=15, colsample_bytree=0.8, subsample=0.8, random_state=seed, verbose=-1),
        'lgb2': LGBMClassifier(n_estimators=350, learning_rate=0.02, max_depth=6, num_leaves=31, colsample_bytree=0.7, subsample=0.7, random_state=seed, verbose=-1),
        'xgb1': XGBClassifier(n_estimators=350, learning_rate=0.03, max_depth=4, colsample_bytree=0.8, subsample=0.8, random_state=seed, eval_metric='logloss'),
        'xgb2': XGBClassifier(n_estimators=350, learning_rate=0.02, max_depth=5, colsample_bytree=0.7, subsample=0.7, random_state=seed, eval_metric='logloss'),
        'et': ExtraTreesClassifier(n_estimators=350, max_depth=14, min_samples_split=4, random_state=seed, n_jobs=-1),
        'rf': RandomForestClassifier(n_estimators=350, max_depth=12, min_samples_split=4, random_state=seed, n_jobs=-1),
        'hgb': HistGradientBoostingClassifier(max_iter=350, learning_rate=0.03, max_depth=5, random_state=seed),
        'lr': LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
    }

model_names = list(get_models().keys())
oof_preds = {m: np.zeros(len(X)) for m in model_names}
test_preds = {m: np.zeros(len(X_test)) for m in model_names}

n_repeats = 3
total_folds = 10 * n_repeats

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y_enc)):
    X_tr, y_tr = X.iloc[train_idx], y_enc[train_idx]
    X_val, y_val = X.iloc[val_idx], y_enc[val_idx]
    
    X_tr_s, X_val_s = X_scaled[train_idx], X_scaled[val_idx]
    
    models = get_models(seed=42 + fold)
    
    for name, model in models.items():
        if name == 'lr':
            model.fit(X_tr_s, y_tr)
            val_p = model.predict_proba(X_val_s)[:, 1]
            test_p = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            val_p = model.predict_proba(X_val)[:, 1]
            test_p = model.predict_proba(X_test)[:, 1]
            
        oof_preds[name][val_idx] += val_p / n_repeats
        test_preds[name] += test_p / total_folds

oof_matrix = np.array([oof_preds[m] for m in model_names])
test_matrix = np.array([test_preds[m] for m in model_names])

def loss_func(weights):
    weights = np.maximum(0, np.array(weights))
    if weights.sum() > 0:
        weights /= weights.sum()
    blend = np.dot(weights, oof_matrix)
    return -roc_auc_score(y_enc, blend)

init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1)] * len(model_names)
res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds)

opt_weights = np.maximum(0, res.x)
if opt_weights.sum() > 0:
    opt_weights /= opt_weights.sum()

final_test_preds = np.dot(opt_weights, test_matrix)

submission = pd.DataFrame({
    'id': np.arange(len(final_test_preds)),
    'Class': final_test_preds
})

submission.to_csv("predictions.csv", index=False)