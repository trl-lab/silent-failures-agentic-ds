import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/dont-overfit-ii/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/dont-overfit-ii/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y = train_df['target'].values.astype(int)

drop_cols = ['id', 'target']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train_raw = train_df[feature_cols].values
X_test_raw = test_df[feature_cols].values

imputer = SimpleImputer(strategy='mean')
X_train_raw = imputer.fit_transform(X_train_raw)
X_test_raw = imputer.transform(X_test_raw)

def add_row_features(X):
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_std = np.std(X, axis=1, keepdims=True)
    row_min = np.min(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    row_pos = np.sum(X > 0, axis=1, keepdims=True)
    return np.hstack([X, row_mean, row_std, row_min, row_max, row_pos])

X_train = add_row_features(X_train_raw)
X_test = add_row_features(X_test_raw)

def get_models():
    return [
        ('log_l1_010', LogisticRegression(penalty='l1', C=0.10, solver='liblinear', random_state=42)),
        ('log_l1_015', LogisticRegression(penalty='l1', C=0.15, solver='liblinear', random_state=42)),
        ('log_l1_020', LogisticRegression(penalty='l1', C=0.20, solver='liblinear', random_state=42)),
        ('log_elastic', LogisticRegression(penalty='elasticnet', solver='saga', C=0.15, l1_ratio=0.5, max_iter=3000, random_state=42)),
        ('lasso_003', Lasso(alpha=0.03, max_iter=3000, random_state=42)),
        ('lasso_005', Lasso(alpha=0.05, max_iter=3000, random_state=42)),
        ('elasticnet_004', ElasticNet(alpha=0.04, l1_ratio=0.5, max_iter=3000, random_state=42)),
        ('lda_lsqr', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ('pipe_select_log', Pipeline([('select', SelectKBest(f_classif, k=35)), ('clf', LogisticRegression(C=0.1, penalty='l2', solver='liblinear'))])),
        ('pipe_select_ridge', Pipeline([('select', SelectKBest(f_classif, k=45)), ('clf', Ridge(alpha=10.0, random_state=42))]))
    ]

models = get_models()
n_models = len(models)
n_samples_train = len(y)
n_samples_test = len(X_test)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

oof_ranks = np.zeros((n_models, n_samples_train))
test_ranks = np.zeros((n_models, n_samples_test))

def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)
    
    for m_idx, (m_name, model_inst) in enumerate(get_models()):
        model_inst.fit(X_tr_sc, y_tr)
        
        va_pred = predict_score(model_inst, X_va_sc)
        te_pred = predict_score(model_inst, X_te_sc)
        
        oof_ranks[m_idx, val_idx] += rankdata(va_pred) / len(va_pred)
        test_ranks[m_idx] += rankdata(te_pred) / len(te_pred)

model_weights = []
for m_idx in range(n_models):
    score = roc_auc_score(y, oof_ranks[m_idx])
    weight = max(0.0, score - 0.5) ** 3
    model_weights.append(weight)

model_weights = np.array(model_weights)
if model_weights.sum() > 0:
    model_weights /= model_weights.sum()
else:
    model_weights = np.ones(n_models) / n_models

final_test_preds = np.zeros(n_samples_test)
for m_idx in range(n_models):
    final_test_preds += test_ranks[m_idx] * model_weights[m_idx]

final_test_preds = (final_test_preds - final_test_preds.min()) / (final_test_preds.max() - final_test_preds.min() + 1e-12)

submission = pd.DataFrame({
    'id': np.arange(n_samples_test),
    'target': final_test_preds
})

submission.to_csv('predictions.csv', index=False)