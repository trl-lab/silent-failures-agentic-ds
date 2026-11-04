import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import os

os.makedirs('./submission', exist_ok=True)
os.makedirs('./working', exist_ok=True)

train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"Target distribution:\n{train['Stars'].value_counts().sort_index()}")

y = train['Stars'].astype(int).values
X = train.drop(columns=['Stars']).copy()
X_test = test.copy()

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
bool_cols = [c for c in X.columns if str(X[c].dtype) == 'bool']
print(f"Categorical: {cat_cols}")
print(f"Numerical: {num_cols}")
print(f"Boolean: {bool_cols}")

for c in bool_cols:
    X[c] = X[c].astype(int)
    X_test[c] = X_test[c].astype(int)

label_encoders = {}
for c in cat_cols:
    X[c] = X[c].astype(str).fillna('missing')
    X_test[c] = X_test[c].astype(str).fillna('missing')
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c])
    label_encoders[c] = le
    unseen = set(X_test[c].unique()) - set(le.classes_)
    if unseen:
        mask = X_test[c].isin(unseen)
        X_test.loc[mask, c] = le.classes_[0]
    X_test[c] = le.transform(X_test[c])

for c in num_cols:
    if c not in bool_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(X[c].median())
        X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(X[c].median())

print(f"Final feature shape: {X.shape}")
print(f"DTypes:\n{X.dtypes}")

label_min = int(y.min())
label_max = int(y.max())
num_classes = label_max - label_min + 1
print(f"Label range: [{label_min}, {label_max}], classes: {num_classes}")

np.random.seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, val_idx = next(skf.split(X, y))
X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
y_tr, y_val = y[tr_idx], y[val_idx]
print(f"Train: {len(y_tr)}, Val: {len(y_val)}")

params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'num_threads': 4,
    'min_data_in_leaf': 5
}

y_tr_adj = y_tr - label_min
y_val_adj = y_val - label_min

dtrain = lgb.Dataset(X_tr.values, label=y_tr_adj)
dval = lgb.Dataset(X_val.values, label=y_val_adj, reference=dtrain)

callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=callbacks)

best_iter = int(model.best_iteration) if model.best_iteration is not None else 1
print(f"Best iteration: {best_iter}")

val_pred_proba = model.predict(X_val.values)
val_pred = np.argmax(val_pred_proba, axis=1) + label_min
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.6f}")

dtrain_full = lgb.Dataset(X.values, label=y - label_min)
model_full = lgb.train(params, dtrain_full, num_boost_round=max(best_iter, 1))

test_pred_proba = model_full.predict(X_test.values)
test_pred = np.argmax(test_pred_proba, axis=1) + label_min

submission = pd.DataFrame({'id': np.arange(len(test_pred), dtype=int), 'Stars': test_pred.astype(int)})
submission.to_csv('./submission/submission.csv', index=False)
print(f"Submission saved: {len(submission)} rows")
print(f"Pred dist:\n{pd.Series(test_pred).value_counts().sort_index()}")

print(f'Final Validation Score: {val_acc}')