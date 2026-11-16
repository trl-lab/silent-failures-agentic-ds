import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Damage Description (total)'

train = train.dropna(subset=[target_col]).reset_index(drop=True)

le = LabelEncoder()
y_train = le.fit_transform(train[target_col])
num_classes = len(le.classes_)

features = [c for c in train.columns if c != target_col]

X_train = train[features].copy()
X_test = test[features].copy()

all_missing = [c for c in features if X_train[c].isna().all() and X_test[c].isna().all()]
if all_missing:
    X_train.drop(columns=all_missing, inplace=True)
    X_test.drop(columns=all_missing, inplace=True)

for c in list(X_train.columns):
    if pd.api.types.is_datetime64_any_dtype(X_train[c]) or 'date' in c.lower() or 'time' in c.lower():
        try:
            dt_tr = pd.to_datetime(X_train[c], errors='coerce')
            dt_te = pd.to_datetime(X_test[c], errors='coerce')
            if dt_tr.notna().sum() > 0:
                X_train[c + '_year'] = dt_tr.dt.year
                X_train[c + '_month'] = dt_tr.dt.month
                X_train[c + '_day'] = dt_tr.dt.day
                X_train[c + '_hour'] = dt_tr.dt.hour

                X_test[c + '_year'] = dt_te.dt.year
                X_test[c + '_month'] = dt_te.dt.month
                X_test[c + '_day'] = dt_te.dt.day
                X_test[c + '_hour'] = dt_te.dt.hour

                X_train.drop(columns=[c], inplace=True)
                X_test.drop(columns=[c], inplace=True)
        except Exception:
            pass

cat_cols = []
for c in X_train.columns:
    if X_train[c].dtype == 'object' or X_train[c].dtype.name == 'category':
        cat_cols.append(c)

if cat_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols].astype(str))
    X_test[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

for c in X_train.columns:
    X_train[c] = pd.to_numeric(X_train[c], errors='coerce').astype(np.float32)
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce').astype(np.float32)

def clean_name(name):
    return re.sub(r'[\[\]<>\:,\"\']', '_', str(name))

X_train.columns = [clean_name(c) for c in X_train.columns]
X_test.columns = [clean_name(c) for c in X_test.columns]

seen = {}
cols = []
for c in X_train.columns:
    if c in seen:
        seen[c] += 1
        cols.append(f"{c}_{seen[c]}")
    else:
        seen[c] = 0
        cols.append(c)
X_train.columns = cols
X_test.columns = cols

class_counts = pd.Series(y_train).value_counts()
min_samples = class_counts.min() if len(class_counts) > 0 else 0
n_splits = min(5, min_samples)

if n_splits < 2:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
else:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((len(X_test), num_classes), dtype=np.float64)
total_weight = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]

    try:
        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        lgb.fit(X_tr, y_tr)
        probs = lgb.predict_proba(X_test)
        if probs.shape[1] == num_classes:
            test_preds += probs
            total_weight += 1.0
    except Exception:
        pass

    try:
        cat = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1
        )
        cat.fit(X_tr, y_tr)
        probs = cat.predict_proba(X_test)
        if probs.shape[1] == num_classes:
            test_preds += probs
            total_weight += 1.0
    except Exception:
        pass

    try:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
            eval_metric='mlogloss' if num_classes > 2 else 'logloss'
        )
        xgb.fit(X_tr, y_tr)
        probs = xgb.predict_proba(X_test)
        if probs.shape[1] == num_classes:
            test_preds += probs
            total_weight += 1.0
    except Exception:
        pass

if total_weight > 0:
    test_preds /= total_weight

pred_indices = np.argmax(test_preds, axis=1)
final_preds = le.inverse_transform(pred_indices)

df_out = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})
df_out.to_csv('predictions.csv', index=False)