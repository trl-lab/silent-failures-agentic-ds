import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

drug_targets = [
    'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc',
    'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd',
    'meth', 'mushrooms', 'nicotine', 'semer', 'vsa'
]

y_raw = train_df['cannabis']
le = LabelEncoder()
y = le.fit_transform(y_raw)

feature_cols = [c for c in train_df.columns if c in test_df.columns and c not in drug_targets and c != 'id']

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

num_classes = len(le.classes_)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_probs = np.zeros((len(X_test_imp), num_classes))
n_models = 4

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_imp, y)):
    X_tr, y_tr = X_train_imp[train_idx], y[train_idx]
    
    cb = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=5, random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr)
    test_probs += cb.predict_proba(X_test_imp) / (5 * n_models)
    
    lgb = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr)
    test_probs += lgb.predict_proba(X_test_imp) / (5 * n_models)
    
    xgb = XGBClassifier(n_estimators=250, learning_rate=0.03, max_depth=4, random_state=42, eval_metric='mlogloss')
    xgb.fit(X_tr, y_tr)
    test_probs += xgb.predict_proba(X_test_imp) / (5 * n_models)
    
    et = ExtraTreesClassifier(n_estimators=250, max_depth=10, random_state=42)
    et.fit(X_tr, y_tr)
    test_probs += et.predict_proba(X_test_imp) / (5 * n_models)

preds_encoded = np.argmax(test_probs, axis=1)
preds_labels = le.inverse_transform(preds_encoded)

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'cannabis': preds_labels
})

sub.to_csv('predictions.csv', index=False)