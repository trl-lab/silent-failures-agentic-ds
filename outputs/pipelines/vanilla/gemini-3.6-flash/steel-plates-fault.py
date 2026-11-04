import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "target"
X = train.drop(columns=[target_col])
y = train[target_col]

X_test = test.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

X_test = X_test[X.columns]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        le_feat = LabelEncoder()
        combined = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
        le_feat.fit(combined)
        X[col] = le_feat.transform(X[col].astype(str))
        X_test[col] = le_feat.transform(X_test[col].astype(str))

X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
X_test = X_test.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

med = X.median()
X = X.fillna(med)
X_test = X_test.fillna(med)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
num_classes = len(le.classes_)

test_preds_lgb = np.zeros((len(X_test), num_classes))
test_preds_xgb = np.zeros((len(X_test), num_classes))
test_preds_cat = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    
    clf_lgb = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    test_preds_lgb += clf_lgb.predict_proba(X_test) / n_splits
    
    clf_xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        eval_metric='mlogloss',
        verbosity=0
    )
    clf_xgb.fit(X_tr, y_tr)
    test_preds_xgb += clf_xgb.predict_proba(X_test) / n_splits

    clf_cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_state=42 + fold,
        verbose=0
    )
    clf_cat.fit(X_tr, y_tr)
    test_preds_cat += clf_cat.predict_proba(X_test) / n_splits

final_probs = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3.0
final_preds_encoded = np.argmax(final_probs, axis=1)
final_preds = le.inverse_transform(final_preds_encoded)

submission = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "target": final_preds
})

submission.to_csv("predictions.csv", index=False)