import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/otto-group-product-classification-challenge/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/otto-group-product-classification-challenge/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'target'
train = train.dropna(subset=[target_col]).reset_index(drop=True)

drop_cols = [c for c in ['id', 'ID', 'Id', target_col] if c in train.columns]
feature_cols = [c for c in train.columns if c not in drop_cols]

for col in feature_cols:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le_f = LabelEncoder()
        train[col] = le_f.fit_transform(train[col])
        test[col] = test[col].map(lambda s: le_f.transform([s])[0] if s in le_f.classes_ else -1)
    else:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

le_target = LabelEncoder()
y = le_target.fit_transform(train[target_col])
num_classes = len(le_target.classes_)

def extract_features(df, feat_cols):
    X = df[feat_cols].values.astype(np.float32)
    feats = pd.DataFrame(X, columns=feat_cols, index=df.index)
    feats['row_sum'] = np.sum(X, axis=1)
    feats['row_zeros'] = np.sum(X == 0, axis=1)
    feats['row_nonzeros'] = np.sum(X > 0, axis=1)
    feats['row_mean'] = np.mean(X, axis=1)
    feats['row_std'] = np.std(X, axis=1)
    feats['row_max'] = np.max(X, axis=1)
    return feats

X_train = extract_features(train, feature_cols)
X_test = extract_features(test, feature_cols)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(test), num_classes), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[trn_idx], y[trn_idx]
    
    clf_lgb = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        metric='multi_logloss',
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    test_preds += clf_lgb.predict_proba(X_test) / (5.0 * 2.0)
    
    clf_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    clf_xgb.fit(X_tr, y_tr)
    test_preds += clf_xgb.predict_proba(X_test) / (5.0 * 2.0)

final_class_indices = np.argmax(test_preds, axis=1)
final_predictions = le_target.inverse_transform(final_class_indices)

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'target': final_predictions
})

sub.to_csv('predictions.csv', index=False)