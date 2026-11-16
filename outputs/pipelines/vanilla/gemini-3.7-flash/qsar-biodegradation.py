import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
target_col = 'Class'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Ensure matching columns in test if target column is present
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Handle categorical/object features if any
for col in X.columns:
    if X[col].dtype.name in ['category', 'object'] or X_test[col].dtype.name in ['category', 'object']:
        col_le = LabelEncoder()
        full_vals = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
        col_le.fit(full_vals)
        X[col] = col_le.transform(X[col].astype(str))
        X_test[col] = col_le.transform(X_test[col].astype(str))
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill any missing values with median
for col in X.columns:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Cross-validation and ensemble modelling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]

    # Models
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42 + fold
    )
    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=5,
        verbose=0,
        random_seed=42 + fold
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42 + fold,
        n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42 + fold,
        n_jobs=-1
    )

    models = [lgbm, xgb, cat, rf, et]
    val_pred_fold = np.zeros(len(X_va))
    test_pred_fold = np.zeros(len(X_test))

    for model in models:
        model.fit(X_tr, y_tr)
        val_pred_fold += model.predict_proba(X_va)[:, 1] / len(models)
        test_pred_fold += model.predict_proba(X_test)[:, 1] / len(models)

    oof_preds[val_idx] = val_pred_fold
    test_preds += test_pred_fold / n_splits

cv_auc = roc_auc_score(y_encoded, oof_preds)

# Save predictions
predictions_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Class': test_preds
})

predictions_df.to_csv('predictions.csv', index=False)