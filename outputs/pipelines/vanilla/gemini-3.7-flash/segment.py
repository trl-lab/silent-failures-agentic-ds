import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/segment/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/segment/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test = test_df.copy()

# Ensure matching columns
feature_cols = [c for c in X.columns if c in X_test.columns]
X = X[feature_cols]
X_test = X_test[feature_cols]

# Handle missing values if any
for col in feature_cols:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

# Cross-validated ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_proba = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    test_preds_proba += lgb.predict_proba(X_test) / 20.0

    # CatBoost
    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr)
    test_preds_proba += cb.predict_proba(X_test) / 20.0

    # ExtraTrees
    et = ExtraTreesClassifier(
        n_estimators=200,
        random_state=42 + fold,
        n_jobs=-1
    )
    et.fit(X_tr, y_tr)
    test_preds_proba += et.predict_proba(X_test) / 20.0

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42 + fold,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    test_preds_proba += rf.predict_proba(X_test) / 20.0

# Final predictions
final_pred_indices = np.argmax(test_preds_proba, axis=1)
final_preds = le.inverse_transform(final_pred_indices)

# Save submission
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})
output_df.to_csv('predictions.csv', index=False)