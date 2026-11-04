import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Load datasets
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/test.parquet')

# Separate features and target
X = train_df.drop('Utility', axis=1).copy()
y = train_df['Utility'].copy()
X_test = test_df.copy()

# Encode target
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y.astype(str))
n_classes = len(target_le.classes_)

# Identify column types
cat_cols = [c for c in X.columns if str(X[c].dtype) in ['category', 'object'] or X[c].dtype == 'object']
num_cols = [c for c in X.columns if c not in cat_cols]

# 1. CatBoost Data Preparation (convert categoricals to string after replacing NaN)
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(object).fillna('__missing__').astype(str)
    X_test_cb[col] = X_test_cb[col].astype(object).fillna('__missing__').astype(str)

# 2. LightGBM Data Preparation (pandas categorical dtype)
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype(object).fillna('__missing__').astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype(object).fillna('__missing__').astype('category')

# 3. HistGradientBoosting Data Preparation (integer encoded categoricals)
X_hgb = X.copy()
X_test_hgb = X_test.copy()
cat_indices_hgb = []
for i, col in enumerate(X.columns):
    if col in cat_cols:
        cat_indices_hgb.append(i)
        oe = LabelEncoder()
        train_vals = X[col].astype(object).fillna('__missing__').astype(str)
        test_vals = X_test[col].astype(object).fillna('__missing__').astype(str)
        comb = pd.concat([train_vals, test_vals], axis=0)
        oe.fit(comb)
        X_hgb[col] = oe.transform(train_vals)
        X_test_hgb[col] = oe.transform(test_vals)

# Stratified K-Fold Ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_probs = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_lgb.iloc[train_idx], y_encoded[train_idx])
    test_probs += lgb_model.predict_proba(X_test_lgb) * 0.4

    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        verbose=False,
        random_seed=42 + fold
    )
    cb_model.fit(X_cb.iloc[train_idx], y_encoded[train_idx])
    test_probs += cb_model.predict_proba(X_test_cb) * 0.4

    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        categorical_features=cat_indices_hgb,
        random_state=42 + fold
    )
    hgb_model.fit(X_hgb.iloc[train_idx], y_encoded[train_idx])
    test_probs += hgb_model.predict_proba(X_test_hgb) * 0.2

# Get final predicted class
final_preds_encoded = np.argmax(test_probs, axis=1)
final_preds = target_le.inverse_transform(final_preds_encoded)

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Utility': final_preds
})
submission.to_csv('predictions.csv', index=False)