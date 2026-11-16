import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

# Load train and test data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
target_col = 'class'
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Fill any missing values if present
for col in feature_cols:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Stratified K-Fold ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_probs = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]

    # LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    test_probs += lgb_model.predict_proba(X_test) / (2 * n_splits)

    # CatBoost Classifier
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr)
    test_probs += cb_model.predict_proba(X_test) / (2 * n_splits)

# Predict class indices and inverse transform to original labels
pred_indices = np.argmax(test_probs, axis=1)
pred_classes = le.inverse_transform(pred_indices)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(pred_classes)),
    'class': pred_classes
})
output_df.to_csv('predictions.csv', index=False)