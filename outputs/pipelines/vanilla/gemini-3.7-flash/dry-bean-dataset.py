import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Class'
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Impute any missing values if present
for col in feature_cols:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Encode target classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

# Stratified K-Fold ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]
    
    # LightGBM Classifier
    lgb = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr)
    test_preds += lgb.predict_proba(X_test) / (skf.n_splits * 2)
    
    # CatBoost Classifier
    cat = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    test_preds += cat.predict_proba(X_test) / (skf.n_splits * 2)

# Convert predicted class indices back to original class names
final_preds = np.argmax(test_preds, axis=1)
final_labels = le.inverse_transform(final_preds)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Class': final_labels
})
output_df.to_csv('predictions.csv', index=False)