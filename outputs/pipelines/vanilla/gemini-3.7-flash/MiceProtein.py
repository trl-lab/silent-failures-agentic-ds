import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/MiceProtein/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/MiceProtein/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
feature_cols = [col for col in train_df.columns if col != target_col]

X_train = train_df[feature_cols].copy()
y_train_raw = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Label encode target
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

n_classes = len(le.classes_)
test_preds_proba = np.zeros((len(X_test), n_classes))

# 2. Train models with Stratified K-Fold ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_imputed, y_train)):
    X_tr, y_tr = X_train_imputed.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_imputed.iloc[val_idx], y_train[val_idx]
    
    models = [
        ExtraTreesClassifier(n_estimators=300, random_state=42 + fold),
        RandomForestClassifier(n_estimators=300, random_state=42 + fold),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42 + fold, verbose=-1),
        CatBoostClassifier(iterations=400, learning_rate=0.05, verbose=0, random_state=42 + fold)
    ]
    
    for model in models:
        model.fit(X_tr, y_tr)
        test_preds_proba += model.predict_proba(X_test_imputed) / (n_splits * len(models))

# 3. Get predicted class labels
final_preds_indices = np.argmax(test_preds_proba, axis=1)
final_preds_labels = le.inverse_transform(final_preds_indices)

# 4. Save predictions to CSV
submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds_labels)),
    'class': final_preds_labels
})
submission_df.to_csv('predictions.csv', index=False)