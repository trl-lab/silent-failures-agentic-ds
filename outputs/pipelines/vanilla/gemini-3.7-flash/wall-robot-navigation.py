import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# Load train and test datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle any missing values if present
for col in feature_cols:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y.astype(str))
n_classes = len(le.classes_)

# Stratified K-Fold ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_proba = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]
    
    # Model 1: LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    
    # Model 2: CatBoost
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42 + fold,
        verbose=False
    )
    cb_model.fit(X_tr, y_tr)
    
    # Model 3: Extra Trees
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42 + fold,
        n_jobs=-1
    )
    et_model.fit(X_tr, y_tr)

    # Accumulate test predictions
    test_preds_proba += (
        0.4 * lgb_model.predict_proba(X_test) +
        0.4 * cb_model.predict_proba(X_test) +
        0.2 * et_model.predict_proba(X_test)
    ) / n_splits

final_class_indices = np.argmax(test_preds_proba, axis=1)
final_predictions = le.inverse_transform(final_class_indices)

# Match the original target data type if possible
if isinstance(train_df[target_col].dtype, pd.CategoricalDtype):
    original_categories = train_df[target_col].cat.categories
    try:
        final_predictions = pd.Categorical(final_predictions, categories=original_categories)
    except Exception:
        pass
elif np.issubdtype(train_df[target_col].dtype, np.number):
    try:
        final_predictions = final_predictions.astype(train_df[target_col].dtype)
    except Exception:
        pass

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'class': final_predictions
})
output_df.to_csv('predictions.csv', index=False)