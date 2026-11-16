import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/car/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/car/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y.astype(str))

# Encode categorical features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = encoder.fit_transform(X.astype(str))
X_test_encoded = encoder.transform(X_test.astype(str))

# Stratified K-Fold CV Ensembling
n_classes = len(le.classes_)
test_preds_proba = np.zeros((len(test_df), n_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y_encoded)):
    X_tr, y_tr = X_encoded[train_idx], y_encoded[train_idx]
    X_va, y_val = X_encoded[val_idx], y_encoded[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    test_preds_proba += lgb_model.predict_proba(X_test_encoded) / (2 * 5)
    
    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr)
    test_preds_proba += cb_model.predict_proba(X_test_encoded) / (2 * 5)

# Get final predicted class labels
final_preds_encoded = np.argmax(test_preds_proba, axis=1)
final_preds = le.inverse_transform(final_preds_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'class': final_preds
})
output_df.to_csv('predictions.csv', index=False)