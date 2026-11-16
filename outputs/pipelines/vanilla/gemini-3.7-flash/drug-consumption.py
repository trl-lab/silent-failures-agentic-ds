import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'cannabis'
feature_cols = [col for col in train_df.columns if col != target_col]

# Handle any unexpected missing values
for col in feature_cols:
    if train_df[col].isnull().any() or test_df[col].isnull().any():
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Scale features for models that require scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Stratified K-Fold setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_probs = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]
    
    X_tr_sc = X_scaled[train_idx]
    X_va_sc = X_scaled[val_idx]

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    test_probs += lgb_model.predict_proba(X_test) / (n_splits * 4)

    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(X_tr, y_tr)
    test_probs += cb_model.predict_proba(X_test) / (n_splits * 4)

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        eval_metric='mlogloss',
        random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr)
    test_probs += xgb_model.predict_proba(X_test) / (n_splits * 4)

    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42 + fold
    )
    lr_model.fit(X_tr_sc, y_tr)
    test_probs += lr_model.predict_proba(X_test_scaled) / (n_splits * 4)

# Generate final predictions
final_pred_indices = np.argmax(test_probs, axis=1)
final_predictions = le.inverse_transform(final_pred_indices)

# Save submission
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'cannabis': final_predictions
})
output_df.to_csv('predictions.csv', index=False)