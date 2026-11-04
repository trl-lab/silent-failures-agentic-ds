import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/test.parquet')

# Define target
target = 'Average or Greater than 30'

# Drop high-cardinality identifier columns that won't be predictive
drop_cols = ['School Code', 'Name', 'Address', 'Zip Code', 'Zip Ext', 'State Desc']

# Features to use
feature_cols = [c for c in train_df.columns if c not in drop_cols and c != target]

X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()
X_test = test_df[feature_cols].copy()

# Convert Prog Length to numeric
X_train['Prog Length'] = pd.to_numeric(X_train['Prog Length'], errors='coerce')
X_test['Prog Length'] = pd.to_numeric(X_test['Prog Length'], errors='coerce')

# Identify categorical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

# Fill missing values
for col in cat_cols:
    X_train[col] = X_train[col].fillna('MISSING')
    X_test[col] = X_test[col].fillna('MISSING')

for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Define models
models = {
    'lgbm': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    'catboost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )
}

# Train models and get validation predictions
val_preds = []
test_preds = []

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    val_preds.append(val_pred)
    test_pred = model.predict_proba(X_test)[:, 1]
    test_preds.append(test_pred)

# Ensemble: average of all models
val_ensemble = np.mean(val_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

# Save predictions
submission_df = pd.DataFrame({
    'id': range(len(test_ensemble)),
    'Average or Greater than 30': test_ensemble
})
submission_df.to_csv('predictions.csv', index=False)