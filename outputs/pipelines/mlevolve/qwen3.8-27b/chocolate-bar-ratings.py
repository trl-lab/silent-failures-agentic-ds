import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

os.makedirs('./submission', exist_ok=True)

# Read data
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

target_col = 'Rating'
feature_cols = [c for c in train.columns if c != target_col and c != 'id']

X_all = train[feature_cols].copy()
y_all = train[target_col].copy()
X_test = test[feature_cols].copy()

# Split FIRST, then preprocess (no leakage)
train_idx, val_idx = train_test_split(
    np.arange(len(train)), test_size=0.2, random_state=42, stratify=y_all.values
)

X_train = X_all.iloc[train_idx].copy()
X_val = X_all.iloc[val_idx].copy()
y_train = y_all.iloc[train_idx].copy()
y_val = y_all.iloc[val_idx].copy()

# Fit encoders on TRAIN ONLY
encoders = {}
for col in X_train.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    encoders[col] = le

def transform_features(df, encs):
    df = df.copy()
    for col, le in encs.items():
        cls_map = {c: i for i, c in enumerate(le.classes_)}
        df[col] = df[col].astype(str).map(cls_map).fillna(-1).astype(int)
    return df

X_train_t = transform_features(X_train, encoders)
X_val_t = transform_features(X_val, encoders)
X_test_t = transform_features(X_test, encoders)

# Coerce to numeric and fill remaining NaNs
for df in (X_train_t, X_val_t, X_test_t):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_t, y_train)

# Validate with official metric (accuracy)
val_preds = model.predict(X_val_t)
val_acc = accuracy_score(y_val, val_preds)
print(f"Final Validation Score: {val_acc:.4f}")

# Test predictions
test_preds = model.predict(X_test_t)

# Write submission
submission = pd.DataFrame({'id': np.arange(len(test_preds)), 'Rating': test_preds})
submission.to_csv('./submission/submission.csv', index=False)