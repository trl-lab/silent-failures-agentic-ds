import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/test.parquet')

# Encode target
target_le = LabelEncoder()
y_train = target_le.fit_transform(train_df['antibody_truth'])

# Identify feature columns (exclude target)
feature_cols = [col for col in train_df.columns if col != 'antibody_truth']

# Encode all categorical features
train_encoded = train_df[feature_cols].copy()
test_encoded = test_df[feature_cols].copy()

# Fill missing values with a string placeholder
for col in feature_cols:
    train_encoded[col] = train_encoded[col].fillna('__MISSING__').astype(str)
    test_encoded[col] = test_encoded[col].fillna('__MISSING__').astype(str)

# Label encode all features
encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    combined = pd.concat([train_encoded[col], test_encoded[col]], axis=0)
    le.fit(combined)
    train_encoded[col] = le.transform(train_encoded[col])
    test_encoded[col] = le.transform(test_encoded[col])
    encoders[col] = le

# Create validation split
X_train, X_val, y_train_split, y_val_split = train_test_split(
    train_encoded, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Define models
models = {
    'lgbm': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    'catboost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
}

# Train models and evaluate on validation set
val_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train_split)
    val_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val_split, val_proba)
    val_scores[name] = score
    print(f"{name}: ROC AUC = {score:.4f}")

# Retrain all models on full training data and predict test
test_probabilities = []
for name, model in models.items():
    model.fit(train_encoded, y_train)
    test_proba = model.predict_proba(test_encoded)[:, 1]
    test_probabilities.append(test_proba)

# Ensemble: average probabilities
final_proba = np.mean(test_probabilities, axis=0)

# Save predictions
submission_df = pd.DataFrame({
    'id': range(len(final_proba)),
    'antibody_truth': final_proba
})
submission_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {submission_df.shape}")