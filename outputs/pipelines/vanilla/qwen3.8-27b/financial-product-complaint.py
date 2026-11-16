import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/test.parquet')

# Define columns to use (drop text and ID columns)
drop_cols = ['Consumer complaint narrative', 'Company public response', 'Complaint ID', 'ZIP code', 'Tags']
drop_cols = [c for c in drop_cols if c in train_df.columns]

X_train = train_df.drop(['Company response to consumer'] + drop_cols, axis=1)
y_train = train_df['Company response to consumer']
X_test = test_df.drop(drop_cols, axis=1)

# Identify categorical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Encode target
target_le = LabelEncoder()
y_train_encoded = target_le.fit_transform(y_train)

# Encode categorical features
cat_le = {}
for col in cat_cols:
    le = LabelEncoder()
    # Combine train and test to fit
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    cat_le[col] = le

# Fill missing numeric values
for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# Define models
models = {
    'LGBM': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbose=-1
    ),
    'XGB': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
}

# Train models and get validation scores
val_probs = []
test_probs = []

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_prob = model.predict_proba(X_val)
    test_prob = model.predict_proba(X_test)
    val_probs.append(val_prob)
    test_probs.append(test_prob)
    val_acc = (val_prob.argmax(axis=1) == y_val).mean()
    print(f"{name} validation accuracy: {val_acc:.4f}")

# Ensemble: average probabilities
val_ensemble = np.mean(val_probs, axis=0)
test_ensemble = np.mean(test_probs, axis=0)

# Final validation accuracy
final_val_acc = (val_ensemble.argmax(axis=1) == y_val).mean()
print(f"Ensemble validation accuracy: {final_val_acc:.4f}")

# Generate predictions
test_pred_idx = test_ensemble.argmax(axis=1)
test_pred = target_le.inverse_transform(test_pred_idx)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_pred)),
    'Company response to consumer': test_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")