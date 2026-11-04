import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'car'
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Encode categorical features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = pd.DataFrame(encoder.fit_transform(X), columns=feature_cols)
X_test_encoded = pd.DataFrame(encoder.transform(X_test), columns=feature_cols)

# Stratified K-Fold Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
num_classes = len(le.classes_)
test_probs = np.zeros((len(test_df), num_classes))

for train_idx, val_idx in skf.split(X_encoded, y_encoded):
    X_tr, y_tr = X_encoded.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X_encoded.iloc[val_idx], y_encoded[val_idx]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_tr, y_tr)
    test_probs += model.predict_proba(X_test_encoded) / n_splits

# Generate final predictions
test_preds_encoded = np.argmax(test_probs, axis=1)
test_preds = le.inverse_transform(test_preds_encoded)

# Save predictions to CSV
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'car': test_preds
})
output_df.to_csv('predictions.csv', index=False)