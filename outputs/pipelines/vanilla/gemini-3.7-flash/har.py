import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/har/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/har/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
X_train = train_df.drop(columns=[target_col])
y_train_raw = train_df[target_col]
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Encode target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)

# Preprocessing: Imputation & Scaling
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_processed = scaler.transform(imputer.transform(X_test))

# Define an ensemble of strong models for high-dimensional feature data
models = [
    ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)),
    ('extra_trees', ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
    ('log_reg', LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1))
]

# Train models and accumulate probability predictions
test_probs = np.zeros((len(X_test), len(label_encoder.classes_)))

for name, model in models:
    model.fit(X_train_processed, y_train)
    test_probs += model.predict_proba(X_test_processed) / len(models)

# Get final predicted class labels
final_preds_encoded = np.argmax(test_probs, axis=1)
final_preds = label_encoder.inverse_transform(final_preds_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})
output_df.to_csv('predictions.csv', index=False)