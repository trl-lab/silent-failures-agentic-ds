import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/test.parquet')

# Separate features and target
X_train = train_df.drop('class', axis=1)
y_raw = train_df['class']
X_test = test_df.copy()

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(y_raw.astype(str))

# Convert features to numeric (handles categorical '0'/'1', floats, and missing values)
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Train LightGBM model
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict probabilities for the positive class (for roc_auc)
probabilities = model.predict_proba(X_test)[:, 1]

# Save predictions to CSV
output_df = pd.DataFrame({
    'id': range(len(probabilities)),
    'class': probabilities
})
output_df.to_csv('predictions.csv', index=False)