import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
feature_cols = [col for col in train_df.columns if col != target_col]

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Encode features as integers / categorical codes
for col in feature_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)
    X_train[col] = X_train[col].cat.codes
    X_test[col] = X_test[col].cat.codes

# Encode target
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)

# Train model
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train_encoded)

# Predict on test set
y_pred_encoded = model.predict(X_test)
y_pred = target_encoder.inverse_transform(y_pred_encoded)

# Save predictions
predictions_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'class': y_pred
})
predictions_df.to_csv('predictions.csv', index=False)