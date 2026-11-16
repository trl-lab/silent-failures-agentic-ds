import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'readmitted'
y_train_raw = train_df[target_col]
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Encode target
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train_raw)

# Preprocess categorical features
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    le = LabelEncoder()
    unique_vals = list(set(X_train[col].unique()).union(set(X_test[col].unique())))
    le.fit(unique_vals)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Train LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict on test data
test_preds = model.predict(X_test)
test_preds_labels = target_encoder.inverse_transform(test_preds)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'readmitted': test_preds_labels
})
output_df.to_csv('predictions.csv', index=False)