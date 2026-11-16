import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Define file paths
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/test.parquet"
output_path = "predictions.csv"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify features and target column
# Assume the last column is the target 'class' based on dataset description
# If multiple columns end similarly, select the one named 'class'
target_col_name = 'class'
if target_col_name not in train_df.columns:
    raise ValueError(f"Target column '{target_col_name}' not found in dataset.")

feature_columns = [col for col in train_df.columns if col != target_col_name]

# Prepare X and y
X_train = train_df[feature_columns]
y_train = train_df[target_col_name].values
X_test = test_df[feature_columns]

# Handle missing values automatically as requested
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Train LightGBM Classifier
model = lgb.LGBMClassifier(objective='binary', eval_metric='auc', n_estimators=1000, random_state=42, verbose=-1)
model.fit(X_train_imputed, y_train)

# Make predictions on test set
y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
y_pred_class = y_pred_proba.round().astype(int)

# Create output dataframe with 'id' and 'class'
predictions_output = pd.DataFrame({
    'id': range(len(y_pred_class)),
    'class': y_pred_class
})

# Save to CSV
predictions_output.to_csv(output_path, index=False)