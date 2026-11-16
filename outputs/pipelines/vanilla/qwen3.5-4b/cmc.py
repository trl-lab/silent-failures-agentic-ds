import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define file paths based on the provided instructions
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/test.parquet'

# Load datasets
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify columns
target_col = 'Contractive_method_used'
# Note: Correcting typo from analysis prompt requirement: "Contraceptive_method_used"
# Based on dataset info: "Contraceptive method used" is the class attribute.
# In the prompt file listing, the column is named `Contraceptive_method_used`.
# Wait, the prompt data file section says: `Target: Contraceptive_method_used`.
# But the Attribute Info (point 7) says: "10. Contraceptive method used".
# And "Information about the dataset" says: "CLASSTYPE: nominal", "Target: `Contraceptive_method_used`".
# I will use the name provided in the OpenML target section for safety.
target_col_corrected = 'Contraceptive_method_used'

# Extract features and target
feature_columns = [col for col in train_df.columns if col != target_col_corrected]

# Separate features and target
X_train = train_df[feature_columns].copy()
y_train = train_df[target_col_corrected].copy()

# Separate test features and target
X_test = test_df[feature_columns].copy()
# We only need to predict; we don't strictly need y_test for evaluation in this script.
y_test_original = test_df[target_col_corrected]  # To ensure we know the original distribution if needed, but we will use our model.

# Step 1: Imputation for Numeric Variables
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train[numeric_features] = X_train[numeric_features].fillna(X_train[numeric_features].median())
X_test[numeric_features] = X_test[numeric_features].fillna(X_train[numeric_features].median())

# Step 2: Imputation for Categorical Variables
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
if len(categorical_features) > 0:
    # Use mode to fill missing categorical values
    train_mode_dict = {}
    for col in categorical_features:
        mode_val = X_train[col].mode().iloc[0]
        train_mode_dict[col] = mode_val
    
    for col in categorical_features:
        X_train[col] = X_train[col].fillna(train_mode_dict[col])
        X_test[col] = X_test[col].fillna(train_mode_dict[col])

# Step 3: One-Hot Encoding
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_encoded = ohe.fit_transform(X_train)
X_test_encoded = ohe.transform(X_test)

# Step 4: Label Encoding Target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Step 5: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_encoded, y_train_encoded)

# Step 6: Predict
y_pred_encoded = model.predict(X_test_encoded)
y_pred = le.inverse_transform(y_pred_encoded)

# Step 7: Create Output DataFrame
results_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'Contraceptive_method_used': y_pred
})

# Step 8: Save to CSV
results_df.to_csv('predictions.csv', index=False, header=True)