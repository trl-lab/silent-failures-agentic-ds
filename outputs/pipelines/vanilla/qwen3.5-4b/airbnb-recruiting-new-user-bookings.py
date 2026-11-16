import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/airbnb-recruiting-new-user-bookings/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/airbnb-recruiting-new-user-bookings/data/test.parquet'
TARGET_COL = 'country_destination'
PREDICTION_FILE = 'predictions.csv'

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Remove id from features if present, ensure target is kept
if TARGET_COL in train_df.columns:
    target = train_df[TARGET_COL]
else:
    raise ValueError(f"Target column {TARGET_COL} not found in train data.")

# Identify features (excluding id if present, excluding target)
feature_columns = [col for col in train_df.columns if col != TARGET_COL and not col.startswith('id')]
if 'id' in feature_columns:
    feature_columns.remove('id')

# Separate numeric and categorical columns for preprocessing
numeric_features = []
categorical_features = []

for col in feature_columns:
    try:
        # Check if it's numeric
        numeric_features.append(col)
        continue
    except TypeError:
        pass
    
    # If it wasn't caught above, try treating as object/categorical
    # Note: Some numeric dtypes might be detected incorrectly if complex, 
    # but standard float/int checks catch most.
    # We assume object/string implies categorical for safety here.
    pass
    
    # Better check
    import inspect
    # Re-evaluate properly
    # Simple heuristic: pandas dtype
    pass

# Robust dtype checking
final_numeric = []
final_categorical = []
for col in feature_columns:
    dtype = train_df[col].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        final_numeric.append(col)
    else:
        final_categorical.append(col)

# Define Preprocessors
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='mode')
one_hot_encoder = OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(transformers=[
    ('num', num_imputer, final_numeric),
    ('cat', cat_imputer, final_categorical)
], remainder='passthrough')

# Target Encoding
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split and Fit Model (Optional cross-validation could be added, but straight fit is faster for single run script)
model = LGBMClassifier(objective='multiclass', num_leaves=31, max_depth=-1, random_state=42, verbose=-1)

# Create full pipeline for preprocessing + modeling (though LGBMClassifier takes raw arrays after transform)
# We manually process for clarity and control over the test transformation pipeline
pipe = Pipeline([
    ('transformer', preprocessor),
    ('model', model)
])

# Fit on Training Data
X_train_processed = pipe.transform(X_train)
y_train_processed = y_train.values.ravel()

# Train the model
model.fit(X_train_processed, y_train_processed)

# --- Testing ---

X_test_processed = pipe.transform(test_df)
predictions_encoded = model.predict(X_test_processed)

# Inverse Transform target
# Note: Need to flatten if 1D array was produced
predictions_decoded = target_encoder.inverse_transform(predictions_encoded).tolist()

# Ensure ID format matches requirements: integer row index starting from 0
output_ids = list(range(len(test_df)))
results_df = pd.DataFrame({
    'id': output_ids,
    'country_destination': predictions_decoded
})

# Save results
results_df.to_csv(PREDICTION_FILE, index=False)