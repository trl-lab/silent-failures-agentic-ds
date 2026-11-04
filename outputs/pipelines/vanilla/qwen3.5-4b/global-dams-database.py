import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load Data
try:
    train_df = pd.read_parquet(TRAIN_PATH)
except Exception:
    raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")

try:
    test_df = pd.read_parquet(TEST_PATH)
except Exception:
    raise FileNotFoundError(f"Test file not found at {TEST_PATH}")

# Define Target
TARGET_COLUMN = 'Res_capacityM3'

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(f"Target column {TARGET_COLUMN} not found in train data")

# Extract Feature Columns
FEATURE_COLUMNS = [col for col in train_df.columns if col != TARGET_COLUMN]

# Validate Test Columns Match
valid_test_features = [col for col in FEATURE_COLUMNS if col in test_df.columns]
if len(valid_test_features) != len(FEATURE_COLUMNS):
    raise ValueError("Test data missing columns present in train data")

X_train_raw = train_df[FEATURE_COLUMNS].copy()
X_test_raw = test_df[valid_test_features].copy()

y_train_target = train_df[TARGET_COLUMN].values

# Step 1: Identify Numeric vs Categorical
numeric_features = []
categorical_features = []

for col in FEATURE_COLUMNS:
    dtype = X_train_raw[col].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        numeric_features.append(col)
    elif pd.api.types.is_object_dtype(dtype):
        categorical_features.append(col)
    else:
        # Try to convert other types to numeric or treat as object
        numeric_features.append(col)

# Step 2: Handle Categorical Features (Label Encoding + NA Imputation)
label_encoders = {}
if categorical_features:
    for col in categorical_features:
        # Impute NA with median of non-null values in train
        median_val = X_train_raw[col].median()
        X_train_raw[col] = X_train_raw[col].fillna(median_val)
        
        # Encode
        le = LabelEncoder()
        X_train_raw[col] = le.fit_transform(X_train_raw[col].astype(str))
        label_encoders[col] = le
        
        # Do the same for test (reuse the fitted encoder)
        X_test_raw[col] = le.transform(X_test_raw[col].astype(str))
        
    # Convert categorical encoded features to float64
    for col in categorical_features:
        X_train_raw[col] = X_train_raw[col].astype(float)
        X_test_raw[col] = X_test_raw[col].astype(float)

# Step 3: Handle Numeric Features (NA Imputation)
if numeric_features:
    imp = SimpleImputer(strategy='median')
    # Fit and transform
    X_train_numeric_imputed = imp.fit_transform(X_train_raw[numeric_features].values.reshape(-1, len(numeric_features)))
    X_test_numeric_imputed = imp.transform(X_test_raw[numeric_features].values.reshape(-1, len(numeric_features)))
    
    # Assign back
    X_train_raw[numeric_features] = np.transpose(X_train_numeric_imputed)
    X_test_raw[numeric_features] = np.transpose(X_test_numeric_imputed)

# Step 4: Standardize Features
scaler = StandardScaler()
X_train_all = scaler.fit_transform(X_train_raw.values)
X_test_all = scaler.transform(X_test_raw.values)

X_train_scaled = pd.DataFrame(X_train_all, columns=FEATURE_COLUMNS)
X_test_scaled = pd.DataFrame(X_test_all, columns=valid_test_features)

# Step 5: Train Model
model = lgb.LGBMRegressor(
    objective='regression', 
    metric='rmse', 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=-1, 
    random_state=42, 
    verbose=-1
)

model.fit(X_train_scaled.values, y_train_target, eval_set=(X_train_scaled.values, y_train_target), verbose=False)

# Step 6: Predictions
y_pred = model.predict(X_test_scaled.values)

# Step 7: Save Results
results_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'Res_capacityM3': y_pred
})
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"Predictions saved to {OUTPUT_PATH}")