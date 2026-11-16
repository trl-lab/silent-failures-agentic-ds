import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Define file paths as specified in the prompt
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Extract features and target based on the Variable Descriptions table
feature_cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", 
                "fM3Long", "fM3Trans", "fAlpha", "fDist"]
target_col = "class"

# Prepare training features and target
y_train = df_train[target_col].values.astype(int)
X_train_raw = df_train[feature_cols].dropna(axis=1, how='all').copy()

# Prepare test features
X_test_raw = df_test[feature_cols].copy()

# Combine train data to handle missing values consistently
X_combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Ensure all feature columns are numeric
for col in feature_cols:
    if X_combined[col].dtype != 'float64' and X_combined[col].dtype != 'int64':
        X_combined[col] = pd.to_numeric(X_combined[col], errors='coerce')

# Impute missing values with median across the combined dataset
imputer = SimpleImputer(strategy='median')
X_combined_imputed = imputer.fit_transform(X_combined)

# Split back into train and test for inference
X_train_imputed = X_combined_imputed[:len(X_train_raw)]
X_test_imputed = X_combined_imputed[len(X_train_raw):]

# Initialize and train LightGBM model
# Using default parameters which are generally robust
model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    verbose=-1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_imputed, y_train)

# Generate predictions for the test set
# LGBMClassifier.predict_proba returns shape (n_samples, n_classes)
probas = model.predict_proba(X_test_imputed)[:, 1]

# Convert probabilities to binary class predictions (threshold 0.5)
predicted_class = (probas >= 0.5).astype(int)

# Create output dataframe
results_df = pd.DataFrame({
    'id': np.arange(len(df_test)),
    'class': predicted_class
})

# Save to csv
results_df.to_csv(OUTPUT_PATH, index=False)