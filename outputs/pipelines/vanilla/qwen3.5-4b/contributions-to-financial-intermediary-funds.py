import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from scipy.sparse import hstack
import lightgbm as lgb

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/contributions-to-financial-intermediary-funds/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/contributions-to-financial-intermediary-funds/data/test.parquet'

# Load Data
print("Loading data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify Target Column
TARGET_COL = "Amount in USD"
# Check if column exists, if not try find exact match
if TARGET_COL not in df_train.columns:
    TARGET_COL = [col for col in df_train.columns if "USD" in col and "Amount" in col][0]

# Separate Features and Target
def prepare_dataframe(df):
    df_clean = df.copy()
    # Drop target column for modeling
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]
    # Drop rows with any missing values in features to prevent imputation confusion later (keep only complete)
    X_nan_count = X.isna().sum()
    # We need to ensure we keep the same rows for prediction if possible, so impute first.
    return X, y, df_clean.columns.tolist()

X_train, y_train, all_cols = prepare_dataframe(df_train)
X_test, y_test, _ = prepare_dataframe(df_test)

# Separate Numerical and Categorical Columns for preprocessing
numerical_features = X_train.select_dtypes(include=['int64', 'float64', 'float32']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Define Transformers
num_imputer = SimpleImputer(strategy='median')
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_imputer, numerical_features),
        ('cat', cat_encoder, categorical_features)
    ],
    remainder='drop'
)

# Compile Pipeline
# Note: To use lightgbm with sparse outputs from OHE efficiently, we handle dense/sparse conversion.
# Since OHE returns sparse by default, we convert to dense or ensure sparse support.
pipe_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', lgb.LGBMRegressor objective="regression", force_col_wise=True))
]
# Actually, LGBMRegressor does not support Pipeline directly via .fit if sparse output changes shape incorrectly.
# It is safer to apply preprocessing before fitting LGBM to avoid SparseData issues.
# Let's use a custom function to handle the transformation properly.

# Apply Preprocessing manually
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# 1. Handle Numerical
X_train_processed[numerical_features] = num_imputer.fit_transform(X_train[numerical_features])
X_test_processed[numerical_features] = num_imputer.transform(X_test[numerical_features])

# 2. Handle Categorical & One Hot Encode
try:
    cat_encoded_train = cat_encoder.fit_transform(X_train[categorical_features])
    cat_encoded_test = cat_encoder.transform(X_test[categorical_features])
except Exception:
    # Fallback for older sklearn versions or different column types
    X_train_processed[categorical_features] = X_train[categorical_features].astype(str)
    X_test_processed[categorical_features] = X_test[categorical_features].astype(str)
    pass

# Concatenate processed data
import warnings
warnings.filterwarnings('ignore')

X_train_combined = pd.DataFrame(cat_encoded_train, columns=cat_encoder.get_feature_names_out(categorical_features)).join(pd.DataFrame(X_train_processed[numerical_features], columns=numerical_features), axis=1).reset_index(drop=True)
X_test_combined = pd.DataFrame(cat_encoded_test, columns=cat_encoder.get_feature_names_out(categorical_features)).join(pd.DataFrame(X_test_processed[numerical_features], columns=numerical_features), axis=1).reset_index(drop=True)

# Prepare datasets for LightGBM
train_dataset = lgb.Dataset(X_train_combined, label=y_train.values)
test_dataset = lgb.Dataset(X_test_combined, label=None, reference=train_dataset)

# Train Model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    'seed': 42,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'n_estimators': 500
}

model = lgb.train(params, train_dataset, num_boost_round=500)

# Predictions
predictions = model.predict(test_dataset.data)

# Create Output DataFrame with required columns: "id" and "Amount in USD"
output_df = pd.DataFrame({
    "id": range(len(predictions)),
    "Amount in USD": predictions
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")