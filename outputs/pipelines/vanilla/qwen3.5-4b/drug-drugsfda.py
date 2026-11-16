import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Set up file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Define target
TARGET = 'marketing_status'
y_train = df_train[TARGET].values

# Prepare features
df_train_feats = df_train.drop(columns=[TARGET])
df_test_feats = df_test.copy()

# Separate feature columns into numerical and categorical for transformation
feature_columns = list(df_train_feats.columns)
numerical_features = []
categorical_features = []

for col in feature_columns:
    if pd.api.types.is_numeric_dtype(df_train_feats[col]):
        numerical_features.append(col)
    elif pd.api.types.is_bool_dtype(df_train_feats[col]):
        numerical_features.append(col)
    else:
        categorical_features.append(col)

# Define transformers
imputer_num = SimpleImputer(strategy='median')
encoder_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
transformer_num = imputer_num

column_transformer = ColumnTransformer(
    transformers=[
        ('num', transformer_num, numerical_features),
        ('cat', encoder_cat, categorical_features)
    ]

# Initialize model
# Target is specified as binary classification task for ROC_AUC
model = LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=None,
    random_state=42,
    silent=1,
    force_col_wise=True,
    verbose_eval=-1,
    num_threads=1
)

# Build and fit pipeline
pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('classifier', model)
])

pipeline.fit(df_train_feats, y_train)

# Make predictions on test set
y_test_probs = pipeline.named_steps['classifier'].predict_proba(df_test_feats)

# Convert probabilities to class labels for submission
# Since task is binary classification (b-classification)
y_pred_labels = y_test_probs[:, 1]  # Assuming first index is negative class, second is positive

# Generate output dataframe
output_data = pd.DataFrame({
    'id': range(len(df_test)),
    'marketing_status': y_pred_labels.astype(int)
})

# Save to csv
output_data.to_csv(OUTPUT_PATH, index=False)