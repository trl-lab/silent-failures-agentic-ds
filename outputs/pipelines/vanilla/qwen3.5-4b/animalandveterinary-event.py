import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# File paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/animalandveterinary-event/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/animalandveterinary-event/data/test.parquet"
output_path = "predictions.csv"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target
target_col = 'serious_ae'

# Extract features
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.copy()

# Identify numerical and categorical columns based on training data
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [col for col in X_train.columns if col not in numeric_features]

# Build Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ]), categorical_features)
    ],
    remainder='passthrough' # Keeps any additional columns not explicitly handled
)

# Initialize LightGBM Classifier
clf = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=1000, 
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8
)

# Create full Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', clf)
])

# Fit the model on training data
model_pipeline.fit(X_train, y_train)

# Transform test data
X_test_processed = model_pipeline.named_steps['preprocessor'].transform(X_test)
proba_scores = model_pipeline.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]

# Create output dataframe
df_predictions = pd.DataFrame({
    'id': list(range(len(X_test))),
    'serious_ae': proba_scores
})

# Save predictions
df_predictions.to_csv(output_path, index=False)