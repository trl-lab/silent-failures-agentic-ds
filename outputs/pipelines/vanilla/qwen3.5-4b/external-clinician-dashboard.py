import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/test.parquet')

# Define target and feature columns
target_col = 'Score'
all_cols = train_df.columns.tolist()
feature_cols = [col for col in all_cols if col != target_col]

# Separate columns into numeric and categorical
numeric_data = train_df[feature_cols].select_dtypes(include=[np.number])
categorical_data = train_df[feature_cols].select_dtypes(include=['object'])
# Ensure columns are selected correctly even if some columns were excluded by selection
numeric_col_names = list(numeric_data.columns)
categorical_col_names = list(categorical_data.columns)

# Ensure uniqueness (e.g. if some columns dropped)
numeric_cols = [col for col in feature_cols if col in numeric_data.columns]
cat_cols = [col for col in feature_cols if col in categorical_data.columns]

# Initialize components
numeric_imputer = SimpleImputer(strategy='median')
numeric_scaler = StandardScaler()
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Define Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_imp', numeric_imputer, numeric_cols),
        ('num_scaler', numeric_scaler, numeric_cols),
        ('cat_enc', cat_encoder, cat_cols)
    ],
    remainder='drop'
)

# Define Model
model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    num_leaves=31,
    learning_rate=0.05,
    max_depth=-1,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose_eval=False
)

# Define Full Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Fit model
pipeline.fit(train_df[feature_cols], train_df[target_col])

# Transform test data and predict
X_test = test_df[feature_cols]
y_pred = pipeline.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'Score': y_pred
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)