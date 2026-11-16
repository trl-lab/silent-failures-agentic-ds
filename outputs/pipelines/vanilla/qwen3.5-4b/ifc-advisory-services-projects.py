import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

# Paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/test.parquet"
OUTPUT_PATH = "predictions.csv"
TARGET_COLUMN = "Estimated Total Budget ($)"

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define target variable
y_train = train_df[TARGET_COLUMN]
X_train_full = train_df.drop(columns=[TARGET_COLUMN])
X_test_full = test_df.copy()

# Prepare features and detect types
features = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in X_train_full.columns if col not in features and col != TARGET_COLUMN]

# Combine all candidate feature columns
all_columns = features + categorical_cols
if len(all_columns) > 0:
    Xt = X_train_full[all_columns].copy()
else:
    Xt = pd.DataFrame(index=X_train_full.index)

# Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ]), features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical_cols)
    ], remainder='drop'
)

# Create Model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42,
    verbose=-1,
    force_col_wise=True
)

# Fit Pipeline
preprocessor_model = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

X_train_processed = preprocessor.fit_transform(X_train_full)
y_train_array = y_train.values.reshape(-1, 1)
preprocessor_model.fit(X_train_processed, y_train_array)

# Transform Test Data
X_test_processed = preprocessor.transform(X_test_full)
predictions = preprocessor_model.predict(X_test_processed)

# Prepare Output DataFrame
ids = list(range(len(test_df)))
results_df = pd.DataFrame({
    'id': ids,
    TARGET_COLUMN: predictions
})

# Ensure columns are in correct order
results_df = results_df[[TARGET_COLUMN, 'id']] # Reorder to be consistent if needed, but spec says id then column
# Actually spec says: two columns: "id" ... and "Estimated Total Budget ($)". Order usually implies index.
# Let's ensure the order matches spec: "id" first, then target.
results_df = results_df[['id', TARGET_COLUMN]]

# Save
results_df.to_csv(OUTPUT_PATH, index=False)