import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/test.parquet'
TARGET_COL = "Gross Issued Amount (US$)"
OUTPUT_FILE = 'predictions.csv'

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Handle Data Consistency
def identify_target(row):
    col_vals = row
    return TARGET_COL in row.columns or TARGET_COL in col_vals.values

# Verify Target Column Exists
assert TARGET_COL in train_df.columns, f"Target column {TARGET_COL} not found in train."
assert TARGET_COL not in test_df.columns, f"Target column {TARGET_COL} found in test. Dropping for prediction only."

# Prepare Features
feature_cols = [col for col in train_df.columns if col != TARGET_COL]
test_feature_cols = [col for col in test_df.columns if col in feature_cols]

# Create Pipeline for Preprocessing
ct = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('encoder', None)
        ])),
    ],
    remapping={},
)

# Robust approach: Separate categorical and numeric columns for automatic handling
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in feature_cols if col not in numerical_cols]

# Define transformers
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ]
)

# Fit and Predict
# We use LightGBM for speed and efficiency on tabular data as requested options
from lightgbm import LGBMRegressor
model = LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, verbose=-1, max_depth=-1)

# Train
X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]

# Apply Preprocessing
pipe = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder=None
)
# Reuse transformers from pipeline construction
pipe._final_estimator = model

X_train_proc = pipe.fit(X_train).transform(X_train)
y_train_arr = y_train.values

# Store models and transformers
# Actually, we should fit the full pipeline including preprocessing
pipe_num = Pipeline([('imputer', SimpleImputer(strategy='median'))])
pipe_cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
full_pipe = ColumnTransformer([('num', pipe_num, numerical_cols),
                               ('cat', pipe_cat, categorical_cols)])

model.fit(full_pipe.transform(X_train), y_train)

# Process Test Data
X_test_proc = full_pipe.transform(test_df[test_feature_cols])

# Predict
y_pred = model.predict(X_test_proc)

# Prepare Output
# Test df reset index to provide 'id' starting from 0
test_id_df = test_df.reset_index(drop=True)
predictions_df = pd.DataFrame({
    'id': range(len(test_id_df)),
    TARGET_COL: y_pred
})

# Save
predictions_df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")