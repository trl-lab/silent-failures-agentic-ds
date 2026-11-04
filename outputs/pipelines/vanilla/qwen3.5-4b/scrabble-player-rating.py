import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/scrabble-player-rating/data/train.parquet')
test = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/scrabble-player-rating/data/test.parquet')

# Prepare feature matrices and labels
X_train = train.drop(columns=['id', 'player_id', 'target'])  # assuming id/target not needed, adjust if different
y_train = train['rating'].values

test_df = test.copy()
X_test = test_df.drop(columns=['id', 'player_id'])  # adjust if needed

# Identify numeric and categorical columns
all_cols = list(X_train.columns) + list(X_test.columns)
# Get column types
col_types = {}
for col in all_cols:
    try:
        col_types[col] = X_train[col].dtype.name == 'object'
    except KeyError:
        pass

numeric_cols = [col for col in all_cols if col_types.get(col, False) == False and col != 'target']
categorical_cols = [col for col in all_cols if col_types.get(col, False) == True]

if not numeric_cols:
    raise ValueError("No numeric columns found")

# Create preprocessor
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ))
])

model.fit(train, y_train)

predictions = model.predict(test)
test_preds = pd.DataFrame({
    'id': range(len(test)),
    'rating': predictions.round(2).astype(float)
})

test_preds.to_csv('predictions.csv', index=False)
print(f"Saved {len(predictions)} predictions to predictions.csv")
print(f"Sample prediction shape: {test_preds.head()}")