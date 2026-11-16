import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load data
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

# Identify target and features
target_col = 'Adjusted MLR'
y = train[target_col].values
feature_cols = [c for c in train.columns if c != target_col]
X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Define column types
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

# --- Validation Phase ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
])
pipe.fit(X_train, y_train)
val_preds = pipe.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Final Validation Score: {rmse:.4f}")

# --- Final Model: refit on all training data ---
final_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
])
final_pipe.fit(X, y)

# Predict on test
test_preds = final_pipe.predict(X_test)

# Write submission
submission = pd.DataFrame({'id': range(len(test)), 'Adjusted MLR': test_preds})
submission.to_csv('./submission/submission.csv', index=False)