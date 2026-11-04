import pandas as pd
import numpy as np
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# File paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/tmdb-box-office-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/tmdb-box-office-prediction/data/test.parquet'

# Load Data
print("Loading data...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Function to handle target transformation
def prepare_for_model(df, train=True):
    df = df.copy()
    
    # Identify target column
    target_col = 'revenue'
    features = [col for col in df.columns if col != target_col]
    df = df[features]
    
    # Handle missing values (Mean/Median fill)
    # CatBoost handles missing values natively, but filling improves stability
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                # For object/categorical, CatBoost handles nulls, but filling mode helps
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '')
    
    return df

# Prepare Training Data
train_processed = prepare_for_model(train_df)
target_y = np.log1p(train_processed[target_col]) # Use log1p for numerical stability
# Drop redundant empty columns if any were created
final_features = train_processed.drop(columns=[train_processed.columns[df.columns.index('revenue')]])

X_train = final_features.values
y_train = target_y

# Prepare Test Data
test_processed = prepare_for_model(test_df)
test_X = test_processed.values

# Train Model
model = cb.CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    early_stopping_rounds=100,
    verbose=False,
    random_state=42
)

model.fit(X_train, y_train)

# Make Predictions
test_predictions = model.predict(test_X)

# Transform back to original scale
final_predictions = np.exp(test_predictions)

# Create Output DataFrame
output_id = list(range(len(test_df)))
results = pd.DataFrame({
    'id': output_id,
    'revenue': final_predictions
})

# Save to CSV
results.to_csv('predictions.csv', index=False)
print(f"Predictions saved to predictions.csv.")