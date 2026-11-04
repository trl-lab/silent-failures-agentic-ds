import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/test.parquet')

# Define feature columns
numeric_cols = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol',
                'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices',
                'Malty', 'number_of_reviews']

categorical_cols = ['Style', 'Brewery']

# Drop high-cardinality text columns that won't help much
drop_cols = ['Name', 'Beer Name (Full)', 'Description']

# Fill missing values
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(train_df[col].median())

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('Unknown')
    test_df[col] = test_df[col].fillna('Unknown')

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Prepare features
feature_cols = numeric_cols + categorical_cols
X = train_df[feature_cols]
y = train_df['review_overall']
X_test = test_df[feature_cols]

# Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'LGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7,
                          num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1),
    'XGB': XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, eval_metric='rmse'),
    'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=7,
                                  random_state=42, verbose=0)
}

val_predictions = []
test_predictions = []

for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"{name} Validation RMSE: {val_rmse:.6f}")
    val_predictions.append(val_pred)
    test_predictions.append(model.predict(X_test))

# Ensemble: average predictions
final_test_pred = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_test_pred)),
    'review_overall': final_test_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")