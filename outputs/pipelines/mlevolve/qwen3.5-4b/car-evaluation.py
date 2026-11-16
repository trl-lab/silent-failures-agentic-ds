import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Check actual column names
print(f"Train columns: {train_df.columns.tolist()}")

# Preprocessing
print("Preprocessing data...")
# Find target column (should be 'car' based on task description)
target_col = "car" if "car" in train_df.columns else "class"
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df

# Encode categorical features with ordinal encoding for safety/maint
# Ordinal encoding: unacceptable=0, acceptable=1, good=2, very_good=3
ordinal_mapping = {
    "unacceptable": 0,
    "acceptable": 1,
    "good": 2,
    "very_good": 3
}

# Create mapping dictionaries for all columns
def create_ordinal_mapping(df):
    mappings = {}
    for col in df.columns:
        unique_vals = df[col].unique()
        if pd.api.types.is_numeric_dtype(df[col]):
            mappings[col] = None
        else:
            # Create ordinal mapping for categorical columns
            mappings[col] = {v: i for i, v in enumerate(unique_vals)}
    return mappings

ordinal_maps = create_ordinal_mapping(X_train)

for col, mapping in ordinal_maps.items():
    if mapping is not None:
        # Apply ordinal encoding
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

# Create interaction features
print("Creating interaction features...")
interaction_features = [
    'buying_doors',
    'maint_persons',
    'safety_lug_boot'
]

for feat in interaction_features:
    # Create interaction feature
    X_train[feat] = X_train['buying'] * X_train['doors'] if feat == 'buying_doors' else None
    X_test[feat] = X_test['buying'] * X_test['doors'] if feat == 'buying_doors' else None

    if feat == 'maint_persons':
        X_train[feat] = X_train['maint'] * X_train['persons']
        X_test[feat] = X_test['maint'] * X_test['persons']
    elif feat == 'safety_lug_boot':
        X_train[feat] = X_train['safety'] * X_train['lug_boot']
        X_test[feat] = X_test['safety'] * X_test['lug_boot']

# Create polynomial features for numeric columns
print("Creating polynomial features...")
X_train['doors_sq'] = X_train['doors'] ** 2
X_train['lug_boot_sq'] = X_train['lug_boot'] ** 2
X_test['doors_sq'] = X_test['doors'] ** 2
X_test['lug_boot_sq'] = X_test['lug_boot'] ** 2

# Split for validation
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
)

# Train model
print("Training model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train_val, y_train_val)

# Early stopping
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [200],
    'learning_rate': [0.1],
    'max_depth': [5]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_val, y_train_val)
best_model = grid_search.best_estimator_

# Validation
print("Evaluating validation...")
y_pred_val = best_model.predict(X_test_val)
val_score = accuracy_score(y_test_val, y_pred_val)
print(f"Final Validation Score: {val_score}")

# Test predictions
print("Generating test predictions...")
y_pred_test = best_model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "id": range(len(y_pred_test)),
    "car": y_pred_test.astype(str)
})

# Save submission
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
