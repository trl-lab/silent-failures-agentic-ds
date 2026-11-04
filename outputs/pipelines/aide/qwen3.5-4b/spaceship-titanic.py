import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create submission directory if not exists
import os

os.makedirs("./working", exist_ok=True)


# Preprocessing function
def preprocess(df, target_col="Transported"):
    # Handle missing values
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical columns
    cat_cols = [
        "Cabin",
        "CryoSleep",
        "Destination",
        "HomePlanet",
        "Name",
        "PassengerId",
        "VIP",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Create feature matrix
    feature_cols = ["Age", "FoodCourt", "RoomService", "ShoppingMall", "Spa", "VRDeck"]
    X = df[feature_cols].values
    y = df[target_col].values

    return X, y


# Preprocess train and test data
X_train, y_train = preprocess(train_df)
X_test, y_test = preprocess(test_df)

# Split training data for validation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# Callbacks for early stopping
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1, verbose_eval=False)]

# Train model
model.fit(X_train_full, y_train_full, eval_set=[(X_val, y_val)], callbacks=callbacks)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {accuracy:.4f}")

# Make predictions on test set
test_predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "Transported": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
