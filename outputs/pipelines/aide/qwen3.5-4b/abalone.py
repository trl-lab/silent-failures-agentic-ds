import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature types
numeric_features = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]
categorical_features = ["Sex"]
target = "Class_number_of_rings"

# Split data for validation
X_train_full = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
)

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
        ("num", "passthrough", numeric_features),
    ]
)

# Create pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_estimators=500,
                early_stopping_rounds=50,
                verbose_eval=False,
            ),
        ),
    ]
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = pipeline.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse_val:.4f}")

# Prepare predictions for test set
test_predictions = pipeline.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "Class_number_of_rings": test_predictions}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
