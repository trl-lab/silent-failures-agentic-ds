import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature matrix and target
X_train = train_df.drop(columns=["New Commitments (US$)"])
y_train = train_df["New Commitments (US$)"]

# Identify numeric columns
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Split data with validation set first
X_train_split, X_val_split, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Create preprocessing pipeline
preprocessor_train = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
)

# Use same preprocessor for validation to avoid double fitting
preprocessor_val = None

# Fit and transform
X_train_processed = preprocessor_train.fit_transform(X_train_split)
X_val_processed = preprocessor_train.transform(X_val_split)

# Define LightGBM model architecture
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    force_col_wise=True,
)

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]

# Train the model
model.fit(
    X_train_processed, y_train,
    eval_set=[(X_val_processed, y_val)],
    callbacks=callbacks
)

# Generate validation predictions
y_val_pred = model.predict(X_val_processed)

# Calculate validation metric using official RMSE
score = mean_squared_error(y_val, y_val_pred, squared=False)

# Generate test predictions
X_test_processed = preprocessor_train.transform(test_df)
y_test_pred = model.predict(X_test_processed)

# Create submission file
submission_df = pd.DataFrame({
    "id": range(len(test_df)),
    "New Commitments (US$)": y_test_pred
})

# Save to correct location
submission_df.to_csv("./submission/submission.csv", index=False)

# Print validation metric as the last line
print(f'Final Validation Score: {score}')