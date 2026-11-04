import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create target variable
train_df["target"] = train_df["revenue"]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop(columns=["revenue"]),
    train_df["revenue"],
    test_size=0.2,
    random_state=42,
)

# Preprocess data
# Handle categorical columns


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
target_col = "revenue"
feature_cols = [col for col in train_df.columns if col not in ["revenue", "id"]]

# Separate ID column for submission
train_ids = train_df["id"].values
test_ids = test_df["id"].values

# Create validation split to avoid data leakage
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols],
    train_df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=None,
)

# Identify categorical and numerical columns
categorical_cols = (
    train_df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
)
numerical_cols = (
    train_df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
)

# Preprocess categorical and numerical columns
cat_processor = Pipeline(
    steps=[
        ("encoder", LabelEncoder()),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ]
)

num_processor = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_processor, categorical_cols),
        ("num", num_processor, numerical_cols),
    ]
)

# Train LightGBM model
cat_features = categorical_cols
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 10,
    "learning_rate": 0.1,
    "verbose": -1,
    "seed": 42,
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=callbacks,
    verbose_eval=-1,
)

# Evaluate on validation set
y_pred_val = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {val_rmse:.2f}")

# Transform test data
X_test = test_df[feature_cols]
X_test_processed = preprocessor.fit_transform(X_train, X_val)

# Predict on test data
test_predictions = model.predict(X_test_processed)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "revenue": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions shape: {submission.shape}")
