import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Ensure 'id' column exists in test_df
if 'id' not in test_df.columns:
    test_df['id'] = range(len(test_df))

# Define target column
target_col = "rating"

# Separate features and target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])

# Identify column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
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
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="MISSING"),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

# Feature engineering
# Create game progression indicator (normalized game_id)
train_df["game_progression"] = train_df["game_id"] / 72773.0
test_df["game_progression"] = test_df["game_id"] / 72773.0

# Create score-based features
train_df["score_squared"] = train_df["score"] ** 2
train_df["score_normalized"] = train_df["score"] / 1132.0
test_df["score_squared"] = test_df["score"] ** 2
test_df["score_normalized"] = test_df["score"] / 1132.0

# Create interaction features
train_df["game_progression_x_score"] = train_df["game_progression"] * train_df["score"]
train_df["game_progression_x_score_squared"] = (
    train_df["game_progression"] * train_df["score"] ** 2
)
test_df["game_progression_x_score"] = test_df["game_progression"] * test_df["score"]
test_df["game_progression_x_score_squared"] = (
    test_df["game_progression"] * test_df["score"] ** 2
)

# Prepare features
feature_columns = [
    "game_id",
    "nickname",
    "score",
    "score_squared",
    "score_normalized",
    "game_progression",
    "game_progression_x_score",
    "game_progression_x_score_squared",
]

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
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
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="MISSING"),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

# Feature engineering
# Create game progression indicator (normalized game_id)
train_df["game_progression"] = train_df["game_id"] / 72773.0
test_df["game_progression"] = test_df["game_id"] / 72773.0

# Create score-based features
train_df["score_squared"] = train_df["score"] ** 2
train_df["score_normalized"] = train_df["score"] / 1132.0
test_df["score_squared"] = test_df["score"] ** 2
test_df["score_normalized"] = test_df["score"] / 1132.0

# Create interaction features
train_df["game_progression_x_score"] = train_df["game_progression"] * train_df["score"]
train_df["game_progression_x_score_squared"] = (
    train_df["game_progression"] * train_df["score"] ** 2
)
test_df["game_progression_x_score"] = test_df["game_progression"] * test_df["score"]
test_df["game_progression_x_score_squared"] = (
    test_df["game_progression"] * test_df["score"] ** 2
)

# Prepare features
feature_columns = [
    "game_id",
    "nickname",
    "score",
    "score_squared",
    "score_normalized",
    "game_progression",
    "game_progression_x_score",
    "game_progression_x_score_squared",
]

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

# Split data for validation FIRST
X_train_split, X_val_split, y_train, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

# Apply preprocessing only to training split
X_train_processed = preprocessor.fit_transform(X_train_split)
X_val_processed = preprocessor.transform(X_val_split)

# Create LightGBM model with custom callbacks
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

# Train with early stopping
model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    callbacks=[lgb.early_stopping(50)],
)

# Predict on validation set
y_val_pred = model.predict(X_val_processed)

# Calculate validation MAE
val_mae = mean_absolute_error(y_val, y_val_pred)

# Print validation score (required for evaluation)
print(f"Final Validation Score: {val_mae:.4f}")

# Predict on test set
y_test_pred = model.predict(preprocessor.transform(X_test))

# Save predictions to submission file
submission_df = pd.DataFrame({"id": test_df["id"].values, "rating": y_test_pred})
submission_df.to_csv("./submission/submission.csv", index=False)