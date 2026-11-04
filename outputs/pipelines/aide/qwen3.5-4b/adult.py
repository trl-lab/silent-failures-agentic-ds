import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", SimpleImputer(strategy="median"), numerical_features),
    ]
)

# Define LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Create training pipeline
train_pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

# Encode target
target_col = "income"
y_train = (train_df[target_col] == ">50K").astype(int)
X_train = train_df[categorical_features + numerical_features]

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_pipeline.fit(X_train_fold, y_train_fold)
    val_pred = train_pipeline.predict(X_val_fold)
    scores.append(accuracy_score(y_val_fold, val_pred))
    print(f"Fold {fold + 1} Accuracy: {scores[-1]:.4f}")

# Calculate mean accuracy
mean_accuracy = np.mean(scores)
print(f"\nMean CV Accuracy: {mean_accuracy:.4f}")

# Prepare test predictions
test_features = test_df[categorical_features + numerical_features]
test_pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
test_pipeline.fit(X_train, y_train)
test_predictions = test_pipeline.predict(test_features)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "income": test_predictions.astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
