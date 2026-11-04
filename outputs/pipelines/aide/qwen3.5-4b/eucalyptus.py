import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature columns and target
feature_cols = [col for col in train_df.columns if col != "Utility"]
target_col = "Utility"

# Create preprocessing pipeline
preprocessor = Pipeline(
    [("categorical", LabelEncoder()), ("numerical", SimpleImputer(strategy="median"))]
)

# Separate features and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Encode categorical features
for col in feature_cols:
    if train_df[col].dtype == "object" or train_df[col].dtype == "category":
        le = LabelEncoder()
        X_train[col] = le.fit_transform(train_df[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    else:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation for evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Train final model on full training data
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_pred)), "Utility": y_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions shape: {submission.shape}")
