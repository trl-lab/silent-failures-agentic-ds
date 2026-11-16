import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
y = train_df["TARGET"]
X_train = train_df.drop("TARGET", axis=1)
X_test = test_df.drop("ID", axis=1)

# Identify feature types
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", "passthrough", categorical_features),
    ]
)

# Create LightGBM model
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "model",
            lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
)

# 5-fold cross-validation for validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y, cv=kf, scoring="roc_auc")
print(f"5-Fold CV ROC AUC: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")

# Train final model on all training data
model.fit(X_train, y)

# Evaluate on validation set (first 20% of training data)
val_split = int(0.8 * len(X_train))
X_val = X_train.iloc[:val_split]
y_val = y.iloc[:val_split]

val_predictions = model.predict(X_val)
val_auc = roc_auc_score(y_val, val_predictions)
print(f"Validation Set ROC AUC: {val_auc:.6f}")

# Make predictions on test set
test_predictions = model.predict(X_test)

# Save submission file
submission = pd.DataFrame({"id": X_test.index, "TARGET": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
