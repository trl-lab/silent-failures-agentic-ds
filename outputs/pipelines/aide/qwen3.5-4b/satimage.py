import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Features and target
feature_cols = [col for col in train_df.columns if col != "class"]
target_col = "class"

# Create feature matrix
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5-fold cross-validation evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(), X_train_scaled, y_train, cv=kf, scoring="accuracy"
)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model
model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1
)
model.fit(X_train_scaled, y_train)

# Predict on test set
predictions = model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({"id": range(len(predictions)), "class": predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
feature_cols = [col for col in train_df.columns if col != "class"]
target_col = "class"

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values

# Scale features to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use 5-fold cross-validation for evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=42,
    ),
    X_train_scaled,
    y_train,
    cv=cv,
    scoring="accuracy",
)
print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model on full training data
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# Predict on test set
predictions = model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({"id": range(len(predictions)), "class": predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
