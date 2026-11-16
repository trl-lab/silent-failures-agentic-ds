import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Combine for preprocessing
combined = pd.concat([train_df, test_df], ignore_index=True)

# Define preprocessing
numeric_features = ["Age", "Fare", "Parch", "SibSp", "Pclass"]
categorical_features = ["Sex", "Embarked", "Cabin", "Ticket", "Name"]

# Fill missing Age with median
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(train_df["Age"].median())

# Create FamilySize feature
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Encode categorical features
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_cabin = LabelEncoder()
le_ticket = LabelEncoder()

train_df["Sex_encoded"] = le_sex.fit_transform(train_df["Sex"])
train_df["Embarked_encoded"] = le_embarked.fit_transform(
    train_df["Embarked"].fillna("S")
)
train_df["Cabin_encoded"] = le_cabin.fit_transform(train_df["Cabin"])
train_df["Ticket_encoded"] = le_ticket.fit_transform(train_df["Ticket"])

test_df["Sex_encoded"] = le_sex.transform(test_df["Sex"])
test_df["Embarked_encoded"] = le_embarked.transform(test_df["Embarked"].fillna("S"))
test_df["Cabin_encoded"] = le_cabin.transform(test_df["Cabin"])
test_df["Ticket_encoded"] = le_ticket.transform(test_df["Ticket"])

# Create feature columns
feature_cols = [
    "Pclass",
    "Sex_encoded",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "Embarked_encoded",
    "Cabin_encoded",
    "Ticket_encoded",
]

# Prepare training data
X_train = combined[feature_cols]
y_train = train_df["Survived"]
X_test = test_df[feature_cols]

# LightGBM parameters
params = {
    "objective": "binary",
    "metric": "accuracy",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_model = None
best_params = params.copy()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    pred_val = model.predict(X_val_fold)
    fold_score = accuracy_score(y_val_fold, (pred_val >= 0.5).astype(int))

    print(f"Fold {fold+1} CV Accuracy: {fold_score:.4f}")

    if fold_score > best_score:
        best_score = fold_score
        best_model = model

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions on test set
test_pred = final_model.predict(X_test)
test_pred_binary = (test_pred >= 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({"id": test_df["PassengerId"], "Survived": test_pred_binary})
submission.to_csv("./working/submission.csv", index=False)
print(f"\nFinal CV Accuracy: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")
