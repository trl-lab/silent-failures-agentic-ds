import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# 2. Feature Engineering
def preprocess_data(df):
    df = df.copy()

    # Extract title from name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Cabin deck
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    # Fare and Age transformations/imputations
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FareLog"] = np.log1p(df["Fare"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # Categorical encoding
    cat_cols = ["Sex", "Embarked", "Title", "Deck"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "FareLog",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "Deck",
    ]
    return df[features]


y_train = train_df["Survived"].values
X_train = preprocess_data(train_df)
X_test = preprocess_data(test_df)

# Align categorical dtypes
for col in X_train.select_dtypes(include="category").columns:
    categories = list(
        set(X_train[col].dropna().unique()).union(set(X_test[col].dropna().unique()))
    )
    X_train[col] = pd.Categorical(X_train[col], categories=categories)
    X_test[col] = pd.Categorical(X_test[col], categories=categories)

# 3. Model Training & 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

oof_binary = (oof_preds >= 0.5).astype(int)
cv_accuracy = accuracy_score(y_train, oof_binary)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 4. Save Predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Survived": (test_preds >= 0.5).astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
