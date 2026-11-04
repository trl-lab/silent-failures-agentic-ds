import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature selection and model training
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
X_test = test_df.drop(columns=["Class"])

# Split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_split, y_train_split)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
X_test = test_df.drop(columns=["Class"])

# Split training data for validation (5-fold CV with train/val split)
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Feature selection using SelectKBest with f_classif
selector = SelectKBest(f_classif, k=15)
X_train_sel = selector.fit_transform(X_train_full, y_train_full)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)

# Train LightGBM model with early stopping
train_data = lgb.Dataset(X_train_sel, label=y_train_full)
valid_data = lgb.Dataset(X_val_sel, label=y_val, reference=train_data)
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val_sel)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Predict on test set
y_test_pred = model.predict(X_test_sel)
y_test_pred = np.round(y_test_pred).astype(int)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_test_pred)), "Class": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
