import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import gini_score
import lightgbm as lgb
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [col for col in train.columns if col not in ["id", "target"]]
X_train = train[feature_cols]
y_train = train["target"]
X_test = test[feature_cols]

# Split into train and validation
from sklearn.model_selection import train_test_split

X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Train with early stopping
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
model.fit(
    X_train_fold,
    y_train_fold,
    eval_set=[(X_val_fold, y_val_fold)],
    callbacks=callbacks,
    verbose=0,
)

# Evaluate on validation set
y_pred = model.predict(X_val_fold, raw_score=True)
normalized_gini = gini_score(y_val_fold, y_pred)
print(f"Normalized Gini Score: {normalized_gini:.4f}")

# Save predictions for test set
y_test_pred = model.predict(X_test, raw_score=True)
submission = pd.DataFrame({"id": test["id"], "target": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import gini_score
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Feature columns (exclude id and target)
feature_cols = [col for col in train.columns if col not in ["id", "target"]]

# Create train/val split for evaluation
X_train = train[feature_cols]
y_train = train["target"]
X_test = test[feature_cols]

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    # LightGBM model parameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "max_depth": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Predict on validation set
    y_pred = model.predict(X_fold_val)

    # Calculate normalized Gini score
    gini = gini_score(y_fold_val, y_pred)
    scores.append(gini)

# Average normalized Gini score
avg_gini = np.mean(scores)
print(f"Average Normalized Gini Score: {avg_gini:.4f}")

# Train final model on full training data
X_train_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    X_train_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Predict on test set
y_test_pred = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test["id"], "target": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
