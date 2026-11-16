import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns
feature_cols = [
    "A_follower_count",
    "A_following_count",
    "A_listed_count",
    "A_mentions_received",
    "A_mentions_sent",
    "A_network_feature_1",
    "A_network_feature_2",
    "A_network_feature_3",
    "A_posts",
    "A_retweets_received",
    "A_retweets_sent",
    "B_follower_count",
    "B_following_count",
    "B_listed_count",
    "B_mentions_received",
    "B_mentions_sent",
    "B_network_feature_1",
    "B_network_feature_2",
    "B_network_feature_3",
    "B_posts",
    "B_retweets_received",
    "B_retweets_sent",
]

# Combine A and B features
train_features = train_df[feature_cols].copy()
train_labels = train_df["Choice"].copy()

# Handle missing values
train_features = train_features.fillna(0)

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_acc = 0
best_params = None
best_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, train_labels)):
    X_train_fold = train_features.iloc[train_idx]
    y_train_fold = train_labels.iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_val_fold = train_labels.iloc[val_idx]

    # Train LightGBM model
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

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
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    val_pred = model.predict(X_val_fold)
    val_pred = (val_pred >= 0.5).astype(int)

    fold_acc = accuracy_score(y_val_fold, val_pred)
    print(f"Fold {fold+1} validation accuracy: {fold_acc:.4f}")

    if fold_acc > best_acc:
        best_acc = fold_acc
        best_params = params.copy()
        best_model = model

# Final model training on full training set (without early stopping since we've selected the best model)
X_train_full = train_features
y_train_full = train_labels
final_model = lgb.train(
    best_params,
    lgb.Dataset(X_train_full, label=y_train_full),
    num_boost_round=1000,
    callbacks=[lgb.log_evaluation(-1)],
)

# Get final accuracy on training set
train_pred = final_model.predict(X_train_full)
train_pred = (train_pred >= 0.5).astype(int)
final_train_acc = accuracy_score(y_train_full, train_pred)
print(f"\nFinal training set accuracy: {final_train_acc:.4f}")

# Load test data and make predictions
test_features = test_df[feature_cols].copy()
test_features = test_features.fillna(0)

test_pred = final_model.predict(test_features)
test_pred = (test_pred >= 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Choice": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
