import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Prepare feature columns (exclude id and target)
feature_cols = [col for col in train.columns if col not in ["id", "target"]]

# Create train/validation split for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_auc = 0
best_model = None

# LightGBM parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train = train["target"].iloc[train_idx]
    y_val = train["target"].iloc[val_idx]

    # Encode features
    le = LabelEncoder()
    for col in feature_cols:
        if train[col].dtype == "object":
            if train[col].nunique() > 10:
                # High cardinality - use label encoding
                X_train[col] = le.fit_transform(train[col])
                X_val[col] = le.transform(X_val[col])
            else:
                X_train[col] = le.fit_transform(train[col])
                X_val[col] = le.transform(X_val[col])

    # Convert to numeric
    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")

    # LightGBM training
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        verbose_eval=False,
    )

    # Predict on validation set
    y_pred = model.predict(X_val)
    fold_auc = roc_auc_score(y_val, y_pred)
    print(f"Fold {fold+1} ROC-AUC: {fold_auc:.4f}")

    if fold_auc > best_auc:
        best_auc = fold_auc
        best_model = model

# Final evaluation on validation fold
if best_model is not None:
    # Get final predictions
    test_features = test[feature_cols]
    test_features = test_features.astype("float32")

    test_pred = best_model.predict(test_features)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    # Save submission
    submission = pd.DataFrame({"id": test["id"], "target": test_pred})
    submission.to_csv("./working/submission.csv", index=False)
    print(f"\nBest validation ROC-AUC: {best_auc:.4f}")
    print("Submission saved to ./working/submission.csv")
else:
    print("No model trained successfully")
