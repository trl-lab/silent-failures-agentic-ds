import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
feature_cols = [
    "age",
    "gender",
    "education",
    "country",
    "ethnicity",
    "nscore",
    "escore",
    "oscore",
    "ascore",
    "cscore",
    "impuslive",
    "ss",
]
target_col = "cannabis"

# Prepare train data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
train_ids = train_df["id"].values

# Prepare test data
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values

# Use 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = []
val_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    # Create LightGBM data
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_cols, reference=train_data
    )

    # Train model with early stopping
    params = {
        "objective": "multiclass",
        "num_class": len(pd.unique(y_train)),
        "metric": "accuracy",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get validation predictions
    val_pred = model.predict(X_val)
    val_preds.extend(val_pred)
    oof_preds.extend(model.predict(X_train))

# Calculate validation accuracy
val_accuracy = accuracy_score(y_train[val_idx], val_preds)
print(f"5-Fold Cross-Validation Accuracy: {val_accuracy:.4f}")

# Make predictions on test data
test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "cannabis": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
