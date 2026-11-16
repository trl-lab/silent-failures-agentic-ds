import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id)
feature_cols = [col for col in train.columns if col not in ["target", "id"]]

# Create LightGBM model with early stopping
model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    train_data = train.iloc[train_idx]
    val_data = train.iloc[val_idx]

    # Prepare data for LightGBM
    train_data = train_data[feature_cols + ["target"]].copy()
    val_data = val_data[feature_cols + ["target"]].copy()

    train_dataset = lgb.Dataset(train_data, label=train_data["target"])
    val_dataset = lgb.Dataset(
        val_data, label=val_data["target"], reference=train_dataset
    )

    # Train with callbacks
    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(0, verbose=False),
    ]

    model.fit(
        train_data[feature_cols],
        train_data["target"],
        feature_names=feature_cols,
        callbacks=callbacks,
        eval_set=[val_data[feature_cols], val_data["target"]],
        eval_name="val",
    )

    # Calculate ROC AUC on validation fold
    val_pred = model.predict(val_data[feature_cols])
    from sklearn.metrics import roc_auc_score

    cv_scores.append(roc_auc_score(val_data["target"], val_pred))

# Calculate mean CV score
mean_cv_score = np.mean(cv_scores)
print(f"5-Fold CV ROC AUC: {mean_cv_score:.4f}")

# Train final model on all training data
train_data = train[feature_cols]
train_target = train["target"]
train_dataset = lgb.Dataset(train_data, label=train_target)

model.fit(
    train_data[feature_cols],
    train_target,
    feature_names=feature_cols,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Generate predictions for test set
test_data = test[feature_cols]
test_pred = model.predict(test_data[feature_cols])

# Create submission file
submission = pd.DataFrame({"id": test["id"], "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
