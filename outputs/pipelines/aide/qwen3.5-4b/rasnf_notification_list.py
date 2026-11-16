import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create submission file
train_df["id"] = train_df.index
test_df["id"] = test_df.index

# Feature engineering
target_col = "num_countries_affected"
feature_cols = [col for col in train_df.columns if col != target_col and col != "id"]

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    cv_scores.append(roc_auc_score(y_val_fold, y_pred))

# Calculate mean CV score
cv_mean = np.mean(cv_scores)
print(f"5-fold CV ROC-AUC: {cv_mean:.4f}")

# Train final model on all training data
model.fit(X_train, y_train, verbose=-1)

# Predict on test set
y_pred_test = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_ids, "num_countries_affected": y_pred_test})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create submission file
train_df["id"] = train_df.index
test_df["id"] = test_df.index

# Feature engineering
target_col = "num_countries_affected"
feature_cols = [col for col in train_df.columns if col != target_col and col != "id"]

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    cv_scores.append(roc_auc_score(y_val_fold, y_pred))

# Calculate mean CV score
cv_mean = np.mean(cv_scores)
print(f"5-fold CV ROC-AUC: {cv_mean:.4f}")

# Train final model on all training data
model.fit(X_train, y_train, verbose=-1)

# Predict on test set
y_pred_test = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_ids, "num_countries_affected": y_pred_test})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
