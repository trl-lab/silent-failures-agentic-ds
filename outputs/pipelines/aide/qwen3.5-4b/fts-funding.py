import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [col for col in train_df.columns if col != "amountUSD"]
X_train = train_df[feature_cols]
y_train = train_df["amountUSD"]
X_test = test_df[feature_cols]

# 5-fold cross-validation
X_train = X_train.astype("float32")
y_train = y_train.astype("float32")

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

# Early stopping callback
early_stopping = lgb.early_stopping(50)
log_eval = lgb.log_evaluation(-1)

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_train.iloc[val_idx])],
        early_stopping=early_stopping,
        callbacks=[log_eval],
    )
    val_pred = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_train.iloc[val_idx], val_pred))

# Calculate CV RMSE
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on full training data
model.fit(X_train, y_train, eval_set=[(X_train, y_train)], callbacks=[early_stopping])

# Generate test predictions
test_pred = model.predict(X_test)
submission = pd.DataFrame({"id": range(len(test_df)), "amountUSD": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature columns
feature_cols = [col for col in train_df.columns if col != "amountUSD"]

# Prepare training data
X_train = train_df[feature_cols].astype("float32")
y_train = train_df["amountUSD"].astype("float32")
X_test = test_df[feature_cols].astype("float32")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(train_data, valid_sets=[val_data], callbacks=callbacks)

    val_pred = model.predict(X_fold_val)
    cv_rmse_scores.append(mean_squared_error(y_fold_val, val_pred))

    print(f"Fold {fold+1} CV RMSE: {np.sqrt(cv_rmse_scores[-1]):.4f}")

# Calculate final CV RMSE
cv_rmse = np.sqrt(np.mean(cv_rmse_scores))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on full training data
train_data = lgb.Dataset(X_train, label=y_train)
callbacks = [lgb.early_stopping(100)]

model.fit(train_data, callbacks=callbacks)

# Generate test predictions
test_pred = model.predict(X_test)
submission = pd.DataFrame({"id": range(len(test_df)), "amountUSD": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
