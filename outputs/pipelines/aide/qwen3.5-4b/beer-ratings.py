import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
features = [col for col in train_df.columns if col != "review_overall"]
target = "review_overall"

# Create train/validation split for evaluation
X_train = train_df[features]
y_train = train_df[target]

# Initialize LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Split into 5 folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get RMSE on validation set
cv_rmse = []
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
        callbacks=callbacks,
        valid_sets=[val_data],
    )

    y_pred = model.predict(X_val_fold)
    cv_rmse.append(mean_squared_error(y_val_fold, y_pred))

print(f"Cross-validation RMSE: {np.sqrt(np.mean(cv_rmse)):4f}")

# Train final model on all training data
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=callbacks,
)

# Generate predictions for test set
y_test_pred = model.predict(test_df[features])

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "review_overall": y_test_pred})

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

# Define features and target
features = [col for col in train_df.columns if col != "review_overall"]
target = "review_overall"

# Create train/validation split for final model
X_train = train_df[features]
y_train = train_df[target]

# Initialize LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = []

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
        callbacks=callbacks,
        valid_sets=[val_data],
    )

    y_pred = model.predict(X_val_fold)
    cv_rmse.append(mean_squared_error(y_val_fold, y_pred))

# Calculate final RMSE on validation set
final_cv_rmse = np.sqrt(np.mean(cv_rmse))
print(f"5-fold Cross-validation RMSE: {final_cv_rmse:.4f}")

# Train final model on all training data
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=callbacks,
)

# Generate predictions for test set
y_test_pred = model.predict(test_df[features])

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "review_overall": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


submission = pd.DataFrame({"id": range(len(test_df)), "review_overall": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
features = [col for col in train_df.columns if col != "review_overall"]
target = "review_overall"

# Create train/validation split for evaluation
X_train = train_df[features]
y_train = train_df[target]

# Initialize LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = []

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
        callbacks=callbacks,
        valid_sets=[val_data],
    )

    y_pred = model.predict(X_val_fold)
    cv_rmse.append(mean_squared_error(y_val_fold, y_pred))

# Calculate final RMSE on validation set
final_cv_rmse = np.sqrt(np.mean(cv_rmse))
print(f"5-fold Cross-validation RMSE: {final_cv_rmse:.4f}")

# Train final model on all training data
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=callbacks,
)

# Generate predictions for test set
y_test_pred = model.predict(test_df[features])

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "review_overall": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


# Check if test data has id column
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# Create submission file
submission = pd.DataFrame({"id": test_ids, "review_overall": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
features = [col for col in train_df.columns if col != "review_overall"]
target = "review_overall"

# Create train/validation split for evaluation
X_train = train_df[features]
y_train = train_df[target]

# Initialize LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = []

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
        callbacks=callbacks,
        valid_sets=[val_data],
    )

    y_pred = model.predict(X_val_fold)
    cv_rmse.append(mean_squared_error(y_val_fold, y_pred))

# Calculate final RMSE on validation set
final_cv_rmse = np.sqrt(np.mean(cv_rmse))
print(f"5-fold Cross-validation RMSE: {final_cv_rmse:.4f}")

# Train final model on all training data
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=callbacks,
)

# Generate predictions for test set
y_test_pred = model.predict(test_df[features])

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "review_overall": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
