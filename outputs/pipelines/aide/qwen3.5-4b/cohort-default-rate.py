import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Average or Greater than 30"
feature_cols = [col for col in train_df.columns if col != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Split training data for validation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Prepare LightGBM data
train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model with callbacks
params = {"objective": "binary", "metric": "auc", "verbose": -1}
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# Evaluate on validation set
y_val_pred = model.predict(val_data, num_iteration=model.best_iteration)
val_metric = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_metric}")

# Predict on test set
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Save submission
submission = pd.DataFrame(
    {"id": test_df.index, "Average or Greater than 30": y_test_pred}
)
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Average or Greater than 30"
feature_cols = [col for col in train_df.columns if col != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Split training data for validation (Hold-out)
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model with callbacks
params = {"objective": "binary", "metric": "auc", "verbose": -1}
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(val_data, num_iteration=model.best_iteration)
val_metric = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_metric}")

# Predict on test set
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Save submission
submission = pd.DataFrame(
    {"id": test_df.index, "Average or Greater than 30": y_test_pred}
)
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Average or Greater than 30"
feature_cols = [col for col in train_df.columns if col != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Split training data for validation (Hold-out)
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model with callbacks
params = {"objective": "binary", "metric": "auc", "verbose": -1}
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(val_data, num_iteration=model.best_iteration)
val_metric = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_metric}")

# Predict on test set
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Save submission
submission = pd.DataFrame(
    {"id": test_df.index, "Average or Greater than 30": y_test_pred}
)
submission.to_csv("./working/submission.csv", index=False)
