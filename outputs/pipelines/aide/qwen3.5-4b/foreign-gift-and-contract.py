import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
id_col = "id" if "id" in test_df.columns else None
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df[target_col]

# Prepare test data
X_test = test_df[feature_cols]
test_ids = test_df[id_col] if id_col else test_df.index

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model with early stopping
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
print(f"5-fold CV RMSE: {np.mean(fold_scores):.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    verbose=-1,
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df[target_col]

# Prepare test data
X_test = test_df[feature_cols]
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
target_col = "Amount"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index if "id" not in test_df.columns else test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=-1,
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False) ** 0.5
    fold_scores.append(fold_rmse)

# Print average CV RMSE
avg_rmse = np.mean(fold_scores)
print(f"5-fold CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

final_model.fit(
    X_train,
    y_train,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose=-1,
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Amount": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
