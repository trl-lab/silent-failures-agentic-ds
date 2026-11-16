import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Feature engineering - use all columns except target
feature_cols = [col for col in train.columns if col != "availability"]
X_train = train[feature_cols]
y_train = train["availability"]
X_test = test[feature_cols]

# Split for validation (80-20)
from sklearn.model_selection import train_test_split

X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Train with callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

model.fit(X_train_full, y_train_full, eval_set=[(X_val, y_val)], callbacks=callbacks)

# Predict on test data
y_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": range(len(y_pred)), "availability": y_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify feature columns (exclude target)
feature_cols = [col for col in train.columns if col != "availability"]

# Preprocess data - handle missing values and ensure consistent types
X_train = train[feature_cols].fillna(0).astype(float)
y_train = train["availability"].astype(int)
X_test = test[feature_cols].fillna(0).astype(float)

# Split training data for validation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Define LightGBM model parameters
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "metric": "accuracy",
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "min_data_in_leaf": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model with callbacks
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose_eval=False,
)

# Predict on test data
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_pred)), "availability": y_pred})
submission.to_csv("./working/submission.csv", index=False)

# Perform 5-fold cross-validation for metric evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_train_full):
    X_train_cv = X_train_full.iloc[train_idx]
    y_train_cv = y_train_full.iloc[train_idx]
    X_val_cv = X_train_full.iloc[val_idx]
    y_val_cv = y_train_full.iloc[val_idx]

    train_data_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
    val_data_cv = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data_cv)

    model_cv = lgb.train(
        params,
        train_data_cv,
        num_boost_round=500,
        valid_sets=[train_data_cv, val_data_cv],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        verbose_eval=False,
    )

    y_pred_cv = model_cv.predict(X_val_cv)
    y_pred_cv = np.round(y_pred_cv).astype(int)
    cv_score = accuracy_score(y_val_cv, y_pred_cv)
    cv_scores.append(cv_score)

cv_accuracy = np.mean(cv_scores)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.4f}")
