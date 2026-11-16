import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.metrics import accuracy_score
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Cover_Type"
feature_cols = [col for col in train.columns if col not in ["id", target_col]]

# Prepare training data
X_train = train[feature_cols].copy()
y_train = train[target_col].copy()

# Prepare test data
X_test = test[feature_cols].copy()

# Split into train/val (80/20)
X_train_val, X_test_val, y_train_val, y_test_val = train_val_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Create preprocessing pipeline
preprocessor = Pipeline(
    [("scaler", StandardScaler()), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
)

# Train model with callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
    X_fold_train = X_train_val.iloc[train_idx]
    y_fold_train = y_train_val.iloc[train_idx]
    X_fold_val = X_train_val.iloc[val_idx]
    y_fold_val = y_train_val.iloc[val_idx]

    # Train model
    model = lgb.train(
        params={"objective": "multinomial", "num_class": 7, "metric": "accuracy"},
        train_data=lgb.Dataset(X_fold_train, label=y_fold_train),
        valid_sets=[lgb.Dataset(X_fold_val, label=y_fold_val)],
        callbacks=callbacks,
        num_boost_round=1000,
    )

    # Predict on validation
    y_pred = model.predict(X_fold_val)
    cv_scores.append(accuracy_score(y_fold_val, y_pred))

# Calculate mean accuracy
mean_accuracy = np.mean(cv_scores)
print(f"5-fold cross-validation accuracy: {mean_accuracy:.4f}")

# Train final model on all training data
X_train_final = X_train.copy()
y_train_final = y_train.copy()

model_final = lgb.train(
    params={"objective": "multinomial", "num_class": 7, "metric": "accuracy"},
    train_data=lgb.Dataset(X_train_final, label=y_train_final),
    callbacks=callbacks,
    num_boost_round=1000,
)

# Predict on test data
y_test_pred = model_final.predict(X_test)
y_test_pred = np.round(y_test_pred).astype(int)

# Create submission
submission = pd.DataFrame({"id": test["id"], "Cover_Type": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Cover_Type"
feature_cols = [col for col in train.columns if col not in ["id", target_col]]

# Prepare training data
X_train = train[feature_cols].copy()
y_train = train[target_col].copy()
X_test = test[feature_cols].copy()

# Split into train/val (80/20)
from sklearn.model_selection import train_test_split

X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Preprocess: scale numerical features
scaler = StandardScaler()
X_train_val_scaled = X_train_val.copy()
X_train_val_scaled[feature_cols] = scaler.fit_transform(X_train_val[feature_cols])
X_test_val_scaled = X_test_val.copy()
X_test_val_scaled[feature_cols] = scaler.transform(X_test_val[feature_cols])

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_scaled, y_train_val)):
    X_fold_train = X_train_val_scaled.iloc[train_idx]
    y_fold_train = y_train_val.iloc[train_idx]
    X_fold_val = X_train_val_scaled.iloc[val_idx]
    y_fold_val = y_train_val.iloc[val_idx]

    model.fit(
        X_fold_train,
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_fold_val)
    cv_scores.append(accuracy_score(y_fold_val, y_pred))

# Calculate mean accuracy
mean_accuracy = np.mean(cv_scores)
print(f"5-fold cross-validation accuracy: {mean_accuracy:.4f}")

# Train final model on all training data
model.fit(
    X_train_val_scaled,
    y_train_val,
    eval_set=[(X_test_val_scaled, y_test_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test data
y_test_pred = model.predict(X_test)
y_test_pred = np.round(y_test_pred).astype(int)

# Create submission
submission = pd.DataFrame({"id": test["id"], "Cover_Type": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
