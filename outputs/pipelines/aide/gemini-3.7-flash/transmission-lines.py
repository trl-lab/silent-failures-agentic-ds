import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "VOLT_CLASS"
drop_cols = [target_col]

# Handle target encoding
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
num_classes = len(le.classes_)

# 2. Identify and preprocess features
features = [col for col in train_df.columns if col not in drop_cols]

# Identify column types
cat_cols = []
num_cols = []

for col in features:
    # Check if object, category or low-cardinality numeric
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Combine train and test for consistent categorical conversion
combined = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

for col in cat_cols:
    # Convert object to string category and label encode
    combined[col] = combined[col].astype(str).fillna("missing")
    col_le = LabelEncoder()
    combined[col] = col_le.fit_transform(combined[col])
    combined[col] = combined[col].astype("category")

# Add frequency encoding for high-cardinality categorical features
for col in cat_cols:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq).astype(float)

X = combined.iloc[: len(train_df)].copy()
X_test = combined.iloc[len(train_df) :].reset_index(drop=True).copy()

# 3. Model Training with 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), num_classes))
test_preds = np.zeros((len(test_df), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 4. Evaluation
oof_classes = np.argmax(oof_preds, axis=1)
oof_accuracy = accuracy_score(y, oof_classes)
print(f"5-Fold CV Out-Of-Fold Accuracy: {oof_accuracy:.5f}")

# 5. Generate Submission
test_pred_classes = np.argmax(test_preds, axis=1)
test_pred_labels = le.inverse_transform(test_pred_classes)

submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "VOLT_CLASS": test_pred_labels}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
