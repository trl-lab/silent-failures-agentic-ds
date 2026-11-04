import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature types
categorical_cols = train_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Handle missing values
train_df = train_df.dropna(subset=["target"])
test_df = test_df.dropna(subset=["target"])

# Create feature matrix
X_train = train_df.drop(columns=["id", "target"])
X_test = test_df.drop(columns=["id", "target"])

# Encode categorical features
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_transformer.fit(X_train[categorical_cols])
X_train_encoded = categorical_transformer.transform(X_train[categorical_cols])
X_test_encoded = categorical_transformer.transform(X_test[categorical_cols])

# Convert to numpy arrays
X_train_encoded = X_train_encoded.toarray()
X_test_encoded = X_test_encoded.toarray()
y_train = train_df["target"].values

# 5-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

roc_auc_scores = []
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded)):
    X_train_fold = X_train_encoded[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_encoded[val_idx]
    y_val_fold = y_train[val_idx]

    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=categorical_cols
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        feature_name=categorical_cols,
        reference=train_data,
    )

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100, verbose=False),
    ]

    model.fit(
        train_data, callbacks=callbacks, valid_sets=[val_data], eval_set=[val_data]
    )

    y_pred = model.predict(X_val_fold)
    roc_auc_scores.append(roc_auc_score(y_val_fold, y_pred))

# Calculate mean ROC AUC
mean_roc_auc = np.mean(roc_auc_scores)
print(f"5-Fold Cross-Validation ROC AUC: {mean_roc_auc:.4f}")

# Train final model on all data
train_data = lgb.Dataset(X_train_encoded, label=y_train, feature_name=categorical_cols)
val_data = lgb.Dataset(X_train_encoded, label=y_train, feature_name=categorical_cols)
callbacks = [
    lgb.early_stopping(50, verbose=False),
    lgb.log_evaluation(100, verbose=False),
]
model.fit(train_data, callbacks=callbacks, valid_sets=[val_data], eval_set=[val_data])

# Generate predictions for test set
y_test_pred = model.predict(X_test_encoded)

# Create submission file
submission = pd.DataFrame({"id": test_df["id"].values, "target": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
