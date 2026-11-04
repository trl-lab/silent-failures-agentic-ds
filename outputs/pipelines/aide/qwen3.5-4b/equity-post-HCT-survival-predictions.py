import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import concordance_index
import warnings

warnings.filterwarnings("ignore")

# Read data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude ID and target)
feature_cols = [col for col in train.columns if col not in ["id", "efs", "efs_time"]]

# Separate features and target
X_train = train[feature_cols].copy()
y_train = train["efs"].copy()
X_test = test[feature_cols].copy()

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("encoder", LabelEncoder()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# Fit preprocessor on training data
preprocessor.fit(X_train)

# Transform training and test data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)

# Use 5-fold stratified cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for each fold
y_pred_all = []
y_true_all = []
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_processed, y_train)):
    X_train_fold = X_train_processed[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_processed[val_idx]
    y_val_fold = y_train[val_idx]

    # Train model on fold
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions on validation set
    val_pred = model.predict(X_val_fold)
    y_pred_all.append(val_pred)
    y_true_all.append(y_val_fold)
    scores.append(concordance_index(y_val_fold, val_pred))

# Calculate average concordance index
avg_score = np.mean(scores)
print(f"5-fold Cross-Validation Concordance Index: {avg_score:.4f}")

# Generate predictions on test set
test_pred = model.predict(X_test_processed)

# Create submission file
submission = pd.DataFrame({"id": test["id"].values, "efs": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
