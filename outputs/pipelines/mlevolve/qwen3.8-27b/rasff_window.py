import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ---------------- Load data ----------------
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "risk_decision"

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Test columns: {list(test.columns)}")
print(f"Target distribution:\n{train[TARGET].value_counts()}")
print(f"Target dtype: {train[TARGET].dtype}")

# ---------------- Identify features ----------------
feature_candidates = [c for c in train.columns if c != TARGET and c != "id"]
print(f"Feature candidates ({len(feature_candidates)}): {feature_candidates}")

# ---------------- Encode target to 0-indexed ----------------
y = train[TARGET].copy()
classes = sorted(y.unique())
n_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
y_encoded = y.map(class_to_idx).astype(int)
print(f"Number of classes: {n_classes}, classes: {classes}")

# ---------------- Prepare feature matrices ----------------
X = train[feature_candidates].copy()
X_test = test[[c for c in feature_candidates if c in test.columns]].copy()

# Ensure X_test has the same columns (and order) as X
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = np.nan
X_test = X_test[X.columns]

print(f"X shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")


# ---------------- Preprocess features ----------------
def preprocess(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("int64") // 10**9
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


X = preprocess(X)
X_test = preprocess(X_test)
print(f"Processed X dtypes:\n{X.dtypes}")

# ---------------- Validation split (stratified) ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# ---------------- Train LightGBM with early stopping ----------------
params = {
    "objective": "multiclass" if n_classes > 2 else "binary",
    "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
    "num_class": n_classes if n_classes > 2 else 1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

cat_features = [col for col in X_train.columns if X_train[col].dtype.name == "category"]
print(f"Categorical features: {cat_features}")

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    categorical_feature=cat_features,
)

# ---------------- Evaluate on validation set ----------------
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc}")

best_iter = (
    model.best_iteration_
    if model.best_iteration_ is not None
    else params["n_estimators"]
)
print(f"Best iteration: {best_iter}")

# ---------------- Retrain on full data ----------------
params_full = params.copy()
params_full["n_estimators"] = best_iter
model_full = lgb.LGBMClassifier(**params_full)
model_full.fit(X, y_encoded, categorical_feature=cat_features)

# ---------------- Predict on test ----------------
test_pred = model_full.predict(X_test)
test_pred_labels = [classes[p] for p in test_pred]

# ---------------- Build submission ----------------
submission = pd.DataFrame(
    {"id": range(len(test_pred_labels)), "risk_decision": test_pred_labels}
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission shape: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

print(f"Final Validation Score: {val_acc}")
