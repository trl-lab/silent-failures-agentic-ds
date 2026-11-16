import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "device_class"

# Encode target labels
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_df[target_col].astype(str))
num_classes = len(target_encoder.classes_)

# Combine data for consistent preprocessing
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()
n_train = len(train_features)
combined_df = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# Identify column types
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if combined_df[col].dtype == "object" or combined_df[col].dtype.name == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Add missing value indicators for columns with missing values
for col in combined_df.columns:
    if combined_df[col].isnull().sum() > 0:
        combined_df[f"{col}_isnan"] = combined_df[col].isnull().astype(int)

# Frequency encoding for categorical columns
for col in cat_cols:
    freq_map = combined_df[col].value_counts(dropna=False, normalize=True).to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq_map).astype(float)
    # Ordinal label encoding
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))

# Create numeric interaction / aggregation features if numerical columns exist
if len(num_cols) >= 2:
    for i in range(min(len(num_cols), 4)):
        for j in range(i + 1, min(len(num_cols), 4)):
            c1, c2 = num_cols[i], num_cols[j]
            combined_df[f"{c1}_add_{c2}"] = combined_df[c1] + combined_df[c2]
            combined_df[f"{c1}_sub_{c2}"] = combined_df[c1] - combined_df[c2]
            combined_df[f"{c1}_mul_{c2}"] = combined_df[c1] * combined_df[c2]
            combined_df[f"{c1}_div_{c2}"] = combined_df[c1] / (
                combined_df[c2].abs() + 1e-6
            )

# Split back into train and test sets
X_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# 2. Stratified K-Fold Cross-Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(X_test), num_classes))

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
    "n_estimators": 1000,
    "n_jobs": -1,
}

# 3. Model Training & Validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test) / n_splits

# 4. Evaluation
oof_classes = np.argmax(oof_preds, axis=1)
oof_accuracy = accuracy_score(y, oof_classes)
print(f"Validation Accuracy: {oof_accuracy:.5f}")

# 5. Generate Submission
os.makedirs("./working", exist_ok=True)
final_test_classes = np.argmax(test_preds, axis=1)
final_predictions = target_encoder.inverse_transform(final_test_classes)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_predictions)), "device_class": final_predictions}
)

submission_df.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
