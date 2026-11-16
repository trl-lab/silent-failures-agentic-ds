import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "cannabis"
id_col = "id"

# Determine feature columns shared between train and test
feature_cols = [
    c
    for c in test_df.columns
    if c in train_df.columns and c not in [id_col, target_col]
]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_prob = np.zeros((len(train_df), num_classes))
test_preds_prob = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1000,
        learning_rate=0.05,
        depth=5,
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    oof_preds_prob[val_idx] = model.predict_proba(X_val)
    test_preds_prob += model.predict_proba(X_test) / skf.n_splits

oof_pred_labels = np.argmax(oof_preds_prob, axis=1)
cv_accuracy = accuracy_score(y_encoded, oof_pred_labels)
print(f"Validation Accuracy: {cv_accuracy:.4f}")

# Prepare submission
os.makedirs("./working", exist_ok=True)
test_pred_indices = np.argmax(test_preds_prob, axis=1)
test_pred_classes = le.inverse_transform(test_pred_indices)

submission = pd.DataFrame(
    {
        "id": test_df[id_col] if id_col in test_df.columns else np.arange(len(test_df)),
        "cannabis": test_pred_classes,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
