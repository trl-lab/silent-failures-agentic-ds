import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load train and test data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Utility"
X = train.drop(columns=[target_col]).copy()
y = train[target_col].copy()
X_test = test.copy()

# Encode target values
le = LabelEncoder()
y_encoded = le.fit_transform(y.astype(str))

# Identify categorical features
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Ensure consistent categorical formatting and handle missing values
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train), len(le.classes_)))
test_preds = np.zeros((len(test), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        loss_function="MultiClass",
        eval_metric="Accuracy",
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

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / skf.n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
acc = accuracy_score(y_encoded, oof_pred_labels)
print(f"5-Fold Cross-Validation Accuracy: {acc:.4f}")

# Generate and save test predictions
test_pred_labels = np.argmax(test_preds, axis=1)
submission = pd.DataFrame(
    {"id": np.arange(len(test)), "Utility": le.inverse_transform(test_pred_labels)}
)
submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
