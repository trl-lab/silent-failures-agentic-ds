import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Target encoding
le = LabelEncoder()
y = le.fit_transform(train_df["class"])

# Ordinal mappings
ordinal_mappings = {
    "buying": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "maint": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
    "persons": {"2": 2, "4": 4, "more": 6},
    "lug_boot": {"small": 1, "med": 2, "big": 3},
    "safety": {"low": 1, "med": 2, "high": 3},
}


def preprocess_and_engineer(df):
    data = pd.DataFrame()
    for col, mapping in ordinal_mappings.items():
        data[f"{col}_num"] = df[col].astype(str).map(mapping)

    # Domain interaction features
    data["total_price"] = data["buying_num"] + data["maint_num"]
    data["capacity_safety"] = data["persons_num"] * data["safety_num"]
    data["comfort_capacity"] = data["doors_num"] * data["persons_num"]
    data["tech_score"] = data["safety_num"] * data["lug_boot_num"] * data["persons_num"]
    return data


X = preprocess_and_engineer(train_df)
X_test = preprocess_and_engineer(test_df)

# 2. 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_probs = np.zeros((len(test_df), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    val_probs = model.predict_proba(X_val)
    oof_preds[val_idx] = np.argmax(val_probs, axis=1)
    test_probs += model.predict_proba(X_test) / skf.n_splits

# 3. Validation Performance
cv_accuracy = accuracy_score(y, oof_preds)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 4. Save Test Predictions
os.makedirs("./working", exist_ok=True)
test_pred_indices = np.argmax(test_probs, axis=1)
test_pred_labels = le.inverse_transform(test_pred_indices)

submission = pd.DataFrame({"id": range(len(test_df)), "class": test_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
