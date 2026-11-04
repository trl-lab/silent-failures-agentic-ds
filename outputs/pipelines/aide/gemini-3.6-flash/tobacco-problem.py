import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from catboost import CatBoostClassifier

    USE_CATBOOST = True
except ImportError:
    import lightgbm as lgb

    USE_CATBOOST = False

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "number_health_problems"

# Separate target
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test = test_df.copy()

# Remove 'id' column if present in features
for id_col in ["id", "ID", "Id"]:
    if id_col in X.columns:
        X = X.drop(columns=[id_col])
    if id_col in X_test.columns:
        X_test = X_test.drop(columns=[id_col])

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Process categorical features
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

if USE_CATBOOST:
    for col in categorical_cols:
        X[col] = X[col].astype(str).fillna("Missing")
        X_test[col] = X_test[col].astype(str).fillna("Missing")
else:
    for col in categorical_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), num_classes))
test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

    if USE_CATBOOST:
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            cat_features=categorical_cols,
            eval_metric="Accuracy",
            random_seed=42 + fold,
            verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        val_probs = model.predict_proba(X_val)
        t_probs = model.predict_proba(X_test)
    else:
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42 + fold,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_probs = model.predict_proba(X_val)
        t_probs = model.predict_proba(X_test)

    oof_preds[val_idx] = val_probs
    test_preds += t_probs / skf.n_splits

oof_classes = np.argmax(oof_preds, axis=1)
accuracy = accuracy_score(y_encoded, oof_classes)
print(f"Validation Accuracy: {accuracy:.5f}")

# Final predictions mapping
test_class_indices = np.argmax(test_preds, axis=1)
test_predictions = le.inverse_transform(test_class_indices)

# Save submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), target_col: test_predictions}
)
sub.to_csv("./working/submission.csv", index=False)
