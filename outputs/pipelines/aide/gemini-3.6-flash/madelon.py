import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load train and test data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class"
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].values
y_raw = train_df[target_col].values

# Extract test features in the same order
X_test = test_df[[c for c in feature_cols if c in test_df.columns]].values

# Target label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(train_df), dtype=int)
test_preds_proba = np.zeros((len(test_df), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Feature selection using ExtraTrees with 500 estimators for stability
    selector_model = ExtraTreesClassifier(
        n_estimators=500, random_state=SEED, n_jobs=-1
    )
    selector_model.fit(X_train, y_train)

    # Select top 20 features (Madelon contains 20 informative features)
    importances = selector_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]

    X_train_sel = X_train[:, top_indices]
    X_val_sel = X_val[:, top_indices]
    X_test_sel = X_test[:, top_indices]

    # Generate degree 2 polynomial interaction features on selected top 20 attributes
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_sel)
    X_val_poly = poly.transform(X_val_sel)
    X_test_poly = poly.transform(X_test_sel)

    # Fit final classifier with entropy split criterion on interaction-augmented feature space
    clf = ExtraTreesClassifier(
        n_estimators=1000,
        criterion="entropy",
        max_features=1.0,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_poly, y_train)

    val_pred = clf.predict(X_val_poly)
    oof_preds[val_idx] = val_pred

    test_preds_proba += clf.predict_proba(X_test_poly) / skf.n_splits

cv_accuracy = accuracy_score(y, oof_preds)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# Map test predictions back to original class labels
test_preds = np.argmax(test_preds_proba, axis=1)
test_preds_labels = le.inverse_transform(test_preds)

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds_labels)), "Class": test_preds_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
