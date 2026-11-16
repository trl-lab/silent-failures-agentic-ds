import os
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
feature_cols = [c for c in train_df.columns if c not in ["id", "target"]]
X = train_df[feature_cols].values
y = train_df["target"].values
X_test = test_df[feature_cols].values

# Set up 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
fold_aucs = []

# Define diverse hyperparameter configurations for Stability Selection pipelines
configurations = [
    {"k": 25, "C": 0.10, "max_samples": 0.85, "n_estimators": 200},
    {"k": 35, "C": 0.15, "max_samples": 0.85, "n_estimators": 200},
    {"k": 45, "C": 0.20, "max_samples": 0.80, "n_estimators": 200},
    {"k": 60, "C": 0.12, "max_samples": 0.80, "n_estimators": 200},
]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Standardize features strictly inside the fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    fold_val_preds = np.zeros(len(val_idx))
    fold_test_preds = np.zeros(len(test_df))

    for cfg_idx, cfg in enumerate(configurations):
        # Base estimator pipeline performing feature selection on each bootstrap replicate
        base_pipeline = Pipeline(
            [
                ("select", SelectKBest(score_func=f_classif, k=cfg["k"])),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        C=cfg["C"],
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=42 + fold * 10 + cfg_idx,
                    ),
                ),
            ]
        )

        bagging_model = BaggingClassifier(
            estimator=base_pipeline,
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            bootstrap=True,
            bootstrap_features=False,
            random_state=42 + fold * 10 + cfg_idx,
            n_jobs=-1,
        )

        bagging_model.fit(X_train_scaled, y_train)

        fold_val_preds += bagging_model.predict_proba(X_val_scaled)[:, 1] / len(
            configurations
        )
        fold_test_preds += bagging_model.predict_proba(X_test_scaled)[:, 1] / len(
            configurations
        )

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / n_splits

    fold_auc = roc_auc_score(y_val, fold_val_preds)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-Of-Fold ROC AUC: {overall_auc:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
