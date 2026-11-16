import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Process target
target_col = "Diagnosis"
if train_df[target_col].dtype == object or isinstance(
    train_df[target_col].iloc[0], str
):
    y = (train_df[target_col] == "M").astype(int).values
else:
    y = train_df[target_col].astype(int).values


def engineer_features(df):
    data = df.copy()
    if "ID" in data.columns:
        data = data.drop(columns=["ID"])
    if target_col in data.columns:
        data = data.drop(columns=[target_col])

    # Feature groups (1: mean, 2: se, 3: worst)
    core_features = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]

    # Worst to mean ratios and relative standard error
    for feat in core_features:
        f1, f2, f3 = f"{feat}1", f"{feat}2", f"{feat}3"
        if f1 in data.columns and f3 in data.columns:
            data[f"{feat}_ratio_worst_mean"] = (data[f3] + 1e-6) / (data[f1] + 1e-6)
        if f1 in data.columns and f2 in data.columns:
            data[f"{feat}_rel_se"] = (data[f2] + 1e-6) / (data[f1] + 1e-6)

    # Geometric and shape interactions
    if "radius1" in data.columns and "perimeter1" in data.columns:
        data["isoperimetric_quotient1"] = (4 * np.pi * data["area1"]) / (
            data["perimeter1"] ** 2 + 1e-6
        )
    if "radius3" in data.columns and "perimeter3" in data.columns:
        data["isoperimetric_quotient3"] = (4 * np.pi * data["area3"]) / (
            data["perimeter3"] ** 2 + 1e-6
        )
    if "concavity1" in data.columns and "concave_points1" in data.columns:
        data["concavity_intensity1"] = data["concavity1"] * data["concave_points1"]
    if "concavity3" in data.columns and "concave_points3" in data.columns:
        data["concavity_intensity3"] = data["concavity3"] * data["concave_points3"]

    return data


X = engineer_features(train_df)
X_test = engineer_features(test_df)

feature_cols = X.columns.tolist()
X = X.values
X_test = X_test.values

# 5-fold Stratified Cross-Validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X[val_idx], y[val_idx]

    pipeline = Pipeline(
        [
            ("power_transform", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    C=1.5,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train_fold, y_train_fold)

    oof_preds[val_idx] = pipeline.predict_proba(X_val_fold)[:, 1]
    test_preds += pipeline.predict_proba(X_test)[:, 1] / n_splits

cv_roc_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold Stratified CV ROC-AUC: {cv_roc_auc:.6f}")

# Save submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Diagnosis": test_preds})
submission.to_csv("./working/submission.csv", index=False)
