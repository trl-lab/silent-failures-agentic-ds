import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class"


# 2. Feature Engineering
def engineer_features(df):
    df = df.copy()

    # Scale log features
    scale_cols = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "ConvexArea",
        "EquivDiameter",
    ]
    for col in scale_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Geometric ratios and interaction features
    if "Area" in df.columns and "ConvexArea" in df.columns:
        df["Area_Convex_Ratio"] = df["Area"] / (df["ConvexArea"] + 1e-6)
    if "MajorAxisLength" in df.columns and "MinorAxisLength" in df.columns:
        df["Axis_Product"] = df["MajorAxisLength"] * df["MinorAxisLength"]
        df["Axis_Diff"] = df["MajorAxisLength"] - df["MinorAxisLength"]
    if "Perimeter" in df.columns and "Area" in df.columns:
        df["Perimeter_Area_Ratio"] = df["Perimeter"] / (df["Area"] + 1e-6)

    return df


train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

feature_cols = [c for c in train_fe.columns if c != target_col and c != "id"]

# 3. Label Encoding for Target
le = LabelEncoder()
y = le.fit_transform(train_fe[target_col])
X = train_fe[feature_cols]
X_test = test_fe[feature_cols]

# 4. Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_fe), len(le.classes_)))
test_preds = np.zeros((len(test_fe), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / skf.n_splits

# 5. Metric Evaluation
oof_classes = np.argmax(oof_preds, axis=1)
overall_accuracy = accuracy_score(y, oof_classes)
print(f"5-Fold Cross-Validation Accuracy Score: {overall_accuracy:.5f}")

# 6. Generate Submission File
test_class_indices = np.argmax(test_preds, axis=1)
test_class_names = le.inverse_transform(test_class_indices)

if "id" in test_df.columns:
    ids = test_df["id"]
else:
    ids = np.arange(len(test_df))

submission = pd.DataFrame({"id": ids, "Class": test_class_names})

os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
print("Submission file created at './working/submission.csv'.")
