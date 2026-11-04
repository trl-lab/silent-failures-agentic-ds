import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
original_df = pd.read_csv("./input/original/train.csv")

test_ids = test_df["id"].copy()

# Target encoding
le = LabelEncoder()
train_df["target"] = le.fit_transform(train_df["NObeyesdad"])
original_df["target"] = le.transform(original_df["NObeyesdad"])


def engineer_features(df):
    df = df.copy()

    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    binary_map = {"no": 0, "yes": 1}
    gender_map = {"Male": 0, "Female": 1}
    mtrans_map = {
        "Public_Transportation": 0,
        "Automobile": 1,
        "Walking": 2,
        "Motorbike": 3,
        "Bike": 4,
    }

    if "CAEC" in df.columns and df["CAEC"].dtype == "object":
        df["CAEC_ord"] = df["CAEC"].map(caec_map).fillna(0).astype(int)
    if "CALC" in df.columns and df["CALC"].dtype == "object":
        df["CALC_ord"] = df["CALC"].map(calc_map).fillna(0).astype(int)
    if "Gender" in df.columns and df["Gender"].dtype == "object":
        df["Gender_num"] = df["Gender"].map(gender_map).fillna(0).astype(int)
    if "MTRANS" in df.columns and df["MTRANS"].dtype == "object":
        df["MTRANS_num"] = df["MTRANS"].map(mtrans_map).fillna(0).astype(int)

    for col in ["FAVC", "SCC", "SMOKE", "family_history_with_overweight"]:
        if col in df.columns and df[col].dtype == "object":
            df[col + "_num"] = df[col].map(binary_map).fillna(0).astype(int)

    # Domain and Interaction Features
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Height_sq"] = df["Height"] ** 2
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["FAF_CH2O"] = df["FAF"] * df["CH2O"]
    df["Water_per_Weight"] = df["CH2O"] / (df["Weight"] + 1e-5)
    df["Activity_Ratio"] = df["FAF"] / (df["TUE"] + 1.0)
    df["FCVC_NCP"] = df["FCVC"] * df["NCP"]

    drop_cols = [
        "id",
        "NObeyesdad",
        "CAEC",
        "CALC",
        "Gender",
        "FAVC",
        "SCC",
        "SMOKE",
        "family_history_with_overweight",
        "MTRANS",
        "target",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    return df[feature_cols]


X_train_full = engineer_features(train_df)
y_train_full = train_df["target"].values
X_test = engineer_features(test_df)
X_original = engineer_features(original_df)
y_original = original_df["target"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_classes = len(le.classes_)
oof_preds = np.zeros((len(train_df), n_classes))
test_preds = np.zeros((len(test_df), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    X_tr = pd.concat([X_train_full.iloc[train_idx], X_original], axis=0).reset_index(
        drop=True
    )
    y_tr = np.concatenate([y_train_full[train_idx], y_original], axis=0)

    X_va = X_train_full.iloc[val_idx].reset_index(drop=True)
    y_va = y_train_full[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_va)
    test_preds += model.predict_proba(X_test) / skf.n_splits

oof_classes = np.argmax(oof_preds, axis=1)
accuracy = accuracy_score(y_train_full, oof_classes)
print(f"Validation Accuracy: {accuracy:.5f}")

# Save test predictions
test_classes = np.argmax(test_preds, axis=1)
test_labels = le.inverse_transform(test_classes)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "NObeyesdad": test_labels})
submission.to_csv("./working/submission.csv", index=False)
