import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# --- Load data ---
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# --- Identify columns ---
target_col = "class"
id_col = "Mouse ID"
meta_cols = ["Genotype", "Treatment type", "Behavior"]
exclude_cols = [target_col, id_col] + meta_cols

protein_cols = [c for c in train_df.columns if c not in exclude_cols]

# --- Build feature matrices ---
def build_features(df, protein_cols):
    X = pd.DataFrame(index=df.index)
    for c in protein_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            X[c] = np.nan
    return X

train_feat = build_features(train_df, protein_cols)
test_feat = build_features(test_df, protein_cols)

# Align columns
common_cols = list(train_feat.columns)
test_feat = test_feat.reindex(columns=common_cols, fill_value=0.0)

# --- Target ---
y_full = train_df[target_col].astype(str)

# --- Split (train/val) ---
X_all = train_feat.values.astype(float)
y_all = y_full.reset_index(drop=True)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# --- Fit imputer + scaler on TRAIN ONLY (no leakage) ---
imputer = SimpleImputer(strategy="median")
X_tr_imp = imputer.fit_transform(X_tr)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(test_feat.values.astype(float))

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# --- Train model ---
model = LogisticRegression(max_iter=5000, random_state=SEED, C=1.0)
model.fit(X_tr_scaled, y_tr)

# --- Validation score (accuracy, matches official metric) ---
val_pred = model.predict(X_val_scaled)
val_score = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_score:.4f}")

# --- Test predictions ---
test_pred = model.predict(X_test_scaled)

# --- Write submission ---
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({
    "id": np.arange(len(test_pred)),
    "class": test_pred
})
submission.to_csv("./submission/submission.csv", index=False)
print("Submission written to ./submission/submission.csv")