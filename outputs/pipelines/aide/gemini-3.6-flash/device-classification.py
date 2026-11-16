import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "device_class"
y_raw = train_df[target_col]
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Remove explicit ID columns if present in feature set
id_cols = [c for c in X_train_raw.columns if c.lower() in ["id", "row_id", "index"]]
if id_cols:
    X_train_raw = X_train_raw.drop(columns=id_cols)
    X_test_raw = X_test_raw.drop(columns=id_cols)

# Encode target label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

# 2. Feature Engineering
df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

cat_cols = []
num_cols = []

for col in df_all.columns:
    if (
        df_all[col].dtype == "object"
        or df_all[col].dtype.name == "category"
        or df_all[col].dtype == "bool"
    ):
        cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_all[col]):
        if df_all[col].nunique() < 10 and df_all[col].dtype in [
            "int64",
            "int32",
            "int16",
            "int8",
        ]:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    else:
        cat_cols.append(col)

# Create frequency encoding for categoricals
for col in cat_cols:
    freq_map = df_all[col].value_counts(dropna=False).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq_map).fillna(0)

# Ordinal encoding for categorical columns
if len(cat_cols) > 0:
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    df_all[cat_cols] = ordinal_encoder.fit_transform(df_all[cat_cols].astype(str))

# Create missing value indicators for numerical features
for col in num_cols:
    if df_all[col].isnull().sum() > 0:
        df_all[f"{col}_isna"] = df_all[col].isnull().astype(int)

# Separate train and test splits
X_train = df_all.iloc[: len(train_df)].copy()
X_test = df_all.iloc[len(train_df) :].copy()

# 3. Model Training with 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_probs = np.zeros((len(X_train), num_classes))
test_probs = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )

    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_probs[val_idx] = clf.predict_proba(X_va)
    test_probs += clf.predict_proba(X_test) / n_splits

# 4. Evaluation and Prediction Output
oof_preds = np.argmax(oof_probs, axis=1)
accuracy = accuracy_score(y, oof_preds)

test_preds = np.argmax(test_probs, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_preds)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "device_class": test_pred_labels}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation Metric (Accuracy): {accuracy:.5f}")
