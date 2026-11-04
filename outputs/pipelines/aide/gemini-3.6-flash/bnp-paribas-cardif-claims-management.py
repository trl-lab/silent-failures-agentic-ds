import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Target setup and column cleaning
y = train["target"].values
drop_cols = ["target"]
if "ID" in train.columns:
    drop_cols.append("ID")
if "id" in train.columns:
    drop_cols.append("id")

# Drop known duplicate features to reduce noise and redundancy
duplicate_cols = ["v8", "v91", "v71", "v47"]
drop_cols.extend([c for c in duplicate_cols if c in train.columns])

feature_cols = [c for c in train.columns if c not in drop_cols]
train_df = train[feature_cols].copy()
test_df = test[[c for c in feature_cols if c in test.columns]].copy()

# Feature engineering: Null count and Zero count
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
train_df["null_count"] = train_df.isnull().sum(axis=1)
test_df["null_count"] = test_df.isnull().sum(axis=1)

train_df["zero_count"] = (train_df[num_cols] == 0).sum(axis=1)
test_df["zero_count"] = (test_df[num_cols] == 0).sum(axis=1)

# Feature engineering: Detailed string decomposition for high-cardinality v22
if "v22" in train_df.columns:
    train_df["v22_len"] = train_df["v22"].fillna("").astype(str).str.len()
    test_df["v22_len"] = test_df["v22"].fillna("").astype(str).str.len()
    train_df["v22_char1"] = train_df["v22"].fillna("").astype(str).str[:1]
    test_df["v22_char1"] = test_df["v22"].fillna("").astype(str).str[:1]
    train_df["v22_char2"] = train_df["v22"].fillna("").astype(str).str[1:2]
    test_df["v22_char2"] = test_df["v22"].fillna("").astype(str).str[1:2]

# Identify categorical features
cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

# Process categoricals and create frequency encoding
for col in cat_cols:
    train_df[col] = train_df[col].fillna("missing").astype(str)
    test_df[col] = test_df[col].fillna("missing").astype(str)

    freq = (
        pd.concat([train_df[col], test_df[col]]).value_counts(normalize=True).to_dict()
    )
    train_df[f"{col}_freq"] = train_df[col].map(freq).astype(float)
    test_df[f"{col}_freq"] = test_df[col].map(freq).astype(float)

# CV Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), start=1):
    X_train, y_train = train_df.iloc[train_idx], y[train_idx]
    X_val, y_val = train_df.iloc[val_idx], y[val_idx]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        eval_metric="Logloss",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test_df)[:, 1] / 5.0

cv_score = log_loss(y, oof_preds)
print(f"5-Fold Stratified CV Log Loss: {cv_score:.5f}")

# Save submission file
sub = pd.DataFrame({"id": np.arange(len(test_df)), "target": test_preds})
sub.to_csv("./working/submission.csv", index=False)
