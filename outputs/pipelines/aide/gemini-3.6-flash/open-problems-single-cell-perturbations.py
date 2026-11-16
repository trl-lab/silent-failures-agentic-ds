import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature and target columns
meta_cols = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
gene_cols = [c for c in train_df.columns if c not in meta_cols]

X_train_meta = train_df[["cell_type", "sm_name"]]
X_test_meta = test_df[["cell_type", "sm_name"]]

Y_train = train_df[gene_cols].values

# Encode categorical features
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = ohe.fit_transform(X_train_meta)
X_test_cat = ohe.transform(X_test_meta)

# Extract TF-IDF features from SMILES chemical strings
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=500)
X_train_smiles = tfidf.fit_transform(train_df["SMILES"].fillna("")).toarray()
X_test_smiles = tfidf.transform(test_df["SMILES"].fillna("")).toarray()

# Concatenate all feature arrays
X_train = np.hstack([X_train_cat, X_train_smiles])
X_test = np.hstack([X_test_cat, X_test_smiles])


# Metric calculation function
def calculate_mrrmse(y_true, y_pred):
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)))


# Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros_like(Y_train)
test_preds = np.zeros((len(test_df), Y_train.shape[1]))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, Y_tr = X_train[train_idx], Y_train[train_idx]
    X_va, Y_va = X_train[val_idx], Y_train[val_idx]

    model = Ridge(alpha=20.0, random_state=42)
    model.fit(X_tr, Y_tr)

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred

    test_preds += model.predict(X_test) / kf.n_splits

    fold_mrrmse = calculate_mrrmse(Y_va, val_pred)
    print(f"Fold {fold + 1} MRRMSE: {fold_mrrmse:.5f}")

cv_mrrmse = calculate_mrrmse(Y_train, oof_preds)
print(f"Overall CV MRRMSE: {cv_mrrmse:.5f}")

# Prepare and save submission
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
formatted_preds = [" ".join(np.round(row, 6).astype(str)) for row in test_preds]

sub = pd.DataFrame({"id": test_ids, "__multi_output__": formatted_preds})

sub.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")
