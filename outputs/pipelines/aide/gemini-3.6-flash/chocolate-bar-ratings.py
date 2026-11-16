import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Encode target variable
target_col = "Rating"
train_y_raw = train_df[target_col]
le = LabelEncoder()
train_y = le.fit_transform(train_y_raw)
num_classes = len(le.classes_)

# Combine features for preprocessing
train_df_feat = train_df.drop(columns=[target_col], errors="ignore")
test_df_feat = test_df.copy()
combined = pd.concat([train_df_feat, test_df_feat], ignore_index=True)


# Helper function to identify columns
def get_col(possible_names, df):
    for col in df.columns:
        for p in possible_names:
            if p.lower() in col.lower():
                return col
    return None


cocoa_col = get_col(["cocoa", "percent"], combined)
ref_col = get_col(["ref"], combined)
date_col = get_col(["review", "date"], combined)
maker_col = get_col(["company", "maker"], combined)
loc_col = get_col(["location"], combined)
origin_col = get_col(["broad", "country", "origin"], combined)
specific_origin_col = get_col(["specific"], combined)
ing_col = get_col(["ingredient"], combined)
char_col = get_col(["characteristic", "memorable", "flavor"], combined)

# Feature 1: Cocoa percent numerical transformation
if cocoa_col:
    combined["cocoa_num"] = (
        combined[cocoa_col].astype(str).str.replace("%", "", regex=False).astype(float)
    )
else:
    combined["cocoa_num"] = 70.0

combined["cocoa_diff_70"] = combined["cocoa_num"] - 70.0
combined["cocoa_sq"] = combined["cocoa_num"] ** 2

# Feature 2: Review Date and REF
if date_col:
    combined["review_year"] = pd.to_numeric(combined[date_col], errors="coerce").fillna(
        2015
    )
else:
    combined["review_year"] = 2015

if ref_col:
    combined["ref_num"] = pd.to_numeric(combined[ref_col], errors="coerce").fillna(0)

# Feature 3: Ingredients parsing
if ing_col:
    ing_str = combined[ing_col].astype(str).str.upper()
    combined["ing_count"] = ing_str.str.extract(r"(\d+)").fillna(0).astype(int)
    for flag in ["B", "S", "C", "V", "L", "SA"]:
        combined[f"ing_{flag}"] = ing_str.str.contains(flag, regex=False).astype(int)
else:
    combined["ing_count"] = 0

# Feature 4: Characteristics Text processing (TF-IDF + SVD)
if char_col:
    char_text = combined[char_col].fillna("").astype(str).str.lower()
    combined["char_len"] = char_text.str.len()
    combined["char_word_count"] = char_text.apply(
        lambda x: len(x.split(",")) if x else 0
    )

    tfidf = TfidfVectorizer(
        max_features=1000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_mat = tfidf.fit_transform(char_text)
    svd = TruncatedSVD(n_components=15, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)
    for i in range(15):
        combined[f"char_svd_{i}"] = svd_feats[:, i]

# Feature 5: Frequency Encodings
cat_cols = [
    c for c in [maker_col, loc_col, origin_col, specific_origin_col] if c is not None
]
for col in cat_cols:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)

# Feature 6: Group level Cocoa aggregations
if maker_col:
    maker_cocoa_mean = combined.groupby(maker_col)["cocoa_num"].transform("mean")
    combined["cocoa_diff_maker_mean"] = combined["cocoa_num"] - maker_cocoa_mean
if loc_col:
    loc_cocoa_mean = combined.groupby(loc_col)["cocoa_num"].transform("mean")
    combined["cocoa_diff_loc_mean"] = combined["cocoa_num"] - loc_cocoa_mean

# Prepare matrix for ML
drop_cols = [c for c in combined.columns if combined[c].dtype == "object"]
features_df = combined.drop(columns=drop_cols)

X_train = features_df.iloc[: len(train_df)].copy().fillna(0)
X_test = features_df.iloc[len(train_df) :].copy().fillna(0)

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), num_classes))
test_preds = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_y)):
    X_tr, y_tr = X_train.iloc[train_idx].copy(), train_y[train_idx]
    X_va, y_va = X_train.iloc[val_idx].copy(), train_y[val_idx]
    X_te = X_test.copy()

    # Out-of-fold text meta-features
    if char_col:
        tr_char = char_text.iloc[: len(train_df)].iloc[train_idx]
        va_char = char_text.iloc[: len(train_df)].iloc[val_idx]
        te_char = char_text.iloc[len(train_df) :]

        tfidf_fold = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        tfidf_tr = tfidf_fold.fit_transform(tr_char)
        tfidf_va = tfidf_fold.transform(va_char)
        tfidf_te = tfidf_fold.transform(te_char)

        meta_lr = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        meta_lr.fit(tfidf_tr, y_tr)

        meta_tr_probs = meta_lr.predict_proba(tfidf_tr)
        meta_va_probs = meta_lr.predict_proba(tfidf_va)
        meta_te_probs = meta_lr.predict_proba(tfidf_te)

        for c_idx in range(meta_tr_probs.shape[1]):
            X_tr[f"meta_lr_p{c_idx}"] = meta_tr_probs[:, c_idx]
            X_va[f"meta_lr_p{c_idx}"] = meta_va_probs[:, c_idx]
            X_te[f"meta_lr_p{c_idx}"] = meta_te_probs[:, c_idx]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=350,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        n_jobs=-1,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_va)
    test_preds += model.predict_proba(X_te) / 5.0

# Calculate CV Accuracy
oof_classes = np.argmax(oof_preds, axis=1)
acc = accuracy_score(train_y, oof_classes)
print(f"5-Fold Cross-Validation Accuracy: {acc:.5f}")

# Save Test Predictions
test_classes = np.argmax(test_preds, axis=1)
test_ratings = le.inverse_transform(test_classes)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Rating": test_ratings})
submission.to_csv("./working/submission.csv", index=False)
