import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Drop missing target rows
train_df = train_df.dropna(subset=["outcomes"]).reset_index(drop=True)

# Encode target
le = LabelEncoder()
y = le.fit_transform(train_df["outcomes"])
num_classes = len(le.classes_)

# Combine data for feature processing
train_len = len(train_df)
feature_cols = [c for c in train_df.columns if c != "outcomes"]
df_all = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0).reset_index(
    drop=True
)

# Classify features
text_cols = []
cat_cols = []
num_cols = []
date_cols = []

for col in feature_cols:
    if "date" in col.lower():
        date_cols.append(col)
    elif df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        sample_vals = df_all[col].dropna().astype(str)
        if len(sample_vals) > 0 and sample_vals.str.len().mean() > 30:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

processed_features = pd.DataFrame(index=df_all.index)

# Date processing
for col in date_cols:
    dt_series = pd.to_datetime(df_all[col], errors="coerce")
    processed_features[f"{col}_year"] = dt_series.dt.year.fillna(-1)
    processed_features[f"{col}_month"] = dt_series.dt.month.fillna(-1)
    processed_features[f"{col}_day"] = dt_series.dt.day.fillna(-1)
    processed_features[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1)
    processed_features[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1)
    processed_features[f"{col}_isnull"] = dt_series.isna().astype(int)

# Categorical processing
for col in cat_cols:
    col_str = df_all[col].fillna("MISSING").astype(str)
    freq = col_str.value_counts(normalize=True).to_dict()
    processed_features[f"{col}_freq"] = col_str.map(freq).fillna(0)

    cat_le = LabelEncoder()
    processed_features[f"{col}_le"] = cat_le.fit_transform(col_str)

# Numerical processing
for col in num_cols:
    processed_features[f"{col}_num"] = pd.to_numeric(
        df_all[col], errors="coerce"
    ).fillna(0)
    processed_features[f"{col}_isnull"] = df_all[col].isna().astype(int)

# Text processing
all_text_cols = text_cols + [c for c in cat_cols if c not in text_cols]
for col in all_text_cols:
    df_all[col] = df_all[col].fillna("").astype(str)

df_all["combined_text"] = df_all[all_text_cols].apply(
    lambda row: " ".join(row.values), axis=1
)

# TF-IDF & SVD
tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df_all["combined_text"])

svd = TruncatedSVD(n_components=50, random_state=42)
svd_feats = svd.fit_transform(tfidf_matrix)
svd_df = pd.DataFrame(svd_feats, columns=[f"svd_{i}" for i in range(50)])

X_all = pd.concat([processed_features, svd_df], axis=1)

X_train = X_all.iloc[:train_len].copy().reset_index(drop=True)
X_test = X_all.iloc[train_len:].copy().reset_index(drop=True)

tfidf_train = tfidf_matrix[:train_len]
tfidf_test = tfidf_matrix[train_len:]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF Logistic Regression meta-features
lr_oof = np.zeros((train_len, num_classes))
lr_test_preds = np.zeros((len(X_test), num_classes))

for train_idx, val_idx in skf.split(X_train, y):
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(tfidf_train[train_idx], y[train_idx])
    lr_oof[val_idx] = lr.predict_proba(tfidf_train[val_idx])
    lr_test_preds += lr.predict_proba(tfidf_test) / 5.0

for c in range(num_classes):
    X_train[f"lr_prob_{c}"] = lr_oof[:, c]
    X_test[f"lr_prob_{c}"] = lr_test_preds[:, c]

# Ensemble Training
oof_preds_cb = np.zeros((train_len, num_classes))
oof_preds_xgb = np.zeros((train_len, num_classes))

test_preds_cb = np.zeros((len(X_test), num_classes))
test_preds_xgb = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    # CatBoost Model
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.08,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=40, verbose=False
    )
    oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)
    test_preds_cb += cb_model.predict_proba(X_test) / 5.0

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.08,
        max_depth=6,
        tree_method="hist",
        objective="multi:softprob",
        num_class=num_classes,
        early_stopping_rounds=40,
        random_state=42 + fold,
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_preds_xgb += xgb_model.predict_proba(X_test) / 5.0

# Blend predictions
oof_preds_blend = 0.5 * oof_preds_cb + 0.5 * oof_preds_xgb
test_preds_blend = 0.5 * test_preds_cb + 0.5 * test_preds_xgb

final_oof_labels = np.argmax(oof_preds_blend, axis=1)
cv_accuracy = accuracy_score(y, final_oof_labels)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# Save Submission
final_test_labels = np.argmax(test_preds_blend, axis=1)
final_test_decoded = le.inverse_transform(final_test_labels)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "outcomes": final_test_decoded}
)
submission.to_csv("./working/submission.csv", index=False)
