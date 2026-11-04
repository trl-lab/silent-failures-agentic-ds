import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Text and Feature Preprocessing
text_cols = ["Brand", "Variety", "Style", "Country"]
for col in text_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna("Unknown").astype(str)
        test_df[col] = test_df[col].fillna("Unknown").astype(str)
    else:
        train_df[col] = "Unknown"
        test_df[col] = "Unknown"


def construct_full_text(df):
    return (
        "brand_"
        + df["Brand"]
        + " "
        + df["Variety"]
        + " style_"
        + df["Style"]
        + " country_"
        + df["Country"]
        + " "
        + df["Brand"]
        + " "
        + df["Country"]
    )


train_text = construct_full_text(train_df)
test_text = construct_full_text(test_df)

# Text length features
train_lens = np.array(
    [
        [
            len(t),
            len(t.split()),
            len(v),
            len(b),
        ]
        for t, v, b in zip(train_text, train_df["Variety"], train_df["Brand"])
    ]
)
test_lens = np.array(
    [
        [
            len(t),
            len(t.split()),
            len(v),
            len(b),
        ]
        for t, v, b in zip(test_text, test_df["Variety"], test_df["Brand"])
    ]
)

scaler = StandardScaler()
train_lens_scaled = scaler.fit_transform(train_lens)
test_lens_scaled = scaler.transform(test_lens)

# Target preparation
target_col = "Stars"
train_df[target_col] = train_df[target_col].astype(str)
y = train_df[target_col].values

classes = np.unique(y)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}
y_idx = np.array([class_to_idx[val] for val in y])

# 3. Model Training with 5-Fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df), dtype=int)
test_decision_sum = np.zeros((len(test_df), len(classes)))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_text, y_idx)):
    fold_train_text = train_text.iloc[train_idx]
    fold_val_text = train_text.iloc[val_idx]

    # Word TF-IDF
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=18000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_word = word_tfidf.fit_transform(fold_train_text)
    X_val_word = word_tfidf.transform(fold_val_text)
    X_test_word = word_tfidf.transform(test_text)

    # Character-wb TF-IDF
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=25000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_char = char_tfidf.fit_transform(fold_train_text)
    X_val_char = char_tfidf.transform(fold_val_text)
    X_test_char = char_tfidf.transform(test_text)

    # Combine sparse text representations with normalized metadata
    X_train_fold = hstack(
        [X_train_word, X_train_char, train_lens_scaled[train_idx]]
    ).tocsr()
    X_val_fold = hstack([X_val_word, X_val_char, train_lens_scaled[val_idx]]).tocsr()
    X_test_fold = hstack([X_test_word, X_test_char, test_lens_scaled]).tocsr()

    y_train_fold = y_idx[train_idx]

    # Train Linear Support Vector Classifier
    model = LinearSVC(
        C=0.25,
        loss="squared_hinge",
        dual=True,
        max_iter=3000,
        random_state=42 + fold,
    )
    model.fit(X_train_fold, y_train_fold)

    val_dec = model.decision_function(X_val_fold)
    if val_dec.ndim == 1:
        # Binary classification fallback if needed
        val_pred = (val_dec > 0).astype(int)
    else:
        val_pred = np.argmax(val_dec, axis=1)

    oof_preds[val_idx] = val_pred

    test_dec = model.decision_function(X_test_fold)
    if test_dec.ndim == 1:
        test_decision_sum[:, 1] += test_dec
        test_decision_sum[:, 0] -= test_dec
    else:
        # Align decision function columns if all classes present
        if len(model.classes_) == len(classes):
            test_decision_sum += test_dec
        else:
            for i, cls_idx in enumerate(model.classes_):
                test_decision_sum[:, cls_idx] += test_dec[:, i]

# 4. Evaluation Metric
oof_accuracy = accuracy_score(y_idx, oof_preds)
print(f"5-Fold OOF Accuracy: {oof_accuracy:.5f}")

# 5. Generate Test Predictions and Save Submission
test_pred_idx = np.argmax(test_decision_sum, axis=1)
test_predictions = [idx_to_class[i] for i in test_pred_idx]

submission = pd.DataFrame({"id": np.arange(len(test_df)), "Stars": test_predictions})

# Preserve original column type if numeric in train
try:
    if pd.api.types.is_numeric_dtype(train_df[target_col].infer_objects().dtype):
        submission["Stars"] = pd.to_numeric(submission["Stars"], errors="ignore")
except Exception:
    pass

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
