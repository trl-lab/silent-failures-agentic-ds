import os
import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack


def main():
    print("Loading data...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Identify text column
    text_col = "comment_text" if "comment_text" in train_df.columns else "text"
    if text_col not in train_df.columns:
        text_cols = [
            c
            for c in train_df.columns
            if c not in ["id", "target"] and train_df[c].dtype == object
        ]
        text_col = text_cols[0]

    print(f"Using text column: {text_col}")
    train_texts = train_df[text_col].fillna("").astype(str).values
    test_texts = test_df[text_col].fillna("").astype(str).values

    y_raw = train_df["target"].values
    y_binary = (y_raw >= 0.5).astype(int)

    print("Extracting TF-IDF word features...")
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=150000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
    )
    word_vec.fit(np.concatenate([train_texts, test_texts]))
    X_train_word = word_vec.transform(train_texts)
    X_test_word = word_vec.transform(test_texts)

    print("Extracting TF-IDF char features...")
    char_vec = TfidfVectorizer(
        ngram_range=(2, 6),
        min_df=10,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="char",
    )
    char_vec.fit(np.concatenate([train_texts, test_texts]))
    X_train_char = char_vec.transform(train_texts)
    X_test_char = char_vec.transform(test_texts)

    X_train = hstack([X_train_word, X_train_char], format="csr")
    X_test = hstack([X_test_word, X_test_char], format="csr")

    del (
        word_vec,
        char_vec,
        X_train_word,
        X_test_word,
        X_train_char,
        X_test_char,
        train_texts,
        test_texts,
    )
    gc.collect()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)

    print(f"Starting {n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_binary)):
        X_tr, y_tr = X_train[train_idx], y_binary[train_idx]
        X_val, y_val = X_train[val_idx], y_binary[val_idx]

        model = LogisticRegression(
            C=2.0, max_iter=200, solver="saga", n_jobs=-1, random_state=42 + fold
        )
        model.fit(X_tr, y_tr)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred

        fold_auc = roc_auc_score(y_val, val_pred)
        print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    overall_auc = roc_auc_score(y_binary, oof_preds)
    print(f"\nOverall Out-of-Fold ROC AUC: {overall_auc:.5f}")

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_df)), "target": test_preds})
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Submission saved to ./working/submission.csv with shape {submission.shape}")


if __name__ == "__main__":
    main()
