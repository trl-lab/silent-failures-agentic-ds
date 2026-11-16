import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_text_stats(texts):
    """Extract fast stylistic, lexical, and domain-specific text features."""
    upper_re = re.compile(r"[A-Z]")
    digit_re = re.compile(r"\d")
    punc_re = re.compile(r'[!?"\'\.,;:()*&^%$#@~`+\-=/\\<>{}\[\]|]')

    toxic_words = {
        "fuck",
        "shit",
        "suck",
        "stupid",
        "idiot",
        "bitch",
        "hate",
        "trash",
        "liar",
        "kill",
        "die",
        "fake",
        "crap",
        "ugly",
        "dumb",
        "asshole",
        "jerk",
    }
    identity_words = {
        "male",
        "female",
        "gay",
        "lesbian",
        "black",
        "white",
        "muslim",
        "christian",
        "jewish",
        "transgender",
        "latino",
        "asian",
        "atheist",
    }

    texts_list = [str(t) for t in texts]

    char_lens = np.array([len(t) for t in texts_list], dtype=np.float32)
    word_counts = np.array([len(t.split()) for t in texts_list], dtype=np.float32)

    upper_counts = np.array(
        [len(upper_re.findall(t)) for t in texts_list], dtype=np.float32
    )
    digit_counts = np.array(
        [len(digit_re.findall(t)) for t in texts_list], dtype=np.float32
    )
    excl_counts = np.array([t.count("!") for t in texts_list], dtype=np.float32)
    quest_counts = np.array([t.count("?") for t in texts_list], dtype=np.float32)
    punc_counts = np.array(
        [len(punc_re.findall(t)) for t in texts_list], dtype=np.float32
    )

    unique_word_counts = np.array(
        [len(set(t.lower().split())) for t in texts_list], dtype=np.float32
    )

    toxic_word_counts = np.array(
        [
            sum(1 for w in t.lower().split() if w.strip(".,!?\"'()[]") in toxic_words)
            for t in texts_list
        ],
        dtype=np.float32,
    )

    identity_word_counts = np.array(
        [
            sum(
                1 for w in t.lower().split() if w.strip(".,!?\"'()[]") in identity_words
            )
            for t in texts_list
        ],
        dtype=np.float32,
    )

    mean_word_len = char_lens / np.maximum(word_counts, 1.0)
    upper_ratio = upper_counts / np.maximum(char_lens, 1.0)
    digit_ratio = digit_counts / np.maximum(char_lens, 1.0)
    excl_ratio = excl_counts / np.maximum(char_lens, 1.0)
    punc_ratio = punc_counts / np.maximum(char_lens, 1.0)
    lexical_diversity = unique_word_counts / np.maximum(word_counts, 1.0)
    toxic_density = toxic_word_counts / np.maximum(word_counts, 1.0)
    identity_density = identity_word_counts / np.maximum(word_counts, 1.0)

    features = np.column_stack(
        [
            char_lens,
            word_counts,
            mean_word_len,
            upper_counts,
            upper_ratio,
            digit_counts,
            digit_ratio,
            excl_counts,
            excl_ratio,
            quest_counts,
            punc_counts,
            punc_ratio,
            unique_word_counts,
            lexical_diversity,
            toxic_word_counts,
            toxic_density,
            identity_word_counts,
            identity_density,
        ]
    )

    feature_names = [
        "char_len",
        "word_count",
        "mean_word_len",
        "upper_count",
        "upper_ratio",
        "digit_count",
        "digit_ratio",
        "excl_count",
        "excl_ratio",
        "quest_count",
        "punc_count",
        "punc_ratio",
        "unique_word_count",
        "lexical_diversity",
        "toxic_word_count",
        "toxic_density",
        "identity_word_count",
        "identity_density",
    ]

    return features, feature_names


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading datasets from ./input...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df["comment_text"] = train_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Stratified Train/Val Split (85% train, 15% validation)
    stratify_target = (train_df["target"] >= 0.5).astype(int)
    tr_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=stratify_target
    )

    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train split: {len(tr_df)} rows, Val split: {len(val_df)} rows")

    # Extract Stylistic & Lexical Dense Features
    print("Extracting domain text features...")
    tr_num_feat, feat_names = extract_text_stats(tr_df["comment_text"])
    val_num_feat, _ = extract_text_stats(val_df["comment_text"])
    test_num_feat, _ = extract_text_stats(test_df["comment_text"])

    scaler = StandardScaler()
    tr_num_scaled = scaler.fit_transform(tr_num_feat)
    val_num_scaled = scaler.transform(val_num_feat)
    test_num_scaled = scaler.transform(test_num_feat)

    # Word-level TF-IDF Vectorization
    print("Fitting Word TF-IDF vectorizer on training split...")
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    tr_word_sparse = word_tfidf.fit_transform(tr_df["comment_text"])
    val_word_sparse = word_tfidf.transform(val_df["comment_text"])
    test_word_sparse = word_tfidf.transform(test_df["comment_text"])

    # Char-level TF-IDF Vectorization
    print("Fitting Char TF-IDF vectorizer on training split...")
    char_tfidf = TfidfVectorizer(
        ngram_range=(2, 5), max_features=25000, sublinear_tf=True, analyzer="char"
    )
    tr_char_sparse = char_tfidf.fit_transform(tr_df["comment_text"])
    val_char_sparse = char_tfidf.transform(val_df["comment_text"])
    test_char_sparse = char_tfidf.transform(test_df["comment_text"])

    # Combine sparse TF-IDF and dense features into CSR matrices
    print("Concatenating feature matrices...")
    X_train = hstack(
        [tr_word_sparse, tr_char_sparse, csr_matrix(tr_num_scaled)]
    ).tocsr()
    X_val = hstack(
        [val_word_sparse, val_char_sparse, csr_matrix(val_num_scaled)]
    ).tocsr()
    X_test = hstack(
        [test_word_sparse, test_char_sparse, csr_matrix(test_num_scaled)]
    ).tocsr()

    print(
        f"Final feature dimensions: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    y_train = (tr_df["target"].values >= 0.5).astype(np.int32)
    y_val = (val_df["target"].values >= 0.5).astype(np.int32)
    test_ids = test_df["id"].values

    classes = np.unique(y_train)

    # Define candidate linear model configurations with distinct objectives
    model_configs = [
        {
            "name": "SGD_log_loss_l2",
            "model": SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                random_state=42,
                learning_rate="optimal",
            ),
            "epochs": 12,
        },
        {
            "name": "SGD_log_loss_elasticnet",
            "model": SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=0.15,
                alpha=1e-5,
                random_state=101,
                learning_rate="optimal",
            ),
            "epochs": 12,
        },
        {
            "name": "SGD_modified_huber_l2",
            "model": SGDClassifier(
                loss="modified_huber",
                penalty="l2",
                alpha=2e-5,
                random_state=2024,
                learning_rate="optimal",
            ),
            "epochs": 12,
        },
    ]

    best_val_preds = []
    best_test_preds = []
    trained_models = {}

    print("Beginning iterative model training and metric tracking...")

    for config in model_configs:
        model_name = config["name"]
        clf = config["model"]
        epochs = config["epochs"]

        best_epoch_auc = -1.0
        best_val_p = None
        best_test_p = None

        print(f"\n--- Training {model_name} ---")
        for epoch in range(1, epochs + 1):
            clf.partial_fit(X_train, y_train, classes=classes)
            val_p = clf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_p)

            if val_auc > best_epoch_auc:
                best_epoch_auc = val_auc
                best_val_p = val_p.copy()
                best_test_p = clf.predict_proba(X_test)[:, 1].copy()

            print(
                f"Epoch {epoch:02d}/{epochs:02d} - Validation ROC-AUC: {val_auc:.5f} (Best: {best_epoch_auc:.5f})"
            )

        trained_models[model_name] = clf
        best_val_preds.append(best_val_p)
        best_test_preds.append(best_test_p)

    # Ensembling probability predictions across diverse loss functions
    ensemble_val_p = np.clip(np.mean(best_val_preds, axis=0), 0.0, 1.0)
    ensemble_test_p = np.clip(np.mean(best_test_preds, axis=0), 0.0, 1.0)

    final_val_score = roc_auc_score(y_val, ensemble_val_p)

    # Save trained model artifacts to disk
    best_model_path = "./working/best_model.joblib"
    joblib.dump(
        {
            "models": trained_models,
            "val_score": final_val_score,
        },
        best_model_path,
    )
    print(f"Saved best model checkpoint to {best_model_path}")

    # Generate final submission file
    submission_path = "./submission/submission.csv"
    sub_df = pd.DataFrame(
        {
            "id": test_ids.astype(np.int64),
            "target": ensemble_test_p,
        }
    )
    sub_df.to_csv(submission_path, index=False)
    print(f"Saved submission file with shape {sub_df.shape} to {submission_path}")

    # Mandatory output format for score evaluation
    print(f"Final Validation Score: {final_val_score:.5f}")


if __name__ == "__main__":
    main()