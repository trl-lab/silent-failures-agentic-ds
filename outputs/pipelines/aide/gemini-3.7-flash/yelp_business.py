import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def main():
    print("Loading datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "stars"
    y_raw = train_df[target_col].values

    # Label encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
    n_classes = len(le_target.classes_)
    print(f"Number of classes: {n_classes} -> {le_target.classes_}")

    # Combine train and test for consistent feature extraction
    n_train = len(train_df)
    n_test = len(test_df)
    df = pd.concat(
        [train_df.drop(columns=[target_col], errors="ignore"), test_df],
        ignore_index=True,
    )

    feature_dfs = []

    # 1. Numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "id" in num_cols:
        num_cols.remove("id")
    print(f"Numeric columns: {num_cols}")
    if num_cols:
        df_num = df[num_cols].copy()
        # Add log transforms for skewed features like review_count
        for col in num_cols:
            if "count" in col.lower() or "review" in col.lower():
                df_num[f"{col}_log1p"] = np.log1p(np.maximum(0, df_num[col].fillna(0)))
        feature_dfs.append(df_num.fillna(-999))

    # 2. Categorical features (low/medium cardinality)
    cat_cols = ["city", "state", "postal_code", "is_open"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in cat_cols:
        s = df[c].astype(str).fillna("missing")
        # Frequency encoding
        freq = s.value_counts(normalize=True)
        feature_dfs.append(
            pd.DataFrame(
                {
                    f"{c}_freq": s.map(freq).fillna(0),
                    f"{c}_le": LabelEncoder().fit_transform(s),
                }
            )
        )

    # 3. Text & Bag-of-features: 'categories', 'attributes', 'hours', 'name'
    text_cols = ["categories", "attributes", "hours", "name"]
    text_cols = [c for c in text_cols if c in df.columns]

    for c in text_cols:
        print(f"Processing text column: {c}")
        s = df[c].fillna("").astype(str)

        # Length features
        length_df = pd.DataFrame(
            {
                f"{c}_char_len": s.str.len(),
                f"{c}_word_count": s.apply(lambda x: len(x.split())),
                f"{c}_item_count": s.apply(
                    lambda x: len(re.split(r"[,;:]+", x)) if x else 0
                ),
            }
        )
        feature_dfs.append(length_df)

        # TF-IDF + TruncatedSVD
        n_comp = 30 if c in ["categories", "attributes"] else 15
        tfidf = TfidfVectorizer(
            max_features=5000, token_pattern=r"(?u)\b\w+\b", lowercase=True
        )
        tfidf_mat = tfidf.fit_transform(s)

        if tfidf_mat.shape[1] > 0:
            svd_comp = min(n_comp, tfidf_mat.shape[1] - 1)
            if svd_comp > 0:
                svd = TruncatedSVD(n_components=svd_comp, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                svd_df = pd.DataFrame(
                    svd_feats, columns=[f"{c}_svd_{i}" for i in range(svd_comp)]
                )
                feature_dfs.append(svd_df)

    # Combine all features
    X_all = pd.concat(feature_dfs, axis=1)
    X = X_all.iloc[:n_train].reset_index(drop=True)
    X_test = X_all.iloc[n_train:].reset_index(drop=True)

    print(f"Total features created: {X.shape[1]}")

    # 5-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((n_train, n_classes), dtype=np.float32)
    test_preds = np.zeros((n_test, n_classes), dtype=np.float32)

    lgb_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    print("\nStarting Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
        )

        fold_acc = accuracy_score(y_val, np.argmax(val_pred, axis=1))
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.5f}")

    oof_labels = np.argmax(oof_preds, axis=1)
    overall_acc = accuracy_score(y, oof_labels)
    print(f"\n==========================================")
    print(f"Overall OOF Accuracy: {overall_acc:.5f}")
    print(f"==========================================")

    # Generate test predictions
    test_labels_encoded = np.argmax(test_preds, axis=1)
    test_stars = le_target.inverse_transform(test_labels_encoded)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_stars)), "stars": test_stars})
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Saved submission to ./working/submission.csv with shape {submission.shape}")


if __name__ == "__main__":
    main()
