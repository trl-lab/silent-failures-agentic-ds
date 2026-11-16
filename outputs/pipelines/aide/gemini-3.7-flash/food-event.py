import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import lightgbm as lgb


def main():
    # 1. Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "outcomes"
    y_raw = train_df[target_col].copy()

    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    train_df = train_df.drop(columns=[target_col])
    n_train = len(train_df)

    # Combine for uniform preprocessing
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 2. Feature Engineering & Preprocessing
    drop_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    for col in combined.columns:
        if combined[col].dtype == "object" or combined[col].dtype.name == "category":
            # Check if it looks like a datetime
            sample_val = (
                combined[col].dropna().iloc[0]
                if len(combined[col].dropna()) > 0
                else ""
            )
            try:
                parsed = pd.to_datetime(combined[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    combined[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(int)
                    combined[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(int)
                    combined[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(int)
                    combined[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(
                        -1
                    ).astype(int)
                    drop_cols.append(col)
                    continue
            except Exception:
                pass

            # Check if text or categorical
            avg_len = combined[col].astype(str).str.len().mean()
            n_unique = combined[col].nunique()
            if avg_len > 30 or n_unique > 500:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            num_cols.append(col)

    combined = combined.drop(columns=drop_cols)

    # Text features using TF-IDF
    tfidf_dfs = []
    for col in text_cols:
        combined[col] = combined[col].fillna("").astype(str)
        tfidf = TfidfVectorizer(
            max_features=50, stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_mat = tfidf.fit_transform(combined[col])
        tfidf_col_names = [f"tfidf_{col}_{i}" for i in range(tfidf_mat.shape[1])]
        tfidf_df = pd.DataFrame(
            tfidf_mat.toarray(), columns=tfidf_col_names, index=combined.index
        )
        tfidf_dfs.append(tfidf_df)

    # Frequency encode and category type for categoricals
    for col in cat_cols:
        combined[col] = combined[col].fillna("Missing").astype(str)
        freq_map = combined[col].value_counts().to_dict()
        combined[f"{col}_freq"] = combined[col].map(freq_map)
        combined[col] = combined[col].astype("category")

    if tfidf_dfs:
        combined = pd.concat([combined.drop(columns=text_cols)] + tfidf_dfs, axis=1)

    # Separate back to train and test
    X = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy()

    # Identify categorical feature column names for LightGBM
    current_cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

    # 3. Stratified 5-Fold Training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((n_train, num_classes))
    test_preds = np.zeros((len(X_test), num_classes))

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=current_cat_cols)
        val_data = lgb.Dataset(
            X_va,
            label=y_val,
            reference=train_data,
            categorical_feature=current_cat_cols,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        val_probs = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_probs
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
        )

    # 4. Evaluation
    oof_classes = np.argmax(oof_preds, axis=1)
    oof_accuracy = accuracy_score(y, oof_classes)
    print(f"Validation Accuracy: {oof_accuracy:.5f}")

    # 5. Generate Submission
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_pred_labels = le.inverse_transform(test_pred_classes)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test_pred_labels)), "outcomes": test_pred_labels}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved submission to ./working/submission.csv")


if __name__ == "__main__":
    main()
