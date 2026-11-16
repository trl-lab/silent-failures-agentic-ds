import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb

warnings.filterwarnings("ignore")


def main():
    # 1. Load Datasets
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Decision type"
    y_raw = train_df[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    train_features = train_df.drop(columns=[target_col]).copy()
    test_features = test_df.copy()

    n_train = len(train_features)
    n_test = len(test_features)

    # 2. Identify Column Types
    date_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    for col in train_features.columns:
        if pd.api.types.is_datetime64_any_dtype(train_features[col]):
            date_cols.append(col)
        elif train_features[col].dtype == "object" or isinstance(
            train_features[col].dtype, pd.CategoricalDtype
        ):
            sample_vals = train_features[col].dropna().astype(str).tolist()
            if not sample_vals:
                cat_cols.append(col)
                continue
            try:
                parsed = pd.to_datetime(
                    train_features[col].dropna().head(100), errors="coerce"
                )
                if parsed.notna().mean() > 0.8:
                    date_cols.append(col)
                    continue
            except Exception:
                pass

            avg_len = np.mean([len(s) for s in sample_vals[:100]])
            if avg_len > 30:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            num_cols.append(col)

    # 3. Process Date Features
    def extract_date_features(df, cols):
        date_feats = pd.DataFrame(index=df.index)
        for col in cols:
            dt = pd.to_datetime(df[col], errors="coerce")
            date_feats[f"{col}_year"] = dt.dt.year.fillna(-1)
            date_feats[f"{col}_month"] = dt.dt.month.fillna(-1)
            date_feats[f"{col}_day"] = dt.dt.day.fillna(-1)
            date_feats[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1)
        return date_feats

    train_date_df = extract_date_features(train_features, date_cols)
    test_date_df = extract_date_features(test_features, date_cols)

    # 4. Process Categorical Features
    train_cat_df = pd.DataFrame(index=train_features.index)
    test_cat_df = pd.DataFrame(index=test_features.index)

    for col in cat_cols:
        tr_s = train_features[col].astype(str).fillna("missing")
        te_s = test_features[col].astype(str).fillna("missing")

        freq = tr_s.value_counts(normalize=True).to_dict()
        train_cat_df[f"{col}_freq"] = tr_s.map(freq).fillna(0)
        test_cat_df[f"{col}_freq"] = te_s.map(freq).fillna(0)

        cat_le = LabelEncoder()
        combined = pd.concat([tr_s, te_s], axis=0)
        cat_le.fit(combined)
        train_cat_df[f"{col}_ord"] = cat_le.transform(tr_s)
        test_cat_df[f"{col}_ord"] = cat_le.transform(te_s)

    # 5. Process Numeric Features
    train_num_df = train_features[num_cols].fillna(-999).copy()
    test_num_df = test_features[num_cols].fillna(-999).copy()

    # 6. Process Text Features
    all_str_cols = text_cols + cat_cols
    if not all_str_cols:
        all_str_cols = train_features.columns.tolist()

    train_text = (
        train_features[all_str_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    test_text = test_features[all_str_cols].fillna("").astype(str).agg(" ".join, axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=30000, sublinear_tf=True
    )
    X_text_word_tr = word_vec.fit_transform(train_text)
    X_text_word_te = word_vec.transform(test_text)

    char_vec = TfidfVectorizer(
        ngram_range=(2, 4), analyzer="char", max_features=20000, sublinear_tf=True
    )
    X_text_char_tr = char_vec.fit_transform(train_text)
    X_text_char_te = char_vec.transform(test_text)

    svd_word = TruncatedSVD(n_components=30, random_state=42)
    word_svd_tr = svd_word.fit_transform(X_text_word_tr)
    word_svd_te = svd_word.transform(X_text_word_te)

    svd_char = TruncatedSVD(n_components=20, random_state=42)
    char_svd_tr = svd_char.fit_transform(X_text_char_tr)
    char_svd_te = svd_char.transform(X_text_char_te)

    oof_lr_meta = np.zeros((n_train, num_classes))
    test_lr_meta = np.zeros((n_test, num_classes))
    oof_nb_meta = np.zeros((n_train, num_classes))
    test_nb_meta = np.zeros((n_test, num_classes))

    # Safe class assignment for fold meta-learners
    for tr_idx, val_idx in skf.split(train_features, y):
        X_tr_fold, y_tr_fold = X_text_word_tr[tr_idx], y[tr_idx]
        X_val_fold = X_text_word_tr[val_idx]

        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
        lr.fit(X_tr_fold, y_tr_fold)
        lr_val_p = lr.predict_proba(X_val_fold)
        for idx_c, cls in enumerate(lr.classes_):
            oof_lr_meta[val_idx, cls] = lr_val_p[:, idx_c]

        lr_test_p = lr.predict_proba(X_text_word_te)
        for idx_c, cls in enumerate(lr.classes_):
            test_lr_meta[:, cls] += lr_test_p[:, idx_c] / 5.0

        nb = MultinomialNB(alpha=0.5)
        nb.fit(X_tr_fold, y_tr_fold)
        nb_val_p = nb.predict_proba(X_val_fold)
        for idx_c, cls in enumerate(nb.classes_):
            oof_nb_meta[val_idx, cls] = nb_val_p[:, idx_c]

        nb_test_p = nb.predict_proba(X_text_word_te)
        for idx_c, cls in enumerate(nb.classes_):
            test_nb_meta[:, cls] += nb_test_p[:, idx_c] / 5.0

    lr_meta_cols = [f"lr_meta_class_{i}" for i in range(num_classes)]
    nb_meta_cols = [f"nb_meta_class_{i}" for i in range(num_classes)]
    svd_word_cols = [f"svd_word_{i}" for i in range(30)]
    svd_char_cols = [f"svd_char_{i}" for i in range(20)]

    train_meta_df = pd.DataFrame(
        np.hstack([oof_lr_meta, oof_nb_meta, word_svd_tr, char_svd_tr]),
        columns=lr_meta_cols + nb_meta_cols + svd_word_cols + svd_char_cols,
        index=train_features.index,
    )
    test_meta_df = pd.DataFrame(
        np.hstack([test_lr_meta, test_nb_meta, word_svd_te, char_svd_te]),
        columns=lr_meta_cols + nb_meta_cols + svd_word_cols + svd_char_cols,
        index=test_features.index,
    )

    X_full_train = pd.concat(
        [train_num_df, train_cat_df, train_date_df, train_meta_df], axis=1
    )
    X_full_test = pd.concat(
        [test_num_df, test_cat_df, test_date_df, test_meta_df], axis=1
    )

    # Clean column names for LightGBM
    X_full_train.columns = [str(c).replace(" ", "_") for c in X_full_train.columns]
    X_full_test.columns = [str(c).replace(" ", "_") for c in X_full_test.columns]

    # 7. Model Training (LightGBM Cross-Validation)
    oof_probs = np.zeros((n_train, num_classes))
    test_probs = np.zeros((n_test, num_classes))

    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42,
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_train, y)):
        X_tr, y_tr = X_full_train.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X_full_train.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        oof_probs[val_idx] = model.predict(X_val)
        test_probs += model.predict(X_full_test) / 5.0

    oof_preds = np.argmax(oof_probs, axis=1)
    val_accuracy = accuracy_score(y, oof_preds)
    print(f"Validation Accuracy: {val_accuracy:.5f}")

    # 8. Save Test Predictions
    test_preds_encoded = np.argmax(test_probs, axis=1)
    test_preds_labels = le.inverse_transform(test_preds_encoded)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(n_test), "Decision type": test_preds_labels}
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
