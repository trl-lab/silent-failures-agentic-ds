import gc
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings("ignore")


def parse_dates(df):
    for col in ["date_added", "date_updated", "read_at", "started_at"]:
        if col in df.columns:
            df[col + "_dt"] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col + "_year"] = df[col + "_dt"].dt.year.fillna(-1).astype("int16")
            df[col + "_month"] = df[col + "_dt"].dt.month.fillna(-1).astype("int8")
            df[col + "_dayofweek"] = (
                df[col + "_dt"].dt.dayofweek.fillna(-1).astype("int8")
            )
            df.drop(columns=[col], inplace=True)

    if "date_added_dt" in df.columns and "date_updated_dt" in df.columns:
        df["days_to_update"] = (
            df["date_updated_dt"] - df["date_added_dt"]
        ).dt.total_seconds() / 86400.0
    if "started_at_dt" in df.columns and "read_at_dt" in df.columns:
        df["reading_duration_days"] = (
            df["read_at_dt"] - df["started_at_dt"]
        ).dt.total_seconds() / 86400.0

    dt_cols = [c for c in df.columns if c.endswith("_dt")]
    df.drop(columns=dt_cols, inplace=True)
    return df


def extract_text_stats(series):
    texts = series.fillna("").astype(str)
    char_len = texts.str.len()
    word_count = texts.str.split().str.len()
    exclamations = texts.str.count("!")
    questions = texts.str.count(r"\?")
    uppercase = texts.apply(lambda s: sum(1 for c in s if c.isupper()))
    upper_ratio = uppercase / (char_len + 1)

    return pd.DataFrame(
        {
            "text_char_len": char_len.astype(np.int32),
            "text_word_count": word_count.astype(np.int32),
            "text_exclamation_cnt": exclamations.astype(np.int16),
            "text_question_cnt": questions.astype(np.int16),
            "text_upper_ratio": upper_ratio.astype(np.float32),
        }
    )


def main():
    print("Loading datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    y = train_df["rating"].values.astype(np.float32)
    n_train = len(train_df)
    n_test = len(test_df)

    global_mean = float(y.mean())

    # Text processing: TF-IDF + Ridge OOF features
    print("Extracting TF-IDF text features and Ridge out-of-fold meta-features...")
    train_text = train_df["review_text"].fillna("").astype(str)
    test_text = test_df["review_text"].fillna("").astype(str)

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_text_train = tfidf.fit_transform(train_text)
    X_text_test = tfidf.transform(test_text)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_text_pred = np.zeros(n_train, dtype=np.float32)
    test_text_pred = np.zeros(n_test, dtype=np.float32)

    for train_idx, val_idx in kf.split(train_df):
        ridge = Ridge(alpha=5.0, solver="auto", random_state=42)
        ridge.fit(X_text_train[train_idx], y[train_idx])
        oof_text_pred[val_idx] = ridge.predict(X_text_train[val_idx])
        test_text_pred += ridge.predict(X_text_test) / kf.n_splits

    del X_text_train, X_text_test, tfidf
    gc.collect()

    print("Engineering tabular and count features...")
    train_stats = extract_text_stats(train_df["review_text"])
    test_stats = extract_text_stats(test_df["review_text"])

    train_df["ridge_text_pred"] = oof_text_pred
    test_df["ridge_text_pred"] = test_text_pred

    for col in train_stats.columns:
        train_df[col] = train_stats[col]
        test_df[col] = test_stats[col]

    del train_stats, test_stats
    gc.collect()

    train_df = parse_dates(train_df)
    test_df = parse_dates(test_df)

    # Frequency encoding
    for cat_col in ["user_id", "book_id"]:
        counts = pd.concat([train_df[cat_col], test_df[cat_col]], axis=0).value_counts()
        train_df[f"{cat_col}_count"] = (
            train_df[cat_col].map(counts).fillna(0).astype(np.int32)
        )
        test_df[f"{cat_col}_count"] = (
            test_df[cat_col].map(counts).fillna(0).astype(np.int32)
        )

    # Target encoding with smoothing (OOF for train, full for test)
    smooth_weight = 10.0
    for cat_col in ["user_id", "book_id"]:
        oof_te = np.full(n_train, global_mean, dtype=np.float32)
        for train_idx, val_idx in kf.split(train_df):
            tr_subset = train_df.iloc[train_idx]
            stats = tr_subset.groupby(cat_col)["rating"].agg(["mean", "count"])
            smoothed = (
                stats["count"] * stats["mean"] + smooth_weight * global_mean
            ) / (stats["count"] + smooth_weight)
            oof_te[val_idx] = (
                train_df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean).values
            )
        train_df[f"{cat_col}_te"] = oof_te

        full_stats = train_df.groupby(cat_col)["rating"].agg(["mean", "count"])
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + smooth_weight * global_mean
        ) / (full_stats["count"] + smooth_weight)
        test_df[f"{cat_col}_te"] = (
            test_df[cat_col]
            .map(full_smoothed)
            .fillna(global_mean)
            .values.astype(np.float32)
        )

    # Numerical features cleaning
    for col in ["n_comments", "n_votes"]:
        train_df[col] = np.log1p(np.maximum(train_df[col], 0))
        test_df[col] = np.log1p(np.maximum(test_df[col], 0))

    drop_cols = ["review_id", "review_text", "user_id", "book_id", "rating"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 63,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 30,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    print("Training LightGBM models with 5-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        )

        fold_rmse = root_mean_squared_error(y_val, val_pred)
        print(f"Fold {fold} RMSE: {fold_rmse:.5f}")

    oof_preds = np.clip(oof_preds, 0.0, 5.0)
    test_preds = np.clip(test_preds, 0.0, 5.0)

    cv_rmse = root_mean_squared_error(y, oof_preds)
    print(f"\n==========================================")
    print(f"Overall Out-of-Fold 5-Fold CV RMSE: {cv_rmse:.5f}")
    print(f"==========================================")

    submission = pd.DataFrame(
        {"id": np.arange(n_test, dtype=np.int64), "rating": test_preds}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(
        f"Submission saved successfully with shape {submission.shape} to ./working/submission.csv"
    )


if __name__ == "__main__":
    main()
