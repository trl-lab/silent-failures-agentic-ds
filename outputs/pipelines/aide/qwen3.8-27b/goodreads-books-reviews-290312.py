import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

SEED = 42
np.random.seed(SEED)

TARGET = "rating"
DATE_COLS = ["date_added", "date_updated", "read_at", "started_at"]


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-free base features from raw columns."""
    X = pd.DataFrame(index=df.index)

    # Numeric review metadata
    n_comments = df["n_comments"].astype("float32")
    n_votes = df["n_votes"].astype("float32")
    X["n_comments"] = n_comments
    X["n_votes"] = n_votes
    X["n_comments_log"] = np.log1p(n_comments.clip(lower=0))
    X["n_votes_log"] = np.log1p(n_votes.clip(lower=0))

    # Date features
    parsed_dates = {}
    for col in DATE_COLS:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", utc=True)
        else:
            dt = pd.Series(pd.NaT, index=df.index)
        parsed_dates[col] = dt

        X[f"{col}_year"] = dt.dt.year.astype("float32")
        X[f"{col}_month"] = dt.dt.month.astype("float32")
        X[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float32")
        X[f"{col}_hour"] = dt.dt.hour.astype("float32")

    # Time differences in days
    def days_between(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a - b).dt.days.astype("float32")

    X["updated_added_days"] = days_between(
        parsed_dates["date_updated"], parsed_dates["date_added"]
    )
    X["read_added_days"] = days_between(
        parsed_dates["read_at"], parsed_dates["date_added"]
    )
    X["started_added_days"] = days_between(
        parsed_dates["started_at"], parsed_dates["date_added"]
    )
    X["read_started_days"] = days_between(
        parsed_dates["read_at"], parsed_dates["started_at"]
    )

    # Simple text features
    if "review_text" in df.columns:
        text = df["review_text"].fillna("").astype(str)
    else:
        text = pd.Series("", index=df.index)

    X["text_len"] = text.str.len().astype("float32")
    X["text_words"] = (text.str.count(" ") + 1).astype("float32")
    X["has_review_to_come"] = (
        text.str.lower().str.contains("review to come").astype("float32")
    )
    X["n_exclamation"] = text.str.count("!").astype("float32")
    X["n_question"] = text.str.count("?").astype("float32")

    return X


def make_stats(ids: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """Compute smoothed target-encoding statistics from a masked subset."""
    sub_ids = ids[mask]
    sub_y = y[mask]
    global_mean = float(np.mean(sub_y)) if len(sub_y) > 0 else 3.0

    s = pd.Series(sub_y, index=sub_ids)
    stats = s.groupby(level=0).agg(["mean", "count"])
    return stats, global_mean


def encode_ids(
    ids: np.ndarray,
    stats: pd.DataFrame,
    global_mean: float,
    prior: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply smoothed target encoding and return encoded values plus counts."""
    if len(stats) == 0:
        return np.full(len(ids), global_mean, dtype="float32"), np.zeros(
            len(ids), dtype="float32"
        )

    smoothed = (stats["count"] * stats["mean"] + prior * global_mean) / (
        stats["count"] + prior
    )
    encoded = (
        pd.Series(ids).map(smoothed).fillna(global_mean).astype("float32").to_numpy()
    )
    counts = pd.Series(ids).map(stats["count"]).fillna(0).astype("float32").to_numpy()
    return encoded, counts


def add_target_features(
    X: pd.DataFrame,
    user_ids: np.ndarray,
    book_ids: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Add target-encoded user/book features computed only from rows in mask."""
    X_out = X.copy()
    user_stats, user_global = make_stats(user_ids, y, mask)
    book_stats, book_global = make_stats(book_ids, y, mask)

    X_out["user_target_enc"], X_out["user_count"] = encode_ids(
        user_ids, user_stats, user_global
    )
    X_out["book_target_enc"], X_out["book_count"] = encode_ids(
        book_ids, book_stats, book_global
    )
    return X_out


def add_safe_cv_target_features(
    X: pd.DataFrame,
    user_ids: np.ndarray,
    book_ids: np.ndarray,
    y: np.ndarray,
    outer_train_mask: np.ndarray,
    outer_val_mask: np.ndarray,
    inner_kfold: KFold,
) -> pd.DataFrame:
    """
    Add leakage-safe target features for one outer CV fold.
    Validation rows use stats from the outer training set.
    Training rows use inner out-of-fold stats from the outer training set only.
    """
    X_out = X.copy()

    # Validation fold: safe because it is excluded from the stats.
    val_user_stats, val_user_global = make_stats(user_ids, y, outer_train_mask)
    val_book_stats, val_book_global = make_stats(book_ids, y, outer_train_mask)

    (
        X_out.loc[outer_val_mask, "user_target_enc"],
        X_out.loc[outer_val_mask, "user_count"],
    ) = encode_ids(user_ids[outer_val_mask], val_user_stats, val_user_global)
    (
        X_out.loc[outer_val_mask, "book_target_enc"],
        X_out.loc[outer_val_mask, "book_count"],
    ) = encode_ids(book_ids[outer_val_mask], val_book_stats, val_book_global)

    # Training rows: inner out-of-fold target encoding within the outer training set.
    outer_train_idx = np.where(outer_train_mask)[0]
    inner_folds = np.empty(len(outer_train_idx), dtype=int)

    for inner_train_idx, inner_val_idx in inner_kfold.split(outer_train_idx):
        inner_val_global_idx = outer_train_idx[inner_val_idx]
        inner_train_mask = np.zeros(len(y), dtype=bool)
        inner_train_mask[outer_train_idx[inner_train_idx]] = True

        inner_user_stats, inner_user_global = make_stats(user_ids, y, inner_train_mask)
        inner_book_stats, inner_book_global = make_stats(book_ids, y, inner_train_mask)

        user_enc, user_count = encode_ids(
            user_ids[inner_val_global_idx], inner_user_stats, inner_user_global
        )
        book_enc, book_count = encode_ids(
            book_ids[inner_val_global_idx], inner_book_stats, inner_book_global
        )

        X_out.loc[inner_val_global_idx, "user_target_enc"] = user_enc
        X_out.loc[inner_val_global_idx, "user_count"] = user_count
        X_out.loc[inner_val_global_idx, "book_target_enc"] = book_enc
        X_out.loc[inner_val_global_idx, "book_count"] = book_count

    return X_out


def train_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray
):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": SEED,
    }

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=True)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def main():
    train = pd.read_parquet("./input/train.parquet")
    y = train[TARGET].astype("float32").to_numpy()
    user_ids = train["user_id"].astype(str).to_numpy()
    book_ids = train["book_id"].astype(str).to_numpy()

    X_base = engineer_base_features(train)
    feature_cols = X_base.columns.tolist()

    # 5-fold cross-validation with leakage-safe target encoding.
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(X_base), start=1):
        outer_train_mask = np.zeros(len(y), dtype=bool)
        outer_val_mask = np.zeros(len(y), dtype=bool)
        outer_train_mask[train_idx] = True
        outer_val_mask[val_idx] = True

        X_fold = add_safe_cv_target_features(
            X_base,
            user_ids,
            book_ids,
            y,
            outer_train_mask,
            outer_val_mask,
            kf_inner,
        )

        X_train = X_fold.loc[
            train_idx,
            feature_cols
            + ["user_target_enc", "user_count", "book_target_enc", "book_count"],
        ]
        X_val = X_fold.loc[
            val_idx,
            feature_cols
            + ["user_target_enc", "user_count", "book_target_enc", "book_count"],
        ]

        model = train_lgbm(X_train, y[train_idx], X_val, y[val_idx])
        pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = float(np.sqrt(mean_squared_error(y[val_idx], pred)))
        fold_rmses.append(rmse)
        print(f"Fold {fold} RMSE: {rmse:.6f}")

    cv_rmse = float(np.mean(fold_rmses))
    print(f"Mean 5-fold CV RMSE: {cv_rmse:.6f}")

    # Refit on the full training set using full-data target encodings.
    full_mask = np.ones(len(y), dtype=bool)
    X_full = add_target_features(X_base, user_ids, book_ids, y, full_mask)
    final_model = train_lgbm(X_full, y, X_full.iloc[:0], y[:0]) if False else None

    # Train final model with a tiny internal validation set only to obtain best_iteration?
    # To keep it simple and avoid extra leakage, train for the average best iteration from CV.
    # We can also train with early stopping on a small holdout, but the CV estimate is already reported.
    # Here we train for a fixed number of rounds equal to the average best iteration from CV.
    # Since we did not store best_iteration, use a conservative fixed round count.
    final_rounds = 1000
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": SEED,
    }
    train_set = lgb.Dataset(X_full, label=y, free_raw_data=True)
    final_model = lgb.train(
        params,
        train_set,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    # Predict test set
    test = pd.read_parquet("./input/test.parquet")
    test_user_ids = (
        test["user_id"].astype(str).to_numpy()
        if "user_id" in test.columns
        else np.array([""] * len(test))
    )
    test_book_ids = (
        test["book_id"].astype(str).to_numpy()
        if "book_id" in test.columns
        else np.array([""] * len(test))
    )

    X_test_base = engineer_base_features(test)
    X_test = add_target_features(
        X_test_base, test_user_ids, test_book_ids, y, full_mask
    )

    final_feature_cols = feature_cols + [
        "user_target_enc",
        "user_count",
        "book_target_enc",
        "book_count",
    ]
    test_pred = final_model.predict(
        X_test[final_feature_cols], num_iteration=final_rounds
    )

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "rating": test_pred.astype("float64"),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
