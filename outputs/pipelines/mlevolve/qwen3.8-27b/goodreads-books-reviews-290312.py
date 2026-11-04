import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import json
import re
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING & FEATURE ENGINEERING
# ============================================================

# Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"\nRating distribution:\n{train_df['rating'].value_counts().sort_index()}")


# Parse date columns
def parse_dates(df):
    """Parse date columns and extract temporal features."""
    date_cols = ["date_added", "date_updated", "read_at", "started_at"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


print("\nParsing dates...")
train_df = parse_dates(train_df)
test_df = parse_dates(test_df)


# Feature engineering function
def engineer_features(
    df, is_train=True, user_stats=None, book_stats=None, global_mean_rating=None
):
    """Engineer features from raw data."""
    feat = pd.DataFrame(index=df.index)

    # === Basic features ===
    feat["n_comments"] = df["n_comments"].clip(lower=0).fillna(0)
    feat["n_votes"] = df["n_votes"].clip(lower=0).fillna(0)
    feat["log_n_comments"] = np.log1p(feat["n_comments"])
    feat["log_n_votes"] = np.log1p(feat["n_votes"])
    feat["comments_votes_ratio"] = feat["n_comments"] / (feat["n_votes"] + 1)

    # === Temporal features ===
    if "started_at" in df.columns and "read_at" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["started_at"]) and pd.api.types.is_datetime64_any_dtype(df["read_at"]):
            reading_duration = (df["read_at"] - df["started_at"]).dt.total_seconds()
            feat["reading_duration_days"] = reading_duration / 86400.0
            feat["reading_duration_days"] = feat["reading_duration_days"].clip(upper=3650)
            feat["log_reading_duration"] = np.log1p(
                feat["reading_duration_days"].clip(lower=0)
            )
            feat["has_reading_duration"] = (
                df["started_at"].notna() & df["read_at"].notna()
            ).astype(int)
        else:
            feat["reading_duration_days"] = 0.0
            feat["log_reading_duration"] = 0.0
            feat["has_reading_duration"] = 0

    if "date_added" in df.columns and "read_at" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["date_added"]) and pd.api.types.is_datetime64_any_dtype(df["read_at"]):
            time_to_read = (df["read_at"] - df["date_added"]).dt.total_seconds()
            feat["time_to_read_days"] = time_to_read / 86400.0
            feat["time_to_read_days"] = feat["time_to_read_days"].clip(upper=3650)
            feat["log_time_to_read"] = np.log1p(feat["time_to_read_days"].clip(lower=0))
        else:
            feat["time_to_read_days"] = 0.0
            feat["log_time_to_read"] = 0.0

    if "date_added" in df.columns and "date_updated" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["date_added"]) and pd.api.types.is_datetime64_any_dtype(df["date_updated"]):
            time_to_update = (df["date_updated"] - df["date_added"]).dt.total_seconds()
            feat["time_to_update_days"] = time_to_update / 86400.0
            feat["time_to_update_days"] = feat["time_to_update_days"].clip(upper=3650)
            feat["log_time_to_update"] = np.log1p(feat["time_to_update_days"].clip(lower=0))
        else:
            feat["time_to_update_days"] = 0.0
            feat["log_time_to_update"] = 0.0

    # Year, month, day of week features
    for col in ["date_added", "date_updated", "read_at", "started_at"]:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            feat[f"{col}_year"] = df[col].dt.year.fillna(0)
            feat[f"{col}_month"] = df[col].dt.month.fillna(0)
            feat[f"{col}_dow"] = df[col].dt.dayofweek.fillna(0)
        else:
            feat[f"{col}_year"] = 0
            feat[f"{col}_month"] = 0
            feat[f"{col}_dow"] = 0

    # === Text features ===
    if "review_text" in df.columns:
        text = df["review_text"].fillna("")
        feat["text_length"] = text.str.len()
        feat["log_text_length"] = np.log1p(feat["text_length"])
        feat["word_count"] = text.str.split().str.len().fillna(0)
        feat["avg_word_length"] = feat["text_length"] / (feat["word_count"] + 1)
        feat["has_text"] = (text.str.len() > 0).astype(int)

        # NEW: Star/rating mentions in text
        star_pattern = r"(\d+)\s*(?:stars?|/5|out of 5)"
        star_matches = text.str.findall(star_pattern, flags=re.IGNORECASE)
        feat["star_mention_count"] = star_matches.apply(len)
        feat["star_mention_value"] = star_matches.apply(
            lambda x: float(x[0]) if x else np.nan
        )

        # NEW: Review length relative to user's average
        if user_stats is not None and "avg_text_length" in user_stats.columns:
            user_avg_len = df["user_id"].map(user_stats["avg_text_length"])
            feat["text_length_vs_user_avg"] = feat["text_length"] / (user_avg_len + 1)
            feat["text_length_vs_user_avg"] = feat["text_length_vs_user_avg"].fillna(1.0)
        else:
            feat["text_length_vs_user_avg"] = 1.0

        # DNF indicators
        dnf_patterns = [
            "did not finish",
            "dnf",
            "abandoned",
            "gave up",
            "stopped reading",
            "put down",
            "put it down",
            "could not finish",
            "unable to finish",
            "dropped",
            "quit",
            "stopped at",
        ]
        dnf_text = text.str.lower()
        feat["is_dnf_text"] = dnf_text.str.contains(
            "|".join(dnf_patterns), na=False
        ).astype(int)

        # Positive sentiment indicators
        positive_words = [
            "amazing",
            "great",
            "love",
            "loved",
            "wonderful",
            "excellent",
            "fantastic",
            "brilliant",
            "outstanding",
            "best",
            "perfect",
            "enjoyed",
            "enjoy",
            "favorite",
            "favourite",
            "recommend",
            "captivating",
            "page-turner",
            "masterpiece",
            "incredible",
        ]
        feat["positive_word_count"] = dnf_text.str.contains(
            "|".join(positive_words), na=False
        ).astype(int)

        # Negative sentiment indicators
        negative_words = [
            "terrible",
            "awful",
            "horrible",
            "boring",
            "waste",
            "poor",
            "bad",
            "worst",
            "disappointing",
            "disappointed",
            "unbearable",
            "dull",
            "tedious",
            "overrated",
            "mediocre",
            "predictable",
            "cliché",
            "cliche",
            "weak",
            "flat",
        ]
        feat["negative_word_count"] = dnf_text.str.contains(
            "|".join(negative_words), na=False
        ).astype(int)

        # Punctuation features
        feat["exclamation_count"] = text.str.count("!")
        feat["question_count"] = text.str.count(r"\?")
        feat["sentence_count"] = text.str.count(r"[.!?]+").fillna(0)

        # Generic review detection
        generic_reviews = [
            "review to come",
            "review to come.",
            "review to come!",
            "review to come!",
        ]
        feat["is_generic_review"] = (
            text.str.lower().str.strip().isin(generic_reviews).astype(int)
        )

        # Paragraph count
        feat["paragraph_count"] = text.str.count(r"\n\s*\n").fillna(0) + 1

        # Spoiler and recommendation mentions
        feat["has_spoiler_mention"] = (
            text.str.lower().str.contains("spoiler", na=False).astype(int)
        )
        feat["has_recommendation"] = (
            text.str.lower().str.contains("recommend", na=False).astype(int)
        )

        # Capital letters ratio
        upper_count = text.str.count(r"[A-Z]")
        feat["upper_ratio"] = upper_count / (feat["text_length"] + 1)

        # Numbers in text
        feat["number_count"] = text.str.count(r"\d")

        # NEW: Exclamation density (computed here since we have word_count)
        feat["exclamation_density"] = feat["exclamation_count"] / (feat["word_count"] + 1)

    # === Book ID features ===
    if "book_id" in df.columns:
        feat["book_id"] = df["book_id"].astype("int64")

    # === User ID features ===
    if "user_id" in df.columns:
        feat["user_id"] = df["user_id"].astype("category")

    # === User-level aggregate features ===
    if user_stats is not None:
        feat["user_mean_rating"] = df["user_id"].map(user_stats["mean_rating"])
        feat["user_std_rating"] = df["user_id"].map(user_stats["std_rating"])
        feat["user_count"] = df["user_id"].map(user_stats["count"])
        feat["user_dnf_rate"] = df["user_id"].map(user_stats["dnf_rate"])
        feat["user_log_count"] = np.log1p(feat["user_count"].fillna(0))

        # Fill NaN for unseen users (safe for both train and test)
        feat["user_mean_rating"] = feat["user_mean_rating"].fillna(
            global_mean_rating if global_mean_rating else 3.0
        )
        feat["user_std_rating"] = feat["user_std_rating"].fillna(1.0)
        feat["user_count"] = feat["user_count"].fillna(0)
        feat["user_dnf_rate"] = feat["user_dnf_rate"].fillna(0.1)

    # === Book-level aggregate features ===
    if book_stats is not None:
        feat["book_mean_rating"] = df["book_id"].map(book_stats["mean_rating"])
        feat["book_std_rating"] = df["book_id"].map(book_stats["std_rating"])
        feat["book_count"] = df["book_id"].map(book_stats["count"])
        feat["book_dnf_rate"] = df["book_id"].map(book_stats["dnf_rate"])
        feat["book_log_count"] = np.log1p(feat["book_count"].fillna(0))

        # Fill NaN for unseen books (safe for both train and test)
        feat["book_mean_rating"] = feat["book_mean_rating"].fillna(
            global_mean_rating if global_mean_rating else 3.0
        )
        feat["book_std_rating"] = feat["book_std_rating"].fillna(1.0)
        feat["book_count"] = feat["book_count"].fillna(0)
        feat["book_dnf_rate"] = feat["book_dnf_rate"].fillna(0.1)

    # === NEW: Bayesian-smoothed target encodings ===
    if user_stats is not None and book_stats is not None:
        gm = global_mean_rating if global_mean_rating else 3.0

        # User Bayesian smoothing: smoothed = (count * mean + m * global_mean) / (count + m)
        m_user = 10  # smoothing parameter for users
        user_count = feat["user_count"].fillna(0) if "user_count" in feat.columns else 0
        user_mean = feat["user_mean_rating"].fillna(gm) if "user_mean_rating" in feat.columns else gm
        feat["user_smoothed_rating"] = (
            (user_count * user_mean + m_user * gm) / (user_count + m_user)
        )

        # Book Bayesian smoothing
        m_book = 5  # smoothing parameter for books (more books, less aggressive)
        book_count = feat["book_count"].fillna(0) if "book_count" in feat.columns else 0
        book_mean = feat["book_mean_rating"].fillna(gm) if "book_mean_rating" in feat.columns else gm
        feat["book_smoothed_rating"] = (
            (book_count * book_mean + m_book * gm) / (book_count + m_book)
        )

        # === NEW: User-book interaction features ===
        feat["user_rating_deviation"] = feat["user_mean_rating"].fillna(gm) - gm
        feat["book_rating_deviation"] = feat["book_mean_rating"].fillna(gm) - gm
        feat["user_book_interaction"] = feat["user_rating_deviation"] * feat["book_rating_deviation"]

        # === NEW: User activity features ===
        feat["user_rating_variance"] = feat["user_std_rating"].fillna(1.0) ** 2

        # === NEW: Book popularity features (already have book_log_count, add more) ===
        feat["book_popularity_log"] = np.log1p(feat["book_count"].fillna(0))
        feat["book_is_popular"] = (feat["book_count"].fillna(0) > 50).astype(int)

    return feat


# Create train/validation split (stratified by rating)
print("\nCreating train/validation split...")
train_df["is_dnf"] = (train_df["rating"] == 0).astype(int)
train_part, val_part = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["rating"]
)
print(f"Train part: {train_part.shape}, Val part: {val_part.shape}")

# Compute user and book statistics from TRAIN PART only (prevent leakage)
print("\nComputing user and book statistics from train part only...")
train_part["_text_len"] = train_part["review_text"].fillna("").str.len()

user_stats = (
    train_part.groupby("user_id")
    .agg(
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        count=("rating", "count"),
        dnf_rate=("is_dnf", "mean"),
        avg_text_length=("_text_len", "mean"),
    )
    .reset_index()
)
user_stats["std_rating"] = user_stats["std_rating"].fillna(0)
user_stats = user_stats.set_index("user_id")

book_stats = (
    train_part.groupby("book_id")
    .agg(
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        count=("rating", "count"),
        dnf_rate=("is_dnf", "mean"),
    )
    .reset_index()
)
book_stats["std_rating"] = book_stats["std_rating"].fillna(0)
book_stats = book_stats.set_index("book_id")

global_mean_rating = train_part["rating"].mean()
print(f"Global mean rating: {global_mean_rating:.4f}")
print(f"Unique users: {len(user_stats)}, Unique books: {len(book_stats)}")

# Engineer features
print("\nEngineering features...")
X_train = engineer_features(
    train_part,
    is_train=True,
    user_stats=user_stats,
    book_stats=book_stats,
    global_mean_rating=global_mean_rating,
)
X_val = engineer_features(
    val_part,
    is_train=True,
    user_stats=user_stats,
    book_stats=book_stats,
    global_mean_rating=global_mean_rating,
)
X_test = engineer_features(
    test_df,
    is_train=False,
    user_stats=user_stats,
    book_stats=book_stats,
    global_mean_rating=global_mean_rating,
)

y_train = train_part["rating"].values
y_val = val_part["rating"].values

print(f"Feature matrix shapes:")
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# Handle NaN values in features
print("\nHandling NaN values...")
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_val[col] = X_val[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

# For categorical features
cat_cols = X_train.select_dtypes(include=["category", "object"]).columns
print(f"Categorical columns: {cat_cols.tolist()}")
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_val[col] = X_val[col].astype("category")
    X_test[col] = X_test[col].astype("category")
    all_cats = X_train[col].cat.categories
    X_val[col] = X_val[col].cat.set_categories(all_cats)
    X_test[col] = X_test[col].cat.set_categories(all_cats)

# Convert to numeric
for col in cat_cols:
    X_train[col] = X_train[col].cat.codes.astype(float)
    X_val[col] = X_val[col].cat.codes.astype(float)
    X_test[col] = X_test[col].cat.codes.astype(float)

print(
    f"\nFinal feature shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
)

# Save processed data
X_train.to_parquet("./working/X_train.parquet")
X_val.to_parquet("./working/X_val.parquet")
X_test.to_parquet("./working/X_test.parquet")
pd.Series(y_train).to_frame("rating").to_parquet("./working/y_train.parquet")
pd.Series(y_val).to_frame("rating").to_parquet("./working/y_val.parquet")

# Save test IDs
if "id" in test_df.columns:
    test_df[["id"]].to_parquet("./working/test_ids.parquet")
else:
    pd.DataFrame({"id": range(len(test_df))}).to_parquet("./working/test_ids.parquet")

# ============================================================
# STEP 2 & 3: MODEL DESIGN, TRAINING & EVALUATION
# ============================================================

# Dual-Model Ensemble:
# Model 1: LightGBM Regressor (captures continuous patterns)
# Model 2: LightGBM Multi-class Classifier (captures discrete ordinal structure)
# Final: 0.5 * reg_pred + 0.5 * multiclass_expected_value

# --- Model 1: LightGBM Regressor ---
print("\n" + "=" * 60)
print("MODEL 1: LightGBM Regressor")
print("=" * 60)

regressor = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=10000,
    learning_rate=0.01,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

print("Training Regressor...")
regressor.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(300, verbose=False)],
)
best_iter_reg = regressor.best_iteration_
print(f"Regressor best iteration: {best_iter_reg}")

reg_val_pred = regressor.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, reg_val_pred))
print(f"Regressor validation RMSE: {reg_val_rmse:.4f}")

# --- Model 2: LightGBM Ordinal Regression (5 binary classifiers) ---
print("\n" + "=" * 60)
print("MODEL 2: LightGBM Ordinal Regression (5 binary classifiers)")
print("=" * 60)

ordinal_models = []
ordinal_best_iters = []

for k in range(1, 6):
    print(f"\n  Training ordinal binary classifier for threshold k={k} (P(y>={k}))...")
    y_binary_train = (y_train >= k).astype(int)
    y_binary_val = (y_val >= k).astype(int)

    ordinal_clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    ordinal_clf.fit(
        X_train,
        y_binary_train,
        eval_set=[(X_val, y_binary_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    ordinal_models.append(ordinal_clf)
    ordinal_best_iters.append(ordinal_clf.best_iteration_)
    print(f"    Best iteration: {ordinal_clf.best_iteration_}")


def ordinal_to_expected_value(models, X):
    """
    Ordinal regression: expected rating = sum of P(y>=k) for k=1..5.
    Each binary classifier predicts P(y>=k), so E[y] = sum_k P(y>=k).
    """
    pred = np.zeros(X.shape[0])
    for model in models:
        pred += model.predict_proba(X)[:, 1]
    return pred


ord_val_pred = ordinal_to_expected_value(ordinal_models, X_val)
ord_val_rmse = np.sqrt(mean_squared_error(y_val, ord_val_pred))
print(f"\nOrdinal regression validation RMSE: {ord_val_rmse:.4f}")

# ============================================================
# ENSEMBLE PREDICTION FUNCTION (identical for val and test)
# ============================================================


def ensemble_predict(model_reg, ordinal_models, X, w_reg):
    """
    Combine regressor and ordinal regression predictions.
    pred = w_reg * reg_pred + (1 - w_reg) * ordinal_expected_value
    """
    reg_pred = model_reg.predict(X)
    ord_pred = ordinal_to_expected_value(ordinal_models, X)
    final_pred = w_reg * reg_pred + (1.0 - w_reg) * ord_pred
    return final_pred


# ============================================================
# OPTIMIZED ENSEMBLE WEIGHT SELECTION VIA GRID SEARCH
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION - OPTIMIZED ENSEMBLE WEIGHTS")
print("=" * 60)

print(f"Regressor RMSE: {reg_val_rmse:.4f}")
print(f"Ordinal RMSE: {ord_val_rmse:.4f}")

best_w_reg = 0.5
best_val_rmse = float("inf")

for w in np.arange(0.0, 1.01, 0.05):
    val_pred_w = ensemble_predict(regressor, ordinal_models, X_val, w)
    rmse_w = np.sqrt(mean_squared_error(y_val, val_pred_w))
    if rmse_w < best_val_rmse:
        best_val_rmse = rmse_w
        best_w_reg = w

val_pred = ensemble_predict(regressor, ordinal_models, X_val, best_w_reg)
val_rmse = best_val_rmse
print(f"Best w_reg: {best_w_reg:.2f}")
print(f"Ensemble RMSE (optimized): {val_rmse:.4f}")

# ============================================================
# SAVE MODELS
# ============================================================
regressor.booster_.save_model("./working/regressor.txt")
for i, model in enumerate(ordinal_models):
    model.booster_.save_model(f"./working/ordinal_clf_k{i+1}.txt")
print("\nModels saved.")

# ============================================================
# TEST PREDICTIONS (identical logic to validation)
# ============================================================
print("\n" + "=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

test_pred = ensemble_predict(regressor, ordinal_models, X_test, best_w_reg)
print(f"Test predictions: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")
print(f"Test pred range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# ============================================================
# SAVE SUBMISSION
# ============================================================
test_ids_df = pd.read_parquet("./working/test_ids.parquet")
if "id" in test_ids_df.columns:
    test_ids = test_ids_df["id"].values
else:
    test_ids = np.arange(len(test_pred))

submission = pd.DataFrame(
    {
        "id": test_ids.astype(int),
        "rating": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head(10))

# ============================================================
# FINAL SCORE
# ============================================================
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Model 1 (Regressor) best iter: {best_iter_reg}")
print(f"Model 2 (Ordinal) best iters: {ordinal_best_iters}")
print(f"Model 1 validation RMSE: {reg_val_rmse:.4f}")
print(f"Model 2 (Ordinal) validation RMSE: {ord_val_rmse:.4f}")
print(f"Ensemble weights: w_reg={best_w_reg:.2f}, w_ordinal={1-best_w_reg:.2f}")
print(f"Ensemble validation RMSE: {val_rmse:.4f}")
print(f"{'='*60}")

score = val_rmse
print(f"Final Validation Score: {score}")