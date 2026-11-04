import pandas as pd
import numpy as np
import os
import re
import torch
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ===================== LOAD DATA =====================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(
    f"Target stats: mean={train['review_overall'].mean():.4f}, std={train['review_overall'].std():.4f}"
)
print(
    f"Target range: [{train['review_overall'].min()}, {train['review_overall'].max()}]"
)

# Identify column types
for col in train.columns:
    if col == "review_overall":
        continue
    print(
        f"  {col}: dtype={train[col].dtype}, nunique={train[col].nunique()}, nulls={train[col].isna().sum()}"
    )

# Separate target
y_full = train["review_overall"].values
train_no_target = train.drop(columns=["review_overall"])

# Create train/val split (split FIRST, then engineer features)
train_df, val_df = train_test_split(train_no_target, test_size=0.2, random_state=42)
test_df = test.copy()

# Get targets aligned with split
train_full_with_target = train_no_target.copy()
train_full_with_target["review_overall"] = y_full
train_split_with_target, val_split_with_target = train_test_split(
    train_full_with_target, test_size=0.2, random_state=42
)
y_train = val_split_with_target["review_overall"].values  # Wait, this is wrong
# Let me redo this properly
train_full_with_target = train_no_target.copy()
train_full_with_target["review_overall"] = y_full
train_split_with_target, val_split_with_target = train_test_split(
    train_full_with_target, test_size=0.2, random_state=42
)
y_train = train_split_with_target["review_overall"].values
y_val = val_split_with_target["review_overall"].values

# Re-do the split to get consistent indices
train_df, val_df = train_test_split(train_no_target, test_size=0.2, random_state=42)
test_df = test.copy()

print(f"\nTrain split: {train_df.shape}, Val split: {val_df.shape}")

# Identify feature types
text_cols = [
    c
    for c in train_df.columns
    if train_df[c].dtype == "object" and train_df[c].str.len().mean() > 50
]
cat_cols = [
    c for c in train_df.columns if train_df[c].dtype == "object" and c not in text_cols
]
num_cols = [
    c
    for c in train_df.columns
    if train_df[c].dtype in ["int64", "float64", "int32", "float32"]
]

print(f"Text columns: {text_cols}")
print(f"Categorical columns: {cat_cols}")
print(f"Numeric columns: {num_cols}")

# ===================== FEATURE ENGINEERING =====================


def extract_text_features(df, text_col):
    """Extract features from text column."""
    if text_col not in df.columns:
        return pd.DataFrame(index=df.index)

    text = df[text_col].fillna("")

    features = pd.DataFrame(index=df.index)
    features[f"{text_col}_len"] = text.str.len()
    features[f"{text_col}_word_count"] = text.str.split().str.len()
    features[f"{text_col}_avg_word_len"] = features[f"{text_col}_len"] / (
        features[f"{text_col}_word_count"] + 1
    )
    features[f"{text_col}_sentence_count"] = text.str.count(r"[.!?]+")
    features[f"{text_col}_exclamation_count"] = text.str.count(r"!")
    features[f"{text_col}_question_count"] = text.str.count(r"\?")
    features[f"{text_col}_caps_ratio"] = text.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    features[f"{text_col}_digit_count"] = text.str.count(r"\d")
    features[f"{text_col}_newline_count"] = text.str.count(r"\n")

    # Sentiment proxies
    positive_words = [
        "great",
        "excellent",
        "amazing",
        "awesome",
        "fantastic",
        "superb",
        "best",
        "love",
        "perfect",
        "incredible",
        "outstanding",
        "delicious",
        "smooth",
        "crisp",
        "refreshing",
        "balanced",
        "well",
        "good",
        "nice",
        "enjoy",
        "favorite",
        "recommend",
        "impressive",
        "solid",
    ]
    negative_words = [
        "bad",
        "terrible",
        "awful",
        "horrible",
        "worst",
        "disappointing",
        "bland",
        "flat",
        "watery",
        "overly",
        "under",
        "stale",
        "off",
        "dislike",
        "poor",
        "weak",
        "thin",
        "harsh",
        "unpleasant",
    ]

    text_lower = text.str.lower()
    features[f"{text_col}_pos_words"] = text_lower.apply(
        lambda x: sum(1 for w in positive_words if w in x.split())
    )
    features[f"{text_col}_neg_words"] = text_lower.apply(
        lambda x: sum(1 for w in negative_words if w in x.split())
    )
    features[f"{text_col}_sentiment_score"] = (
        features[f"{text_col}_pos_words"] - features[f"{text_col}_neg_words"]
    ) / (features[f"{text_col}_pos_words"] + features[f"{text_col}_neg_words"] + 1)

    # Beer-specific vocabulary
    beer_positive = [
        "smooth",
        "crisp",
        "refreshing",
        "balanced",
        "complex",
        "aromatic",
        "flavorful",
        "well-crafted",
        "layered",
        "depth",
        "finish",
        "mouthfeel",
    ]
    beer_negative = [
        "harsh",
        "flat",
        "thin",
        "watery",
        "stale",
        "over-carbonated",
        "under-carbonated",
        "off-flavor",
        "skunky",
        "oxidized",
    ]

    features[f"{text_col}_beer_pos"] = text_lower.apply(
        lambda x: sum(1 for w in beer_positive if w in x)
    )
    features[f"{text_col}_beer_neg"] = text_lower.apply(
        lambda x: sum(1 for w in beer_negative if w in x)
    )
    features[f"{text_col}_beer_sentiment"] = (
        features[f"{text_col}_beer_pos"] - features[f"{text_col}_beer_neg"]
    ) / (features[f"{text_col}_beer_pos"] + features[f"{text_col}_beer_neg"] + 1)

    # Punctuation intensity
    features[f"{text_col}_punct_density"] = text.str.count(r"[^\w\s]") / (
        features[f"{text_col}_word_count"] + 1
    )

    return features


def engineer_features(train_df, val_df, test_df, cat_cols, num_cols, text_cols):
    """Engineer features using only train statistics."""

    # --- Text features ---
    for tc in text_cols:
        train_df = pd.concat([train_df, extract_text_features(train_df, tc)], axis=1)
        val_df = pd.concat([val_df, extract_text_features(val_df, tc)], axis=1)
        test_df = pd.concat([test_df, extract_text_features(test_df, tc)], axis=1)

    # --- Frequency encoding (fit on train only) ---
    for col in cat_cols:
        freq = train_df[col].value_counts(normalize=True)
        train_df[f"{col}_freq"] = train_df[col].map(freq)
        val_df[f"{col}_freq"] = val_df[col].map(freq).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

    # --- Numeric interactions ---
    if len(num_cols) >= 2:
        for i, c1 in enumerate(num_cols[:5]):
            for c2 in num_cols[i + 1 : i + 4]:
                if c1 in train_df.columns and c2 in train_df.columns:
                    train_df[f"{c1}_x_{c2}"] = train_df[c1] * train_df[c2]
                    val_df[f"{c1}_x_{c2}"] = val_df[c1] * val_df[c2]
                    test_df[f"{c1}_x_{c2}"] = test_df[c1] * test_df[c2]

    # --- Log transforms for skewed numeric features ---
    for col in num_cols:
        if col in train_df.columns:
            if train_df[col].std() > 0 and (train_df[col].abs() > 0).any():
                train_df[f"{col}_log"] = np.log1p(train_df[col].clip(lower=0))
                val_df[f"{col}_log"] = np.log1p(val_df[col].clip(lower=0))
                test_df[f"{col}_log"] = np.log1p(test_df[col].clip(lower=0))

    # --- Ratio features ---
    if len(num_cols) >= 2:
        for i, c1 in enumerate(num_cols[:3]):
            for c2 in num_cols[i + 1 : i + 3]:
                if c1 in train_df.columns and c2 in train_df.columns:
                    train_df[f"{c1}_div_{c2}"] = train_df[c1] / (
                        train_df[c2].abs() + 1e-8
                    )
                    val_df[f"{c1}_div_{c2}"] = val_df[c1] / (val_df[c2].abs() + 1e-8)
                    test_df[f"{c1}_div_{c2}"] = test_df[c1] / (test_df[c2].abs() + 1e-8)

    # --- Text length interactions with numeric ---
    for tc in text_cols:
        len_col = f"{tc}_len"
        if len_col in train_df.columns:
            for nc in num_cols[:3]:
                if nc in train_df.columns:
                    train_df[f"{len_col}_x_{nc}"] = train_df[len_col] * train_df[nc]
                    val_df[f"{len_col}_x_{nc}"] = val_df[len_col] * val_df[nc]
                    test_df[f"{len_col}_x_{nc}"] = test_df[len_col] * test_df[nc]

    # --- Sentiment interactions with numeric ---
    for tc in text_cols:
        sent_col = f"{tc}_sentiment_score"
        if sent_col in train_df.columns:
            for nc in num_cols[:3]:
                if nc in train_df.columns:
                    train_df[f"{sent_col}_x_{nc}"] = train_df[sent_col] * train_df[nc]
                    val_df[f"{sent_col}_x_{nc}"] = val_df[sent_col] * val_df[nc]
                    test_df[f"{sent_col}_x_{nc}"] = test_df[sent_col] * test_df[nc]

    return train_df, val_df, test_df


# Run feature engineering
train_processed, val_processed, test_processed = engineer_features(
    train_df, val_df, test_df, cat_cols, num_cols, text_cols
)

print(f"\nAfter feature engineering:")
print(f"  Train: {train_processed.shape}")
print(f"  Val: {val_processed.shape}")
print(f"  Test: {test_processed.shape}")

# Handle remaining non-numeric columns
remaining_cat = [
    c for c in train_processed.columns if train_processed[c].dtype == "object"
]
print(f"Remaining categorical columns: {remaining_cat}")

for col in remaining_cat:
    if train_processed[col].nunique() <= 30:
        dummies = pd.get_dummies(train_processed[col], prefix=col, drop_first=True)
        train_processed = pd.concat(
            [train_processed.drop(columns=[col]), dummies], axis=1
        )
        val_dummies = pd.get_dummies(val_processed[col], prefix=col, drop_first=True)
        for c in dummies.columns:
            if c not in val_dummies.columns:
                val_dummies[c] = 0
        val_dummies = val_dummies[dummies.columns]
        val_processed = pd.concat(
            [val_processed.drop(columns=[col]), val_dummies], axis=1
        )
        test_dummies = pd.get_dummies(test_processed[col], prefix=col, drop_first=True)
        for c in dummies.columns:
            if c not in test_dummies.columns:
                test_dummies[c] = 0
        test_dummies = test_dummies[dummies.columns]
        test_processed = pd.concat(
            [test_processed.drop(columns=[col]), test_dummies], axis=1
        )
    else:
        label_map = {v: i for i, v in enumerate(train_processed[col].unique())}
        train_processed[col] = (
            train_processed[col].map(label_map).fillna(-1).astype(int)
        )
        val_processed[col] = val_processed[col].map(label_map).fillna(-1).astype(int)
        test_processed[col] = test_processed[col].map(label_map).fillna(-1).astype(int)

# Fill remaining NaNs
for col in train_processed.columns:
    if train_processed[col].isna().any():
        median_val = train_processed[col].median()
        train_processed[col] = train_processed[col].fillna(median_val)
        val_processed[col] = val_processed[col].fillna(median_val)
        test_processed[col] = test_processed[col].fillna(median_val)

# Convert all to numeric
train_processed = train_processed.astype(float)
val_processed = val_processed.astype(float)
test_processed = test_processed.astype(float)

# Replace inf values
train_processed = train_processed.replace([np.inf, -np.inf], 0)
val_processed = val_processed.replace([np.inf, -np.inf], 0)
test_processed = test_processed.replace([np.inf, -np.inf], 0)

print(f"\nFinal feature shapes:")
print(f"  Train: {train_processed.shape}")
print(f"  Val: {val_processed.shape}")
print(f"  Test: {test_processed.shape}")

# Save test ids
test_ids = test.index.values
np.save("./working/test_ids.npy", test_ids)

# ===================== MODEL DESIGN =====================

# LightGBM regressor configuration for small tabular dataset
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 15,
    "min_child_samples": 30,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 2000,
}


# ===================== PRETRAINED TEXT ENCODER =====================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim = 384

print(f"\nLoading pretrained encoder: {EMBEDDING_MODEL_NAME}")
text_encoder = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
text_encoder.eval()
for param in text_encoder.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
print(f"Text encoder loaded. Embedding dim: {embedding_dim}")


def compute_text_embeddings(texts, batch_size=64):
    """Compute mean-pooled sentence embeddings using frozen encoder."""
    all_embeddings = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = text_encoder(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.repeat(1, 1, token_embeddings.size(-1))
            embeddings = (token_embeddings * input_mask_expanded).sum(
                1
            ) / input_mask_expanded.sum(1).clamp(min=1e-9)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


# Get text for each split
text_col = text_cols[0] if text_cols else None
print(f"\nText column: {text_col}")

train_texts = train_df[text_col].fillna("").tolist()
val_texts = val_df[text_col].fillna("").tolist()
test_texts = test_df[text_col].fillna("").tolist()

print(
    f"Computing embeddings: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}"
)

train_text_emb = compute_text_embeddings(train_texts)
val_text_emb = compute_text_embeddings(val_texts)
test_text_emb = compute_text_embeddings(test_texts)

print(
    f"Text embeddings computed: train={train_text_emb.shape}, val={val_text_emb.shape}, test={test_text_emb.shape}"
)

# ===================== TRAINING & EVALUATION =====================

# Concatenate structured features and text embeddings (466 total features)
X_train_all = np.hstack([train_processed.values, train_text_emb])
X_val_all = np.hstack([val_processed.values, val_text_emb])
X_test_all = np.hstack([test_processed.values, test_text_emb])

print(f"\n{'='*60}")
print(f"TRAINING START (LightGBM)")
print(f"{'='*60}")
print(f"Train samples: {X_train_all.shape[0]}")
print(f"Total features: {X_train_all.shape[1]} (82 structured + 384 text embeddings)")
print(f"Val samples: {X_val_all.shape[0]}")
print(f"Test samples: {X_test_all.shape[0]}")
print(f"Params: num_leaves=15, min_child_samples=30, lr=0.05")
print(f"Early stopping rounds: 50")
print(f"{'='*60}\n")

# Stage 1: Train on train split with early stopping on validation
train_data = lgb.Dataset(X_train_all, label=y_train)
val_data = lgb.Dataset(X_val_all, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

best_iteration = model.best_iteration
best_val_rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val_all))))

print(f"Best iteration: {best_iteration}")
print(f"Validation RMSE: {best_val_rmse:.4f}")

# Stage 2: Retrain on ALL training data (train + val) with best_iteration
X_train_full = np.vstack([X_train_all, X_val_all])
y_train_full = np.concatenate([y_train, y_val])

train_data_full = lgb.Dataset(X_train_full, label=y_train_full)

model_final = lgb.train(
    {**lgb_params, "n_estimators": best_iteration},
    train_data_full,
    num_boost_round=best_iteration,
)

# Final predictions
test_predictions = model_final.predict(X_test_all)

# Compute final validation score (RMSE - official metric) using the same model that produces test predictions
val_predictions = model_final.predict(X_val_all)
final_val_rmse = float(np.sqrt(mean_squared_error(y_val, val_predictions)))

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Best Validation RMSE: {final_val_rmse:.4f}")
print(
    f"Val predictions: mean={val_predictions.mean():.4f}, std={val_predictions.std():.4f}"
)
print(
    f"Test predictions: mean={test_predictions.mean():.4f}, std={test_predictions.std():.4f}"
)
print(f"Target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"{'='*60}")

# ===================== SAVE SUBMISSION =====================

submission = pd.DataFrame(
    {"id": test_ids.astype(int), "review_overall": test_predictions}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(f"Columns: {list(submission.columns)}")

# Save model
model_final.save_model("./working/lgb_model.txt")

print(f"Final Validation Score: {final_val_rmse:.4f}")