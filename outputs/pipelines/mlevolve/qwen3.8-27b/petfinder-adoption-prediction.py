import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import os
import pickle
from itertools import combinations

# ============================================================
# SETUP
# ============================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(
    f"AdoptionSpeed distribution:\n{train['AdoptionSpeed'].value_counts().sort_index()}"
)

# ============================================================
# 2. CREATE STRATIFIED TRAIN/VALIDATION SPLIT
# ============================================================
train_data, val_data = train_test_split(
    train, test_size=0.15, random_state=42, stratify=train["AdoptionSpeed"]
)
print(f"Train split: {train_data.shape}, Val split: {val_data.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# --- 3.1: Text Feature Engineering ---


def extract_text_features(text_series):
    """Extract rich text features from description/name."""
    features = pd.DataFrame(index=text_series.index)

    features["text_len"] = text_series.str.len().fillna(0)
    features["word_count"] = text_series.str.split().str.len().fillna(0)
    features["avg_word_len"] = features["text_len"] / (features["word_count"] + 1)
    features["unique_word_ratio"] = text_series.apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )
    features["num_digits"] = text_series.str.count(r"\d").fillna(0)
    features["num_special_chars"] = text_series.str.count(r"[^\w\s]").fillna(0)
    features["num_exclamation"] = text_series.str.count(r"!").fillna(0)
    features["num_question"] = text_series.str.count(r"\?").fillna(0)
    features["num_upper_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper())
    )
    features["upper_ratio"] = features["num_upper_words"] / (features["word_count"] + 1)

    char_3gram_counts = text_series.apply(
        lambda x: len(set(re.findall(r".{3}", str(x).lower())))
    )
    features["char_3gram_diversity"] = char_3gram_counts

    features["has_number"] = (text_series.str.contains(r"\d", na=False)).astype(int)
    features["has_url"] = (
        text_series.str.contains(r"http|www|\.com", na=False)
    ).astype(int)
    features["has_emoticon"] = (
        text_series.str.contains(r"[:;][-o]*[)\](]", na=False)
    ).astype(int)

    positive_words = [
        "friendly",
        "loving",
        "gentle",
        "sweet",
        "playful",
        "good",
        "great",
        "adorable",
        "beautiful",
        "happy",
        "smart",
        "clever",
        "calm",
        "mellow",
        "affectionate",
        "cuddly",
        "snuggly",
        "kind",
        "loyal",
        "devoted",
        "easy",
        "quick",
        "fast",
        "ready",
    ]
    negative_words = [
        "aggressive",
        "bite",
        "bitten",
        "fearful",
        "anxious",
        "reactive",
        "dominant",
        "territorial",
        "destructive",
        "escape",
        "escaped",
        "difficult",
        "challenging",
        "severe",
        "serious",
        "critical",
        "ill",
        "sick",
        "disease",
        "cancer",
        "tumor",
        "injured",
        "injury",
    ]
    urgency_words = [
        "urgent",
        "immediately",
        "asap",
        "now",
        "quickly",
        "fast",
        "before",
        "deadline",
        "time-sensitive",
        "last chance",
    ]
    health_words = [
        "vaccinated",
        "spayed",
        "neutered",
        "sterilized",
        "dewormed",
        "healthy",
        "checkup",
        "vet",
        "doctor",
        "treatment",
        "medication",
    ]

    def count_words(text, word_list):
        text_lower = str(text).lower()
        return sum(1 for w in word_list if w in text_lower)

    features["positive_word_count"] = text_series.apply(
        lambda x: count_words(x, positive_words)
    )
    features["negative_word_count"] = text_series.apply(
        lambda x: count_words(x, negative_words)
    )
    features["urgency_word_count"] = text_series.apply(
        lambda x: count_words(x, urgency_words)
    )
    features["health_word_count"] = text_series.apply(
        lambda x: count_words(x, health_words)
    )
    features["sentiment_score"] = (
        features["positive_word_count"] - features["negative_word_count"]
    )

    return features


def extract_name_features(name_series):
    """Extract features from pet names."""
    features = pd.DataFrame(index=name_series.index)

    features["name_len"] = name_series.str.len().fillna(0)
    features["name_is_common"] = name_series.isin(
        [
            "Baby",
            "Lucky",
            "No Name",
            "Brownie",
            "Max",
            "Bella",
            "Charlie",
            "Luna",
            "Daisy",
            "Milo",
            "Coco",
            "Rocky",
            "Duke",
            "Rex",
            "Shadow",
            "Ginger",
            "Molly",
            "Rosie",
            "Tiger",
            "Buddy",
            "Sunny",
            "Pepper",
            "Oreo",
            "Nala",
        ]
    ).astype(int)
    features["name_starts_upper"] = name_series.apply(
        lambda x: 1 if str(x)[0].isupper() else 0
    ).fillna(0)
    features["name_all_lower"] = name_series.apply(
        lambda x: 1 if str(x) == str(x).lower() else 0
    ).fillna(0)
    features["name_has_digit"] = name_series.str.contains(r"\d", na=False).astype(int)
    features["name_is_no_name"] = (name_series == "No Name").astype(int)

    return features


print("Extracting text features...")
train_text_feat = extract_text_features(train_data["Description"])
val_text_feat = extract_text_features(val_data["Description"])
test_text_feat = extract_text_features(test["Description"])

train_name_feat = extract_name_features(train_data["Name"])
val_name_feat = extract_name_features(val_data["Name"])
test_name_feat = extract_name_features(test["Name"])

# --- 3.2: Numeric Feature Engineering ---


def create_numeric_features(df):
    """Create interaction and transformation features from numeric/categorical columns."""
    features = pd.DataFrame(index=df.index)

    features["age_squared"] = df["Age"] ** 2
    features["age_log"] = np.log1p(df["Age"])
    features["is_baby"] = (df["Age"] <= 3).astype(int)
    features["is_young"] = (df["Age"] <= 12).astype(int)
    features["is_senior"] = (df["Age"] >= 60).astype(int)
    features["is_very_senior"] = (df["Age"] >= 96).astype(int)

    features["photo_log"] = np.log1p(df["PhotoAmt"])
    features["video_log"] = np.log1p(df["VideoAmt"])
    features["total_media"] = df["PhotoAmt"] + df["VideoAmt"]
    features["media_ratio"] = df["VideoAmt"] / (df["PhotoAmt"] + 1)
    features["has_video"] = (df["VideoAmt"] > 0).astype(int)
    features["many_photos"] = (df["PhotoAmt"] >= 10).astype(int)

    features["fee_log"] = np.log1p(df["Fee"])
    features["is_free"] = (df["Fee"] == 0).astype(int)
    features["is_expensive"] = (df["Fee"] > 1000).astype(int)

    features["quantity_log"] = np.log1p(df["Quantity"])
    features["is_single"] = (df["Quantity"] == 1).astype(int)
    features["is_multiple"] = (df["Quantity"] > 1).astype(int)
    features["is_many"] = (df["Quantity"] >= 5).astype(int)

    features["sterilized_young"] = ((df["Sterilized"] == 1) & (df["Age"] <= 24)).astype(
        int
    )
    features["sterilized_senior"] = (
        (df["Sterilized"] == 1) & (df["Age"] >= 60)
    ).astype(int)

    features["healthy_young"] = ((df["Health"] == 1) & (df["Age"] <= 24)).astype(int)
    features["unhealthy_senior"] = ((df["Health"] == 3) & (df["Age"] >= 60)).astype(int)

    features["male_young"] = ((df["Gender"] == 2) & (df["Age"] <= 24)).astype(int)
    features["female_young"] = ((df["Gender"] == 1) & (df["Age"] <= 24)).astype(int)

    features["is_cat"] = (df["Type"] == 1).astype(int)
    features["is_dog"] = (df["Type"] == 2).astype(int)

    features["has_two_breeds"] = (df["Breed2"] > 0).astype(int)
    features["same_breed"] = (df["Breed1"] == df["Breed2"]).astype(int)
    features["breed1_is_common"] = (
        df["Breed1"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(int)
    )

    features["has_multiple_colors"] = (df["Color3"] > 0).astype(int)
    features["has_white"] = (
        (df["Color1"] == 1) | (df["Color2"] == 1) | (df["Color3"] == 1)
    ).astype(int)
    features["has_black"] = (
        (df["Color1"] == 2) | (df["Color2"] == 2) | (df["Color3"] == 2)
    ).astype(int)
    features["has_brown"] = (
        (df["Color1"] == 3) | (df["Color2"] == 3) | (df["Color3"] == 3)
    ).astype(int)
    features["has_gray"] = (
        (df["Color1"] == 5) | (df["Color2"] == 5) | (df["Color3"] == 5)
    ).astype(int)

    features["long_hair_cat"] = ((df["FurLength"] == 3) & (df["Type"] == 1)).astype(int)
    features["long_hair_dog"] = ((df["FurLength"] == 3) & (df["Type"] == 2)).astype(int)

    features["large_young"] = ((df["MaturitySize"] >= 3) & (df["Age"] <= 24)).astype(
        int
    )
    features["small_senior"] = ((df["MaturitySize"] <= 2) & (df["Age"] >= 60)).astype(
        int
    )

    features["all_medical"] = (
        (df["Vaccinated"] == 1) & (df["Dewormed"] == 1) & (df["Sterilized"] == 1)
    ).astype(int)
    features["medical_count"] = (
        (df["Vaccinated"] == 1).astype(int)
        + (df["Dewormed"] == 1).astype(int)
        + (df["Sterilized"] == 1).astype(int)
    )

    # state_center removed; State is already included as a categorical column
    # and state-level aggregations are handled in create_state_features

    return features


print("Creating numeric features...")
train_num_feat = create_numeric_features(train_data)
val_num_feat = create_numeric_features(val_data)
test_num_feat = create_numeric_features(test)

# --- 3.3: Frequency Encoding (fit on train only) ---


def create_frequency_encodings(train_df, val_df, test_df, columns):
    """Create frequency encodings fit only on training data."""
    train_enc = pd.DataFrame(index=train_df.index)
    val_enc = pd.DataFrame(index=val_df.index)
    test_enc = pd.DataFrame(index=test_df.index)

    for col in columns:
        freq = train_df[col].value_counts(normalize=True)
        train_enc[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
        val_enc[f"{col}_freq"] = val_df[col].map(freq).fillna(0)
        test_enc[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

    return train_enc, val_enc, test_enc


freq_cols = [
    "Breed1",
    "Breed2",
    "State",
    "RescuerID",
    "MaturitySize",
    "FurLength",
    "Gender",
]
print("Creating frequency encodings...")
train_freq, val_freq, test_freq = create_frequency_encodings(
    train_data, val_data, test, freq_cols
)

# --- 3.4: Target-Mean Encoding (fit on train only, with smoothing) ---


def create_target_mean_encoding(
    train_df, val_df, test_df, columns, target="AdoptionSpeed", smoothing=10
):
    """Create target-mean encodings with Bayesian smoothing, fit only on training data."""
    global_mean = train_df[target].mean()

    train_enc = pd.DataFrame(index=train_df.index)
    val_enc = pd.DataFrame(index=val_df.index)
    test_enc = pd.DataFrame(index=test_df.index)

    for col in columns:
        stats = train_df.groupby(col)[target].agg(["mean", "count"])
        smoothed_mean = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        train_enc[f"{col}_target_mean"] = (
            train_df[col].map(smoothed_mean).fillna(global_mean)
        )
        val_enc[f"{col}_target_mean"] = (
            val_df[col].map(smoothed_mean).fillna(global_mean)
        )
        test_enc[f"{col}_target_mean"] = (
            test_df[col].map(smoothed_mean).fillna(global_mean)
        )

    return train_enc, val_enc, test_enc


tme_cols = [
    "Breed1",
    "Breed2",
    "State",
    "RescuerID",
    "Type",
    "MaturitySize",
    "FurLength",
]
print("Creating target-mean encodings...")
train_tme, val_tme, test_tme = create_target_mean_encoding(
    train_data, val_data, test, tme_cols
)

# --- 3.5: RescuerID Aggregation Features ---


def create_rescuer_features(train_df, val_df, test_df):
    """Create aggregated features per rescuer, fit only on train."""
    rescuer_stats = (
        train_df.groupby("RescuerID")
        .agg(
            rescuer_count=("PetID", "count"),
            rescuer_mean_age=("Age", "mean"),
            rescuer_mean_fee=("Fee", "mean"),
            rescuer_mean_photos=("PhotoAmt", "mean"),
            rescuer_mean_adoption=("AdoptionSpeed", "mean"),
            rescuer_std_adoption=("AdoptionSpeed", "std"),
            rescuer_cat_ratio=("Type", lambda x: (x == 1).mean()),
            rescuer_young_ratio=("Age", lambda x: (x <= 12).mean()),
        )
        .reset_index()
    )
    rescuer_stats["rescuer_std_adoption"] = rescuer_stats[
        "rescuer_std_adoption"
    ].fillna(0)

    def merge_rescuer(df):
        merged = df[["RescuerID"]].merge(rescuer_stats, on="RescuerID", how="left")
        for col in [
            "rescuer_count",
            "rescuer_mean_age",
            "rescuer_mean_fee",
            "rescuer_mean_photos",
            "rescuer_mean_adoption",
            "rescuer_std_adoption",
            "rescuer_cat_ratio",
            "rescuer_young_ratio",
        ]:
            merged[col] = merged[col].fillna(0)
        merged = merged.drop(columns=["RescuerID"])
        merged.index = df.index
        return merged

    return merge_rescuer(train_df), merge_rescuer(val_df), merge_rescuer(test_df)


print("Creating rescuer features...")
train_rescuer, val_rescuer, test_rescuer = create_rescuer_features(
    train_data, val_data, test
)

# --- 3.6: Breed Pair Features ---


def create_breed_pair_features(train_df, val_df, test_df):
    """Create features for breed combinations."""

    def get_breed_pair(df):
        pairs = []
        for _, row in df.iterrows():
            b1, b2 = row["Breed1"], row["Breed2"]
            if b2 > 0:
                pairs.append(f"{min(b1,b2)}_{max(b1,b2)}")
            else:
                pairs.append(f"{b1}_single")
        return pd.Series(pairs, index=df.index)

    train_pair = get_breed_pair(train_df)
    val_pair = get_breed_pair(val_df)
    test_pair = get_breed_pair(test_df)

    pair_freq = train_pair.value_counts(normalize=True)

    train_enc = pd.DataFrame(index=train_df.index)
    val_enc = pd.DataFrame(index=val_df.index)
    test_enc = pd.DataFrame(index=test_df.index)

    train_enc["breed_pair_freq"] = train_pair.map(pair_freq).fillna(0)
    val_enc["breed_pair_freq"] = val_pair.map(pair_freq).fillna(0)
    test_enc["breed_pair_freq"] = test_pair.map(pair_freq).fillna(0)

    pair_target = pd.DataFrame(
        {"pair": train_pair, "target": train_df["AdoptionSpeed"]}
    )
    pair_target_mean = pair_target.groupby("pair")["target"].mean()
    global_mean = train_df["AdoptionSpeed"].mean()

    train_enc["breed_pair_target_mean"] = train_pair.map(pair_target_mean).fillna(
        global_mean
    )
    val_enc["breed_pair_target_mean"] = val_pair.map(pair_target_mean).fillna(
        global_mean
    )
    test_enc["breed_pair_target_mean"] = test_pair.map(pair_target_mean).fillna(
        global_mean
    )

    return train_enc, val_enc, test_enc


print("Creating breed pair features...")
train_breed_pair, val_breed_pair, test_breed_pair = create_breed_pair_features(
    train_data, val_data, test
)

# --- 3.7: State-Level Aggregation ---


def create_state_features(train_df, val_df, test_df):
    """Create state-level aggregated features."""
    state_stats = (
        train_df.groupby("State")
        .agg(
            state_count=("PetID", "count"),
            state_mean_adoption=("AdoptionSpeed", "mean"),
            state_std_adoption=("AdoptionSpeed", "std"),
            state_mean_age=("Age", "mean"),
            state_mean_fee=("Fee", "mean"),
            state_young_ratio=("Age", lambda x: (x <= 12).mean()),
            state_cat_ratio=("Type", lambda x: (x == 1).mean()),
        )
        .reset_index()
    )
    state_stats["state_std_adoption"] = state_stats["state_std_adoption"].fillna(0)

    def merge_state(df):
        merged = df[["State"]].merge(state_stats, on="State", how="left")
        for col in [
            "state_count",
            "state_mean_adoption",
            "state_std_adoption",
            "state_mean_age",
            "state_mean_fee",
            "state_young_ratio",
            "state_cat_ratio",
        ]:
            if col == "state_mean_adoption":
                merged[col] = merged[col].fillna(train_df["AdoptionSpeed"].mean())
            elif col == "state_count":
                merged[col] = merged[col].fillna(1)
            else:
                merged[col] = merged[col].fillna(0)
        merged = merged.drop(columns=["State"])
        merged.index = df.index
        return merged

    return merge_state(train_df), merge_state(val_df), merge_state(test_df)


print("Creating state features...")
train_state, val_state, test_state = create_state_features(train_data, val_data, test)

# --- 3.8: Combine All Features ---

print("Combining all features...")

drop_cols = ["AdoptionSpeed", "PetID", "Description", "Name", "RescuerID"]


def combine_features(
    base_df,
    text_feat,
    name_feat,
    num_feat,
    freq_feat,
    tme_feat,
    rescuer_feat,
    breed_pair_feat,
    state_feat,
):
    base = base_df.drop(columns=[c for c in drop_cols if c in base_df.columns])
    combined = pd.concat(
        [
            base,
            text_feat,
            name_feat,
            num_feat,
            freq_feat,
            tme_feat,
            rescuer_feat,
            breed_pair_feat,
            state_feat,
        ],
        axis=1,
    )
    return combined


train_features = combine_features(
    train_data,
    train_text_feat,
    train_name_feat,
    train_num_feat,
    train_freq,
    train_tme,
    train_rescuer,
    train_breed_pair,
    train_state,
)
val_features = combine_features(
    val_data,
    val_text_feat,
    val_name_feat,
    val_num_feat,
    val_freq,
    val_tme,
    val_rescuer,
    val_breed_pair,
    val_state,
)
test_features = combine_features(
    test,
    test_text_feat,
    test_name_feat,
    test_num_feat,
    test_freq,
    test_tme,
    test_rescuer,
    test_breed_pair,
    test_state,
)

# Handle NaN and infinities
for df in [train_features, val_features, test_features]:
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

print(
    f"Final feature shapes - Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}"
)

# ============================================================
# 4. PREPARE DATA FOR LIGHTGBM
# ============================================================
X_train = train_features.copy()
X_val = val_features.copy()
X_test = test_features.copy()

train_labels = train_data["AdoptionSpeed"].values.astype(np.int64)
val_labels = val_data["AdoptionSpeed"].values.astype(np.int64)

# Define categorical columns for LightGBM native handling
categorical_cols = [
    "Breed1", "Breed2", "State", "Type", "Gender", "MaturitySize",
    "FurLength", "Color1", "Color2", "Color3", "Vaccinated",
    "Dewormed", "Sterilized", "Health"
]

# Ensure categorical columns are of category dtype for LightGBM
for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")
        X_test[col] = X_test[col].astype("category")

# Ensure all other columns are numeric
for col in X_train.columns:
    if col not in categorical_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce").fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

print(f"Label distribution (train): {np.bincount(train_labels, minlength=5)}")
print(f"Feature columns: {X_train.shape[1]}, Categorical cols: {len(categorical_cols)}")

# ============================================================
# 5. LIGHTGBM ORDINAL REGRESSION ENSEMBLE
# ============================================================
import lightgbm as lgb


def create_ordinal_targets(labels, num_classes=5):
    """Create binary ordinal targets: P(y >= k) for k=1..num_classes-1."""
    targets = []
    for k in range(1, num_classes):
        targets.append((labels >= k).astype(np.float32))
    return targets


def train_lgbm_multiclass_ensemble(X_train, y_train, X_val, y_val, categorical_cols, num_classes=5):
    """
    Train LightGBM multi-class classifier ensemble.
    Returns list of 3 LGBMClassifier models with different hyperparameters.
    """
    model_configs = [
        {
            "n_estimators": 3000,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "min_child_samples": 30,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        {
            "n_estimators": 3000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_child_samples": 20,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    ]

    models = []
    for i, config in enumerate(model_configs):
        print(f"  Training Model {chr(65+i)}: lr={config['learning_rate']}, "
              f"num_leaves={config['num_leaves']}, n_estimators={config['n_estimators']}")

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
            random_state=42 + i,
            **config,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(500),
            ],
        )
        models.append(model)
        print(f"  Model {chr(65+i)}: best_iteration={model.best_iteration_}")

    return models


def predict_multiclass_ensemble(models, X):
    """
    Get averaged probability vectors from multi-class ensemble.
    Returns:
        avg_probs: (n_samples, num_classes) averaged probability matrix
        expected_value: (n_samples,) expected value = sum(k * P(k) for k=0..4)
    """
    num_classes = 5
    avg_probs = np.zeros((len(X), num_classes), dtype=np.float64)
    for model in models:
        probs = model.predict_proba(X, num_iteration=model.best_iteration_)
        avg_probs += probs
    avg_probs /= len(models)
    expected_value = avg_probs @ np.arange(num_classes, dtype=np.float64)
    return avg_probs, expected_value


# ============================================================
# 8. QWK METRIC (Official Competition Metric)
# ============================================================


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """Compute QWK exactly as per the official competition metric."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    C = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1

    W = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = (i - j) ** 2 / (num_classes - 1) ** 2

    row_sum = C.sum(axis=1)
    col_sum = C.sum(axis=0)
    total = C.sum()
    E = np.outer(row_sum, col_sum) / total

    numerator = (W * C).sum()
    denominator = (W * E).sum()

    if denominator == 0:
        return 1.0

    return 1.0 - numerator / denominator


# ============================================================
# 9. PREDICTION HELPER (LightGBM)
# ============================================================


def get_continuous_predictions_lgbm(models, X):
    """Get continuous expected value predictions from LightGBM multiclass ensemble."""
    _, expected_value = predict_multiclass_ensemble(models, X)
    return expected_value


# ============================================================
# 10. QWK-OPTIMAL THRESHOLD SEARCH
# ============================================================


def find_optimal_thresholds(val_continuous, val_labels, num_classes=5):
    """Search for optimal decision thresholds that maximize QWK."""
    best_qwk = -1
    best_thresholds = [0.5, 1.5, 2.5, 3.5]

    coarse_range = np.arange(0.0, 4.01, 0.25)

    best_coarse_qwk = -1
    best_coarse_thresholds = [0.5, 1.5, 2.5, 3.5]

    for t1 in coarse_range:
        for t2 in coarse_range:
            if t2 <= t1:
                continue
            for t3 in coarse_range:
                if t3 <= t2:
                    continue
                for t4 in coarse_range:
                    if t4 <= t3:
                        continue
                    preds = np.zeros(len(val_continuous), dtype=int)
                    preds[val_continuous >= t1] = 1
                    preds[val_continuous >= t2] = 2
                    preds[val_continuous >= t3] = 3
                    preds[val_continuous >= t4] = 4
                    qwk = quadratic_weighted_kappa(val_labels, preds, num_classes)
                    if qwk > best_coarse_qwk:
                        best_coarse_qwk = qwk
                        best_coarse_thresholds = [t1, t2, t3, t4]

    fine_range = np.arange(0.0, 4.01, 0.05)
    best_fine_qwk = best_coarse_qwk
    best_fine_thresholds = best_coarse_thresholds

    for t1 in fine_range:
        if abs(t1 - best_coarse_thresholds[0]) > 0.3:
            continue
        for t2 in fine_range:
            if t2 <= t1 or abs(t2 - best_coarse_thresholds[1]) > 0.3:
                continue
            for t3 in fine_range:
                if t3 <= t2 or abs(t3 - best_coarse_thresholds[2]) > 0.3:
                    continue
                for t4 in fine_range:
                    if t4 <= t3 or abs(t4 - best_coarse_thresholds[3]) > 0.3:
                        continue
                    preds = np.zeros(len(val_continuous), dtype=int)
                    preds[val_continuous >= t1] = 1
                    preds[val_continuous >= t2] = 2
                    preds[val_continuous >= t3] = 3
                    preds[val_continuous >= t4] = 4
                    qwk = quadratic_weighted_kappa(val_labels, preds, num_classes)
                    if qwk > best_fine_qwk:
                        best_fine_qwk = qwk
                        best_fine_thresholds = [t1, t2, t3, t4]

    return best_fine_qwk, best_fine_thresholds


def apply_thresholds(continuous_preds, thresholds):
    """Apply decision thresholds to continuous predictions to get class labels."""
    preds = np.zeros(len(continuous_preds), dtype=int)
    preds[continuous_preds >= thresholds[0]] = 1
    preds[continuous_preds >= thresholds[1]] = 2
    preds[continuous_preds >= thresholds[2]] = 3
    preds[continuous_preds >= thresholds[3]] = 4
    return preds


# ============================================================
# 11. LIGHTGBM MULTI-CLASS ENSEMBLE TRAINING
# ============================================================

num_classes = 5
print(f"\nTraining LightGBM Multi-Class Ensemble (3 models)...")
print(f"  Features: {X_train.shape[1]}, Categorical: {len(categorical_cols)}")
print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

lgbm_models = train_lgbm_multiclass_ensemble(
    X_train, train_labels, X_val, val_labels, categorical_cols, num_classes
)

# ============================================================
# 12. GENERATE PREDICTIONS FROM LIGHTGBM MULTI-CLASS ENSEMBLE
# ============================================================

val_probs, val_continuous = predict_multiclass_ensemble(lgbm_models, X_val)
test_probs, test_continuous = predict_multiclass_ensemble(lgbm_models, X_test)

print(f"\nValidation expected value - mean: {val_continuous.mean():.4f}, "
      f"std: {val_continuous.std():.4f}, min: {val_continuous.min():.4f}, max: {val_continuous.max():.4f}")
print(f"Test expected value - mean: {test_continuous.mean():.4f}, "
      f"std: {test_continuous.std():.4f}, min: {test_continuous.min():.4f}, max: {test_continuous.max():.4f}")

# Argmax predictions from averaged probabilities
val_argmax = np.argmax(val_probs, axis=1).astype(int)
test_argmax = np.argmax(test_probs, axis=1).astype(int)

argmax_qwk = quadratic_weighted_kappa(val_labels, val_argmax, num_classes=5)
print(f"QWK with argmax: {argmax_qwk:.4f}")
print(f"Argmax prediction distribution: {np.bincount(val_argmax, minlength=5)}")

# ============================================================
# 13. QWK-OPTIMAL THRESHOLD SEARCH ON VALIDATION
# ============================================================

print("\nRunning QWK-optimal threshold search...")
optimal_qwk, optimal_thresholds = find_optimal_thresholds(
    val_continuous, val_labels, num_classes=5
)
print(f"Optimal thresholds: {optimal_thresholds}")
print(f"QWK with optimal thresholds: {optimal_qwk:.4f}")

val_rounded = np.clip(np.round(val_continuous), 0, 4).astype(int)
standard_qwk = quadratic_weighted_kappa(val_labels, val_rounded, num_classes=5)
print(f"QWK with standard rounding: {standard_qwk:.4f}")

# ============================================================
# 14. COMPARE ARGMAX VS THRESHOLD-BASED PREDICTIONS
# ============================================================

print(f"\nComparison:")
print(f"  Argmax QWK: {argmax_qwk:.4f}")
print(f"  Optimal thresholds QWK: {optimal_qwk:.4f}")
print(f"  Standard rounding QWK: {standard_qwk:.4f}")

# Pick the best method
best_method_qwk = max(argmax_qwk, optimal_qwk, standard_qwk)
if best_method_qwk == argmax_qwk:
    val_preds_final = val_argmax
    test_preds_final = test_argmax
    print("  -> Using argmax of averaged probabilities")
elif best_method_qwk == optimal_qwk:
    val_preds_final = apply_thresholds(val_continuous, optimal_thresholds)
    test_preds_final = apply_thresholds(test_continuous, optimal_thresholds)
    print(f"  -> Using optimal thresholds {optimal_thresholds}")
else:
    val_preds_final = val_rounded
    test_preds_final = np.clip(np.round(test_continuous), 0, 4).astype(int)
    print("  -> Using standard rounding")

final_val_qwk = quadratic_weighted_kappa(val_labels, val_preds_final, num_classes=5)
print(f"\nFinal validation QWK: {final_val_qwk:.4f}")
print(
    f"Validation prediction distribution: {np.bincount(val_preds_final, minlength=5)}"
)
print(f"Test prediction distribution: {np.bincount(test_preds_final, minlength=5)}")

# ============================================================
# 16. CREATE SUBMISSION
# ============================================================

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": range(len(test_preds_final)), "AdoptionSpeed": test_preds_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 17. FINAL SCORE
# ============================================================

print(f"\n{'='*50}")
print(f"Model: LightGBM Multi-Class Ensemble (3 models)")
print(f"Architecture: 3x LGBMClassifier (multiclass, class_weight='balanced')")
print(f"  Model A: lr=0.02, num_leaves=127, n_estimators=3000")
print(f"  Model B: lr=0.03, num_leaves=63, n_estimators=3000")
print(f"  Model C: lr=0.05, num_leaves=255, n_estimators=2000")
print(f"Features: {X_train.shape[1]} (incl. {len(categorical_cols)} categorical)")
print(f"Prediction: Averaged probabilities -> argmax/threshold")
print(f"Validation QWK: {final_val_qwk:.4f}")
print(f"{'='*50}")

score = final_val_qwk
print(f"Final Validation Score: {score}")