import os
import numpy as np
import pandas as pd
import lightgbm as lgb

# -----------------------------
# Setup
# -----------------------------
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

SEED = 42
np.random.seed(SEED)

# -----------------------------
# Load data
# -----------------------------
train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Basic cleaning / sanity checks
if "rating" not in train.columns:
    raise ValueError("Train data must contain the target column 'rating'.")

# Ensure required columns exist
for col in ["game_id", "nickname", "score"]:
    if col not in train.columns:
        raise ValueError(f"Train data is missing required column: {col}")

# Drop rows with missing target or required features
train = train.dropna(subset=["rating", "game_id", "nickname", "score"]).copy()
test = test.dropna(subset=["game_id", "nickname", "score"]).copy()

# Coerce numeric types
train["game_id"] = pd.to_numeric(train["game_id"], errors="coerce")
train["score"] = pd.to_numeric(train["score"], errors="coerce")
train["rating"] = pd.to_numeric(train["rating"], errors="coerce")

test["game_id"] = pd.to_numeric(test["game_id"], errors="coerce")
test["score"] = pd.to_numeric(test["score"], errors="coerce")

# Remove rows that became invalid after coercion
train = train.dropna(subset=["game_id", "score", "rating"]).copy()
test = test.dropna(subset=["game_id", "score"]).copy()

# Keep original test order for submission
test = test.reset_index(drop=True).copy()

# Sort training data temporally by game_id, then nickname for stable ordering
train = train.sort_values(["game_id", "nickname"], kind="mergesort").reset_index(
    drop=True
)

# -----------------------------
# Feature engineering
# -----------------------------
FEATURES = [
    "score",
    "game_id",
    "nickname_code",
    "is_unseen_player",
    "player_count",
    "player_mean_score",
    "player_std_score",
    "player_min_score",
    "player_max_score",
    "player_last_game_id",
    "player_first_game_id",
    "games_since_last",
    "score_minus_player_mean",
    "score_z",
]

CATEGORICAL_FEATURES = ["nickname_code"]


def build_mapping(train_part: pd.DataFrame) -> dict:
    """Create a stable integer mapping for nicknames seen in train_part."""
    names = train_part["nickname"].dropna().unique()
    return {name: i + 1 for i, name in enumerate(sorted(names))}


def add_features(
    df: pd.DataFrame, train_part: pd.DataFrame, mapping: dict
) -> pd.DataFrame:
    """
    Add features to df using only statistics computed from train_part.
    This prevents leakage from validation/test rows.
    """
    out = df.copy()

    # Encode nickname using mapping built from the training portion only
    out["nickname_code"] = out["nickname"].map(mapping).fillna(0).astype(int)
    out["is_unseen_player"] = (out["nickname_code"] == 0).astype(int)

    # Global fallbacks from the training portion
    global_mean_score = float(train_part["score"].mean()) if len(train_part) else 0.0
    global_mean_game_id = (
        float(train_part["game_id"].mean()) if len(train_part) else 0.0
    )

    # Player-level historical aggregates from the training portion only
    if len(train_part) > 0:
        agg = train_part.groupby("nickname", as_index=False).agg(
            player_count=("score", "size"),
            player_mean_score=("score", "mean"),
            player_std_score=("score", "std"),
            player_min_score=("score", "min"),
            player_max_score=("score", "max"),
            player_last_game_id=("game_id", "max"),
            player_first_game_id=("game_id", "min"),
        )
    else:
        agg = pd.DataFrame(columns=["nickname"])

    out = out.merge(agg, on="nickname", how="left")

    # Fill unseen / missing player statistics with safe fallbacks
    out["player_count"] = out["player_count"].fillna(0).astype(int)
    out["player_mean_score"] = out["player_mean_score"].fillna(global_mean_score)
    out["player_std_score"] = out["player_std_score"].fillna(0.0)
    out["player_min_score"] = out["player_min_score"].fillna(global_mean_score)
    out["player_max_score"] = out["player_max_score"].fillna(global_mean_score)
    out["player_last_game_id"] = out["player_last_game_id"].fillna(global_mean_game_id)
    out["player_first_game_id"] = out["player_first_game_id"].fillna(
        global_mean_game_id
    )

    # Derived features
    out["games_since_last"] = (out["game_id"] - out["player_last_game_id"]).clip(
        lower=0
    )
    out["score_minus_player_mean"] = out["score"] - out["player_mean_score"]
    out["score_z"] = np.where(
        out["player_std_score"] > 1e-8,
        out["score_minus_player_mean"] / out["player_std_score"],
        0.0,
    )

    # Ensure all feature columns exist and are numeric
    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out[FEATURES]


# -----------------------------
# Model training helper
# -----------------------------
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> lgb.Booster:
    train_ds = lgb.Dataset(
        data=X_train,
        label=y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    val_ds = lgb.Dataset(
        data=X_val,
        label=y_val,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_ds,
        free_raw_data=False,
    )

    params = {
        "objective": "regression_l1",  # directly optimize MAE
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=3000,
        valid_sets=[val_ds],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    return model


# -----------------------------
# Time-series expanding-window validation
# -----------------------------
def time_series_validation(train_df: pd.DataFrame, n_folds: int = 5) -> float:
    """
    Split the chronologically sorted training data into n_folds consecutive chunks.
    For each chunk after the first, train on all previous chunks and validate on the current chunk.
    This avoids using future information.
    """
    n = len(train_df)
    if n < 100 or n_folds < 2:
        # Fallback to a simple holdout if data is too small
        split = int(n * 0.8)
        tr = train_df.iloc[:split]
        va = train_df.iloc[split:]
        mapping = build_mapping(tr)
        X_tr = add_features(tr, tr, mapping)
        X_va = add_features(va, tr, mapping)
        model = train_model(X_tr, tr["rating"], X_va, va["rating"])
        pred = model.predict(X_va)
        return float(np.mean(np.abs(pred - va["rating"].values)))

    # Create roughly equal consecutive chunks
    chunk_bounds = np.linspace(0, n, n_folds + 1, dtype=int)

    maes = []
    for i in range(1, n_folds):
        train_idx = slice(chunk_bounds[i - 1], chunk_bounds[i])
        val_idx = slice(chunk_bounds[i], chunk_bounds[i + 1])

        train_part = train_df.iloc[train_idx].copy()
        val_part = train_df.iloc[val_idx].copy()

        if len(train_part) == 0 or len(val_part) == 0:
            continue

        mapping = build_mapping(train_part)
        X_train = add_features(train_part, train_part, mapping)
        X_val = add_features(val_part, train_part, mapping)

        model = train_model(X_train, train_part["rating"], X_val, val_part["rating"])
        pred = model.predict(X_val)
        mae = float(np.mean(np.abs(pred - val_part["rating"].values)))
        maes.append(mae)

    if len(maes) == 0:
        return float("nan")

    return float(np.mean(maes))


# -----------------------------
# Evaluate
# -----------------------------
print("Running time-series expanding-window validation...")
val_mae = time_series_validation(train, n_folds=5)
print(f"Validation MAE: {val_mae:.6f}")

# -----------------------------
# Final training on full train and test prediction
# -----------------------------
print("Training final model on full training data...")
final_mapping = build_mapping(train)
X_train_final = add_features(train, train, final_mapping)
y_train_final = train["rating"]

# For final training, use a small internal holdout only for early stopping.
# This does not affect the final fitted model, which is retrained on all data.
n = len(train)
early_split = int(n * 0.9)
X_early_train = X_train_final.iloc[:early_split]
y_early_train = y_train_final.iloc[:early_split]
X_early_val = X_train_final.iloc[early_split:]
y_early_val = y_train_final.iloc[early_split:]

final_model = train_model(X_early_train, y_early_train, X_early_val, y_early_val)

# Retrain on the full training set using the best iteration found during early stopping
best_iter = max(1, int(final_model.best_iteration))
final_model_full = lgb.train(
    {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbose": -1,
    },
    lgb.Dataset(
        data=X_train_final,
        label=y_train_final,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    ),
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(-1)],
)

# Prepare test features using full training data only
X_test = add_features(test, train, final_mapping)
test_pred = final_model_full.predict(X_test)

# Ensure predictions are finite
test_pred = np.nan_to_num(
    test_pred,
    nan=np.nanmean(y_train_final),
    posinf=np.max(y_train_final),
    neginf=np.min(y_train_final),
)

# -----------------------------
# Save submission
# -----------------------------
submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "rating": test_pred,
    }
)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to {SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print(submission.head())
