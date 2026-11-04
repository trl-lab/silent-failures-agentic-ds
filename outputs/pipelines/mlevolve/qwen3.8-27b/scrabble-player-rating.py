import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import os
import joblib

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING & FEATURE ENGINEERING
# ============================================================

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Check if test has rating column
if "rating" in test_df.columns:
    test_df = test_df.drop(columns=["rating"])

# Add id column for test
test_df["id"] = range(len(test_df))

# Player-level train/validation split (no leakage)
player_groups = train_df["nickname"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(gss.split(train_df, groups=player_groups))

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train split: {train_split.shape}, Val split: {val_split.shape}")
print(f"Unique players in train split: {train_split['nickname'].nunique()}")
print(f"Unique players in val split: {val_split['nickname'].nunique()}")
print(
    f"Overlap in players: {len(set(train_split['nickname']) & set(val_split['nickname']))}"
)

# Compute player-level features from TRAIN SPLIT only (no leakage)
player_stats = (
    train_split.groupby("nickname")
    .agg(
        player_mean_score=("score", "mean"),
        player_std_score=("score", "std"),
        player_min_score=("score", "min"),
        player_max_score=("score", "max"),
        player_median_score=("score", "median"),
        player_num_games=("score", "count"),
        player_score_range=("score", lambda x: x.max() - x.min()),
        player_score_q25=("score", lambda x: x.quantile(0.25)),
        player_score_q75=("score", lambda x: x.quantile(0.75)),
        player_score_skew=("score", lambda x: x.skew() if len(x) > 2 else 0),
        player_score_kurt=("score", lambda x: x.kurtosis() if len(x) > 3 else 0),
    )
    .reset_index()
)

# Player score percentile rank within their own games
def _percentile_rank(s):
    return s.rank(pct=True).mean()

# Player score trend (linear regression slope of score vs game_id)
def _score_trend(df):
    if len(df) < 2:
        return 0.0
    x = df["game_id"].values
    y = df["score"].values
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return ((x - x_mean) * (y - y_mean)).sum() / denom

# Player score volatility (coefficient of variation)
def _score_cv(s):
    m = s.mean()
    if abs(m) < 1e-8:
        return 0.0
    return s.std() / abs(m)

# Player score momentum (recent vs early games comparison)
def _score_momentum(df):
    if len(df) < 4:
        return 0.0
    df_sorted = df.sort_values("game_id")
    n = len(df_sorted)
    half = n // 2
    early = df_sorted.iloc[:half]["score"].mean()
    recent = df_sorted.iloc[half:]["score"].mean()
    return recent - early

# Compute additional score-based player features
player_extra_stats = (
    train_split.groupby("nickname")
    .apply(
        lambda g: pd.Series(
            {
                "player_score_percentile_rank": _percentile_rank(g["score"]),
                "player_score_trend": _score_trend(g),
                "player_score_volatility": _score_cv(g["score"]),
                "player_score_momentum": _score_momentum(g),
            }
        )
    )
    .reset_index()
)

# Game_id-based temporal features
train_split = train_split.sort_values(["nickname", "game_id"]).reset_index(drop=True)
val_split = val_split.sort_values(["nickname", "game_id"]).reset_index(drop=True)
test_df = test_df.sort_values(["nickname", "game_id"]).reset_index(drop=True)

# games_before: count of earlier games by same player (within the dataset)
train_split["games_before"] = train_split.groupby("nickname").cumcount()
val_split["games_before"] = val_split.groupby("nickname").cumcount()
test_df["games_before"] = test_df.groupby("nickname").cumcount()

# time_since_last_game: difference from previous game_id for same player
train_split["time_since_last_game"] = train_split.groupby("nickname")["game_id"].diff().fillna(train_split["game_id"])
val_split["time_since_last_game"] = val_split.groupby("nickname")["game_id"].diff().fillna(val_split["game_id"])
test_df["time_since_last_game"] = test_df.groupby("nickname")["game_id"].diff().fillna(test_df["game_id"])

# Merge extra player stats
player_stats = player_stats.merge(player_extra_stats, on="nickname", how="left")

# Compute player stats from FULL train_df for test set (consistent with final model training)
full_player_stats_test = (
    train_df.groupby("nickname")
    .agg(
        player_mean_score=("score", "mean"),
        player_std_score=("score", "std"),
        player_min_score=("score", "min"),
        player_max_score=("score", "max"),
        player_median_score=("score", "median"),
        player_num_games=("score", "count"),
        player_score_range=("score", lambda x: x.max() - x.min()),
        player_score_q25=("score", lambda x: x.quantile(0.25)),
        player_score_q75=("score", lambda x: x.quantile(0.75)),
        player_score_skew=("score", lambda x: x.skew() if len(x) > 2 else 0),
        player_score_kurt=("score", lambda x: x.kurtosis() if len(x) > 3 else 0),
    )
    .reset_index()
)

full_player_extra_stats_test = (
    train_df.groupby("nickname")
    .apply(
        lambda g: pd.Series(
            {
                "player_score_percentile_rank": _percentile_rank(g["score"]),
                "player_score_trend": _score_trend(g),
                "player_score_volatility": _score_cv(g["score"]),
                "player_score_momentum": _score_momentum(g),
            }
        )
    )
    .reset_index()
)

full_player_stats_test = full_player_stats_test.merge(full_player_extra_stats_test, on="nickname", how="left")

# Merge player stats back to train_split and val_split (from train_split only - no leakage)
train_split = train_split.merge(player_stats, on="nickname", how="left")
val_split = val_split.merge(player_stats, on="nickname", how="left")
# Merge full-train player stats to test (consistent with final model training features)
test_df = test_df.merge(full_player_stats_test, on="nickname", how="left")

# Impute for players not found
global_mean_score = train_split["score"].mean()

for col in player_stats.columns:
    if col != "nickname":
        fill_val = 0 if col in ["player_num_games"] else global_mean_score
        train_split[col] = train_split[col].fillna(fill_val)
        val_split[col] = val_split[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)

# Additional features
train_split["score_relative_to_player_mean"] = (
    train_split["score"] - train_split["player_mean_score"]
)
val_split["score_relative_to_player_mean"] = (
    val_split["score"] - val_split["player_mean_score"]
)
test_df["score_relative_to_player_mean"] = (
    test_df["score"] - test_df["player_mean_score"]
)

train_split["score_zscore_player"] = (
    train_split["score"] - train_split["player_mean_score"]
) / (train_split["player_std_score"] + 1e-8)
val_split["score_zscore_player"] = (
    val_split["score"] - val_split["player_mean_score"]
) / (val_split["player_std_score"] + 1e-8)
test_df["score_zscore_player"] = (
    test_df["score"] - test_df["player_mean_score"]
) / (test_df["player_std_score"] + 1e-8)

# Game ID features (temporal) - use full train max for consistency with final model
game_id_max = train_df["game_id"].max()
train_split["game_id_normalized"] = train_split["game_id"] / game_id_max
val_split["game_id_normalized"] = val_split["game_id"] / game_id_max
test_df["game_id_normalized"] = test_df["game_id"] / game_id_max

# Score transformation features
train_split["score_squared"] = train_split["score"] ** 2
val_split["score_squared"] = val_split["score"] ** 2
test_df["score_squared"] = test_df["score"] ** 2
train_split["score_log"] = np.log1p(train_split["score"].clip(lower=0))
val_split["score_log"] = np.log1p(val_split["score"].clip(lower=0))
test_df["score_log"] = np.log1p(test_df["score"].clip(lower=0))
train_split["score_sqrt"] = np.sqrt(train_split["score"].clip(lower=0))
val_split["score_sqrt"] = np.sqrt(val_split["score"].clip(lower=0))
test_df["score_sqrt"] = np.sqrt(test_df["score"].clip(lower=0))

# Interaction features
train_split["score_x_num_games"] = train_split["score"] * train_split["player_num_games"]
val_split["score_x_num_games"] = val_split["score"] * val_split["player_num_games"]
test_df["score_x_num_games"] = test_df["score"] * test_df["player_num_games"]

# Define feature columns
exclude_cols = ["id", "nickname", "rating", "game_id"]
feature_cols = [c for c in train_split.columns if c not in exclude_cols]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# ============================================================
# STEP 2 & 3: MODEL TRAINING & EVALUATION
# ============================================================

# Ensure test_df has all feature columns (in case some player stats columns differ)
for col in feature_cols:
    if col not in test_df.columns:
        test_df[col] = 0.0

# Prepare feature matrices
X_train = train_split[feature_cols].values.astype(np.float64)
y_train = train_split["rating"].values.astype(np.float64)
X_val = val_split[feature_cols].values.astype(np.float64)
y_val = val_split["rating"].values.astype(np.float64)
# Re-derive X_test after ensuring all columns exist
X_test = test_df[feature_cols].values.astype(np.float64)

# Define LightGBM Model 1: L1 objective, shallow trees, low learning rate
model1 = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=3.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Define LightGBM Model 2: L1 objective, deeper trees, more regularization
model2 = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=10,
    num_leaves=255,
    min_child_samples=15,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=2.0,
    reg_lambda=5.0,
    random_state=123,
    n_jobs=-1,
    verbose=-1,
)

# Train Model 1 with early stopping
callbacks1 = [
    lgb.early_stopping(stopping_rounds=500, verbose=False),
    lgb.log_evaluation(period=500),
]

model1.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=callbacks1,
)

print(f"Model 1 best iteration: {model1.best_iteration_}")

# Train Model 2 with early stopping
callbacks2 = [
    lgb.early_stopping(stopping_rounds=500, verbose=False),
    lgb.log_evaluation(period=500),
]

model2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=callbacks2,
)

print(f"Model 2 best iteration: {model2.best_iteration_}")

# Validate: ensemble prediction on validation set
val_preds1 = model1.predict(X_val)
val_preds2 = model2.predict(X_val)
val_preds = 0.5 * val_preds1 + 0.5 * val_preds2
val_mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MAE (ensemble): {val_mae:.4f}")
print(f"Validation MAE (model1): {mean_absolute_error(y_val, val_preds1):.4f}")
print(f"Validation MAE (model2): {mean_absolute_error(y_val, val_preds2):.4f}")

# Retrain on FULL training data with the best iteration counts
best_iter1 = model1.best_iteration_
best_iter2 = model2.best_iteration_
print(f"Retraining on full training data with model1={best_iter1}, model2={best_iter2} iterations...")

# Apply same feature engineering to full train_df
full_player_stats = (
    train_df.groupby("nickname")
    .agg(
        player_mean_score=("score", "mean"),
        player_std_score=("score", "std"),
        player_min_score=("score", "min"),
        player_max_score=("score", "max"),
        player_median_score=("score", "median"),
        player_num_games=("score", "count"),
        player_score_range=("score", lambda x: x.max() - x.min()),
        player_score_q25=("score", lambda x: x.quantile(0.25)),
        player_score_q75=("score", lambda x: x.quantile(0.75)),
        player_score_skew=("score", lambda x: x.skew() if len(x) > 2 else 0),
        player_score_kurt=("score", lambda x: x.kurtosis() if len(x) > 3 else 0),
    )
    .reset_index()
)

# Compute extra score-based features for full training data
full_player_extra_stats = (
    train_df.groupby("nickname")
    .apply(
        lambda g: pd.Series(
            {
                "player_score_percentile_rank": _percentile_rank(g["score"]),
                "player_score_trend": _score_trend(g),
                "player_score_volatility": _score_cv(g["score"]),
                "player_score_momentum": _score_momentum(g),
            }
        )
    )
    .reset_index()
)

full_player_stats = full_player_stats.merge(full_player_extra_stats, on="nickname", how="left")
train_df = train_df.merge(full_player_stats, on="nickname", how="left")

# Temporal features for full training data
train_df = train_df.sort_values(["nickname", "game_id"]).reset_index(drop=True)
train_df["games_before"] = train_df.groupby("nickname").cumcount()
train_df["time_since_last_game"] = train_df.groupby("nickname")["game_id"].diff().fillna(train_df["game_id"])

for col in full_player_stats.columns:
    if col != "nickname":
        fill_val = 0 if col in ["player_num_games"] else train_df["score"].mean()
        train_df[col] = train_df[col].fillna(fill_val)

train_df["score_relative_to_player_mean"] = train_df["score"] - train_df["player_mean_score"]
train_df["score_zscore_player"] = (
    train_df["score"] - train_df["player_mean_score"]
) / (train_df["player_std_score"] + 1e-8)
train_df["game_id_normalized"] = train_df["game_id"] / game_id_max
train_df["score_squared"] = train_df["score"] ** 2
train_df["score_log"] = np.log1p(train_df["score"].clip(lower=0))
train_df["score_sqrt"] = np.sqrt(train_df["score"].clip(lower=0))
train_df["score_x_num_games"] = train_df["score"] * train_df["player_num_games"]

full_model1 = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=best_iter1,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=3.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

full_model2 = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=best_iter2,
    learning_rate=0.02,
    max_depth=10,
    num_leaves=255,
    min_child_samples=15,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=2.0,
    reg_lambda=5.0,
    random_state=123,
    n_jobs=-1,
    verbose=-1,
)

# Prepare full training data
X_full = train_df[feature_cols].values.astype(np.float64)
y_full = train_df["rating"].values.astype(np.float64)

full_model1.fit(X_full, y_full)
full_model2.fit(X_full, y_full)

# Save the best models
joblib.dump(full_model1, "./working/best_model1.joblib")
joblib.dump(full_model2, "./working/best_model2.joblib")

# Feature importance
importance = pd.DataFrame(
    {"feature": feature_cols, "importance": full_model1.feature_importances_}
).sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

# Recompute X_test using the same pipeline as full training (features from full train_df)
# Re-sort and recompute temporal features for test consistent with full model
test_df_sorted = test_df.sort_values(["nickname", "game_id"]).reset_index(drop=True)
test_df_sorted["games_before"] = test_df_sorted.groupby("nickname").cumcount()
test_df_sorted["time_since_last_game"] = test_df_sorted.groupby("nickname")["game_id"].diff().fillna(test_df_sorted["game_id"])

# Merge full player stats (already done above via full_player_stats_test)
# Ensure game_id_normalized uses same game_id_max
test_df_sorted["game_id_normalized"] = test_df_sorted["game_id"] / game_id_max

# Recompute derived features
test_df_sorted["score_relative_to_player_mean"] = test_df_sorted["score"] - test_df_sorted["player_mean_score"]
test_df_sorted["score_zscore_player"] = (
    test_df_sorted["score"] - test_df_sorted["player_mean_score"]
) / (test_df_sorted["player_std_score"] + 1e-8)
test_df_sorted["score_squared"] = test_df_sorted["score"] ** 2
test_df_sorted["score_log"] = np.log1p(test_df_sorted["score"].clip(lower=0))
test_df_sorted["score_sqrt"] = np.sqrt(test_df_sorted["score"].clip(lower=0))
test_df_sorted["score_x_num_games"] = test_df_sorted["score"] * test_df_sorted["player_num_games"]

X_test_final = test_df_sorted[feature_cols].values.astype(np.float64)

# Test inference using ensemble of full-data models
test_preds1 = full_model1.predict(X_test_final)
test_preds2 = full_model2.predict(X_test_final)
test_preds = 0.5 * test_preds1 + 0.5 * test_preds2

# Generate submission
submission = pd.DataFrame({"id": test_df_sorted["id"].astype(int), "rating": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction range: [{test_preds.min():.2f}, {test_preds.max():.2f}]")

# Final validation score (official metric: MAE)
score = val_mae
print(f"Final Validation Score: {score}")