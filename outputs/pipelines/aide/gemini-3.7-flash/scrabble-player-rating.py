import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


def main():
    # Load dataset
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Ensure output directory exists
    os.makedirs("./working", exist_ok=True)

    # Combine for consistent relational and structural feature engineering
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    if "rating" not in test_df.columns:
        test_df["rating"] = np.nan

    # Preserve original ordering
    train_df["orig_order"] = np.arange(len(train_df))
    test_df["orig_order"] = np.arange(len(test_df))

    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Detect bots
    combined["is_bot"] = (
        combined["nickname"].astype(str).str.contains("Bot|bot", regex=True).astype(int)
    )

    # Opponent and game-level features (paired games by game_id)
    # Sort by game_id to align pairs
    combined = combined.sort_values(by=["game_id", "orig_order"]).reset_index(drop=True)

    # Calculate opponent stats if 2 players per game_id
    game_grouped = combined.groupby("game_id")
    combined["game_player_count"] = game_grouped["nickname"].transform("count")
    combined["game_total_score"] = game_grouped["score"].transform("sum")
    combined["game_mean_score"] = game_grouped["score"].transform("mean")
    combined["game_max_score"] = game_grouped["score"].transform("max")
    combined["game_min_score"] = game_grouped["score"].transform("min")

    # Opponent score and differential
    combined["opp_score"] = combined["game_total_score"] - combined["score"]
    combined["score_diff"] = combined["score"] - combined["opp_score"]
    combined["score_ratio"] = (combined["score"] + 1.0) / (combined["opp_score"] + 1.0)
    combined["is_win"] = (combined["score"] > combined["opp_score"]).astype(float)
    combined["is_tie"] = (combined["score"] == combined["opp_score"]).astype(float)

    # Restore original dataset splits
    train_data = (
        combined[combined["is_test"] == 0]
        .sort_values("orig_order")
        .reset_index(drop=True)
    )
    test_data = (
        combined[combined["is_test"] == 1]
        .sort_values("orig_order")
        .reset_index(drop=True)
    )

    # Player-level aggregate statistics computed strictly from train or aggregated globally where safe
    # Frequency encoding
    freq_map = train_data["nickname"].value_counts()
    train_data["nickname_freq"] = train_data["nickname"].map(freq_map).fillna(0)
    test_data["nickname_freq"] = test_data["nickname"].map(freq_map).fillna(0)

    # Overall player score aggregates from train
    player_stats = (
        train_data.groupby("nickname")
        .agg(
            player_mean_score=("score", "mean"),
            player_std_score=("score", "std"),
            player_min_score=("score", "min"),
            player_max_score=("score", "max"),
            player_median_score=("score", "median"),
            player_win_rate=("is_win", "mean"),
            player_mean_score_diff=("score_diff", "mean"),
        )
        .reset_index()
    )

    train_data = train_data.merge(player_stats, on="nickname", how="left")
    test_data = test_data.merge(player_stats, on="nickname", how="left")

    # Feature columns for LightGBM
    features = [
        "score",
        "game_player_count",
        "game_total_score",
        "game_mean_score",
        "game_max_score",
        "game_min_score",
        "opp_score",
        "score_diff",
        "score_ratio",
        "is_win",
        "is_tie",
        "is_bot",
        "nickname_freq",
        "player_mean_score",
        "player_std_score",
        "player_min_score",
        "player_max_score",
        "player_median_score",
        "player_win_rate",
        "player_mean_score_diff",
    ]

    # Categorical handling for nickname
    train_data["nickname_cat"] = train_data["nickname"].astype("category")
    test_data["nickname_cat"] = pd.Categorical(
        test_data["nickname"], categories=train_data["nickname_cat"].cat.categories
    )
    features.append("nickname_cat")

    # Target variable
    y_train = train_data["rating"].values
    X_train = train_data[features].copy()
    X_test = test_data[features].copy()

    # Out-of-fold target encoding for nickname rating
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train_data))
    test_predictions = np.zeros(len(test_data))

    global_mean_rating = y_train.mean()
    X_train["nickname_target_enc"] = global_mean_rating
    X_test["nickname_target_enc"] = global_mean_rating

    # Calculate out-of-fold target encoding
    for train_idx, val_idx in kf.split(train_data):
        train_fold = train_data.iloc[train_idx]
        te_map = train_fold.groupby("nickname")["rating"].mean()
        X_train.loc[val_idx, "nickname_target_enc"] = (
            train_data.iloc[val_idx]["nickname"].map(te_map).fillna(global_mean_rating)
        )

    # Full train target encoding for test set
    full_te_map = train_data.groupby("nickname")["rating"].mean()
    X_test["nickname_target_enc"] = (
        test_data["nickname"].map(full_te_map).fillna(global_mean_rating)
    )

    model_features = features + ["nickname_target_enc"]

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # 5-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx][model_features], y_train[train_idx]
        X_va, y_val = X_train.iloc[val_idx][model_features], y_train[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        oof_predictions[val_idx] = model.predict(X_va)
        test_predictions += model.predict(X_test[model_features]) / kf.n_splits

    # Compute overall CV MAE
    cv_mae = mean_absolute_error(y_train, oof_predictions)
    print(f"5-Fold CV MAE: {cv_mae:.4f}")

    # Generate submission file
    submission = pd.DataFrame(
        {"id": np.arange(len(test_predictions)), "rating": test_predictions}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(
        "Submission saved successfully to ./working/submission.csv with shape:",
        submission.shape,
    )


if __name__ == "__main__":
    main()
