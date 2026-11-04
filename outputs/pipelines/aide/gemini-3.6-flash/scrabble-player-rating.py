import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Flag train and test
train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["rating"] = np.nan

# Combine datasets for feature engineering
df = pd.concat([train_df, test_df], ignore_index=True)

# Opponent pairing per game
df["turn_order"] = df.groupby("game_id").cumcount()
opp_df = df[["game_id", "turn_order", "nickname", "score"]].copy()
opp_df["turn_order"] = 1 - opp_df["turn_order"]
opp_df.rename(columns={"nickname": "opp_nickname", "score": "opp_score"}, inplace=True)

df = pd.merge(df, opp_df, on=["game_id", "turn_order"], how="left")
df["opp_score"] = df["opp_score"].fillna(df["score"])
df["opp_nickname"] = df["opp_nickname"].fillna("Unknown")

# Score differential interaction features
df["score_diff"] = df["score"] - df["opp_score"]
df["score_total"] = df["score"] + df["opp_score"]
df["score_ratio"] = (df["score"] + 100.0) / (df["opp_score"] + 100.0)
df["is_win"] = (df["score_diff"] > 0).astype(int)


# Bot detection features
def identify_bot(name_series):
    return name_series.str.contains(
        "Bot", case=False, na=False
    ) | name_series.str.contains("BB-8", case=False, na=False)


df["is_bot"] = identify_bot(df["nickname"]).astype(int)
df["opp_is_bot"] = identify_bot(df["opp_nickname"]).astype(int)
df["bot_match"] = df["is_bot"] * 2 + df["opp_is_bot"]

# Parse datetime and additional meta-features if present
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["year"] = df["created_at"].dt.year
    df["month"] = df["created_at"].dt.month
    df["day"] = df["created_at"].dt.day
    df["dayofweek"] = df["created_at"].dt.dayofweek
    df["hour"] = df["created_at"].dt.hour
    df["timestamp"] = df["created_at"].astype("int64") // 10**9

extra_cats = [
    col for col in ["rating_mode", "lexicon", "time_control_name"] if col in df.columns
]
for cat in extra_cats:
    df[cat] = df[cat].astype("category").cat.codes

# Historical player performance aggregations
player_stats = (
    df.groupby("nickname")
    .agg(
        p_games=("game_id", "count"),
        p_mean_score=("score", "mean"),
        p_std_score=("score", "std"),
        p_max_score=("score", "max"),
        p_min_score=("score", "min"),
        p_mean_diff=("score_diff", "mean"),
        p_std_diff=("score_diff", "std"),
        p_win_rate=("is_win", "mean"),
    )
    .reset_index()
)

opp_stats = player_stats.copy()
opp_stats.columns = [
    "opp_" + col if col != "nickname" else "opp_nickname" for col in opp_stats.columns
]

df = pd.merge(df, player_stats, on="nickname", how="left")
df = pd.merge(df, opp_stats, on="opp_nickname", how="left")

# Categorical label encoding
all_nicknames = pd.concat([df["nickname"], df["opp_nickname"]]).unique()
nick_map = {name: idx for idx, name in enumerate(all_nicknames)}
df["nickname_code"] = df["nickname"].map(nick_map).fillna(-1).astype(int)
df["opp_nickname_code"] = df["opp_nickname"].map(nick_map).fillna(-1).astype(int)

# Separate train and test sets
df_train = df[df["is_train"] == 1].reset_index(drop=True)
df_test = df[df["is_train"] == 0].reset_index(drop=True)

drop_cols = ["game_id", "nickname", "opp_nickname", "rating", "is_train", "created_at"]
feature_cols = [c for c in df_train.columns if c not in drop_cols]

for c in feature_cols:
    df_train[c] = df_train[c].fillna(-999)
    df_test[c] = df_test[c].fillna(-999)

# GroupKFold CV and out-of-fold target encoding
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(df_train))
test_preds = np.zeros(len(df_test))
m_smooth = 10.0

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df_train, df_train["rating"], groups=df_train["game_id"])
):
    tr_df = df_train.iloc[train_idx].copy()
    va_df = df_train.iloc[val_idx].copy()
    te_df = df_test.copy()

    global_mean = tr_df["rating"].mean()

    # Bayesian target encoding
    p_target = tr_df.groupby("nickname")["rating"].agg(["count", "sum"])
    p_te = (
        (p_target["sum"] + m_smooth * global_mean) / (p_target["count"] + m_smooth)
    ).to_dict()

    tr_df["p_target_enc"] = tr_df["nickname"].map(p_te).fillna(global_mean)
    va_df["p_target_enc"] = va_df["nickname"].map(p_te).fillna(global_mean)
    te_df["p_target_enc"] = te_df["nickname"].map(p_te).fillna(global_mean)

    tr_df["opp_target_enc"] = tr_df["opp_nickname"].map(p_te).fillna(global_mean)
    va_df["opp_target_enc"] = va_df["opp_nickname"].map(p_te).fillna(global_mean)
    te_df["opp_target_enc"] = te_df["opp_nickname"].map(p_te).fillna(global_mean)

    fold_features = feature_cols + ["p_target_enc", "opp_target_enc"]

    X_tr = tr_df[fold_features]
    y_tr = tr_df["rating"]
    X_va = va_df[fold_features]
    y_va = va_df["rating"]
    X_te = te_df[fold_features]

    model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        eval_metric="mae",
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_te) / gkf.n_splits

# Compute evaluation metric
oof_mae = mean_absolute_error(df_train["rating"], oof_preds)
print(f"Validation MAE: {oof_mae:.4f}")

# Save output predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(df_test)), "rating": test_preds})
submission.to_csv("./working/submission.csv", index=False)
