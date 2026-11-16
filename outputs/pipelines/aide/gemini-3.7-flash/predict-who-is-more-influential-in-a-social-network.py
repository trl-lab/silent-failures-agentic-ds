import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

base_features = [
    "follower_count",
    "following_count",
    "listed_count",
    "mentions_received",
    "mentions_sent",
    "network_feature_1",
    "network_feature_2",
    "network_feature_3",
    "posts",
    "retweets_received",
    "retweets_sent",
]


def swap_pairwise_df(df):
    swapped = df.copy()
    for feat in base_features:
        a_col = f"A_{feat}"
        b_col = f"B_{feat}"
        swapped[a_col] = df[b_col].values
        swapped[b_col] = df[a_col].values
    return swapped


# 2. Pairwise & Intra-Profile Feature Engineering
def create_features(df):
    feats = pd.DataFrame(index=df.index)

    for prefix in ["A", "B"]:
        followers = df[f"{prefix}_follower_count"].astype(float)
        following = df[f"{prefix}_following_count"].astype(float)
        listed = df[f"{prefix}_listed_count"].astype(float)
        posts = df[f"{prefix}_posts"].astype(float)
        retweets_rec = df[f"{prefix}_retweets_received"].astype(float)
        mentions_rec = df[f"{prefix}_mentions_received"].astype(float)
        net1 = df[f"{prefix}_network_feature_1"].astype(float)
        net2 = df[f"{prefix}_network_feature_2"].astype(float)

        feats[f"{prefix}_log_follower"] = np.log1p(np.maximum(0, followers))
        feats[f"{prefix}_log_following"] = np.log1p(np.maximum(0, following))
        feats[f"{prefix}_log_listed"] = np.log1p(np.maximum(0, listed))
        feats[f"{prefix}_follower_to_following"] = np.log1p(
            np.maximum(0, followers)
        ) - np.log1p(np.maximum(0, following))
        feats[f"{prefix}_listed_to_follower"] = (listed + 1.0) / (followers + 1.0)
        feats[f"{prefix}_retweets_per_post"] = (retweets_rec + 1.0) / (posts + 1.0)
        feats[f"{prefix}_mentions_per_post"] = (mentions_rec + 1.0) / (posts + 1.0)
        feats[f"{prefix}_retweets_to_mentions"] = (retweets_rec + 1.0) / (
            mentions_rec + 1.0
        )
        feats[f"{prefix}_net_ratio"] = (net1 + 1.0) / (net2 + 1.0)

    for feat in base_features:
        a_val = df[f"A_{feat}"].astype(float)
        b_val = df[f"B_{feat}"].astype(float)

        feats[f"A_{feat}"] = a_val
        feats[f"B_{feat}"] = b_val
        feats[f"{feat}_diff"] = a_val - b_val

        log_a = np.log1p(np.maximum(0, a_val))
        log_b = np.log1p(np.maximum(0, b_val))
        feats[f"{feat}_log_diff"] = log_a - log_b
        feats[f"{feat}_rel_diff"] = (a_val - b_val) / (a_val + b_val + 1.0)
        feats[f"{feat}_ratio"] = (a_val + 1.0) / (b_val + 1.0)

    feats["ff_ratio_diff"] = (
        feats["A_follower_to_following"] - feats["B_follower_to_following"]
    )
    feats["listed_ratio_diff"] = (
        feats["A_listed_to_follower"] - feats["B_listed_to_follower"]
    )
    feats["retweets_per_post_diff"] = (
        feats["A_retweets_per_post"] - feats["B_retweets_per_post"]
    )
    feats["mentions_per_post_diff"] = (
        feats["A_mentions_per_post"] - feats["B_mentions_per_post"]
    )

    return feats


X_train_raw = create_features(train_df)
y_train = train_df["Choice"].values

X_test_orig = create_features(test_df)
X_test_swapped = create_features(swap_pairwise_df(test_df))

# 3. Model Training with 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
    train_fold_df = train_df.iloc[train_idx].copy()
    val_fold_df = train_df.iloc[val_idx].copy()

    X_tr_orig = create_features(train_fold_df)
    y_tr_orig = y_train[train_idx]

    X_tr_swap = create_features(swap_pairwise_df(train_fold_df))
    y_tr_swap = 1 - y_tr_orig

    X_train_aug = pd.concat([X_tr_orig, X_tr_swap], ignore_index=True)
    y_train_aug = np.concatenate([y_tr_orig, y_tr_swap])

    X_val_orig = create_features(val_fold_df)
    X_val_swap = create_features(swap_pairwise_df(val_fold_df))
    y_val_orig = y_train[val_idx]

    trn_data = lgb.Dataset(X_train_aug, label=y_train_aug)
    val_data = lgb.Dataset(X_val_orig, label=y_val_orig, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    # Symmetric inference
    val_p_orig = model.predict(X_val_orig, num_iteration=model.best_iteration)
    val_p_swap = model.predict(X_val_swap, num_iteration=model.best_iteration)
    val_probs = 0.5 * (val_p_orig + (1.0 - val_p_swap))

    oof_preds[val_idx] = val_probs

    test_p_orig = model.predict(X_test_orig, num_iteration=model.best_iteration)
    test_p_swap = model.predict(X_test_swapped, num_iteration=model.best_iteration)
    test_preds += 0.5 * (test_p_orig + (1.0 - test_p_swap)) / n_splits

    fold_acc = accuracy_score(y_val_orig, (val_probs >= 0.5).astype(int))
    print(f"Fold {fold + 1} Accuracy: {fold_acc:.5f}")

# 4. Evaluation
oof_binary = (oof_preds >= 0.5).astype(int)
overall_accuracy = accuracy_score(y_train, oof_binary)
print(f"\nOverall Out-of-Fold Accuracy: {overall_accuracy:.5f}")

# 5. Create Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Choice": (test_preds >= 0.5).astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv successfully.")
