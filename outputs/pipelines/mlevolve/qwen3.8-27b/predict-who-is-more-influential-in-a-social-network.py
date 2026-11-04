import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution (train): {train_df['Choice'].value_counts().to_dict()}")

# ============================================================
# 2. IDENTIFY FEATURE COLUMNS
# ============================================================
feature_cols = [c for c in train_df.columns if c.startswith(("A_", "B_"))]
print(f"Number of raw features: {len(feature_cols)}")

a_cols = [c for c in feature_cols if c.startswith("A_")]
b_cols = [c for c in feature_cols if c.startswith("B_")]

a_to_b = {}
for a_col in a_cols:
    suffix = a_col[2:]
    b_col = f"B_{suffix}"
    if b_col in b_cols:
        a_to_b[a_col] = b_col

print(f"Paired features: {len(a_to_b)}")


# ============================================================
# 3. FEATURE ENGINEERING FUNCTION
# ============================================================
def engineer_features(df):
    """
    Create comparative features from paired A/B social network metrics.
    """
    df = df.copy()
    new_cols = {}

    # Log transforms
    for col in feature_cols:
        new_cols[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Comparative features for each A-B pair
    for a_col, b_col in a_to_b.items():
        suffix = a_col[2:]
        a_vals = df[a_col].astype(float)
        b_vals = df[b_col].astype(float)

        new_cols[f"diff_{suffix}"] = a_vals - b_vals

        a_safe = a_vals.clip(lower=1e-6)
        b_safe = b_vals.clip(lower=1e-6)
        new_cols[f"log_ratio_{suffix}"] = np.log(a_safe) - np.log(b_safe)

        new_cols[f"abs_diff_{suffix}"] = np.abs(a_vals - b_vals)
        new_cols[f"min_{suffix}"] = np.minimum(a_vals, b_vals)
        new_cols[f"max_{suffix}"] = np.maximum(a_vals, b_vals)

        a_log = np.log1p(a_vals.clip(lower=0))
        b_log = np.log1p(b_vals.clip(lower=0))
        new_cols[f"log_diff_{suffix}"] = a_log - b_log

        denom = (a_vals + b_vals).clip(lower=1e-6)
        new_cols[f"rel_diff_{suffix}"] = (a_vals - b_vals) / denom

    # Aggregate features
    new_cols["log_sum_A"] = sum(new_cols[f"log_{c}"] for c in a_cols)
    new_cols["log_sum_B"] = sum(new_cols[f"log_{c}"] for c in b_cols)
    new_cols["log_sum_diff"] = new_cols["log_sum_A"] - new_cols["log_sum_B"]

    new_cols["log_mean_A"] = np.mean([new_cols[f"log_{c}"] for c in a_cols], axis=0)
    new_cols["log_mean_B"] = np.mean([new_cols[f"log_{c}"] for c in b_cols], axis=0)
    new_cols["log_mean_diff"] = new_cols["log_mean_A"] - new_cols["log_mean_B"]

    # --- Composite/Interaction Features ---

    # (1) Engagement rate features
    for prefix in ['A_', 'B_']:
        posts = df[f"{prefix}posts"].clip(lower=1e-6)
        new_cols[f"{prefix}engagement_mentions"] = df[f"{prefix}mentions_received"] / posts
        new_cols[f"{prefix}engagement_retweets"] = df[f"{prefix}retweets_received"] / posts

    new_cols["diff_engagement_mentions"] = new_cols["A_engagement_mentions"] - new_cols["B_engagement_mentions"]
    new_cols["diff_engagement_retweets"] = new_cols["A_engagement_retweets"] - new_cols["B_engagement_retweets"]

    # (2) Influence density features
    for prefix in ['A_', 'B_']:
        posts = df[f"{prefix}posts"].clip(lower=1e-6)
        new_cols[f"{prefix}influence_density_followers"] = df[f"{prefix}follower_count"] / posts
        new_cols[f"{prefix}influence_density_listed"] = df[f"{prefix}listed_count"] / posts

    new_cols["diff_influence_density_followers"] = new_cols["A_influence_density_followers"] - new_cols["B_influence_density_followers"]
    new_cols["diff_influence_density_listed"] = new_cols["A_influence_density_listed"] - new_cols["B_influence_density_listed"]

    # (3) Cross-metric interactions
    for prefix in ['A_', 'B_']:
        new_cols[f"{prefix}cross_follower_mentions"] = df[f"{prefix}follower_count"] * np.log1p(df[f"{prefix}mentions_received"].clip(lower=0))
        new_cols[f"{prefix}cross_following_retweets"] = df[f"{prefix}following_count"] * np.log1p(df[f"{prefix}retweets_received"].clip(lower=0))

    new_cols["diff_cross_follower_mentions"] = new_cols["A_cross_follower_mentions"] - new_cols["B_cross_follower_mentions"]
    new_cols["diff_cross_following_retweets"] = new_cols["A_cross_following_retweets"] - new_cols["B_cross_following_retweets"]

    # (4) Squared ratio features to capture non-linear dominance
    for a_col, b_col in a_to_b.items():
        suffix = a_col[2:]
        a_vals = df[a_col].astype(float).clip(lower=1e-6)
        b_vals = df[b_col].astype(float).clip(lower=1e-6)
        ratio = a_vals / b_vals
        new_cols[f"squared_ratio_{suffix}"] = (ratio - 1.0) ** 2

    # (5) Product of log ratios across different metric types
    key_metrics = ['follower_count', 'mentions_received', 'retweets_received', 'posts', 'listed_count']
    log_ratios = []
    for metric in key_metrics:
        a_col = f"A_{metric}"
        b_col = f"B_{metric}"
        if a_col in a_to_b:
            a_vals = df[a_col].astype(float).clip(lower=1e-6)
            b_vals = df[b_col].astype(float).clip(lower=1e-6)
            log_ratios.append(np.log(a_vals) - np.log(b_vals))
    if log_ratios:
        new_cols["product_log_ratios_key"] = np.prod(log_ratios, axis=0)
        new_cols["sum_log_ratios_key"] = np.sum(log_ratios, axis=0)

    # Add all new columns at once to avoid fragmentation
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    return df


# ============================================================
# 4. APPLY FEATURE ENGINEERING
# ============================================================
train_eng = engineer_features(train_df)
test_eng = engineer_features(test_df)

final_feature_cols = [c for c in train_eng.columns if c not in ["Choice", "id"]]
print(f"Total features after engineering: {len(final_feature_cols)}")

# ============================================================
# 5. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
y = train_eng["Choice"].values
X = train_eng[final_feature_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test = test_eng[final_feature_cols]

print(f"Train split: {X_train.shape[0]} samples")
print(f"Validation split: {X_val.shape[0]} samples")
print(f"Val target distribution: {pd.Series(y_val).value_counts().to_dict()}")
print(f"Test: {X_test.shape[0]} samples")


# ============================================================
# 6. LIGHTGBM ENSEMBLE TRAINING
# ============================================================
n_features = X_train.shape[1]
print(f"Number of features: {n_features}")

# ============================================================
# 6a. LIGHTGBM NUM_LEAVES HYPERPARAMETER SEARCH
# ============================================================
print(f"\n{'='*50}")
print(f"LightGBM num_leaves search: [15, 25, 31]")
print(f"{'='*50}")

best_num_leaves = 31
best_nl_acc = 0.0

for nl in [15, 25, 31]:
    nl_val_accs = []
    for seed in [42, 123, 456]:
        model_nl = lgb.LGBMClassifier(
            num_leaves=nl,
            learning_rate=0.02,
            n_estimators=5000,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )
        model_nl.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='accuracy',
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        val_p = model_nl.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, (val_p > 0.5).astype(int))
        nl_val_accs.append(acc)
    mean_acc = np.mean(nl_val_accs)
    print(f"  num_leaves={nl}: mean val acc = {mean_acc:.4f} (accs: {[f'{a:.4f}' for a in nl_val_accs]})")
    if mean_acc > best_nl_acc:
        best_nl_acc = mean_acc
        best_num_leaves = nl

print(f"  Selected num_leaves: {best_num_leaves} (acc: {best_nl_acc:.4f})")

# ============================================================
# 6b. LIGHTGBM ENSEMBLE TRAINING (with best num_leaves)
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]
lgb_val_proba = np.zeros(X_val.shape[0])
lgb_test_proba = np.zeros(X_test.shape[0])
seed_val_accs = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*50}")
    print(f"LightGBM Seed {seed_idx+1}/{len(SEEDS)} (seed={seed}, num_leaves={best_num_leaves})")
    print(f"{'='*50}")

    model = lgb.LGBMClassifier(
        num_leaves=best_num_leaves,
        learning_rate=0.02,
        n_estimators=5000,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='accuracy',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    val_preds = (val_proba > 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    seed_val_accs.append(val_acc)

    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Seed {seed} val acc: {val_acc:.4f}")

    lgb_val_proba += val_proba
    lgb_test_proba += test_proba

# ============================================================
# 6c. XGBOOST ENSEMBLE TRAINING
# ============================================================
XGB_SEEDS = [42, 123, 456]
xgb_val_proba = np.zeros(X_val.shape[0])
xgb_test_proba = np.zeros(X_test.shape[0])
xgb_seed_val_accs = []

for seed_idx, seed in enumerate(XGB_SEEDS):
    print(f"\n{'='*50}")
    print(f"XGBoost Seed {seed_idx+1}/{len(XGB_SEEDS)} (seed={seed})")
    print(f"{'='*50}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    xgb_params = {
        'max_depth': 5,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'learning_rate': 0.02,
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'seed': seed,
        'nthread': -1,
        'verbosity': 0
    }

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    val_proba = xgb_model.predict(dval)
    test_proba = xgb_model.predict(dtest)

    val_preds = (val_proba > 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    xgb_seed_val_accs.append(val_acc)

    print(f"  Best iteration: {xgb_model.best_iteration}")
    print(f"  Seed {seed} val acc: {val_acc:.4f}")

    xgb_val_proba += val_proba
    xgb_test_proba += test_proba

# ============================================================
# 7. WEIGHTED ENSEMBLE COMBINATION (1/8 per model)
# ============================================================
n_lgb = len(SEEDS)       # 5
n_xgb = len(XGB_SEEDS)   # 3
n_total = n_lgb + n_xgb  # 8

# Average within each algorithm, then combine with equal weight per model
lgb_avg_val = lgb_val_proba / n_lgb
lgb_avg_test = lgb_test_proba / n_lgb
xgb_avg_val = xgb_val_proba / n_xgb
xgb_avg_test = xgb_test_proba / n_xgb

# Equal weight per model: each model gets 1/8, so LGBM group gets 5/8, XGB group gets 3/8
avg_val_proba = (n_lgb / n_total) * lgb_avg_val + (n_xgb / n_total) * xgb_avg_val
avg_test_proba = (n_lgb / n_total) * lgb_avg_test + (n_xgb / n_total) * xgb_avg_test

val_preds = (avg_val_proba > 0.5).astype(int)
test_preds = (avg_test_proba > 0.5).astype(int)

# ============================================================
# 8. FINAL VALIDATION SCORE
# ============================================================
final_score = accuracy_score(y_val, val_preds)
print(f"\n{'='*50}")
print(f"Ensemble Validation Accuracy: {final_score:.4f}")
print(f"LightGBM per-seed val accs: {[f'{a:.4f}' for a in seed_val_accs]}")
print(f"LightGBM mean val acc: {np.mean(seed_val_accs):.4f}")
print(f"XGBoost per-seed val accs: {[f'{a:.4f}' for a in xgb_seed_val_accs]}")
print(f"XGBoost mean val acc: {np.mean(xgb_seed_val_accs):.4f}")
print(f"Prediction distribution (val): {np.bincount(val_preds).tolist()}")
print(f"Prediction distribution (test): {np.bincount(test_preds).tolist()}")

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

test_ids = (
    test_eng["id"].values if "id" in test_eng.columns else np.arange(len(test_eng))
)
submission = pd.DataFrame({"id": test_ids, "Choice": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")

print(f"Final Validation Score: {final_score:.4f}")
