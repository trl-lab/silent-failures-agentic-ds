import pandas as pd
import numpy as np
import os
import json
import warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Fraud rate in train: {train['isFraud'].mean():.4f}")

# ============================================================
# 2. TIME-BASED SPLIT (respect temporal ordering)
# ============================================================
train = train.sort_values("TransactionDT").reset_index(drop=True)

time_sorted = train["TransactionDT"].values
split_time = np.percentile(time_sorted, 85)

train_mask = train["TransactionDT"] < split_time
val_mask = train["TransactionDT"] >= split_time

train_split = train[train_mask].reset_index(drop=True)
val_split = train[val_mask].reset_index(drop=True)

print(f"Train split: {len(train_split)} rows, val split: {len(val_split)} rows")
print(f"Train fraud rate: {train_split['isFraud'].mean():.4f}")
print(f"Val fraud rate: {val_split['isFraud'].mean():.4f}")
print(f"Time split point: {split_time}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(df, is_train=True, train_df=None):
    """
    Engineer features for fraud detection.
    Features are computed using ONLY training data statistics to prevent leakage.
    """
    df = df.copy()

    # --- Basic temporal features ---
    df["TransactionDT"] = pd.to_numeric(df["TransactionDT"])

    # Hour of day, day of week (relative to epoch)
    df["hour"] = (df["TransactionDT"] % 86400) // 3600
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)

    # --- Transaction amount features ---
    df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(
        0
    )
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    df["amt_squared"] = df["TransactionAmt"] ** 2
    df["amt_cubed"] = df["TransactionAmt"] ** 3
    df["amt_sqrt"] = np.sqrt(df["TransactionAmt"])

    # --- Product features ---
    df["ProductCD"] = df["ProductCD"].fillna("UNKNOWN")
    if train_df is not None:
        product_counts = train_df["ProductCD"].value_counts()
        df["product_freq"] = df["ProductCD"].map(product_counts).fillna(0)

    # --- Card features ---
    for col in ["card1", "card2", "card3", "card4", "card5", "card6"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Address and distance features ---
    for col in ["addr1", "addr2", "dist1", "dist2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interaction features for addr/dist
    if "addr1" in df.columns and "dist1" in df.columns:
        df["addr1_dist1_ratio"] = df["addr1"] / (df["dist1"] + 1)
    if "addr2" in df.columns and "dist2" in df.columns:
        df["addr2_dist2_ratio"] = df["addr2"] / (df["dist2"] + 1)
    if "addr1" in df.columns and "addr2" in df.columns:
        df["addr_diff"] = (df["addr1"] - df["addr2"]).abs()
    if "dist1" in df.columns and "dist2" in df.columns:
        df["dist_diff"] = (df["dist1"] - df["dist2"]).abs()

    # --- C1-C146 features: basic stats ---
    c_cols = [col for col in df.columns if col.startswith("C") and col[1:].isdigit()]
    p_cols = [col for col in df.columns if col.startswith("P") and col[1:].isdigit()]
    d_cols = [col for col in df.columns if col.startswith("D") and col[1:].isdigit()]

    for col in c_cols + p_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- D features (device) ---
    for col in d_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Aggregate statistics across C features ---
    if c_cols:
        c_matrix = df[c_cols].fillna(-1)
        df["C_mean"] = c_matrix.mean(axis=1)
        df["C_std"] = c_matrix.std(axis=1)
        df["C_max"] = c_matrix.max(axis=1)
        df["C_min"] = c_matrix.min(axis=1)
        df["C_range"] = df["C_max"] - df["C_min"]
        df["C_nunique"] = c_matrix.nunique(axis=1)
        df["C_nonzero"] = (c_matrix != 0).sum(axis=1)
        df["C_neg"] = (c_matrix < 0).sum(axis=1)

        # Interaction: C features with amount
        df["C_mean_x_amt"] = df["C_mean"] * df["log_amt"]
        df["C_std_x_amt"] = df["C_std"] * df["log_amt"]

    # --- P features aggregates ---
    if p_cols:
        p_matrix = df[p_cols].fillna(-1)
        df["P_mean"] = p_matrix.mean(axis=1)
        df["P_std"] = p_matrix.std(axis=1)
        df["P_max"] = p_matrix.max(axis=1)
        df["P_min"] = p_matrix.min(axis=1)
        df["P_range"] = df["P_max"] - df["P_min"]

    # --- D features aggregates ---
    if d_cols:
        d_matrix = df[d_cols].fillna(-1)
        df["D_mean"] = d_matrix.mean(axis=1)
        df["D_std"] = d_matrix.std(axis=1)
        df["D_max"] = d_matrix.max(axis=1)
        df["D_min"] = d_matrix.min(axis=1)
        df["D_range"] = df["D_max"] - df["D_min"]

    # --- Cross-group interactions ---
    df["CP_interaction"] = df.get("C_mean", 0) * df.get("P_mean", 0)
    df["CD_interaction"] = df.get("C_mean", 0) * df.get("D_mean", 0)
    df["PD_interaction"] = df.get("P_mean", 0) * df.get("D_mean", 0)

    # --- Time-based behavioral features (per card1-card2 pair) ---
    if "card1" in df.columns and "card2" in df.columns:
        customer_id = df["card1"].astype(str) + "_" + df["card2"].astype(str)

        df = df.sort_values("TransactionDT").reset_index(drop=True)

        # Group by customer and compute time gaps
        df["time_gap"] = df.groupby(customer_id)["TransactionDT"].diff().fillna(0)
        df["log_time_gap"] = np.log1p(df["time_gap"])

        # Rolling statistics (using expanding window to avoid future leakage)
        grp = df.groupby(customer_id)

        # Expanding mean/std of amount
        df["amt_expanding_mean"] = (
            grp["TransactionAmt"]
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(0)
        )
        df["amt_expanding_std"] = (
            grp["TransactionAmt"]
            .transform(lambda x: x.expanding().std().shift(1))
            .fillna(0)
        )
        df["amt_expanding_count"] = (
            grp["TransactionAmt"]
            .transform(lambda x: x.expanding().count().shift(1))
            .fillna(0)
        )

        # Deviation from expanding mean
        df["amt_deviation"] = (df["TransactionAmt"] - df["amt_expanding_mean"]) / (
            df["amt_expanding_std"] + 1
        )

        # Rolling window (last 10 transactions)
        df["amt_rolling_mean_10"] = (
            grp["TransactionAmt"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            .fillna(0)
        )
        df["amt_rolling_std_10"] = (
            grp["TransactionAmt"]
            .transform(lambda x: x.rolling(10, min_periods=1).std().shift(1))
            .fillna(0)
        )
        df["amt_rolling_count_10"] = (
            grp["TransactionAmt"]
            .transform(lambda x: x.rolling(10, min_periods=1).count().shift(1))
            .fillna(0)
        )

        # Rolling time gap stats
        df["timegap_rolling_mean_10"] = (
            grp["time_gap"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            .fillna(0)
        )

        # Customer transaction count so far
        df["cust_txn_count"] = grp["TransactionDT"].cumcount().astype(int)
        df["log_cust_txn_count"] = np.log1p(df["cust_txn_count"])

        # Is this the first transaction for this customer?
        df["is_first_txn"] = (df["cust_txn_count"] == 0).astype(int)

        # Time since customer's first transaction
        first_txn_time = grp["TransactionDT"].transform("first")
        df["time_since_first"] = df["TransactionDT"] - first_txn_time
        df["log_time_since_first"] = np.log1p(df["time_since_first"])

    # --- Product change features ---
    if "ProductCD" in df.columns and "card1" in df.columns:
        customer_id = df["card1"].astype(str) + "_" + df["card2"].astype(str)
        grp = df.groupby(customer_id)
        prev_product = grp["ProductCD"].shift(1)
        df["product_changed"] = (df["ProductCD"] != prev_product).astype(int).fillna(0)

        df["n_unique_products"] = grp["ProductCD"].cumcount() + 1
        # Count unique products seen so far per customer (expanding nunique)
        # Use first-occurrence indicator + cumsum for vectorized expanding nunique
        df["is_first_occurrence"] = df.groupby(customer_id)["ProductCD"].transform(
            lambda x: ~x.duplicated(keep="first")
        ).astype(int)
        df["product_diversity"] = df.groupby(customer_id)[
            "is_first_occurrence"
        ].cumsum()
        df.drop(columns=["is_first_occurrence"], inplace=True)

    # --- Card feature interactions ---
    if "card3" in df.columns and "card4" in df.columns:
        df["card3_card4_interaction"] = df["card3"] * df["card4"]
    if "card5" in df.columns and "card6" in df.columns:
        df["card5_card6_interaction"] = df["card5"] * df["card6"]

    # --- Amount relative to customer history ---
    if "card1" in df.columns:
        df["amt_zscore"] = df["amt_deviation"]

    return df, c_cols, p_cols, d_cols


# ============================================================
# 4. APPLY FEATURE ENGINEERING
# ============================================================
print("Engineering features for train split...")
train_feat, c_cols, p_cols, d_cols = engineer_features(
    train_split, is_train=True, train_df=train_split
)

print("Engineering features for validation split...")
val_feat, _, _, _ = engineer_features(val_split, is_train=False, train_df=train_split)

print("Engineering features for test...")
test_feat, _, _, _ = engineer_features(test, is_train=False, train_df=train_split)

# ============================================================
# 5. FEATURE SELECTION & CLEANUP
# ============================================================
exclude_cols = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
]
feature_cols = [
    col
    for col in train_feat.columns
    if col not in exclude_cols and col not in ["ProductCD"]
]

# Ensure all feature columns are numeric
for col in feature_cols:
    train_feat[col] = pd.to_numeric(train_feat[col], errors="coerce")
    val_feat[col] = pd.to_numeric(val_feat[col], errors="coerce")
    test_feat[col] = pd.to_numeric(test_feat[col], errors="coerce")

# Fill remaining NaNs with -1
train_feat[feature_cols] = train_feat[feature_cols].fillna(-1)
val_feat[feature_cols] = val_feat[feature_cols].fillna(-1)
test_feat[feature_cols] = test_feat[feature_cols].fillna(-1)

# Clip extreme values to prevent numerical issues
for col in feature_cols:
    q01 = train_feat[col].quantile(0.01)
    q99 = train_feat[col].quantile(0.99)
    if q01 < q99:
        train_feat[col] = train_feat[col].clip(q01, q99)
        val_feat[col] = val_feat[col].clip(q01, q99)
        test_feat[col] = test_feat[col].clip(q01, q99)

print(f"\nTotal features: {len(feature_cols)}")
print(f"Train features shape: {train_feat[feature_cols].shape}")
print(f"Val features shape: {val_feat[feature_cols].shape}")
print(f"Test features shape: {test_feat[feature_cols].shape}")

# ============================================================
# 6. MODEL DESIGN: LightGBM for imbalanced tabular fraud detection
# ============================================================
n_pos = train_feat["isFraud"].sum()
n_neg = len(train_feat) - n_pos
scale_pos_weight = n_neg / n_pos
print(
    f"Class imbalance: {n_pos} fraud vs {n_neg} non-fraud (ratio: {scale_pos_weight:.1f}:1)"
)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 50,
    "min_child_weight": 5,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "is_unbalance": False,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# ============================================================
# 7. TRAIN WITH EARLY STOPPING
# ============================================================
X_train = train_feat[feature_cols].values
y_train = train_feat["isFraud"].values.astype(np.int32)

X_val = val_feat[feature_cols].values
y_val = val_feat["isFraud"].values.astype(np.int32)

X_test = test_feat[feature_cols].values

train_lgb = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=feature_cols,
    free_raw_data=False,
)

val_lgb = lgb.Dataset(
    X_val,
    label=y_val,
    feature_name=feature_cols,
    free_raw_data=False,
    reference=train_lgb,
)

print("\nTraining LightGBM model with early stopping...")

model = lgb.train(
    lgb_params,
    train_lgb,
    num_boost_round=3000,
    valid_sets=[train_lgb, val_lgb],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)

best_iteration = model.best_iteration
best_val_auc = model.best_score["valid"]["auc"]
print(f"Best iteration: {best_iteration}")
print(f"Best validation AUC (from LightGBM): {best_val_auc:.6f}")

# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================
val_pred = model.predict(X_val, num_iteration=best_iteration)
test_pred = model.predict(X_test, num_iteration=best_iteration)

# ============================================================
# 9. EVALUATE ON VALIDATION SET (ROC-AUC - official metric)
# ============================================================
val_auc = roc_auc_score(y_val, val_pred)
print(f"\nValidation ROC-AUC: {val_auc:.6f}")

# ============================================================
# 10. FEATURE IMPORTANCE (top 20)
# ============================================================
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame(
    {"feature": feature_cols, "importance": importance}
).sort_values("importance", ascending=False)

print("\nTop 20 features by gain importance:")
print(feat_imp.head(20).to_string(index=False))

# ============================================================
# 11. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "isFraud": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, min={test_pred.min():.4f}, max={test_pred.max():.4f}"
)

# ============================================================
# 12. FINAL OUTPUT
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")