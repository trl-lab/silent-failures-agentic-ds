import gc
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# 1. Setup paths and parameters
input_dir = "./input"
output_dir = "./working"
submission_dir = "./submission"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

print("Loading raw parquet files...")
train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

print(f"Raw Train shape: {train_df.shape}, Raw Test shape: {test_df.shape}")

# Ensure sorted by TransactionDT for strict time-based split
train_df = train_df.sort_values("TransactionDT").reset_index(drop=True)

# 2. Strict 80/20 Time Split FIRST to avoid temporal leakage
split_idx = int(len(train_df) * 0.8)
train_split = train_df.iloc[:split_idx].copy()
val_split = train_df.iloc[split_idx:].copy()

del train_df
gc.collect()

print(
    f"Train split shape: {train_split.shape}, Validation split shape: {val_split.shape}"
)


def create_base_features(df):
    """Generates time features, D-norm features, composite entity keys, null counts, and velocity features."""
    df = df.copy()

    # Time-of-day and day-of-week features from TransactionDT (seconds)
    df["dt_hour"] = (df["TransactionDT"] // 3600) % 24
    df["dt_day"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["dt_day_idx"] = df["TransactionDT"] / (3600 * 24)

    # Row-wise null count signature
    df["null_count"] = df.isnull().sum(axis=1).astype("int16")

    # D-feature normalization: converting relative day deltas into stationary anchor timestamps for all available D1-D15
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]
    for col in d_cols:
        df[f"{col}_norm"] = df[col] - df["dt_day_idx"]

    # Email domain features
    if "P_emaildomain" in df.columns:
        df["P_email_bin"] = df["P_emaildomain"].astype(str).str.split(".").str[0]
    if "R_emaildomain" in df.columns:
        df["R_email_bin"] = df["R_emaildomain"].astype(str).str.split(".").str[0]

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["P_R_email_match"] = (
            df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)
        ).astype(int)

    # Composite user entity keys
    card1_str = df["card1"].astype(str) if "card1" in df.columns else ""
    card2_str = df["card2"].astype(str) if "card2" in df.columns else ""
    addr1_str = df["addr1"].astype(str) if "addr1" in df.columns else ""
    addr2_str = df["addr2"].astype(str) if "addr2" in df.columns else ""
    p_email_str = (
        df["P_emaildomain"].astype(str) if "P_emaildomain" in df.columns else ""
    )
    r_email_str = (
        df["R_emaildomain"].astype(str) if "R_emaildomain" in df.columns else ""
    )

    df["uid1"] = card1_str + "_" + addr1_str
    df["uid2"] = df["uid1"] + "_" + p_email_str
    df["uid3"] = card1_str + "_" + addr1_str + "_" + p_email_str + "_" + r_email_str
    df["uid4"] = card1_str + "_" + card2_str + "_" + addr1_str + "_" + addr2_str

    # Temporal transaction velocity (time delta since previous transaction per card1 and uid1)
    for key in ["card1", "uid1"]:
        if key in df.columns:
            df[f"{key}_dt_diff"] = df.groupby(key)["TransactionDT"].diff().fillna(-1).astype("float32")

    # Transaction Amount logarithmic transform & fractional part
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_decimal"] = (
            df["TransactionAmt"] - df["TransactionAmt"].astype(int)
        ).round(4)

    return df


print("Creating base domain features...")
train_split = create_base_features(train_split)
val_split = create_base_features(val_split)
test_df = create_base_features(test_df)

# 3. Compute group statistics strictly on train_split
group_cols = [
    "card1",
    "uid1",
    "uid2",
    "uid3",
    "uid4",
    "ProductCD",
    "card2",
    "card3",
    "card5",
    "addr1",
]
d_norm_cols = [f"D{i}_norm" for i in range(1, 16) if f"D{i}_norm" in train_split.columns]
agg_cols = ["TransactionAmt"] + d_norm_cols

new_cols_train = {}
new_cols_val = {}
new_cols_test = {}

for g_col in group_cols:
    if g_col not in train_split.columns:
        continue

    # Frequency encoding
    freq_map = train_split[g_col].value_counts(dropna=False).to_dict()
    new_cols_train[f"{g_col}_freq"] = (
        train_split[g_col].map(freq_map).fillna(0).astype("float32")
    )
    new_cols_val[f"{g_col}_freq"] = (
        val_split[g_col].map(freq_map).fillna(0).astype("float32")
    )
    new_cols_test[f"{g_col}_freq"] = (
        test_df[g_col].map(freq_map).fillna(0).astype("float32")
    )

    for a_col in agg_cols:
        if a_col not in train_split.columns:
            continue

        # Group stats fit on train_split: mean, std, min, max
        stats_df = train_split.groupby(g_col)[a_col].agg(["mean", "std", "min", "max"])
        mean_map = stats_df["mean"].to_dict()
        std_map = stats_df["std"].fillna(0).to_dict()
        min_map = stats_df["min"].to_dict()
        max_map = stats_df["max"].to_dict()

        for df_target, target_dict in [
            (train_split, new_cols_train),
            (val_split, new_cols_val),
            (test_df, new_cols_test),
        ]:
            mean_val = df_target[g_col].map(mean_map).astype("float32")
            std_val = df_target[g_col].map(std_map).astype("float32")
            min_val = df_target[g_col].map(min_map).astype("float32")
            max_val = df_target[g_col].map(max_map).astype("float32")

            target_dict[f"{g_col}_{a_col}_mean"] = mean_val
            target_dict[f"{g_col}_{a_col}_std"] = std_val
            target_dict[f"{g_col}_{a_col}_min"] = min_val
            target_dict[f"{g_col}_{a_col}_max"] = max_val

            # Relative transaction amount ratio & difference for key user entities
            if a_col == "TransactionAmt" and g_col in ["card1", "uid1", "uid2", "uid4"]:
                target_dict[f"{g_col}_TransactionAmt_ratio"] = (
                    df_target["TransactionAmt"] / (mean_val + 1e-5)
                ).astype("float32")
                target_dict[f"{g_col}_TransactionAmt_diff"] = (
                    df_target["TransactionAmt"] - mean_val
                ).astype("float32")

train_split = pd.concat(
    [train_split, pd.DataFrame(new_cols_train, index=train_split.index)], axis=1
)
val_split = pd.concat(
    [val_split, pd.DataFrame(new_cols_val, index=val_split.index)], axis=1
)
test_df = pd.concat(
    [test_df, pd.DataFrame(new_cols_test, index=test_df.index)], axis=1
)

# 4. Handle Categorical Columns with consistent encoding
cat_cols = [
    col
    for col in train_split.columns
    if train_split[col].dtype == "object" or train_split[col].dtype.name == "category"
]

for c in cat_cols:
    train_split[c] = train_split[c].astype(str).fillna("missing")
    val_split[c] = val_split[c].astype(str).fillna("missing")
    test_df[c] = test_df[c].astype(str).fillna("missing")

    unique_cats = train_split[c].unique()
    cat_to_code = {cat: idx for idx, cat in enumerate(unique_cats)}

    train_split[c] = train_split[c].map(cat_to_code).fillna(-1).astype("int32")
    val_split[c] = val_split[c].map(cat_to_code).fillna(-1).astype("int32")
    test_df[c] = test_df[c].map(cat_to_code).fillna(-1).astype("int32")


# 5. Optimize Memory Types (Downcast Float64/Int64)
def downcast_dtypes(df):
    for col in df.columns:
        if col in ["TransactionID", "isFraud"]:
            continue
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype("float32")
        elif col_type == "int64":
            df[col] = df[col].astype("int32")
    return df


train_split = downcast_dtypes(train_split)
val_split = downcast_dtypes(val_split)
test_df = downcast_dtypes(test_df)

gc.collect()

# 6. Extract Features and Target
exclude_cols = ["TransactionID", "isFraud"]
features = [col for col in train_split.columns if col not in exclude_cols]

X_train, y_train = train_split[features], train_split["isFraud"]
X_val, y_val = val_split[features], val_split["isFraud"]
X_test = test_df[features]

print(
    f"Features count: {len(features)}, Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
)

# 7. Model Parameters & LightGBM Datasets
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "max_depth": 12,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.3,
    "reg_lambda": 5.0,
    "scale_pos_weight": 1.0,
    "n_jobs": -1,
    "seed": 42,
    "verbose": -1,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100),
]

train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
)
val_data = lgb.Dataset(
    X_val,
    label=y_val,
    reference=train_data,
    categorical_feature=cat_cols,
    free_raw_data=False,
)

# 8. Train LightGBM & CatBoost Models
print("Starting LightGBM model training...")
lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=callbacks,
)

lgb_val_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

print("Starting CatBoost model training...")
cat_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 100,
    "thread_count": -1,
}

cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=100,
)

cat_val_preds = cat_model.predict_proba(X_val)[:, 1]
cat_test_preds = cat_model.predict_proba(X_test)[:, 1]

print("Starting XGBoost model training...")
xgb_model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=50,
    eval_metric="auc",
    n_jobs=-1,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

xgb_val_preds = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]

# 9. Optimal Weight Optimization using scipy.optimize
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_val = weights[0] * lgb_val_preds + weights[1] * cat_val_preds + weights[2] * xgb_val_preds
    return -roc_auc_score(y_val, blend_val)

init_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x / np.sum(res.x)
print(f"Optimized blend weights: LGB={best_weights[0]:.4f}, CAT={best_weights[1]:.4f}, XGB={best_weights[2]:.4f}")

val_preds = best_weights[0] * lgb_val_preds + best_weights[1] * cat_val_preds + best_weights[2] * xgb_val_preds
test_preds = best_weights[0] * lgb_test_preds + best_weights[1] * cat_test_preds + best_weights[2] * xgb_test_preds

val_auc = roc_auc_score(y_val, val_preds)

print("Generating test predictions...")
submission_path = os.path.join(submission_dir, "submission.csv")
submission_df = pd.DataFrame({"id": np.arange(len(test_df)), "isFraud": test_preds})
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path} with shape {submission_df.shape}")

print(f"Final Validation Score: {val_auc}")
