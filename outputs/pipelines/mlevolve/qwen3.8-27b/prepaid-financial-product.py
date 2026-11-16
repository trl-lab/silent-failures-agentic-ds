import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "current_status"
id_col = "id" if "id" in train.columns else None

y_raw = train[target_col].values
X = train.drop(columns=[target_col])

# Convert target to binary 0/1
if y_raw.dtype == object or y_raw.dtype == "category":
    unique_vals = sorted(pd.unique(y_raw))
    print(f"Target unique values: {unique_vals}")
    # Map the second value (or '1'/'yes'/'active' etc.) to 1, first to 0
    # Use a deterministic mapping: sort alphabetically, last value = 1
    val_to_num = {v: i for i, v in enumerate(unique_vals)}
    y = np.array([val_to_num[v] for v in y_raw])
else:
    y = y_raw.astype(int)
print(f"Target dtype: {y.dtype}, unique: {np.unique(y)}, mean: {y.mean():.4f}")
X_test = test.copy()

if id_col and id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col and id_col in X_test.columns:
    X_test = X_test.drop(columns=[id_col])

# Identify feature types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in X.columns if X[c].dtype == bool]

for c in bool_cols:
    X[c] = X[c].astype(int)
    X_test[c] = X_test[c].astype(int)
    if c in categorical_cols:
        categorical_cols.remove(c)
    if c not in numeric_cols:
        numeric_cols.append(c)

print(
    f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}"
)

# ============================================================
# 2. CREATE TRAIN/VALIDATION SPLIT (FIRST, before any fitting)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")
print(f"Train target mean: {y_train.mean():.4f}, Val target mean: {y_val.mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df,
    numeric_cols,
    categorical_cols,
    freq_maps=None,
    target_enc_maps=None,
    numeric_stats=None,
    fit=False,
):
    df = df.copy()

    # Missing value indicators
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # Frequency encoding
    if fit:
        freq_maps = {}
        for col in categorical_cols:
            if col in df.columns:
                freq_maps[col] = df[col].value_counts(normalize=True)
    else:
        freq_maps = freq_maps or {}

    for col in categorical_cols:
        if col in df.columns:
            if fit:
                df[f"{col}_freq"] = df[col].map(freq_maps.get(col, pd.Series()))
            else:
                df[f"{col}_freq"] = (
                    df[col].map(freq_maps.get(col, pd.Series())).fillna(0)
                )

    # Target encoding (applied separately)
    if not fit and target_enc_maps is not None:
        for col in categorical_cols:
            if col in df.columns and col in target_enc_maps:
                df[f"{col}_target_enc"] = (
                    df[col]
                    .map(target_enc_maps[col])
                    .fillna(target_enc_maps.get("__global_mean__", 0.5))
                )

    # Numerical stats
    if fit:
        numeric_stats = {}
        for col in numeric_cols:
            if col in df.columns:
                numeric_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "median": df[col].median(),
                }
    else:
        numeric_stats = numeric_stats or {}

    # Log transform
    for col in numeric_cols:
        if col in df.columns:
            col_vals = df[col].dropna()
            if len(col_vals) > 0 and col_vals.min() > 0:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            elif len(col_vals) > 0:
                df[f"{col}_log"] = np.sign(df[col]) * np.log1p(df[col].abs())

    # Z-score
    for col in numeric_cols:
        if col in df.columns and col in numeric_stats:
            stats = numeric_stats[col]
            if stats["std"] and stats["std"] > 0:
                df[f"{col}_zscore"] = (df[col] - stats["mean"]) / stats["std"]

    # Interaction features
    num_in_df = [c for c in numeric_cols if c in df.columns]
    if len(num_in_df) >= 2:
        for i in range(min(len(num_in_df), 15)):
            for j in range(i + 1, min(len(num_in_df), 15)):
                c1, c2 = num_in_df[i], num_in_df[j]
                if df[c2].abs().sum() > 0:
                    df[f"{c1}_div_{c2}"] = df[c1] / (df[c2].abs() + 1e-8)

    # Categorical interactions
    cat_in_df = [c for c in categorical_cols if c in df.columns]
    if len(cat_in_df) >= 2:
        for i in range(min(len(cat_in_df), 5)):
            for j in range(i + 1, min(len(cat_in_df), 5)):
                c1, c2 = cat_in_df[i], cat_in_df[j]
                df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    # Aggregate features
    df["_n_nonnull"] = df.notna().sum(axis=1)
    df["_n_null"] = df.isna().sum(axis=1)

    return df, freq_maps, numeric_stats


def compute_target_encoding_cv(X_train, y_train, categorical_cols, n_splits=5):
    """Compute OOF target encodings for training, and full-train maps for val/test."""
    oof_enc = np.zeros((len(X_train), len(categorical_cols)))
    target_enc_maps = {}
    global_mean = y_train.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for idx, col in enumerate(categorical_cols):
        if col not in X_train.columns:
            continue
        oof_enc[:, idx] = global_mean
        for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
            fold_map = pd.Series(y_train[fold_train_idx]).groupby(
                X_train[col].iloc[fold_train_idx]
            ).mean()
            oof_enc[fold_val_idx, idx] = (
                X_train[col].iloc[fold_val_idx].map(fold_map).fillna(global_mean).values
            )
        # Full training map for val/test
        full_map = pd.Series(y_train).groupby(X_train[col].values).mean()
        target_enc_maps[col] = full_map
    target_enc_maps["__global_mean__"] = global_mean
    return target_enc_maps, oof_enc


def finalize_features(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in ["float64", "float32"]:
                df[col] = df[col].fillna(-999.0)
            else:
                df[col] = df[col].fillna(0)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-999.0)
            except:
                df[col] = df[col].astype("category")
    return df


# Apply feature engineering
print("Engineering features...")
X_train_eng, freq_maps, numeric_stats = engineer_features(
    X_train, numeric_cols, categorical_cols, fit=True
)
target_enc_maps, oof_enc = compute_target_encoding_cv(X_train, y_train, categorical_cols)

for idx, col in enumerate(categorical_cols):
    if col in X_train_eng.columns:
        X_train_eng[f"{col}_target_enc"] = oof_enc[:, idx]

X_val_eng, _, _ = engineer_features(
    X_val,
    numeric_cols,
    categorical_cols,
    freq_maps=freq_maps,
    target_enc_maps=target_enc_maps,
    numeric_stats=numeric_stats,
    fit=False,
)
for col in categorical_cols:
    if col in X_val_eng.columns and col in target_enc_maps:
        X_val_eng[f"{col}_target_enc"] = (
            X_val_eng[col].map(target_enc_maps[col]).fillna(target_enc_maps["__global_mean__"])
        )

X_test_eng, _, _ = engineer_features(
    X_test,
    numeric_cols,
    categorical_cols,
    freq_maps=freq_maps,
    target_enc_maps=target_enc_maps,
    numeric_stats=numeric_stats,
    fit=False,
)
for col in categorical_cols:
    if col in X_test_eng.columns and col in target_enc_maps:
        X_test_eng[f"{col}_target_enc"] = (
            X_test_eng[col].map(target_enc_maps[col]).fillna(target_enc_maps["__global_mean__"])
        )

# Finalize
X_train_final = finalize_features(X_train_eng)
X_val_final = finalize_features(X_val_eng)
X_test_final = finalize_features(X_test_eng)

common_cols = X_train_final.columns.tolist()
X_val_final = X_val_final.reindex(columns=common_cols, fill_value=-999.0)
X_test_final = X_test_final.reindex(columns=common_cols, fill_value=-999.0)

print(
    f"Final shapes: Train={X_train_final.shape}, Val={X_val_final.shape}, Test={X_test_final.shape}"
)

# ============================================================
# 4. MODEL TRAINING: MULTI-ALGORITHM ENSEMBLE
# ============================================================
pos_count = int(y_train.sum())
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / max(pos_count, 1)
print(f"Class balance: {pos_count} pos / {neg_count} neg, spw={scale_pos_weight:.3f}")

val_preds_all = []
test_preds_all = []
val_aucs_all = []

# --- LightGBM (3 seeds) ---
print("\n--- Training LightGBM ---")
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "min_split_gain": 0.01,
    "scale_pos_weight": scale_pos_weight,
    "verbose": -1,
}

lgb_seeds = [42, 123, 777]
for seed in lgb_seeds:
    params = lgb_params.copy()
    params["seed"] = seed
    params["n_estimators"] = 3000

    train_data = lgb.Dataset(
        X_train_final.values,
        label=y_train,
        feature_name=common_cols,
        free_raw_data=True,
    )
    val_data = lgb.Dataset(
        X_val_final.values,
        label=y_val,
        feature_name=common_cols,
        reference=train_data,
        free_raw_data=True,
    )

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )

    best_iter = model.best_iteration
    val_pred = model.predict(X_val_final.values, num_iteration=best_iter)
    test_pred = model.predict(X_test_final.values, num_iteration=best_iter)

    val_preds_all.append(val_pred)
    test_preds_all.append(test_pred)
    fold_auc = roc_auc_score(y_val, val_pred)
    val_aucs_all.append(fold_auc)
    print(f"  LGB Seed {seed}: best_iter={best_iter}, val_AUC={fold_auc:.4f}")

# --- XGBoost (2 seeds) ---
print("\n--- Training XGBoost ---")
xgb_seeds = [42, 123]
for seed in xgb_seeds:
    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=2.0,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=seed,
        early_stopping_rounds=150,
        verbosity=0,
    )
    xgb_model.fit(
        X_train_final.values,
        y_train,
        eval_set=[(X_val_final.values, y_val)],
        verbose=False,
    )

    val_pred = xgb_model.predict_proba(X_val_final.values)[:, 1]
    test_pred = xgb_model.predict_proba(X_test_final.values)[:, 1]

    val_preds_all.append(val_pred)
    test_preds_all.append(test_pred)
    fold_auc = roc_auc_score(y_val, val_pred)
    val_aucs_all.append(fold_auc)
    print(f"  XGB Seed {seed}: best_iter={xgb_model.best_iteration}, val_AUC={fold_auc:.4f}")

# --- CatBoost (2 seeds) ---
print("\n--- Training CatBoost ---")
cb_seeds = [42, 123]
for seed in cb_seeds:
    cb_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.7,
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=0,
        early_stopping_rounds=150,
    )
    cb_model.fit(
        X_train_final.values,
        y_train,
        eval_set=(X_val_final.values, y_val),
        verbose=0,
    )

    val_pred = cb_model.predict_proba(X_val_final.values)[:, 1]
    test_pred = cb_model.predict_proba(X_test_final.values)[:, 1]

    val_preds_all.append(val_pred)
    test_preds_all.append(test_pred)
    fold_auc = roc_auc_score(y_val, val_pred)
    val_aucs_all.append(fold_auc)
    print(f"  CB Seed {seed}: best_iter={cb_model.get_best_iteration()}, val_AUC={fold_auc:.4f}")

# ============================================================
# 5. WEIGHTED ENSEMBLE PREDICTIONS & VALIDATION
# ============================================================
# Use inverse validation AUC as weights (higher AUC -> higher weight)
val_aucs_arr = np.array(val_aucs_all)
weights = 1.0 / (1.0 - val_aucs_arr)  # inverse of (1 - AUC) gives more weight to better models
weights = weights / weights.sum()  # normalize to sum to 1

print("\n--- Ensemble Weights ---")
model_names = (
    [f"LGB_s{s}" for s in lgb_seeds]
    + [f"XGB_s{s}" for s in xgb_seeds]
    + [f"CB_s{s}" for s in cb_seeds]
)
for name, auc, w in zip(model_names, val_aucs_arr, weights):
    print(f"  {name}: AUC={auc:.4f}, weight={w:.4f}")

val_preds = np.average(val_preds_all, axis=0, weights=weights)
test_preds = np.average(test_preds_all, axis=0, weights=weights)

val_auc = roc_auc_score(y_val, val_preds)
print(f"\nWeighted Ensemble Val ROC-AUC: {val_auc:.4f}")
print(f"Test preds range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")

# ============================================================
# 6. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

if id_col and id_col in test.columns:
    submission = pd.DataFrame({"id": test[id_col].values, "current_status": test_preds})
else:
    submission = pd.DataFrame(
        {"id": range(len(test_preds)), "current_status": test_preds}
    )

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head().to_string(index=False))

# ============================================================
# FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")