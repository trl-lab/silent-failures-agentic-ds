import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "APR"
id_col = "id" if "id" in train.columns else None

if id_col:
    train_ids = train[id_col].values
    test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))
else:
    train_ids = np.arange(len(train))
    test_ids = np.arange(len(test))

feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")

# ============================================================
# 3. SPLIT TRAIN INTO TRAIN/VAL FIRST (before any transformation)
# ============================================================
val_size = 0.15
train_df, val_df = train_test_split(train, test_size=val_size, random_state=42)
print(f"Train split: {train_df.shape}, Val split: {val_df.shape}")

# ============================================================
# 4. FEATURE ENGINEERING (fit on train, transform on val/test)
# ============================================================

# --- 4a: Handle missing values ---
numeric_medians = {}
for col in numeric_cols:
    if col in train_df.columns:
        numeric_medians[col] = train_df[col].median()

cat_fill = {}
for col in categorical_cols:
    if col in train_df.columns:
        cat_fill[col] = (
            train_df[col].mode()[0] if not train_df[col].mode().empty else "MISSING"
        )


def impute_numeric(df, medians):
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(medians.get(col, df[col].median()))
    return df


def impute_categorical(df, fills):
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(fills.get(col, "MISSING"))
    return df


# --- 4b: Log transform for skewed numeric features ---
skewed_cols = []
for col in numeric_cols:
    if col in train_df.columns:
        positive = train_df[col].dropna()
        if len(positive) > 0 and (positive > 0).sum() / len(positive) > 0.9:
            skew = positive.skew()
            if abs(skew) > 1.0 and positive.min() > 0:
                skewed_cols.append(col)


def apply_log_transform(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col].clip(lower=0))
    return df


# --- 4c: Ratio and interaction features ---
def create_interactions(df):
    df = df.copy()
    col_set = set(df.columns)

    income_cols = [c for c in df.columns if "income" in c.lower() or "INC" in c.upper()]
    debt_cols = [c for c in df.columns if "debt" in c.lower() or "DEBT" in c.upper()]
    score_cols = [
        c
        for c in df.columns
        if "score" in c.lower() or "SCORE" in c.upper() or "FICO" in c.upper()
    ]
    term_cols = [
        c
        for c in df.columns
        if "term" in c.lower() or "TERM" in c.upper() or "month" in c.lower()
    ]
    loan_cols = [
        c
        for c in df.columns
        if "loan" in c.lower()
        or "amt" in c.lower()
        or "AMT" in c.upper()
        or "amount" in c.lower()
    ]

    if income_cols and debt_cols:
        for ic in income_cols[:2]:
            for dc in debt_cols[:2]:
                if ic in df.columns and dc in df.columns:
                    df[f"{dc}_to_{ic}"] = df[dc] / (df[ic] + 1e-8)

    if score_cols and term_cols:
        for sc in score_cols[:2]:
            for tc in term_cols[:2]:
                if sc in df.columns and tc in df.columns:
                    df[f"{sc}_x_{tc}"] = df[sc] * df[tc]

    if loan_cols and term_cols:
        for lc in loan_cols[:2]:
            for tc in term_cols[:2]:
                if lc in df.columns and tc in df.columns:
                    df[f"{lc}_per_{tc}"] = df[lc] / (df[tc] + 1e-8)

    if income_cols and loan_cols:
        for ic in income_cols[:2]:
            for lc in loan_cols[:2]:
                if ic in df.columns and lc in df.columns:
                    df[f"{lc}_to_{ic}"] = df[lc] / (df[ic] + 1e-8)

    return df


# --- 4d: Target encoding for high-cardinality categoricals (K-fold) ---
high_card_threshold = 10
target_enc_cols = []
target_enc_mappings = {}


def compute_target_encoding_from_split(train_split, col, target_col, smoothing=10):
    """Compute target encoding mapping from a training split."""
    global_mean = train_split[target_col].mean()
    agg = pd.DataFrame(
        {"col": train_split[col].values, "target": train_split[target_col].values}
    )
    stats = agg.groupby("col")["target"].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    mapping = smoothed.to_dict()
    return mapping, global_mean


def compute_kfold_target_encoding(train_df, col, target_col, n_folds=5, smoothing=10):
    """K-fold target encoding to prevent self-leakage."""
    global_mean = train_df[target_col].mean()
    n = len(train_df)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    encoded = np.full(n, global_mean, dtype=np.float64)

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

        train_split = train_df.iloc[train_idx]
        mapping, _ = compute_target_encoding_from_split(train_split, col, target_col, smoothing)

        val_values = train_df.iloc[val_idx][col].values
        encoded[val_idx] = np.array([mapping.get(v, global_mean) for v in val_values])

    return encoded


for col in categorical_cols:
    if col in train_df.columns:
        nunique = train_df[col].nunique()
        if nunique > high_card_threshold:
            target_enc_cols.append(col)
            # Compute full mapping from all training data (for test/val inference)
            target_enc_mappings[col], _ = compute_target_encoding_from_split(
                train_df, col, target_col
            )


# --- 4d2: Hierarchical multi-level target encoding ---
# Institution + Plan interaction target encoding
hierarchical_enc_cols = []
hierarchical_enc_mappings = {}

# Identify the institution and plan columns
institution_col = "Institution Name"
plan_col = "Plan Name"

if institution_col in train_df.columns and plan_col in train_df.columns:
    # Create interaction column
    train_df["_inst_plan"] = train_df[institution_col].astype(str) + "_" + train_df[plan_col].astype(str)
    val_df["_inst_plan"] = val_df[institution_col].astype(str) + "_" + val_df[plan_col].astype(str)
    test["_inst_plan"] = test[institution_col].astype(str) + "_" + test[plan_col].astype(str)

    # Compute full mapping for test/val inference
    global_mean_hp = train_df[target_col].mean()
    agg_hp = pd.DataFrame({
        "col": train_df["_inst_plan"].values,
        "target": train_df[target_col].values
    })
    stats_hp = agg_hp.groupby("col")["target"].agg(["mean", "count"])
    smoothed_hp = (stats_hp["count"] * stats_hp["mean"] + 10 * global_mean_hp) / (
        stats_hp["count"] + 10
    )
    hierarchical_enc_mappings["_inst_plan"] = smoothed_hp.to_dict()
    hierarchical_enc_mappings["_inst_plan_global_mean"] = global_mean_hp
    hierarchical_enc_cols.append("_inst_plan")

    # Institution-level APR variance/std as features
    inst_stats = train_df.groupby(institution_col)[target_col].agg(["std", "mean"])
    inst_stats.columns = ["inst_apr_std", "inst_apr_mean"]
    inst_stats["inst_apr_std"] = inst_stats["inst_apr_std"].fillna(0)

    # Store for later use
    inst_apr_std_map = inst_stats["inst_apr_std"].to_dict()
    inst_apr_mean_map = inst_stats["inst_apr_mean"].to_dict()

    # Apply to train (K-fold style to avoid leakage)
    n_hp = len(train_df)
    np.random.seed(42)
    indices_hp = np.arange(n_hp)
    np.random.shuffle(indices_hp)
    folds_hp = np.array_split(indices_hp, 5)

    train_df["_inst_plan_te"] = np.full(n_hp, global_mean_hp, dtype=np.float64)
    for fold_idx in range(5):
        val_idx = folds_hp[fold_idx]
        train_idx = np.concatenate([folds_hp[i] for i in range(5) if i != fold_idx])
        train_split = train_df.iloc[train_idx]
        agg_f = pd.DataFrame({
            "col": train_split["_inst_plan"].values,
            "target": train_split[target_col].values
        })
        stats_f = agg_f.groupby("col")["target"].agg(["mean", "count"])
        smoothed_f = (stats_f["count"] * stats_f["mean"] + 10 * global_mean_hp) / (
            stats_f["count"] + 10
        )
        mapping_f = smoothed_f.to_dict()
        val_values = train_df.iloc[val_idx]["_inst_plan"].values
        train_df.iloc[val_idx, train_df.columns.get_loc("_inst_plan_te")] = np.array(
            [mapping_f.get(v, global_mean_hp) for v in val_values]
        )

    # Apply to val and test using full mapping
    val_df["_inst_plan_te"] = val_df["_inst_plan"].map(hierarchical_enc_mappings["_inst_plan"]).fillna(global_mean_hp)
    test["_inst_plan_te"] = test["_inst_plan"].map(hierarchical_enc_mappings["_inst_plan"]).fillna(global_mean_hp)

    # Institution APR std feature - K-fold on train to prevent leakage
    n_inst = len(train_df)
    np.random.seed(42)
    indices_inst = np.arange(n_inst)
    np.random.shuffle(indices_inst)
    folds_inst = np.array_split(indices_inst, 5)

    train_df["inst_apr_std"] = np.full(n_inst, 0.0, dtype=np.float64)
    train_df["inst_apr_mean"] = np.full(n_inst, global_mean_hp, dtype=np.float64)

    for fold_idx in range(5):
        val_idx = folds_inst[fold_idx]
        train_idx = np.concatenate([folds_inst[i] for i in range(5) if i != fold_idx])
        train_split = train_df.iloc[train_idx]
        inst_stats_f = train_split.groupby(institution_col)[target_col].agg(["std", "mean"])
        inst_stats_f.columns = ["inst_apr_std_f", "inst_apr_mean_f"]
        inst_stats_f["inst_apr_std_f"] = inst_stats_f["inst_apr_std_f"].fillna(0)
        std_map_f = inst_stats_f["inst_apr_std_f"].to_dict()
        mean_map_f = inst_stats_f["inst_apr_mean_f"].to_dict()
        val_values = train_df.iloc[val_idx][institution_col].values
        train_df.iloc[val_idx, train_df.columns.get_loc("inst_apr_std")] = np.array(
            [std_map_f.get(v, 0.0) for v in val_values]
        )
        train_df.iloc[val_idx, train_df.columns.get_loc("inst_apr_mean")] = np.array(
            [mean_map_f.get(v, global_mean_hp) for v in val_values]
        )

    # Apply full mapping to val and test
    val_df["inst_apr_std"] = val_df[institution_col].map(inst_apr_std_map).fillna(0)
    test["inst_apr_std"] = test[institution_col].map(inst_apr_std_map).fillna(0)
    val_df["inst_apr_mean"] = val_df[institution_col].map(inst_apr_mean_map).fillna(global_mean_hp)
    test["inst_apr_mean"] = test[institution_col].map(inst_apr_mean_map).fillna(global_mean_hp)

    # Add rate_type binary feature: 1 if variable, 0 if fixed
    rate_col = "Availability of Credit Card Plan"
    train_df["rate_type"] = train_df[rate_col].str.lower().str.contains("variable", na=False).astype(int)
    val_df["rate_type"] = val_df[rate_col].str.lower().str.contains("variable", na=False).astype(int)
    test["rate_type"] = test[rate_col].str.lower().str.contains("variable", na=False).astype(int)

    # K-fold target encoding for institution+plan+rate_type triple
    train_df["_inst_plan_rate"] = (
        train_df[institution_col].astype(str)
        + "_"
        + train_df[plan_col].astype(str)
        + "_"
        + train_df["rate_type"].astype(str)
    )
    val_df["_inst_plan_rate"] = (
        val_df[institution_col].astype(str)
        + "_"
        + val_df[plan_col].astype(str)
        + "_"
        + val_df["rate_type"].astype(str)
    )
    test["_inst_plan_rate"] = (
        test[institution_col].astype(str)
        + "_"
        + test[plan_col].astype(str)
        + "_"
        + test["rate_type"].astype(str)
    )

    # Full mapping for val/test
    global_mean_ipr = train_df[target_col].mean()
    agg_ipr = pd.DataFrame({
        "col": train_df["_inst_plan_rate"].values,
        "target": train_df[target_col].values
    })
    stats_ipr = agg_ipr.groupby("col")["target"].agg(["mean", "count"])
    smoothed_ipr = (stats_ipr["count"] * stats_ipr["mean"] + 10 * global_mean_ipr) / (
        stats_ipr["count"] + 10
    )
    ipr_mapping_full = smoothed_ipr.to_dict()

    # K-fold on train
    train_df["_inst_plan_rate_te"] = np.full(n_inst, global_mean_ipr, dtype=np.float64)
    for fold_idx in range(5):
        val_idx = folds_inst[fold_idx]
        train_idx = np.concatenate([folds_inst[i] for i in range(5) if i != fold_idx])
        train_split = train_df.iloc[train_idx]
        agg_f = pd.DataFrame({
            "col": train_split["_inst_plan_rate"].values,
            "target": train_split[target_col].values
        })
        stats_f = agg_f.groupby("col")["target"].agg(["mean", "count"])
        smoothed_f = (stats_f["count"] * stats_f["mean"] + 10 * global_mean_ipr) / (
            stats_f["count"] + 10
        )
        mapping_f = smoothed_f.to_dict()
        val_values = train_df.iloc[val_idx]["_inst_plan_rate"].values
        train_df.iloc[val_idx, train_df.columns.get_loc("_inst_plan_rate_te")] = np.array(
            [mapping_f.get(v, global_mean_ipr) for v in val_values]
        )

    # Apply full mapping to val and test
    val_df["_inst_plan_rate_te"] = val_df["_inst_plan_rate"].map(ipr_mapping_full).fillna(global_mean_ipr)
    test["_inst_plan_rate_te"] = test["_inst_plan_rate"].map(ipr_mapping_full).fillna(global_mean_ipr)


# --- 4e-placeholder (moved above pipeline) ---
def create_credit_card_plan_indicator(df):
    """Create binary indicator: 1 if variable rate, 0 if fixed rate."""
    df = df.copy()
    if "Availability of Credit Card Plan" in df.columns:
        df["Credit Card Plan Variable"] = (
            df["Availability of Credit Card Plan"].str.lower().str.contains("variable", na=False)
            .astype(int)
        )
    return df


# --- 4f: Polynomial features for key numeric pairs ---
def create_poly_features(df, numeric_cols, max_pairs=5, top_features=None):
    df = df.copy()
    num_features = [c for c in numeric_cols if c in df.columns]

    if len(num_features) >= 2:
        if top_features is None:
            variances = [(c, df[c].var()) for c in num_features if df[c].var() > 0]
            variances.sort(key=lambda x: x[1], reverse=True)
            top_features = [c for c, v in variances[:6]]

        for i in range(min(len(top_features), max_pairs)):
            for j in range(i + 1, min(len(top_features), max_pairs)):
                c1, c2 = top_features[i], top_features[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    return df


# ============================================================
# 5. APPLY ALL TRANSFORMS (fit on train, apply to all)
# ============================================================


def full_pipeline_transform(df, is_train=False, fold_indices=None):
    """Apply all feature engineering transforms.

    When is_train=True and fold_indices is provided, uses K-fold target encoding
    to prevent self-leakage. When is_train=False, uses pre-computed mapping.
    """
    df = df.copy()

    df = impute_numeric(df, numeric_medians)
    df = impute_categorical(df, cat_fill)
    df = apply_log_transform(df, skewed_cols)
    df = create_interactions(df)
    df = create_credit_card_plan_indicator(df)

    # Target encoding
    for col in target_enc_cols:
        if col in df.columns:
            if is_train and fold_indices is not None:
                # K-fold target encoding to prevent self-leakage
                encoded = compute_kfold_target_encoding(df, col, target_col, n_folds=5, smoothing=10)
                df[col + "_te"] = encoded
            elif is_train:
                # Fallback: single-pass (should not be used in final pipeline)
                mapping, global_mean = compute_target_encoding_from_split(df, col, target_col)
                df[col + "_te"] = df[col].map(mapping).fillna(global_mean)
            else:
                mapping = target_enc_mappings.get(col, {})
                global_mean = train_df[target_col].mean()
                df[col + "_te"] = df[col].map(mapping).fillna(global_mean)

    df = create_poly_features(df, numeric_cols, top_features=top_features_for_poly)
    return df


# Pre-compute top variance features from train only (to ensure consistency across splits)
num_features_all = [c for c in numeric_cols if c in train_df.columns]
variances_train = [(c, train_df[c].var()) for c in num_features_all if train_df[c].var() > 0]
variances_train.sort(key=lambda x: x[1], reverse=True)
top_features_for_poly = [c for c, v in variances_train[:6]]

# Apply to train (fit) with K-fold target encoding to prevent self-leakage
n_train = len(train_df)
np.random.seed(42)
indices = np.arange(n_train)
np.random.shuffle(indices)
folds = np.array_split(indices, 5)

train_processed = train_df.copy()
train_processed = impute_numeric(train_processed, numeric_medians)
train_processed = impute_categorical(train_processed, cat_fill)
train_processed = apply_log_transform(train_processed, skewed_cols)
train_processed = create_interactions(train_processed)
train_processed = create_credit_card_plan_indicator(train_processed)

for col in target_enc_cols:
    if col in train_processed.columns:
        encoded = compute_kfold_target_encoding(train_processed, col, target_col, n_folds=5, smoothing=10)
        train_processed[col + "_te"] = encoded

train_processed = create_poly_features(train_processed, numeric_cols)

# Apply to val and test (transform only, using pre-computed mapping)
val_processed = full_pipeline_transform(val_df, is_train=False)
test_processed = full_pipeline_transform(test, is_train=False)

# Add hierarchical features that were computed directly on train_df/val_df/test
# (they were added before the pipeline, so they persist in the copies)
if "_inst_plan_te" in train_processed.columns:
    pass  # Already in train_processed via train_df.copy()
elif "_inst_plan_te" in train_df.columns:
    train_processed["_inst_plan_te"] = train_df["_inst_plan_te"].values

if "_inst_plan_te" in val_processed.columns:
    pass
elif "_inst_plan_te" in val_df.columns:
    val_processed["_inst_plan_te"] = val_df["_inst_plan_te"].values

if "_inst_plan_te" in test_processed.columns:
    pass
elif "_inst_plan_te" in test.columns:
    test_processed["_inst_plan_te"] = test["_inst_plan_te"].values

# Add inst_apr_std, inst_apr_mean, rate_type, _inst_plan_rate_te if not already present
for extra_col in ["inst_apr_std", "inst_apr_mean", "rate_type", "_inst_plan_rate_te"]:
    if extra_col in train_df.columns and extra_col not in train_processed.columns:
        train_processed[extra_col] = train_df[extra_col].values
    if extra_col in val_df.columns and extra_col not in val_processed.columns:
        val_processed[extra_col] = val_df[extra_col].values
    if extra_col in test.columns and extra_col not in test_processed.columns:
        test_processed[extra_col] = test[extra_col].values

# Ensure val/test have same columns as train (in case of missing interaction features)
for col in train_processed.columns:
    if col not in val_processed.columns:
        val_processed[col] = 0.0
    if col not in test_processed.columns:
        test_processed[col] = 0.0

# ============================================================
# 6. FINALIZE FEATURE SET
# ============================================================
exclude_cols = [target_col] + ([id_col] if id_col else []) + target_enc_cols + ["_inst_plan", "_inst_plan_rate"]
final_feature_cols = [c for c in train_processed.columns if c not in exclude_cols]

# Reorder val/test to match train column order
val_processed = val_processed[final_feature_cols + [target_col]]
test_processed = test_processed[final_feature_cols]

# Ensure all features are numeric
for col in final_feature_cols:
    if train_processed[col].dtype == object:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_processed[col] = oe.fit_transform(train_processed[[col]])
        val_processed[col] = oe.transform(val_processed[[col]])
        test_processed[col] = oe.transform(test_processed[[col]])
    elif train_processed[col].dtype == "category":
        train_processed[col] = train_processed[col].cat.codes
        val_processed[col] = val_processed[col].cat.codes
        test_processed[col] = test_processed[col].cat.codes

# Replace any remaining NaN/inf
for df_ref in [train_processed, val_processed, test_processed]:
    df_ref[final_feature_cols] = df_ref[final_feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    df_ref[final_feature_cols] = df_ref[final_feature_cols].fillna(0)

print(f"Final feature count: {len(final_feature_cols)}")

# ============================================================
# 7. EXTRACT NUMERIC ARRAYS
# ============================================================
X_train = train_processed[final_feature_cols].values.astype(np.float32)
y_train = train_processed[target_col].values.astype(np.float32)
X_val = val_processed[final_feature_cols].values.astype(np.float32)
y_val = val_processed[target_col].values.astype(np.float32)
X_test = test_processed[final_feature_cols].values.astype(np.float32)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"Target stats - Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"Target stats - Val:   mean={y_val.mean():.4f}, std={y_val.std():.4f}")

# ============================================================
# 8. DETERMINE APR BOUNDS FOR CLIPPING
# ============================================================
apr_lower = max(0, y_train.min() - 0.5)
apr_upper = min(40, y_train.max() + 0.5)
print(f"APR clipping bounds: [{apr_lower:.2f}, {apr_upper:.2f}]")

# ============================================================
# 9. MULTI-SEED LIGHTGBM + CATBOOST ENSEMBLE
# ============================================================
base_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_samples": 10,
    "min_child_weight": 3.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "min_split_gain": 0.05,
    "verbose": -1,
}

seeds = [42, 123, 456, 789, 1024]
num_leaves_list = [23, 31, 47, 31, 47]
n_lgb_models = len(seeds) + 2  # 5 standard + 2 huber
n_cb_models = len(seeds)
n_xgb_models = len(seeds)
n_ridge_models = 2
n_models = n_lgb_models + n_cb_models + n_xgb_models + n_ridge_models

train_ds = lgb.Dataset(X_train, label=y_train, feature_name=list(final_feature_cols))
val_ds = lgb.Dataset(
    X_val, label=y_val, feature_name=list(final_feature_cols), reference=train_ds
)

models = []
val_preds = np.zeros((n_models, len(X_val)))
test_preds = np.zeros((n_models, len(X_test)))
val_rmses = np.zeros(n_models)

# --- Train LightGBM models ---
print(f"\nTraining {n_lgb_models} LightGBM models (5 standard + 2 huber)...")

# Standard LGB models with varying num_leaves
for i, seed in enumerate(seeds):
    params = base_params.copy()
    params["seed"] = seed
    params["num_leaves"] = num_leaves_list[i]

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=3000,
        valid_sets=[val_ds],
        callbacks=callbacks,
    )

    models.append(model)

    vp = model.predict(X_val, num_iteration=model.best_iteration)
    tp = model.predict(X_test, num_iteration=model.best_iteration)

    val_preds[i] = vp
    test_preds[i] = tp
    val_rmses[i] = np.sqrt(mean_squared_error(y_val, vp))

    print(
        f"  [LGB] Seed {seed}, leaves={num_leaves_list[i]}: best_iter={model.best_iteration}, val_rmse={val_rmses[i]:.4f}"
    )

# Huber loss LGB models
for i, seed in enumerate([42, 789]):
    params = base_params.copy()
    params["seed"] = seed
    params["objective"] = "huber"
    params["alpha"] = 0.9  # huber delta parameter
    params["num_leaves"] = 31

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=3000,
        valid_sets=[val_ds],
        callbacks=callbacks,
    )

    models.append(model)

    vp = model.predict(X_val, num_iteration=model.best_iteration)
    tp = model.predict(X_test, num_iteration=model.best_iteration)

    idx = len(seeds) + i
    val_preds[idx] = vp
    test_preds[idx] = tp
    val_rmses[idx] = np.sqrt(mean_squared_error(y_val, vp))

    print(
        f"  [LGB-Huber] Seed {seed}: best_iter={model.best_iteration}, val_rmse={val_rmses[idx]:.4f}"
    )

# --- Train CatBoost models ---
print(f"\nTraining {n_cb_models} CatBoost models with different seeds...")

for i, seed in enumerate(seeds):
    cb_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=0,
    )

    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )

    models.append(cb_model)

    vp = cb_model.predict(X_val)
    tp = cb_model.predict(X_test)

    idx = n_lgb_models + i
    val_preds[idx] = vp
    test_preds[idx] = tp
    val_rmses[idx] = np.sqrt(mean_squared_error(y_val, vp))

    print(
        f"  [CB]  Seed {seed}: best_iter={cb_model.get_best_iteration()}, val_rmse={val_rmses[idx]:.4f}"
    )

# --- Train XGBoost models ---
print(f"\nTraining {n_xgb_models} XGBoost models with different seeds...")

for i, seed in enumerate(seeds):
    xgb_model = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        n_estimators=3000,
        random_state=seed,
        verbosity=0,
        tree_method="hist",
        early_stopping_rounds=50,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    models.append(xgb_model)

    best_iter = xgb_model.best_iteration
    vp = xgb_model.predict(X_val, iteration_range=(0, best_iter + 1))
    tp = xgb_model.predict(X_test, iteration_range=(0, best_iter + 1))

    idx = n_lgb_models + n_cb_models + i
    val_preds[idx] = vp
    test_preds[idx] = tp
    val_rmses[idx] = np.sqrt(mean_squared_error(y_val, vp))

    print(
        f"  [XGB] Seed {seed}: best_iter={best_iter}, val_rmse={val_rmses[idx]:.4f}"
    )

# --- Train Ridge models ---
print(f"\nTraining {n_ridge_models} Ridge models...")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

for i, alpha in enumerate([1.0, 10.0]):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)

    models.append(ridge_model)

    vp = ridge_model.predict(X_val_scaled)
    tp = ridge_model.predict(X_test_scaled)

    idx = n_lgb_models + n_cb_models + n_xgb_models + i
    val_preds[idx] = vp
    test_preds[idx] = tp
    val_rmses[idx] = np.sqrt(mean_squared_error(y_val, vp))

    print(
        f"  [Ridge] alpha={alpha}: val_rmse={val_rmses[idx]:.4f}"
    )

# ============================================================
# 10. INVERSE-RMSE WEIGHTED ENSEMBLE (LGB + CB + XGB)
# ============================================================
weights = 1.0 / val_rmses
weights = weights / weights.sum()

print(f"\nEnsemble weights ({n_models} models): {np.round(weights, 4)}")

val_pred_ensemble = np.average(val_preds, axis=0, weights=weights)
test_pred_ensemble = np.average(test_preds, axis=0, weights=weights)

# Apply clipping (same for val and test)
val_pred_final = np.clip(val_pred_ensemble, apr_lower, apr_upper)
test_pred_final = np.clip(test_pred_ensemble, apr_lower, apr_upper)

# ============================================================
# 11. EVALUATE WITH OFFICIAL METRIC (RMSE)
# ============================================================
val_rmse_ensemble = np.sqrt(mean_squared_error(y_val, val_pred_final))
val_rmse_unclipped = np.sqrt(mean_squared_error(y_val, val_pred_ensemble))
baseline_rmse = np.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_train.mean())))

print(f"\n--- Results ---")
print(f"Baseline RMSE (mean):      {baseline_rmse:.4f}")
print(f"Ensemble RMSE (unclipped): {val_rmse_unclipped:.4f}")
print(f"Ensemble RMSE (clipped):   {val_rmse_ensemble:.4f}")
print(
    f"Improvement over baseline: {(1 - val_rmse_ensemble / baseline_rmse) * 100:.2f}%"
)

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================
importance = models[0].feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=final_feature_cols).sort_values(ascending=False)
print(f"\nTop 10 features by gain:")
for i, (f, v) in enumerate(feat_imp.head(10).items()):
    print(f"  {i+1}. {f}: {v:.2f}")

# ============================================================
# 13. SAVE SUBMISSION
# ============================================================
test_ids_arr = (
    np.array(test_ids, dtype=int)
    if not isinstance(test_ids, np.ndarray)
    else test_ids.astype(int)
)

submission = pd.DataFrame({"id": test_ids_arr, "APR": test_pred_final})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")
print(submission.head())

# ============================================================
# 14. FINAL SCORE
# ============================================================
score = val_rmse_ensemble
print(f"Final Validation Score: {score}")