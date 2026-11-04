import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Set random seed for reproducibility
RANDOM_STATE = 42
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ---------------------------------------------------------
print("Loading train and test parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Loaded raw train shape: {train_df.shape}, test shape: {test_df.shape}")

id_col = "id"
target_col = "target"
target = train_df[target_col].values

# Drop synthetic noise ps_calc_* columns
calc_cols = [c for c in train_df.columns if c.startswith("ps_calc_")]
train_df.drop(columns=calc_cols, inplace=True)
test_df.drop(columns=calc_cols, inplace=True)
print(f"Dropped {len(calc_cols)} uninformative ps_calc_* columns.")

feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]
n_train = len(train_df)

combined_df = pd.concat(
    [train_df[feature_cols], test_df[feature_cols]], axis=0
).reset_index(drop=True)

# Missing Value Indicators (-1 marker)
print("Engineering missingness, group aggregations, and interaction features...")
combined_df["missing_count_total"] = (combined_df == -1).sum(axis=1)

ind_cols = [c for c in feature_cols if "ps_ind_" in c]
reg_cols = [c for c in feature_cols if "ps_reg_" in c]
car_cols = [c for c in feature_cols if "ps_car_" in c]
bin_cols = [c for c in feature_cols if c.endswith("_bin")]
cat_cols = [c for c in feature_cols if c.endswith("_cat")]
num_cols = [c for c in feature_cols if not c.endswith("_cat") and not c.endswith("_bin")]

combined_df["missing_count_ind"] = (combined_df[ind_cols] == -1).sum(axis=1)
combined_df["missing_count_reg"] = (combined_df[reg_cols] == -1).sum(axis=1)
combined_df["missing_count_car"] = (combined_df[car_cols] == -1).sum(axis=1)

# Cleanly convert -1 to np.nan for numerical features
for col in num_cols:
    combined_df[col] = combined_df[col].replace(-1, np.nan)

# Group Aggregations of continuous features grouped by ps_car_11_cat
if "ps_car_11_cat" in combined_df.columns:
    for group_target in ["ps_car_13", "ps_reg_03"]:
        if group_target in combined_df.columns:
            grouped = combined_df.groupby("ps_car_11_cat")[group_target]
            combined_df[f"{group_target}_mean_by_car_11"] = grouped.transform("mean").astype(np.float32)
            combined_df[f"{group_target}_std_by_car_11"] = grouped.transform("std").fillna(0).astype(np.float32)
            combined_df[f"{group_target}_min_by_car_11"] = grouped.transform("min").astype(np.float32)
            combined_df[f"{group_target}_max_by_car_11"] = grouped.transform("max").astype(np.float32)
            combined_df[f"{group_target}_diff_mean_by_car_11"] = (
                combined_df[group_target] - combined_df[f"{group_target}_mean_by_car_11"]
            ).astype(np.float32)

# Ensure non-negative indexing for categorical features (removes negative value warnings)
for col in cat_cols:
    min_val = combined_df[col].min()
    if min_val < 0:
        combined_df[col] = (combined_df[col] - min_val).astype(np.int32)
    else:
        combined_df[col] = combined_df[col].astype(np.int32)

# Binary feature aggregations
combined_df["bin_feature_sum"] = combined_df[bin_cols].sum(axis=1)
combined_df["ind_bin_sum"] = combined_df[[c for c in bin_cols if "ps_ind_" in c]].sum(
    axis=1
)

# Regional arithmetic interactions
if "ps_reg_01" in combined_df.columns and "ps_reg_02" in combined_df.columns:
    combined_df["ps_reg_01_plus_02"] = (
        combined_df["ps_reg_01"] + combined_df["ps_reg_02"]
    )
    combined_df["ps_reg_01_mult_02"] = (
        combined_df["ps_reg_01"] * combined_df["ps_reg_02"]
    )

if "ps_reg_01" in combined_df.columns and "ps_reg_03" in combined_df.columns:
    combined_df["ps_reg_01_plus_03"] = (
        combined_df["ps_reg_01"] + combined_df["ps_reg_03"]
    )
    combined_df["ps_reg_01_mult_03"] = (
        combined_df["ps_reg_01"] * combined_df["ps_reg_03"]
    )

if "ps_reg_02" in combined_df.columns and "ps_reg_03" in combined_df.columns:
    combined_df["ps_reg_02_plus_03"] = (
        combined_df["ps_reg_02"] + combined_df["ps_reg_03"]
    )
    combined_df["ps_reg_02_mult_03"] = (
        combined_df["ps_reg_02"] * combined_df["ps_reg_03"]
    )

# Vehicle spec interactions
if "ps_car_13" in combined_df.columns and "ps_reg_03" in combined_df.columns:
    combined_df["ps_car_13_x_ps_reg_03"] = (
        combined_df["ps_car_13"] * combined_df["ps_reg_03"]
    )

if "ps_car_13" in combined_df.columns and "ps_car_15" in combined_df.columns:
    combined_df["ps_car_13_x_ps_car_15"] = (
        combined_df["ps_car_13"] * combined_df["ps_car_15"]
    )

if "ps_car_12" in combined_df.columns:
    combined_df["ps_car_12_sq"] = combined_df["ps_car_12"] ** 2

if "ps_car_15" in combined_df.columns:
    combined_df["ps_car_15_sq"] = combined_df["ps_car_15"] ** 2

# Frequency encoding for categoricals
for col in cat_cols:
    freq_map = combined_df[col].value_counts(normalize=True).to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq_map).astype(np.float32)

# High-order categorical interactions
if "ps_car_01_cat" in combined_df.columns and "ps_car_06_cat" in combined_df.columns:
    cat_inter = (
        combined_df["ps_car_01_cat"].astype(str)
        + "_"
        + combined_df["ps_car_06_cat"].astype(str)
    )
    freq_map_inter = cat_inter.value_counts(normalize=True).to_dict()
    combined_df["ps_car_01_06_inter_freq"] = cat_inter.map(freq_map_inter).astype(
        np.float32
    )

# Split back to train and test
train_fe = combined_df.iloc[:n_train].copy().reset_index(drop=True)
test_fe = combined_df.iloc[n_train:].copy().reset_index(drop=True)

train_fe["id"] = train_df["id"].values
train_fe["target"] = target
test_fe["id"] = test_df["id"].values

# 5-fold Stratified split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_fe["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_fe, train_fe["target"])):
    train_fe.loc[val_idx, "fold"] = fold


# ---------------------------------------------------------
# 2. Metric and Model Architecture
# ---------------------------------------------------------
def compute_normalized_gini(y_true, y_pred):
    """Computes Normalized Gini coefficient (2 * ROC_AUC - 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return 0.0
    return 2.0 * roc_auc_score(y_true[valid_mask], y_pred[valid_mask]) - 1.0


def lgb_normalized_gini(preds, train_data):
    """Custom LightGBM evaluation metric function returning ('gini', score, True)."""
    labels = train_data.get_label()
    score = compute_normalized_gini(labels, preds)
    return "gini", score, True


def get_lgbm_model_params(random_state: int = 42) -> dict:
    """Returns regularized LightGBM parameters for low signal-to-noise risk data."""
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 80,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "max_bin": 255,
        "n_jobs": -1,
        "random_state": random_state,
        "verbose": -1,
    }


def get_xgb_model_params(random_state: int = 42) -> dict:
    """Returns regularized histogram XGBoost parameters for noisy risk data."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 2.0,
        "reg_lambda": 6.0,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }


def calc_smoothed_te(train_series, target_series, test_series, smoothing=10.0):
    """Calculates Bayesian smoothed target encoding."""
    if not isinstance(target_series, pd.Series):
        target_series = pd.Series(target_series, index=train_series.index)
    global_mean = float(target_series.mean())
    grouped = target_series.groupby(train_series)
    counts = grouped.count()
    means = grouped.mean()
    smooth_map = (counts * means + smoothing * global_mean) / (counts + smoothing)
    train_encoded = train_series.map(smooth_map).fillna(global_mean).astype(np.float32)
    test_encoded = test_series.map(smooth_map).fillna(global_mean).astype(np.float32)
    return train_encoded, test_encoded, smooth_map, global_mean


# ---------------------------------------------------------
# 3. Training, Evaluation, and Heterogeneous Ensembling
# ---------------------------------------------------------
ignore_cols = ["id", "target", "fold"]
base_features = [col for col in train_fe.columns if col not in ignore_cols]
cat_features = [
    col for col in base_features if col.endswith("_cat") and not col.endswith("_freq")
]
te_target_cols = [
    col
    for col in ["ps_car_11_cat", "ps_car_01_cat", "ps_car_06_cat", "ps_ind_05_cat"]
    if col in base_features
]

print(
    f"Starting 5-Fold Cross-Validation training on {len(base_features)} base features with OOF Target Encoding..."
)

n_splits = 5
oof_preds_lgb = np.zeros(len(train_fe), dtype=np.float64)
oof_preds_xgb = np.zeros(len(train_fe), dtype=np.float64)
test_preds_lgb = np.zeros(len(test_fe), dtype=np.float64)
test_preds_xgb = np.zeros(len(test_fe), dtype=np.float64)

for fold in range(n_splits):
    train_idx = train_fe["fold"] != fold
    val_idx = train_fe["fold"] == fold

    X_train_fold = train_fe.loc[train_idx, base_features].copy()
    y_train_fold = train_fe.loc[train_idx, "target"].values
    X_val_fold = train_fe.loc[val_idx, base_features].copy()
    y_val_fold = train_fe.loc[val_idx, "target"].values
    X_test_fold = test_fe[base_features].copy()

    # Leak-free Bayesian smoothed target encoding computed strictly on fold train data
    for col in te_target_cols:
        te_train, te_val, smooth_map, global_mean = calc_smoothed_te(
            X_train_fold[col], y_train_fold, X_val_fold[col], smoothing=10.0
        )
        te_test = X_test_fold[col].map(smooth_map).fillna(global_mean).astype(np.float32)

        X_train_fold[f"{col}_te"] = te_train
        X_val_fold[f"{col}_te"] = te_val
        X_test_fold[f"{col}_te"] = te_test

    all_fold_features = X_train_fold.columns.tolist()
    fold_cat_features = [
        col
        for col in all_fold_features
        if col.endswith("_cat") and not col.endswith("_freq") and not col.endswith("_te")
    ]

    # --- 1. LightGBM Model ---
    dtrain_lgb = lgb.Dataset(
        X_train_fold,
        label=y_train_fold,
        categorical_feature=fold_cat_features,
        free_raw_data=False,
    )
    dval_lgb = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        categorical_feature=fold_cat_features,
        reference=dtrain_lgb,
        free_raw_data=False,
    )

    lgb_params = get_lgbm_model_params(random_state=RANDOM_STATE + fold * 10)
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=dtrain_lgb,
        num_boost_round=1500,
        valid_sets=[dval_lgb],
        valid_names=["valid"],
        feval=lgb_normalized_gini,
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds_lgb = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
    oof_preds_lgb[val_idx] = val_preds_lgb
    test_preds_lgb += (
        lgb_model.predict(X_test_fold, num_iteration=lgb_model.best_iteration) / n_splits
    )
    lgb_gini = compute_normalized_gini(y_val_fold, val_preds_lgb)

    # --- 2. XGBoost Model ---
    dtrain_xgb = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval_xgb = xgb.DMatrix(X_val_fold, label=y_val_fold)
    dtest_xgb = xgb.DMatrix(X_test_fold)

    xgb_params = get_xgb_model_params(random_state=RANDOM_STATE + fold * 10)
    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain_xgb,
        num_boost_round=1500,
        evals=[(dval_xgb, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )

    val_preds_xgb = xgb_model.predict(dval_xgb)
    oof_preds_xgb[val_idx] = val_preds_xgb
    test_preds_xgb += xgb_model.predict(dtest_xgb) / n_splits
    xgb_gini = compute_normalized_gini(y_val_fold, val_preds_xgb)

    fold_blend = 0.5 * val_preds_lgb + 0.5 * val_preds_xgb
    fold_blend_gini = compute_normalized_gini(y_val_fold, fold_blend)
    print(
        f"Fold {fold + 1}/{n_splits} | LGB Gini: {lgb_gini:.5f} (iter {lgb_model.best_iteration}) | "
        f"XGB Gini: {xgb_gini:.5f} (iter {xgb_model.best_iteration}) | Blend Gini: {fold_blend_gini:.5f}"
    )

# Optimize ensemble blending weight on full Out-Of-Fold predictions
y_true_all = train_fe["target"].values
lgb_oof_gini = compute_normalized_gini(y_true_all, oof_preds_lgb)
xgb_oof_gini = compute_normalized_gini(y_true_all, oof_preds_xgb)
print(f"Overall LGBM OOF Gini: {lgb_oof_gini:.5f}")
print(f"Overall XGBoost OOF Gini: {xgb_oof_gini:.5f}")

best_w = 0.5
best_ensemble_gini = -1.0
for w in np.linspace(0.0, 1.0, 101):
    blended_oof = w * oof_preds_lgb + (1.0 - w) * oof_preds_xgb
    score = compute_normalized_gini(y_true_all, blended_oof)
    if score > best_ensemble_gini:
        best_ensemble_gini = score
        best_w = w

print(
    f"Optimal Blending Weights -> LightGBM: {best_w:.2f}, XGBoost: {1.0 - best_w:.2f}"
)
print(f"Overall Heterogeneous Ensemble OOF Normalized Gini: {best_ensemble_gini:.5f}")

# Final blended test predictions
final_test_preds = best_w * test_preds_lgb + (1.0 - best_w) * test_preds_xgb

# Save submission
submission_df = pd.DataFrame({"id": test_fe["id"].astype(int), "target": final_test_preds})
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert len(submission_df) == len(
    test_fe
), f"Expected {len(test_fe)} rows, got {len(submission_df)}"
assert (
    not submission_df["target"].isnull().any()
), "Submission contains null target values"
assert submission_df.columns.tolist() == ["id", "target"], "Incorrect column format"

print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}"
)
print(f"Final Validation Score: {best_ensemble_gini}")
