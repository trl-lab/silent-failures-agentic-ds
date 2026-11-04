import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "antibody_truth"

# Separate features and target
y_raw = train_df[target_col]
if y_raw.dtype == "object" or str(y_raw.dtype) == "category" or y_raw.dtype == "bool":
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
else:
    y = y_raw.astype(int).values

train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()

# Drop ID columns if present
id_cols = [
    c for c in train_features.columns if c.lower() in ["id", "sample_id", "index"]
]
train_features = train_features.drop(columns=id_cols)
test_features = test_features.drop(columns=id_cols)

# Combine for uniform feature engineering
df_all = pd.concat([train_features, test_features], axis=0, ignore_index=True)
n_train = len(train_features)

# Add missing count feature
df_all["missing_count"] = df_all.isna().sum(axis=1)

# Identify column types
cat_cols = []
num_cols = []

for c in df_all.columns:
    if c == "missing_count":
        continue
    if (
        df_all[c].dtype == "object"
        or str(df_all[c].dtype) == "category"
        or df_all[c].dtype == "bool"
    ):
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Process categorical missing values
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("Missing")

# Frequency encoding for categoricals
for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True)
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Group-level numerical aggregations across categoricals
for cat in cat_cols[:5]:  # Select key categoricals with moderate cardinality
    if df_all[cat].nunique() > 1 and df_all[cat].nunique() < 500:
        for num in num_cols[:5]:  # Select key numerical features
            grp_mean = df_all.groupby(cat)[num].transform("mean")
            df_all[f"{num}_mean_by_{cat}"] = grp_mean
            df_all[f"{num}_diff_mean_by_{cat}"] = df_all[num] - grp_mean

# Pairwise categorical interactions
if len(cat_cols) >= 2:
    for i in range(min(3, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            col_i, col_j = cat_cols[i], cat_cols[j]
            comb_name = f"{col_i}_X_{col_j}"
            df_all[comb_name] = (
                df_all[col_i].astype(str) + "_" + df_all[col_j].astype(str)
            )
            cat_cols.append(comb_name)

# Ensure categoricals are strictly string type for CatBoost
for col in cat_cols:
    df_all[col] = df_all[col].astype(str)

X_train_full = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_full = df_all.iloc[n_train:].copy().reset_index(drop=True)

# 5-Fold Stratified Cross Validation with Multi-Seed Ensembling & Rank Averaging
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 123, 456, 789, 2024]

oof_preds = np.zeros(len(X_train_full))
test_preds = np.zeros(len(X_test_full))

m_smooth = 10  # Target encoding smoothing parameter

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y)):
    X_tr = X_train_full.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train_full.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test_full.copy()

    # Out-of-fold target encoding for categoricals
    global_mean = y_tr.mean()
    for col in cat_cols[:10]:
        stats = (
            pd.DataFrame({"cat": X_tr[col], "target": y_tr})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
            stats["count"] + m_smooth
        )

        te_map = smooth.to_dict()
        X_tr[f"{col}_te"] = X_tr[col].map(te_map).fillna(global_mean)
        X_va[f"{col}_te"] = X_va[col].map(te_map).fillna(global_mean)
        X_te[f"{col}_te"] = X_te[col].map(te_map).fillna(global_mean)

    cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
    val_pool = Pool(X_va, y_va, cat_features=cat_indices)
    test_pool = Pool(X_te, cat_features=cat_indices)

    fold_val_ranks = np.zeros(len(val_idx))
    fold_test_ranks = np.zeros(len(X_test_full))

    for seed in SEEDS:
        model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=3.0,
            eval_metric="AUC",
            random_seed=seed + fold * 100,
            early_stopping_rounds=100,
            verbose=False,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_prob = model.predict_proba(val_pool)[:, 1]
        test_prob = model.predict_proba(test_pool)[:, 1]

        # Convert to percentile ranks for optimal AUC rank-based ensembling
        fold_val_ranks += rankdata(val_prob) / (len(val_prob) * len(SEEDS))
        fold_test_ranks += rankdata(test_prob) / (len(test_prob) * len(SEEDS))

    oof_preds[val_idx] = fold_val_ranks
    test_preds += fold_test_ranks / skf.n_splits

# Compute Overall OOF Validation ROC AUC
overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC AUC Score: {overall_auc:.5f}")

# Format submission with test predictions converted to percentile ranks
final_test_ranks = rankdata(test_preds) / len(test_preds)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_ranks)), "antibody_truth": final_test_ranks}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
