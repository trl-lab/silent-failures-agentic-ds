import os
from itertools import combinations
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# 1. Load Data
print("Loading raw parquet data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_ids = train_df["id"].values
test_ids = test_df["id"].values
y_train = train_df["target"].values.astype(np.float32)

train_df = train_df.drop(columns=["id", "target"])
test_df = test_df.drop(columns=["id"])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Binary Feature Processing
bin_3_map = {"T": 1, "F": 0}
bin_4_map = {"Y": 1, "N": 0}

combined_df["bin_3_num"] = combined_df["bin_3"].map(bin_3_map).astype(np.float32)
combined_df["bin_4_num"] = combined_df["bin_4"].map(bin_4_map).astype(np.float32)

# 3. Ordinal Feature Mapping
ord_1_map = {"Novice": 1, "Contributor": 2, "Expert": 3, "Master": 4, "Grandmaster": 5}
ord_2_map = {
    "Freezing": 1,
    "Cold": 2,
    "Warm": 3,
    "Hot": 4,
    "Boiling Hot": 5,
    "Lava Hot": 6,
}

combined_df["ord_1_num"] = combined_df["ord_1"].map(ord_1_map).astype(np.float32)
combined_df["ord_2_num"] = combined_df["ord_2"].map(ord_2_map).astype(np.float32)

ord_3_sorted = sorted(combined_df["ord_3"].unique())
ord_3_map = {val: i + 1 for i, val in enumerate(ord_3_sorted)}
combined_df["ord_3_num"] = combined_df["ord_3"].map(ord_3_map).astype(np.float32)

ord_4_sorted = sorted(combined_df["ord_4"].unique())
ord_4_map = {val: i + 1 for i, val in enumerate(ord_4_sorted)}
combined_df["ord_4_num"] = combined_df["ord_4"].map(ord_4_map).astype(np.float32)

ord_5_sorted = sorted(combined_df["ord_5"].unique())
ord_5_map = {val: i + 1 for i, val in enumerate(ord_5_sorted)}
combined_df["ord_5_num"] = combined_df["ord_5"].map(ord_5_map).astype(np.float32)
combined_df["ord_5_char1"] = combined_df["ord_5"].str[0]
combined_df["ord_5_char2"] = combined_df["ord_5"].str[1]

# 4. Cyclical Time Features
combined_df["day_sin"] = np.sin(2 * np.pi * combined_df["day"] / 7.0).astype(np.float32)
combined_df["day_cos"] = np.cos(2 * np.pi * combined_df["day"] / 7.0).astype(np.float32)
combined_df["month_sin"] = np.sin(2 * np.pi * combined_df["month"] / 12.0).astype(
    np.float32
)
combined_df["month_cos"] = np.cos(2 * np.pi * combined_df["month"] / 12.0).astype(
    np.float32
)

# 5. Domain & Pairwise Interaction Features
low_med_cols = [
    "bin_0", "bin_1", "bin_2", "bin_3", "bin_4",
    "nom_0", "nom_1", "nom_2", "nom_3", "nom_4",
    "ord_0", "ord_1", "ord_2", "ord_3", "ord_4",
    "day", "month"
]

interaction_cols = []
for c1, c2 in combinations(low_med_cols, 2):
    col_name = f"{c1}_{c2}"
    combined_df[col_name] = combined_df[c1].astype(str) + "_" + combined_df[c2].astype(str)
    interaction_cols.append(col_name)

# 6. Frequency Encodings
high_card_cols = ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "ord_5", "day_month"]
freq_features = []
for col in high_card_cols:
    freq_map = combined_df.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    freq_feat = combined_df[col].map(freq_map).fillna(0.0).values.astype(np.float32).reshape(-1, 1)
    freq_features.append(freq_feat)

X_freq = np.hstack(freq_features)

# 7. Out-of-Fold Target Encoding
te_cols = [
    "nom_0",
    "nom_1",
    "nom_2",
    "nom_3",
    "nom_4",
    "nom_5",
    "nom_6",
    "nom_7",
    "nom_8",
    "nom_9",
    "ord_1",
    "ord_2",
    "ord_3",
    "ord_4",
    "ord_5",
    "day",
    "month",
]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_te = np.zeros((n_train, len(te_cols)), dtype=np.float32)
test_te = np.zeros((len(test_df), len(te_cols)), dtype=np.float32)

train_part = combined_df.iloc[:n_train].copy()
test_part = combined_df.iloc[n_train:].copy()

smoothing = 10.0
global_mean = np.mean(y_train)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_part, y_train)):
    fold_trn = train_part.iloc[trn_idx]
    fold_val = train_part.iloc[val_idx]

    for c_idx, col in enumerate(te_cols):
        fold_trn_col = fold_trn[col].values
        fold_y = y_train[trn_idx]
        df_tmp = pd.DataFrame({"cat": fold_trn_col, "target": fold_y})
        stats = df_tmp.groupby("cat")["target"].agg(["count", "mean"])

        smooth_map = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        train_te[val_idx, c_idx] = (
            fold_val[col].map(smooth_map).fillna(global_mean).values
        )
        test_te[:, c_idx] += (
            test_part[col].map(smooth_map).fillna(global_mean).values / 5.0
        )

X_te = np.vstack([train_te, test_te])

# Standardize numerical features
num_cols = [
    "bin_0",
    "bin_1",
    "bin_2",
    "bin_3_num",
    "bin_4_num",
    "ord_0",
    "ord_1_num",
    "ord_2_num",
    "ord_3_num",
    "ord_4_num",
    "ord_5_num",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]
X_dense_raw = combined_df[num_cols].values.astype(np.float32)
X_dense_combined = np.hstack([X_dense_raw, X_freq, X_te])

X_dense_combined = np.nan_to_num(X_dense_combined, nan=0.0)
scaler = StandardScaler()
X_dense_train = scaler.fit_transform(X_dense_combined[:n_train])
X_dense_test = scaler.transform(X_dense_combined[n_train:])
X_dense = np.vstack([X_dense_train, X_dense_test])

# 8. Sparse One-Hot Encoding
cat_cols = [
    "bin_0",
    "bin_1",
    "bin_2",
    "bin_3",
    "bin_4",
    "nom_0",
    "nom_1",
    "nom_2",
    "nom_3",
    "nom_4",
    "nom_5",
    "nom_6",
    "nom_7",
    "nom_8",
    "nom_9",
    "ord_0",
    "ord_1",
    "ord_2",
    "ord_3",
    "ord_4",
    "ord_5",
    "ord_5_char1",
    "ord_5_char2",
    "day",
    "month",
] + interaction_cols

ohe = OneHotEncoder(dtype=np.float32, handle_unknown="ignore")
ohe.fit(combined_df.iloc[:n_train][cat_cols].astype(str))
X_ohe = ohe.transform(combined_df[cat_cols].astype(str))

# Combine Sparse One-Hot and Scaled Dense Features
X_sparse = sparse.hstack([X_ohe, sparse.csr_matrix(X_dense)], format="csr")

X_train_final = X_sparse[:n_train]
X_test_final = X_sparse[n_train:]

# 9. Model Training and Out-of-Fold Evaluation
print("Starting 5-fold cross-validation training...")
n_test = X_test_final.shape[0]

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 30,
    "random_state": 42,
    "n_estimators": 1500,
    "verbose": -1,
    "n_jobs": -1,
}

model_configs = [
    ("LR (C=1.0)", lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)),
    ("LR (C=2.5)", lambda: LogisticRegression(C=2.5, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)),
    ("LR (C=5.0)", lambda: LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)),
    ("LR (C=8.0)", lambda: LogisticRegression(C=8.0, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)),
    ("LR (C=12.0)", lambda: LogisticRegression(C=12.0, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)),
    ("LightGBM", lambda: lgb.LGBMClassifier(**lgb_params)),
]

n_models = len(model_configs)
oof_preds = np.zeros((n_models, n_train), dtype=np.float32)
test_preds = np.zeros((n_models, n_test), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_final, y_train)):
    X_trn, y_trn = X_train_final[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train_final[val_idx], y_train[val_idx]

    fold_scores = []
    for m_idx, (m_name, model_fn) in enumerate(model_configs):
        clf = model_fn()
        if "LightGBM" in m_name:
            clf.fit(
                X_trn,
                y_trn,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            clf.fit(X_trn, y_trn)

        val_pred = clf.predict_proba(X_val)[:, 1]
        oof_preds[m_idx, val_idx] = val_pred
        test_preds[m_idx] += clf.predict_proba(X_test_final)[:, 1] / 5.0

        score_fold = roc_auc_score(y_val, val_pred)
        fold_scores.append(f"{m_name}: {score_fold:.5f}")

    print(f"Fold {fold+1} - " + " | ".join(fold_scores))

# Log individual model OOF scores
for m_idx, (m_name, _) in enumerate(model_configs):
    score_m = roc_auc_score(y_train, oof_preds[m_idx])
    print(f"OOF AUC for {m_name}: {score_m:.5f}")

# Logit-transform OOF and test predictions for meta-learner
eps = 1e-6
X_meta_train = logit(np.clip(oof_preds.T, eps, 1.0 - eps))
X_meta_test = logit(np.clip(test_preds.T, eps, 1.0 - eps))

# Train Logistic Regression meta-learner on logit-transformed OOF probabilities
meta_oof = np.zeros(n_train, dtype=np.float32)
meta_test = np.zeros(n_test, dtype=np.float32)

for trn_idx, val_idx in skf.split(X_meta_train, y_train):
    meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_clf.fit(X_meta_train[trn_idx], y_train[trn_idx])
    meta_oof[val_idx] = meta_clf.predict_proba(X_meta_train[val_idx])[:, 1]
    meta_test += meta_clf.predict_proba(X_meta_test)[:, 1] / 5.0

meta_score = roc_auc_score(y_train, meta_oof)
print(f"Meta Logistic Regression OOF AUC: {meta_score:.5f}")

# Refined OOF stacking and aggregation focusing on top-performing base models and meta-learner
base_auc_scores = [roc_auc_score(y_train, oof_preds[i]) for i in range(n_models)]
top_threshold = max(base_auc_scores) - 0.02
top_indices = [i for i, score in enumerate(base_auc_scores) if score >= top_threshold]

print(f"Top base models selected for ensembling: {[model_configs[i][0] for i in top_indices]}")

meta_oof_rank = rankdata(meta_oof) / len(meta_oof)
meta_test_rank = rankdata(meta_test) / len(meta_test)

if len(top_indices) > 0:
    top_oof_ranks = np.column_stack([rankdata(oof_preds[i]) / len(y_train) for i in top_indices])
    top_test_ranks = np.column_stack([rankdata(test_preds[i]) / len(test_preds[i]) for i in top_indices])

    top_oof_avg = np.mean(top_oof_ranks, axis=1)
    top_test_avg = np.mean(top_test_ranks, axis=1)

    blend_oof = 0.90 * meta_oof_rank + 0.10 * top_oof_avg
    if roc_auc_score(y_train, blend_oof) > roc_auc_score(y_train, meta_oof_rank):
        final_oof = blend_oof
        final_test_preds = 0.90 * meta_test_rank + 0.10 * top_test_avg
    else:
        final_oof = meta_oof_rank
        final_test_preds = meta_test_rank
else:
    final_oof = meta_oof_rank
    final_test_preds = meta_test_rank

best_score = roc_auc_score(y_train, final_oof)

# 10. Test Predictions Inference & Save
sub_df = pd.DataFrame({"id": test_ids.astype(int), "target": final_test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

# Final Validation Score
print(f"Final Validation Score: {best_score}")