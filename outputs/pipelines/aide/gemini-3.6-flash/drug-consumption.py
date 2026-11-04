import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "cannabis"
id_col = "id" if "id" in test_df.columns else None
base_features = [c for c in test_df.columns if c != "id"]


# 2. Domain Feature Engineering
def extract_domain_features(df):
    df = df.copy()

    nscore = df["nscore"] if "nscore" in df.columns else 0
    escore = df["escore"] if "escore" in df.columns else 0
    oscore = df["oscore"] if "oscore" in df.columns else 0
    ascore = df["ascore"] if "ascore" in df.columns else 0
    cscore = df["cscore"] if "cscore" in df.columns else 0

    imp_col = [c for c in df.columns if "imp" in c.lower()]
    imp = df[imp_col[0]] if len(imp_col) > 0 else 0
    ss = df["ss"] if "ss" in df.columns else 0

    # Big Five summary statistics
    big5_matrix = df[["nscore", "escore", "oscore", "ascore", "cscore"]].values
    df["big5_mean"] = big5_matrix.mean(axis=1)
    df["big5_std"] = big5_matrix.std(axis=1)
    df["big5_sum"] = big5_matrix.sum(axis=1)

    # Psychological Domain Indices
    df["risk_index"] = (oscore + ss + imp) - (cscore + ascore)
    df["stability_index"] = (cscore + ascore) - nscore

    # Trait interactions & ratios
    df["oscore_ss"] = oscore * ss
    df["cscore_imp"] = cscore * imp
    df["nscore_cscore_diff"] = nscore - cscore
    df["oscore_cscore_ratio"] = oscore / (cscore.abs() + 1e-4)
    df["ss_cscore_ratio"] = ss / (cscore.abs() + 1e-4)
    df["ss_imp_sum"] = ss + imp
    df["ss_imp_prod"] = ss * imp

    return df


combined = pd.concat(
    [train_df[base_features], test_df[base_features]], axis=0
).reset_index(drop=True)
combined = extract_domain_features(combined)

# Group-level demographic aggregations
demo_cols = ["age", "country", "gender", "education", "ethnicity"]
val_cols = ["ss", "oscore", "big5_mean", "risk_index"]

for group_col in demo_cols:
    if group_col in combined.columns:
        rounded_group = combined[group_col].round(4)
        for val_col in val_cols:
            if val_col in combined.columns:
                group_mean = combined.groupby(rounded_group)[val_col].transform("mean")
                combined[f"{val_col}_mean_by_{group_col}"] = group_mean
                combined[f"{val_col}_diff_by_{group_col}"] = (
                    combined[val_col] - group_mean
                )

train_feat = combined.iloc[: len(train_df)].copy()
test_feat = combined.iloc[len(train_df) :].copy()

train_feat = train_feat.fillna(train_feat.mean())
test_feat = test_feat.fillna(train_feat.mean())

# 3. Target Encoding
le = LabelEncoder()
y = le.fit_transform(train_df[target_col])
num_classes = len(le.classes_)

# 4. Stratified K-Fold Cross-Validation with Ordinal & Multi-class Models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

num_mc_models = 5
num_cum_models = num_classes - 1  # 6 cumulative binary targets for 7 classes
num_reg_models = 2  # LGBM Regressor + Scaled Ridge Regressor

total_meta_feats = (num_mc_models * num_classes) + num_cum_models + num_reg_models

oof_base_features = np.zeros((len(train_df), total_meta_feats))
test_base_features = np.zeros((len(test_df), total_meta_feats))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_feat, y)):
    X_train, y_train = train_feat.iloc[train_idx], y[train_idx]
    X_val, y_val = train_feat.iloc[val_idx], y[val_idx]

    # --- Section A: Multi-class Base Models ---
    model_lgb = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(X_train, y_train)

    model_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=9, random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_train, y_train)

    model_hgb = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.02, max_leaf_nodes=15, random_state=42 + fold
    )
    model_hgb.fit(X_train, y_train)

    model_rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42 + fold, n_jobs=-1
    )
    model_rf.fit(X_train, y_train)

    model_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, max_iter=1000, random_state=42 + fold, multi_class="multinomial"
        ),
    )
    model_lr.fit(X_train, y_train)

    val_mc_probs = np.hstack(
        [
            model_lgb.predict_proba(X_val),
            model_et.predict_proba(X_val),
            model_hgb.predict_proba(X_val),
            model_rf.predict_proba(X_val),
            model_lr.predict_proba(X_val),
        ]
    )

    test_mc_probs = np.hstack(
        [
            model_lgb.predict_proba(test_feat),
            model_et.predict_proba(test_feat),
            model_hgb.predict_proba(test_feat),
            model_rf.predict_proba(test_feat),
            model_lr.predict_proba(test_feat),
        ]
    )

    # --- Section B: Ordinal Cumulative Binary Models (y >= k) ---
    val_cum_probs = []
    test_cum_probs = []
    for k in range(1, num_classes):
        y_train_bin = (y_train >= k).astype(int)
        model_cum = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=10,
            random_state=42 + fold,
            verbosity=-1,
        )
        model_cum.fit(X_train, y_train_bin)
        val_cum_probs.append(model_cum.predict_proba(X_val)[:, 1:2])
        test_cum_probs.append(model_cum.predict_proba(test_feat)[:, 1:2])

    val_cum_mat = np.hstack(val_cum_probs)
    test_cum_mat = np.hstack(test_cum_probs)

    # --- Section C: Ordinal Continuous Regression Intensity Models ---
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        random_state=42 + fold,
        verbosity=-1,
    )
    reg_lgb.fit(X_train, y_train)

    reg_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg_ridge.fit(X_train, y_train)

    val_reg = np.hstack(
        [
            reg_lgb.predict(X_val)[:, np.newaxis],
            reg_ridge.predict(X_val)[:, np.newaxis],
        ]
    )
    test_reg = np.hstack(
        [
            reg_lgb.predict(test_feat)[:, np.newaxis],
            reg_ridge.predict(test_feat)[:, np.newaxis],
        ]
    )

    # Combine fold meta-features
    val_meta_all = np.hstack([val_mc_probs, val_cum_mat, val_reg])
    test_meta_all = np.hstack([test_mc_probs, test_cum_mat, test_reg])

    oof_base_features[val_idx] = val_meta_all
    test_base_features += test_meta_all / skf.n_splits

# 5. Meta-Learner Stacking
meta_model = LogisticRegression(
    C=0.05, max_iter=1000, random_state=42, multi_class="multinomial"
)

oof_meta_probs = np.zeros((len(train_df), num_classes))
test_meta_probs = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(oof_base_features, y)):
    X_tr_meta, y_tr_meta = oof_base_features[train_idx], y[train_idx]
    X_val_meta = oof_base_features[val_idx]

    meta_model.fit(X_tr_meta, y_tr_meta)
    oof_meta_probs[val_idx] = meta_model.predict_proba(X_val_meta)
    test_meta_probs += meta_model.predict_proba(test_base_features) / skf.n_splits


# 6. Post-Processing Optimization: Powell Class Multipliers
def loss_func(weights, probs, targets):
    scaled_probs = probs * weights
    preds = np.argmax(scaled_probs, axis=1)
    return -accuracy_score(targets, preds)


init_weights = np.ones(num_classes)
opt_result = minimize(
    loss_func,
    init_weights,
    args=(oof_meta_probs, y),
    method="Powell",
    options={"maxiter": 1000, "ftol": 1e-5},
)
opt_weights = opt_result.x

# Compute Final Cross-Validation Accuracy
oof_classes_opt = np.argmax(oof_meta_probs * opt_weights, axis=1)
cv_accuracy = accuracy_score(y, oof_classes_opt)
print(f"5-Fold CV Accuracy: {cv_accuracy:.4f}")

# 7. Generate Predictions & Save Submission File
test_classes = np.argmax(test_meta_probs * opt_weights, axis=1)
test_labels = le.inverse_transform(test_classes)

os.makedirs("./working", exist_ok=True)
sub_id = (
    test_df[id_col] if id_col and id_col in test_df.columns else np.arange(len(test_df))
)

submission = pd.DataFrame({"id": sub_id, "cannabis": test_labels})
submission.to_csv("./working/submission.csv", index=False)
