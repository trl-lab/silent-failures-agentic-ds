import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Ensure output directories exist
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set deterministic random seeds
np.random.seed(42)

# ==============================================================================
# 1. DATA LOADING & ROW-WISE STATISTICAL FEATURE EXTRACTION
# ==============================================================================
print("Loading train and test parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

id_col = "id"
target_col = "target"
feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

train_ids = train_df[id_col].values
test_ids = test_df[id_col].values
y_train = train_df[target_col].astype(int).values

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()


def extract_row_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sample-wise (row-wise) statistical aggregates and distribution metrics.
    Because these operate per-instance independently, there is zero data leakage.
    """
    values = df_features.values.astype(np.float64)
    feats = pd.DataFrame(index=df_features.index)

    # Basic moments
    feats["row_mean"] = np.mean(values, axis=1)
    feats["row_std"] = np.std(values, axis=1)
    feats["row_median"] = np.median(values, axis=1)
    feats["row_min"] = np.min(values, axis=1)
    feats["row_max"] = np.max(values, axis=1)
    feats["row_skew"] = stats.skew(values, axis=1)
    feats["row_kurtosis"] = stats.kurtosis(values, axis=1)

    # Quantile spreads
    q10 = np.percentile(values, 10, axis=1)
    q25 = np.percentile(values, 25, axis=1)
    q75 = np.percentile(values, 75, axis=1)
    q90 = np.percentile(values, 90, axis=1)
    feats["row_q10"] = q10
    feats["row_q25"] = q25
    feats["row_q75"] = q75
    feats["row_q90"] = q90
    feats["row_iqr"] = q75 - q25
    feats["row_range"] = feats["row_max"] - feats["row_min"]

    # Vector norms
    feats["row_l1_norm"] = np.linalg.norm(values, ord=1, axis=1)
    feats["row_l2_norm"] = np.linalg.norm(values, ord=2, axis=1)

    # Extreme value counts / distribution tails
    feats["row_pos_count"] = np.sum(values > 0, axis=1)
    feats["row_neg_count"] = np.sum(values < 0, axis=1)
    feats["row_gt_1"] = np.sum(np.abs(values) > 1.0, axis=1)
    feats["row_gt_2"] = np.sum(np.abs(values) > 2.0, axis=1)
    feats["row_gt_2_5"] = np.sum(np.abs(values) > 2.5, axis=1)

    return feats


print("Extracting row-wise statistical features...")
train_row_feats = extract_row_features(X_train_raw)
test_row_feats = extract_row_features(X_test_raw)

X_train_combined = pd.concat([X_train_raw, train_row_feats], axis=1)
X_test_combined = pd.concat([X_test_raw, test_row_feats], axis=1)

X_train = X_train_combined.values
X_test = X_test_combined.values
n_samples, n_features = X_train.shape
n_test = len(X_test)

print(
    f"Engineered dataset shapes: Train={X_train.shape}, Test={X_test.shape}, Features={n_features}"
)

# ==============================================================================
# 2. MODEL ARCHITECTURES AND BUILDERS
# ==============================================================================


def select_stable_features(
    X_train_f: np.ndarray,
    y_train_f: np.ndarray,
    n_bootstrap: int = 30,
    subsample_ratio: float = 0.8,
    top_k: int = 35,
    random_state: int = 42,
) -> np.ndarray:
    """
    Leak-free fold-level stability feature selection using randomized L1 Logistic Regression.
    Identifies features with robust non-zero coefficients across resampled splits.
    """
    n_samples_f, n_features_f = X_train_f.shape
    feature_freq = np.zeros(n_features_f, dtype=float)
    rng = np.random.RandomState(random_state)

    for b in range(n_bootstrap):
        boot_idx = rng.choice(
            n_samples_f, size=int(n_samples_f * subsample_ratio), replace=True
        )
        c_val = rng.uniform(0.05, 0.20)
        lr = LogisticRegression(
            penalty="l1",
            C=c_val,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=rng.randint(0, 100000),
        )
        lr.fit(X_train_f[boot_idx], y_train_f[boot_idx])
        feature_freq += (np.abs(lr.coef_[0]) > 1e-5).astype(float)

    selected_indices = np.argsort(-feature_freq)[:top_k]
    return np.sort(selected_indices)


def build_bagged_l1(
    C: float = 0.1,
    n_estimators: int = 50,
    max_features: float = 0.7,
    random_state: int = 42,
):
    base_lr = LogisticRegression(
        penalty="l1",
        C=C,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    return BaggingClassifier(
        estimator=base_lr,
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=1,
    )


def build_l1_logistic(C: float = 0.1, random_state: int = 42):
    return LogisticRegression(
        penalty="l1",
        C=C,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )


def build_elasticnet_logistic(
    C: float = 0.1, l1_ratio: float = 0.7, random_state: int = 42
):
    return LogisticRegression(
        penalty="elasticnet",
        C=C,
        l1_ratio=l1_ratio,
        solver="saga",
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )


def build_calibrated_linearsvc(C: float = 0.08, random_state: int = 42):
    base_svc = LinearSVC(
        penalty="l1",
        dual=False,
        C=C,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)


def build_shrinkage_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def build_calibrated_ridge(alpha: float = 10.0, random_state: int = 42):
    base_ridge = RidgeClassifier(
        alpha=alpha, class_weight="balanced", random_state=random_state
    )
    return CalibratedClassifierCV(estimator=base_ridge, method="sigmoid", cv=3)


# ==============================================================================
# 3. REPEATED STRATIFIED K-FOLD TRAINING & EVALUATION PIPELINE
# ==============================================================================
n_repeats = 10
n_splits_per_repeat = 5
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits_per_repeat, n_repeats=n_repeats, random_state=42
)
cv_splits_repeated = list(rskf.split(X_train, y_train))
total_splits = len(cv_splits_repeated)

print(
    f"Starting 10x5 Repeated Stratified K-Fold cross-validation ({total_splits} total folds)..."
)

oof_repeats = np.zeros((n_repeats, n_samples), dtype=float)
test_predictions_accum = np.zeros(n_test, dtype=float)
fold_scores = []

for idx, (train_idx, val_idx) in enumerate(cv_splits_repeated):
    repeat_idx = idx // n_splits_per_repeat
    fold_num = (idx % n_splits_per_repeat) + 1

    # Isolate fold partitions
    X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
    X_te_fold = X_test.copy()

    # Leak-free feature standardisation strictly fitted on fold train set
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    X_te_scaled = scaler.transform(X_te_fold)

    # Leak-free stability feature selection strictly computed on fold training split
    selected_feat_idx = select_stable_features(
        X_tr_scaled, y_tr_fold, n_bootstrap=30, subsample_ratio=0.8, top_k=35, random_state=42 + idx
    )
    X_tr_sel = X_tr_scaled[:, selected_feat_idx]
    X_val_sel = X_val_scaled[:, selected_feat_idx]
    X_te_sel = X_te_scaled[:, selected_feat_idx]

    # Instantiate specialized suite of regularized linear estimators
    # Some operate on selected stable subspace, some on multi-scale L1/ElasticNet spaces
    models = {
        "bag_l1": (build_bagged_l1(C=0.1, n_estimators=50, max_features=0.7, random_state=42 + idx), "full"),
        "l1_c005": (build_l1_logistic(C=0.05, random_state=42 + idx), "full"),
        "l1_c010": (build_l1_logistic(C=0.10, random_state=42 + idx), "full"),
        "l1_c015": (build_l1_logistic(C=0.15, random_state=42 + idx), "full"),
        "l1_c025": (build_l1_logistic(C=0.25, random_state=42 + idx), "full"),
        "elastic_07": (build_elasticnet_logistic(C=0.10, l1_ratio=0.70, random_state=42 + idx), "full"),
        "elastic_085": (build_elasticnet_logistic(C=0.15, l1_ratio=0.85, random_state=42 + idx), "full"),
        "calibrated_svc": (build_calibrated_linearsvc(C=0.08, random_state=42 + idx), "full"),
        "lda_shrinkage": (build_shrinkage_lda(), "selected"),
        "calibrated_ridge": (build_calibrated_ridge(alpha=10.0, random_state=42 + idx), "selected"),
        "l1_sel_c01": (build_l1_logistic(C=0.10, random_state=42 + idx), "selected"),
    }

    val_preds_ranked = []
    te_preds_ranked = []
    fold_saved_models = {}

    for name, (model, feat_mode) in models.items():
        tr_data = X_tr_sel if feat_mode == "selected" else X_tr_scaled
        val_data = X_val_sel if feat_mode == "selected" else X_val_scaled
        te_data = X_te_sel if feat_mode == "selected" else X_te_scaled

        model.fit(tr_data, y_tr_fold)
        val_p = model.predict_proba(val_data)[:, 1]
        te_p = model.predict_proba(te_data)[:, 1]

        # Rank-normalization within fold
        val_preds_ranked.append(rankdata(val_p) / len(val_p))
        te_preds_ranked.append(rankdata(te_p) / len(te_p))
        fold_saved_models[name] = model

    # Rank-averaged ensemble blend
    fold_val_pred = np.mean(val_preds_ranked, axis=0)
    fold_te_pred = np.mean(te_preds_ranked, axis=0)

    oof_repeats[repeat_idx, val_idx] = fold_val_pred
    test_predictions_accum += fold_te_pred / total_splits

    fold_auc = roc_auc_score(y_val_fold, fold_val_pred)
    fold_scores.append(fold_auc)

    # Save fold artifacts
    joblib.dump(
        {
            "scaler": scaler,
            "selected_features": selected_feat_idx,
            "models": fold_saved_models,
        },
        f"./working/models/fold_{idx}.joblib",
    )

    print(
        f"Iter {idx + 1:02d}/{total_splits:02d} [Repeat {repeat_idx + 1}, Fold {fold_num}] - Val ROC-AUC: {fold_auc:.5f}"
    )

# ==============================================================================
# 4. FINAL AGGREGATED EVALUATION & SUBMISSION GENERATION
# ==============================================================================
oof_repeat_ranks = np.zeros_like(oof_repeats)
for r in range(n_repeats):
    oof_repeat_ranks[r] = rankdata(oof_repeats[r]) / n_samples
overall_oof_preds = np.mean(oof_repeat_ranks, axis=0)
final_val_score = roc_auc_score(y_train, overall_oof_preds)

pd.DataFrame(
    {"id": train_ids, "target": y_train, "oof_pred": overall_oof_preds}
).to_csv("./working/oof_predictions.csv", index=False)

# Format & write test submission
test_predictions_clipped = np.clip(test_predictions_accum, 0.0, 1.0)
submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "target": test_predictions_clipped}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Mean Fold ROC-AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
print(
    f"Submission generated at ./submission/submission.csv with {len(submission_df)} rows."
)
print(f"Final Validation Score: {final_val_score:.6f}")
