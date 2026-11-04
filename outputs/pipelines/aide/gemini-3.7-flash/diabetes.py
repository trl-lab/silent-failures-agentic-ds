import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from catboost import CatBoostClassifier

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
target_str = train_df[target_col].astype(str).str.strip().str.lower()
y = np.where(target_str.isin(["tested_positive", "1", "positive", "true"]), 1, 0)

feature_cols = [c for c in train_df.columns if c != target_col]


def engineer_features(df):
    data = df.copy()

    col_mapping = {
        "glucose": "plas",
        "blood_pressure": "pres",
        "skin_thickness": "skin",
        "insulin": "insu",
        "bmi": "mass",
        "pedigree": "pedi",
        "pregnancies": "preg",
        "age": "age",
    }
    for k, v in col_mapping.items():
        if k in data.columns and v not in data.columns:
            data.rename(columns={k: v}, inplace=True)

    # Biologically invalid zeros that represent missing measurements
    zero_as_nan_cols = ["plas", "pres", "skin", "insu", "mass"]
    actual_nan_cols = [c for c in zero_as_nan_cols if c in data.columns]
    for col in actual_nan_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace(0, np.nan)

    # Missingness indicators
    data["missing_count"] = data[actual_nan_cols].isna().sum(axis=1)

    # Domain interaction & ratio features
    if "plas" in data.columns and "insu" in data.columns:
        data["homa_ir"] = (data["plas"] * data["insu"]) / 405.0
        data["glucose_insulin_ratio"] = data["plas"] / (data["insu"] + 1e-5)

    if "mass" in data.columns and "plas" in data.columns:
        data["metabolic_risk"] = data["mass"] * data["plas"]

    if "preg" in data.columns and "age" in data.columns:
        data["preg_age_ratio"] = data["preg"] / (data["age"] + 1e-5)

    if "pedi" in data.columns and "age" in data.columns:
        data["genetic_age_risk"] = data["pedi"] * data["age"]

    if "plas" in data.columns and "age" in data.columns:
        data["plas_age_interaction"] = data["plas"] * data["age"]

    if "mass" in data.columns and "pres" in data.columns:
        data["mass_pres_ratio"] = data["mass"] / (data["pres"] + 1e-5)

    return data


X_train_df = engineer_features(train_df[feature_cols])
X_test_df = engineer_features(test_df[feature_cols])

num_cols = [c for c in X_train_df.columns if c != target_col]
X_train = X_train_df[num_cols].values.astype(np.float64)
X_test = X_test_df[num_cols].values.astype(np.float64)

seeds = [42, 101, 2024, 777]
n_splits = 5

oof_ensemble_preds = np.zeros(len(train_df))
test_ensemble_preds = np.zeros(len(test_df))

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_seed = np.zeros(len(train_df))
    test_seed = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train[train_idx], y[train_idx]
        X_val, y_val = X_train[val_idx], y[val_idx]

        # Model 1: Regularized Logistic Regression with MICE + PowerTransformer
        lr_pipe = Pipeline(
            [
                ("imputer", IterativeImputer(max_iter=30, random_state=seed + fold)),
                ("power", PowerTransformer(method="yeo-johnson")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.35,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )

        # Model 2: RBF Kernel Support Vector Classifier
        svc_pipe = Pipeline(
            [
                ("imputer", IterativeImputer(max_iter=30, random_state=seed + fold)),
                ("power", PowerTransformer(method="yeo-johnson")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=0.8,
                        kernel="rbf",
                        probability=True,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )

        # Model 3: Regularized HistGradientBoosting
        hgb_pipe = Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=70,
                        learning_rate=0.035,
                        max_leaf_nodes=12,
                        min_samples_leaf=14,
                        l2_regularization=2.0,
                        random_state=seed + fold,
                    ),
                )
            ]
        )

        # Model 4: CatBoost with native missing handling
        cb_model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            random_seed=seed + fold,
            verbose=False,
        )

        # Model 5: ExtraTrees with MICE Imputer
        et_pipe = Pipeline(
            [
                ("imputer", IterativeImputer(max_iter=30, random_state=seed + fold)),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        max_depth=5,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )

        lr_pipe.fit(X_tr, y_tr)
        svc_pipe.fit(X_tr, y_tr)
        hgb_pipe.fit(X_tr, y_tr)
        cb_model.fit(X_tr, y_tr)
        et_pipe.fit(X_tr, y_tr)

        models = [lr_pipe, svc_pipe, hgb_pipe, cb_model, et_pipe]
        weights = [0.35, 0.25, 0.15, 0.15, 0.10]

        val_rank_sum = np.zeros(len(val_idx))
        test_rank_sum = np.zeros(len(test_df))

        for model, w in zip(models, weights):
            val_p = model.predict_proba(X_val)[:, 1]
            test_p = model.predict_proba(X_test)[:, 1]

            val_r = rankdata(val_p) / len(val_p)
            test_r = rankdata(test_p) / len(test_p)

            val_rank_sum += w * val_r
            test_rank_sum += w * test_r

        oof_seed[val_idx] = val_rank_sum
        test_seed += test_rank_sum / n_splits

    oof_ensemble_preds += oof_seed / len(seeds)
    test_ensemble_preds += test_seed / len(seeds)

cv_auc = roc_auc_score(y, oof_ensemble_preds)
print(f"Multi-Seed Rank-Averaged Ensemble 5-Fold CV ROC-AUC: {cv_auc:.5f}")

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": test_ensemble_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
