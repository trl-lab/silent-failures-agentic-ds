import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")


def main():
    # 1. Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")
    target_col = "Class"

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))
    print(f"Target distribution: {np.bincount(y)} (classes: {le.classes_})")

    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    # 2. Setup Cross-Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_names = ["lgb", "xgb", "cat", "rf", "et", "lr", "svc"]
    oof_preds = {m: np.zeros(len(train_df)) for m in model_names}
    test_preds = {m: np.zeros(len(test_df)) for m in model_names}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_imp, y)):
        X_tr, y_tr = X_train_imp.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train_imp.iloc[val_idx], y[val_idx]
        X_te = X_test_imp.copy()

        # Scaled versions for linear models and SVM
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        # 1. LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_preds["lgb"][val_idx] = lgb_clf.predict_proba(X_va)[:, 1]
        test_preds["lgb"] += lgb_clf.predict_proba(X_te)[:, 1] / n_splits

        # 2. XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds["xgb"][val_idx] = xgb_clf.predict_proba(X_va)[:, 1]
        test_preds["xgb"] += xgb_clf.predict_proba(X_te)[:, 1] / n_splits

        # 3. CatBoost
        cat_clf = CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            eval_metric="AUC",
            early_stopping_rounds=50,
            verbose=False,
        )
        cat_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds["cat"][val_idx] = cat_clf.predict_proba(X_va)[:, 1]
        test_preds["cat"] += cat_clf.predict_proba(X_te)[:, 1] / n_splits

        # 4. Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf_clf.fit(X_tr, y_tr)
        oof_preds["rf"][val_idx] = rf_clf.predict_proba(X_va)[:, 1]
        test_preds["rf"] += rf_clf.predict_proba(X_te)[:, 1] / n_splits

        # 5. Extra Trees
        et_clf = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_clf.fit(X_tr, y_tr)
        oof_preds["et"][val_idx] = et_clf.predict_proba(X_va)[:, 1]
        test_preds["et"] += et_clf.predict_proba(X_te)[:, 1] / n_splits

        # 6. Logistic Regression
        lr_clf = LogisticRegression(C=0.3, max_iter=1000, random_state=42 + fold)
        lr_clf.fit(X_tr_sc, y_tr)
        oof_preds["lr"][val_idx] = lr_clf.predict_proba(X_va_sc)[:, 1]
        test_preds["lr"] += lr_clf.predict_proba(X_te_sc)[:, 1] / n_splits

        # 7. Support Vector Classifier (RBF kernel)
        svc_clf = SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=42 + fold,
            gamma="scale",
        )
        svc_clf.fit(X_tr_sc, y_tr)
        oof_preds["svc"][val_idx] = svc_clf.predict_proba(X_va_sc)[:, 1]
        test_preds["svc"] += svc_clf.predict_proba(X_te_sc)[:, 1] / n_splits

    # 3. Print validation scores
    print("\n--- Out-of-Fold ROC-AUC Scores ---")
    for m in model_names:
        score = roc_auc_score(y, oof_preds[m])
        print(f"{m.upper()}: {score:.5f}")

    # 4. Ensemble Weight Optimization
    pred_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        sum_w = np.sum(w)
        if sum_w == 0:
            return 0.0
        w = w / sum_w
        blended = pred_matrix @ w
        return -roc_auc_score(y, blended)

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1)] * len(model_names)
    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 500},
    )

    best_weights = np.maximum(res.x, 0)
    best_weights /= np.sum(best_weights)

    print("\nOptimized Ensemble Weights:")
    for m, w in zip(model_names, best_weights):
        print(f"{m.upper()}: {w:.4f}")

    final_oof = pred_matrix @ best_weights
    final_cv_auc = roc_auc_score(y, final_oof)
    print(f"\nFinal Ensemble Out-of-Fold ROC-AUC: {final_cv_auc:.5f}")

    # 5. Predict on test and save submission
    final_test_preds = test_matrix @ best_weights

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test_df)), target_col: final_test_preds}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Saved submission to ./working/submission.csv with shape {submission.shape}")


if __name__ == "__main__":
    main()
