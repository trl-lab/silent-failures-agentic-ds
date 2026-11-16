import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb


def extract_target(df):
    target = df["Class"]
    s = target.astype(str).str.strip()
    if (s == "2").any() or (s == "2.0").any():
        return np.where((s == "2") | (s == "2.0"), 1, 0)
    else:
        return np.where((s == "1") | (s == "1.0"), 1, 0)


def engineer_features(df):
    data = pd.DataFrame(index=df.index)

    v1 = df["V1"].astype(float)  # Recency (months since last donation)
    v2 = df["V2"].astype(float)  # Frequency (total donations)
    v4 = df["V4"].astype(float)  # Time (months since first donation)

    data["recency"] = v1
    data["frequency"] = v2
    data["time"] = v4

    # Domain features
    avg_interval = v4 / np.maximum(v2, 1.0)
    data["avg_interval"] = avg_interval
    data["active_span"] = np.maximum(0.0, v4 - v1)
    data["recency_to_time"] = v1 / (v4 + 1.0)
    data["overdue_ratio"] = v1 / (avg_interval + 1.0)
    data["overdue_months"] = v1 - avg_interval
    data["is_single_donor"] = (v2 <= 1.0).astype(float)
    data["donations_per_month"] = v2 / (v4 + 1.0)

    # Nonlinear transformations
    data["log_recency"] = np.log1p(v1)
    data["log_frequency"] = np.log1p(v2)
    data["log_time"] = np.log1p(v4)
    data["log_avg_interval"] = np.log1p(avg_interval)

    return data


def main():
    os.makedirs("./working", exist_ok=True)

    # Load dataset
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    y = extract_target(train)

    feature_cols = [c for c in train.columns if c != "Class"]
    X_train_df = engineer_features(train[feature_cols])
    X_test_df = engineer_features(test[feature_cols])

    X = X_train_df.values
    X_test = X_test_df.values

    # Setup Stratified Cross-Validation
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)

        seed = 42 + fold

        # 1. Logistic Regression
        lr = LogisticRegression(C=0.2, max_iter=1000, random_state=seed)
        lr.fit(X_tr_sc, y_tr)
        p_lr_va = lr.predict_proba(X_va_sc)[:, 1]
        p_lr_te = lr.predict_proba(X_te_sc)[:, 1]

        # 2. Support Vector Classifier
        svc = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
        svc.fit(X_tr_sc, y_tr)
        p_svc_va = svc.predict_proba(X_va_sc)[:, 1]
        p_svc_te = svc.predict_proba(X_te_sc)[:, 1]

        # 3. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            num_leaves=3,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_lgb_va = lgb_model.predict_proba(X_va)[:, 1]
        p_lgb_te = lgb_model.predict_proba(X_test)[:, 1]

        # 4. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="auc",
            early_stopping_rounds=30,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb_va = xgb_model.predict_proba(X_va)[:, 1]
        p_xgb_te = xgb_model.predict_proba(X_test)[:, 1]

        # 5. Extra Trees
        et = ExtraTreesClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=4, random_state=seed
        )
        et.fit(X_tr, y_tr)
        p_et_va = et.predict_proba(X_va)[:, 1]
        p_et_te = et.predict_proba(X_test)[:, 1]

        # 6. Random Forest
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=4, random_state=seed
        )
        rf.fit(X_tr, y_tr)
        p_rf_va = rf.predict_proba(X_va)[:, 1]
        p_rf_te = rf.predict_proba(X_test)[:, 1]

        # Rank transform fold predictions for robust blending
        va_preds_list = [
            p_lr_va,
            p_svc_va,
            p_lgb_va,
            p_xgb_va,
            p_et_va,
            p_rf_va,
        ]
        te_preds_list = [
            p_lr_te,
            p_svc_te,
            p_lgb_te,
            p_xgb_te,
            p_et_te,
            p_rf_te,
        ]

        weights = [0.20, 0.15, 0.20, 0.20, 0.125, 0.125]

        fold_oof_rank = np.zeros(len(val_idx))
        fold_test_rank = np.zeros(len(test))

        for p_va, p_te, w in zip(va_preds_list, te_preds_list, weights):
            fold_oof_rank += w * (rankdata(p_va) / len(p_va))
            fold_test_rank += w * (rankdata(p_te) / len(p_te))

        oof_preds[val_idx] = fold_oof_rank
        test_preds += fold_test_rank / n_splits

    cv_auc = roc_auc_score(y, oof_preds)
    print(f"5-Fold Cross-Validation Ensembled ROC AUC: {cv_auc:.5f}")

    sub = pd.DataFrame({"id": np.arange(len(test_preds)), "Class": test_preds})
    sub.to_csv("./working/submission.csv", index=False)
    print(f"Submission saved to ./working/submission.csv with shape {sub.shape}")


if __name__ == "__main__":
    main()
