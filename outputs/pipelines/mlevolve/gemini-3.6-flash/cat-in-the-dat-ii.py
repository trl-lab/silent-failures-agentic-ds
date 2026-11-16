import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from xgboost import XGBClassifier


def process_and_engineer_features():
    print("Starting data processing and feature engineering...")
    os.makedirs("./working", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_ids = train_df["id"].values
    test_ids = test_df["id"].values
    y_train = train_df["target"].values

    train_df = train_df.drop(columns=["id", "target"])
    test_df = test_df.drop(columns=["id"])

    n_train = len(train_df)
    n_test = len(test_df)

    # Combine temporarily for consistent categorical mappings
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 2. Domain Mappings for Ordinal & Binary Features
    ord_1_map = {
        "Novice": 1,
        "Contributor": 2,
        "Expert": 3,
        "Master": 4,
        "Grandmaster": 5,
    }
    ord_2_map = {
        "Freezing": 1,
        "Cold": 2,
        "Warm": 3,
        "Hot": 4,
        "Boiling Hot": 5,
        "Lava Hot": 6,
    }
    ord_3_map = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    ord_4_map = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

    bin_3_map = {"F": 0, "T": 1}
    bin_4_map = {"N": 0, "Y": 1}

    full_df["bin_3_num"] = full_df["bin_3"].map(bin_3_map)
    full_df["bin_4_num"] = full_df["bin_4"].map(bin_4_map)

    full_df["ord_1_num"] = full_df["ord_1"].map(ord_1_map)
    full_df["ord_2_num"] = full_df["ord_2"].map(ord_2_map)
    full_df["ord_3_num"] = full_df["ord_3"].map(ord_3_map)
    full_df["ord_4_num"] = full_df["ord_4"].map(ord_4_map)

    # ord_5 split into char1 and char2 ranks
    ord_5_str = full_df["ord_5"].astype(str).str.rjust(2, " ")
    char_map = {
        c: i + 1
        for i, c in enumerate(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    }
    full_df["ord_5_char1"] = ord_5_str.str[0].map(char_map)
    full_df["ord_5_char2"] = ord_5_str.str[1].map(char_map)
    full_df["ord_5_rank_sum"] = full_df["ord_5_char1"].fillna(0) + full_df[
        "ord_5_char2"
    ].fillna(0)

    # 3. Cyclical Trigonometric Features
    full_df["day_sin"] = np.sin(2 * np.pi * full_df["day"] / 7.0)
    full_df["day_cos"] = np.cos(2 * np.pi * full_df["day"] / 7.0)
    full_df["month_sin"] = np.sin(2 * np.pi * full_df["month"] / 12.0)
    full_df["month_cos"] = np.cos(2 * np.pi * full_df["month"] / 12.0)

    # 4. Missing Value Indicators
    original_cols = [c for c in train_df.columns]
    for col in original_cols:
        full_df[f"{col}_isna"] = full_df[col].isna().astype(np.int8)

    # 5. Pairwise Feature Interactions
    interactions = [
        ("nom_0", "nom_1"),
        ("nom_0", "nom_2"),
        ("nom_0", "day"),
        ("day", "month"),
        ("ord_1", "ord_2"),
        ("nom_5", "nom_6"),
        ("nom_7", "nom_8"),
        ("nom_0", "month"),
    ]

    for c1, c2 in interactions:
        full_df[f"{c1}_{c2}"] = full_df[c1].astype(str) + "_" + full_df[c2].astype(str)

    # Convert remaining string/object columns to category codes consistently before split
    object_cols = full_df.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        full_df[col] = pd.Categorical(full_df[col]).codes

    # Split back to Train and Test
    train_features = full_df.iloc[:n_train].copy()
    test_features = full_df.iloc[n_train:].copy()

    # 6. Frequency Encoding (Fitted on Train)
    freq_cols = original_cols + [f"{c1}_{c2}" for c1, c2 in interactions]
    for col in freq_cols:
        freq_map = train_features[col].value_counts(dropna=False).to_dict()
        train_features[f"{col}_freq"] = (
            train_features[col].map(freq_map).fillna(0).astype(np.int32)
        )
        test_features[f"{col}_freq"] = (
            test_features[col].map(freq_map).fillna(0).astype(np.int32)
        )

    # 7. Out-Of-Fold Smoothed Target Encoding
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
    ] + [f"{c1}_{c2}" for c1, c2 in interactions]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m_smoothing = 10.0
    global_mean = y_train.mean()

    fold_indices = np.zeros(n_train, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y_train)):
        fold_indices[val_idx] = fold

    for col in te_cols:
        train_features[f"{col}_te"] = np.nan
        test_te_folds = np.zeros((n_test, 5))

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y_train)):
            tr_sub = train_features.iloc[train_idx]
            y_sub = y_train[train_idx]

            stats = (
                pd.DataFrame({col: tr_sub[col], "target": y_sub})
                .groupby(col, dropna=False)["target"]
                .agg(["count", "mean"])
            )
            smooth = (stats["count"] * stats["mean"] + m_smoothing * global_mean) / (
                stats["count"] + m_smoothing
            )
            smooth_dict = smooth.to_dict()

            val_sub = train_features.iloc[val_idx]
            col_te_loc = train_features.columns.get_loc(f"{col}_te")
            train_features.iloc[val_idx, col_te_loc] = (
                val_sub[col].map(smooth_dict).fillna(global_mean).values
            )
            test_te_folds[:, fold] = (
                test_features[col].map(smooth_dict).fillna(global_mean).values
            )

        test_features[f"{col}_te"] = test_te_folds.mean(axis=1)

    # 8. Category encoding already handled on full_df prior to splitting

    train_features["target"] = y_train
    train_features["fold"] = fold_indices
    train_features["id"] = train_ids
    test_features["id"] = test_ids

    print(
        f"Data processing completed. Processed train shape: {train_features.shape}, test shape: {test_features.shape}"
    )
    return train_features, test_features


def get_lgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_callbacks():
    return [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]


def train_and_evaluate():
    train_df, test_df = process_and_engineer_features()

    y_true = train_df["target"].values
    fold_indices = train_df["fold"].values
    features = [c for c in train_df.columns if c not in ["id", "target", "fold"]]

    n_train = len(train_df)
    n_test = len(test_df)
    n_folds = len(np.unique(fold_indices))

    oof_preds_lgb = np.zeros(n_train, dtype=np.float64)
    oof_preds_xgb = np.zeros(n_train, dtype=np.float64)

    test_preds_lgb = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)

    X_test = test_df[features]

    # 5-Fold Stratified CV
    for fold in range(n_folds):
        tr_mask = fold_indices != fold
        val_mask = fold_indices == fold

        X_tr, y_tr = train_df.loc[tr_mask, features], y_true[tr_mask]
        X_val, y_val = train_df.loc[val_mask, features], y_true[val_mask]

        # LightGBM Classifier
        lgb_model = get_lgbm_model(random_state=42 + fold)
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=get_callbacks())

        val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_mask] = val_pred_lgb
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_folds

        # XGBoost Classifier
        xgb_model = get_xgb_model(random_state=42 + fold)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
        oof_preds_xgb[val_mask] = val_pred_xgb
        test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_folds

        fold_auc_lgb = roc_auc_score(y_val, val_pred_lgb)
        fold_auc_xgb = roc_auc_score(y_val, val_pred_xgb)

        print(
            f"Fold {fold + 1}/{n_folds} - LGBM ROC-AUC: {fold_auc_lgb:.5f} | XGB ROC-AUC: {fold_auc_xgb:.5f}"
        )

    # Optimize Ensemble Blend Weights on OOF
    best_w = 0.5
    best_auc = 0.0

    for w in np.linspace(0.0, 1.0, 101):
        blend_oof = w * oof_preds_lgb + (1.0 - w) * oof_preds_xgb
        score = roc_auc_score(y_true, blend_oof)
        if score > best_auc:
            best_auc = score
            best_w = w

    final_test_preds = best_w * test_preds_lgb + (1.0 - best_w) * test_preds_xgb
    print(
        f"Optimal Ensemble Blending Weight -> LightGBM: {best_w:.2f}, XGBoost: {1.0 - best_w:.2f}"
    )

    # Output Final Submission
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub = pd.DataFrame({"id": test_df["id"].astype(int), "target": final_test_preds})
    sub.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {best_auc}")


if __name__ == "__main__":
    train_and_evaluate()