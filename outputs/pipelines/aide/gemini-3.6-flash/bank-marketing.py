import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")


def load_data():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")
    return train_df, test_df


def process_target(train_df):
    target_col = "Class"
    y_raw = train_df[target_col]

    if pd.api.types.is_numeric_dtype(y_raw):
        unique_vals = set(y_raw.unique())
        if unique_vals == {1, 2}:
            return (y_raw == 2).astype(int).values
        elif unique_vals == {0, 1}:
            return y_raw.values.astype(int)

    y_str = y_raw.astype(str).str.lower().str.strip()
    unique_strs = set(y_str.unique())
    if "2" in unique_strs or "yes" in unique_strs:
        target_map = {"1": 0, "2": 1, "no": 0, "yes": 1}
        return y_str.map(target_map).values.astype(int)
    else:
        sorted_vals = sorted(list(unique_strs))
        target_map = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return y_str.map(target_map).values.astype(int)


def create_features(df):
    df = df.copy()

    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    if "month" in df.columns:
        df["month_num"] = (
            df["month"].astype(str).str.lower().map(month_map).fillna(0).astype(int)
        )
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)

    # Domain features
    df["balance_per_age"] = df["balance"] / (df["age"] + 1e-5)
    df["signed_log_balance"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["is_negative_balance"] = (df["balance"] < 0).astype(int)

    df["is_previous_contacted"] = (df["pdays"] != -1).astype(int)
    df["pdays_clean"] = np.where(df["pdays"] == -1, np.nan, df["pdays"])
    df["pdays_log"] = np.log1p(np.maximum(0, df["pdays_clean"].fillna(0)))

    df["total_contacts"] = df["campaign"] + df["previous"]
    df["duration_min"] = df["duration"] / 60.0
    df["duration_log"] = np.log1p(np.maximum(0, df["duration"]))
    df["duration_per_campaign"] = df["duration"] / (df["campaign"] + 1e-5)
    df["previous_ratio"] = df["previous"] / (df["campaign"] + df["previous"] + 1.0)

    return df


def add_fold_features(X_tr, y_tr, X_va, X_te, cat_cols, te_cols, m=10.0):
    X_tr_out = X_tr.copy()
    X_va_out = X_va.copy()
    X_te_out = X_te.copy()

    global_mean = float(y_tr.mean())

    # Frequency encoding on train fold
    for c in cat_cols:
        freq_map = X_tr[c].value_counts(normalize=True).to_dict()
        X_tr_out[f"{c}_freq"] = X_tr[c].map(freq_map).fillna(0.0).astype(float)
        X_va_out[f"{c}_freq"] = X_va[c].map(freq_map).fillna(0.0).astype(float)
        X_te_out[f"{c}_freq"] = X_te[c].map(freq_map).fillna(0.0).astype(float)

    # Group statistics
    for group_col in ["job", "month", "education"]:
        if group_col in X_tr.columns:
            for num_col in ["duration", "balance"]:
                grp_mean = (
                    X_tr.groupby(group_col, observed=False)[num_col].mean().to_dict()
                )
                X_tr_out[f"{num_col}_diff_{group_col}"] = X_tr[num_col] - X_tr[
                    group_col
                ].map(grp_mean).fillna(0)
                X_va_out[f"{num_col}_diff_{group_col}"] = X_va[num_col] - X_va[
                    group_col
                ].map(grp_mean).fillna(0)
                X_te_out[f"{num_col}_diff_{group_col}"] = X_te[num_col] - X_te[
                    group_col
                ].map(grp_mean).fillna(0)

    # Out-of-fold target encoding for train, global map for va/te
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in te_cols:
        te_name = f"{col}_te"
        X_tr_out[te_name] = np.nan

        for in_tr_idx, in_va_idx in skf_inner.split(X_tr, y_tr):
            in_X_tr, in_y_tr = X_tr.iloc[in_tr_idx], y_tr[in_tr_idx]
            in_X_va = X_tr.iloc[in_va_idx]

            stats = (
                pd.DataFrame({"cat": in_X_tr[col], "target": in_y_tr})
                .groupby("cat", observed=False)["target"]
                .agg(["count", "mean"])
            )
            smooth = (stats["count"] * stats["mean"] + m * global_mean) / (
                stats["count"] + m
            )
            te_map = smooth.to_dict()

            X_tr_out.iloc[in_va_idx, X_tr_out.columns.get_loc(te_name)] = (
                in_X_va[col].map(te_map).fillna(global_mean)
            )

        # Full train map for validation and test sets
        full_stats = (
            pd.DataFrame({"cat": X_tr[col], "target": y_tr})
            .groupby("cat", observed=False)["target"]
            .agg(["count", "mean"])
        )
        full_smooth = (full_stats["count"] * full_stats["mean"] + m * global_mean) / (
            full_stats["count"] + m
        )
        full_map = full_smooth.to_dict()

        X_tr_out[te_name] = X_tr_out[te_name].fillna(global_mean).astype(float)
        X_va_out[te_name] = X_va[col].map(full_map).fillna(global_mean).astype(float)
        X_te_out[te_name] = X_te[col].map(full_map).fillna(global_mean).astype(float)

    return X_tr_out, X_va_out, X_te_out


def main():
    train_df, test_df = load_data()
    y = process_target(train_df)

    train_df = train_df.drop(columns=["Class"], errors="ignore")
    n_train = len(train_df)

    combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    combined = create_features(combined)

    cat_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    cat_cols = [c for c in cat_cols if c in combined.columns]

    for c in cat_cols:
        combined[c] = combined[c].astype("category")

    X_all = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test))

    te_cols = ["job", "month", "poutcome", "education"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y)):
        X_tr = X_all.iloc[train_idx].copy().reset_index(drop=True)
        y_tr = y[train_idx]
        X_va = X_all.iloc[val_idx].copy().reset_index(drop=True)
        y_va = y[val_idx]
        X_te_fold = X_test.copy().reset_index(drop=True)

        X_tr, X_va, X_te_fold = add_fold_features(
            X_tr, y_tr, X_va, X_te_fold, cat_cols, te_cols, m=10.0
        )

        # LightGBM Model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_preds_lgb = lgb_model.predict_proba(X_va)[:, 1]
        test_preds_lgb = lgb_model.predict_proba(X_te_fold)[:, 1]

        # CatBoost Model
        cb_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.02,
            depth=6,
            eval_metric="AUC",
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        val_preds_cb = cb_model.predict_proba(X_va)[:, 1]
        test_preds_cb = cb_model.predict_proba(X_te_fold)[:, 1]

        # XGBoost Model
        xgb_model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="auc",
            enable_categorical=True,
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_preds_xgb = xgb_model.predict_proba(X_va)[:, 1]
        test_preds_xgb = xgb_model.predict_proba(X_te_fold)[:, 1]

        # Ensemble predictions
        fold_val_preds = (
            0.40 * val_preds_lgb + 0.35 * val_preds_cb + 0.25 * val_preds_xgb
        )
        fold_test_preds = (
            0.40 * test_preds_lgb + 0.35 * test_preds_cb + 0.25 * test_preds_xgb
        )

        oof_preds[val_idx] = fold_val_preds
        test_preds += fold_test_preds / n_splits

    cv_auc = roc_auc_score(y, oof_preds)
    print(f"5-Fold Cross-Validation ROC AUC: {cv_auc:.5f}")

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Class": test_preds})
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
