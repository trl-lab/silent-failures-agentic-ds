import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
id_col = "id" if "id" in test_df.columns else None

raw_y = train_df[target_col].astype(str).str.lower().str.strip(" .")
y = raw_y.apply(lambda x: 1 if x in ["ad", "1", "1.0", "true"] else 0).values

feature_cols = [c for c in train_df.columns if c != target_col and c != id_col]


# 2. Data Cleaning Helper
def clean_feature_matrix(df, cols):
    data = df[cols].copy()
    for col in cols:
        data[col] = data[col].astype(str).str.strip()
        data[col] = data[col].replace(
            ["?", "None", "nan", "NaN", "", "unknown", "Unknown"], np.nan
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


X_train_raw = clean_feature_matrix(train_df, feature_cols)
X_test_raw = clean_feature_matrix(test_df, feature_cols)

cont_cols = feature_cols[:3]
bin_cols = feature_cols[3:]

n_train = len(train_df)
n_test = len(test_df)
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

model_names = ["lgb", "xgb", "cat", "et", "lr"]
n_models = len(model_names)

oof_preds = np.zeros((n_train, n_models))
test_preds_folds = np.zeros((n_test, n_models, n_splits))

# 3. Stratified K-Fold CV Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y)):
    tr_raw, va_raw = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
    te_raw = X_test_raw.copy()

    # Engineer Continuous Features
    tr_cont, va_cont, te_cont = (
        tr_raw[cont_cols].copy(),
        va_raw[cont_cols].copy(),
        te_raw[cont_cols].copy(),
    )

    # Missing indicators
    for c in cont_cols:
        tr_cont[f"{c}_isna"] = tr_cont[c].isna().astype(np.float32)
        va_cont[f"{c}_isna"] = va_cont[c].isna().astype(np.float32)
        te_cont[f"{c}_isna"] = te_cont[c].isna().astype(np.float32)

        medians = tr_cont[c].median()
        if pd.isna(medians):
            medians = 0.0
        tr_cont[c] = tr_cont[c].fillna(medians)
        va_cont[c] = va_cont[c].fillna(medians)
        te_cont[c] = te_cont[c].fillna(medians)

    def add_geo_features(df_c):
        c0, c1, c2 = df_c[cont_cols[0]], df_c[cont_cols[1]], df_c[cont_cols[2]]
        df_c["area"] = c0 * c1
        df_c["calc_aratio"] = c0 / (c1 + 1e-5)
        df_c["aratio_diff"] = np.abs(df_c["calc_aratio"] - c2)
        df_c["log_c0"] = np.log1p(np.maximum(0, c0))
        df_c["log_c1"] = np.log1p(np.maximum(0, c1))
        df_c["log_area"] = np.log1p(np.maximum(0, df_c["area"]))
        return df_c

    tr_cont = add_geo_features(tr_cont)
    va_cont = add_geo_features(va_cont)
    te_cont = add_geo_features(te_cont)

    # Engineer Binary Features & Expanded 50-component SVD
    tr_bin = tr_raw[bin_cols].fillna(0).values.astype(np.float32)
    va_bin = va_raw[bin_cols].fillna(0).values.astype(np.float32)
    te_bin = te_raw[bin_cols].fillna(0).values.astype(np.float32)

    tr_bin_stats = np.column_stack(
        [
            tr_bin.sum(axis=1),
            tr_bin.mean(axis=1),
            tr_bin.std(axis=1),
        ]
    )
    va_bin_stats = np.column_stack(
        [
            va_bin.sum(axis=1),
            va_bin.mean(axis=1),
            va_bin.std(axis=1),
        ]
    )
    te_bin_stats = np.column_stack(
        [
            te_bin.sum(axis=1),
            te_bin.mean(axis=1),
            te_bin.std(axis=1),
        ]
    )

    svd = TruncatedSVD(n_components=50, random_state=42 + fold)
    tr_svd = svd.fit_transform(tr_bin)
    va_svd = svd.transform(va_bin)
    te_svd = svd.transform(te_bin)

    # Assemble Feature Matrices
    X_tr = np.hstack([tr_cont.values, tr_bin, tr_bin_stats, tr_svd])
    X_va = np.hstack([va_cont.values, va_bin, va_bin_stats, va_svd])
    X_te = np.hstack([te_cont.values, te_bin, te_bin_stats, te_svd])

    # Drop zero variance columns based on train fold
    var = np.var(X_tr, axis=0)
    non_const = var > 1e-12
    X_tr, X_va, X_te = X_tr[:, non_const], X_va[:, non_const], X_te[:, non_const]

    y_tr, y_va = y[train_idx], y[val_idx]

    # Scaled inputs for linear model
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds[val_idx, 0] = model_lgb.predict_proba(X_va)[:, 1]
    test_preds_folds[:, 0, fold] = model_lgb.predict_proba(X_te)[:, 1]

    # Model 2: XGBoost
    model_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds[val_idx, 1] = model_xgb.predict_proba(X_va)[:, 1]
    test_preds_folds[:, 1, fold] = model_xgb.predict_proba(X_te)[:, 1]

    # Model 3: CatBoost
    model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_preds[val_idx, 2] = model_cat.predict_proba(X_va)[:, 1]
    test_preds_folds[:, 2, fold] = model_cat.predict_proba(X_te)[:, 1]

    # Model 4: ExtraTrees
    model_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=15, random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_tr, y_tr)
    oof_preds[val_idx, 3] = model_et.predict_proba(X_va)[:, 1]
    test_preds_folds[:, 3, fold] = model_et.predict_proba(X_te)[:, 1]

    # Model 5: Regularized Logistic Regression
    model_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42 + fold)
    model_lr.fit(X_tr_sc, y_tr)
    oof_preds[val_idx, 4] = model_lr.predict_proba(X_va_sc)[:, 1]
    test_preds_folds[:, 4, fold] = model_lr.predict_proba(X_te_sc)[:, 1]

# 4. Meta-Learner Stacking
meta_model = LogisticRegression(C=1.0, random_state=42)
meta_model.fit(oof_preds, y)
final_oof_probs = meta_model.predict_proba(oof_preds)[:, 1]

overall_auc = roc_auc_score(y, final_oof_probs)
print(f"Final CV Stacking ROC AUC: {overall_auc:.6f}")

# Fold-wise meta predictions for test set
fold_test_meta_preds = np.zeros((n_test, n_splits))
for fold in range(n_splits):
    fold_test_meta_preds[:, fold] = meta_model.predict_proba(
        test_preds_folds[:, :, fold]
    )[:, 1]
final_test_probs = fold_test_meta_preds.mean(axis=1)

# 5. Output Submission File
os.makedirs("./working", exist_ok=True)
sub_ids = test_df[id_col].values if id_col else np.arange(n_test)
submission = pd.DataFrame({"id": sub_ids, "class": final_test_probs})
submission.to_csv("./working/submission.csv", index=False)
