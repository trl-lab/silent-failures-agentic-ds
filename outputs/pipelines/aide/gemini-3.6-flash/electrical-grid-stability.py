import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Set seed for reproducibility
SEED = 42


def load_data():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")
    return train_df, test_df


def engineer_features(df):
    data = df.copy()

    tau_cols = ["tau1", "tau2", "tau3", "tau4"]
    p_cols = ["p1", "p2", "p3", "p4"]
    g_cols = ["g1", "g2", "g3", "g4"]

    # Basic sum and summary features
    data["tau_sum"] = data[tau_cols].sum(axis=1)
    data["tau_mean"] = data[tau_cols].mean(axis=1)
    data["tau_std"] = data[tau_cols].std(axis=1)

    data["g_sum"] = data[g_cols].sum(axis=1)
    data["g_mean"] = data[g_cols].mean(axis=1)
    data["g_std"] = data[g_cols].std(axis=1)

    data["p_balance"] = data[p_cols].sum(axis=1)
    data["p_abs_sum"] = data[p_cols].abs().sum(axis=1)

    # Node-level interaction features
    for i in range(1, 5):
        t, p, g = f"tau{i}", f"p{i}", f"g{i}"
        data[f"tg_{i}"] = data[t] * data[g]
        data[f"t_over_g_{i}"] = data[t] / (data[g] + 1e-8)
        data[f"g_over_t_{i}"] = data[g] / (data[t] + 1e-8)
        data[f"pt_{i}"] = data[p] * data[t]
        data[f"pg_{i}"] = data[p] * data[g]
        data[f"p_over_tg_{i}"] = data[p] / (data[t] * data[g] + 1e-8)
        data[f"kinetic_{i}"] = (data[p] ** 2) / (data[t] * data[g] + 1e-8)

    tg_cols = [f"tg_{i}" for i in range(1, 5)]
    data["tg_sum"] = data[tg_cols].sum(axis=1)
    data["tg_mean"] = data[tg_cols].mean(axis=1)
    data["tg_std"] = data[tg_cols].std(axis=1)

    # Consumer vs Producer summary features
    tau_cons = ["tau2", "tau3", "tau4"]
    g_cons = ["g2", "g3", "g4"]
    p_cons = ["p2", "p3", "p4"]
    tg_cons = ["tg_2", "tg_3", "tg_4"]

    data["tau_cons_mean"] = data[tau_cons].mean(axis=1)
    data["g_cons_mean"] = data[g_cons].mean(axis=1)
    data["p_cons_mean"] = data[p_cons].mean(axis=1)
    data["tg_cons_mean"] = data[tg_cons].mean(axis=1)

    # Ratios between Producer (node 1) and Consumer averages
    data["tau_prod_cons_ratio"] = data["tau1"] / (data["tau_cons_mean"] + 1e-8)
    data["g_prod_cons_ratio"] = data["g1"] / (data["g_cons_mean"] + 1e-8)
    data["tg_prod_cons_ratio"] = data["tg_1"] / (data["tg_cons_mean"] + 1e-8)

    # Distribution statistics across nodes
    for cols, prefix in [
        (tau_cols, "tau"),
        (p_cols, "p"),
        (g_cols, "g"),
        (tg_cols, "tg"),
    ]:
        data[f"{prefix}_min"] = data[cols].min(axis=1)
        data[f"{prefix}_max"] = data[cols].max(axis=1)

    return data


def encode_targets(train_df):
    target_col = "stabf"
    leakage_col = "stab"

    y_raw = train_df[target_col]
    if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
        y_binary = (y_raw.astype(str).str.lower() == "unstable").astype(int).values
    else:
        y_binary = y_raw.astype(int).values

    y_stab = train_df[leakage_col].values if leakage_col in train_df.columns else None
    return y_binary, y_stab


def main():
    train_df, test_df = load_data()

    y_binary, y_stab = encode_targets(train_df)

    cols_to_drop = [col for col in ["stabf", "stab", "id"] if col in train_df.columns]
    X_train_raw = train_df.drop(columns=cols_to_drop)

    cols_to_drop_test = [
        col for col in ["stabf", "stab", "id"] if col in test_df.columns
    ]
    X_test_raw = test_df.drop(columns=cols_to_drop_test)

    X_train = engineer_features(X_train_raw)
    X_test = engineer_features(X_test_raw)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Out-of-fold and test prediction arrays
    oof_lgb_reg = np.zeros(len(X_train))
    oof_cat_reg = np.zeros(len(X_train))
    oof_xgb_reg = np.zeros(len(X_train))

    oof_lgb_clf = np.zeros(len(X_train))
    oof_cat_clf = np.zeros(len(X_train))
    oof_xgb_clf = np.zeros(len(X_train))

    test_preds_lgb_reg = np.zeros(len(X_test))
    test_preds_cat_reg = np.zeros(len(X_test))
    test_preds_xgb_reg = np.zeros(len(X_test))

    test_preds_lgb_clf = np.zeros(len(X_test))
    test_preds_cat_clf = np.zeros(len(X_test))
    test_preds_xgb_clf = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_binary)):
        X_tr, y_tr_bin = X_train.iloc[train_idx], y_binary[train_idx]
        X_va, y_va_bin = X_train.iloc[val_idx], y_binary[val_idx]
        y_tr_stab = y_stab[train_idx]
        y_va_stab = y_stab[val_idx]

        # 1. LightGBM Regressor on continuous stability target
        model_lgb_reg = lgb.LGBMRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED + fold,
            verbosity=-1,
        )
        model_lgb_reg.fit(
            X_tr,
            y_tr_stab,
            eval_set=[(X_va, y_va_stab)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb_reg[val_idx] = model_lgb_reg.predict(X_va)
        test_preds_lgb_reg += model_lgb_reg.predict(X_test) / skf.n_splits

        # 2. CatBoost Regressor on continuous stability target
        model_cat_reg = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.03,
            depth=6,
            eval_metric="RMSE",
            random_seed=SEED + fold,
            verbose=0,
        )
        model_cat_reg.fit(
            X_tr,
            y_tr_stab,
            eval_set=(X_va, y_va_stab),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        oof_cat_reg[val_idx] = model_cat_reg.predict(X_va)
        test_preds_cat_reg += model_cat_reg.predict(X_test) / skf.n_splits

        # 3. XGBoost Regressor on continuous stability target
        model_xgb_reg = xgb.XGBRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
            random_state=SEED + fold,
            early_stopping_rounds=100,
            tree_method="hist",
        )
        model_xgb_reg.fit(X_tr, y_tr_stab, eval_set=[(X_va, y_va_stab)], verbose=False)
        oof_xgb_reg[val_idx] = model_xgb_reg.predict(X_va)
        test_preds_xgb_reg += model_xgb_reg.predict(X_test) / skf.n_splits

        # 4. LightGBM Classifier on binary target
        model_lgb_clf = lgb.LGBMClassifier(
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED + fold,
            verbosity=-1,
        )
        model_lgb_clf.fit(
            X_tr,
            y_tr_bin,
            eval_set=[(X_va, y_va_bin)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb_clf[val_idx] = model_lgb_clf.predict_proba(X_va)[:, 1]
        test_preds_lgb_clf += model_lgb_clf.predict_proba(X_test)[:, 1] / skf.n_splits

        # 5. CatBoost Classifier on binary target
        model_cat_clf = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            random_seed=SEED + fold,
            verbose=0,
        )
        model_cat_clf.fit(
            X_tr,
            y_tr_bin,
            eval_set=(X_va, y_va_bin),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        oof_cat_clf[val_idx] = model_cat_clf.predict_proba(X_va)[:, 1]
        test_preds_cat_clf += model_cat_clf.predict_proba(X_test)[:, 1] / skf.n_splits

        # 6. XGBoost Classifier on binary target
        model_xgb_clf = xgb.XGBClassifier(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=SEED + fold,
            early_stopping_rounds=100,
            tree_method="hist",
        )
        model_xgb_clf.fit(X_tr, y_tr_bin, eval_set=[(X_va, y_va_bin)], verbose=False)
        oof_xgb_clf[val_idx] = model_xgb_clf.predict_proba(X_va)[:, 1]
        test_preds_xgb_clf += model_xgb_clf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Calibrate continuous stability predictions into probabilities
    reg_oof_combined = (oof_lgb_reg + oof_cat_reg + oof_xgb_reg) / 3.0
    reg_test_combined = (
        test_preds_lgb_reg + test_preds_cat_reg + test_preds_xgb_reg
    ) / 3.0

    calibrator = LogisticRegression()
    calibrator.fit(reg_oof_combined.reshape(-1, 1), y_binary)

    oof_reg_prob = calibrator.predict_proba(reg_oof_combined.reshape(-1, 1))[:, 1]
    test_reg_prob = calibrator.predict_proba(reg_test_combined.reshape(-1, 1))[:, 1]

    clf_oof_combined = (oof_lgb_clf + oof_cat_clf + oof_xgb_clf) / 3.0
    clf_test_combined = (
        test_preds_lgb_clf + test_preds_cat_clf + test_preds_xgb_clf
    ) / 3.0

    # Final ensemble blending continuous regression and classifier probabilities
    oof_final = 0.65 * oof_reg_prob + 0.35 * clf_oof_combined
    test_final = 0.65 * test_reg_prob + 0.35 * clf_test_combined

    auc_reg = roc_auc_score(y_binary, oof_reg_prob)
    auc_clf = roc_auc_score(y_binary, clf_oof_combined)
    cv_auc = roc_auc_score(y_binary, oof_final)

    print(f"OOF ROC AUC - Continuous Regression Models: {auc_reg:.6f}")
    print(f"OOF ROC AUC - Binary Classification Models: {auc_clf:.6f}")
    print(f"Overall Ensemble OOF ROC AUC:               {cv_auc:.6f}")

    # Save test set predictions
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_final)), "stabf": test_final})
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved submission to ./working/submission.csv")


if __name__ == "__main__":
    main()
