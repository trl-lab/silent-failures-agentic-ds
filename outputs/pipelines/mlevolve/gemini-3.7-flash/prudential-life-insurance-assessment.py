import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def load_data(input_dir="./input"):
    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def engineer_features(train_df, test_df):
    target_col = "Response"
    y_train = train_df[target_col].copy() if target_col in train_df.columns else None

    test_id = (
        test_df["Id"].copy()
        if "Id" in test_df.columns
        else pd.Series(np.arange(len(test_df)))
    )
    train_id = (
        train_df["Id"].copy()
        if "Id" in train_df.columns
        else pd.Series(np.arange(len(train_df)))
    )

    df_train = train_df.drop(columns=[target_col, "Id"], errors="ignore").copy()
    df_test = test_df.drop(columns=["Id"], errors="ignore").copy()

    # Frequency encoding for high-cardinality categoricals (fitted on df_train)
    for cat_col in ["Medical_History_2", "Employment_Info_2", "Product_Info_2", "InsuredInfo_1"]:
        if cat_col in df_train.columns:
            freq = df_train[cat_col].value_counts(dropna=False, normalize=True)
            df_train[f"{cat_col}_freq"] = df_train[cat_col].map(freq).fillna(0)
            df_test[f"{cat_col}_freq"] = df_test[cat_col].map(freq).fillna(0)

    # Strict Out-Of-Fold Target Encoding for high-cardinality categoricals
    if y_train is not None:
        te_cols = ["Medical_History_2", "Employment_Info_2", "Product_Info_2", "InsuredInfo_1"]
        global_mean = float(y_train.mean())
        smoothing = 10.0
        te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for cat_col in te_cols:
            if cat_col in df_train.columns:
                col_tr_str = df_train[cat_col].astype(str)
                col_te_str = df_test[cat_col].astype(str)
                oof_te = np.full(len(df_train), global_mean, dtype=np.float64)

                for fold_tr_idx, fold_va_idx in te_skf.split(df_train, y_train):
                    tr_s = col_tr_str.iloc[fold_tr_idx]
                    tr_y = y_train.iloc[fold_tr_idx]
                    stats = tr_y.groupby(tr_s).agg(["count", "mean"])
                    smooth_map = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                        stats["count"] + smoothing
                    )
                    oof_te[fold_va_idx] = (
                        col_tr_str.iloc[fold_va_idx].map(smooth_map).fillna(global_mean).values
                    )

                df_train[f"{cat_col}_te"] = oof_te

                full_stats = y_train.groupby(col_tr_str).agg(["count", "mean"])
                full_smooth = (
                    full_stats["count"] * full_stats["mean"] + smoothing * global_mean
                ) / (full_stats["count"] + smoothing)
                df_test[f"{cat_col}_te"] = (
                    col_te_str.map(full_smooth).fillna(global_mean).values
                )

    # Hierarchical group aggregation statistics (mean, std, min, max)
    for grp_col in ["Medical_History_2", "Product_Info_2"]:
        if grp_col in df_train.columns:
            for num_col in ["BMI", "Ins_Age", "Wt"]:
                if num_col in df_train.columns:
                    grp_stats = df_train.groupby(grp_col)[num_col].agg(["mean", "std", "min", "max"])
                    num_mean = float(df_train[num_col].mean())
                    num_min = float(df_train[num_col].min())
                    num_max = float(df_train[num_col].max())

                    df_train[f"{grp_col}_{num_col}_mean"] = df_train[grp_col].map(grp_stats["mean"]).fillna(num_mean)
                    df_train[f"{grp_col}_{num_col}_std"] = df_train[grp_col].map(grp_stats["std"]).fillna(0.0)
                    df_train[f"{grp_col}_{num_col}_min"] = df_train[grp_col].map(grp_stats["min"]).fillna(num_min)
                    df_train[f"{grp_col}_{num_col}_max"] = df_train[grp_col].map(grp_stats["max"]).fillna(num_max)

                    df_test[f"{grp_col}_{num_col}_mean"] = df_test[grp_col].map(grp_stats["mean"]).fillna(num_mean)
                    df_test[f"{grp_col}_{num_col}_std"] = df_test[grp_col].map(grp_stats["std"]).fillna(0.0)
                    df_test[f"{grp_col}_{num_col}_min"] = df_test[grp_col].map(grp_stats["min"]).fillna(num_min)
                    df_test[f"{grp_col}_{num_col}_max"] = df_test[grp_col].map(grp_stats["max"]).fillna(num_max)

    # 1. Product_Info_2 decomposition (Alphanumeric: character group + numeric level)
    if "Product_Info_2" in df_train.columns:
        for df in [df_train, df_test]:
            df["Product_Info_2_char"] = df["Product_Info_2"].astype(str).str[0]
            df["Product_Info_2_num"] = pd.to_numeric(
                df["Product_Info_2"].astype(str).str[1:], errors="coerce"
            ).fillna(-1)

        le_char = LabelEncoder()
        df_train["Product_Info_2_char_encoded"] = le_char.fit_transform(df_train["Product_Info_2_char"])
        char_map = {val: i for i, val in enumerate(le_char.classes_)}
        df_test["Product_Info_2_char_encoded"] = df_test["Product_Info_2_char"].map(char_map).fillna(-1)

        le_full = LabelEncoder()
        df_train["Product_Info_2_encoded"] = le_full.fit_transform(df_train["Product_Info_2"].astype(str))
        full_map = {val: i for i, val in enumerate(le_full.classes_)}
        df_test["Product_Info_2_encoded"] = df_test["Product_Info_2"].astype(str).map(full_map).fillna(-1)

        df_train = df_train.drop(columns=["Product_Info_2", "Product_Info_2_char"])
        df_test = df_test.drop(columns=["Product_Info_2", "Product_Info_2_char"])

    # 2. Medical keyword aggregations & SVD representations
    med_keyword_cols = [c for c in df_train.columns if c.startswith("Medical_Keyword_")]
    if med_keyword_cols:
        for df in [df_train, df_test]:
            df["Medical_Keyword_Sum"] = df[med_keyword_cols].sum(axis=1)
            df["Medical_Keyword_Mean"] = df[med_keyword_cols].mean(axis=1)
            df["Medical_Keyword_Std"] = df[med_keyword_cols].std(axis=1)
            df["Medical_Keyword_Active_Count"] = (df[med_keyword_cols] > 0).sum(axis=1)
        svd = TruncatedSVD(n_components=5, random_state=42)
        svd_tr = svd.fit_transform(df_train[med_keyword_cols].fillna(0))
        svd_te = svd.transform(df_test[med_keyword_cols].fillna(0))
        for i in range(5):
            df_train[f"Med_Keyword_SVD_{i+1}"] = svd_tr[:, i]
            df_test[f"Med_Keyword_SVD_{i+1}"] = svd_te[:, i]

    combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # 3. Informative Missingness Features
    combined["Total_Nulls"] = combined.isnull().sum(axis=1)

    med_hist_cols = [c for c in combined.columns if c.startswith("Medical_History_")]
    fam_hist_cols = [c for c in combined.columns if c.startswith("Family_Hist_")]
    emp_info_cols = [c for c in combined.columns if c.startswith("Employment_Info_")]

    if med_hist_cols:
        combined["Med_Hist_Nulls"] = combined[med_hist_cols].isnull().sum(axis=1)
        combined["Med_Hist_Mean"] = combined[med_hist_cols].mean(axis=1)
        combined["Med_Hist_Std"] = combined[med_hist_cols].std(axis=1)
        combined["Med_Hist_Max"] = combined[med_hist_cols].max(axis=1)
        combined["Med_Hist_Min"] = combined[med_hist_cols].min(axis=1)

    if fam_hist_cols:
        combined["Fam_Hist_Nulls"] = combined[fam_hist_cols].isnull().sum(axis=1)
        combined["Fam_Hist_Mean"] = combined[fam_hist_cols].mean(axis=1)
        combined["Fam_Hist_Std"] = combined[fam_hist_cols].std(axis=1)
        combined["Fam_Hist_Max"] = combined[fam_hist_cols].max(axis=1)

    if emp_info_cols:
        combined["Emp_Info_Nulls"] = combined[emp_info_cols].isnull().sum(axis=1)

    # 4. Actuarial and Physical Interaction Features
    if all(c in combined.columns for c in ["BMI", "Ins_Age"]):
        combined["BMI_x_Age"] = combined["BMI"] * combined["Ins_Age"]
        combined["Age_over_BMI"] = combined["Ins_Age"] / (combined["BMI"] + 1e-5)
        combined["BMI_sq"] = combined["BMI"] ** 2
        combined["Age_sq"] = combined["Ins_Age"] ** 2

    if all(c in combined.columns for c in ["Ht", "Wt"]):
        combined["Wt_over_Ht"] = combined["Wt"] / (combined["Ht"] + 1e-5)
        combined["Ht_over_Wt"] = combined["Ht"] / (combined["Wt"] + 1e-5)
        combined["Ht_x_Wt"] = combined["Ht"] * combined["Wt"]

    if all(c in combined.columns for c in ["BMI", "Wt"]):
        combined["BMI_x_Wt"] = combined["BMI"] * combined["Wt"]

    if all(c in combined.columns for c in ["Wt", "Ins_Age"]):
        combined["Wt_x_Age"] = combined["Wt"] * combined["Ins_Age"]

    if all(c in combined.columns for c in ["BMI", "Employment_Info_1"]):
        combined["BMI_x_Emp1"] = combined["BMI"] * combined["Employment_Info_1"].fillna(
            0
        )

    for col in combined.select_dtypes(include=["object", "category"]).columns:
        combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))

    n_train = len(df_train)
    X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

    return X_train, y_train, X_test, train_id, test_id


def create_folds(X_train, y_train, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(len(X_train), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(X_train, y_train)):
        folds[val_idx] = fold
    return folds


class OrdinalThresholdOptimizer:
    """Optimizes continuous regression cutoffs [c1, c2, ..., c7] to maximize Quadratic Weighted Kappa."""

    def __init__(self, initial_cutoffs=None):
        if initial_cutoffs is None:
            self.cutoffs = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype=np.float64)
        else:
            self.cutoffs = np.sort(np.array(initial_cutoffs, dtype=np.float64))

    def digitize(self, continuous_preds, cutoffs=None):
        if cutoffs is None:
            cutoffs = self.cutoffs
        ordered_cutoffs = np.sort(cutoffs)
        return np.clip(np.digitize(continuous_preds, ordered_cutoffs) + 1, 1, 8)

    def _loss(self, cutoffs, continuous_preds, y_true):
        discrete_preds = self.digitize(continuous_preds, cutoffs)
        return -cohen_kappa_score(y_true, discrete_preds, weights="quadratic")

    def _coordinate_descent(self, continuous_preds, y_true, init_cutoffs, max_iter=15, step_sizes=(0.1, 0.05, 0.02, 0.01, 0.005)):
        current_cutoffs = np.sort(init_cutoffs.copy())
        best_loss = self._loss(current_cutoffs, continuous_preds, y_true)

        for _ in range(max_iter):
            improved = False
            for i in range(len(current_cutoffs)):
                low = current_cutoffs[i - 1] + 1e-4 if i > 0 else 0.5
                high = current_cutoffs[i + 1] - 1e-4 if i < len(current_cutoffs) - 1 else 8.5
                val = current_cutoffs[i]

                for step in step_sizes:
                    for direction in [-1, 1]:
                        cand = np.clip(val + direction * step, low, high)
                        test_cutoffs = current_cutoffs.copy()
                        test_cutoffs[i] = cand
                        loss = self._loss(test_cutoffs, continuous_preds, y_true)
                        if loss < best_loss:
                            best_loss = loss
                            current_cutoffs[i] = cand
                            val = cand
                            improved = True
            if not improved:
                break
        return current_cutoffs, best_loss

    def fit(self, continuous_preds, y_true, n_trials=20, random_state=42):
        rng = np.random.RandomState(random_state)
        best_cutoffs = np.sort(self.cutoffs.copy())
        best_loss = self._loss(best_cutoffs, continuous_preds, y_true)

        # Base candidate initializations
        candidate_starts = [
            self.cutoffs.copy(),
            np.percentile(continuous_preds, [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]),
            np.percentile(continuous_preds, np.linspace(100 / 9, 800 / 9, 7)),
            np.array([1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2]),
        ]

        # Add jittered starts
        for _ in range(n_trials):
            jitter = rng.normal(0, 0.25, size=len(self.cutoffs))
            candidate_starts.append(np.sort(self.cutoffs + jitter))

        for init_c in candidate_starts:
            for method in ["Powell", "Nelder-Mead"]:
                res = minimize(
                    self._loss,
                    np.sort(init_c),
                    args=(continuous_preds, y_true),
                    method=method,
                    options={"maxiter": 500, "disp": False},
                )
                cand_cutoffs = np.sort(res.x)
                cand_loss = self._loss(cand_cutoffs, continuous_preds, y_true)
                if cand_loss < best_loss:
                    best_loss = cand_loss
                    best_cutoffs = cand_cutoffs

            refined_c, refined_loss = self._coordinate_descent(
                continuous_preds, y_true, init_c
            )
            if refined_loss < best_loss:
                best_loss = refined_loss
                best_cutoffs = refined_c

        best_cutoffs, _ = self._coordinate_descent(
            continuous_preds,
            y_true,
            best_cutoffs,
            max_iter=25,
            step_sizes=(0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001),
        )
        self.cutoffs = np.sort(best_cutoffs)
        return self


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data & Feature Engineering
    train_raw, test_raw = load_data("./input")
    X_train, y_train, X_test, train_ids, test_ids = engineer_features(
        train_raw, test_raw
    )
    fold_indices = create_folds(X_train, y_train, n_splits=5, random_state=42)

    features = list(X_train.columns)
    n_folds = 5

    oof_lgb = np.zeros(len(X_train), dtype=np.float64)
    oof_xgb = np.zeros(len(X_train), dtype=np.float64)
    oof_cat = np.zeros(len(X_train), dtype=np.float64)
    oof_mc_lgb = np.zeros(len(X_train), dtype=np.float64)

    test_lgb = np.zeros(len(X_test), dtype=np.float64)
    test_xgb = np.zeros(len(X_test), dtype=np.float64)
    test_cat = np.zeros(len(X_test), dtype=np.float64)
    test_mc_lgb = np.zeros(len(X_test), dtype=np.float64)

    # 2. 5-Fold Stratified Cross-Validation
    for fold in range(n_folds):
        train_mask = fold_indices != fold
        val_mask = fold_indices == fold

        X_tr, y_tr = X_train.loc[train_mask, features], y_train.iloc[train_mask].values
        X_va, y_va = X_train.loc[val_mask, features], y_train.iloc[val_mask].values
        X_te = X_test[features]

        # LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=7,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )

        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            booster="gbtree",
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )

        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # CatBoost Regressor
        cat_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=False,
        )

        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        # Multiclass LightGBM for Expected Value computation
        y_tr_mc = (y_tr - 1).astype(int)
        y_va_mc = (y_va - 1).astype(int)
        mc_weights = np.arange(1, 9, dtype=np.float64)

        mc_lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=8,
            metric="multi_logloss",
            boosting_type="gbdt",
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=45,
            max_depth=6,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )

        mc_lgb_model.fit(
            X_tr,
            y_tr_mc,
            eval_set=[(X_va, y_va_mc)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_probs = mc_lgb_model.predict_proba(X_va)
        test_probs = mc_lgb_model.predict_proba(X_te)

        oof_lgb[val_mask] = lgb_model.predict(X_va)
        oof_xgb[val_mask] = xgb_model.predict(X_va)
        oof_cat[val_mask] = cat_model.predict(X_va)
        oof_mc_lgb[val_mask] = val_probs @ mc_weights

        test_lgb += lgb_model.predict(X_te) / n_folds
        test_xgb += xgb_model.predict(X_te) / n_folds
        test_cat += cat_model.predict(X_te) / n_folds
        test_mc_lgb += (test_probs @ mc_weights) / n_folds

        val_blend = (
            oof_lgb[val_mask]
            + oof_xgb[val_mask]
            + oof_cat[val_mask]
            + oof_mc_lgb[val_mask]
        ) / 4.0
        fold_discrete_baseline = np.clip(np.round(val_blend), 1, 8).astype(int)
        fold_qwk = cohen_kappa_score(y_va, fold_discrete_baseline, weights="quadratic")
        print(f"Fold {fold + 1}/{n_folds} - Continuous Blend Val QWK: {fold_qwk:.5f}")

    # 3. Ensemble Blending Weights Optimization on OOF
    y_true_all = y_train.values
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_mc_lgb])
    test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_mc_lgb])

    def blend_mse_loss(weights):
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-12)
        pred = oof_matrix @ weights
        return np.mean((pred - y_true_all) ** 2)

    init_weights = np.array([0.25, 0.25, 0.25, 0.25])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    opt_weights_res = minimize(
        blend_mse_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_weights_res.x / np.sum(opt_weights_res.x)

    oof_blend = oof_matrix @ best_weights
    test_blend = test_matrix @ best_weights

    # 4. Multi-Start Post-Processing: Ordinal Threshold Calibration
    optimizer = OrdinalThresholdOptimizer(
        initial_cutoffs=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    )
    optimizer.fit(oof_blend, y_true_all, n_trials=20, random_state=42)

    # 5. Out-of-fold Evaluation
    oof_discrete = optimizer.digitize(oof_blend)
    final_score = cohen_kappa_score(y_true_all, oof_discrete, weights="quadratic")

    # 6. Test Inference & Submission Generation
    test_discrete = optimizer.digitize(test_blend)

    submission = pd.DataFrame(
        {"id": np.arange(len(X_test), dtype=int), "Response": test_discrete}
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()