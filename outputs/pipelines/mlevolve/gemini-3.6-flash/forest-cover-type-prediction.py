import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")


def process_and_engineer_features():
    """Engineers domain-specific spatial and topographic features with leak-free processing."""
    print("Starting data processing and feature engineering...")

    os.makedirs("./working", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "Cover_Type" in train_df.columns:
        train_y = train_df["Cover_Type"].values - 1
    else:
        raise ValueError("Target column 'Cover_Type' not found in train data.")

    train_id = (
        train_df["Id"] if "Id" in train_df.columns else pd.Series(range(len(train_df)))
    )
    test_id = (
        test_df["Id"] if "Id" in test_df.columns else pd.Series(range(len(test_df)))
    )

    df_train_feats = train_df.drop(
        columns=[col for col in ["Id", "Cover_Type"] if col in train_df.columns]
    ).copy()
    df_test_feats = test_df.drop(
        columns=[col for col in ["Id", "Cover_Type"] if col in test_df.columns]
    ).copy()

    n_train = len(df_train_feats)
    combined = pd.concat([df_train_feats, df_test_feats], axis=0, ignore_index=True)

    # 3D Euclidean Hydrology Distance
    combined["Euc_Distance_To_Hydrology"] = np.sqrt(
        combined["Horizontal_Distance_To_Hydrology"] ** 2
        + combined["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Hydrology Elevation Offset
    combined["Hydrology_Elevation_Offset"] = (
        combined["Elevation"] - combined["Vertical_Distance_To_Hydrology"]
    )

    # Cyclic Aspect Transformations
    aspect_rad = np.radians(combined["Aspect"])
    combined["Aspect_Sin"] = np.sin(aspect_rad)
    combined["Aspect_Cos"] = np.cos(aspect_rad)

    # Linear Combinations of Spatial Distances
    combined["Dist_Hydrology_Plus_Roadways"] = (
        combined["Horizontal_Distance_To_Hydrology"]
        + combined["Horizontal_Distance_To_Roadways"]
    )
    combined["Dist_Hydrology_Minus_Roadways"] = (
        combined["Horizontal_Distance_To_Hydrology"]
        - combined["Horizontal_Distance_To_Roadways"]
    )
    combined["Dist_Hydrology_Plus_Fire"] = (
        combined["Horizontal_Distance_To_Hydrology"]
        + combined["Horizontal_Distance_To_Fire_Points"]
    )
    combined["Dist_Hydrology_Minus_Fire"] = (
        combined["Horizontal_Distance_To_Hydrology"]
        - combined["Horizontal_Distance_To_Fire_Points"]
    )
    combined["Dist_Roadways_Plus_Fire"] = (
        combined["Horizontal_Distance_To_Roadways"]
        + combined["Horizontal_Distance_To_Fire_Points"]
    )
    combined["Dist_Roadways_Minus_Fire"] = (
        combined["Horizontal_Distance_To_Roadways"]
        - combined["Horizontal_Distance_To_Fire_Points"]
    )

    # Hillshade Aggregation Statistics
    hillshade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    combined["Hillshade_Mean"] = combined[hillshade_cols].mean(axis=1)
    combined["Hillshade_Min"] = combined[hillshade_cols].min(axis=1)
    combined["Hillshade_Max"] = combined[hillshade_cols].max(axis=1)
    combined["Hillshade_Range"] = combined["Hillshade_Max"] - combined["Hillshade_Min"]
    combined["Hillshade_Noon_Minus_9am"] = (
        combined["Hillshade_Noon"] - combined["Hillshade_9am"]
    )
    combined["Hillshade_3pm_Minus_Noon"] = (
        combined["Hillshade_3pm"] - combined["Hillshade_Noon"]
    )
    combined["Hillshade_3pm_Minus_9am"] = (
        combined["Hillshade_3pm"] - combined["Hillshade_9am"]
    )

    # Physical Terrain Interactions
    eps = 1e-5
    combined["Hydrology_Slope_Arctan"] = np.arctan2(
        combined["Vertical_Distance_To_Hydrology"],
        combined["Horizontal_Distance_To_Hydrology"] + eps,
    )
    combined["Total_Insolation"] = (
        combined["Hillshade_9am"]
        + combined["Hillshade_Noon"]
        + combined["Hillshade_3pm"]
    )

    # Distance Aggregations across Hydrology, Roadways, Fire Points
    dist_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    combined["Dist_Mean"] = combined[dist_cols].mean(axis=1)
    combined["Dist_Min"] = combined[dist_cols].min(axis=1)
    combined["Dist_Max"] = combined[dist_cols].max(axis=1)
    combined["Dist_Std"] = combined[dist_cols].std(axis=1)

    # Distance Ratios
    combined["Hydro_To_Road_Ratio"] = combined["Horizontal_Distance_To_Hydrology"] / (
        combined["Horizontal_Distance_To_Roadways"] + eps
    )
    combined["Hydro_To_Fire_Ratio"] = combined["Horizontal_Distance_To_Hydrology"] / (
        combined["Horizontal_Distance_To_Fire_Points"] + eps
    )
    combined["Road_To_Fire_Ratio"] = combined["Horizontal_Distance_To_Roadways"] / (
        combined["Horizontal_Distance_To_Fire_Points"] + eps
    )

    # Terrain & Slope Adjustments
    slope_rad = np.radians(combined["Slope"])
    combined["Slope_In_Radians"] = slope_rad
    combined["Elevation_Cos_Slope"] = combined["Elevation"] * np.cos(slope_rad)

    # Categoricals from One-Hot Matrices
    soil_cols = [c for c in combined.columns if c.startswith("Soil_Type")]
    wild_cols = [c for c in combined.columns if c.startswith("Wilderness_Area")]

    if soil_cols:
        combined["Soil_Type_Code"] = np.argmax(combined[soil_cols].values, axis=1) + 1
    if wild_cols:
        combined["Wilderness_Area_Code"] = (
            np.argmax(combined[wild_cols].values, axis=1) + 1
        )

    # Remove Zero-Variance Columns
    zero_var_cols = [c for c in combined.columns if combined[c].nunique() <= 1]
    if zero_var_cols:
        combined.drop(columns=zero_var_cols, inplace=True)

    train_processed = combined.iloc[:n_train].copy()
    test_processed = combined.iloc[n_train:].copy()

    # Leak-Free Group Aggregations (Fitted strictly on Train Data)
    group_vars = [
        "Elevation",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Slope",
    ]

    if "Soil_Type_Code" in train_processed.columns:
        soil_stats = train_processed.groupby("Soil_Type_Code")[group_vars].agg(
            ["mean", "std"]
        )
        soil_stats.columns = [f"Soil_{var}_{stat}" for var, stat in soil_stats.columns]
        soil_stats = soil_stats.reset_index()

        train_processed = train_processed.merge(
            soil_stats, on="Soil_Type_Code", how="left"
        )
        test_processed = test_processed.merge(
            soil_stats, on="Soil_Type_Code", how="left"
        )

        for var in group_vars:
            train_processed[f"Soil_{var}_std"] = train_processed[
                f"Soil_{var}_std"
            ].fillna(0.0)
            test_processed[f"Soil_{var}_std"] = test_processed[
                f"Soil_{var}_std"
            ].fillna(0.0)

            train_processed[f"{var}_Diff_Soil_Mean"] = (
                train_processed[var] - train_processed[f"Soil_{var}_mean"]
            )
            test_processed[f"{var}_Diff_Soil_Mean"] = (
                test_processed[var] - test_processed[f"Soil_{var}_mean"]
            )

    if "Wilderness_Area_Code" in train_processed.columns:
        wild_stats = train_processed.groupby("Wilderness_Area_Code")[group_vars].agg(
            ["mean", "std"]
        )
        wild_stats.columns = [f"Wild_{var}_{stat}" for var, stat in wild_stats.columns]
        wild_stats = wild_stats.reset_index()

        train_processed = train_processed.merge(
            wild_stats, on="Wilderness_Area_Code", how="left"
        )
        test_processed = test_processed.merge(
            wild_stats, on="Wilderness_Area_Code", how="left"
        )

        for var in group_vars:
            train_processed[f"Wild_{var}_std"] = train_processed[
                f"Wild_{var}_std"
            ].fillna(0.0)
            test_processed[f"Wild_{var}_std"] = test_processed[
                f"Wild_{var}_std"
            ].fillna(0.0)

            train_processed[f"{var}_Diff_Wild_Mean"] = (
                train_processed[var] - train_processed[f"Wild_{var}_mean"]
            )
            test_processed[f"{var}_Diff_Wild_Mean"] = (
                test_processed[var] - test_processed[f"Wild_{var}_mean"]
            )

    train_processed["Id"] = train_id.values
    train_processed["Cover_Type"] = train_y
    test_processed["Id"] = test_id.values

    # Stratified 5-Fold Cross Validation Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_processed["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(train_processed, train_processed["Cover_Type"])
    ):
        train_processed.loc[val_idx, "fold"] = fold_idx

    train_processed.to_parquet("./working/train_processed.parquet", index=False)
    test_processed.to_parquet("./working/test_processed.parquet", index=False)

    print("Feature Engineering completed successfully.")


def train_and_evaluate_ensemble():
    """Trains multi-model ensemble across 5 folds, optimizes blend weights, and generates test submission."""
    print("Starting ensemble training and evaluation pipeline...")

    train_df = pd.read_parquet("./working/train_processed.parquet")
    test_df = pd.read_parquet("./working/test_processed.parquet")

    ignore_cols = ["Id", "Cover_Type", "fold"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    num_classes = 7
    n_folds = train_df["fold"].nunique()
    n_train = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros((n_train, num_classes))
    oof_xgb = np.zeros((n_train, num_classes))
    oof_cb = np.zeros((n_train, num_classes))
    oof_et = np.zeros((n_train, num_classes))
    oof_rf = np.zeros((n_train, num_classes))

    test_lgb = np.zeros((n_test, num_classes))
    test_xgb = np.zeros((n_test, num_classes))
    test_cb = np.zeros((n_test, num_classes))
    test_et = np.zeros((n_test, num_classes))
    test_rf = np.zeros((n_test, num_classes))

    for fold in range(n_folds):
        train_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        train_fold = train_df[train_idx].copy()
        val_fold = train_df[val_idx].copy()

        X_tr = train_fold[feature_cols].values
        y_tr = train_fold["Cover_Type"].values
        X_va = val_fold[feature_cols].values
        y_va = val_fold["Cover_Type"].values
        X_te = test_df[feature_cols].values

        # Sanitize numeric inputs
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        val_indices = train_df.index[val_idx].values

        # 1. LightGBM
        lgb_params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": 10,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
        lgb_tr = lgb.Dataset(X_tr, label=y_tr)
        lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr)

        lgb_model = lgb.train(
            lgb_params,
            lgb_tr,
            num_boost_round=1200,
            valid_sets=[lgb_va],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        oof_lgb[val_indices] = lgb_model.predict(
            X_va, num_iteration=lgb_model.best_iteration
        )
        test_lgb += (
            lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / n_folds
        )

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=1200,
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_xgb[val_indices] = xgb_model.predict_proba(X_va)
        test_xgb += xgb_model.predict_proba(X_te) / n_folds

        # 3. CatBoost
        cb_model = cb.CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=1200,
            learning_rate=0.05,
            depth=7,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
        )
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=0,
        )

        oof_cb[val_indices] = cb_model.predict_proba(X_va)
        test_cb += cb_model.predict_proba(X_te) / n_folds

        # 4. Extra Trees Classifier
        et_model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        oof_et[val_indices] = et_model.predict_proba(X_va)
        test_et += et_model.predict_proba(X_te) / n_folds

        # 5. Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf_model.fit(X_tr, y_tr)
        oof_rf[val_indices] = rf_model.predict_proba(X_va)
        test_rf += rf_model.predict_proba(X_te) / n_folds

        fold_lgb_acc = accuracy_score(y_va, np.argmax(oof_lgb[val_indices], axis=1))
        fold_xgb_acc = accuracy_score(y_va, np.argmax(oof_xgb[val_indices], axis=1))
        fold_cb_acc = accuracy_score(y_va, np.argmax(oof_cb[val_indices], axis=1))
        fold_et_acc = accuracy_score(y_va, np.argmax(oof_et[val_indices], axis=1))
        fold_rf_acc = accuracy_score(y_va, np.argmax(oof_rf[val_indices], axis=1))

        print(
            f"Fold {fold+1}/{n_folds} Accuracies -> "
            f"LGB: {fold_lgb_acc:.4f} | XGB: {fold_xgb_acc:.4f} | CB: {fold_cb_acc:.4f} | ET: {fold_et_acc:.4f} | RF: {fold_rf_acc:.4f}"
        )

    # Optimize Ensemble Blend Weights
    y_true_all = train_df["Cover_Type"].values

    def loss_func(weights):
        w = np.maximum(0, np.array(weights))
        if np.sum(w) == 0:
            return 0.0
        w = w / np.sum(w)
        blend_probs = (
            w[0] * oof_lgb
            + w[1] * oof_xgb
            + w[2] * oof_cb
            + w[3] * oof_et
            + w[4] * oof_rf
        )
        preds = np.argmax(blend_probs, axis=1)
        return -accuracy_score(y_true_all, preds)

    init_weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    bounds = [(0, 1)] * 5
    res = minimize(loss_func, init_weights, bounds=bounds, method="Nelder-Mead")

    opt_w = np.maximum(0, np.array(res.x))
    opt_w = opt_w / np.sum(opt_w)

    final_oof_probs = (
        opt_w[0] * oof_lgb
        + opt_w[1] * oof_xgb
        + opt_w[2] * oof_cb
        + opt_w[3] * oof_et
        + opt_w[4] * oof_rf
    )
    final_oof_preds = np.argmax(final_oof_probs, axis=1)
    final_acc = accuracy_score(y_true_all, final_oof_preds)

    # Compute Final Test Predictions & Save Submission File
    final_test_probs = (
        opt_w[0] * test_lgb
        + opt_w[1] * test_xgb
        + opt_w[2] * test_cb
        + opt_w[3] * test_et
        + opt_w[4] * test_rf
    )
    test_class_preds = np.argmax(final_test_probs, axis=1) + 1

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_df)), "Cover_Type": test_class_preds}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")

    print(f"Final Validation Score: {final_acc}")


if __name__ == "__main__":
    process_and_engineer_features()
    train_and_evaluate_ensemble()
