import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def engineer_features(df, base_cols):
    X = df[base_cols].copy()
    eps = 1e-6

    # Specific geometric / moment ratios tailored to silhouette classification
    ratios = pd.DataFrame(index=df.index)

    # Elongation and aspect ratios
    if "pr.axis_aspect_ratio" in X.columns and "max.length_aspect_ratio" in X.columns:
        ratios["aspect_ratio_diff"] = (
            X["pr.axis_aspect_ratio"] - X["max.length_aspect_ratio"]
        )
        ratios["aspect_ratio_ratio"] = X["pr.axis_aspect_ratio"] / (
            X["max.length_aspect_ratio"].abs() + eps
        )

    if "compactness" in X.columns and "circularity" in X.columns:
        ratios["compact_circ_ratio"] = X["compactness"] / (X["circularity"].abs() + eps)

    if "elongatedness" in X.columns and "compactness" in X.columns:
        ratios["elong_compact_ratio"] = X["elongatedness"] / (
            X["compactness"].abs() + eps
        )

    # Variance and gyration relationships
    if (
        "scaled_variance_along_major_axis" in X.columns
        and "scaled_variance_along_minor_axis" in X.columns
    ):
        ratios["var_major_minor_ratio"] = X["scaled_variance_along_major_axis"] / (
            X["scaled_variance_along_minor_axis"].abs() + eps
        )
        ratios["var_diff"] = (
            X["scaled_variance_along_major_axis"]
            - X["scaled_variance_along_minor_axis"]
        )

    # Rectangularity combinations
    if (
        "pr.axis_rectangularity" in X.columns
        and "max.length_rectangularity" in X.columns
    ):
        ratios["rect_diff"] = (
            X["pr.axis_rectangularity"] - X["max.length_rectangularity"]
        )

    # Moment / Kurtosis differences
    if (
        "kurtosis_about_major_axis" in X.columns
        and "kurtosis_about_minor_axis" in X.columns
    ):
        ratios["kurt_diff"] = (
            X["kurtosis_about_major_axis"] - X["kurtosis_about_minor_axis"]
        )

    if (
        "skewness_about_major_axis" in X.columns
        and "skewness_about_minor_axis" in X.columns
    ):
        ratios["skew_diff"] = (
            X["skewness_about_major_axis"] - X["skewness_about_minor_axis"]
        )

    return pd.concat([X, ratios], axis=1)


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "class"
    raw_feature_cols = [c for c in train_df.columns if c != target_col]

    # Impute missing values with medians if any exist
    for col in raw_feature_cols:
        if train_df[col].isnull().any() or test_df[col].isnull().any():
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            test_df[col] = test_df[col].fillna(med)

    X_train_full = engineer_features(train_df, raw_feature_cols)
    X_test_full = engineer_features(test_df, raw_feature_cols)

    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col])
    num_classes = len(le.classes_)
    n_samples = len(train_df)
    n_test = len(test_df)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_names = [
        "svc_rbf",
        "svc_poly",
        "extra_trees",
        "random_forest",
        "grad_boost",
        "mlp",
        "log_reg",
        "lightgbm",
    ]
    n_models = len(model_names)

    oof_preds = {name: np.zeros((n_samples, num_classes)) for name in model_names}
    test_preds = {name: np.zeros((n_test, num_classes)) for name in model_names}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y)):
        X_tr, y_tr = X_train_full.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train_full.iloc[val_idx], y[val_idx]
        X_te = X_test_full.copy()

        # 1. SVC (RBF)
        svc_rbf = make_pipeline(
            StandardScaler(),
            SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
        )
        svc_rbf.fit(X_tr, y_tr)
        oof_preds["svc_rbf"][val_idx] = svc_rbf.predict_proba(X_va)
        test_preds["svc_rbf"] += svc_rbf.predict_proba(X_te) / n_splits

        # 2. SVC (Poly)
        svc_poly = make_pipeline(
            StandardScaler(),
            SVC(C=2.0, kernel="poly", degree=3, probability=True, random_state=42),
        )
        svc_poly.fit(X_tr, y_tr)
        oof_preds["svc_poly"][val_idx] = svc_poly.predict_proba(X_va)
        test_preds["svc_poly"] += svc_poly.predict_proba(X_te) / n_splits

        # 3. ExtraTreesClassifier
        et = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=16,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et.fit(X_tr, y_tr)
        oof_preds["extra_trees"][val_idx] = et.predict_proba(X_va)
        test_preds["extra_trees"] += et.predict_proba(X_te) / n_splits

        # 4. RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        oof_preds["random_forest"][val_idx] = rf.predict_proba(X_va)
        test_preds["random_forest"] += rf.predict_proba(X_te) / n_splits

        # 5. GradientBoostingClassifier
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.85,
            random_state=42 + fold,
        )
        gb.fit(X_tr, y_tr)
        oof_preds["grad_boost"][val_idx] = gb.predict_proba(X_va)
        test_preds["grad_boost"] += gb.predict_proba(X_te) / n_splits

        # 6. MLPClassifier
        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.02,
                max_iter=500,
                random_state=42 + fold,
                early_stopping=True,
                n_iter_no_change=25,
            ),
        )
        mlp.fit(X_tr, y_tr)
        oof_preds["mlp"][val_idx] = mlp.predict_proba(X_va)
        test_preds["mlp"] += mlp.predict_proba(X_te) / n_splits

        # 7. LogisticRegression
        lr = make_pipeline(
            PowerTransformer(),
            LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        )
        lr.fit(X_tr, y_tr)
        oof_preds["log_reg"][val_idx] = lr.predict_proba(X_va)
        test_preds["log_reg"] += lr.predict_proba(X_te) / n_splits

        # 8. LightGBM
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
        lgb_params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.04,
            "num_leaves": 25,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42 + fold,
            "verbosity": -1,
            "n_jobs": -1,
        }
        lgb_model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            num_boost_round=600,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds["lightgbm"][val_idx] = lgb_model.predict(
            X_va, num_iteration=lgb_model.best_iteration
        )
        test_preds["lightgbm"] += (
            lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / n_splits
        )

    # Print individual model CV scores
    for name in model_names:
        acc = accuracy_score(y, np.argmax(oof_preds[name], axis=1))
        print(f"Model {name} CV Accuracy: {acc:.5f}")

    # Optimize ensemble weights to maximize CV accuracy (smooth surrogate via log loss or direct soft-blend)
    oof_matrix = np.array([oof_preds[m] for m in model_names])  # shape: (M, N, C)

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(0, w)
        w = w / (np.sum(w) + 1e-12)
        blended = np.tensordot(w, oof_matrix, axes=(0, 0))
        # Compute multi-class cross entropy
        eps_clip = 1e-15
        blended_clipped = np.clip(blended, eps_clip, 1 - eps_clip)
        one_hot = np.eye(num_classes)[y]
        return -np.mean(np.sum(one_hot * np.log(blended_clipped), axis=1))

    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    res = minimize(loss_func, init_weights, bounds=bounds, method="SLSQP")
    opt_weights = res.x / np.sum(res.x)

    final_oof = np.tensordot(opt_weights, oof_matrix, axes=(0, 0))
    cv_accuracy = accuracy_score(y, np.argmax(final_oof, axis=1))
    print(f"\nFinal Ensemble CV Accuracy: {cv_accuracy:.5f}")

    test_matrix = np.array([test_preds[m] for m in model_names])
    final_test_probs = np.tensordot(opt_weights, test_matrix, axes=(0, 0))
    test_pred_labels = le.inverse_transform(np.argmax(final_test_probs, axis=1))

    # Save predictions
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test_df)), "class": test_pred_labels}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved successfully to ./working/submission.csv")


if __name__ == "__main__":
    main()
