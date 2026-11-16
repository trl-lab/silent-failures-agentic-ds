import json
import os
from typing import Any, Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_morphological_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-grounded geometric, shape factor, scale-invariant,

    and interaction features from raw bean physical dimensions.
    """
    df = df.copy()
    eps = 1e-8

    # 1. Non-convex area discrepancy & geometric fill ratios
    df["Area_diff_ConvexArea"] = df["ConvexArea"] - df["Area"]
    df["Area_ratio_ConvexArea"] = df["Area"] / (df["ConvexArea"] + eps)

    # 2. Perimeter interaction ratios
    df["Perimeter_to_Area"] = df["Perimeter"] / (df["Area"] + eps)
    df["Perimeter_to_MajorAxis"] = df["Perimeter"] / (df["MajorAxisLength"] + eps)
    df["Perimeter_to_MinorAxis"] = df["Perimeter"] / (df["MinorAxisLength"] + eps)
    df["Perimeter_to_EquivDiameter"] = df["Perimeter"] / (df["EquivDiameter"] + eps)
    df["Perimeter_sq_to_Area"] = (df["Perimeter"] ** 2) / (df["Area"] + eps)

    # 3. Axis dimensions & Ellipse approximations
    df["Major_to_Minor_diff"] = df["MajorAxisLength"] - df["MinorAxisLength"]
    df["Major_to_Minor_sum"] = df["MajorAxisLength"] + df["MinorAxisLength"]
    df["Major_Minor_prod"] = df["MajorAxisLength"] * df["MinorAxisLength"]
    df["Ellipse_Area_Approx"] = (
        np.pi * (df["MajorAxisLength"] / 2.0) * (df["MinorAxisLength"] / 2.0)
    )
    df["Area_to_EllipseArea_Ratio"] = df["Area"] / (df["Ellipse_Area_Approx"] + eps)
    df["ConvexArea_to_EllipseArea_Ratio"] = df["ConvexArea"] / (
        df["Ellipse_Area_Approx"] + eps
    )

    # 4. Bounding box estimates & ratios
    df["BoundingBox_Area_Est"] = df["Area"] / (df["Extent"] + eps)
    df["BoundingBox_to_Ellipse_Ratio"] = df["BoundingBox_Area_Est"] / (
        df["Ellipse_Area_Approx"] + eps
    )

    # 5. Log-scale transformations for scale features
    for col in [
        "Area",
        "ConvexArea",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "EquivDiameter",
    ]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # 6. Shape factor non-linear combinations & ratios
    df["ShapeFactor_Ratio_1_2"] = df["ShapeFactor1"] / (df["ShapeFactor2"] + eps)
    df["ShapeFactor_Ratio_3_4"] = df["ShapeFactor3"] / (df["ShapeFactor4"] + eps)
    df["ShapeFactor_Prod_1_2"] = df["ShapeFactor1"] * df["ShapeFactor2"]
    df["ShapeFactor_Prod_3_4"] = df["ShapeFactor3"] * df["ShapeFactor4"]

    # 7. Form-factor interactions
    df["Compactness_sq"] = df["Compactness"] ** 2
    df["Eccentricity_sq"] = df["Eccentricity"] ** 2
    df["Roundness_Solidity_prod"] = df["Roundness"] * df["Solidity"]
    df["Roundness_to_Solidity"] = df["Roundness"] / (df["Solidity"] + eps)
    df["Extent_Solidity_prod"] = df["Extent"] * df["Solidity"]
    df["Aspect_Eccentricity_prod"] = df["AspectRatio"] * df["Eccentricity"]

    # 8. High-order morphological invariants & theoretical discrepancies
    df["Rectangularity"] = df["Area"] / (df["Major_Minor_prod"] + eps)
    df["Convexity"] = df["ConvexArea"] / (df["Area"] + eps)
    df["Circularity"] = 4.0 * np.pi * df["Area"] / ((df["Perimeter"] ** 2) + eps)
    df["Circularity_to_Roundness"] = df["Circularity"] / (df["Roundness"] + eps)

    # Theoretical shape factor deviations
    df["SF1_diff"] = df["ShapeFactor1"] - (df["MajorAxisLength"] / (df["Area"] + eps))
    df["SF2_diff"] = df["ShapeFactor2"] - (df["MinorAxisLength"] / (df["Area"] + eps))
    df["SF3_diff"] = df["ShapeFactor3"] - (
        4.0 * df["Area"] / (np.pi * (df["MajorAxisLength"] ** 2) + eps)
    )
    df["SF4_diff"] = df["ShapeFactor4"] - (
        4.0 * df["Area"] / (np.pi * df["Major_Minor_prod"] + eps)
    )

    # Ramanujan perimeter approximation and ratio
    h = ((df["Major_to_Minor_diff"]) / (df["Major_to_Minor_sum"] + eps)) ** 2
    df["Perimeter_Ellipse_Ramanujan"] = (
        np.pi
        * (df["Major_to_Minor_sum"] / 2.0)
        * (1.0 + (3.0 * h) / (10.0 + np.sqrt(np.clip(4.0 - 3.0 * h, 0, None)) + eps))
    )
    df["Perimeter_to_Ellipse_Perimeter"] = df["Perimeter"] / (
        df["Perimeter_Ellipse_Ramanujan"] + eps
    )
    df["Polar_Eccentricity_Est"] = np.sqrt(
        np.clip(
            1.0 - (df["MinorAxisLength"] / (df["MajorAxisLength"] + eps)) ** 2,
            0.0,
            1.0,
        )
    )
    df["Polar_Eccentricity_Ratio"] = df["Polar_Eccentricity_Est"] / (
        df["Eccentricity"] + eps
    )
    df["Moment_Ratio_1"] = (
        df["MajorAxisLength"] ** 2 + df["MinorAxisLength"] ** 2
    ) / (df["Area"] + eps)
    df["Equiv_to_MajorMinor_GeomMean"] = df["EquivDiameter"] / (
        np.sqrt(df["Major_Minor_prod"]) + eps
    )

    # Domain-grounded geometric invariants
    df["Normalized_Elongation_Index"] = (
        df["MajorAxisLength"] - df["MinorAxisLength"]
    ) / (df["MajorAxisLength"] + df["MinorAxisLength"] + eps)
    df["Stokes_Equivalent_Radius"] = np.sqrt(df["Area"] / (np.pi + eps))
    df["Boundary_Roughness_Index"] = df["Perimeter"] / (
        2.0 * np.sqrt(np.pi * df["Area"] + eps)
    )

    # Clean non-finite entries
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    return df


class TabularResBlock(nn.Module):
    """Residual block with LayerNorm, Dropout, SiLU activations, and skip connection."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular Residual Neural Network for multi-class classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            blocks.append(TabularResBlock(in_dim, dropout_rate=dropout_rate))
            if in_dim != out_dim:
                blocks.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout_rate),
                    )
                )
        blocks.append(TabularResBlock(hidden_dims[-1], dropout_rate=dropout_rate))
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.head(x)


def build_neural_model_and_optimizer(
    input_dim: int,
    num_classes: int = 7,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    total_epochs: int = 70,
):
    model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(256, 256, 128),
        dropout_rate=0.15,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6
    )
    return model, criterion, optimizer, scheduler


def train_stacking_meta_learner(
    oof_list: List[np.ndarray],
    test_list: List[np.ndarray],
    y_true: np.ndarray,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """Trains a regularized multi-class stacking meta-learner on out-of-fold probability inputs."""
    oof_meta = np.hstack(oof_list)
    test_meta = np.hstack(test_list)

    meta_model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    meta_model.fit(oof_meta, y_true)
    oof_probs = meta_model.predict_proba(oof_meta)
    test_probs = meta_model.predict_proba(test_meta)
    return oof_probs, test_probs, meta_model


def get_lightgbm_params(num_classes: int = 7, seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(num_classes: int = 7, seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(num_classes: int = 7, seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Encode multi-class target
    label_encoder = LabelEncoder()
    train_df["target"] = label_encoder.fit_transform(train_df["Class"])
    class_mapping = {int(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    num_classes = len(class_mapping)

    # Feature engineering
    feature_cols_raw = [c for c in train_df.columns if c not in ["Class", "target"]]
    X_train_fe = create_morphological_features(train_df[feature_cols_raw])
    X_test_fe = create_morphological_features(test_df[feature_cols_raw])

    feature_cols = list(X_train_fe.columns)
    train_df = pd.concat([X_train_fe, train_df[["Class", "target"]]], axis=1)
    test_df = X_test_fe

    # Setup Stratified 10-Fold Cross Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["target"])):
        train_df.loc[val_idx, "fold"] = fold

    num_folds = 10
    n_train = len(train_df)
    n_test = len(test_df)

    oof_preds_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
    test_preds_lgb = np.zeros((n_test, num_classes), dtype=np.float32)

    oof_preds_xgb = np.zeros((n_train, num_classes), dtype=np.float32)
    test_preds_xgb = np.zeros((n_test, num_classes), dtype=np.float32)

    oof_preds_cat = np.zeros((n_train, num_classes), dtype=np.float32)
    test_preds_cat = np.zeros((n_test, num_classes), dtype=np.float32)

    oof_preds_et = np.zeros((n_train, num_classes), dtype=np.float32)
    test_preds_et = np.zeros((n_test, num_classes), dtype=np.float32)

    oof_preds_rf = np.zeros((n_train, num_classes), dtype=np.float32)
    test_preds_rf = np.zeros((n_test, num_classes), dtype=np.float32)

    oof_preds_nn = np.zeros((n_train, num_classes), dtype=np.float32)
    test_preds_nn = np.zeros((n_test, num_classes), dtype=np.float32)

    X_test_raw = test_df[feature_cols].values

    for fold in range(num_folds):
        train_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        X_train_base = train_df.loc[train_mask, feature_cols].values
        y_train = train_df.loc[train_mask, "target"].values
        X_val_base = train_df.loc[val_mask, feature_cols].values
        y_val = train_df.loc[val_mask, "target"].values
        X_test_base = test_df[feature_cols].values

        # Fold-isolated supervised and unsupervised representations fitted strictly on training fold
        morph_scaler = StandardScaler()
        X_tr_norm = morph_scaler.fit_transform(X_train_base)
        X_va_norm = morph_scaler.transform(X_val_base)
        X_te_norm = morph_scaler.transform(X_test_base)

        # 1. Fisher Linear Discriminant Analysis components (6 components for 7 classes)
        lda = LinearDiscriminantAnalysis(n_components=6)
        tr_lda = lda.fit_transform(X_tr_norm, y_train)
        va_lda = lda.transform(X_va_norm)
        te_lda = lda.transform(X_te_norm)

        # 2. Class-centroid Euclidean distance features (7 distances)
        tr_centroids = np.array(
            [X_tr_norm[y_train == c].mean(axis=0) for c in range(num_classes)]
        )
        tr_cent_dist = np.linalg.norm(
            X_tr_norm[:, np.newaxis, :] - tr_centroids[np.newaxis, :, :], axis=2
        )
        va_cent_dist = np.linalg.norm(
            X_va_norm[:, np.newaxis, :] - tr_centroids[np.newaxis, :, :], axis=2
        )
        te_cent_dist = np.linalg.norm(
            X_te_norm[:, np.newaxis, :] - tr_centroids[np.newaxis, :, :], axis=2
        )

        # 3. K-Means prototype spatial distances
        kmeans_7 = KMeans(n_clusters=7, random_state=SEED + fold, n_init=10)
        tr_dist_7 = kmeans_7.fit_transform(X_tr_norm)
        va_dist_7 = kmeans_7.transform(X_va_norm)
        te_dist_7 = kmeans_7.transform(X_te_norm)

        kmeans_14 = KMeans(n_clusters=14, random_state=SEED + fold, n_init=10)
        tr_dist_14 = kmeans_14.fit_transform(X_tr_norm)
        va_dist_14 = kmeans_14.transform(X_va_norm)
        te_dist_14 = kmeans_14.transform(X_te_norm)

        X_train_raw = np.hstack(
            [X_train_base, tr_lda, tr_cent_dist, tr_dist_7, tr_dist_14]
        )
        X_val_raw = np.hstack(
            [X_val_base, va_lda, va_cent_dist, va_dist_7, va_dist_14]
        )
        X_test_raw = np.hstack(
            [X_test_base, te_lda, te_cent_dist, te_dist_7, te_dist_14]
        )

        # 1. LightGBM
        lgb_params = get_lightgbm_params(num_classes=num_classes, seed=SEED + fold)
        dtrain = lgb.Dataset(X_train_raw, label=y_train)
        dval = lgb.Dataset(X_val_raw, label=y_val, reference=dtrain)
        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1200,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds_lgb[val_mask] = lgb_model.predict(
            X_val_raw, num_iteration=lgb_model.best_iteration
        )
        test_preds_lgb += (
            lgb_model.predict(X_test_raw, num_iteration=lgb_model.best_iteration)
            / num_folds
        )

        # 2. XGBoost
        xgb_params = get_xgboost_params(num_classes=num_classes, seed=SEED + fold)
        xgb_model = xgb.XGBClassifier(
            **xgb_params, n_estimators=1200, early_stopping_rounds=50
        )
        xgb_model.fit(
            X_train_raw,
            y_train,
            eval_set=[(X_val_raw, y_val)],
            verbose=False,
        )
        oof_preds_xgb[val_mask] = xgb_model.predict_proba(X_val_raw)
        test_preds_xgb += xgb_model.predict_proba(X_test_raw) / num_folds

        # 3. CatBoost
        cat_params = get_catboost_params(num_classes=num_classes, seed=SEED + fold)
        cat_model = cb.CatBoostClassifier(
            **cat_params, iterations=1200, early_stopping_rounds=50
        )
        cat_model.fit(
            X_train_raw,
            y_train,
            eval_set=(X_val_raw, y_val),
            verbose=False,
        )
        oof_preds_cat[val_mask] = cat_model.predict_proba(X_val_raw)
        test_preds_cat += cat_model.predict_proba(X_test_raw) / num_folds

        # 4. ExtraTrees Classifier
        et_model = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        et_model.fit(X_train_raw, y_train)
        oof_preds_et[val_mask] = et_model.predict_proba(X_val_raw)
        test_preds_et += et_model.predict_proba(X_test_raw) / num_folds

        # 5. Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=350,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        rf_model.fit(X_train_raw, y_train)
        oof_preds_rf[val_mask] = rf_model.predict_proba(X_val_raw)
        test_preds_rf += rf_model.predict_proba(X_test_raw) / num_folds

        # 6. PyTorch TabularResNet
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, drop_last=False
        )

        nn_model, criterion, optimizer, scheduler = build_neural_model_and_optimizer(
            input_dim=X_train_raw.shape[1],
            num_classes=num_classes,
            learning_rate=1e-3,
            weight_decay=1e-4,
            label_smoothing=0.05,
            total_epochs=70,
        )
        nn_model.to(device)

        best_val_loss = float("inf")
        best_nn_val_probs = None
        best_nn_test_probs = None

        for epoch in range(70):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation step
            nn_model.eval()
            with torch.no_grad():
                v_inputs = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
                v_logits = nn_model(v_inputs)
                v_loss = criterion(
                    v_logits, torch.tensor(y_val, dtype=torch.long).to(device)
                ).item()

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_nn_val_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
                    t_inputs = torch.tensor(X_test_scaled, dtype=torch.float32).to(
                        device
                    )
                    t_logits = nn_model(t_inputs)
                    best_nn_test_probs = torch.softmax(t_logits, dim=-1).cpu().numpy()

        oof_preds_nn[val_mask] = best_nn_val_probs
        test_preds_nn += best_nn_test_probs / num_folds

        fold_acc_blend = accuracy_score(
            y_val,
            np.argmax(
                (
                    oof_preds_lgb[val_mask]
                    + oof_preds_xgb[val_mask]
                    + oof_preds_cat[val_mask]
                    + oof_preds_et[val_mask]
                    + oof_preds_rf[val_mask]
                    + oof_preds_nn[val_mask]
                )
                / 6.0,
                axis=1,
            ),
        )
        print(f"Fold {fold + 1}/{num_folds} Mean Blend Accuracy: {fold_acc_blend:.5f}")

    # Collect predictions across all models
    oof_models = [
        oof_preds_lgb,
        oof_preds_xgb,
        oof_preds_cat,
        oof_preds_et,
        oof_preds_rf,
        oof_preds_nn,
    ]
    test_models = [
        test_preds_lgb,
        test_preds_xgb,
        test_preds_cat,
        test_preds_et,
        test_preds_rf,
        test_preds_nn,
    ]
    model_names = [
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "ExtraTrees",
        "RandomForest",
        "TabularResNet",
    ]

    y_true = train_df["target"].values
    for name, oof in zip(model_names, oof_models):
        acc = accuracy_score(y_true, np.argmax(oof, axis=1))
        print(f"OOF Accuracy - {name}: {acc:.5f}")

    # Train regularized Multi-Class Stacking Meta-Learner
    oof_meta_probs, test_meta_probs, meta_model = train_stacking_meta_learner(
        oof_models, test_models, y_true, seed=SEED
    )

    oof_pred_labels = np.argmax(oof_meta_probs, axis=1)
    overall_accuracy = accuracy_score(y_true, oof_pred_labels)

    # Final submission formatting
    test_pred_labels = np.argmax(test_meta_probs, axis=1)
    predicted_class_names = [class_mapping[idx] for idx in test_pred_labels]

    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(predicted_class_names), dtype=int),
            "Class": predicted_class_names,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {overall_accuracy}")


if __name__ == "__main__":
    main()