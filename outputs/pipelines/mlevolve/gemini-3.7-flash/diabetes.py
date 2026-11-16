import copy
import os
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary column naming variants to standard canonical names."""
    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "preg" in col_lower:
            rename_map[col] = "pregnancies"
        elif "plas" in col_lower or "gluc" in col_lower:
            rename_map[col] = "glucose"
        elif "pres" in col_lower or "blood" in col_lower or "bp" in col_lower:
            rename_map[col] = "blood_pressure"
        elif "skin" in col_lower:
            rename_map[col] = "skin_thickness"
        elif "insu" in col_lower:
            rename_map[col] = "insulin"
        elif "mass" in col_lower or "bmi" in col_lower:
            rename_map[col] = "bmi"
        elif "pedi" in col_lower:
            rename_map[col] = "diabetes_pedigree"
        elif "age" in col_lower:
            rename_map[col] = "age"
        elif "class" in col_lower or "target" in col_lower or "outcome" in col_lower:
            rename_map[col] = "target"
        else:
            rename_map[col] = col_lower
    return df.rename(columns=rename_map)


def parse_target(series: pd.Series) -> pd.Series:
    """Standardize target values to binary 0/1 integers."""
    mapping = {
        "tested_positive": 1,
        "tested_negative": 0,
        "positive": 1,
        "negative": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
        1.0: 1,
        0.0: 0,
        True: 1,
        False: 0,
    }
    return series.map(mapping).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform domain-informed feature engineering:
    - Treat biologically implausible zeros as missing.
    - Create missingness indicator flags.
    - Create metabolic ratios, clinical risk factors, and non-linear interactions.
    """
    df_feat = df.copy()
    zero_as_missing_cols = [
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
    ]

    missing_count = np.zeros(len(df_feat), dtype=np.float32)
    for col in zero_as_missing_cols:
        if col in df_feat.columns:
            is_zero = (df_feat[col] == 0) | df_feat[col].isna()
            df_feat[f"{col}_is_missing"] = is_zero.astype(float)
            missing_count += is_zero.astype(float)
            df_feat.loc[is_zero, col] = np.nan

    df_feat["total_missing_count"] = missing_count

    eps = 1e-5
    df_feat["homa_ir_proxy"] = (df_feat["glucose"] * df_feat["insulin"]) / 405.0
    df_feat["insulin_to_glucose"] = df_feat["insulin"] / (df_feat["glucose"] + eps)
    df_feat["glucose_to_bmi"] = df_feat["glucose"] / (df_feat["bmi"] + eps)
    df_feat["bmi_age_interaction"] = df_feat["bmi"] * df_feat["age"]
    df_feat["glucose_age_interaction"] = df_feat["glucose"] * df_feat["age"]
    df_feat["pedigree_age_interaction"] = df_feat["diabetes_pedigree"] * df_feat["age"]
    df_feat["glucose_pedigree_interaction"] = (
        df_feat["glucose"] * df_feat["diabetes_pedigree"]
    )
    df_feat["fertility_rate"] = df_feat["pregnancies"] / np.maximum(
        df_feat["age"] - 14.0, 1.0
    )

    # Clinical novel biomarkers
    df_feat["matsuda_index"] = 10000.0 / (
        np.sqrt(np.maximum(df_feat["glucose"] * df_feat["insulin"], 0.0)) + eps
    )
    df_feat["subcutaneous_adiposity_ratio"] = df_feat["skin_thickness"] / (
        df_feat["bmi"] + eps
    )
    df_feat["metabolic_pressure_index"] = (
        df_feat["bmi"] * df_feat["blood_pressure"]
    ) / 1000.0

    df_feat["is_impaired_glucose"] = (df_feat["glucose"] >= 140.0).astype(float)
    df_feat["is_obese"] = (df_feat["bmi"] >= 30.0).astype(float)
    df_feat["is_severe_obese"] = (df_feat["bmi"] >= 35.0).astype(float)
    df_feat["is_hypertension"] = (df_feat["blood_pressure"] >= 80.0).astype(float)
    df_feat["metabolic_syndrome_risk"] = (
        (df_feat["glucose"] >= 140.0).astype(float)
        + (df_feat["bmi"] >= 30.0).astype(float)
        + (df_feat["blood_pressure"] >= 80.0).astype(float)
        + (df_feat["age"] >= 35.0).astype(float)
    )

    df_feat["log_insulin"] = np.log1p(np.maximum(df_feat["insulin"], 0))
    df_feat["log_diabetes_pedigree"] = np.log1p(
        np.maximum(df_feat["diabetes_pedigree"], 0)
    )
    df_feat["log_pregnancies"] = np.log1p(np.maximum(df_feat["pregnancies"], 0))
    df_feat["log_homa_ir"] = np.log1p(np.maximum(df_feat["homa_ir_proxy"], 0))
    df_feat["log_matsuda"] = np.log1p(np.maximum(df_feat["matsuda_index"], 0))

    return df_feat


class ResidualBlock(nn.Module):
    """
    A regularized residual block tailored for small-to-medium tabular feature spaces.
    Applies Linear -> LayerNorm -> SiLU -> Dropout -> Linear -> LayerNorm -> Skip Connection.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        return self.activation(out + residual)


class TabularResNet(nn.Module):
    """
    Deep Tabular Residual Network for binary clinical classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


def train_tabular_resnet(
    X_tr_np: np.ndarray,
    y_tr_np: np.ndarray,
    X_va_np: np.ndarray,
    y_va_np: np.ndarray,
    input_dim: int,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
):
    pos_weight = float((len(y_tr_np) - y_tr_np.sum()) / (y_tr_np.sum() + 1e-5))
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=64,
        num_blocks=2,
        dropout_rate=0.3,
        output_dim=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    tr_tensor_x = torch.tensor(X_tr_np, dtype=torch.float32)
    tr_tensor_y = torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(tr_tensor_x, tr_tensor_y),
        batch_size=batch_size,
        shuffle=True,
    )

    va_tensor_x = torch.tensor(X_va_np, dtype=torch.float32).to(device)

    best_val_auc = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(va_tensor_x)
            va_preds = torch.sigmoid(va_logits).cpu().numpy().ravel()

        val_auc = roc_auc_score(y_va_np, va_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        final_va_preds = torch.sigmoid(model(va_tensor_x)).cpu().numpy().ravel()

    return model, final_va_preds


def get_lightgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=random_state,
        n_jobs=-1,
    )


def get_extratrees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_hist_gbdt_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=random_state,
    )


def get_logistic_model(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        C=0.5,
        solver="saga",
        max_iter=2000,
        random_state=random_state,
    )


def main():
    set_seed(42)
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    # 2. Standardize column names
    train_df = standardize_columns(train_raw)
    test_df = standardize_columns(test_raw)

    target_col = "target"
    y = parse_target(train_df[target_col])
    X_train_raw = train_df.drop(columns=[target_col])
    X_test_raw = test_df.copy()
    if target_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[target_col])

    # 3. Feature engineering
    X_train_eng = build_features(X_train_raw)
    X_test_eng = build_features(X_test_raw)
    feature_cols = list(X_train_eng.columns)
    input_dim = len(feature_cols)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_targets = np.zeros(len(train_df), dtype=np.int32)
    oof_nn_preds = np.zeros(len(train_df), dtype=np.float64)
    oof_lgb_preds = np.zeros(len(train_df), dtype=np.float64)
    oof_xgb_preds = np.zeros(len(train_df), dtype=np.float64)
    oof_et_preds = np.zeros(len(train_df), dtype=np.float64)
    oof_hgb_preds = np.zeros(len(train_df), dtype=np.float64)
    oof_lr_preds = np.zeros(len(train_df), dtype=np.float64)

    test_nn_preds_folds = np.zeros(len(test_raw), dtype=np.float64)
    test_lgb_preds_folds = np.zeros(len(test_raw), dtype=np.float64)
    test_xgb_preds_folds = np.zeros(len(test_raw), dtype=np.float64)
    test_et_preds_folds = np.zeros(len(test_raw), dtype=np.float64)
    test_hgb_preds_folds = np.zeros(len(test_raw), dtype=np.float64)
    test_lr_preds_folds = np.zeros(len(test_raw), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_eng, y)):
        X_tr = X_train_eng.iloc[train_idx].copy()
        y_tr = y.iloc[train_idx].values
        X_va = X_train_eng.iloc[val_idx].copy()
        y_va = y.iloc[val_idx].values
        X_te = X_test_eng.copy()

        # Leak-free Bayesian Ridge MICE Imputation fitted strictly on training fold
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=42 + fold,
            sample_posterior=False,
        )
        X_tr_imp = imputer.fit_transform(X_tr[feature_cols])
        X_va_imp = imputer.transform(X_va[feature_cols])
        X_te_imp = imputer.transform(X_te[feature_cols])

        # Standard Scaling fitted strictly on training fold
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_va_scaled = scaler.transform(X_va_imp)
        X_te_scaled = scaler.transform(X_te_imp)

        # 1. TabularResNet
        nn_model, val_nn_pred = train_tabular_resnet(
            X_tr_scaled,
            y_tr,
            X_va_scaled,
            y_va,
            input_dim=input_dim,
            epochs=80,
            batch_size=32,
            lr=1e-3,
            device=device,
        )
        with torch.no_grad():
            te_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            test_nn_pred = torch.sigmoid(nn_model(te_tensor_x)).cpu().numpy().ravel()

        # 2. LightGBM
        lgb_model = get_lightgbm_model(random_state=42 + fold)
        lgb_model.fit(
            X_tr_imp,
            y_tr,
            eval_set=[(X_va_imp, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_lgb_pred = lgb_model.predict_proba(X_va_imp)[:, 1]
        test_lgb_pred = lgb_model.predict_proba(X_te_imp)[:, 1]

        # 3. XGBoost
        xgb_model = get_xgboost_model(random_state=42 + fold)
        xgb_model.fit(
            X_tr_imp,
            y_tr,
            eval_set=[(X_va_imp, y_va)],
            verbose=False,
        )
        val_xgb_pred = xgb_model.predict_proba(X_va_imp)[:, 1]
        test_xgb_pred = xgb_model.predict_proba(X_te_imp)[:, 1]

        # 4. ExtraTrees
        et_model = get_extratrees_model(random_state=42 + fold)
        et_model.fit(X_tr_imp, y_tr)
        val_et_pred = et_model.predict_proba(X_va_imp)[:, 1]
        test_et_pred = et_model.predict_proba(X_te_imp)[:, 1]

        # 5. HistGradientBoosting
        hgb_model = get_hist_gbdt_model(random_state=42 + fold)
        hgb_model.fit(X_tr_imp, y_tr)
        val_hgb_pred = hgb_model.predict_proba(X_va_imp)[:, 1]
        test_hgb_pred = hgb_model.predict_proba(X_te_imp)[:, 1]

        # 6. ElasticNet Logistic Regression
        lr_model = get_logistic_model(random_state=42 + fold)
        lr_model.fit(X_tr_scaled, y_tr)
        val_lr_pred = lr_model.predict_proba(X_va_scaled)[:, 1]
        test_lr_pred = lr_model.predict_proba(X_te_scaled)[:, 1]

        oof_targets[val_idx] = y_va
        oof_nn_preds[val_idx] = val_nn_pred
        oof_lgb_preds[val_idx] = val_lgb_pred
        oof_xgb_preds[val_idx] = val_xgb_pred
        oof_et_preds[val_idx] = val_et_pred
        oof_hgb_preds[val_idx] = val_hgb_pred
        oof_lr_preds[val_idx] = val_lr_pred

        test_nn_preds_folds += test_nn_pred / n_splits
        test_lgb_preds_folds += test_lgb_pred / n_splits
        test_xgb_preds_folds += test_xgb_pred / n_splits
        test_et_preds_folds += test_et_pred / n_splits
        test_hgb_preds_folds += test_hgb_pred / n_splits
        test_lr_preds_folds += test_lr_pred / n_splits

    oof_matrix = np.column_stack(
        [
            oof_nn_preds,
            oof_lgb_preds,
            oof_xgb_preds,
            oof_et_preds,
            oof_hgb_preds,
            oof_lr_preds,
        ]
    )

    test_matrix = np.column_stack(
        [
            test_nn_preds_folds,
            test_lgb_preds_folds,
            test_xgb_preds_folds,
            test_et_preds_folds,
            test_hgb_preds_folds,
            test_lr_preds_folds,
        ]
    )

    # Simplex meta-optimization maximizing OOF ROC-AUC
    n_models = oof_matrix.shape[1]

    def auc_objective(weights: np.ndarray) -> float:
        w = weights / (np.sum(weights) + 1e-12)
        blend = np.dot(oof_matrix, w)
        return -roc_auc_score(oof_targets, blend)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        auc_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = (
        res.x / np.sum(res.x) if np.sum(res.x) > 0 else init_weights
    )

    # Global Out-of-Fold Evaluation
    oof_ensemble_preds = np.dot(oof_matrix, best_weights)
    final_score = roc_auc_score(oof_targets, oof_ensemble_preds)

    # Test Ensemble Predictions
    test_ensemble_preds = np.dot(test_matrix, best_weights)

    # Save final submission
    submission = pd.DataFrame(
        {"id": np.arange(len(test_ensemble_preds)), "class": test_ensemble_preds}
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
