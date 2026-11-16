import copy
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ==========================================
# 1. Feature Engineering Strategy
# ==========================================
def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific geometric, relative severity, and statistical feature engineering
    for breast cancer diagnostic cell nuclei measurements.
    """
    df = df_in.copy()

    core_metrics = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]

    # Geometric & Shape Features
    for suffix in ["1", "2", "3"]:
        area_col = f"area{suffix}"
        peri_col = f"perimeter{suffix}"
        rad_col = f"radius{suffix}"

        if area_col in df.columns and peri_col in df.columns:
            df[f"area_to_perimeter_{suffix}"] = df[area_col] / (df[peri_col] + 1e-8)
            df[f"circularity_{suffix}"] = (4 * np.pi * df[area_col]) / (
                df[peri_col] ** 2 + 1e-8
            )

        if rad_col in df.columns and peri_col in df.columns:
            df[f"radius_perimeter_ratio_{suffix}"] = df[peri_col] / (
                2 * np.pi * df[rad_col] + 1e-8
            )

    # Multi-tier Ratio Features
    for metric in core_metrics:
        m1 = f"{metric}1"
        m2 = f"{metric}2"
        m3 = f"{metric}3"

        if m1 in df.columns and m2 in df.columns:
            df[f"{metric}_rel_se"] = df[m2] / (df[m1] + 1e-8)

        if m1 in df.columns and m3 in df.columns:
            df[f"{metric}_worst_ratio"] = df[m3] / (df[m1] + 1e-8)
            df[f"{metric}_worst_spread"] = df[m3] - df[m1]

        # Cross-tier Statistical Summary Features
        if m1 in df.columns and m2 in df.columns and m3 in df.columns:
            tier_df = df[[m1, m2, m3]]
            df[f"{metric}_cross_mean"] = tier_df.mean(axis=1)
            df[f"{metric}_cross_std"] = tier_df.std(axis=1)
            df[f"{metric}_cross_min"] = tier_df.min(axis=1)
            df[f"{metric}_cross_max"] = tier_df.max(axis=1)
            df[f"{metric}_cross_range"] = (
                df[f"{metric}_cross_max"] - df[f"{metric}_cross_min"]
            )

    # Composite Interaction Features
    if "concave_points3" in df.columns and "radius3" in df.columns:
        df["concavity_size_product_3"] = df["concave_points3"] * df["radius3"]
    if "compactness3" in df.columns and "concavity3" in df.columns:
        df["shape_irregularity_3"] = df["compactness3"] * df["concavity3"]
    if "texture3" in df.columns and "area3" in df.columns:
        df["texture_area_interaction_3"] = df["texture3"] * df["area3"]

    # Log-transform skewed magnitude features
    skewed_cols = [c for c in df.columns if "area" in c or "perimeter" in c]
    for col in skewed_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[f"{col}_log"] = np.log1p(np.maximum(0, df[col]))

    return df


# ==========================================
# 2. PyTorch Tabular Neural Architecture
# ==========================================
class TabularResBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def create_pytorch_model_and_optimizer(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-2
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=64, num_blocks=2, dropout_rate=0.25
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


# ==========================================
# 3. Main Data Processing & Training Pipeline
# ==========================================
def main():
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Diagnosis"
    id_col = (
        "ID"
        if "ID" in train_df.columns
        else ("id" if "id" in train_df.columns else None)
    )

    # Standardize Target Encoding
    if train_df[target_col].dtype == object or isinstance(
        train_df[target_col].iloc[0], str
    ):
        target_map = {"M": 1, "B": 0, "1": 1, "0": 0}
        y_train = train_df[target_col].map(target_map).astype(int).values
    else:
        y_train = train_df[target_col].astype(int).values

    # Separate raw features
    drop_cols = [c for c in [target_col, id_col] if c and c in train_df.columns]
    X_train_raw = train_df.drop(columns=drop_cols)
    drop_cols_test = [c for c in [target_col, id_col] if c and c in test_df.columns]
    X_test_raw = test_df.drop(columns=drop_cols_test)

    # Apply Feature Engineering
    X_train_fe = engineer_features(X_train_raw)
    X_test_fe = engineer_features(X_test_raw)

    common_cols = [col for col in X_train_fe.columns if col in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].replace([np.inf, -np.inf], np.nan)
    X_test_fe = X_test_fe[common_cols].replace([np.inf, -np.inf], np.nan)

    X_train_mat = X_train_fe.values
    X_test_mat = X_test_fe.values

    n_samples, n_features = X_train_mat.shape
    n_test = X_test_mat.shape[0]

    # Model Ensembling Setup: 10-fold Repeated Stratified K-Fold across 3 seeds
    n_splits = 10
    n_repeats = 3
    total_folds = n_splits * n_repeats
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    n_models = 7
    oof_model_preds = np.zeros((n_samples, n_models))
    test_model_preds = np.zeros((n_test, n_models))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train_mat, y_train)):
        X_tr, y_tr = X_train_mat[train_idx], y_train[train_idx]
        X_va, y_va = X_train_mat[val_idx], y_train[val_idx]

        # Strict Leak-Free Imputation & Feature Transformation Per Fold
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_va_imp = imputer.transform(X_va)
        X_test_imp = imputer.transform(X_test_mat)

        # Standard scaling for tree-based models
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(X_tr_imp)
        X_va_proc = scaler.transform(X_va_imp)
        X_test_proc = scaler.transform(X_test_imp)

        # Quantile Transformation for non-tree models
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, X_tr.shape[0]),
            random_state=42 + fold,
        )
        X_tr_qt = qt.fit_transform(X_tr_imp)
        X_va_qt = qt.transform(X_va_imp)
        X_test_qt = qt.transform(X_test_imp)

        # Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.015,
            max_depth=3,
            num_leaves=7,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42 + fold,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_proc,
            y_tr,
            eval_set=[(X_va_proc, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_val_pred = lgb_model.predict_proba(X_va_proc)[:, 1]
        lgb_test_pred = lgb_model.predict_proba(X_test_proc)[:, 1]

        # Model 2: XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.015,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42 + fold,
            eval_metric="auc",
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr_proc, y_tr, eval_set=[(X_va_proc, y_va)], verbose=False)
        xgb_val_pred = xgb_model.predict_proba(X_va_proc)[:, 1]
        xgb_test_pred = xgb_model.predict_proba(X_test_proc)[:, 1]

        # Model 3: ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            max_features="sqrt",
            min_samples_split=4,
            random_state=42 + fold,
        )
        et_model.fit(X_tr_proc, y_tr)
        et_val_pred = et_model.predict_proba(X_va_proc)[:, 1]
        et_test_pred = et_model.predict_proba(X_test_proc)[:, 1]

        # Model 4: ElasticNet Logistic Regression (on QuantileTransformed data)
        lr_model = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            C=0.3,
            solver="saga",
            max_iter=2000,
            random_state=42 + fold,
        )
        lr_model.fit(X_tr_qt, y_tr)
        lr_val_pred = lr_model.predict_proba(X_va_qt)[:, 1]
        lr_test_pred = lr_model.predict_proba(X_test_qt)[:, 1]

        # Model 5: PyTorch TabularResNet (on QuantileTransformed data)
        nn_model, nn_criterion, nn_optimizer, nn_scheduler = (
            create_pytorch_model_and_optimizer(n_features)
        )
        nn_model = nn_model.to(device)

        tr_ds = TensorDataset(
            torch.tensor(X_tr_qt, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        )
        va_ds = TensorDataset(
            torch.tensor(X_va_qt, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
        )
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(nn_model.state_dict())

        for epoch in range(100):
            nn_model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                nn_optimizer.zero_grad()
                out = nn_model(bx)
                loss = nn_criterion(out, by)
                loss.backward()
                nn_optimizer.step()
            nn_scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = nn_model(bx)
                    val_loss += nn_criterion(out, by).item() * bx.size(0)
            val_loss /= len(X_va_qt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(nn_model.state_dict())

        nn_model.load_state_dict(best_weights)
        nn_model.eval()
        with torch.no_grad():
            nn_val_logits = nn_model(
                torch.tensor(X_va_qt, dtype=torch.float32).to(device)
            )
            nn_val_pred = torch.sigmoid(nn_val_logits).cpu().numpy().flatten()
            nn_test_logits = nn_model(
                torch.tensor(X_test_qt, dtype=torch.float32).to(device)
            )
            nn_test_pred = torch.sigmoid(nn_test_logits).cpu().numpy().flatten()

        # Model 6: CatBoost
        cb_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.015,
            depth=4,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_tr_proc, y_tr, eval_set=(X_va_proc, y_va), verbose=False)
        cb_val_pred = cb_model.predict_proba(X_va_proc)[:, 1]
        cb_test_pred = cb_model.predict_proba(X_test_proc)[:, 1]

        # Model 7: Calibrated Support Vector Classifier (RBF kernel, on QuantileTransformed data)
        base_svc = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42 + fold)
        svc_model = CalibratedClassifierCV(estimator=base_svc, cv=3)
        svc_model.fit(X_tr_qt, y_tr)
        svc_val_pred = svc_model.predict_proba(X_va_qt)[:, 1]
        svc_test_pred = svc_model.predict_proba(X_test_qt)[:, 1]

        fold_val_preds = [
            lgb_val_pred,
            xgb_val_pred,
            et_val_pred,
            lr_val_pred,
            nn_val_pred,
            cb_val_pred,
            svc_val_pred,
        ]
        fold_test_preds = [
            lgb_test_pred,
            xgb_test_pred,
            et_test_pred,
            lr_test_pred,
            nn_test_pred,
            cb_test_pred,
            svc_test_pred,
        ]

        for m in range(n_models):
            oof_model_preds[val_idx, m] += fold_val_preds[m] / n_repeats
            test_model_preds[:, m] += fold_test_preds[m] / total_folds

    # Out-Of-Fold Rank-Normalized Meta-Learning Stacking
    def rank_normalize(arr: np.ndarray) -> np.ndarray:
        res = np.zeros_like(arr, dtype=float)
        for col in range(arr.shape[1]):
            res[:, col] = rankdata(arr[:, col]) / arr.shape[0]
        return res

    X_meta_oof = rank_normalize(oof_model_preds)
    X_meta_test = rank_normalize(test_model_preds)

    meta_learner = LogisticRegressionCV(
        penalty="l2",
        solver="lbfgs",
        Cs=10,
        cv=5,
        max_iter=2000,
        random_state=42,
    )
    meta_learner.fit(X_meta_oof, y_train)

    oof_preds = meta_learner.predict_proba(X_meta_oof)[:, 1]
    test_preds = meta_learner.predict_proba(X_meta_test)[:, 1]

    # Compute overall Out-Of-Fold ROC-AUC score
    final_auc = roc_auc_score(y_train, oof_preds)

    # Save Submission File
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame({"id": np.arange(n_test), "Diagnosis": test_preds})
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_auc}")


if __name__ == "__main__":
    main()