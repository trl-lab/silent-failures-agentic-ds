import copy
import math
import os
import pickle
import itertools
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# Set deterministic seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. Feature Engineering Pipeline
# ==============================================================================
def engineer_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Applies domain-informed psychological feature engineering, demographic

    aggregations, polynomial interactions, and dimensionality reduction strictly
    without data leakage.
    """
    train = df_train.copy()
    test = df_test.copy()

    # Identify base feature columns (handle UCI typo 'impuslive' vs 'impulsive')
    imp_col = "impuslive" if "impuslive" in train.columns else "impulsive"

    personality_cols = [
        "nscore",
        "escore",
        "oscore",
        "ascore",
        "cscore",
        imp_col,
        "ss",
    ]

    dfs = [train, test]
    for df in dfs:
        # Psychological Domain Composite Indices
        # Externalizing drug-seeking spectrum
        df["feat_externalizing_risk"] = (
            df["ss"] + df[imp_col] + df["oscore"] - df["cscore"] - df["ascore"]
        )

        # Internalizing spectrum: high neuroticism, low extraversion
        df["feat_internalizing_risk"] = df["nscore"] - df["escore"]

        # Sensation-seeking & Impulsivity synergy
        df["feat_ss_x_impulsive"] = df["ss"] * df[imp_col]

        # Openness & Sensation seeking interaction
        df["feat_ss_x_oscore"] = df["ss"] * df["oscore"]

        # Self-regulation & Conscientiousness balance
        df["feat_oscore_minus_cscore"] = df["oscore"] - df["cscore"]
        df["feat_nscore_minus_cscore"] = df["nscore"] - df["cscore"]
        df["feat_ss_minus_cscore"] = df["ss"] - df["cscore"]
        df["feat_impulsive_minus_cscore"] = df[imp_col] - df["cscore"]

        # Prosocial compliance index
        df["feat_prosocial_index"] = df["ascore"] + df["cscore"]

        # Personality vector magnitude (norm) & distribution moments
        pers_vals = df[personality_cols].values
        df["feat_pers_norm"] = np.linalg.norm(pers_vals, axis=1)
        df["feat_pers_mean"] = np.mean(pers_vals, axis=1)
        df["feat_pers_std"] = np.std(pers_vals, axis=1)
        df["feat_pers_max"] = np.max(pers_vals, axis=1)
        df["feat_pers_min"] = np.min(pers_vals, axis=1)

        # Polynomial non-linear terms for key predictors
        df["feat_ss_sq"] = df["ss"] ** 2
        df["feat_oscore_sq"] = df["oscore"] ** 2
        df["feat_cscore_sq"] = df["cscore"] ** 2
        df["feat_impulsive_sq"] = df[imp_col] ** 2
        df["feat_nscore_sq"] = df["nscore"] ** 2

        # Age-personality interactions
        if "age" in df.columns:
            df["feat_age_x_ss"] = df["age"] * df["ss"]
            df["feat_age_x_oscore"] = df["age"] * df["oscore"]
            df["feat_age_x_extrisk"] = df["age"] * df["feat_externalizing_risk"]

        # Country / Education interactions if present
        if "education" in df.columns:
            df["feat_edu_x_cscore"] = df["education"] * df["cscore"]
            df["feat_edu_x_oscore"] = df["education"] * df["oscore"]

    # Demographic Group-Level Aggregations (Fitted strictly on train to avoid leakage)
    group_keys = [k for k in ["country", "age", "education"] if k in train.columns]
    for g_col in group_keys:
        for val_col in ["ss", "oscore", "feat_externalizing_risk", "cscore"]:
            group_stats = (
                train.groupby(g_col)[val_col].agg(["mean", "std"]).reset_index()
            )
            group_stats.columns = [
                g_col,
                f"feat_{g_col}_{val_col}_grp_mean",
                f"feat_{g_col}_{val_col}_grp_std",
            ]
            group_stats[f"feat_{g_col}_{val_col}_grp_std"] = group_stats[
                f"feat_{g_col}_{val_col}_grp_std"
            ].fillna(0.0)

            train = train.merge(group_stats, on=g_col, how="left")
            test = test.merge(group_stats, on=g_col, how="left")

            # Group-relative differences
            train[f"feat_{g_col}_{val_col}_diff"] = (
                train[val_col] - train[f"feat_{g_col}_{val_col}_grp_mean"]
            )
            test[f"feat_{g_col}_{val_col}_diff"] = (
                test[val_col] - test[f"feat_{g_col}_{val_col}_grp_mean"]
            )

            # Impute unseen test levels with train global mean
            global_mean = train[val_col].mean()
            train[f"feat_{g_col}_{val_col}_grp_mean"] = train[
                f"feat_{g_col}_{val_col}_grp_mean"
            ].fillna(global_mean)
            test[f"feat_{g_col}_{val_col}_grp_mean"] = test[
                f"feat_{g_col}_{val_col}_grp_mean"
            ].fillna(global_mean)
            train[f"feat_{g_col}_{val_col}_grp_std"] = train[
                f"feat_{g_col}_{val_col}_grp_std"
            ].fillna(0.0)
            test[f"feat_{g_col}_{val_col}_grp_std"] = test[
                f"feat_{g_col}_{val_col}_grp_std"
            ].fillna(0.0)
            train[f"feat_{g_col}_{val_col}_diff"] = train[
                f"feat_{g_col}_{val_col}_diff"
            ].fillna(0.0)
            test[f"feat_{g_col}_{val_col}_diff"] = test[
                f"feat_{g_col}_{val_col}_diff"
            ].fillna(0.0)

    return train, test


# ==============================================================================
# 2. Model Architectures & Factories
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual Block with LayerNorm, Swish (SiLU) activation, and Dropout for tabular features."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep regularized Residual Tabular Network for multi-class personality-behavior prediction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
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

        self.res_blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.classifier_head(h)


def get_neural_criterion(label_smoothing: float = 0.05) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_neural_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 80,
    min_lr: float = 1e-6,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optimizer, scheduler


def get_lgb_classifier(
    num_classes: int,
    random_state: int = 42,
    n_estimators: int = 600,
    learning_rate: float = 0.03,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def get_xgb_classifier(
    num_classes: int,
    random_state: int = 42,
    n_estimators: int = 600,
    learning_rate: float = 0.03,
) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        gamma=0.1,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )


def get_cat_classifier(
    num_classes: int,
    random_state: int = 42,
    iterations: int = 600,
    learning_rate: float = 0.03,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=False,
        early_stopping_rounds=50,
    )


def get_et_classifier(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_rf_classifier(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_lr_classifier(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )


def optimize_ensemble_weights(
    oof_probs_list: list, y_true: np.ndarray
) -> np.ndarray:
    """Finds optimal ensemble probability mixing weights minimizing multiclass log loss."""
    n_models = len(oof_probs_list)
    eps = 1e-15

    def loss_func(weights: np.ndarray) -> float:
        w_norm = weights / np.sum(weights)
        blended = np.zeros_like(oof_probs_list[0])
        for w, p in zip(w_norm, oof_probs_list):
            blended += w * p
        blended = np.clip(blended, eps, 1.0 - eps)
        log_p = np.log(blended)
        return -float(np.mean(log_p[np.arange(len(y_true)), y_true]))

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    best_weights = res.x / np.sum(res.x)
    return best_weights


# ==============================================================================
# 3. Main Cross-Validation Training & Inference Pipeline
# ==============================================================================
def main():
    # Setup directories
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 1. Load Raw Data
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    id_col = "id"
    target_col = "cannabis"

    if id_col not in test_df.columns:
        test_ids = test_df.index.values
    else:
        test_ids = test_df[id_col].values

    # Target Encoding
    target_raw = train_df[target_col].copy()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(target_raw.astype(str))
    num_classes = len(label_encoder.classes_)

    # Exclude other target drug columns if present in train
    drug_targets = [
        "alcohol",
        "amphet",
        "amyl",
        "benzos",
        "caff",
        "cannabis",
        "choc",
        "coke",
        "crack",
        "ecstasy",
        "heroin",
        "ketamine",
        "legalh",
        "lsd",
        "meth",
        "mushrooms",
        "nicotine",
        "semer",
        "vsa",
    ]
    cols_to_drop_train = [c for c in drug_targets if c in train_df.columns]

    X_train_raw = train_df.drop(columns=cols_to_drop_train)
    X_test_raw = test_df.copy()

    if id_col in X_train_raw.columns:
        X_train_raw = X_train_raw.drop(columns=[id_col])
    if id_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[id_col])

    # 2. Domain Feature Engineering
    X_train_eng, X_test_eng = engineer_features(X_train_raw, X_test_raw)

    feature_names = [col for col in X_train_eng.columns if col in X_test_eng.columns]
    X_train_final = (
        X_train_eng[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    X_test_final = (
        X_test_eng[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    X_train_arr = X_train_final.values
    X_test_arr = X_test_final.values
    num_features = X_train_arr.shape[1]

    # 3. Stratified 5-Fold Cross-Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_train_arr, y_train))
    n_splits = len(cv_splits)
    n_train = len(y_train)
    n_test = len(X_test_arr)

    oof_lgb = np.zeros((n_train, num_classes))
    oof_xgb = np.zeros((n_train, num_classes))
    oof_cat = np.zeros((n_train, num_classes))
    oof_et = np.zeros((n_train, num_classes))
    oof_rf = np.zeros((n_train, num_classes))
    oof_lr = np.zeros((n_train, num_classes))
    oof_nn = np.zeros((n_train, num_classes))

    test_lgb_probs = np.zeros((n_test, num_classes))
    test_xgb_probs = np.zeros((n_test, num_classes))
    test_cat_probs = np.zeros((n_test, num_classes))
    test_et_probs = np.zeros((n_test, num_classes))
    test_rf_probs = np.zeros((n_test, num_classes))
    test_lr_probs = np.zeros((n_test, num_classes))
    test_nn_probs = np.zeros((n_test, num_classes))

    # 4. Cross-Validation Training Loop
    for fold, (trn_idx, val_idx) in enumerate(cv_splits):
        X_tr_raw, y_tr = X_train_arr[trn_idx], y_train[trn_idx]
        X_val_raw, y_val = X_train_arr[val_idx], y_train[val_idx]
        X_te_raw = X_test_arr.copy()

        # Fit PCA and KMeans strictly on fold train data
        scaler_pers = StandardScaler()
        tr_scaled = scaler_pers.fit_transform(X_tr_raw)
        val_scaled = scaler_pers.transform(X_val_raw)
        te_scaled = scaler_pers.transform(X_te_raw)

        pca = PCA(n_components=3, random_state=42)
        tr_pca = pca.fit_transform(tr_scaled)
        val_pca = pca.transform(val_scaled)
        te_pca = pca.transform(te_scaled)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(tr_scaled)
        tr_km = kmeans.transform(tr_scaled)
        val_km = kmeans.transform(val_scaled)
        te_km = kmeans.transform(te_scaled)

        X_tr = np.hstack([X_tr_raw, tr_pca, tr_km])
        X_val = np.hstack([X_val_raw, val_pca, val_km])
        X_test_fold = np.hstack([X_te_raw, te_pca, te_km])

        # Model 1: LightGBM
        lgb_clf = get_lgb_classifier(num_classes=num_classes, random_state=42 + fold)
        lgb_clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_val)
        test_lgb_probs += lgb_clf.predict_proba(X_test_fold) / n_splits

        # Model 2: XGBoost
        xgb_clf = get_xgb_classifier(num_classes=num_classes, random_state=42 + fold)
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_clf.predict_proba(X_val)
        test_xgb_probs += xgb_clf.predict_proba(X_test_fold) / n_splits

        # Model 3: CatBoost
        cat_clf = get_cat_classifier(num_classes=num_classes, random_state=42 + fold)
        cat_clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = cat_clf.predict_proba(X_val)
        test_cat_probs += cat_clf.predict_proba(X_test_fold) / n_splits

        # Model 4: ExtraTrees
        et_clf = get_et_classifier(random_state=42 + fold)
        et_clf.fit(X_tr, y_tr)
        oof_et[val_idx] = et_clf.predict_proba(X_val)
        test_et_probs += et_clf.predict_proba(X_test_fold) / n_splits

        # Model 5: RandomForest
        rf_clf = get_rf_classifier(random_state=42 + fold)
        rf_clf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf_clf.predict_proba(X_val)
        test_rf_probs += rf_clf.predict_proba(X_test_fold) / n_splits

        # Scaled data strictly fold-wise for linear and neural models
        scaler_fold = StandardScaler()
        X_tr_scaled = scaler_fold.fit_transform(X_tr)
        X_val_scaled = scaler_fold.transform(X_val)
        X_test_scaled = scaler_fold.transform(X_test_fold)

        # Model 6: Logistic Regression
        lr_clf = get_lr_classifier(random_state=42 + fold)
        lr_clf.fit(X_tr_scaled, y_tr)
        oof_lr[val_idx] = lr_clf.predict_proba(X_val_scaled)
        test_lr_probs += lr_clf.predict_proba(X_test_scaled) / n_splits

        # Model 7: TabularResNet
        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        nn_model = TabularResNet(
            input_dim=X_tr.shape[1],
            num_classes=num_classes,
            hidden_dim=128,
            num_blocks=2,
            dropout_rate=0.25,
        ).to(device)

        criterion = get_neural_criterion(label_smoothing=0.05)
        epochs = 75
        optimizer, scheduler = get_neural_optimizer_and_scheduler(
            nn_model, lr=1.5e-3, weight_decay=1e-4, epochs=epochs
        )

        best_val_loss = float("inf")
        best_model_weights = None

        for epoch in range(epochs):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = nn_model(bx)
                    val_loss += criterion(logits, by).item() * len(by)
            val_loss /= len(y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(nn_model.state_dict())

        nn_model.load_state_dict(best_model_weights)
        nn_model.eval()

        with torch.no_grad():
            val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            val_probs = torch.softmax(nn_model(val_t), dim=1).cpu().numpy()
            oof_nn[val_idx] = val_probs

            test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            test_probs = torch.softmax(nn_model(test_t), dim=1).cpu().numpy()
            test_nn_probs += test_probs / n_splits

    # 5. Out-of-Fold Evaluation and Weighted Probability Optimization
    oof_list = [oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf, oof_lr, oof_nn]
    test_list = [
        test_lgb_probs,
        test_xgb_probs,
        test_cat_probs,
        test_et_probs,
        test_rf_probs,
        test_lr_probs,
        test_nn_probs,
    ]
    optimal_weights = optimize_ensemble_weights(oof_list, y_train)

    oof_ensemble_probs = np.zeros_like(oof_lgb)
    for w, p in zip(optimal_weights, oof_list):
        oof_ensemble_probs += w * p
    oof_predictions = np.argmax(oof_ensemble_probs, axis=1)

    overall_accuracy = accuracy_score(y_train, oof_predictions)

    # 6. Generate and Save Submission File
    final_test_probs = np.zeros_like(test_lgb_probs)
    for w, p in zip(optimal_weights, test_list):
        final_test_probs += w * p
    test_pred_indices = np.argmax(final_test_probs, axis=1)
    test_pred_labels = label_encoder.inverse_transform(test_pred_indices)

    submission_df = pd.DataFrame({"id": test_ids, "cannabis": test_pred_labels})
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    # Final Validation Score line required for metric tracking
    print(f"Final Validation Score: {overall_accuracy:.6f}")


if __name__ == "__main__":
    main()