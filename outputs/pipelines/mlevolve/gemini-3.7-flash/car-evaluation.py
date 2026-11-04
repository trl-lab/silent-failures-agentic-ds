import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# 1. Data Loading & Feature Engineering
# =========================================================================


def load_raw_data(data_dir="./input"):
    """Load raw train and test datasets from parquet files."""
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    return df_train, df_test


def identify_target_column(df_train):
    """Identify the target column in the training dataset."""
    possible_targets = ["car", "class", "target", "evaluation"]
    for col in possible_targets:
        if col in df_train.columns:
            return col
    return df_train.columns[-1]


class CarFeatureEngineer:
    """
    Feature engineering pipeline tailored for the Car Evaluation dataset.
    Extracts ordinal values, interaction terms, hierarchical boundary flags,
    and categorical cross-features with strict isolation against leakage.
    """

    def __init__(self):
        # Calibrated ordinal scale mappings
        self.buying_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
        self.maint_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
        self.doors_map = {"2": 2, "3": 3, "4": 4, "5more": 5}
        self.persons_map = {"2": 2, "4": 4, "more": 6}
        self.lug_boot_map = {"small": 0, "med": 1, "big": 2}
        self.safety_map = {"low": 0, "med": 1, "high": 2}

        self.freq_enc_maps = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _extract_base_ordinal_and_domain_features(self, df):
        """Construct ordinal and hierarchical interaction features."""
        df_feat = pd.DataFrame(index=df.index)

        # 1. Monotonic Ordinal Encodings
        df_feat["buying_num"] = (
            df["buying"].astype(str).map(self.buying_map).fillna(1.5).astype(float)
        )
        df_feat["maint_num"] = (
            df["maint"].astype(str).map(self.maint_map).fillna(1.5).astype(float)
        )
        df_feat["doors_num"] = (
            df["doors"].astype(str).map(self.doors_map).fillna(3.5).astype(float)
        )
        df_feat["persons_num"] = (
            df["persons"].astype(str).map(self.persons_map).fillna(4.0).astype(float)
        )
        df_feat["lug_boot_num"] = (
            df["lug_boot"].astype(str).map(self.lug_boot_map).fillna(1.0).astype(float)
        )
        df_feat["safety_num"] = (
            df["safety"].astype(str).map(self.safety_map).fillna(1.0).astype(float)
        )

        # 2. Hierarchical Cost & Capacity Composites
        df_feat["total_cost"] = df_feat["buying_num"] + df_feat["maint_num"]
        df_feat["cost_product"] = df_feat["buying_num"] * df_feat["maint_num"]
        df_feat["cost_diff"] = df_feat["buying_num"] - df_feat["maint_num"]

        df_feat["capacity_sum"] = df_feat["doors_num"] + df_feat["persons_num"]
        df_feat["capacity_prod"] = df_feat["doors_num"] * df_feat["persons_num"]
        df_feat["lug_person_ratio"] = df_feat["lug_boot_num"] / (
            df_feat["persons_num"] + 1e-5
        )

        # 3. Safety & Utility Interaction Terms
        df_feat["safety_cost_ratio"] = (df_feat["safety_num"] + 1.0) / (
            df_feat["total_cost"] + 1.0
        )
        df_feat["safety_person_prod"] = df_feat["safety_num"] * df_feat["persons_num"]
        df_feat["safety_lug_prod"] = df_feat["safety_num"] * df_feat["lug_boot_num"]
        df_feat["overall_utility"] = (
            df_feat["safety_num"] * 2.0
            + df_feat["persons_num"]
            + df_feat["lug_boot_num"]
        ) - df_feat["total_cost"]

        # 4. Critical Logical Boundary Flags (Bottleneck Decision Logic)
        df_feat["is_safety_low"] = (df["safety"].astype(str) == "low").astype(float)
        df_feat["is_persons_2"] = (df["persons"].astype(str) == "2").astype(float)
        df_feat["is_deterministic_unacc"] = (
            (df["safety"].astype(str) == "low") | (df["persons"].astype(str) == "2")
        ).astype(float)
        df_feat["is_high_cost"] = (
            (df_feat["buying_num"] >= 2) & (df_feat["maint_num"] >= 2)
        ).astype(float)
        df_feat["is_vgood_eligible"] = (
            (df["safety"].astype(str) == "high")
            & (df["persons"].astype(str) != "2")
            & (df["buying"].astype(str).isin(["low", "med"]))
            & (df["maint"].astype(str).isin(["low", "med"]))
        ).astype(float)

        # 5. Categorical Interaction Combinations
        cat_combos = {
            "cat_buying_maint": df["buying"].astype(str)
            + "_"
            + df["maint"].astype(str),
            "cat_safety_persons": df["safety"].astype(str)
            + "_"
            + df["persons"].astype(str),
            "cat_safety_lug": df["safety"].astype(str)
            + "_"
            + df["lug_boot"].astype(str),
            "cat_buying_safety": df["buying"].astype(str)
            + "_"
            + df["safety"].astype(str),
            "cat_doors_persons": df["doors"].astype(str)
            + "_"
            + df["persons"].astype(str),
        }
        df_combos = pd.DataFrame(cat_combos, index=df.index)

        return df_feat, df_combos

    def fit(self, df_train):
        """Fit frequency encoders and numeric scalers strictly on training data."""
        _, df_combos = self._extract_base_ordinal_and_domain_features(df_train)

        raw_cats = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        for col in raw_cats:
            freq = df_train[col].astype(str).value_counts(normalize=True).to_dict()
            self.freq_enc_maps[f"{col}_freq"] = freq

        for col in df_combos.columns:
            freq = df_combos[col].value_counts(normalize=True).to_dict()
            self.freq_enc_maps[f"{col}_freq"] = freq

        df_feat, _ = self._extract_base_ordinal_and_domain_features(df_train)
        self.scaler.fit(df_feat)
        self.feature_names_ = list(df_feat.columns) + [
            f"{k}" for k in self.freq_enc_maps.keys()
        ]
        self.is_fitted = True
        return self

    def transform(self, df):
        """Transform dataset applying learned stateful transformations."""
        if not self.is_fitted:
            raise RuntimeError("CarFeatureEngineer must be fitted before transform.")

        df_feat, df_combos = self._extract_base_ordinal_and_domain_features(df)

        raw_cats = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        for col in raw_cats:
            mapping = self.freq_enc_maps[f"{col}_freq"]
            df_feat[f"{col}_freq"] = (
                df[col].astype(str).map(mapping).fillna(0.0).astype(float)
            )

        for col in df_combos.columns:
            mapping = self.freq_enc_maps[f"{col}_freq"]
            df_feat[f"{col}_freq"] = (
                df_combos[col].map(mapping).fillna(0.0).astype(float)
            )

        return df_feat


# =========================================================================
# 2. Model Architectures & Builders
# =========================================================================


def build_lightgbm_classifier(
    num_classes: int = 4,
    n_estimators: int = 500,
    learning_rate: float = 0.04,
    max_depth: int = 6,
    num_leaves: int = 31,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> lgb.LGBMClassifier:
    """Build an sklearn-compatible LightGBM Classifier instance."""
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
        verbose=-1,
    )


def build_xgboost_classifier(
    num_classes: int = 4,
    n_estimators: int = 500,
    learning_rate: float = 0.04,
    max_depth: int = 5,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Build an sklearn-compatible XGBoost Classifier instance."""
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
    )


class ResidualBlock(nn.Module):
    """Residual Tabular Dense Block with LayerNorm, GELU, and Dropout regularization."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNetClassifier(nn.Module):
    """
    Deep Residual MLP architecture tailored for tabular multi-class classification.
    Processes continuous and encoded categorical features with non-linear projections
    and skip connections to prevent gradient vanishing on small-to-medium datasets.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def get_loss_function(class_weights=None, label_smoothing: float = 0.03) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    min_lr: float = 1e-6,
):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "LayerNorm" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optimizer, scheduler


def train_tabular_resnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 2e-3,
    device: str = "cpu",
):
    """Train deep TabularResNet model with cosine decay and early stopping."""
    input_dim = X_train.shape[1]

    tensor_x_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    tensor_x_val = torch.tensor(X_val, dtype=torch.float32)
    tensor_y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = TabularResNetClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = [total_samples / (num_classes * max(1, count)) for count in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = get_loss_function(
        class_weights=class_weights_tensor, label_smoothing=0.03
    )
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr=lr, weight_decay=1e-4, epochs=epochs
    )

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(tensor_x_val.to(device))
            val_loss = criterion(val_logits, tensor_y_val.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    return model


# =========================================================================
# 3. Full Training, Evaluation & Submission Pipeline
# =========================================================================


def main():
    seed_everything(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading datasets from ./input ...")
    df_train_raw, df_test_raw = load_raw_data("./input")
    target_col = identify_target_column(df_train_raw)

    print(
        f"Loaded: Train shape={df_train_raw.shape}, Test shape={df_test_raw.shape}, Target='{target_col}'"
    )

    X_raw = df_train_raw.drop(columns=[target_col]).copy()
    y_raw = df_train_raw[target_col].copy()

    id_col = "id" if "id" in df_test_raw.columns else None
    if id_col and id_col in X_raw.columns:
        X_raw = X_raw.drop(columns=[id_col])

    if id_col:
        test_ids = df_test_raw[id_col].values
        X_test_raw = df_test_raw.drop(columns=[id_col]).copy()
    else:
        test_ids = np.arange(len(df_test_raw))
        X_test_raw = df_test_raw.copy()

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_raw.astype(str))
    target_classes = list(target_encoder.classes_)
    num_classes = len(target_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dedicated 80/20 Hold-out Validation Split
    train_idx_holdout, val_idx_holdout = train_test_split(
        np.arange(len(df_train_raw)),
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    fe_holdout = CarFeatureEngineer()
    fe_holdout.fit(X_raw.iloc[train_idx_holdout])

    X_train_holdout = fe_holdout.transform(X_raw.iloc[train_idx_holdout]).values
    y_train_holdout = y_encoded[train_idx_holdout]
    X_val_holdout = fe_holdout.transform(X_raw.iloc[val_idx_holdout]).values
    y_val_holdout = y_encoded[val_idx_holdout]

    lgb_holdout = build_lightgbm_classifier(
        num_classes=num_classes,
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        class_weight="balanced",
    )
    lgb_holdout.fit(
        X_train_holdout,
        y_train_holdout,
        eval_set=[(X_val_holdout, y_val_holdout)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    xgb_holdout = build_xgboost_classifier(
        num_classes=num_classes,
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        random_state=42,
    )
    xgb_holdout.set_params(early_stopping_rounds=50)
    xgb_holdout.fit(
        X_train_holdout,
        y_train_holdout,
        eval_set=[(X_val_holdout, y_val_holdout)],
        verbose=False,
    )

    nn_holdout = train_tabular_resnet(
        X_train_holdout,
        y_train_holdout,
        X_val_holdout,
        y_val_holdout,
        num_classes=num_classes,
        epochs=70,
        batch_size=32,
        lr=2e-3,
        device=device,
    )

    holdout_prob_lgb = lgb_holdout.predict_proba(X_val_holdout)
    holdout_prob_xgb = xgb_holdout.predict_proba(X_val_holdout)
    with torch.no_grad():
        tensor_va_h = torch.tensor(X_val_holdout, dtype=torch.float32).to(device)
        holdout_prob_nn = nn_holdout.predict_proba(tensor_va_h).cpu().numpy()

    holdout_ensemble_probs = (
        0.45 * holdout_prob_lgb + 0.40 * holdout_prob_xgb + 0.15 * holdout_prob_nn
    )
    holdout_final_preds = np.argmax(holdout_ensemble_probs, axis=1)
    final_validation_score = accuracy_score(y_val_holdout, holdout_final_preds)

    # 2. Stratified 5-Fold Cross-Validation for Full Out-of-Fold & Test Inference
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros((len(X_raw), num_classes))
    oof_preds_xgb = np.zeros((len(X_raw), num_classes))
    oof_preds_nn = np.zeros((len(X_raw), num_classes))

    test_preds_lgb = np.zeros((len(X_test_raw), num_classes))
    test_preds_xgb = np.zeros((len(X_test_raw), num_classes))
    test_preds_nn = np.zeros((len(X_test_raw), num_classes))

    n_folds = 5
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_encoded)):
        fe_fold = CarFeatureEngineer()
        fe_fold.fit(X_raw.iloc[train_idx])
        X_tr = fe_fold.transform(X_raw.iloc[train_idx]).values
        y_tr = y_encoded[train_idx]
        X_va = fe_fold.transform(X_raw.iloc[val_idx]).values
        y_va = y_encoded[val_idx]
        X_test = fe_fold.transform(X_test_raw).values

        # LightGBM
        lgb_model = build_lightgbm_classifier(
            num_classes=num_classes,
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            random_state=42 + fold,
            class_weight="balanced",
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_va)
        test_preds_lgb += lgb_model.predict_proba(X_test) / n_folds

        # XGBoost
        xgb_model = build_xgboost_classifier(
            num_classes=num_classes,
            n_estimators=400,
            learning_rate=0.04,
            max_depth=5,
            random_state=42 + fold,
        )
        xgb_model.set_params(early_stopping_rounds=50)
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_va)
        test_preds_xgb += xgb_model.predict_proba(X_test) / n_folds

        # TabularResNet
        nn_model = train_tabular_resnet(
            X_tr,
            y_tr,
            X_va,
            y_va,
            num_classes=num_classes,
            epochs=70,
            batch_size=32,
            lr=2e-3,
            device=device,
        )
        with torch.no_grad():
            tensor_va = torch.tensor(X_va, dtype=torch.float32).to(device)
            tensor_te = torch.tensor(X_test, dtype=torch.float32).to(device)
            oof_preds_nn[val_idx] = nn_model.predict_proba(tensor_va).cpu().numpy()
            test_preds_nn += nn_model.predict_proba(tensor_te).cpu().numpy() / n_folds

    oof_ensemble_probs = (
        0.45 * oof_preds_lgb + 0.40 * oof_preds_xgb + 0.15 * oof_preds_nn
    )
    oof_ensemble_acc = accuracy_score(y_encoded, np.argmax(oof_ensemble_probs, axis=1))
    print(f"5-Fold Cross-Validation OOF Accuracy: {oof_ensemble_acc:.5f}")

    # 3. Final Test Prediction & Submission Generation
    final_test_probs = (
        0.45 * test_preds_lgb + 0.40 * test_preds_xgb + 0.15 * test_preds_nn
    )
    final_test_pred_indices = np.argmax(final_test_probs, axis=1)
    final_test_labels = target_encoder.inverse_transform(final_test_pred_indices)

    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            "car": final_test_labels,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission successfully saved to {submission_path}")
    print(f"Holdout Validation Accuracy: {final_validation_score:.5f}")
    print(f"Final Validation Score: {final_validation_score:.5f}")


if __name__ == "__main__":
    main()