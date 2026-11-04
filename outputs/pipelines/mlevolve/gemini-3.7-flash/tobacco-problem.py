import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# =========================================================================
# 0. Global Configuration and Reproducibility
# =========================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Feature Engineering
# =========================================================================
def engineer_features(df_input):
    df = df_input.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1.1 Add missingness counts and indicators
    df["row_missing_count"] = df.isnull().sum(axis=1).astype(float)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f"{col}_is_missing"] = df[col].isnull().astype(float)

    # 1.2 Identify binary indicator features and compute survey sum/mean
    binary_cols = [
        c
        for c in num_cols
        if df[c].dropna().isin([0, 1, 0.0, 1.0]).all() and df[c].nunique() <= 2
    ]
    if len(binary_cols) >= 2:
        df["binary_sum"] = df[binary_cols].sum(axis=1)
        df["binary_mean"] = df[binary_cols].mean(axis=1)

    # 1.3 Numeric aggregations
    pure_num_cols = [c for c in num_cols if c not in binary_cols]
    if len(pure_num_cols) >= 2:
        df["num_features_sum"] = df[pure_num_cols].sum(axis=1)
        df["num_features_mean"] = df[pure_num_cols].mean(axis=1)
        df["num_features_std"] = df[pure_num_cols].std(axis=1).fillna(0)
        df["num_features_max"] = df[pure_num_cols].max(axis=1)
        df["num_features_min"] = df[pure_num_cols].min(axis=1)
        df["num_features_nonzero_count"] = (
            (df[pure_num_cols] > 0).sum(axis=1).astype(float)
        )

    # 1.4 Log1p transformation on continuous positive columns
    for col in pure_num_cols:
        if (df[col] >= 0).all():
            df[f"{col}_log1p"] = np.log1p(df[col])

    return df


# =========================================================================
# 2. Deep Tabular Neural Network Architecture & Custom Losses
# =========================================================================
class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class LabelSmoothedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            smooth_targets = torch.full_like(probs, self.smoothing / num_classes)
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.smoothing + (self.smoothing / num_classes),
            )

        focal_weight = (1.0 - probs) ** self.gamma
        loss = -torch.sum(smooth_targets * focal_weight * log_probs, dim=-1)
        return loss.mean()


def build_neural_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 40,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    return optimizer, scheduler


# =========================================================================
# 3. GBDT Model Configuration Factories
# =========================================================================
def build_lgbm_model(num_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 15,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": -1,
    }


def build_catboost_model(num_classes: int, seed: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=800,
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=4.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


def build_xgboost_model(num_classes: int, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


def optimize_class_multipliers(oof_probs, y_true_idx, n_classes):
    init_c = np.ones(n_classes)

    def obj(c):
        c_pos = np.maximum(c, 1e-4)
        scaled_probs = oof_probs * c_pos
        preds = np.argmax(scaled_probs, axis=1)
        return -accuracy_score(y_true_idx, preds)

    res = minimize(
        obj,
        init_c,
        method="Nelder-Mead",
        options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-5},
    )
    best_c = np.maximum(res.x, 1e-4)
    raw_acc = accuracy_score(y_true_idx, np.argmax(oof_probs, axis=1))
    opt_acc = accuracy_score(y_true_idx, np.argmax(oof_probs * best_c, axis=1))
    if opt_acc >= raw_acc:
        return best_c
    return init_c


# =========================================================================
# 4. Main Execution Pipeline
# =========================================================================
def main():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "number_health_problems"
    id_col = "id" if "id" in test_df.columns else None

    y_train_full = train_df[target_col].copy()
    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_test_raw = test_df.copy()

    if id_col and id_col in X_train_raw.columns:
        X_train_raw = X_train_raw.drop(columns=[id_col])
    if id_col and id_col in X_test_raw.columns:
        test_ids = X_test_raw[id_col].copy()
        X_test_raw = X_test_raw.drop(columns=[id_col])
    else:
        test_ids = pd.Series(np.arange(len(X_test_raw)), name="id")

    # Apply Feature Engineering
    X_train_fe = engineer_features(X_train_raw)
    X_test_fe = engineer_features(X_test_raw)

    for col in X_train_fe.columns:
        if col not in X_test_fe.columns:
            X_test_fe[col] = 0.0
    for col in X_test_fe.columns:
        if col not in X_train_fe.columns:
            X_train_fe[col] = 0.0
    X_test_fe = X_test_fe[X_train_fe.columns]

    classes = np.sort(np.unique(y_train_full))
    n_classes = len(classes)
    n_samples = len(y_train_full)
    n_test = len(test_ids)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    cat_cols = X_train_fe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    oof_lgb_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_cb_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_xgb_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_et_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_rf_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_nn_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

    test_lgb_probs = np.zeros((n_test, n_classes), dtype=np.float64)
    test_cb_probs = np.zeros((n_test, n_classes), dtype=np.float64)
    test_xgb_probs = np.zeros((n_test, n_classes), dtype=np.float64)
    test_et_probs = np.zeros((n_test, n_classes), dtype=np.float64)
    test_rf_probs = np.zeros((n_test, n_classes), dtype=np.float64)
    test_nn_probs = np.zeros((n_test, n_classes), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y_train_full)):
        X_tr = X_train_fe.iloc[train_idx].copy()
        y_tr_raw = y_train_full.iloc[train_idx].copy()
        X_va = X_train_fe.iloc[val_idx].copy()
        y_va_raw = y_train_full.iloc[val_idx].copy()
        X_te = X_test_fe.copy()

        y_tr_idx = np.array([class_to_idx[y] for y in y_tr_raw])
        y_va_idx = np.array([class_to_idx[y] for y in y_va_raw])

        # Strict split-isolated categorical encoding
        if len(cat_cols) > 0:
            cat_imputer_values = {
                c: X_tr[c].mode()[0] if not X_tr[c].mode().empty else "missing"
                for c in cat_cols
            }
            for c in cat_cols:
                X_tr[c] = X_tr[c].fillna(cat_imputer_values[c]).astype(str)
                X_va[c] = X_va[c].fillna(cat_imputer_values[c]).astype(str)
                X_te[c] = X_te[c].fillna(cat_imputer_values[c]).astype(str)

                freq_map = X_tr[c].value_counts(normalize=True).to_dict()
                X_tr[f"{c}_freq"] = X_tr[c].map(freq_map).fillna(0.0)
                X_va[f"{c}_freq"] = X_va[c].map(freq_map).fillna(0.0)
                X_te[f"{c}_freq"] = X_te[c].map(freq_map).fillna(0.0)

            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr[cat_cols] = oe.fit_transform(X_tr[cat_cols])
            X_va[cat_cols] = oe.transform(X_va[cat_cols])
            X_te[cat_cols] = oe.transform(X_te[cat_cols])

        # Strict split-isolated numeric median imputation
        all_cols = list(X_tr.columns)
        medians = X_tr[all_cols].median()
        X_tr[all_cols] = X_tr[all_cols].fillna(medians)
        X_va[all_cols] = X_va[all_cols].fillna(medians)
        X_te[all_cols] = X_te[all_cols].fillna(medians)

        # Standard scaling for neural network
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
        )
        X_va_scaled = pd.DataFrame(
            scaler.transform(X_va), columns=X_va.columns, index=X_va.index
        )
        X_te_scaled = pd.DataFrame(
            scaler.transform(X_te), columns=X_te.columns, index=X_te.index
        )

        # 1. Train LightGBM
        lgb_params = build_lgbm_model(num_classes=n_classes, seed=SEED + fold)
        dtrain = lgb.Dataset(X_tr, label=y_tr_idx)
        dval = lgb.Dataset(X_va, label=y_va_idx, reference=dtrain)
        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        lgb_val_pred = lgb_model.predict(X_va)
        lgb_test_pred = lgb_model.predict(X_te)
        oof_lgb_probs[val_idx] = lgb_val_pred
        test_lgb_probs += lgb_test_pred / N_SPLITS

        # 2. Train CatBoost
        cb_model = build_catboost_model(num_classes=n_classes, seed=SEED + fold)
        cb_model.fit(
            X_tr,
            y_tr_idx,
            eval_set=(X_va, y_va_idx),
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_val_pred = cb_model.predict_proba(X_va)
        cb_test_pred = cb_model.predict_proba(X_te)
        oof_cb_probs[val_idx] = cb_val_pred
        test_cb_probs += cb_test_pred / N_SPLITS

        # 3. Train XGBoost
        xgb_model = build_xgboost_model(num_classes=n_classes, seed=SEED + fold)
        xgb_model.fit(X_tr, y_tr_idx, eval_set=[(X_va, y_va_idx)], verbose=False)
        xgb_val_pred = xgb_model.predict_proba(X_va)
        xgb_test_pred = xgb_model.predict_proba(X_te)
        oof_xgb_probs[val_idx] = xgb_val_pred
        test_xgb_probs += xgb_test_pred / N_SPLITS

        # 3.1 Train ExtraTrees and Random Forest
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr_idx)
        et_val_pred = et_model.predict_proba(X_va)
        et_test_pred = et_model.predict_proba(X_te)
        oof_et_probs[val_idx] = et_val_pred
        test_et_probs += et_test_pred / N_SPLITS

        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        rf_model.fit(X_tr, y_tr_idx)
        rf_val_pred = rf_model.predict_proba(X_va)
        rf_test_pred = rf_model.predict_proba(X_te)
        oof_rf_probs[val_idx] = rf_val_pred
        test_rf_probs += rf_test_pred / N_SPLITS

        # 4. Train Tabular ResNet
        input_dim = X_tr_scaled.shape[1]
        nn_model = TabularResNet(
            input_dim=input_dim,
            num_classes=n_classes,
            hidden_dim=128,
            num_blocks=2,
            dropout_rate=0.2,
        ).to(device)

        criterion = LabelSmoothedFocalLoss(gamma=1.5, label_smoothing=0.05)
        optimizer, scheduler = build_neural_optimizer_and_scheduler(
            nn_model, lr=1e-3, weight_decay=1e-4, num_epochs=40
        )

        tr_tensor_x = torch.tensor(X_tr_scaled.values, dtype=torch.float32)
        tr_tensor_y = torch.tensor(y_tr_idx, dtype=torch.long)
        va_tensor_x = torch.tensor(X_va_scaled.values, dtype=torch.float32)
        va_tensor_y = torch.tensor(y_va_idx, dtype=torch.long)
        te_tensor_x = torch.tensor(X_te_scaled.values, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            TensorDataset(tr_tensor_x, tr_tensor_y),
            batch_size=64,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(va_tensor_x, va_tensor_y),
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(40):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = nn_model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item() * len(by)
                    val_samples += len(by)
            avg_val_loss = val_loss / max(val_samples, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        if best_weights is not None:
            nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

        nn_model.eval()
        with torch.no_grad():
            va_logits = nn_model(va_tensor_x.to(device))
            nn_val_pred = (
                torch.softmax(va_logits, dim=-1).cpu().numpy().astype(np.float64)
            )
            te_logits = nn_model(te_tensor_x)
            nn_test_pred = (
                torch.softmax(te_logits, dim=-1).cpu().numpy().astype(np.float64)
            )

        oof_nn_probs[val_idx] = nn_val_pred
        test_nn_probs += nn_test_pred / N_SPLITS

        fold_ensemble_val = (
            0.25 * lgb_val_pred
            + 0.25 * cb_val_pred
            + 0.15 * xgb_val_pred
            + 0.15 * et_val_pred
            + 0.10 * rf_val_pred
            + 0.10 * nn_val_pred
        )
        fold_acc = accuracy_score(
            y_va_raw, [classes[i] for i in np.argmax(fold_ensemble_val, axis=1)]
        )
        print(f"Fold {fold} Ensemble Accuracy: {fold_acc:.5f}")

    # Final Ensemble Aggregation & Probability Multiplier Optimization
    y_train_idx_full = np.array([class_to_idx[y] for y in y_train_full])

    final_oof_probs = (
        0.25 * oof_lgb_probs
        + 0.25 * oof_cb_probs
        + 0.15 * oof_xgb_probs
        + 0.15 * oof_et_probs
        + 0.10 * oof_rf_probs
        + 0.10 * oof_nn_probs
    )
    final_test_probs = (
        0.25 * test_lgb_probs
        + 0.25 * test_cb_probs
        + 0.15 * test_xgb_probs
        + 0.15 * test_et_probs
        + 0.10 * test_rf_probs
        + 0.10 * test_nn_probs
    )

    best_multipliers = optimize_class_multipliers(
        final_oof_probs, y_train_idx_full, n_classes
    )
    oof_final_pred_indices = np.argmax(final_oof_probs * best_multipliers, axis=1)
    oof_final_predictions = np.array([classes[i] for i in oof_final_pred_indices])
    final_accuracy = accuracy_score(y_train_full, oof_final_predictions)

    test_final_pred_indices = np.argmax(final_test_probs * best_multipliers, axis=1)
    test_final_predictions = [classes[i] for i in test_final_pred_indices]

    submission_df = pd.DataFrame(
        {
            "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
            "number_health_problems": test_final_predictions,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Submission shape: {submission_df.shape}")
    print(f"Final Validation Score: {final_accuracy}")


if __name__ == "__main__":
    main()
