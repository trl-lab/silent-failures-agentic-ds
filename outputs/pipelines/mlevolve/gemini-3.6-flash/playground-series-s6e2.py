import json
import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Set seed for deterministic reproducibility across frameworks
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =============================================================================
# 1. PYTORCH TABULAR RESNET WITH SWIGLU ACTIVATION
# =============================================================================


class SwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features * 2)

    def forward(self, x):
        x_proj, gate = self.fc(x).chunk(2, dim=-1)
        return x_proj * F.silu(gate)


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.swiglu = SwiGLU(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.swiglu(out)
        out = self.norm2(out)
        out = self.dropout(self.fc2(out))
        return residual + out


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits.squeeze(-1)


class TabularDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# =============================================================================
# 2. FEATURE PIPELINE LOADER & PREPROCESSING
# =============================================================================


def build_clinical_features(df):
    df = df.copy()
    expected_max_hr = 220.0 - df["Age"]
    df["hr_reserve"] = expected_max_hr - df["Max HR"]
    df["hr_ratio"] = df["Max HR"] / (expected_max_hr + 1e-5)
    df["rate_pressure_product"] = df["BP"] * df["Max HR"]
    df["st_slope_prod"] = df["ST depression"] * df["Slope of ST"]
    df["st_depression_per_slope"] = df["ST depression"] / (df["Slope of ST"] + 1e-5)
    df["vessels_thallium_prod"] = df["Number of vessels fluro"] * df["Thallium"]
    df["vessels_per_age"] = df["Number of vessels fluro"] / (df["Age"] + 1e-5)
    df["duke_proxy"] = (
        df["Max HR"] - (5.0 * df["ST depression"]) - (4.0 * df["Exercise angina"])
    )
    df["chol_age_ratio"] = df["Cholesterol"] / (df["Age"] + 1e-5)
    df["bp_age_ratio"] = df["BP"] / (df["Age"] + 1e-5)

    high_chol = (df["Cholesterol"] > 240).astype(int)
    high_bp = (df["BP"] > 130).astype(int)
    high_age = (df["Age"] > 55).astype(int)
    has_angina = df["Exercise angina"].astype(int)
    has_fbs = df["FBS over 120"].astype(int)
    df["risk_factor_count"] = high_chol + high_bp + high_age + has_angina + has_fbs
    df["age_decile"] = pd.qcut(df["Age"], q=5, labels=False, duplicates="drop")
    return df


def prepare_dataset():
    proc_train_path = "./working/processed_train.parquet"
    proc_test_path = "./working/processed_test.parquet"
    target_col = "Heart Disease"

    if os.path.exists(proc_train_path) and os.path.exists(proc_test_path):
        train_df = pd.read_parquet(proc_train_path)
        test_df = pd.read_parquet(proc_test_path)
    else:
        train_raw = pd.read_parquet("./input/train.parquet")
        test_raw = pd.read_parquet("./input/test.parquet")

        if target_col in train_raw.columns:
            if train_raw[target_col].dtype == object:
                target_map = {"Presence": 1, "Absence": 0}
                train_raw[target_col] = (
                    train_raw[target_col].map(target_map).fillna(0).astype(int)
                )

        y_tr = train_raw[target_col].values
        train_fe = build_clinical_features(train_raw.drop(columns=[target_col]))
        test_fe = build_clinical_features(test_raw)

        group_cols = ["age_decile", "Chest pain type"]
        num_aggregates = ["Max HR", "ST depression", "Cholesterol", "BP"]

        for g_col in group_cols:
            for n_col in num_aggregates:
                means = train_fe.groupby(g_col)[n_col].mean().to_dict()
                stds = train_fe.groupby(g_col)[n_col].std().to_dict()
                train_fe[f"{n_col}_mean_by_{g_col}"] = train_fe[g_col].map(means)
                train_fe[f"{n_col}_std_by_{g_col}"] = train_fe[g_col].map(stds)
                test_fe[f"{n_col}_mean_by_{g_col}"] = test_fe[g_col].map(means)
                test_fe[f"{n_col}_std_by_{g_col}"] = test_fe[g_col].map(stds)

        cat_cols = [
            "Chest pain type",
            "Thallium",
            "Slope of ST",
            "EKG results",
            "Number of vessels fluro",
        ]
        for col in cat_cols:
            freq = train_fe[col].value_counts(normalize=True).to_dict()
            train_fe[f"{col}_freq"] = train_fe[col].map(freq)
            test_fe[f"{col}_freq"] = test_fe[col].map(freq).fillna(0)

        feature_cols = [c for c in train_fe.columns if c not in ["id", target_col]]
        train_df = train_fe[feature_cols].copy()
        train_df[target_col] = y_tr
        if "id" in train_raw.columns:
            train_df["id"] = train_raw["id"]

        test_df = test_fe[feature_cols].copy()
        if "id" in test_raw.columns:
            test_df["id"] = test_raw["id"]

    return train_df, test_df


# =============================================================================
# 3. TRAIN PYTORCH FOLD
# =============================================================================


def train_pytorch_fold(X_tr_proc, y_tr, X_va_proc, y_va, X_te_proc, device, epochs=12):
    input_dim = X_tr_proc.shape[1]
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout=0.15
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    train_ds = TabularDataset(X_tr_proc, y_tr)
    val_ds = TabularDataset(X_va_proc)
    test_ds = TabularDataset(X_te_proc)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    best_val_auc = -1.0
    best_val_preds = None
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx in val_loader:
                bx = bx.to(device)
                probs = torch.sigmoid(model(bx)).cpu().numpy()
                val_preds_list.append(probs)

        val_preds = np.concatenate(val_preds_list)
        val_auc = roc_auc_score(y_va, val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_preds = val_preds
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for bx in test_loader:
            bx = bx.to(device)
            probs = torch.sigmoid(model(bx)).cpu().numpy()
            test_preds_list.append(probs)

    test_preds = np.concatenate(test_preds_list)
    return best_val_preds, test_preds


# =============================================================================
# 4. MAIN CROSS-VALIDATION, ENSEMBLE BLENDING & SUBMISSION GENERATION
# =============================================================================


def main():
    seed_everything(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    target_col = "Heart Disease"
    train_df, test_df = prepare_dataset()

    feature_cols = [c for c in train_df.columns if c not in ["id", target_col]]
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    X_test = test_df[feature_cols].values

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cb = np.zeros(len(X))
    oof_nn = np.zeros(len(X))

    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_cb = np.zeros(len(X_test))
    test_preds_nn = np.zeros(len(X_test))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. LightGBM Classifier
        model_lgb = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_child_samples=25,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
        test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits

        # 2. XGBoost Classifier
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
        test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

        # 3. CatBoost Classifier
        model_cb = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            early_stopping_rounds=50,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_idx] = model_cb.predict_proba(X_va)[:, 1]
        test_preds_cb += model_cb.predict_proba(X_test)[:, 1] / n_splits

        # 4. PyTorch TabularResNet
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_va_proc = scaler.transform(imputer.transform(X_va))
        X_te_proc = scaler.transform(imputer.transform(X_test))

        nn_val_preds, nn_test_preds = train_pytorch_fold(
            X_tr_proc, y_tr, X_va_proc, y_va, X_te_proc, device, epochs=12
        )
        oof_nn[val_idx] = nn_val_preds
        test_preds_nn += nn_test_preds / n_splits

        print(
            f"Fold {fold+1}/{n_splits} complete - LGB AUC:"
            f" {roc_auc_score(y_va, oof_lgb[val_idx]):.5f}"
        )

    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_nn]
    )

    def loss_func(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        blend_pred = np.dot(oof_matrix, weights)
        return -roc_auc_score(y, blend_pred)

    init_weights = [0.35, 0.30, 0.25, 0.10]
    bounds = [(0, 1) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

    res = minimize(
        loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )
    opt_weights = res.x / np.sum(res.x)

    oof_final = np.dot(oof_matrix, opt_weights)
    final_cv_score = roc_auc_score(y, oof_final)

    test_final = np.dot(test_matrix, opt_weights)

    sub = pd.DataFrame({"id": test_ids, "Heart Disease": test_final})
    sub.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_cv_score}")


if __name__ == "__main__":
    main()