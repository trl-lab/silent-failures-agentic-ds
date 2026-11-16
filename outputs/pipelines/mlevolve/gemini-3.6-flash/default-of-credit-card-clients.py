import os
import random
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ==========================================
# 1. Reproducibility & Environment Setup
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ==========================================
# 2. Data Preprocessing & Feature Engineering
# ==========================================
def load_and_preprocess_data():
    """Load train and test datasets, align column names, and prepare IDs."""
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    rename_dict = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "default payment next month",
    }

    train_df.rename(columns=rename_dict, inplace=True)
    test_df.rename(columns=rename_dict, inplace=True)

    train_df["id"] = np.arange(len(train_df))
    test_df["id"] = np.arange(len(test_df))

    return train_df, test_df


def engineer_features(df):
    """Perform domain-specific feature engineering."""
    df = df.copy()

    # 1. Clean categorical anomalies
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x in [0, 5, 6] else x)
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 3 if x == 0 else x)

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]
    pay_amt_cols = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    # 2. Multi-period Credit Utilization Ratios
    for i, col in enumerate(bill_cols, 1):
        df[f"utilization_{i}"] = df[col] / (df["LIMIT_BAL"] + 1.0)

    util_cols = [f"utilization_{i}" for i in range(1, 7)]
    df["utilization_mean"] = df[util_cols].mean(axis=1)
    df["utilization_max"] = df[util_cols].max(axis=1)
    df["utilization_min"] = df[util_cols].min(axis=1)
    df["utilization_std"] = df[util_cols].std(axis=1)
    df["utilization_momentum"] = df["utilization_1"] - df["utilization_6"]

    # 3. Payment-to-Bill Ratios, Satisfaction Completeness & Deficits across months
    for i in range(1, 6):
        df[f"pay_ratio_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].abs() + 1.0)
        df[f"pay_sat_completeness_{i}"] = df[f"PAY_AMT{i}"] / np.maximum(df[f"BILL_AMT{i+1}"], 1.0)
        df[f"deficit_{i}"] = df[f"BILL_AMT{i}"] - df[f"PAY_AMT{i}"]
    df["deficit_6"] = df["BILL_AMT6"] - df["PAY_AMT6"]
    df["pay_ratio_6"] = df["PAY_AMT6"] / (df["BILL_AMT6"].abs() + 1.0)
    df["pay_sat_completeness_6"] = df["PAY_AMT6"] / np.maximum(df["BILL_AMT6"], 1.0)

    # Payment-to-bill ratio acceleration
    for i in range(1, 6):
        df[f"pay_ratio_acc_{i}"] = df[f"pay_ratio_{i}"] - df[f"pay_ratio_{i+1}"]

    deficit_cols = [f"deficit_{i}" for i in range(1, 7)]
    pay_ratio_cols = [f"pay_ratio_{i}" for i in range(1, 7)]
    df["deficit_mean"] = df[deficit_cols].mean(axis=1)
    df["deficit_sum"] = df[deficit_cols].sum(axis=1)
    df["deficit_max"] = df[deficit_cols].max(axis=1)
    df["pay_ratio_mean"] = df[pay_ratio_cols].mean(axis=1)
    df["pay_ratio_min"] = df[pay_ratio_cols].min(axis=1)

    # 4. Payment Delay Trajectory Statistics & Temporal Velocity
    df["pay_delay_max"] = df[pay_cols].max(axis=1)
    df["pay_delay_min"] = df[pay_cols].min(axis=1)
    df["pay_delay_mean"] = df[pay_cols].mean(axis=1)
    df["pay_delay_std"] = df[pay_cols].std(axis=1)
    df["pay_delay_recent_trend"] = df["PAY_0"] - df["PAY_6"]
    df["pay_delay_latest_diff"] = df["PAY_0"] - df["PAY_2"]
    df["num_months_delayed"] = (df[pay_cols] > 0).sum(axis=1)
    df["num_months_prepaid"] = (df[pay_cols] < 0).sum(axis=1)

    # Temporal repayment velocity features and severe delinquency count
    df["pay_vel_0_2"] = df["PAY_0"] - df["PAY_2"]
    df["pay_vel_2_3"] = df["PAY_2"] - df["PAY_3"]
    df["pay_vel_3_4"] = df["PAY_3"] - df["PAY_4"]
    df["pay_vel_4_5"] = df["PAY_4"] - df["PAY_5"]
    df["pay_vel_5_6"] = df["PAY_5"] - df["PAY_6"]
    df["severe_delinquency_count"] = (df[pay_cols] >= 2).sum(axis=1)

    # 5. Bill and Payment Aggregations & Dynamics
    df["bill_amt_mean"] = df[bill_cols].mean(axis=1)
    df["bill_amt_std"] = df[bill_cols].std(axis=1)
    df["bill_amt_max"] = df[bill_cols].max(axis=1)
    df["bill_amt_min"] = df[bill_cols].min(axis=1)
    df["bill_amt_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    df["pay_amt_mean"] = df[pay_amt_cols].mean(axis=1)
    df["pay_amt_std"] = df[pay_amt_cols].std(axis=1)
    df["pay_amt_sum"] = df[pay_amt_cols].sum(axis=1)
    df["pay_amt_max"] = df[pay_amt_cols].max(axis=1)
    df["pay_amt_trend"] = df["PAY_AMT1"] - df["PAY_AMT6"]

    df["total_pay_to_bill_ratio"] = df["pay_amt_sum"] / (
        df[bill_cols].abs().sum(axis=1) + 1.0
    )

    # 6. Demographics & Categorical Interaction Terms
    df["age_bin"] = pd.cut(
        df["AGE"],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=False,
        include_lowest=True,
    )
    df["sex_edu"] = df["SEX"] * 10 + df["EDUCATION"]
    df["sex_marriage"] = df["SEX"] * 10 + df["MARRIAGE"]
    df["edu_marriage"] = df["EDUCATION"] * 10 + df["MARRIAGE"]
    df["pay_0_edu"] = (df["PAY_0"] + 10) * 10 + df["EDUCATION"]

    df["limit_per_age"] = df["LIMIT_BAL"] / (df["AGE"] + 1e-5)
    df["bill_per_age"] = df["bill_amt_mean"] / (df["AGE"] + 1e-5)

    # Demographic subgroup benchmark statistics & Temporal Delinquency Metrics
    exp_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
    df["exp_delinquency_index"] = (df[pay_cols].values * exp_weights).sum(axis=1)

    pay_pos = (df[pay_cols].values > 0).astype(int)
    streaks = np.zeros(len(df), dtype=int)
    for k in range(pay_pos.shape[1]):
        mask = (pay_pos[:, : k + 1].sum(axis=1) == k + 1)
        streaks[mask] = k + 1
    df["delay_streak_count"] = streaks

    # Payment severity interaction features (risk per credit unit)
    df["pay_0_limit_ratio"] = df["PAY_0"] / (df["LIMIT_BAL"] + 1.0)
    df["pay_delay_mean_limit_ratio"] = df["pay_delay_mean"] / (df["LIMIT_BAL"] + 1.0)
    df["pay_delay_max_limit_ratio"] = df["pay_delay_max"] / (df["LIMIT_BAL"] + 1.0)
    df["severe_delinq_limit_ratio"] = df["severe_delinquency_count"] / (df["LIMIT_BAL"] + 1.0)

    # 7. Non-linear Log Transformations
    df["log_limit_bal"] = np.log1p(np.maximum(0, df["LIMIT_BAL"]))
    df["log_pay_amt_sum"] = np.log1p(np.maximum(0, df["pay_amt_sum"]))
    df["log_bill_amt_mean"] = np.log1p(np.maximum(0, df["bill_amt_mean"]))

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def apply_group_benchmark_stats(train_df, test_df):
    """Compute demographic group statistics strictly on training data and map onto test data."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    group_cols = ["EDUCATION", "MARRIAGE", "age_bin"]
    bench_cols = ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "PAY_0"]

    tr_keys = pd.Series(list(zip(*[train_df[g] for g in group_cols])), index=train_df.index)
    te_keys = pd.Series(list(zip(*[test_df[g] for g in group_cols])), index=test_df.index)

    for col in bench_cols:
        grp_mean_dict = train_df.groupby(group_cols)[col].mean().to_dict()
        grp_std_dict = (train_df.groupby(group_cols)[col].std().fillna(0) + 1e-5).to_dict()

        global_mean = train_df[col].mean()
        global_std = train_df[col].std() + 1e-5

        tr_g_mean = tr_keys.map(grp_mean_dict).fillna(global_mean)
        tr_g_std = tr_keys.map(grp_std_dict).fillna(global_std)
        te_g_mean = te_keys.map(grp_mean_dict).fillna(global_mean)
        te_g_std = te_keys.map(grp_std_dict).fillna(global_std)

        train_df[f"{col}_group_mean_ratio"] = train_df[col] / (tr_g_mean + 1e-5)
        train_df[f"{col}_group_zscore"] = (train_df[col] - tr_g_mean) / tr_g_std

        test_df[f"{col}_group_mean_ratio"] = test_df[col] / (te_g_mean + 1e-5)
        test_df[f"{col}_group_zscore"] = (test_df[col] - te_g_mean) / te_g_std

    return train_df, test_df


def apply_leakage_free_encodings(train_df, test_df):
    """Apply frequency encodings fitted strictly on the training set."""
    cat_cols = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "age_bin",
        "sex_edu",
        "sex_marriage",
        "edu_marriage",
    ]

    for col in cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

    return train_df, test_df


def apply_oof_target_encoding(
    train_df, test_df, target_col, cat_cols, n_splits=10, seed=42, smoothing=10.0
):
    """Apply out-of-fold target encoding for categorical interactions."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    global_mean = train_df[target_col].mean()

    for col in cat_cols:
        te_col = f"{col}_te"
        train_df[te_col] = np.nan

        for tr_idx, val_idx in skf.split(train_df, train_df[target_col]):
            tr_fold = train_df.iloc[tr_idx]
            stats = tr_fold.groupby(col)[target_col].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            train_df.loc[train_df.index[val_idx], te_col] = (
                train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean)
            )

        full_stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        test_df[te_col] = test_df[col].map(full_smoothed).fillna(global_mean)

    return train_df, test_df


# ==========================================
# 3. Model Architecture Definitions
# ==========================================
class FeatureGatingModule(nn.Module):
    """Squeeze-and-Excitation feature attention for tabular inputs."""

    def __init__(self, in_features: int, reduction_ratio: int = 4):
        super(FeatureGatingModule, self).__init__()
        reduced_dim = max(8, in_features // reduction_ratio)
        self.fc1 = nn.Linear(in_features, reduced_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(reduced_dim, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.fc2(self.act(self.fc1(x))))
        return x * weights


class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)

        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class TabularResNet(nn.Module):
    """Deep Tabular ResNet with Feature Attention and Residual Skip Connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.3,
    ):
        super(TabularResNet, self).__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.gating = FeatureGatingModule(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.gating(x)
        x = F.gelu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x)
        logits = self.head(x)
        return logits.squeeze(-1)


class FocalLoss(nn.Module):
    """Focal Loss tailored for imbalanced binary credit default prediction."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


def build_pytorch_optimizer_and_scheduler(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 35
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    return optimizer, scheduler


# ==========================================
# 4. GBDT Hyperparameters & Factory
# ==========================================
def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 35,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_estimators": 1500,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.025,
        "max_depth": 6,
        "min_child_weight": 6,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_estimators": 1500,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.025,
        "depth": 6,
        "rsm": 0.70,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "iterations": 1500,
        "verbose": False,
        "thread_count": -1,
    }


class ModelFactory:

    @staticmethod
    def create_lgb(seed: int = 42) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(**get_lgb_params(seed))

    @staticmethod
    def create_xgb(seed: int = 42) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(**get_xgb_params(seed))

    @staticmethod
    def create_catboost(seed: int = 42) -> cb.CatBoostClassifier:
        return cb.CatBoostClassifier(**get_catboost_params(seed))

    @staticmethod
    def create_hist_gb(seed: int = 42) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.03,
            max_iter=600,
            l2_regularization=1.5,
            min_samples_leaf=30,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=30,
        )

    @staticmethod
    def create_nn(
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.3,
    ) -> nn.Module:
        return TabularResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        )


# ==========================================
# 5. Cross-Validation, Blending & Execution
# ==========================================
def main():
    seed = 42
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data()
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    train_df, test_df = apply_group_benchmark_stats(train_df, test_df)
    train_df, test_df = apply_leakage_free_encodings(train_df, test_df)

    target_col = "default payment next month"
    id_col = "id"

    # Out-of-fold target encoding for categorical interactions
    interaction_cats = ["PAY_0", "pay_0_edu", "sex_edu", "sex_marriage", "edu_marriage"]
    train_df, test_df = apply_oof_target_encoding(
        train_df, test_df, target_col, interaction_cats, n_splits=10, seed=seed
    )

    feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

    X = train_df[feature_cols].copy().fillna(0).values.astype(np.float32)
    y = train_df[target_col].copy().values.astype(int)
    X_test = test_df[feature_cols].copy().fillna(0).values.astype(np.float32)

    n_samples_train = len(train_df)
    n_samples_test = len(test_df)
    n_splits = 10

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_lgb = np.zeros(n_samples_train)
    oof_xgb = np.zeros(n_samples_train)
    oof_cat = np.zeros(n_samples_train)
    oof_hist = np.zeros(n_samples_train)
    oof_nn = np.zeros(n_samples_train)

    test_lgb = np.zeros(n_samples_test)
    test_xgb = np.zeros(n_samples_test)
    test_cat = np.zeros(n_samples_test)
    test_hist = np.zeros(n_samples_test)
    test_nn = np.zeros(n_samples_test)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. LightGBM
        lgb_model = ModelFactory.create_lgb(seed=seed + fold)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

        # 2. XGBoost
        xgb_model = ModelFactory.create_xgb(seed=seed + fold)
        xgb_model.set_params(early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_splits

        # 3. CatBoost
        cat_model = ModelFactory.create_catboost(seed=seed + fold)
        cat_model.set_params(early_stopping_rounds=50)
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits

        # 4. HistGradientBoosting
        hist_model = ModelFactory.create_hist_gb(seed=seed + fold)
        hist_model.fit(X_tr, y_tr)
        oof_hist[val_idx] = hist_model.predict_proba(X_va)[:, 1]
        test_hist += hist_model.predict_proba(X_test)[:, 1] / n_splits

        # 5. PyTorch TabularResNet
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_test_scaled = scaler.transform(X_test)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        nn_model = ModelFactory.create_nn(
            input_dim=X.shape[1], hidden_dim=256, num_blocks=3, dropout_rate=0.3
        ).to(device)

        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer, scheduler = build_pytorch_optimizer_and_scheduler(
            nn_model, lr=1e-3, weight_decay=1e-4, epochs=35
        )

        best_val_auc = 0.0
        best_model_weights = None

        for epoch in range(35):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_preds = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    logits = nn_model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)

            val_auc = roc_auc_score(y_va, np.array(val_preds))
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

        if best_model_weights is not None:
            nn_model.load_state_dict(best_model_weights)

        nn_model.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            oof_nn[val_idx] = torch.sigmoid(nn_model(va_tensor)).cpu().numpy()

            test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            test_nn += torch.sigmoid(nn_model(test_tensor)).cpu().numpy() / n_splits

        print(
            f"Fold {fold+1}/{n_splits} complete | LGB AUC: {roc_auc_score(y_va, oof_lgb[val_idx]):.4f} | "
            f"XGB AUC: {roc_auc_score(y_va, oof_xgb[val_idx]):.4f} | "
            f"Cat AUC: {roc_auc_score(y_va, oof_cat[val_idx]):.4f} | "
            f"Hist AUC: {roc_auc_score(y_va, oof_hist[val_idx]):.4f} | "
            f"NN AUC: {best_val_auc:.4f}"
        )

    # Rank-normalize out-of-fold and test predictions across base models prior to stacking
    oof_preds = [oof_lgb, oof_xgb, oof_cat, oof_hist, oof_nn]
    test_preds = [test_lgb, test_xgb, test_cat, test_hist, test_nn]

    oof_ranks = [rankdata(p) / len(p) for p in oof_preds]
    test_ranks = [rankdata(p) / len(p) for p in test_preds]

    # Regularized Logistic Regression Stacking Meta-Learner
    OOF_meta = np.column_stack(oof_ranks)
    Test_meta = np.column_stack(test_ranks)

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    meta_model.fit(OOF_meta, y)

    oof_blended = meta_model.predict_proba(OOF_meta)[:, 1]
    final_score = roc_auc_score(y, oof_blended)

    test_blended = meta_model.predict_proba(Test_meta)[:, 1]

    # Save submission file
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame(
        {
            "id": test_df[id_col].values.astype(int),
            "default payment next month": test_blended,
        }
    )
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()