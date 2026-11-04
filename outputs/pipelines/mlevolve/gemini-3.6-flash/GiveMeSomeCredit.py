import json
import math
import os
import sys
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# =============================================================================
print("Loading raw datasets from ./input...")
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "SeriousDlqin2yrs"

# Preserve test ID for final submission formatting
if "id" in test_raw.columns:
    test_ids = test_raw["id"].values
elif "Unnamed: 0" in test_raw.columns:
    test_ids = test_raw["Unnamed: 0"].values
else:
    test_ids = np.arange(len(test_raw))


def extract_domain_features(df):
    """Engineer core domain features for tabular credit scoring."""
    data = df.copy()

    for col in ["id", "Unnamed: 0", target_col]:
        if col in data.columns:
            data = data.drop(columns=[col])

    # 1. Age Cleaning & Features
    data["age_clean"] = data["age"].apply(
        lambda x: np.nan if (x < 18 or x > 100) else x
    )

    # 2. Delinquency Sentinel Codes (96 and 98 indicate missing/extreme statuses)
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    for col in delinq_cols:
        data[f"{col}_is_sentinel"] = (data[col] >= 96).astype(float)
        data[f"{col}_clean"] = data[col].apply(lambda x: np.nan if x >= 96 else x)

    clean_delinq_cols = [f"{col}_clean" for col in delinq_cols]
    data["total_delinquent_clean"] = data[clean_delinq_cols].sum(axis=1)
    data["weighted_delinquency"] = (
        1.0 * data["NumberOfTime30-59DaysPastDueNotWorse_clean"].fillna(0)
        + 2.0 * data["NumberOfTime60-89DaysPastDueNotWorse_clean"].fillna(0)
        + 3.0 * data["NumberOfTimes90DaysLate_clean"].fillna(0)
    )
    data["has_any_delinquency"] = (data["total_delinquent_clean"] > 0).astype(float)

    # 3. Revolving Utilization Features
    data["util_over_1"] = (data["RevolvingUtilizationOfUnsecuredLines"] > 1.0).astype(
        float
    )
    data["util_over_10"] = (data["RevolvingUtilizationOfUnsecuredLines"] > 10.0).astype(
        float
    )
    data["util_is_zero"] = (data["RevolvingUtilizationOfUnsecuredLines"] == 0.0).astype(
        float
    )
    data["log_utilization"] = np.log1p(
        data["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0, upper=100)
    )

    # 4. Income & Debt Ratio Disambiguation
    data["income_isna"] = data["MonthlyIncome"].isna().astype(float)
    data["income_is_zero"] = (data["MonthlyIncome"] == 0).astype(float)

    has_income = (~data["MonthlyIncome"].isna()) & (data["MonthlyIncome"] > 0)
    data["implied_monthly_debt"] = np.where(
        has_income, data["DebtRatio"] * data["MonthlyIncome"], data["DebtRatio"]
    )
    data["implied_disposable_income"] = np.where(
        has_income, data["MonthlyIncome"] - data["implied_monthly_debt"], np.nan
    )
    data["log_monthly_income"] = np.log1p(data["MonthlyIncome"].clip(lower=0))
    data["log_implied_monthly_debt"] = np.log1p(
        data["implied_monthly_debt"].clip(lower=0)
    )

    # Dependents & Ratios
    data["dependents_clean"] = data["NumberOfDependents"].fillna(0)
    data["dependents_isna"] = data["NumberOfDependents"].isna().astype(float)
    data["has_dependents"] = (data["dependents_clean"] > 0).astype(float)
    data["income_per_dependent"] = data["MonthlyIncome"] / (
        data["dependents_clean"] + 1.0
    )
    data["debt_per_dependent"] = data["implied_monthly_debt"] / (
        data["dependents_clean"] + 1.0
    )

    # 5. Open Lines & Real Estate Ratios
    data["total_credit_lines"] = (
        data["NumberOfOpenCreditLinesAndLoans"] + data["NumberRealEstateLoansOrLines"]
    )
    data["real_estate_ratio"] = data["NumberRealEstateLoansOrLines"] / (
        data["total_credit_lines"] + 1e-5
    )
    data["lines_per_adult_year"] = data["total_credit_lines"] / np.maximum(
        data["age_clean"].fillna(45) - 17, 1.0
    )
    data["income_per_credit_line"] = data["MonthlyIncome"] / (
        data["NumberOfOpenCreditLinesAndLoans"] + 1.0
    )

    # 6. Delinquency and Utilization Interactions
    data["delinq_per_open_line"] = data["total_delinquent_clean"] / (
        data["NumberOfOpenCreditLinesAndLoans"] + 1.0
    )
    data["late90_ratio"] = data["NumberOfTimes90DaysLate_clean"] / (
        data["total_delinquent_clean"] + 1e-5
    )
    data["util_times_total_lines"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] * data["total_credit_lines"]
    )

    return data


print("Engineering domain features...")
df_train_feat = extract_domain_features(train_raw)
df_test_feat = extract_domain_features(test_raw)

# Fit age decile bins on training data only
train_age = df_train_feat["age_clean"].fillna(df_train_feat["age_clean"].median())
age_bins, bin_edges = pd.qcut(
    train_age, q=10, labels=False, duplicates="drop", retbins=True
)
df_train_feat["age_group"] = age_bins

bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
df_test_feat["age_group"] = pd.cut(
    df_test_feat["age_clean"].fillna(train_age.median()), bins=bin_edges, labels=False
)

df_train_feat["dep_group"] = df_train_feat["dependents_clean"].clip(upper=4).astype(int)
df_test_feat["dep_group"] = df_test_feat["dependents_clean"].clip(upper=4).astype(int)

df_train_feat["age_dep_group"] = (
    df_train_feat["age_group"].astype(str) + "_" + df_train_feat["dep_group"].astype(str)
)
df_test_feat["age_dep_group"] = (
    df_test_feat["age_group"].astype(str) + "_" + df_test_feat["dep_group"].astype(str)
)

group_agg = (
    df_train_feat.groupby("age_dep_group")[
        ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "log_utilization", "log_implied_monthly_debt"]
    ]
    .agg(["median", "std"])
)
group_agg.columns = [f"{c}_{stat}" for c, stat in group_agg.columns]

df_train_feat = df_train_feat.merge(group_agg, on="age_dep_group", how="left")
df_test_feat = df_test_feat.merge(group_agg, on="age_dep_group", how="left")

group_stats = (
    df_train_feat.groupby("age_group")[["log_utilization", "log_implied_monthly_debt"]]
    .mean()
    .rename(
        columns={
            "log_utilization": "age_grp_util_mean",
            "log_implied_monthly_debt": "age_grp_debt_mean",
        }
    )
)

df_train_feat = df_train_feat.merge(group_stats, on="age_group", how="left")
df_test_feat = df_test_feat.merge(group_stats, on="age_group", how="left")

df_train_feat["util_diff_from_age_group"] = (
    df_train_feat["log_utilization"] - df_train_feat["age_grp_util_mean"]
)
df_test_feat["util_diff_from_age_group"] = (
    df_test_feat["log_utilization"] - df_test_feat["age_grp_util_mean"]
)

df_train_feat["debt_diff_from_age_group"] = (
    df_train_feat["log_implied_monthly_debt"] - df_train_feat["age_grp_debt_mean"]
)
df_test_feat["debt_diff_from_age_group"] = (
    df_test_feat["log_implied_monthly_debt"] - df_test_feat["age_grp_debt_mean"]
)

feature_cols = [c for c in df_train_feat.columns if c not in ["age_group", "dep_group", "age_dep_group"]]
train_medians = df_train_feat[feature_cols].median()

df_train_imputed = df_train_feat[feature_cols].fillna(train_medians)
df_test_imputed = df_test_feat[feature_cols].fillna(train_medians)

y_all = train_raw[target_col].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(len(df_train_imputed), dtype=int)

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df_train_imputed, y_all)):
    folds[va_idx] = fold_idx

X_all = df_train_imputed.values
X_test = df_test_imputed.values
num_features = len(feature_cols)
num_folds = 5

print(
    f"Preprocessed features: {num_features}, Train rows: {len(X_all)}, Test rows: {len(X_test)}"
)


# =============================================================================
# 2. NEURAL NETWORK DEFINITIONS
# =============================================================================
class AsymmetricFocalLoss(nn.Module):

    def __init__(self, gamma_pos=1.0, gamma_neg=3.5, clip=0.02, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.view_as(probs)
        p_pos = probs.clamp(min=self.clip) if self.clip else probs
        p_neg = (1.0 - probs).clamp(min=self.clip) if self.clip else (1.0 - probs)

        loss_pos = (
            -targets * ((1.0 - p_pos) ** self.gamma_pos) * torch.log(p_pos + self.eps)
        )
        loss_neg = (
            -(1.0 - targets) * (p_pos**self.gamma_neg) * torch.log(p_neg + self.eps)
        )
        return (loss_pos + loss_neg).mean()


class GatedResidualBlock(nn.Module):

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        self.gate = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.gate(x)
        h = F.silu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(residual + h)


class GatedTabularNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.input_layer(x))
        h = self.input_norm(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# =============================================================================
# 3. STRATIFIED K-FOLD MODEL TRAINING
# =============================================================================
import xgboost as xgb

oof_lgb = np.zeros(len(df_train_imputed))
oof_xgb = np.zeros(len(df_train_imputed))
oof_cat = np.zeros(len(df_train_imputed))
oof_nn = np.zeros(len(df_train_imputed))

test_preds_lgb = np.zeros(len(df_test_imputed))
test_preds_xgb = np.zeros(len(df_test_imputed))
test_preds_cat = np.zeros(len(df_test_imputed))
test_preds_nn = np.zeros(len(df_test_imputed))

print(
    "Starting 5-Fold Cross Validation across LightGBM, XGBoost, CatBoost, and PyTorch Neural Net..."
)

for fold in range(num_folds):
    train_idx = folds != fold
    val_idx = folds == fold

    X_train_f, y_train_f = X_all[train_idx], y_all[train_idx]
    X_val_f, y_val_f = X_all[val_idx], y_all[val_idx]

    # --- 1. LightGBM ---
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        objective="binary",
        metric="auc",
    )
    model_lgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val_f)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / num_folds

    # --- 2. XGBoost ---
    model_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.2,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_f)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / num_folds

    # --- 3. CatBoost ---
    model_cat = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=5.0,
        scale_pos_weight=1.0,
        random_seed=42 + fold,
        verbose=False,
        eval_metric="AUC",
        thread_count=-1,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_val_f)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / num_folds

    # --- 4. PyTorch GatedTabularNet ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_f, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=2048, shuffle=True, num_workers=2
    )

    nn_model = GatedTabularNet(
        input_dim=num_features, hidden_dim=128, num_blocks=3, dropout=0.25
    ).to(device)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=3.5)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=1e-5
    )

    best_val_auc = 0.0
    best_nn_val_preds = None
    best_nn_test_preds = None

    for epoch in range(25):
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
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            val_logits = nn_model(val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
            val_auc = roc_auc_score(y_val_f, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_nn_val_preds = val_probs

                test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(
                    device
                )
                best_nn_test_preds = (
                    torch.sigmoid(nn_model(test_tensor)).cpu().numpy().ravel()
                )

    oof_nn[val_idx] = best_nn_val_preds
    test_preds_nn += best_nn_test_preds / num_folds

    fold_lgb_auc = roc_auc_score(y_val_f, oof_lgb[val_idx])
    fold_xgb_auc = roc_auc_score(y_val_f, oof_xgb[val_idx])
    fold_cat_auc = roc_auc_score(y_val_f, oof_cat[val_idx])
    fold_nn_auc = roc_auc_score(y_val_f, oof_nn[val_idx])
    print(
        f"Fold {fold+1}/5 - LGB: {fold_lgb_auc:.5f} | XGB: {fold_xgb_auc:.5f} | CAT: {fold_cat_auc:.5f} | NN: {fold_nn_auc:.5f}"
    )

# =============================================================================
# 4. ENSEMBLE OPTIMIZATION & EVALUATION
# =============================================================================
auc_lgb = roc_auc_score(y_all, oof_lgb)
auc_xgb = roc_auc_score(y_all, oof_xgb)
auc_cat = roc_auc_score(y_all, oof_cat)
auc_nn = roc_auc_score(y_all, oof_nn)

print(
    f"Overall OOF AUC -> LightGBM: {auc_lgb:.5f} | XGBoost: {auc_xgb:.5f} | CatBoost: {auc_cat:.5f} | NeuralNet: {auc_nn:.5f}"
)

# Apply percentile rank transformation prior to ensembling
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_cat_rank = rankdata(oof_cat) / len(oof_cat)
oof_nn_rank = rankdata(oof_nn) / len(oof_nn)

test_preds_lgb_rank = rankdata(test_preds_lgb) / len(test_preds_lgb)
test_preds_xgb_rank = rankdata(test_preds_xgb) / len(test_preds_xgb)
test_preds_cat_rank = rankdata(test_preds_cat) / len(test_preds_cat)
test_preds_nn_rank = rankdata(test_preds_nn) / len(test_preds_nn)


def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = (
        w[0] * oof_lgb_rank
        + w[1] * oof_xgb_rank
        + w[2] * oof_cat_rank
        + w[3] * oof_nn_rank
    )
    return -roc_auc_score(y_all, blend_oof)


init_weights = [0.35, 0.35, 0.20, 0.10]
bounds = [(0, 1)] * 4
res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds)

opt_weights = res.x / np.sum(res.x)
print(
    f"Optimized Weights -> LGB: {opt_weights[0]:.3f}, XGB: {opt_weights[1]:.3f}, CAT: {opt_weights[2]:.3f}, NN: {opt_weights[3]:.3f}"
)

oof_ensemble = (
    opt_weights[0] * oof_lgb_rank
    + opt_weights[1] * oof_xgb_rank
    + opt_weights[2] * oof_cat_rank
    + opt_weights[3] * oof_nn_rank
)
final_val_score = roc_auc_score(y_all, oof_ensemble)

# =============================================================================
# 5. SUBMISSION GENERATION
# =============================================================================
test_ensemble = (
    opt_weights[0] * test_preds_lgb_rank
    + opt_weights[1] * test_preds_xgb_rank
    + opt_weights[2] * test_preds_cat_rank
    + opt_weights[3] * test_preds_nn_rank
)

submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "SeriousDlqin2yrs": test_ensemble}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission with shape {submission_df.shape} to {submission_path}")

# Required final output string
print(f"Final Validation Score: {final_val_score}")
