import json
import os
import re
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set global configurations
RANDOM_STATE = 42
N_SPLITS = 10
TARGET_COL = "SeriousDlqin2yrs"

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
print("Loading train and test datasets from ./input/...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]


# 2. Domain-Specific Feature Engineering
def engineer_features(
    df: pd.DataFrame, train_stats: dict = None
) -> (pd.DataFrame, dict):
    """
    Transforms raw credit bureau data into domain-informed risk features.
    Computes summary stats solely on the training split to prevent leakage.
    """
    data = df.copy()
    if train_stats is None:
        stats = {
            "MonthlyIncome_median": float(data["MonthlyIncome"].median()),
            "NumberOfDependents_median": float(data["NumberOfDependents"].median()),
            "age_median": float(data[data["age"] > 0]["age"].median()),
            "freq_maps": {},
            "rank_arrays": {},
            "income_model": None,
        }
    else:
        stats = train_stats

    # A. Missing Value Flags & Sentinel Handling
    data["MonthlyIncome_isna"] = data["MonthlyIncome"].isna().astype(np.int32)
    data["NumberOfDependents_isna"] = data["NumberOfDependents"].isna().astype(np.int32)

    # Age sentinels
    data["is_age_zero"] = (data["age"] <= 0).astype(np.int32)
    data["age_cleaned"] = np.where(data["age"] <= 0, stats["age_median"], data["age"])

    # Delinquency sentinels (96 and 98 represent missing/special codes)
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    data["is_delinquency_sentinel"] = (
        (data[delinq_cols[0]] >= 96)
        | (data[delinq_cols[1]] >= 96)
        | (data[delinq_cols[2]] >= 96)
    ).astype(np.int32)

    for col in delinq_cols:
        data[f"{col}_cleaned"] = np.where(data[col] >= 96, 0, data[col])

    dependents_imputed = data["NumberOfDependents"].fillna(
        stats["NumberOfDependents_median"]
    )
    data["dependents_cleaned"] = dependents_imputed

    # B. ML-Based Missing Income Imputation (Trained strictly on training rows)
    impute_features = [
        "age_cleaned",
        "dependents_cleaned",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "is_delinquency_sentinel",
    ]
    if train_stats is None:
        valid_inc_mask = data["MonthlyIncome"].notna() & (data["MonthlyIncome"] > 0)
        inc_regressor = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
        inc_regressor.fit(
            data.loc[valid_inc_mask, impute_features],
            np.log1p(data.loc[valid_inc_mask, "MonthlyIncome"]),
        )
        stats["income_model"] = inc_regressor

    predicted_log_income = stats["income_model"].predict(data[impute_features])
    predicted_income = np.expm1(np.clip(predicted_log_income, 0, 15))
    income_imputed = np.where(
        data["MonthlyIncome"].isna(), predicted_income, data["MonthlyIncome"]
    )
    income_imputed = np.clip(income_imputed, 0, 1e7)
    data["MonthlyIncome_imputed"] = income_imputed

    # C. Delinquency Aggregations & Non-Linear Severity Scores
    data["TotalPastDue"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse_cleaned"]
        + data["NumberOfTime60-89DaysPastDueNotWorse_cleaned"]
        + data["NumberOfTimes90DaysLate_cleaned"]
    )
    data["WeightedPastDue"] = (
        1.0 * data["NumberOfTime30-59DaysPastDueNotWorse_cleaned"]
        + 2.0 * data["NumberOfTime60-89DaysPastDueNotWorse_cleaned"]
        + 3.0 * data["NumberOfTimes90DaysLate_cleaned"]
    )
    # Exponential severity captures non-linear compounding default risks
    data["ExpDelinquencySeverity"] = (
        np.exp(0.4 * np.clip(data["NumberOfTime30-59DaysPastDueNotWorse_cleaned"], 0, 10))
        + np.exp(0.7 * np.clip(data["NumberOfTime60-89DaysPastDueNotWorse_cleaned"], 0, 10))
        + np.exp(1.1 * np.clip(data["NumberOfTimes90DaysLate_cleaned"], 0, 10))
        - 3.0
    )
    data["HasAnyPastDue"] = (data["TotalPastDue"] > 0).astype(np.int32)
    data["Has90DaysLate"] = (data["NumberOfTimes90DaysLate_cleaned"] > 0).astype(np.int32)

    # D. Utilization Features & Anomalies
    data["is_utilization_over_1"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] > 1.0
    ).astype(np.int32)
    data["is_utilization_over_2"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] > 2.0
    ).astype(np.int32)
    data["is_utilization_over_10"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] > 10.0
    ).astype(np.int32)
    data["is_utilization_over_100"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] > 100.0
    ).astype(np.int32)
    data["is_utilization_zero"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] == 0.0
    ).astype(np.int32)
    data["log_revolving_util"] = np.log1p(
        np.clip(data["RevolvingUtilizationOfUnsecuredLines"], 0, 100)
    )

    # E. Debt Obligations & Solvency Proxies
    data["is_high_debt_ratio"] = (data["DebtRatio"] > 1.0).astype(np.int32)
    data["log_debt_ratio"] = np.log1p(np.clip(data["DebtRatio"], 0, 1e6))
    data["log_monthly_income"] = np.log1p(income_imputed)

    # When MonthlyIncome was missing, DebtRatio in bureau data is the absolute monthly debt
    estimated_monthly_debt = np.where(
        data["MonthlyIncome_isna"] == 1,
        data["DebtRatio"],
        data["DebtRatio"] * income_imputed,
    )
    data["estimated_monthly_debt"] = estimated_monthly_debt
    data["log_estimated_monthly_debt"] = np.log1p(
        np.clip(estimated_monthly_debt, 0, 1e7)
    )

    # Disposable income proxy and debt-to-income ratio
    data["disposable_income_proxy"] = np.clip(
        income_imputed - estimated_monthly_debt, -1e6, 1e7
    )
    data["is_negative_disposable"] = (data["disposable_income_proxy"] < 0).astype(np.int32)
    data["Debt_to_Income_Ratio"] = estimated_monthly_debt / (income_imputed + 50.0)

    # F. Household & Capacity Ratios
    data["HouseholdSize"] = dependents_imputed + 1.0
    data["Income_per_Person"] = income_imputed / data["HouseholdSize"]
    data["log_income_per_person"] = np.log1p(np.clip(data["Income_per_Person"], 0, 1e7))
    data["Debt_per_Person"] = estimated_monthly_debt / data["HouseholdSize"]

    # G. Credit Lines & Loan Mix
    total_lines = data["NumberOfOpenCreditLinesAndLoans"]
    real_estate_lines = data["NumberRealEstateLoansOrLines"]
    data["NonRealEstateLines"] = np.maximum(0, total_lines - real_estate_lines)
    data["RealEstateShare"] = real_estate_lines / (total_lines + 1e-4)
    data["Income_per_CreditLine"] = income_imputed / (total_lines + 1.0)
    data["Debt_per_CreditLine"] = estimated_monthly_debt / (total_lines + 1.0)
    data["Debt_per_RealEstateLine"] = estimated_monthly_debt / (real_estate_lines + 1.0)
    data["Debt_per_NonRealEstateLine"] = estimated_monthly_debt / (data["NonRealEstateLines"] + 1.0)
    data["Income_per_RealEstateLine"] = income_imputed / (real_estate_lines + 1.0)
    data["Income_per_NonRealEstateLine"] = income_imputed / (data["NonRealEstateLines"] + 1.0)
    data["Lines_per_Age"] = total_lines / (data["age_cleaned"] + 1e-4)

    # H. Advanced Delinquency Ratios & Severity Interactions
    data["Late90_to_Lines"] = data["NumberOfTimes90DaysLate_cleaned"] / (total_lines + 1.0)
    data["Late90_to_PastDue"] = data["NumberOfTimes90DaysLate_cleaned"] / (data["TotalPastDue"] + 1.0)
    data["PastDue_to_Age"] = data["TotalPastDue"] / (data["age_cleaned"] + 1.0)
    data["PastDue_to_Lines"] = data["TotalPastDue"] / (total_lines + 1.0)

    data["Credit_Capacity_per_Age"] = (income_imputed / (data["DebtRatio"] + 1.0)) / (
        data["age_cleaned"] + 1.0
    )
    data["log_credit_capacity"] = np.log1p(np.clip(data["Credit_Capacity_per_Age"], 0, 1e7))

    data["Util_x_DebtRatio"] = data["log_revolving_util"] * data["log_debt_ratio"]
    data["PastDue_x_Util"] = data["WeightedPastDue"] * data["log_revolving_util"]
    data["PastDue_x_DebtRatio"] = data["WeightedPastDue"] * data["log_debt_ratio"]

    # I. Binned Attributes for Categorization & Target Encoding
    data["age_bin"] = pd.cut(
        data["age_cleaned"],
        bins=[-np.inf, 30, 40, 50, 60, 70, np.inf],
        labels=False,
    ).astype(np.int32)
    data["delinq_bin"] = np.clip(data["TotalPastDue"], 0, 5).astype(np.int32)
    data["age_x_delinq_bin"] = data["age_bin"] * 10 + data["delinq_bin"]
    data["lines_bin"] = pd.cut(
        total_lines,
        bins=[-np.inf, 2, 5, 10, 15, np.inf],
        labels=False,
    ).astype(np.int32)
    data["dependents_bin"] = np.clip(dependents_imputed, 0, 4).astype(np.int32)

    # J. Frequency Encodings
    freq_cols = [
        "age_cleaned",
        "NumberOfDependents",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "TotalPastDue",
    ]
    for fc in freq_cols:
        if fc not in stats["freq_maps"]:
            stats["freq_maps"][fc] = data[fc].value_counts(normalize=True).to_dict()
        data[f"{fc}_freq"] = data[fc].map(stats["freq_maps"][fc]).fillna(0).astype(np.float64)

    # K. Percentile Rank Transformations
    rank_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "ExpDelinquencySeverity",
        "DebtRatio",
        "MonthlyIncome_imputed",
        "Income_per_Person",
        "estimated_monthly_debt",
        "disposable_income_proxy",
    ]
    for rc in rank_cols:
        clean_s = data[rc].values
        if rc not in stats["rank_arrays"]:
            stats["rank_arrays"][rc] = np.sort(clean_s)
        sorted_ref = stats["rank_arrays"][rc]
        data[f"{rc}_rank"] = (
            np.searchsorted(sorted_ref, clean_s, side="right") / float(len(sorted_ref))
        ).astype(np.float64)

    # Composite Monotonic Risk Index
    data["CompositeRiskIndex"] = (
        0.35 * data["RevolvingUtilizationOfUnsecuredLines_rank"]
        + 0.45 * data["ExpDelinquencySeverity_rank"]
        + 0.20 * data["estimated_monthly_debt_rank"]
    ).astype(np.float64)

    return data, stats


print("Processing features...")
train_processed, computed_stats = engineer_features(train_df, train_stats=None)
test_processed, _ = engineer_features(test_df, train_stats=computed_stats)

# Sanitize feature names to remove special characters (e.g. hyphens) for LightGBM compatibility
train_processed.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in train_processed.columns]
test_processed.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in test_processed.columns]

# Stratified K-Fold Split
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_processed["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, train_processed[TARGET_COL])
):
    train_processed.loc[val_idx, "fold"] = fold

excluded_cols = {TARGET_COL, "fold", "id", "Unnamed_0", "Unnamed_0_", "index"}
feature_cols = [c for c in train_processed.columns if c not in excluded_cols]

print(f"Dataset prepared with {len(feature_cols)} features across {N_SPLITS} folds.")

# 3. Model Training & Evaluation Loop
class TabularNN(nn.Module):
    def __init__(self, in_features, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
        )
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.act1 = nn.SiLU()
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.act1(self.ln1(x0 + self.block1(x0)))
        x2 = self.block2(x1)
        out = self.head(x2)
        return out


X = train_processed[feature_cols].copy()
y = train_processed[TARGET_COL].values
X_test = test_processed[feature_cols].copy()

oof_lgb = np.zeros(len(train_processed), dtype=np.float64)
oof_xgb = np.zeros(len(train_processed), dtype=np.float64)
oof_cat = np.zeros(len(train_processed), dtype=np.float64)
oof_nn = np.zeros(len(train_processed), dtype=np.float64)
test_preds_lgb = np.zeros(len(test_processed), dtype=np.float64)
test_preds_xgb = np.zeros(len(test_processed), dtype=np.float64)
test_preds_cat = np.zeros(len(test_processed), dtype=np.float64)
test_preds_nn = np.zeros(len(test_processed), dtype=np.float64)

folds = sorted(train_processed["fold"].unique())

for fold in folds:
    val_mask = train_processed["fold"] == fold
    train_mask = ~val_mask

    X_tr, y_tr = X.loc[train_mask].copy(), y[train_mask]
    X_val, y_val = X.loc[val_mask].copy(), y[val_mask]
    X_te = X_test.copy()

    # Leak-free out-of-fold target encoding for discrete risk bins
    te_cols = ["age_bin", "delinq_bin", "age_x_delinq_bin", "lines_bin", "dependents_bin"]
    global_mean = float(np.mean(y_tr))
    smoothing = 10.0

    for col in te_cols:
        counts = pd.Series(y_tr).groupby(X_tr[col].values).agg(["count", "mean"])
        smoothed = (counts["count"] * counts["mean"] + smoothing * global_mean) / (
            counts["count"] + smoothing
        )
        te_map = smoothed.to_dict()
        X_tr[f"{col}_te"] = X_tr[col].map(te_map).fillna(global_mean).astype(np.float64)
        X_val[f"{col}_te"] = X_val[col].map(te_map).fillna(global_mean).astype(np.float64)
        X_te[f"{col}_te"] = X_te[col].map(te_map).fillna(global_mean).astype(np.float64)

    # --- LightGBM Model ---
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "n_estimators": 2500,
        "random_state": RANDOM_STATE + fold,
        "n_jobs": -1,
        "verbose": -1,
    }

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=lgb_callbacks,
    )

    val_preds_lgb = model_lgb.predict_proba(X_val)[:, 1]
    oof_lgb[val_mask] = val_preds_lgb
    test_preds_lgb += model_lgb.predict_proba(X_te)[:, 1] / len(folds)
    fold_lgb_auc = roc_auc_score(y_val, val_preds_lgb)

    # --- XGBoost Model ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "n_estimators": 2500,
        "early_stopping_rounds": 50,
        "random_state": RANDOM_STATE + fold,
        "n_jobs": -1,
    }

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_preds_xgb = model_xgb.predict_proba(X_val)[:, 1]
    oof_xgb[val_mask] = val_preds_xgb
    test_preds_xgb += model_xgb.predict_proba(X_te)[:, 1] / len(folds)
    fold_xgb_auc = roc_auc_score(y_val, val_preds_xgb)

    # --- CatBoost Model ---
    cat_params = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE + fold,
        "verbose": False,
        "early_stopping_rounds": 50,
        "thread_count": -1,
    }

    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_preds_cat = model_cat.predict_proba(X_val)[:, 1]
    oof_cat[val_mask] = val_preds_cat
    test_preds_cat += model_cat.predict_proba(X_te)[:, 1] / len(folds)
    fold_cat_auc = roc_auc_score(y_val, val_preds_cat)

    # --- PyTorch Tabular NN Model with Quantile Normalization ---
    qt = QuantileTransformer(
        n_quantiles=min(2000, len(X_tr)),
        output_distribution="normal",
        random_state=RANDOM_STATE,
        subsample=200000,
    )
    X_tr_scaled = qt.fit_transform(np.nan_to_num(X_tr.values, nan=0.0))
    X_val_scaled = qt.transform(np.nan_to_num(X_val.values, nan=0.0))
    X_te_scaled = qt.transform(np.nan_to_num(X_te.values, nan=0.0))

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = TabularNN(in_features=X_tr.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auc = 0.0
    best_nn_preds = None
    best_state = None

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = nn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_preds_list.append(torch.sigmoid(nn_model(batch_x)).cpu().numpy())
        val_preds_epoch = np.vstack(val_preds_list).ravel()
        val_auc_epoch = roc_auc_score(y_val, val_preds_epoch)

        if val_auc_epoch > best_val_auc:
            best_val_auc = val_auc_epoch
            best_nn_preds = val_preds_epoch
            best_state = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict(best_state)
    oof_nn[val_mask] = best_nn_preds
    fold_nn_auc = best_val_auc

    te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
    nn_model.eval()
    with torch.no_grad():
        test_preds_fold_nn = torch.sigmoid(nn_model(te_tensor)).cpu().numpy().ravel()
    test_preds_nn += test_preds_fold_nn / len(folds)

    fold_blend_auc = roc_auc_score(
        y_val,
        (
            rankdata(val_preds_lgb)
            + rankdata(val_preds_xgb)
            + rankdata(val_preds_cat)
            + rankdata(best_nn_preds)
        )
        / 4.0,
    )
    print(
        f"Fold {fold} | LGB AUC: {fold_lgb_auc:.5f} | XGB AUC: {fold_xgb_auc:.5f} | CAT AUC: {fold_cat_auc:.5f} | NN AUC: {fold_nn_auc:.5f} | Blend AUC: {fold_blend_auc:.5f}"
    )

# 4. Out-of-Fold Ensemble Evaluation & Metric Optimization
total_lgb_auc = roc_auc_score(y, oof_lgb)
total_xgb_auc = roc_auc_score(y, oof_xgb)
total_cat_auc = roc_auc_score(y, oof_cat)
total_nn_auc = roc_auc_score(y, oof_nn)

# Rank-transform out-of-fold predictions to uniform [0, 1] quantile rankings
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_cat_rank = rankdata(oof_cat) / len(oof_cat)
oof_nn_rank = rankdata(oof_nn) / len(oof_nn)


def objective_fn(weights):
    w = np.maximum(0, weights)
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    blend = (
        w[0] * oof_lgb_rank
        + w[1] * oof_xgb_rank
        + w[2] * oof_cat_rank
        + w[3] * oof_nn_rank
    )
    return -roc_auc_score(y, blend)


res = minimize(
    objective_fn,
    [0.25, 0.25, 0.25, 0.25],
    method="Nelder-Mead",
    options={"maxiter": 500},
)
opt_w = np.maximum(0, res.x)
opt_w = opt_w / opt_w.sum()

oof_blend = (
    opt_w[0] * oof_lgb_rank
    + opt_w[1] * oof_xgb_rank
    + opt_w[2] * oof_cat_rank
    + opt_w[3] * oof_nn_rank
)
final_validation_score = roc_auc_score(y, oof_blend)

print(
    f"Overall OOF - LightGBM: {total_lgb_auc:.5f} | XGBoost: {total_xgb_auc:.5f} | CatBoost: {total_cat_auc:.5f} | NN: {total_nn_auc:.5f}"
)
print(
    f"Optimal Weights: LGB={opt_w[0]:.4f}, XGB={opt_w[1]:.4f}, CAT={opt_w[2]:.4f}, NN={opt_w[3]:.4f} | Ensemble AUC: {final_validation_score:.5f}"
)

# 5. Generate and Verify Submission
test_lgb_rank = rankdata(test_preds_lgb) / len(test_preds_lgb)
test_xgb_rank = rankdata(test_preds_xgb) / len(test_preds_xgb)
test_cat_rank = rankdata(test_preds_cat) / len(test_preds_cat)
test_nn_rank = rankdata(test_preds_nn) / len(test_preds_nn)

submission_preds = (
    opt_w[0] * test_lgb_rank
    + opt_w[1] * test_xgb_rank
    + opt_w[2] * test_cat_rank
    + opt_w[3] * test_nn_rank
)
test_ids = np.arange(len(test_processed), dtype=int)
submission_df = pd.DataFrame({"id": test_ids, "SeriousDlqin2yrs": submission_preds})

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission successfully saved to {submission_path}")

assert list(submission_df.columns) == ["id", "SeriousDlqin2yrs"]
assert len(submission_df) == len(test_processed)
assert not submission_df.isna().any().any()

# Required Final Validation Score Output
print(f"Final Validation Score: {final_validation_score}")
