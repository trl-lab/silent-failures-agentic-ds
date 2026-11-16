import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Set deterministic random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Helper & Feature Engineering Functions
# =========================================================================
def standardize_col_name(col: str) -> str:
    """Standardizes column names by converting to lowercase and stripping punctuation."""
    c = col.strip().lower()
    c = re.sub(r"[\s\-]+", "_", c)
    c = re.sub(r"[^\w_]", "", c)
    return c


def parse_target(val) -> int:
    """Parses binary churn target value to integer {0, 1}."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().rstrip(".")
    if s in {"true", "1", "yes", "churn", "t", "y", "1.0"}:
        return 1
    return 0


def find_col(df, patterns):
    """Finds first matching column name in df from a list of regex patterns."""
    for col in df.columns:
        for p in patterns:
            if re.search(p, col):
                return col
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates rich domain features for telecom churn prediction."""
    res = df.copy()

    day_mins_col = find_col(res, [r"total_day_min", r"day_min"])
    day_calls_col = find_col(res, [r"total_day_call", r"day_call"])
    day_charge_col = find_col(res, [r"total_day_charge", r"day_charge"])

    eve_mins_col = find_col(res, [r"total_eve_min", r"eve_min"])
    eve_calls_col = find_col(res, [r"total_eve_call", r"eve_call"])
    eve_charge_col = find_col(res, [r"total_eve_charge", r"eve_charge"])

    night_mins_col = find_col(res, [r"total_night_min", r"night_min"])
    night_calls_col = find_col(res, [r"total_night_call", r"night_call"])
    night_charge_col = find_col(res, [r"total_night_charge", r"night_charge"])

    intl_mins_col = find_col(res, [r"total_intl_min", r"intl_min"])
    intl_calls_col = find_col(res, [r"total_intl_call", r"intl_call"])
    intl_charge_col = find_col(res, [r"total_intl_charge", r"intl_charge"])

    cust_serv_col = find_col(
        res, [r"customer_service_call", r"custserv_call", r"cust_serv"]
    )
    acct_len_col = find_col(res, [r"account_length", r"account_len"])
    vmail_msgs_col = find_col(
        res, [r"number_vmail_messages", r"vmail_message", r"voice_mail_message"]
    )
    intl_plan_col = find_col(res, [r"international_plan", r"intl_plan"])
    vmail_plan_col = find_col(res, [r"voice_mail_plan", r"vmail_plan"])

    for col in [
        day_mins_col,
        day_calls_col,
        day_charge_col,
        eve_mins_col,
        eve_calls_col,
        eve_charge_col,
        night_mins_col,
        night_calls_col,
        night_charge_col,
        intl_mins_col,
        intl_calls_col,
        intl_charge_col,
        cust_serv_col,
        acct_len_col,
        vmail_msgs_col,
    ]:
        if col is not None and col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").fillna(0.0)

    # Binary plan flags
    if intl_plan_col in res.columns:
        res["intl_plan_binary"] = (
            res[intl_plan_col]
            .astype(str)
            .str.lower()
            .str.contains(r"yes|true|1")
            .astype(float)
        )
    else:
        res["intl_plan_binary"] = 0.0

    if vmail_plan_col in res.columns:
        res["vmail_plan_binary"] = (
            res[vmail_plan_col]
            .astype(str)
            .str.lower()
            .str.contains(r"yes|true|1")
            .astype(float)
        )
    else:
        res["vmail_plan_binary"] = 0.0

    # Total usage aggregations
    day_m = res[day_mins_col].astype(float) if day_mins_col else 0.0
    eve_m = res[eve_mins_col].astype(float) if eve_mins_col else 0.0
    night_m = res[night_mins_col].astype(float) if night_mins_col else 0.0
    intl_m = res[intl_mins_col].astype(float) if intl_mins_col else 0.0

    day_c = res[day_calls_col].astype(float) if day_calls_col else 0.0
    eve_c = res[eve_calls_col].astype(float) if eve_calls_col else 0.0
    night_c = res[night_calls_col].astype(float) if night_calls_col else 0.0
    intl_c = res[intl_calls_col].astype(float) if intl_calls_col else 0.0

    day_ch = res[day_charge_col].astype(float) if day_charge_col else 0.0
    eve_ch = res[eve_charge_col].astype(float) if eve_charge_col else 0.0
    night_ch = res[night_charge_col].astype(float) if night_charge_col else 0.0
    intl_ch = res[intl_charge_col].astype(float) if intl_charge_col else 0.0

    res["total_minutes"] = day_m + eve_m + night_m + intl_m
    res["total_calls"] = day_c + eve_c + night_c + intl_c
    res["total_charge"] = day_ch + eve_ch + night_ch + intl_ch
    res["total_domestic_minutes"] = day_m + eve_m + night_m
    res["total_domestic_charge"] = day_ch + eve_ch + night_ch

    # Minutes and charge proportions
    res["day_mins_ratio"] = day_m / (res["total_minutes"] + 1e-4)
    res["eve_mins_ratio"] = eve_m / (res["total_minutes"] + 1e-4)
    res["night_mins_ratio"] = night_m / (res["total_minutes"] + 1e-4)
    res["intl_mins_ratio"] = intl_m / (res["total_minutes"] + 1e-4)

    res["day_charge_ratio"] = day_ch / (res["total_charge"] + 1e-4)
    res["eve_charge_ratio"] = eve_ch / (res["total_charge"] + 1e-4)
    res["night_charge_ratio"] = night_ch / (res["total_charge"] + 1e-4)
    res["intl_charge_ratio"] = intl_ch / (res["total_charge"] + 1e-4)

    # Time-of-day ratios
    res["day_to_eve_mins_ratio"] = day_m / (eve_m + 1e-4)
    res["day_to_night_mins_ratio"] = day_m / (night_m + 1e-4)
    res["eve_to_night_mins_ratio"] = eve_m / (night_m + 1e-4)

    # Minutes per call & charge per call
    res["day_mins_per_call"] = day_m / (day_c + 1e-4)
    res["eve_mins_per_call"] = eve_m / (eve_c + 1e-4)
    res["night_mins_per_call"] = night_m / (night_c + 1e-4)
    res["intl_mins_per_call"] = intl_m / (intl_c + 1e-4)
    res["overall_mins_per_call"] = res["total_minutes"] / (res["total_calls"] + 1e-4)

    res["day_charge_per_call"] = day_ch / (day_c + 1e-4)
    res["eve_charge_per_call"] = eve_ch / (eve_c + 1e-4)
    res["night_charge_per_call"] = night_ch / (night_c + 1e-4)
    res["intl_charge_per_call"] = intl_ch / (intl_c + 1e-4)
    res["overall_charge_per_call"] = res["total_charge"] / (res["total_calls"] + 1e-4)

    # Effective tariff rates (charge per minute)
    res["day_rate"] = day_ch / (day_m + 1e-4)
    res["eve_rate"] = eve_ch / (eve_m + 1e-4)
    res["night_rate"] = night_ch / (night_m + 1e-4)
    res["intl_rate"] = intl_ch / (intl_m + 1e-4)
    res["effective_overall_rate"] = res["total_charge"] / (res["total_minutes"] + 1e-4)

    # Customer service call intensity & risk inflection points
    if cust_serv_col in res.columns:
        cs = res[cust_serv_col].astype(float)
        res["high_service_calls_flag"] = (cs >= 4).astype(float)
        res["service_calls_ge_3"] = (cs >= 3).astype(float)
        res["service_calls_per_total_calls"] = cs / (res["total_calls"] + 1.0)
        res["service_calls_per_total_charge"] = cs / (res["total_charge"] + 1.0)
        if acct_len_col in res.columns:
            acct_val = res[acct_len_col].astype(float)
            res["service_call_rate_per_month"] = cs / ((acct_val / 30.0) + 1.0)
            res["service_call_rate_per_day"] = cs / (acct_val + 1.0)

    # Account tenure based metrics
    if acct_len_col in res.columns:
        acct = res[acct_len_col].astype(float)
        res["charge_per_tenure_day"] = res["total_charge"] / (acct + 1.0)
        res["minutes_per_tenure_day"] = res["total_minutes"] / (acct + 1.0)
        res["calls_per_tenure_day"] = res["total_calls"] / (acct + 1.0)
        res["tenure_months"] = acct / 30.0

    # Plan interactions
    res["intl_plan_usage_interaction"] = res["intl_plan_binary"] * intl_m
    res["intl_plan_charge_interaction"] = res["intl_plan_binary"] * intl_ch
    res["intl_plan_calls_interaction"] = res["intl_plan_binary"] * intl_c

    if vmail_msgs_col in res.columns:
        vmail_cnt = res[vmail_msgs_col].astype(float)
        tenure_m = res["tenure_months"] if "tenure_months" in res.columns else 1.0
        res["vmail_msgs_per_month"] = vmail_cnt / (tenure_m + 1.0)
        res["has_vmail_with_zero_msgs"] = (
            (res["vmail_plan_binary"] == 1.0) & (vmail_cnt == 0)
        ).astype(float)
        res["has_vmail_msgs_without_plan"] = (
            (res["vmail_plan_binary"] == 0.0) & (vmail_cnt > 0)
        ).astype(float)

    # High-expenditure and high-volume risk indicators & non-linear risk heuristics
    res["high_day_charge_flag"] = (day_ch > 40.0).astype(float)
    res["high_intl_charge_flag"] = (intl_ch > 3.5).astype(float)
    res["high_total_charge_flag"] = (res["total_charge"] > 75.0).astype(float)
    res["high_day_charge_no_vmail"] = (
        (day_ch > 40.0) & (res["vmail_plan_binary"] == 0.0)
    ).astype(float)
    res["high_day_charge_no_intl"] = (
        (day_ch > 40.0) & (res["intl_plan_binary"] == 0.0)
    ).astype(float)
    res["intl_plan_cost_ratio"] = res["intl_plan_binary"] * (
        intl_ch / (res["total_charge"] + 1e-4)
    )
    if cust_serv_col in res.columns:
        res["service_calls_to_day_calls"] = cs / (day_c + 1.0)
        res["service_calls_to_intl_calls"] = cs / (intl_c + 1.0)
        res["high_service_call_high_charge"] = (
            (cs >= 4) & (res["total_charge"] > 60.0)
        ).astype(float)

    return res


# =========================================================================
# 2. Deep Tabular Neural Network Architecture (TabResNet)
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                nn.Sequential(
                    ResidualBlock(hidden_dims[i], dropout_rate=dropout_rate),
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.SiLU(),
                )
            )

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        for block in self.res_blocks:
            feat = block(feat)
        logits = self.head(feat)
        return logits.squeeze(-1)


def get_neural_components(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 35,
    pos_weight: float = 1.0,
):
    pos_weight_tensor = (
        torch.tensor([pos_weight], dtype=torch.float32, device=device)
        if pos_weight != 1.0
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return criterion, optimizer, scheduler


# =========================================================================
# 3. Model Factories for GBDT and Ensembles
# =========================================================================
def get_lgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        importance_type="gain",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.80,
        colsample_bylevel=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=1.0,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


# =========================================================================
# 4. Data Loading & Preprocessing
# =========================================================================
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

train_df = train_raw.copy()
test_df = test_raw.copy()
train_df.columns = [standardize_col_name(c) for c in train_df.columns]
test_df.columns = [standardize_col_name(c) for c in test_df.columns]

target_col = find_col(train_df, [r"^class$", r"^churn$", r"^target$"])
if target_col is None:
    target_col = "class"

y_train = train_df[target_col].apply(parse_target).astype(int).values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = (
    test_df.drop(columns=[target_col])
    if target_col in test_df.columns
    else test_df.copy()
)

X_train_fe = engineer_features(X_train_raw)
X_test_fe = engineer_features(X_test_raw)

# Drop phone number or raw ID columns if present to avoid high-cardinality noise
id_cols = [
    c
    for c in X_train_fe.columns
    if re.search(r"phone|phone_number", c, re.IGNORECASE)
]
if id_cols:
    X_train_fe.drop(columns=id_cols, inplace=True, errors="ignore")
    X_test_fe.drop(columns=id_cols, inplace=True, errors="ignore")

# Group-level usage deviations (state, area code)
group_cols = [c for c in ["state", "area_code"] if c in X_train_fe.columns]
metric_candidates = [
    find_col(X_train_fe, [r"total_day_min", r"day_min"]),
    find_col(X_train_fe, [r"total_day_charge", r"day_charge"]),
    find_col(X_train_fe, [r"customer_service_call", r"custserv_call", r"cust_serv"]),
]
metric_cols = [m for m in metric_candidates if m is not None]

for grp in group_cols:
    for m_col in metric_cols:
        X_train_fe[m_col] = pd.to_numeric(X_train_fe[m_col], errors="coerce").fillna(0.0)
        X_test_fe[m_col] = pd.to_numeric(X_test_fe[m_col], errors="coerce").fillna(0.0)
        grp_s = X_train_fe[grp].astype(str)
        test_grp_s = X_test_fe[grp].astype(str)
        grp_means = X_train_fe.groupby(grp_s, observed=False)[m_col].mean().to_dict()
        overall_m = float(X_train_fe[m_col].mean())
        X_train_fe[f"{grp}_{m_col}_diff"] = (
            X_train_fe[m_col] - grp_s.map(grp_means).fillna(overall_m).astype(float)
        )
        X_test_fe[f"{grp}_{m_col}_diff"] = (
            X_test_fe[m_col] - test_grp_s.map(grp_means).fillna(overall_m).astype(float)
        )

# Encode categoricals with Bayesian Target Encoding and leak-free statistics
cat_cols = [
    c
    for c in X_train_fe.columns
    if X_train_fe[c].dtype == "object"
    or str(X_train_fe[c].dtype) == "category"
    or X_train_fe[c].dtype == "bool"
]

te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
smoothing_m = 10.0
global_mean = float(np.mean(y_train))

for c in cat_cols:
    train_vals = X_train_fe[c].astype(str)
    test_vals = X_test_fe[c].astype(str)

    # Frequency encoding
    freq_map = train_vals.value_counts(normalize=True).to_dict()
    X_train_fe[f"{c}_freq"] = train_vals.map(freq_map).fillna(0.0).astype(float)
    X_test_fe[f"{c}_freq"] = test_vals.map(freq_map).fillna(0.0).astype(float)

    # Bayesian Target Encoding
    cat_counts = pd.Series(y_train).groupby(train_vals.values).count()
    cat_sums = pd.Series(y_train).groupby(train_vals.values).sum()
    smoothed_map = (
        (cat_sums + smoothing_m * global_mean) / (cat_counts + smoothing_m)
    ).to_dict()
    X_test_fe[f"{c}_target_enc"] = (
        test_vals.map(smoothed_map).fillna(global_mean).astype(float)
    )

    oof_te = np.full(len(X_train_fe), global_mean, dtype=float)
    for tr_i, val_i in te_skf.split(X_train_fe, y_train):
        tr_y = y_train[tr_i]
        tr_c = train_vals.iloc[tr_i].values
        fold_counts = pd.Series(tr_y).groupby(tr_c).count()
        fold_sums = pd.Series(tr_y).groupby(tr_c).sum()
        fold_map = (
            (fold_sums + smoothing_m * global_mean) / (fold_counts + smoothing_m)
        ).to_dict()
        oof_te[val_i] = (
            train_vals.iloc[val_i].map(fold_map).fillna(global_mean).values
        )
    X_train_fe[f"{c}_target_enc"] = oof_te

    # Label Encoding
    le = LabelEncoder()
    all_unique = np.unique(np.concatenate([train_vals.unique(), test_vals.unique()]))
    le.fit(all_unique)
    X_train_fe[f"{c}_encoded"] = le.transform(train_vals)
    X_test_fe[f"{c}_encoded"] = le.transform(test_vals)

    X_train_fe.drop(columns=[c], inplace=True)
    X_test_fe.drop(columns=[c], inplace=True)

X_train_fe = (
    X_train_fe.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
X_test_fe = (
    X_test_fe.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

feature_columns = [col for col in X_train_fe.columns if col in X_test_fe.columns]
X_train_mat = X_train_fe[feature_columns].values.astype(np.float32)
X_test_mat = X_test_fe[feature_columns].values.astype(np.float32)
n_features = len(feature_columns)
n_samples = len(y_train)
n_test = len(X_test_mat)

# =========================================================================
# 5. Stratified 10-Fold Cross-Validation Training
# =========================================================================
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_cb = np.zeros(n_samples, dtype=np.float32)
oof_nn = np.zeros(n_samples, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)
test_cb = np.zeros(n_test, dtype=np.float32)
test_nn = np.zeros(n_test, dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_mat, y_train)):
    X_tr, y_tr = X_train_mat[train_idx], y_train[train_idx]
    X_va, y_va = X_train_mat[val_idx], y_train[val_idx]

    # --- Model 1: LightGBM ---
    lgb_model = get_lgbm_model(random_state=42 + fold)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_preds_lgb = lgb_model.predict_proba(X_va)[:, 1]
    oof_lgb[val_idx] = val_preds_lgb
    test_lgb += lgb_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # --- Model 2: XGBoost ---
    xgb_model = get_xgb_model(random_state=42 + fold)
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    val_preds_xgb = xgb_model.predict_proba(X_va)[:, 1]
    oof_xgb[val_idx] = val_preds_xgb
    test_xgb += xgb_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # --- Model 3: CatBoost Classifier ---
    cb_model = get_catboost_model(random_state=42 + fold)
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    val_preds_cb = cb_model.predict_proba(X_va)[:, 1]
    oof_cb[val_idx] = val_preds_cb
    test_cb += cb_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # --- Model 4: Tabular Residual Neural Network ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_scaled = scaler.transform(X_va).astype(np.float32)
    X_te_scaled = scaler.transform(X_test_mat).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled), torch.tensor(y_tr, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

    nn_model = TabularResNet(
        input_dim=n_features, hidden_dims=[256, 128, 64], dropout_rate=0.25
    ).to(device)

    pos_weight = float((len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-5))
    criterion, optimizer, scheduler = get_neural_components(
        nn_model,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=35,
        pos_weight=min(pos_weight, 4.0),
    )

    best_val_auc = -1.0
    best_weights = None

    for epoch in range(35):
        nn_model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = nn_model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(torch.tensor(X_va_scaled).to(device))
            val_preds_epoch = torch.sigmoid(val_logits).cpu().numpy()
            epoch_val_auc = roc_auc_score(y_va, val_preds_epoch)

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        val_preds_nn = (
            torch.sigmoid(nn_model(torch.tensor(X_va_scaled).to(device))).cpu().numpy()
        )
        oof_nn[val_idx] = val_preds_nn
        test_preds_nn = (
            torch.sigmoid(nn_model(torch.tensor(X_te_scaled).to(device))).cpu().numpy()
        )
        test_nn += test_preds_nn / n_folds

    print(
        f"Fold {fold+1}/{n_folds} - LGB AUC: {roc_auc_score(y_va, val_preds_lgb):.5f} | XGB AUC: {roc_auc_score(y_va, val_preds_xgb):.5f} | CB AUC: {roc_auc_score(y_va, val_preds_cb):.5f} | NN AUC: {roc_auc_score(y_va, val_preds_nn):.5f}"
    )


# =========================================================================
# 6. Ensemble Optimization & Submission Generation
# =========================================================================
# Rank-percentile normalization to align probability distributions
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_cb_rank = rankdata(oof_cb) / len(oof_cb)
oof_nn_rank = rankdata(oof_nn) / len(oof_nn)

test_lgb_rank = rankdata(test_lgb) / len(test_lgb)
test_xgb_rank = rankdata(test_xgb) / len(test_xgb)
test_cb_rank = rankdata(test_cb) / len(test_cb)
test_nn_rank = rankdata(test_nn) / len(test_nn)

oof_matrix = np.column_stack([oof_lgb_rank, oof_xgb_rank, oof_cb_rank, oof_nn_rank])
test_matrix = np.column_stack([test_lgb_rank, test_xgb_rank, test_cb_rank, test_nn_rank])


def blend_objective(weights_raw):
    w = np.exp(weights_raw) / np.sum(np.exp(weights_raw))
    pred = oof_matrix @ w
    return -roc_auc_score(y_train, pred)


initial_weights = np.array([1.0, 1.0, 1.0, 0.5], dtype=float)
res = minimize(blend_objective, initial_weights, method="Nelder-Mead")
optimal_weights = np.exp(res.x) / np.sum(np.exp(res.x))

oof_final = oof_matrix @ optimal_weights
final_validation_score = float(roc_auc_score(y_train, oof_final))
test_final = test_matrix @ optimal_weights

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_final), dtype=int), "class": test_final}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Optimal Ensemble Weights [LGB, XGB, CB, NN]: {np.round(optimal_weights, 4)}")
print(f"Final Validation Score: {final_validation_score}")