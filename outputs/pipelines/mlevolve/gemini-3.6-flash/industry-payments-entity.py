import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Global Setup & Reproducibility
# ------------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ------------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "entityPayment"
if target_col not in train_df.columns:
    raise ValueError(f"Target column {target_col} not found in train data.")

# Ensure target is non-negative
train_df[target_col] = np.clip(train_df[target_col].values, 0, None)
y_train = train_df[target_col].values.astype(np.float64)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()

n_train = len(train_features)
n_test = len(test_features)

df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

ignore_cols = [c for c in ["id", "ID", "Id"] if c in df_all.columns]
feature_cols = [c for c in df_all.columns if c not in ignore_cols]

# A. Parse Datetime columns
for col in list(feature_cols):
    if df_all[col].dtype == "object" or pd.api.types.is_string_dtype(df_all[col]):
        try:
            parsed_date = pd.to_datetime(df_all[col], errors="coerce")
            if parsed_date.notna().sum() > 0.5 * len(df_all):
                df_all[col + "_year"] = parsed_date.dt.year
                df_all[col + "_month"] = parsed_date.dt.month
                df_all[col + "_day"] = parsed_date.dt.day
                df_all[col + "_dayofweek"] = parsed_date.dt.dayofweek
                df_all[col + "_dayofyear"] = parsed_date.dt.dayofyear
                df_all[col + "_quarter"] = parsed_date.dt.quarter
                df_all[col + "_is_month_end"] = parsed_date.dt.is_month_end.astype(int)
                df_all[col + "_month_sin"] = np.sin(
                    2 * np.pi * df_all[col + "_month"] / 12.0
                )
                df_all[col + "_month_cos"] = np.cos(
                    2 * np.pi * df_all[col + "_month"] / 12.0
                )
                df_all[col + "_dow_sin"] = np.sin(
                    2 * np.pi * df_all[col + "_dayofweek"] / 7.0
                )
                df_all[col + "_dow_cos"] = np.cos(
                    2 * np.pi * df_all[col + "_dayofweek"] / 7.0
                )
                df_all.drop(columns=[col], inplace=True)
        except Exception:
            pass

cat_cols = []
num_cols = []
for col in df_all.columns:
    if col in ignore_cols:
        continue
    if pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

# B. Frequency Encoding for Categoricals
for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True).to_dict()
    df_all[col + "_freq"] = df_all[col].map(freq).fillna(0).astype(np.float32)

for col in cat_cols:
    df_all[col] = df_all[col].astype(str).astype("category")

# C. Row-level Summary Statistics on Numerical Features
if len(num_cols) >= 2:
    num_arr = df_all[num_cols].fillna(0).values
    df_all["num_mean"] = np.mean(num_arr, axis=1)
    df_all["num_std"] = np.std(num_arr, axis=1)
    df_all["num_min"] = np.min(num_arr, axis=1)
    df_all["num_max"] = np.max(num_arr, axis=1)

# Impute median values for numerical features (computed from train only to avoid leakage)
for col in df_all.columns:
    if col in ignore_cols:
        continue
    if pd.api.types.is_numeric_dtype(df_all[col]):
        train_median = df_all.iloc[:n_train][col].median()
        df_all[col] = df_all[col].fillna(train_median).astype(np.float32)

X_train_full = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_full = df_all.iloc[n_train:].copy().reset_index(drop=True)

X_train_df = X_train_full.drop(columns=ignore_cols, errors="ignore")
X_test_df = X_test_full.drop(columns=ignore_cols, errors="ignore")

# D. Setup Cross-Validation Splits
n_splits = 5
bins = pd.qcut(
    y_train, q=min(10, len(np.unique(y_train))), labels=False, duplicates="drop"
)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
target_cat_col = cat_cols[0] if len(cat_cols) > 0 else None

# Numeric matrix for models requiring encoded numerical inputs
X_train_num = X_train_df.copy()
X_test_num = X_test_df.copy()
for col in cat_cols:
    if col in X_train_num.columns:
        X_train_num[col] = X_train_num[col].cat.codes
        X_test_num[col] = X_test_num[col].cat.codes


# ------------------------------------------------------------------------------
# 2. Custom Neural Architecture & Loss Function
# ------------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):
    """Gated Residual Dense Layer with LayerNorm and Dropout for Tabular Features."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        h1, h2 = self.fc1(x_norm).chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        h = self.dropout(h)
        h = self.fc2(h)
        gate_weights = torch.sigmoid(self.gate(x_norm))
        return residual + gate_weights * h


class TabularAttentionResNet(nn.Module):
    """Deep Tabular Network with Multi-Head Attention & Gated Residual Blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout_rate) for _ in range(num_layers)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h_attn_in = h.unsqueeze(1)
        h_attn, _ = self.self_attn(
            self.attn_norm(h_attn_in),
            self.attn_norm(h_attn_in),
            self.attn_norm(h_attn_in),
        )
        h = h + h_attn.squeeze(1)

        for block in self.blocks:
            h = block(h)

        out = self.head(h)
        out = F.softplus(out)
        return out.squeeze(-1)


class RobustAsymmetricHuberLoss(nn.Module):
    """Asymmetric Huber Loss tailored for zero-inflated heavy-tailed payment targets."""

    def __init__(self, delta: float = 100.0, underpredict_penalty: float = 1.2):
        super().__init__()
        self.delta = delta
        self.underpredict_penalty = underpredict_penalty

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        abs_diff = torch.abs(diff)

        huber_loss = torch.where(
            abs_diff <= self.delta,
            0.5 * (diff**2),
            self.delta * (abs_diff - 0.5 * self.delta),
        )

        weights = torch.where(
            diff > 0, self.underpredict_penalty, 1.0 / self.underpredict_penalty
        )
        return torch.mean(weights * huber_loss)


def build_neural_model_and_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
):
    model = TabularAttentionResNet(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout_rate=0.2,
    ).to(device)

    criterion = RobustAsymmetricHuberLoss(delta=100.0, underpredict_penalty=1.2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    return model, criterion, optimizer, scheduler


# ------------------------------------------------------------------------------
# 3. Model Configuration Specifications
# ------------------------------------------------------------------------------
lgb_tweedie_params = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.25,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 7,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

lgb_log_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": SEED + 100,
    "n_jobs": -1,
    "verbosity": -1,
}

xgb_huber_params = {
    "objective": "reg:pseudohubererror",
    "eval_metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "huber_slope": 50.0,
    "n_jobs": -1,
}

# ------------------------------------------------------------------------------
# 4. Multi-Model Training & Cross-Validation
# ------------------------------------------------------------------------------
oof_preds_lgb_tw = np.zeros(n_train, dtype=np.float64)
oof_preds_lgb_log = np.zeros(n_train, dtype=np.float64)
oof_preds_xgb = np.zeros(n_train, dtype=np.float64)
oof_preds_cat = np.zeros(n_train, dtype=np.float64)
oof_preds_nn = np.zeros(n_train, dtype=np.float64)

test_preds_lgb_tw = np.zeros(n_test, dtype=np.float64)
test_preds_lgb_log = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cat = np.zeros(n_test, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

y_train_log = np.log1p(y_train)

cat_features_idx = [
    c
    for c in cat_cols
    if c in X_train_df.columns and X_train_df[c].dtype.name == "category"
]

X_tr_num_clean = X_train_num.copy()
X_te_num_clean = X_test_num.copy()
for col in X_tr_num_clean.columns:
    med = X_tr_num_clean[col].median()
    if pd.isna(med):
        med = 0.0
    X_tr_num_clean[col] = X_tr_num_clean[col].fillna(med)
    X_te_num_clean[col] = X_te_num_clean[col].fillna(med)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_df, bins)):
    X_tr, y_tr = X_train_df.iloc[tr_idx].copy(), y_train[tr_idx]
    X_va, y_va = X_train_df.iloc[va_idx].copy(), y_train[va_idx]
    X_te_fold = X_test_df.copy()

    if target_cat_col is not None:
        smoothing = 10.0
        tr_target_mean = np.mean(y_tr)
        stats = (
            pd.DataFrame({"cat": X_tr[target_cat_col], "target": y_tr})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth_enc = (stats["count"] * stats["mean"] + smoothing * tr_target_mean) / (
            stats["count"] + smoothing
        )
        enc_map = smooth_enc.to_dict()
        X_tr["oof_target_enc"] = X_tr[target_cat_col].map(enc_map).fillna(tr_target_mean)
        X_va["oof_target_enc"] = X_va[target_cat_col].map(enc_map).fillna(tr_target_mean)
        X_te_fold["oof_target_enc"] = X_te_fold[target_cat_col].map(enc_map).fillna(tr_target_mean)

    X_tr_num_f, y_tr_num_f = X_tr_num_clean.iloc[tr_idx], y_train[tr_idx]
    X_va_num_f, y_va_num_f = X_tr_num_clean.iloc[va_idx], y_train[va_idx]
    y_tr_log_f, y_va_log_f = y_train_log[tr_idx], y_train_log[va_idx]

    # Model 1: LightGBM Tweedie
    model_lgb_tw = lgb.LGBMRegressor(**lgb_tweedie_params)
    model_lgb_tw.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_preds_lgb_tw[va_idx] = np.clip(model_lgb_tw.predict(X_va), 0, None)
    test_preds_lgb_tw += np.clip(model_lgb_tw.predict(X_te_fold), 0, None) / n_splits

    # Model 2: LightGBM Log1p
    model_lgb_log = lgb.LGBMRegressor(**lgb_log_params)
    model_lgb_log.fit(
        X_tr,
        y_tr_log_f,
        eval_set=[(X_va, y_va_log_f)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    val_log_p = model_lgb_log.predict(X_va)
    res_var = np.var(y_va_log_f - val_log_p)
    oof_preds_lgb_log[va_idx] = np.expm1(val_log_p + 0.5 * res_var)
    test_preds_lgb_log += (
        np.expm1(model_lgb_log.predict(X_te_fold) + 0.5 * res_var) / n_splits
    )

    # Model 3: XGBoost Huber
    model_xgb = xgb.XGBRegressor(
        **xgb_huber_params, early_stopping_rounds=50, random_state=SEED + fold
    )
    model_xgb.fit(
        X_tr_num_f, y_tr_num_f, eval_set=[(X_va_num_f, y_va_num_f)], verbose=0
    )
    oof_preds_xgb[va_idx] = np.clip(model_xgb.predict(X_va_num_f), 0, None)
    test_preds_xgb += np.clip(model_xgb.predict(X_te_num_clean), 0, None) / n_splits

    # Model 4: CatBoost Regressor
    model_cat = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        early_stopping_rounds=50,
        random_seed=SEED + fold,
        verbose=0,
    )
    model_cat.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_features_idx if len(cat_features_idx) > 0 else None,
    )
    oof_preds_cat[va_idx] = np.clip(model_cat.predict(X_va), 0, None)
    test_preds_cat += np.clip(model_cat.predict(X_te_fold), 0, None) / n_splits

    # Model 5: TabularAttentionResNet (PyTorch)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_num_f.values)
    X_va_sc = scaler.transform(X_va_num_f.values)
    X_te_sc = scaler.transform(X_te_num_clean.values)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr_num_f, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va_num_f, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    nn_model, nn_loss_fn, nn_optimizer, nn_scheduler = build_neural_model_and_optimizer(
        input_dim=X_tr_num_f.shape[1],
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
    )

    best_val_rmse = float("inf")
    best_weights = None
    patience, patience_cnt = 12, 0

    for epoch in range(50):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            nn_optimizer.zero_grad()
            out = nn_model(bx)
            loss = nn_loss_fn(out, by)
            loss.backward()
            nn_optimizer.step()
        nn_scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                val_preds_list.append(nn_model(bx).cpu().numpy())
        val_preds_arr = np.concatenate(val_preds_list, axis=0)
        curr_val_rmse = root_mean_squared_error(y_va_num_f, val_preds_arr)

        if curr_val_rmse < best_val_rmse:
            best_val_rmse = curr_val_rmse
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        nn_va_pred = nn_model(va_tensor).cpu().numpy()
        nn_te_pred = nn_model(te_tensor).cpu().numpy()

    oof_preds_nn[va_idx] = np.clip(nn_va_pred, 0, None)
    test_preds_nn += np.clip(nn_te_pred, 0, None) / n_splits

# ------------------------------------------------------------------------------
# 5. Non-Negative SLSQP Meta-Optimization Blending
# ------------------------------------------------------------------------------
all_oof = [
    oof_preds_lgb_tw,
    oof_preds_lgb_log,
    oof_preds_xgb,
    oof_preds_cat,
    oof_preds_nn,
]
all_test = [
    test_preds_lgb_tw,
    test_preds_lgb_log,
    test_preds_xgb,
    test_preds_cat,
    test_preds_nn,
]


def blend_rmse(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = sum(wi * pi for wi, pi in zip(w, all_oof))
    return root_mean_squared_error(y_train, np.clip(pred, 0, None))


init_weights = [1.0 / len(all_oof)] * len(all_oof)
bounds = [(0, 1)] * len(all_oof)
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    blend_rmse,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = res.x / np.sum(res.x)

# Out-Of-Fold Final Evaluation
final_oof_preds = sum(w * p for w, p in zip(opt_weights, all_oof))
final_oof_preds = np.clip(final_oof_preds, 0, None)
final_val_score = root_mean_squared_error(y_train, final_oof_preds)

# ------------------------------------------------------------------------------
# 6. Final Test Predictions & Submission File Generation
# ------------------------------------------------------------------------------
final_test_preds = sum(w * p for w, p in zip(opt_weights, all_test))
final_test_preds = np.clip(final_test_preds, 0, None)

os.makedirs("./submission", exist_ok=True)
sub_df = pd.DataFrame({"id": test_ids, "entityPayment": final_test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")