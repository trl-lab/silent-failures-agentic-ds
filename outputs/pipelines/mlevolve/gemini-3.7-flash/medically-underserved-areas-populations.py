import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Preprocessing
# =========================================================================
print("Loading raw datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

target_col = "IMU Score"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

y_train = train_df[target_col].copy()


def clean_numeric_series(s):
    """Clean string columns containing numeric data (percentages, currency, commas)."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    cleaned = (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


feature_cols = [c for c in train_df.columns if c != target_col]

for col in feature_cols:
    if col in test_df.columns:
        if train_df[col].dtype == "object":
            sample_non_null = train_df[col].dropna().astype(str).head(100)
            if (
                len(sample_non_null) > 0
                and sample_non_null.str.match(r"^-?[\d,\.]+%?$").mean() > 0.8
            ):
                train_df[col] = clean_numeric_series(train_df[col])
                test_df[col] = clean_numeric_series(test_df[col])

df_all = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0).reset_index(
    drop=True
)
n_train = len(train_df)

# Column type detection
datetime_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        datetime_cols.append(col)
    elif df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notna().sum() > 0.7 * len(df_all[col].dropna()):
                df_all[col] = parsed
                datetime_cols.append(col)
                continue
        except Exception:
            pass

        avg_len = df_all[col].dropna().astype(str).str.len().mean()
        unique_ratio = df_all[col].nunique() / (len(df_all[col].dropna()) + 1e-6)
        if avg_len > 50 or unique_ratio > 0.5:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)

print(
    f"Detected columns - Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}, Datetime: {len(datetime_cols)}, Text: {len(text_cols)}"
)

# =========================================================================
# 2. Feature Engineering
# =========================================================================
# Datetime Features
for col in datetime_cols:
    dt_series = pd.to_datetime(df_all[col], errors="coerce")
    df_all[f"{col}_year"] = dt_series.dt.year.astype("float32")
    df_all[f"{col}_month"] = dt_series.dt.month.astype("float32")
    df_all[f"{col}_day"] = dt_series.dt.day.astype("float32")
    df_all[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float32")
    df_all[f"{col}_quarter"] = dt_series.dt.quarter.astype("float32")
    min_date = dt_series.min()
    if pd.notna(min_date):
        df_all[f"{col}_elapsed_days"] = (dt_series - min_date).dt.days.astype("float32")
    df_all.drop(columns=[col], inplace=True)

# Missingness Indicators
for col in num_cols:
    if df_all[col].isna().sum() > 0.02 * len(df_all):
        df_all[f"{col}_isna"] = df_all[col].isna().astype(np.float32)

# Preserve raw categorical dataframe for strict OOF target encoding
raw_cat_df = df_all[cat_cols].copy()

# Domain Specific Non-Linear Ratios & Health Index Interactions
col_map = {c.lower(): c for c in num_cols}
poverty_col = next((col_map[k] for k in col_map if "poverty" in k or "pov" in k), None)
elderly_col = next(
    (col_map[k] for k in col_map if "elderly" in k or "65" in k or "age" in k), None
)
physician_col = next(
    (
        col_map[k]
        for k in col_map
        if "physician" in k
        or "doctor" in k
        or "provider" in k
        or "pcp" in k
        or "ratio" in k
    ),
    None,
)
infant_col = next(
    (col_map[k] for k in col_map if "infant" in k or "mortality" in k or "imr" in k),
    None,
)
pop_col = next((col_map[k] for k in col_map if "pop" in k or "population" in k), None)

eps = 1e-5

# Non-linear log transforms for positive numeric columns
for col in num_cols:
    min_v = df_all[col].min()
    if pd.notna(min_v) and min_v >= 0:
        df_all[f"{col}_log1p"] = np.log1p(df_all[col].clip(lower=0).fillna(0).values).astype(np.float32)

if physician_col and pop_col:
    df_all["feat_physicians_per_1k_pop"] = (
        (df_all[physician_col] / (df_all[pop_col].clip(lower=1.0))) * 1000.0
    ).astype(np.float32)
    df_all["feat_pop_per_physician"] = (
        df_all[pop_col] / (df_all[physician_col].abs() + eps)
    ).astype(np.float32)

if poverty_col and elderly_col:
    df_all["feat_poverty_x_elderly"] = (df_all[poverty_col] * df_all[elderly_col]).astype(np.float32)
    df_all["feat_poverty_plus_elderly"] = (df_all[poverty_col] + df_all[elderly_col]).astype(np.float32)

if poverty_col and infant_col:
    df_all["feat_poverty_x_infant"] = (df_all[poverty_col] * df_all[infant_col]).astype(np.float32)

if elderly_col and infant_col:
    df_all["feat_elderly_x_infant"] = (df_all[elderly_col] * df_all[infant_col]).astype(np.float32)

if poverty_col and physician_col:
    df_all["feat_poverty_div_physician"] = (
        df_all[poverty_col] / (df_all[physician_col].abs() + eps)
    ).astype(np.float32)

if elderly_col and physician_col:
    df_all["feat_elderly_div_physician"] = (
        df_all[elderly_col] / (df_all[physician_col].abs() + eps)
    ).astype(np.float32)

if infant_col and physician_col:
    df_all["feat_infant_div_physician"] = (
        df_all[infant_col] / (df_all[physician_col].abs() + eps)
    ).astype(np.float32)

# Composite Health Vulnerability Index & Burden
identified_metrics = [c for c in [poverty_col, elderly_col, infant_col] if c is not None]
if len(identified_metrics) >= 2:
    composite_sum = 0
    raw_burden = 0
    for c in identified_metrics:
        std_val = df_all[c].std()
        std_val = std_val if std_val > 0 else 1.0
        composite_sum += (df_all[c] - df_all[c].mean()) / std_val
        raw_burden += df_all[c].fillna(0)
    if physician_col:
        p_std = df_all[physician_col].std()
        p_std = p_std if p_std > 0 else 1.0
        composite_sum -= (df_all[physician_col] - df_all[physician_col].mean()) / p_std
        df_all["feat_vulnerability_burden"] = (
            raw_burden / (df_all[physician_col].abs() + eps)
        ).astype(np.float32)
    df_all["feat_composite_vulnerability_index"] = composite_sum.astype(np.float32)

# Categorical Frequency and Code Encodings
for col in cat_cols:
    freq_map = df_all[col].value_counts(dropna=False, normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq_map).astype(np.float32)
    df_all[f"{col}_cat_code"] = df_all[col].astype("category").cat.codes.astype("int32")

# Multi-level Geographic Group Aggregations
geo_cols = [
    c
    for c in cat_cols
    if any(
        k in c.lower() for k in ["state", "county", "fips", "region", "type", "status"]
    )
]
agg_num_cols = [
    c
    for c in [poverty_col, elderly_col, physician_col, infant_col, pop_col]
    if c is not None
]

df_all.drop(columns=cat_cols, inplace=True, errors="ignore")

# Text Feature Extraction (Length only globally)
for col in text_cols:
    df_all[f"{col}_len"] = df_all[col].fillna("").astype(str).str.len()
    df_all[f"{col}_word_count"] = (
        df_all[col].fillna("").astype(str).apply(lambda x: len(x.split()))
    )
    df_all.drop(columns=[col], inplace=True)

# Clean column names for GBDT algorithms
clean_col_names = [
    re.sub(r"[^\w\s]", "_", str(c)).replace(" ", "_") for c in df_all.columns
]
df_all.columns = clean_col_names

# Separate Train and Test
X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

for c in X_train.columns:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype(np.float32)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float32)
    med_val = X_train[c].median()
    if pd.isna(med_val):
        med_val = 0.0
    X_train[c] = X_train[c].fillna(med_val)
    X_test[c] = X_test[c].fillna(med_val)

# 10-Fold Stratified K-Fold setup
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
target_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")

y = y_train.values.astype(np.float32)
n_test = len(X_test)

fold_indices = np.zeros(len(y_train), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, target_bins)):
    fold_indices[val_idx] = fold

# Out-of-fold smoothed Target Encoding strictly inside CV folds
m_smooth = 10.0
global_mean = float(np.mean(y))
cats_train = raw_cat_df.iloc[:n_train].reset_index(drop=True)
cats_test = raw_cat_df.iloc[n_train:].reset_index(drop=True)

for col in cat_cols:
    oof_te = np.zeros(n_train, dtype=np.float32)
    test_te = np.zeros(n_test, dtype=np.float32)
    for fold in range(n_folds):
        val_mask = fold_indices == fold
        trn_mask = ~val_mask
        fold_cats = cats_train.iloc[trn_mask][col]
        fold_target = y[trn_mask]
        stats = pd.DataFrame({"cat": fold_cats, "y": fold_target}).groupby("cat")["y"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (stats["count"] + m_smooth)
        s_dict = smoothed.to_dict()
        oof_te[val_mask] = cats_train.iloc[val_mask][col].map(s_dict).fillna(global_mean).values
        test_te += cats_test[col].map(s_dict).fillna(global_mean).values / n_folds
    X_train[f"{col}_target_enc"] = oof_te.astype(np.float32)
    X_test[f"{col}_target_enc"] = test_te.astype(np.float32)

feature_cols_final = list(X_test.columns)


# =========================================================================
# 3. Model Architecture & Configurations
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        num_res_blocks: int = 2,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        res_blocks = [
            ResidualBlock(hidden_dims[0], dropout_rate=dropout_rate)
            for _ in range(num_res_blocks)
        ]
        self.res_layers = nn.Sequential(*res_blocks)

        downsample_layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            downsample_layers.extend(
                [
                    nn.BatchNorm1d(in_dim),
                    nn.Linear(in_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = h_dim
        self.downsample = nn.Sequential(*downsample_layers)

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_layers(h)
        h = self.downsample(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_lgbm_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_samples": 20,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3000,
        "learning_rate": 0.025,
        "depth": 5,
        "l2_leaf_reg": 4.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


def get_et_model(seed: int = 42) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=350,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=seed,
    )


# =========================================================================
# 4. Training & Out-of-Fold Evaluation Loop
# =========================================================================
oof_preds = {
    "lgb": np.zeros(n_train, dtype=np.float32),
    "xgb": np.zeros(n_train, dtype=np.float32),
    "et": np.zeros(n_train, dtype=np.float32),
    "nn": np.zeros(n_train, dtype=np.float32),
}
test_preds = {
    "lgb": np.zeros(n_test, dtype=np.float32),
    "xgb": np.zeros(n_test, dtype=np.float32),
    "et": np.zeros(n_test, dtype=np.float32),
    "nn": np.zeros(n_test, dtype=np.float32),
}

if HAS_CATBOOST:
    oof_preds["cb"] = np.zeros(n_train, dtype=np.float32)
    test_preds["cb"] = np.zeros(n_test, dtype=np.float32)

for fold in range(n_folds):
    val_idx = fold_indices == fold
    trn_idx = ~val_idx

    X_tr_df, y_tr = X_train.iloc[trn_idx], y[trn_idx]
    X_va_df, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_params = get_lgbm_params(seed=42 + fold)
    lgb_reg = lgb.LGBMRegressor(**lgb_params)
    lgb_reg.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds["lgb"][val_idx] = lgb_reg.predict(X_va_df)
    test_preds["lgb"] += lgb_reg.predict(X_test) / n_folds

    # XGBoost
    xgb_params = get_xgb_params(seed=42 + fold)
    xgb_params["early_stopping_rounds"] = 50
    xgb_reg = xgb.XGBRegressor(**xgb_params)
    xgb_reg.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)
    oof_preds["xgb"][val_idx] = xgb_reg.predict(X_va_df)
    test_preds["xgb"] += xgb_reg.predict(X_test) / n_folds

    # CatBoost
    if HAS_CATBOOST:
        cb_params = get_catboost_params(seed=42 + fold)
        cb_reg = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        cb_reg.fit(X_tr_df, y_tr, eval_set=(X_va_df, y_va), verbose=False)
        oof_preds["cb"][val_idx] = cb_reg.predict(X_va_df)
        test_preds["cb"] += cb_reg.predict(X_test) / n_folds

    # ExtraTrees Regressor
    et_reg = get_et_model(seed=42 + fold)
    et_reg.fit(X_tr_df, y_tr)
    oof_preds["et"][val_idx] = et_reg.predict(X_va_df)
    test_preds["et"] += et_reg.predict(X_test) / n_folds

    # PyTorch Tabular ResNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_df.values).astype(np.float32)
    X_va_scaled = scaler.transform(X_va_df.values).astype(np.float32)
    X_test_scaled = scaler.transform(X_test.values).astype(np.float32)

    train_dataset = TensorDataset(torch.tensor(X_tr_scaled), torch.tensor(y_tr))
    val_dataset = TensorDataset(torch.tensor(X_va_scaled), torch.tensor(y_va))
    test_tensor = torch.tensor(X_test_scaled).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    nn_model = TabularResNet(
        input_dim=len(feature_cols_final),
        hidden_dims=[256, 128, 64],
        num_res_blocks=2,
        dropout_rate=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights_path = f"./working/models/tabular_resnet_fold_{fold}.pt"
    patience = 15
    patience_counter = 0

    for epoch in range(60):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = nn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_out = nn_model(val_x)
                v_loss = criterion(val_out, val_y)
                val_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(nn_model.state_dict(), best_weights_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(torch.load(best_weights_path, map_location=device))
    nn_model.eval()

    val_preds_list = []
    with torch.no_grad():
        for val_x, _ in val_loader:
            val_x = val_x.to(device)
            val_preds_list.append(nn_model(val_x).cpu().numpy())
    oof_preds["nn"][val_idx] = np.concatenate(val_preds_list)

    with torch.no_grad():
        test_nn_fold = nn_model(test_tensor).cpu().numpy()
        test_preds["nn"] += test_nn_fold / n_folds

# =========================================================================
# 5. Constrained Blending Optimization & Final Predictions
# =========================================================================
model_names = list(oof_preds.keys())
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])


def rmse_objective(weights):
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-12)
    pred_blend = np.dot(oof_matrix, weights)
    return np.sqrt(mean_squared_error(y, pred_blend))


initial_weights = [1.0 / len(model_names)] * len(model_names)
bounds = [(0.0, 1.0) for _ in range(len(model_names))]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    rmse_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 200},
)

if opt_res.success and not np.any(np.isnan(opt_res.x)):
    best_weights = opt_res.x / np.sum(opt_res.x)
else:
    best_weights = np.array(initial_weights)

final_oof_pred = np.dot(oof_matrix, best_weights)
final_oof_rmse = float(np.sqrt(mean_squared_error(y, final_oof_pred)))

final_test_pred = np.dot(test_matrix, best_weights)
final_test_pred = np.clip(final_test_pred, a_min=0.0, a_max=100.0)

sub = pd.DataFrame(
    {"id": np.arange(len(final_test_pred)), "IMU Score": final_test_pred}
)
sub.to_csv("./submission/submission.csv", index=False)
print("Submission saved successfully to ./submission/submission.csv")

print(f"Final Validation Score: {final_oof_rmse}")