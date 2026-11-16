import os
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    cb_available = True
except ImportError:
    cb_available = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Seed & Environment Setup
# ---------------------------------------------------------
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)


# ---------------------------------------------------------
# 2. Data Cleaning & Sentinel Missing Value Processing
# ---------------------------------------------------------
def clean_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if col in ["Date", "Time"]:
            continue
        if df[col].dtype == object or df[col].dtype == "string":
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # UCI Air Quality sentinel value for missing data is -200
        df[col] = df[col].replace(-200.0, np.nan)
        df[col] = df[col].replace(-200, np.nan)
    return df


train_clean = clean_dataframe(train_raw)
test_clean = clean_dataframe(test_raw)

# Drop missing target values in training set
train_clean = train_clean.dropna(subset=["CO(GT)"]).reset_index(drop=True)

# Preserve original sample order via index tracking
train_clean["is_test"] = 0
train_clean["orig_idx"] = train_clean.index

test_clean["is_test"] = 1
test_clean["orig_idx"] = test_clean.index

# Combine for contiguous time-series feature alignment
combined = pd.concat([train_clean, test_clean], ignore_index=True)


# ---------------------------------------------------------
# 3. Datetime Parsing & Feature Engineering
# ---------------------------------------------------------
def parse_datetime(df):
    time_str = df["Time"].astype(str).str.replace(".", ":", regex=False)
    date_str = df["Date"].astype(str)
    combined_dt_str = date_str + " " + time_str

    dt = pd.to_datetime(combined_dt_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    null_mask = dt.isna()
    if null_mask.any():
        dt[null_mask] = pd.to_datetime(combined_dt_str[null_mask], errors="coerce")
    return dt


combined["dt"] = parse_datetime(combined)
combined = combined.sort_values("dt").reset_index(drop=True)

# Temporal & Diurnal Cyclical Features
combined["hour"] = combined["dt"].dt.hour
combined["dayofweek"] = combined["dt"].dt.dayofweek
combined["day"] = combined["dt"].dt.day
combined["month"] = combined["dt"].dt.month
combined["is_weekend"] = combined["dayofweek"].isin([5, 6]).astype(int)

combined["sin_hour"] = np.sin(2 * np.pi * combined["hour"] / 24.0)
combined["cos_hour"] = np.cos(2 * np.pi * combined["hour"] / 24.0)
combined["sin_month"] = np.sin(2 * np.pi * combined["month"] / 12.0)
combined["cos_month"] = np.cos(2 * np.pi * combined["month"] / 12.0)
combined["sin_dow"] = np.sin(2 * np.pi * combined["dayofweek"] / 7.0)
combined["cos_dow"] = np.cos(2 * np.pi * combined["dayofweek"] / 7.0)

combined["is_morning_rush"] = combined["hour"].isin([7, 8, 9, 10]).astype(int)
combined["is_evening_rush"] = combined["hour"].isin([17, 18, 19, 20]).astype(int)

# Identify feature columns
sensor_cols = [
    c
    for c in [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
    ]
    if c in combined.columns
]
env_cols = [c for c in ["T", "RH", "AH"] if c in combined.columns]
gt_cols = [
    c for c in ["NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"] if c in combined.columns
]

# Missingness Indicators
for col in sensor_cols + env_cols + gt_cols:
    combined[f"{col}_isna"] = combined[col].isna().astype(int)

# Thermodynamic & Environmental Cross-Interactions
if "T" in combined.columns and "RH" in combined.columns:
    combined["dew_point_approx"] = combined["T"] - ((100.0 - combined["RH"]) / 5.0)
    combined["T_RH_prod"] = combined["T"] * combined["RH"]
    combined["T_kelvin"] = combined["T"] + 273.15
    combined["RH_over_Tkelvin"] = combined["RH"] / combined["T_kelvin"]

if "T" in combined.columns and "AH" in combined.columns:
    combined["T_AH_prod"] = combined["T"] * combined["AH"]

if "RH" in combined.columns and "AH" in combined.columns:
    combined["AH_over_RH"] = combined["AH"] / (combined["RH"] + 1e-5)

# Sensor Transfer Response Features
for s in sensor_cols:
    if "T_kelvin" in combined.columns:
        combined[f"{s}_per_T"] = combined[s] / combined["T_kelvin"]
    if "RH" in combined.columns:
        combined[f"{s}_per_RH"] = combined[s] / (combined["RH"] + 1e-5)
    if "AH" in combined.columns:
        combined[f"{s}_per_AH"] = combined[s] / (combined["AH"] + 1e-5)

    combined[f"{s}_nat_log"] = np.log(np.maximum(1e-5, combined[s]))
    combined[f"{s}_log"] = np.log1p(np.maximum(0, combined[s]))
    combined[f"{s}_sqrt"] = np.sqrt(np.maximum(0, combined[s]))

# Pairwise Sensor Dynamics & Ratios Across All PT08 Sensors
for s1, s2 in itertools.combinations(sensor_cols, 2):
    combined[f"{s1}_{s2}_ratio"] = combined[s1] / (combined[s2] + 1e-5)
    combined[f"{s2}_{s1}_ratio"] = combined[s2] / (combined[s1] + 1e-5)
    combined[f"{s1}_{s2}_prod"] = combined[s1] * combined[s2]
    log_s1 = np.log(np.maximum(1e-5, combined[s1]))
    log_s2 = np.log(np.maximum(1e-5, combined[s2]))
    combined[f"{s1}_{s2}_nat_log_diff"] = log_s1 - log_s2

# Reference Co-pollutant Exhaust Interactions
if "C6H6(GT)" in combined.columns:
    combined["C6H6_sq"] = combined["C6H6(GT)"] ** 2
if "NOx(GT)" in combined.columns and "NO2(GT)" in combined.columns:
    combined["NOx_NO2_ratio"] = combined["NOx(GT)"] / (combined["NO2(GT)"] + 1e-5)
if "C6H6(GT)" in combined.columns and "NOx(GT)" in combined.columns:
    combined["C6H6_NOx_prod"] = combined["C6H6(GT)"] * combined["NOx(GT)"]

# Physical Sensor-to-Reference Calibration Ratios
for s in sensor_cols:
    for g in gt_cols:
        combined[f"{s}_{g}_ratio"] = combined[s] / (combined[g] + 1e-5)
        combined[f"{g}_{s}_ratio"] = combined[g] / (combined[s] + 1e-5)

# Log-Domain Power-Law Interactions
for s in sensor_cols:
    log_s = np.log1p(np.maximum(0, combined[s]))
    for e in env_cols:
        e_val = combined["T_kelvin"] if e == "T" and "T_kelvin" in combined.columns else combined[e]
        log_e = np.log1p(np.maximum(0, e_val))
        combined[f"{s}_{e}_log_prod"] = log_s * log_e
        combined[f"{s}_{e}_log_ratio"] = log_s / (log_e + 1e-5)
        combined[f"{s}_{e}_log_diff"] = log_s - log_e

if len(sensor_cols) > 0:
    s_vals = combined[sensor_cols].values
    combined["sensor_mean"] = np.nanmean(s_vals, axis=1)
    combined["sensor_std"] = np.nanstd(s_vals, axis=1)

# ---------------------------------------------------------
# 4. Bidirectional Lags/Leads, EWM & Centered Rolling Windows
# ---------------------------------------------------------
lag_features = [
    c
    for c in sensor_cols + env_cols + gt_cols
    if c in combined.columns
]

for col in lag_features:
    # Lags & Leads
    combined[f"{col}_lag1"] = combined[col].shift(1)
    combined[f"{col}_lag2"] = combined[col].shift(2)
    combined[f"{col}_lag3"] = combined[col].shift(3)
    combined[f"{col}_lag24"] = combined[col].shift(24)
    combined[f"{col}_lead1"] = combined[col].shift(-1)
    combined[f"{col}_lead2"] = combined[col].shift(-2)
    combined[f"{col}_lead3"] = combined[col].shift(-3)
    combined[f"{col}_lead24"] = combined[col].shift(-24)

    combined[f"{col}_diff1"] = combined[col] - combined[f"{col}_lag1"]
    combined[f"{col}_diff24"] = combined[col] - combined[f"{col}_lag24"]
    combined[f"{col}_diff_lead1"] = combined[col] - combined[f"{col}_lead1"]

    # Past rolling statistics
    s_past = combined[col].shift(1)
    combined[f"{col}_roll_mean_3h"] = s_past.rolling(3, min_periods=1).mean()
    combined[f"{col}_roll_std_3h"] = s_past.rolling(3, min_periods=1).std()
    combined[f"{col}_roll_mean_6h"] = s_past.rolling(6, min_periods=1).mean()
    combined[f"{col}_roll_mean_12h"] = s_past.rolling(12, min_periods=1).mean()
    combined[f"{col}_roll_mean_24h"] = s_past.rolling(24, min_periods=1).mean()

    # Centered rolling windows (3h, 5h, 12h, 24h)
    combined[f"{col}_center_roll_mean_3h"] = combined[col].rolling(3, center=True, min_periods=1).mean()
    combined[f"{col}_center_roll_std_3h"] = combined[col].rolling(3, center=True, min_periods=1).std()
    combined[f"{col}_center_roll_mean_5h"] = combined[col].rolling(5, center=True, min_periods=1).mean()
    combined[f"{col}_center_roll_std_5h"] = combined[col].rolling(5, center=True, min_periods=1).std()
    combined[f"{col}_center_roll_mean_12h"] = combined[col].rolling(12, center=True, min_periods=1).mean()
    combined[f"{col}_center_roll_std_12h"] = combined[col].rolling(12, center=True, min_periods=1).std()
    combined[f"{col}_center_roll_mean_24h"] = combined[col].rolling(24, center=True, min_periods=1).mean()
    combined[f"{col}_center_roll_std_24h"] = combined[col].rolling(24, center=True, min_periods=1).std()

    # Exponential Weighted Moving Averages (EWM)
    combined[f"{col}_ewm_span3"] = combined[col].ewm(span=3, min_periods=1).mean()
    combined[f"{col}_ewm_span6"] = combined[col].ewm(span=6, min_periods=1).mean()
    combined[f"{col}_ewm_span12"] = combined[col].ewm(span=12, min_periods=1).mean()
    combined[f"{col}_ewm_span24"] = combined[col].ewm(span=24, min_periods=1).mean()

# ---------------------------------------------------------
# 5. Restore Original Train and Test Order
# ---------------------------------------------------------
train_proc = (
    combined[combined["is_test"] == 0].sort_values("orig_idx").reset_index(drop=True)
)
test_proc = (
    combined[combined["is_test"] == 1].sort_values("orig_idx").reset_index(drop=True)
)

drop_cols = ["Date", "Time", "dt", "is_test", "orig_idx", "CO(GT)"]
feature_cols = [c for c in train_proc.columns if c not in drop_cols]

X_df = train_proc[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
y_arr = np.array(train_proc["CO(GT)"].values, dtype=np.float32)
X_test_df = test_proc[feature_cols].replace([np.inf, -np.inf], np.nan).copy()


# ---------------------------------------------------------
# 6. PyTorch Tabular ResNet Neural Architecture
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.Mish()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.Mish()
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------
# 7. Model Training & Out-of-Fold Cross Validation
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_list = ["lgb", "xgb", "nn"]
if cb_available:
    models_list.append("cb")
num_models = len(models_list)

oof_matrix = np.zeros((len(X_df), num_models))
test_matrix = np.zeros((len(X_test_df), num_models))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 1500,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y_arr)):
    X_tr_fold, y_tr_fold = X_df.iloc[train_idx], y_arr[train_idx]
    X_va_fold, y_va_fold = X_df.iloc[val_idx], y_arr[val_idx]

    # --- 1. LightGBM Model ---
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=[(X_va_fold, y_va_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    val_p_lgb = np.maximum(0, model_lgb.predict(X_va_fold))
    t_p_lgb = np.maximum(0, model_lgb.predict(X_test_df))

    oof_matrix[val_idx, 0] += val_p_lgb
    test_matrix[:, 0] += t_p_lgb / 5.0

    # --- 2. XGBoost Model ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )
    model_xgb.fit(
        X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)], verbose=False
    )
    val_p_xgb = np.maximum(0, model_xgb.predict(X_va_fold))
    t_p_xgb = np.maximum(0, model_xgb.predict(X_test_df))

    oof_matrix[val_idx, 1] += val_p_xgb
    test_matrix[:, 1] += t_p_xgb / 5.0

    # --- 3. PyTorch Tabular ResNet Model ---
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = scaler.fit_transform(imputer.fit_transform(X_tr_fold))
    X_va_imp = scaler.transform(imputer.transform(X_va_fold))
    X_te_imp = scaler.transform(imputer.transform(X_test_df))

    ds_tr = TensorDataset(
        torch.tensor(X_tr_imp, dtype=torch.float32),
        torch.tensor(y_tr_fold, dtype=torch.float32),
    )
    ds_va = TensorDataset(
        torch.tensor(X_va_imp, dtype=torch.float32),
        torch.tensor(y_va_fold, dtype=torch.float32),
    )

    loader_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)

    nn_model = TabularResNet(
        input_dim=X_tr_imp.shape[1], hidden_dim=128, num_blocks=2, dropout=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights_nn = None
    patience, patience_cnt = 10, 0

    for epoch in range(80):
        nn_model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for bx, by in loader_va:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_samples += len(by)

        val_loss = val_loss / val_samples
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_nn = {k: v.cpu() for k, v in nn_model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights_nn is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights_nn.items()})

    nn_model.eval()
    with torch.no_grad():
        val_p_nn_raw = (
            nn_model(torch.tensor(X_va_imp, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        t_p_nn_raw = (
            nn_model(torch.tensor(X_te_imp, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    val_p_nn = np.maximum(0, val_p_nn_raw)
    t_p_nn = np.maximum(0, t_p_nn_raw)

    oof_matrix[val_idx, 2] += val_p_nn
    test_matrix[:, 2] += t_p_nn / 5.0

    # --- 4. CatBoost Model (If Available) ---
    if cb_available:
        model_cb = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
        )
        model_cb.fit(
            X_tr_fold, y_tr_fold, eval_set=(X_va_fold, y_va_fold), use_best_model=True
        )
        val_p_cb = np.maximum(0, model_cb.predict(X_va_fold))
        t_p_cb = np.maximum(0, model_cb.predict(X_test_df))

        oof_matrix[val_idx, 3] += val_p_cb
        test_matrix[:, 3] += t_p_cb / 5.0


# ---------------------------------------------------------
# 8. SLSQP Optimization for Dynamic Ensemble Weights
# ---------------------------------------------------------
def objective(weights):
    w = weights / np.sum(weights)
    blend = oof_matrix @ w
    blend = np.maximum(0, blend)
    return root_mean_squared_error(y_arr, blend)


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1)] * num_models
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x / np.sum(opt_res.x)

oof_final = np.maximum(0, oof_matrix @ best_weights)
test_final = np.maximum(0, test_matrix @ best_weights)

# ---------------------------------------------------------
# 9. Output Predictions & Score Tracking
# ---------------------------------------------------------
os.makedirs("./submission", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_final)), "CO(GT)": test_final})
sub.to_csv("./submission/submission.csv", index=False)

cv_score = root_mean_squared_error(y_arr, oof_final)
print(f"Final Validation Score: {cv_score}")
