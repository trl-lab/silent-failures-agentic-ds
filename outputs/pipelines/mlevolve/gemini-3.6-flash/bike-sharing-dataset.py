import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Reproducibility seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Directory creation
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

train_raw["dteday"] = pd.to_datetime(train_raw["dteday"])
test_raw["dteday"] = pd.to_datetime(test_raw["dteday"])

if "instant" in train_raw.columns:
    train_raw = train_raw.sort_values(by="instant").reset_index(drop=True)
else:
    train_raw = train_raw.sort_values(by=["dteday", "hr"]).reset_index(drop=True)

# 2. Chronological Train / Validation Split (80% Train, 20% Val)
split_idx = int(len(train_raw) * 0.8)
train_df = train_raw.iloc[:split_idx].copy()
val_df = train_raw.iloc[split_idx:].copy()
test_df = test_raw.copy()

target_col = "cnt"
y_train = train_df[target_col].values
y_val = val_df[target_col].values


# 3. Feature Engineering Pipeline
def engineer_features(df):
    df_out = df.copy()

    df_out["day"] = df_out["dteday"].dt.day
    df_out["dayofweek"] = df_out["dteday"].dt.dayofweek
    df_out["dayofyear"] = df_out["dteday"].dt.dayofyear
    df_out["is_weekend"] = (df_out["dayofweek"] >= 5).astype(int)

    df_out["hr_sin"] = np.sin(2 * np.pi * df_out["hr"] / 24.0)
    df_out["hr_cos"] = np.cos(2 * np.pi * df_out["hr"] / 24.0)
    df_out["mnth_sin"] = np.sin(2 * np.pi * df_out["mnth"] / 12.0)
    df_out["mnth_cos"] = np.cos(2 * np.pi * df_out["mnth"] / 12.0)
    df_out["weekday_sin"] = np.sin(2 * np.pi * df_out["weekday"] / 7.0)
    df_out["weekday_cos"] = np.cos(2 * np.pi * df_out["weekday"] / 7.0)

    df_out["temp_diff"] = df_out["atemp"] - df_out["temp"]
    df_out["temp_hum_ratio"] = df_out["temp"] / (df_out["hum"] + 1e-4)
    df_out["temp_wind_prod"] = df_out["temp"] * df_out["windspeed"]
    df_out["humid_wind_prod"] = df_out["hum"] * df_out["windspeed"]
    df_out["high_humidity"] = (df_out["hum"] > 0.8).astype(int)
    df_out["high_wind"] = (df_out["windspeed"] > 0.5).astype(int)

    df_out["workingday_hr"] = (
        df_out["workingday"].astype(str) + "_" + df_out["hr"].astype(str)
    )
    df_out["season_weathersit"] = (
        df_out["season"].astype(str) + "_" + df_out["weathersit"].astype(str)
    )

    is_morning_peak = df_out["hr"].isin([7, 8, 9]) & (df_out["workingday"] == 1)
    is_evening_peak = df_out["hr"].isin([16, 17, 18, 19]) & (df_out["workingday"] == 1)
    df_out["is_peak_commute"] = (is_morning_peak | is_evening_peak).astype(int)

    drop_cols = ["dteday", "instant", "casual", "registered", "cnt"]
    cols_to_drop = [c for c in drop_cols if c in df_out.columns]
    df_out = df_out.drop(columns=cols_to_drop)

    return df_out


X_train_df = engineer_features(train_df)
X_val_df = engineer_features(val_df)
X_test_df = engineer_features(test_df)

# Target encoding strictly derived from training split
hr_workingday_target_map = (
    train_df.groupby(["workingday", "hr"])["cnt"].mean().to_dict()
)
global_cnt_mean = float(train_df["cnt"].mean())


def apply_target_encoding(df_in, target_map, default_val):
    return np.array(
        [
            target_map.get((w, h), default_val)
            for w, h in zip(df_in["workingday"], df_in["hr"])
        ]
    )


X_train_df["target_enc_hr_work"] = apply_target_encoding(
    train_df, hr_workingday_target_map, global_cnt_mean
)
X_val_df["target_enc_hr_work"] = apply_target_encoding(
    val_df, hr_workingday_target_map, global_cnt_mean
)
X_test_df["target_enc_hr_work"] = apply_target_encoding(
    test_df, hr_workingday_target_map, global_cnt_mean
)

# Column alignment
feature_cols = [c for c in X_train_df.columns]
X_train_df = X_train_df[feature_cols]
X_val_df = X_val_df[feature_cols]
X_test_df = X_test_df[feature_cols]

# Log1p target transformation
y_train_log = np.log1p(y_train).astype(np.float32)
y_val_log = np.log1p(y_val).astype(np.float32)

cat_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "workingday_hr",
    "season_weathersit",
]
cat_cols = [c for c in cat_cols if c in X_train_df.columns]
num_cols = [c for c in X_train_df.columns if c not in cat_cols]

train_medians = X_train_df[num_cols].median()
X_train_df[num_cols] = X_train_df[num_cols].fillna(train_medians)
X_val_df[num_cols] = X_val_df[num_cols].fillna(train_medians)
X_test_df[num_cols] = X_test_df[num_cols].fillna(train_medians)

# Categorical handling for GBDT models
X_train_gbdt = X_train_df.copy()
X_val_gbdt = X_val_df.copy()
X_test_gbdt = X_test_df.copy()

for col in cat_cols:
    categories = X_train_df[col].astype(str).unique()
    cat_type = pd.CategoricalDtype(categories=categories)
    X_train_gbdt[col] = X_train_gbdt[col].astype(str).astype(cat_type)
    X_val_gbdt[col] = X_val_gbdt[col].astype(str).astype(cat_type)
    X_test_gbdt[col] = X_test_gbdt[col].astype(str).astype(cat_type)

X_train_cat = X_train_df.copy()
X_val_cat = X_val_df.copy()
X_test_cat = X_test_df.copy()

for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_val_cat[col] = X_val_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

# Preprocessing for Neural Network
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_df[num_cols]).astype(np.float32)
X_val_num = scaler.transform(X_val_df[num_cols]).astype(np.float32)
X_test_num = scaler.transform(X_test_df[num_cols]).astype(np.float32)

cat_dims = {}
X_train_cat_nn = np.zeros((len(X_train_df), len(cat_cols)), dtype=np.int64)
X_val_cat_nn = np.zeros((len(X_val_df), len(cat_cols)), dtype=np.int64)
X_test_cat_nn = np.zeros((len(X_test_df), len(cat_cols)), dtype=np.int64)

for idx, col in enumerate(cat_cols):
    unique_vals = {val: i for i, val in enumerate(X_train_df[col].astype(str).unique())}
    num_classes = len(unique_vals) + 1
    emb_dim = min(50, max(4, (num_classes + 1) // 2))
    cat_dims[col] = (num_classes, emb_dim)

    X_train_cat_nn[:, idx] = [unique_vals[val] for val in X_train_df[col].astype(str)]
    X_val_cat_nn[:, idx] = [
        unique_vals.get(val, len(unique_vals)) for val in X_val_df[col].astype(str)
    ]
    X_test_cat_nn[:, idx] = [
        unique_vals.get(val, len(unique_vals)) for val in X_test_df[col].astype(str)
    ]

# --- Train Model 1: LightGBM ---
lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train_gbdt,
    y_train_log,
    eval_set=[(X_val_gbdt, y_val_log)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

val_pred_lgb_log = lgb_model.predict(X_val_gbdt)
test_pred_lgb_log = lgb_model.predict(X_test_gbdt)

# --- Train Model 2: XGBoost ---
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    enable_categorical=True,
)

xgb_model.fit(
    X_train_gbdt,
    y_train_log,
    eval_set=[(X_val_gbdt, y_val_log)],
    verbose=False,
)

val_pred_xgb_log = xgb_model.predict(X_val_gbdt)
test_pred_xgb_log = xgb_model.predict(X_test_gbdt)

# --- Train Model 3: CatBoost ---
cat_model = cb.CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
)

cat_model.fit(
    X_train_cat,
    y_train_log,
    eval_set=(X_val_cat, y_val_log),
    cat_features=cat_cols,
    use_best_model=True,
)

val_pred_cat_log = cat_model.predict(X_val_cat)
test_pred_cat_log = cat_model.predict(X_test_cat)


# --- Train Model 4: PyTorch TabularResNet ---
class TabularResBlock(nn.Module):

    def __init__(self, in_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        cat_dims: dict,
        num_continuous: int,
        emb_drop: float = 0.1,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, emb_dim)
                for num_classes, emb_dim in cat_dims.values()
            ]
        )
        self.emb_drop = nn.Dropout(emb_drop)
        total_emb_dim = sum(emb_dim for num_classes, emb_dim in cat_dims.values())
        in_dim = total_emb_dim + num_continuous

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x_cat: torch.Tensor = None, x_num: torch.Tensor = None
    ) -> torch.Tensor:
        emb_outs = [
            emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)
        ]
        cat_features = self.emb_drop(torch.cat(emb_outs, dim=1))
        x = torch.cat([cat_features, x_num], dim=1)
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(
    torch.tensor(X_train_cat_nn, dtype=torch.long),
    torch.tensor(X_train_num, dtype=torch.float32),
    torch.tensor(y_train_log, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_cat_nn, dtype=torch.long),
    torch.tensor(X_val_num, dtype=torch.float32),
    torch.tensor(y_val_log, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)

nn_model = TabularResNet(
    cat_dims=cat_dims,
    num_continuous=X_train_num.shape[1],
    emb_drop=0.1,
    hidden_dim=128,
    num_blocks=2,
    dropout_rate=0.15,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

best_val_loss = float("inf")
best_weights = None
patience = 15
patience_counter = 0

for epoch in range(1, 101):
    nn_model.train()
    for x_c, x_n, y_b in train_loader:
        x_c, x_n, y_b = x_c.to(device), x_n.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(nn_model(x_c, x_n), y_b)
        loss.backward()
        optimizer.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_c, x_n, y_b in val_loader:
            x_c, x_n, y_b = x_c.to(device), x_n.to(device), y_b.to(device)
            val_loss += criterion(nn_model(x_c, x_n), y_b).item() * len(y_b)
    val_loss /= len(val_dataset)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

if best_weights is not None:
    nn_model.load_state_dict(best_weights)


def predict_nn(loader):
    nn_model.eval()
    preds = []
    with torch.no_grad():
        for x_c, x_n in loader:
            x_c, x_n = x_c.to(device), x_n.to(device)
            preds.append(nn_model(x_c, x_n).cpu().numpy())
    return np.concatenate(preds)


val_pred_nn_log = predict_nn(
    DataLoader(
        TensorDataset(
            torch.tensor(X_val_cat_nn, dtype=torch.long),
            torch.tensor(X_val_num, dtype=torch.float32),
        ),
        batch_size=512,
        shuffle=False,
    )
)
test_pred_nn_log = predict_nn(
    DataLoader(
        TensorDataset(
            torch.tensor(X_test_cat_nn, dtype=torch.long),
            torch.tensor(X_test_num, dtype=torch.float32),
        ),
        batch_size=512,
        shuffle=False,
    )
)

# --- SLSQP Convex Optimization for Ensemble Weights ---
val_preds_matrix = [
    val_pred_lgb_log,
    val_pred_xgb_log,
    val_pred_cat_log,
    val_pred_nn_log,
]
test_preds_matrix = [
    test_pred_lgb_log,
    test_pred_xgb_log,
    test_pred_cat_log,
    test_pred_nn_log,
]


def objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_log = sum(w[i] * val_preds_matrix[i] for i in range(len(w)))
    blend_cnt = np.clip(np.expm1(blend_log), 0, None)
    return np.sqrt(np.mean((y_val - blend_cnt) ** 2))


init_w = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1) for _ in range(len(val_preds_matrix))]
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

opt_res = minimize(
    objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints
)
optimal_weights = opt_res.x / np.sum(opt_res.x)

final_val_log = sum(w * pred for w, pred in zip(optimal_weights, val_preds_matrix))
final_val_cnt = np.clip(np.expm1(final_val_log), 0, None)
final_rmse = float(np.sqrt(np.mean((y_val - final_val_cnt) ** 2)))

# --- Submission File Generation ---
final_test_log = sum(w * pred for w, pred in zip(optimal_weights, test_preds_matrix))
final_test_cnt = np.clip(np.expm1(final_test_log), 0, None)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_cnt)), "cnt": final_test_cnt}
)
submission_df.to_csv("./submission/submission.csv", index=False)

# Required score logging requirement
print(f"Final Validation Score: {final_rmse}")
