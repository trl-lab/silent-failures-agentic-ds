import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

# 1. Environment Setup & Data Loading
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "fatalities"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "fatal" in c.lower() or "target" in c.lower()
    ]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

y_train_full = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

id_col = None
for candidate in ["id", "ID", "event_id_cnty", "data_id"]:
    if candidate in X_train_raw.columns and candidate in X_test_raw.columns:
        id_col = candidate
        break

# 2. Feature Engineering Functions
date_cols = [
    c
    for c in train_df.columns
    if "date" in c.lower()
    or "time" in c.lower()
    or c.lower() in ["year", "month", "day"]
]


def extract_datetime_features(df, date_col):
    df_feat = pd.DataFrame(index=df.index)
    try:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df_feat[f"{date_col}_year"] = dt.dt.year.fillna(-1).astype(int)
        df_feat[f"{date_col}_month"] = dt.dt.month.fillna(-1).astype(int)
        df_feat[f"{date_col}_day"] = dt.dt.day.fillna(-1).astype(int)
        df_feat[f"{date_col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
        df_feat[f"{date_col}_dayofyear"] = dt.dt.dayofyear.fillna(-1).astype(int)
        df_feat[f"{date_col}_quarter"] = dt.dt.quarter.fillna(-1).astype(int)
        df_feat[f"{date_col}_is_month_end"] = dt.dt.is_month_end.fillna(False).astype(
            int
        )

        month_rad = 2 * np.pi * dt.dt.month / 12.0
        df_feat[f"{date_col}_sin_month"] = np.sin(month_rad).fillna(0)
        df_feat[f"{date_col}_cos_month"] = np.cos(month_rad).fillna(0)

        dow_rad = 2 * np.pi * dt.dt.dayofweek / 7.0
        df_feat[f"{date_col}_sin_dow"] = np.sin(dow_rad).fillna(0)
        df_feat[f"{date_col}_cos_dow"] = np.cos(dow_rad).fillna(0)
    except Exception:
        pass
    return df_feat


lat_cols = [c for c in train_df.columns if c.lower() in ["latitude", "lat", "y"]]
lon_cols = [
    c for c in train_df.columns if c.lower() in ["longitude", "lon", "long", "x"]
]


def extract_spatial_features(df, lat_col, lon_col):
    df_feat = pd.DataFrame(index=df.index)
    lat = pd.to_numeric(df[lat_col], errors="coerce").fillna(0)
    lon = pd.to_numeric(df[lon_col], errors="coerce").fillna(0)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    df_feat["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df_feat["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df_feat["coord_z"] = np.sin(lat_rad)
    df_feat["coord_rot45_1"] = 0.7071 * lat + 0.7071 * lon
    df_feat["coord_rot45_2"] = -0.7071 * lat + 0.7071 * lon
    return df_feat


feature_dfs_train = []
feature_dfs_test = []
processed_date_cols = set()

for dc in date_cols:
    if dc in X_train_raw.columns and dc in X_test_raw.columns:
        if not np.issubdtype(X_train_raw[dc].dtype, np.number):
            f_tr = extract_datetime_features(X_train_raw, dc)
            f_te = extract_datetime_features(X_test_raw, dc)
            if not f_tr.empty and not f_te.empty:
                feature_dfs_train.append(f_tr)
                feature_dfs_test.append(f_te)
                processed_date_cols.add(dc)

if lat_cols and lon_cols:
    lat_c = lat_cols[0]
    lon_c = lon_cols[0]
    if lat_c in X_train_raw.columns and lon_c in X_test_raw.columns:
        f_tr = extract_spatial_features(X_train_raw, lat_c, lon_c)
        f_te = extract_spatial_features(X_test_raw, lat_c, lon_c)
        feature_dfs_train.append(f_tr)
        feature_dfs_test.append(f_te)

numeric_cols = []
cat_cols = []
exclude_cols = set(processed_date_cols)
if id_col:
    exclude_cols.add(id_col)

for col in X_train_raw.columns:
    if col in exclude_cols:
        continue
    if pd.api.types.is_numeric_dtype(X_train_raw[col]):
        numeric_cols.append(col)
    else:
        cat_cols.append(col)

freq_df_train = pd.DataFrame(index=X_train_raw.index)
freq_df_test = pd.DataFrame(index=X_test_raw.index)
for c in cat_cols:
    freq_map = X_train_raw[c].astype(str).value_counts(normalize=True).to_dict()
    freq_df_train[f"{c}_freq"] = (
        X_train_raw[c].astype(str).map(freq_map).fillna(0).astype(float)
    )
    freq_df_test[f"{c}_freq"] = (
        X_test_raw[c].astype(str).map(freq_map).fillna(0).astype(float)
    )

feature_dfs_train.append(freq_df_train)
feature_dfs_test.append(freq_df_test)

encoded_cat_train = pd.DataFrame(index=X_train_raw.index)
encoded_cat_test = pd.DataFrame(index=X_test_raw.index)
for c in cat_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_vals = X_train_raw[[c]].astype(str)
    test_vals = X_test_raw[[c]].astype(str)
    encoded_cat_train[c] = encoder.fit_transform(train_vals)
    encoded_cat_test[c] = encoder.transform(test_vals)

feature_dfs_train.append(encoded_cat_train)
feature_dfs_test.append(encoded_cat_test)

num_train = X_train_raw[numeric_cols].copy()
num_test = X_test_raw[numeric_cols].copy()
for c in numeric_cols:
    median_val = num_train[c].median()
    if pd.isna(median_val):
        median_val = 0.0
    num_train[c] = num_train[c].fillna(median_val)
    num_test[c] = num_test[c].fillna(median_val)

feature_dfs_train.append(num_train)
feature_dfs_test.append(num_test)

X_train_all = pd.concat(feature_dfs_train, axis=1)
X_test_all = pd.concat(feature_dfs_test, axis=1)

X_train_all = X_train_all.loc[:, ~X_train_all.columns.duplicated()].copy()
X_test_all = X_test_all.loc[:, ~X_test_all.columns.duplicated()].copy()

common_cols = [c for c in X_train_all.columns if c in X_test_all.columns]
X_train_all = X_train_all[common_cols]
X_test_all = X_test_all[common_cols]

train_indices, val_indices = train_test_split(
    np.arange(len(X_train_all)), test_size=0.20, random_state=42
)
X_tr = X_train_all.iloc[train_indices].copy().reset_index(drop=True)
y_tr = y_train_full.iloc[train_indices].copy().reset_index(drop=True)
X_val = X_train_all.iloc[val_indices].copy().reset_index(drop=True)
y_val = y_train_full.iloc[val_indices].copy().reset_index(drop=True)
X_te = X_test_all.copy().reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y_tr.mean())
top_cat_cols = cat_cols[: min(5, len(cat_cols))]

for c in top_cat_cols:
    oof_target_enc = pd.Series(index=X_tr.index, dtype=float)
    tr_raw_col = X_train_raw.iloc[train_indices][c].astype(str).reset_index(drop=True)
    val_raw_col = X_train_raw.iloc[val_indices][c].astype(str).reset_index(drop=True)
    test_raw_col = X_test_raw[c].astype(str).reset_index(drop=True)
    test_target_enc = pd.Series(0.0, index=X_te.index, dtype=float)
    val_target_enc = pd.Series(0.0, index=X_val.index, dtype=float)

    for train_idx, kf_val_idx in kf.split(X_tr, y_tr):
        tr_cat = tr_raw_col.iloc[train_idx]
        tr_y = y_tr.iloc[train_idx]
        val_cat = tr_raw_col.iloc[kf_val_idx]

        means = tr_y.groupby(tr_cat).mean()
        counts = tr_y.groupby(tr_cat).count()
        smooth = 10
        smoothed_means = (counts * means + smooth * global_mean) / (counts + smooth)

        oof_target_enc.iloc[kf_val_idx] = val_cat.map(smoothed_means).fillna(global_mean)
        test_target_enc += (
            test_raw_col.map(smoothed_means).fillna(global_mean) / kf.n_splits
        )
        val_target_enc += (
            val_raw_col.map(smoothed_means).fillna(global_mean) / kf.n_splits
        )

    X_tr[f"{c}_target_enc"] = oof_target_enc.values
    X_val[f"{c}_target_enc"] = val_target_enc.values
    X_te[f"{c}_target_enc"] = test_target_enc.values


# 3. Model Architecture & Definitions
def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0, None)
    return root_mean_squared_error(y_true, y_pred)


def get_lgbm_params(random_state=42):
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 2000,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(random_state=42):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 2000,
        "n_jobs": -1,
        "tree_method": "hist",
    }


class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout_rate=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ConflictTabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        for next_dim in hidden_dims[1:]:
            self.res_blocks.append(
                TabularResidualBlock(current_dim, dropout_rate=dropout_rate)
            )
            if next_dim != current_dim:
                self.res_blocks.append(
                    nn.Sequential(
                        nn.Linear(current_dim, next_dim),
                        nn.LayerNorm(next_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout_rate),
                    )
                )
                current_dim = next_dim

        self.head = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.Linear(current_dim, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.res_blocks:
            x = layer(x)
        return self.head(x).squeeze(-1)


# 4. Model Training: LightGBM
model_lgb = lgb.LGBMRegressor(**get_lgbm_params(random_state=42))
model_lgb.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(-1),
    ],
)
val_preds_lgb = np.clip(model_lgb.predict(X_val), 0, None)
test_preds_lgb = np.clip(model_lgb.predict(X_te), 0, None)

# 5. Model Training: XGBoost
model_xgb = XGBRegressor(**get_xgb_params(random_state=42), early_stopping_rounds=50)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
val_preds_xgb = np.clip(model_xgb.predict(X_val), 0, None)
test_preds_xgb = np.clip(model_xgb.predict(X_te), 0, None)

# 6. Model Training: Deep Tabular ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = StandardScaler()
X_tr_scaled = np.nan_to_num(
    scaler.fit_transform(X_tr.values), nan=0.0, posinf=0.0, neginf=0.0
)
X_val_scaled = np.nan_to_num(
    scaler.transform(X_val.values), nan=0.0, posinf=0.0, neginf=0.0
)
X_te_scaled = np.nan_to_num(
    scaler.transform(X_te.values), nan=0.0, posinf=0.0, neginf=0.0
)

train_dataset = TensorDataset(
    torch.tensor(X_tr_scaled, dtype=torch.float32),
    torch.tensor(y_tr.values, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val.values, dtype=torch.float32),
)
test_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

nn_model = ConflictTabularResNet(input_dim=X_tr.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

best_val_loss = float("inf")
best_nn_state = None
patience = 10
patience_counter = 0

for epoch in range(1, 41):
    nn_model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = nn_model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            val_loss += loss.item() * len(batch_y)
    val_loss /= len(val_dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_nn_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

if best_nn_state is not None:
    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_state.items()})

nn_model.eval()
val_preds_nn_list = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        preds = nn_model(batch_x)
        val_preds_nn_list.append(preds.cpu().numpy())
val_preds_nn = np.clip(np.concatenate(val_preds_nn_list), 0, None)

test_preds_nn_list = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        preds = nn_model(batch_x)
        test_preds_nn_list.append(preds.cpu().numpy())
test_preds_nn = np.clip(np.concatenate(test_preds_nn_list), 0, None)

# 7. Validation Ensembling & Final Predictions
val_preds_matrix = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_nn])
test_preds_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])


def ensemble_loss(weights):
    w = np.array(weights)
    if np.sum(w) == 0:
        return 1e9
    w = w / np.sum(w)
    blended = np.dot(val_preds_matrix, w)
    return compute_rmse(y_val, np.clip(blended, 0, None))


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)

optimal_weights = (
    opt_res.x / np.sum(opt_res.x)
    if opt_res.success
    else np.array([1 / 3, 1 / 3, 1 / 3])
)

final_val_preds = np.clip(np.dot(val_preds_matrix, optimal_weights), 0, None)
final_test_preds = np.clip(np.dot(test_preds_matrix, optimal_weights), 0, None)
final_score = compute_rmse(y_val, final_val_preds)

# 8. Output Submission
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds), dtype=int), "fatalities": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")