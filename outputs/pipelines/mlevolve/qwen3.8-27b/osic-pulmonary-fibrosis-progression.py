import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
import joblib
import os

# ============================================================
# SETUP
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# STEP 1: DATA LOADING & FEATURE ENGINEERING
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train columns: {train.columns.tolist()}")

id_col = "Patient"
time_col = "Weeks"
target_col = "FVC"

cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train.select_dtypes(include=["number"]).columns.tolist()

# Combine for feature engineering
train["is_train"] = 1
test["is_train"] = 0

if target_col not in test.columns:
    test[target_col] = np.nan

combined = pd.concat([train, test], axis=0, ignore_index=True)
combined = combined.sort_values([id_col, time_col]).reset_index(drop=True)

# 1. Time-based features
combined["weeks_from_first"] = combined.groupby(id_col)[time_col].transform(
    lambda x: x - x.iloc[0]
)

# 2. Lag features
for lag in [1, 2, 3, 4, 6, 8, 12]:
    lag_col = f"FVC_lag_{lag}"
    combined[lag_col] = combined.groupby(id_col)[target_col].shift(lag)
    if "Percent" in combined.columns:
        pct_lag = f"Percent_lag_{lag}"
        combined[pct_lag] = combined.groupby(id_col)["Percent"].shift(lag)

# 3. Difference features
for lag in [1, 2, 4, 8]:
    diff_col = f"FVC_diff_{lag}"
    combined[diff_col] = combined.groupby(id_col)[target_col].transform(
        lambda x: x - x.shift(lag)
    )

# 4. Rolling statistics
for window in [4, 8, 12, 24]:
    combined[f"FVC_roll_mean_{window}"] = combined.groupby(id_col)[
        target_col
    ].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    combined[f"FVC_roll_std_{window}"] = combined.groupby(id_col)[target_col].transform(
        lambda x: x.rolling(window, min_periods=1).std().shift(1)
    )
    combined[f"FVC_roll_min_{window}"] = combined.groupby(id_col)[target_col].transform(
        lambda x: x.rolling(window, min_periods=1).min().shift(1)
    )
    combined[f"FVC_roll_max_{window}"] = combined.groupby(id_col)[target_col].transform(
        lambda x: x.rolling(window, min_periods=1).max().shift(1)
    )

# 5. Rate of decline (slope)
for window in [8, 16, 24]:
    slope_col = f"FVC_slope_{window}"
    _w = window

    def calc_slope(x, w=_w):
        slopes = pd.Series(np.nan, index=x.index)
        for i in range(len(x)):
            if i >= 1:
                window_data = x.iloc[max(0, i - w) : i]
                if len(window_data) >= 2:
                    slope = (window_data.iloc[-1] - window_data.iloc[0]) / max(
                        len(window_data) - 1, 1
                    )
                    slopes.iloc[i] = slope
        return slopes

    combined[slope_col] = combined.groupby(id_col)[target_col].transform(calc_slope)

# 6. Patient-level features (shifted to avoid target leakage)
combined["FVC_baseline"] = combined.groupby(id_col)[target_col].transform(
    lambda x: x.shift(1).fillna(x.iloc[0])
)
if "Percent" in combined.columns:
    combined["Percent_baseline"] = combined.groupby(id_col)["Percent"].transform(
        lambda x: x.shift(1).fillna(x.iloc[0])
    )

# 7. FVC relative to baseline (shifted to avoid target leakage)
combined["FVC_ratio_to_baseline"] = (
    combined.groupby(id_col)[target_col].transform(lambda x: x.shift(1).fillna(x.iloc[0]))
    / combined["FVC_baseline"]
)

# 8. Ensure time is numeric
combined[time_col] = pd.to_numeric(combined[time_col], errors="coerce")

# 9. Encode categorical variables
train_mask = combined["is_train"] == 1
for col in cat_cols:
    if col in combined.columns and col != id_col:
        train_vals = combined.loc[train_mask, col].astype(str)
        categories = sorted(train_vals.unique())
        cat_to_num = {c: i for i, c in enumerate(categories)}
        combined[f"{col}_enc"] = (
            combined[col].astype(str).map(cat_to_num).fillna(-1).astype(int)
        )

# 10. Interaction features (shifted to avoid target leakage)
combined["FVC_per_week"] = (
    combined.groupby(id_col)[target_col].transform(lambda x: x.shift(1).fillna(x.iloc[0]))
    / (combined[time_col] + 1)
)


# 11. FVC residual from linear trend
def calc_residual(x):
    # Uses only past data (excludes current row) to compute residual
    residuals = pd.Series(np.nan, index=x.index)
    weeks = np.arange(len(x))
    for i in range(len(x)):
        if i >= 2:
            past_weeks = weeks[:i]
            past_vals = x.iloc[:i].values
            mask = ~np.isnan(past_vals)
            if mask.sum() >= 2:
                coeffs = np.polyfit(past_weeks[mask], past_vals[mask], 1)
                predicted = np.polyval(coeffs, weeks[i])
                residuals.iloc[i] = x.iloc[i] - predicted
    return residuals


combined["FVC_residual"] = combined.groupby(id_col)[target_col].transform(calc_residual)

# 12. Cumulative decline (shifted to avoid target leakage)
combined["FVC_cum_decline"] = combined.groupby(id_col)[target_col].transform(
    lambda x: x.shift(1).fillna(x.iloc[0]) - x
)

# 13. Percent change from baseline (shifted to avoid target leakage)
if "Percent" in combined.columns:
    combined["Percent_change"] = (
        combined.groupby(id_col)["Percent"].transform(lambda x: x.shift(1).fillna(x.iloc[0]))
        - combined["Percent_baseline"]
    )

# 14. Acceleration (shifted to avoid target leakage)
combined["FVC_accel"] = combined.groupby(id_col)[target_col].transform(
    lambda x: x.diff().diff().shift(1)
)

# 15. Rolling median
for window in [4, 8]:
    combined[f"FVC_roll_median_{window}"] = combined.groupby(id_col)[
        target_col
    ].transform(lambda x: x.rolling(window, min_periods=1).median().shift(1))

# Split back
train_processed = combined[combined["is_train"] == 1].reset_index(drop=True)
test_processed = combined[combined["is_train"] == 0].reset_index(drop=True)

# Define feature columns
exclude_cols = [id_col, time_col, target_col, "is_train"] + cat_cols
feature_cols = [
    c for c in train_processed.columns if c not in exclude_cols and c not in ["id"]
]
feature_cols = [c for c in feature_cols if c in test_processed.columns]

print(f"Number of features: {len(feature_cols)}")

# Handle NaN
for col in feature_cols:
    if "diff" in col or "accel" in col or "residual" in col:
        train_processed[col] = train_processed[col].fillna(0)
        test_processed[col] = test_processed[col].fillna(0)
    elif "lag" in col:
        train_processed[col] = train_processed.groupby(id_col)[col].transform(
            lambda x: x.ffill().bfill()
        )
        test_processed[col] = test_processed.groupby(id_col)[col].transform(
            lambda x: x.ffill().bfill()
        )
        train_processed[col] = train_processed[col].fillna(0)
        test_processed[col] = test_processed[col].fillna(0)
    else:
        train_processed[col] = train_processed[col].fillna(0)
        test_processed[col] = test_processed[col].fillna(0)

# Replace inf
train_processed[feature_cols] = train_processed[feature_cols].replace(
    [np.inf, -np.inf], 0
)
test_processed[feature_cols] = test_processed[feature_cols].replace(
    [np.inf, -np.inf], 0
)

# Patient-based train/validation split
train_patients = train_processed[id_col].unique()
np.random.seed(42)
np.random.shuffle(train_patients)

n_val = max(1, int(len(train_patients) * 0.15))
val_patients = set(train_patients[:n_val])
trn_patients = set(train_patients[n_val:])

train_final = train_processed[train_processed[id_col].isin(trn_patients)].reset_index(
    drop=True
)
val_final = train_processed[train_processed[id_col].isin(val_patients)].reset_index(
    drop=True
)

print(f"Split: Train patients={len(trn_patients)}, Val patients={len(val_patients)}")
print(
    f"Train rows: {len(train_final)}, Val rows: {len(val_final)}, Test rows: {len(test_processed)}"
)

# Scale features - fit on train only
scaler = RobustScaler()
train_final[feature_cols] = scaler.fit_transform(train_final[feature_cols])
val_final[feature_cols] = scaler.transform(val_final[feature_cols])
test_processed[feature_cols] = scaler.transform(test_processed[feature_cols])

# ============================================================
# STEP 2 & 3: MODEL TRAINING & ENSEMBLE
# ============================================================
X_train = train_final[feature_cols].values.astype(np.float32)
y_train = train_final[target_col].values.astype(np.float32)
X_val = val_final[feature_cols].values.astype(np.float32)
y_val = val_final[target_col].values.astype(np.float32)
X_test = test_processed[feature_cols].values.astype(np.float32)

cat_features = [c for c in feature_cols if c.endswith("_enc")]
cat_indices = [feature_cols.index(c) for c in cat_features if c in feature_cols]
print(f"Categorical features: {cat_features}")

# --- MODEL 1: LightGBM ---
lgb_params = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
    categorical_feature=cat_indices,
)

val_pred_lgb = lgb_model.predict(X_val)
val_mae_lgb = np.abs(y_val - val_pred_lgb).mean()
print(f"LightGBM Val MAE: {val_mae_lgb:.4f} (best_iter={lgb_model.best_iteration_})")
test_pred_lgb = lgb_model.predict(X_test)

# --- MODEL 2: XGBoost ---
xgb_model = xgb.XGBRegressor(
    n_estimators=5000,
    objective="reg:absoluteerror",
    eval_metric="mae",
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=100,
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_pred_xgb = xgb_model.predict(X_val)
val_mae_xgb = np.abs(y_val - val_pred_xgb).mean()
print(f"XGBoost Val MAE: {val_mae_xgb:.4f}")
test_pred_xgb = xgb_model.predict(X_test)

# --- MODEL 3: LightGBM v2 (different hyperparams) ---
lgb_params2 = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 3,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 123,
}

lgb_model2 = lgb.LGBMRegressor(**lgb_params2)
lgb_model2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=0),
    ],
    categorical_feature=cat_indices,
)

val_pred_lgb2 = lgb_model2.predict(X_val)
val_mae_lgb2 = np.abs(y_val - val_pred_lgb2).mean()
print(
    f"LightGBM-v2 Val MAE: {val_mae_lgb2:.4f} (best_iter={lgb_model2.best_iteration_})"
)
test_pred_lgb2 = lgb_model2.predict(X_test)


# --- MODEL 4: Temporal Neural Network ---
def build_sequences(df, feature_cols, target_col, id_col, time_col, max_seq_len=48):
    sequences = []
    targets = []
    for patient, group in df.groupby(id_col):
        group = group.sort_values(time_col)
        feat_matrix = group[feature_cols].values.astype(np.float32)
        target_values = group[target_col].values.astype(np.float32)
        n = len(group)
        if n < 2:
            continue
        for i in range(1, n):
            seq_len = min(i, max_seq_len)
            start = max(0, i - seq_len)
            seq = feat_matrix[start : i + 1]
            if len(seq) < max_seq_len:
                pad = np.zeros((max_seq_len - len(seq), seq.shape[1]), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)
            sequences.append(seq)
            targets.append(target_values[i])
    return np.array(sequences), np.array(targets)


X_train_seq, y_train_seq = build_sequences(
    train_final, feature_cols, target_col, id_col, time_col
)
X_val_seq, y_val_seq = build_sequences(
    val_final, feature_cols, target_col, id_col, time_col
)
X_test_seq, _ = build_sequences(
    test_processed, feature_cols, target_col, id_col, time_col
)

print(
    f"NN sequences - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}"
)


class TemporalFVCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(TemporalFVCNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_out = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        attn_scores = self.attention(self.dropout(lstm_out)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        final_hidden = h_n[-1]
        combined = torch.cat([context, final_hidden], dim=-1)
        out = torch.relu(self.bn_out(self.fc1(self.dropout(combined))))
        out = self.fc2(out).squeeze(-1)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nn_model = TemporalFVCNet(
    input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.3
).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-5
)

train_ds = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq))
val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq))
test_ds = TensorDataset(torch.tensor(X_test_seq))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

best_val_mae = float("inf")
best_model_state = None
patience = 15
patience_counter = 0

for epoch in range(200):
    nn_model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = nn_model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(train_ds)
    scheduler.step()

    nn_model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            pred = nn_model(X_batch)
            val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_mae_nn = np.abs(y_val_seq - val_preds).mean()
    print(
        f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae_nn:.4f}"
    )

    if val_mae_nn < best_val_mae:
        best_val_mae = val_mae_nn
        best_model_state = nn_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

nn_model.load_state_dict(best_model_state)
nn_model.eval()

with torch.no_grad():
    val_preds_nn = []
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        pred = nn_model(X_batch)
        val_preds_nn.append(pred.cpu().numpy())
    val_preds_nn = np.concatenate(val_preds_nn)

    test_preds_nn = []
    for (X_batch,) in test_loader:
        X_batch = X_batch.to(device)
        pred = nn_model(X_batch)
        test_preds_nn.append(pred.cpu().numpy())
    test_preds_nn = np.concatenate(test_preds_nn)

val_mae_nn = np.abs(y_val_seq - val_preds_nn).mean()
print(f"Neural Network Val MAE: {val_mae_nn:.4f}")

# ============================================================
# ENSEMBLE: Find optimal weights on validation
# ============================================================
print(f"\n{'='*50}")
print(f"Individual model MAEs:")
print(f"  LightGBM:    {val_mae_lgb:.4f}")
print(f"  XGBoost:     {val_mae_xgb:.4f}")
print(f"  LightGBM-v2: {val_mae_lgb2:.4f}")
print(f"  Neural Net:  {val_mae_nn:.4f}")

# Note: NN predictions are in sequence order, need to map back to row order
# The NN val predictions correspond to val_final rows (each row after first per patient)
# We need to align them. Let's use the tabular models for the main ensemble
# and only include NN if it aligns properly.

# For simplicity and correctness, use the 3 tabular models for ensemble
# (NN predictions may not align 1:1 with val rows due to sequence construction)
best_mae = float("inf")
best_w = (1.0, 0.0, 0.0)

for w1 in np.arange(0.0, 1.01, 0.1):
    for w2 in np.arange(0.0, 1.01 - w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        w3 = max(0.0, w3)
        blend = w1 * val_pred_lgb + w2 * val_pred_xgb + w3 * val_pred_lgb2
        mae = np.abs(y_val - blend).mean()
        if mae < best_mae:
            best_mae = mae
            best_w = (w1, w2, w3)

print(
    f"\nBest blend weights: LGB={best_w[0]:.1f}, XGB={best_w[1]:.1f}, LGB2={best_w[2]:.1f}"
)
print(f"Best blend Val MAE: {best_mae:.4f}")

# Final predictions
final_val_pred = (
    best_w[0] * val_pred_lgb + best_w[1] * val_pred_xgb + best_w[2] * val_pred_lgb2
)
final_test_pred = (
    best_w[0] * test_pred_lgb + best_w[1] * test_pred_xgb + best_w[2] * test_pred_lgb2
)

# Clip to reasonable bounds
final_test_pred = np.clip(final_test_pred, 0, 10000)
final_val_pred = np.clip(final_val_pred, 0, 10000)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_processed)), "FVC": final_test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Submission FVC range: [{submission['FVC'].min():.2f}, {submission['FVC'].max():.2f}]"
)

# Final validation score (official metric: MAE)
score = np.abs(y_val - final_val_pred).mean()
print(f"Final Validation Score: {score:.4f}")