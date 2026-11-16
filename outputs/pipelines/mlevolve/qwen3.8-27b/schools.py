import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import json
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CREATE DIRECTORIES
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

TARGET = "Enrollment_FT_Teachers_Ratio"
ID_COL = "id"

if ID_COL not in train.columns:
    train[ID_COL] = range(len(train))
if ID_COL not in test.columns:
    test[ID_COL] = range(len(test))

y_train_full = train[TARGET].copy()
X_train_full = train.drop(columns=[TARGET])
X_test = test.copy()

train_ids = X_train_full[ID_COL].copy()
test_ids = X_test[ID_COL].copy()

X_train_full = X_train_full.drop(columns=[ID_COL])
X_test = X_test.drop(columns=[ID_COL])

# Identify column types
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(
    include=["object", "category"]
).columns.tolist()
bool_cols = [c for c in X_train_full.columns if X_train_full[c].dtype == bool]

print(
    f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}, Bool cols: {len(bool_cols)}"
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================


def engineer_features(X, y=None, is_train=True, fit_params=None):
    X = X.copy()
    if fit_params is None:
        fit_params = {}

    # Handle boolean columns
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(np.int8)

    # Categorical encoding
    for col in categorical_cols:
        if col not in X.columns:
            continue
        if is_train:
            fit_params[f"card_{col}"] = X[col].nunique()
        n_unique = fit_params.get(f"card_{col}", X[col].nunique())
        if n_unique <= 15:
            dummies = pd.get_dummies(
                X[col], prefix=col, drop_first=False, dtype=np.float32
            )
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:
            X[col] = X[col].astype("category")
            X[col] = X[col].cat.codes.astype(np.float32)

    # Numeric feature engineering
    for col in numeric_cols:
        if col not in X.columns:
            continue
        if is_train:
            col_data = X[col].dropna()
            if len(col_data) > 0 and col_data.min() > 0 and col_data.skew() > 1.0:
                fit_params[f"log_{col}"] = True
            else:
                fit_params[f"log_{col}"] = False
        if fit_params.get(f"log_{col}", False):
            X[col] = np.log1p(X[col].clip(lower=0))
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    # Ratio features
    enroll_cols = [
        c
        for c in X.columns
        if "enroll" in c.lower() or "student" in c.lower() or "pupil" in c.lower()
    ]
    teacher_cols = [
        c for c in X.columns if "teacher" in c.lower() or "staff" in c.lower()
    ]
    for e_col in enroll_cols[:5]:
        for t_col in teacher_cols[:5]:
            if e_col in X.columns and t_col in X.columns:
                X[f"ratio_{e_col}_per_{t_col}"] = X[e_col] / (X[t_col] + 1e-8)

    # Square features
    key_numeric = [c for c in numeric_cols if c in X.columns][:10]
    for col in key_numeric:
        if col in X.columns:
            X[f"{col}_sq"] = X[col] ** 2

    # Group-level aggregations
    id_like_cols = [
        c
        for c in X.columns
        if any(
            k in c.lower()
            for k in ["school", "district", "region", "state", "county", "zip", "code"]
        )
    ]
    if len(id_like_cols) > 0 and is_train:
        group_col = id_like_cols[0]
        if group_col in X.columns and X[group_col].nunique() < len(X) * 0.9:
            for num_col in [
                c
                for c in X.select_dtypes(include=[np.number]).columns
                if c != group_col
            ][:10]:
                if num_col in X.columns:
                    X[f"{num_col}_grp_mean"] = X.groupby(group_col)[num_col].transform(
                        "mean"
                    )
                    X[f"{num_col}_grp_std"] = (
                        X.groupby(group_col)[num_col].transform("std").fillna(0)
                    )

    # Impute missing values
    if is_train:
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isnull().any():
                fit_params[f"imp_{col}"] = X[col].median()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            if X[col].isnull().any():
                fit_params[f"imp_{col}"] = (
                    X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "UNKNOWN"
                )

    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(fit_params.get(f"imp_{col}", 0))
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(fit_params.get(f"imp_{col}", "UNKNOWN"))

    # Scale numeric features
    if is_train:
        for col in X.select_dtypes(include=[np.number]).columns:
            col_std = X[col].std()
            col_mean = X[col].mean()
            fit_params[f"scale_{col}"] = (col_mean, col_std if col_std > 1e-10 else 1.0)
        for col in X.select_dtypes(include=[np.number]).columns:
            mean, std = fit_params.get(f"scale_{col}", (0, 1))
            X[col] = (X[col] - mean) / std
    else:
        for col in X.select_dtypes(include=[np.number]).columns:
            if f"scale_{col}" in fit_params:
                mean, std = fit_params[f"scale_{col}"]
                X[col] = (X[col] - mean) / std

    # Cast to float32
    for col in X.columns:
        if X[col].dtype != np.float32:
            try:
                X[col] = X[col].astype(np.float32)
            except:
                pass

    return X, fit_params


# ============================================================
# TRAIN/VALIDATION SPLIT
# ============================================================
valid_ratio = 0.15
valid_indices = np.random.RandomState(42).choice(
    len(X_train_full), size=int(len(X_train_full) * valid_ratio), replace=False
)
valid_mask = np.zeros(len(X_train_full), dtype=bool)
valid_mask[valid_indices] = True

X_train_split = X_train_full[~valid_mask].copy()
X_val_split = X_train_full[valid_mask].copy()
y_train_split = y_train_full[~valid_mask].copy()
y_val_split = y_train_full[valid_mask].copy()

print(f"Train split: {X_train_split.shape[0]}, Val split: {X_val_split.shape[0]}")

# ============================================================
# APPLY FEATURE ENGINEERING
# ============================================================
print("Applying feature engineering...")
X_train_feat, fit_params = engineer_features(
    X_train_split, y_train_split, is_train=True
)
X_val_feat, _ = engineer_features(
    X_val_split, y_val_split, is_train=False, fit_params=fit_params
)
X_test_feat, _ = engineer_features(X_test, None, is_train=False, fit_params=fit_params)

# Align columns
train_cols = X_train_feat.columns.tolist()
for col in train_cols:
    if col not in X_val_feat.columns:
        X_val_feat[col] = 0
    if col not in X_test_feat.columns:
        X_test_feat[col] = 0
X_val_feat = X_val_feat[train_cols]
X_test_feat = X_test_feat[train_cols]

X_train_feat = X_train_feat.fillna(0)
X_val_feat = X_val_feat.fillna(0)
X_test_feat = X_test_feat.fillna(0)

print(
    f"Feature matrix shapes: Train={X_train_feat.shape}, Val={X_val_feat.shape}, Test={X_test_feat.shape}"
)

# Save to working directory
X_train_feat.to_parquet("./working/X_train.parquet", index=False)
X_val_feat.to_parquet("./working/X_val.parquet", index=False)
X_test_feat.to_parquet("./working/X_test.parquet", index=False)
y_train_split.to_frame(TARGET).to_parquet("./working/y_train.parquet", index=False)
y_val_split.to_frame(TARGET).to_parquet("./working/y_val.parquet", index=False)
pd.DataFrame({"id": test_ids}).to_parquet("./working/test_ids.parquet", index=False)

# ============================================================
# STAGE 1: TRAIN LIGHTGBM (log-space)
# ============================================================
print("\n" + "=" * 60)
print("STAGE 1: Training LightGBM (log-space)")
print("=" * 60)

EPS = 1e-6
y_train_log = np.log1p(np.clip(y_train_split.values, 0, None) + EPS)
y_val_log = np.log1p(np.clip(y_val_split.values, 0, None) + EPS)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train_feat,
    y_train_log,
    eval_set=[(X_val_feat, y_val_log)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

print(f"  Best iteration: {lgb_model.best_iteration_}")

val_pred_lgbm_log = lgb_model.predict(X_val_feat)
val_pred_lgbm = np.clip(np.expm1(np.maximum(val_pred_lgbm_log, 0)), 0, None)
test_pred_lgbm_log = lgb_model.predict(X_test_feat)
test_pred_lgbm = np.clip(np.expm1(np.maximum(test_pred_lgbm_log, 0)), 0, None)

rmse_lgbm = np.sqrt(mean_squared_error(y_val_split, val_pred_lgbm))
print(f"  LGBM RMSE (original scale): {rmse_lgbm:.4f}")

# ============================================================
# STAGE 2: TRAIN MLP ON RESIDUALS
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: Training MLP (residual model)")
print("=" * 60)

train_pred_lgbm = np.clip(np.expm1(np.maximum(lgb_model.predict(X_train_feat), 0)), 0, None)
y_train_residual = y_train_split.values - train_pred_lgbm
y_val_residual = y_val_split.values - val_pred_lgbm

print(
    f"  Residual stats - mean: {y_train_residual.mean():.4f}, std: {y_train_residual.std():.4f}"
)

n_features = X_train_feat.shape[1]
X_train_t = torch.tensor(X_train_feat.values, dtype=torch.float32)
y_train_res_t = torch.tensor(y_train_residual, dtype=torch.float32)
X_val_t = torch.tensor(X_val_feat.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test_feat.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_res_t)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super(TabularMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out.squeeze(-1)


mlp_model = TabularMLP(
    input_dim=n_features, hidden_dims=[256, 128, 64], dropout_rate=0.2
).to(device)

optimizer = AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epochs = 150
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
criterion = nn.HuberLoss(delta=1.0)

best_val_rmse = float("inf")
best_model_state = None
patience_counter = 0
patience = 15

for epoch in range(num_epochs):
    mlp_model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = mlp_model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)

    scheduler.step()
    avg_loss = epoch_loss / len(train_dataset)

    mlp_model.eval()
    with torch.no_grad():
        val_pred_res = mlp_model(X_val_t.to(device)).cpu().numpy()
        val_pred_combined = val_pred_lgbm + val_pred_res
        val_pred_combined = np.clip(val_pred_combined, 0, None)
        val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred_combined))

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.6f} | Val RMSE: {val_rmse:.4f}"
        )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = {
            k: v.cpu().clone() for k, v in mlp_model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

mlp_model.load_state_dict(best_model_state)
mlp_model.eval()
print(f"  Best MLP val RMSE: {best_val_rmse:.4f}")

# ============================================================
# FINAL PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FINAL PREDICTIONS")
print("=" * 60)


def predict_combined(model, X_lgbm_pred, X_torch):
    model.eval()
    with torch.no_grad():
        X_dev = X_torch.to(device)
        pred_res = model(X_dev).cpu().numpy()
    pred_final = X_lgbm_pred + pred_res
    pred_final = np.clip(pred_final, 0, None)
    return pred_final


val_pred_final = predict_combined(mlp_model, val_pred_lgbm, X_val_t)
test_pred_final = predict_combined(mlp_model, test_pred_lgbm, X_test_t)

# Ensure no NaN/Inf in predictions
test_pred_final = np.nan_to_num(test_pred_final, nan=0.0, posinf=0.0, neginf=0.0)
val_pred_final = np.nan_to_num(val_pred_final, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# EVALUATE
# ============================================================
final_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred_final))
print(f"\n  LGBM only RMSE: {rmse_lgbm:.4f}")
print(f"  Combined RMSE: {final_rmse:.4f}")
print(f"  Improvement: {rmse_lgbm - final_rmse:.4f}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Enrollment_FT_Teachers_Ratio": test_pred_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"  Predictions range: [{submission['Enrollment_FT_Teachers_Ratio'].min():.4f}, {submission['Enrollment_FT_Teachers_Ratio'].max():.4f}]"
)
print(f"  Predictions mean: {submission['Enrollment_FT_Teachers_Ratio'].mean():.4f}")

score = final_rmse
print(f"Final Validation Score: {score}")