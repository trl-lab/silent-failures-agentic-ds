import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "CAPACITY"
id_col = None

for col in train.columns:
    if col.lower() in ["id", "index", "row_id", "sample_id"]:
        id_col = col
        break

if id_col is None:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

y_train = train[target_col].values
train = train.drop(columns=[target_col])

cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train.select_dtypes(include=["number"]).columns.tolist()

if id_col is not None:
    if id_col in cat_cols:
        cat_cols.remove(id_col)
    if id_col in num_cols:
        num_cols.remove(id_col)

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
y_binned = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
train_data, val_data = train_test_split(
    train, test_size=0.15, random_state=42, stratify=y_binned
)
y_val = y_train[val_data.index]
y_train_split = y_train[train_data.index]

print(
    f"Train split: {train_data.shape}, Val split: {val_data.shape}, Test: {test.shape}"
)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def basic_feature_engineering(df, cat_cols, num_cols):
    """Target-independent feature engineering."""
    df = df.copy()

    # Handle missing values
    for col in num_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING")

    # Interaction features between numerical columns
    if len(num_cols) >= 2:
        for i in range(min(len(num_cols), 5)):
            for j in range(i + 1, min(len(num_cols), 5)):
                c1, c2 = num_cols[i], num_cols[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                    df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-8)

    # Polynomial features
    for col in num_cols[:5]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2
            df[f"{col}_log"] = np.log1p(np.abs(df[col])) * np.sign(df[col])

    # Categorical interactions
    if len(cat_cols) >= 2:
        for i in range(min(len(cat_cols), 4)):
            for j in range(i + 1, min(len(cat_cols), 4)):
                c1, c2 = cat_cols[i], cat_cols[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    # Categorical + numerical interactions
    for c in cat_cols[:3]:
        for n in num_cols[:3]:
            if c in df.columns and n in df.columns:
                df[f"{c}_x_{n}"] = df[c].astype(str) + "_" + (df[n] // 1).astype(str)

    return df


print("Applying basic feature engineering...")
train_data = basic_feature_engineering(train_data, cat_cols, num_cols)
val_data = basic_feature_engineering(val_data, cat_cols, num_cols)
test = basic_feature_engineering(test, cat_cols, num_cols)

# Update column lists
new_cat_cols = [
    c
    for c in train_data.columns
    if train_data[c].dtype == "object" or train_data[c].dtype.name == "category"
]
new_num_cols = [
    c
    for c in train_data.columns
    if train_data[c].dtype in ["int64", "float64", "int32", "float32"]
]

if id_col is not None:
    if id_col in new_cat_cols:
        new_cat_cols.remove(id_col)
    if id_col in new_num_cols:
        new_num_cols.remove(id_col)

print(f"After basic FE - Cat: {len(new_cat_cols)}, Num: {len(new_num_cols)}")


# ============================================================
# 5. TARGET ENCODING (K-Fold to prevent leakage)
# ============================================================
def target_encode_kfold(train_df, val_df, test_df, cat_cols, target, k=5, seed=42):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for col in cat_cols:
        if col not in train_df.columns:
            continue
        train_df[f"{col}_te"] = 0.0
        global_mean = target.mean()

        for train_idx, val_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            fold_target = target.iloc[train_idx]
            temp_df = pd.DataFrame(
                {"cat": fold_train[col].values, "target": fold_target.values}
            )
            cat_means = temp_df.groupby("cat")["target"].mean()
            train_df.iloc[val_idx, train_df.columns.get_loc(f"{col}_te")] = (
                fold_val[col].map(cat_means).fillna(global_mean).values
            )

        full_means = pd.DataFrame(
            {"cat": train_df[col].values, "target": target.values}
        )
        full_cat_means = full_means.groupby("cat")["target"].mean()
        val_df[f"{col}_te"] = val_df[col].map(full_cat_means).fillna(global_mean).values
        test_df[f"{col}_te"] = (
            test_df[col].map(full_cat_means).fillna(global_mean).values
        )

    return train_df, val_df, test_df


print("Applying target encoding...")
y_train_split_series = pd.Series(y_train_split, index=train_data.index)
train_data, val_data, test = target_encode_kfold(
    train_data, val_data, test, new_cat_cols, y_train_split_series, k=5
)

# ============================================================
# 6. SCALE NUMERICAL FEATURES
# ============================================================
scaler = RobustScaler()
scaler.fit(train_data[new_num_cols].values)
train_data[new_num_cols] = scaler.transform(train_data[new_num_cols].values)
val_data[new_num_cols] = scaler.transform(val_data[new_num_cols].values)
test[new_num_cols] = scaler.transform(test[new_num_cols].values)

# Add target back
train_data[target_col] = y_train_split
val_data[target_col] = y_val

# Save processed data
train_data.to_parquet("./working/train_processed.parquet", index=False)
val_data.to_parquet("./working/val_processed.parquet", index=False)
test.to_parquet("./working/test_processed.parquet", index=False)
joblib.dump(scaler, "./working/scaler.pkl")

feature_columns = [c for c in train_data.columns if c != target_col and c != id_col]
metadata = {
    "cat_cols": new_cat_cols,
    "num_cols": new_num_cols,
    "id_col": id_col,
    "target_col": target_col,
    "feature_columns": feature_columns,
}
joblib.dump(metadata, "./working/metadata.pkl")

print(f"Feature columns: {len(feature_columns)}")

# ============================================================
# 7. QUICK LIGHTGBM BASELINE (sanity check)
# ============================================================
print("\n--- LightGBM Baseline ---")
lgb_cat_cols = [c for c in feature_columns if c in new_cat_cols]
lgb_num_cols = [c for c in feature_columns if c in new_num_cols]

train_X_lgb = train_data[feature_columns].copy()
val_X_lgb = val_data[feature_columns].copy()
test_X_lgb = test[feature_columns].copy()

for col in lgb_cat_cols:
    train_X_lgb[col] = train_X_lgb[col].astype("category")
    val_X_lgb[col] = val_X_lgb[col].astype("category")
    test_X_lgb[col] = test_X_lgb[col].astype("category")

dtrain = lgb.Dataset(
    train_X_lgb, label=train_data[target_col], categorical_feature=lgb_cat_cols
)
dval = lgb.Dataset(
    val_X_lgb,
    label=val_data[target_col],
    categorical_feature=lgb_cat_cols,
    reference=dtrain,
)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
)
lgb_val_pred = lgb_model.predict(val_X_lgb)
lgb_test_pred = lgb_model.predict(test_X_lgb)
lgb_val_rmse = np.sqrt(mean_squared_error(val_data[target_col], lgb_val_pred))
print(f"LightGBM Validation RMSE: {lgb_val_rmse:.6f}")


# ============================================================
# 8. TABNET MODEL DEFINITION
# ============================================================
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, output_dim, n_shared=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_shared)
        self.fc2 = nn.Linear(n_shared, output_dim * 2)
        self.bn = nn.BatchNorm1d(output_dim * 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.bn(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class SparseFeatureSelection(nn.Module):
    def __init__(self, input_dim, n_shared=8):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_shared)
        self.bn = nn.BatchNorm1d(n_shared)
        self.fc_out = nn.Linear(n_shared, input_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.bn(x)
        x = self.fc_out(x)
        return F.softmax(x, dim=-1)


class TabNetStep(nn.Module):
    def __init__(self, input_dim, d, n_shared=8):
        super().__init__()
        self.sfs = SparseFeatureSelection(input_dim, n_shared)
        self.feb = FeatureBlock(input_dim, d, n_shared)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x, prior_scale):
        attn = self.sfs(x)
        attn = attn * prior_scale
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        x_transformed = self.feb(x * attn)
        x_transformed = self.bn(x_transformed)
        return x_transformed, attn


class TabNetRegressor(nn.Module):
    def __init__(self, input_dim, d=32, n_steps=3, n_shared=8, dropout=0.15):
        super().__init__()
        self.d = d
        self.n_steps = n_steps
        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.steps = nn.ModuleList(
            [TabNetStep(input_dim, d, n_shared) for _ in range(n_steps)]
        )
        self.output_fc = nn.Linear(d * n_steps, 16)
        self.output_bn = nn.BatchNorm1d(16)
        self.output_dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_bn(x)
        outputs = []
        prior_scale = torch.ones(x.shape[0], x.shape[1], device=x.device)
        for step in self.steps:
            out, attn = step(x, prior_scale)
            outputs.append(out)
            prior_scale = 1.0 - attn
        x_out = torch.cat(outputs, dim=-1)
        x_out = self.output_fc(x_out)
        x_out = self.output_bn(x_out)
        x_out = F.relu(x_out)
        x_out = self.output_dropout(x_out)
        x_out = self.output_head(x_out)
        return x_out.squeeze(-1)


class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


# ============================================================
# 9. PREPARE TENSORS FOR TABNET
# ============================================================
# Encode categoricals as integers
cat_mappings = {}
for col in new_cat_cols:
    if col in train_data.columns:
        categories = sorted(train_data[col].astype(str).unique())
        mapping = {c: i for i, c in enumerate(categories)}
        cat_mappings[col] = mapping


def prepare_tensor(df, feature_cols, cat_cols, cat_mappings):
    X = df[feature_cols].copy()
    for col in cat_cols:
        if col in X.columns:
            if col in cat_mappings:
                X[col] = X[col].astype(str).map(cat_mappings[col]).fillna(0).astype(int)
            else:
                X[col] = 0
    for col in X.columns:
        if X[col].dtype == object or (
            hasattr(X[col].dtype, "name") and X[col].dtype.name == "category"
        ):
            X[col] = (
                X[col]
                .astype(str)
                .map({v: i for i, v in enumerate(X[col].unique())})
                .fillna(0)
                .astype(int)
            )
    X = X.fillna(0).astype(np.float32)
    return torch.tensor(X.values, dtype=torch.float32)


X_train = prepare_tensor(train_data, feature_columns, new_cat_cols, cat_mappings)
X_val = prepare_tensor(val_data, feature_columns, new_cat_cols, cat_mappings)
X_test = prepare_tensor(test, feature_columns, new_cat_cols, cat_mappings)

y_train_t = torch.tensor(train_data[target_col].values, dtype=torch.float32)
y_val_t = torch.tensor(val_data[target_col].values, dtype=torch.float32)

y_mean = y_train_t.mean().item()
y_std = y_train_t.std().item() + 1e-8
y_train_norm = (y_train_t - y_mean) / y_std
y_val_norm = (y_val_t - y_mean) / y_std

print(f"\nTabNet Input dim: {X_train.shape[1]}")
print(f"Target stats: mean={y_mean:.4f}, std={y_std:.4f}")

# ============================================================
# 10. TABNET TRAINING
# ============================================================
input_dim = X_train.shape[1]
d = 32
n_steps = 3
n_shared = 8
dropout = 0.15

model = TabNetRegressor(
    input_dim=input_dim, d=d, n_steps=n_steps, n_shared=n_shared, dropout=dropout
)
n_params = sum(p.numel() for p in model.parameters())
print(f"TabNetRegressor params: {n_params:,}")

criterion = RMSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))

num_epochs = 400
batch_size = 32
patience = 40
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

train_dataset = TensorDataset(X_train, y_train_norm)
val_dataset = TensorDataset(X_val, y_val_norm)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

print(f"\nTraining TabNetRegressor...")
print(
    f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Batch: {batch_size}, Epochs: {num_epochs}"
)

best_val_rmse = float("inf")
best_model_state = None
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    train_loss /= max(n_batches, 1)

    model.eval()
    val_loss = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item()
            n_val_batches += 1
    val_loss /= max(n_val_batches, 1)

    val_rmse_actual = val_loss * y_std
    train_rmse_actual = train_loss * y_std
    scheduler.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | Train RMSE: {train_rmse_actual:.4f} | Val RMSE: {val_rmse_actual:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    if val_rmse_actual < best_val_rmse:
        best_val_rmse = val_rmse_actual
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1}. Best Val RMSE: {best_val_rmse:.4f}"
            )
            break

# ============================================================
# 11. FINAL PREDICTIONS & SUBMISSION
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    val_pred_norm = model(X_val).numpy()
    val_pred = val_pred_norm * y_std + y_mean

    test_pred_norm = model(X_test).numpy()
    test_pred = test_pred_norm * y_std + y_mean

val_rmse_final = np.sqrt(mean_squared_error(y_val_t.numpy(), val_pred))
print(f"\nTabNet Validation RMSE: {val_rmse_final:.6f}")

# Use the better of LightGBM and TabNet for submission
# Ensemble: average predictions
ensemble_val_pred = (lgb_val_pred + val_pred) / 2.0
ensemble_val_rmse = np.sqrt(mean_squared_error(y_val_t.numpy(), ensemble_val_pred))
print(f"Ensemble Validation RMSE: {ensemble_val_rmse:.6f}")

# Choose best approach
if ensemble_val_rmse < min(lgb_val_rmse, val_rmse_final):
    final_test_pred = (lgb_test_pred + test_pred) / 2.0
    final_val_rmse = ensemble_val_rmse
    print("Using ensemble for submission")
elif lgb_val_rmse < val_rmse_final:
    final_test_pred = lgb_test_pred
    final_val_rmse = lgb_val_rmse
    print("Using LightGBM for submission")
else:
    final_test_pred = test_pred
    final_val_rmse = val_rmse_final
    print("Using TabNet for submission")

# Save submission
if id_col is not None and id_col in test.columns:
    submission = pd.DataFrame({"id": test[id_col].values, "CAPACITY": final_test_pred})
else:
    submission = pd.DataFrame(
        {"id": range(len(final_test_pred)), "CAPACITY": final_test_pred}
    )

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())

print(f"Final Validation Score: {final_val_rmse}")
