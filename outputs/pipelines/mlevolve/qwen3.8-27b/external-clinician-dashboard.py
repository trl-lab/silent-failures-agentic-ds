import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings
import json

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(
    f"Target 'Score' stats: mean={train['Score'].mean():.4f}, std={train['Score'].std():.4f}, min={train['Score'].min():.4f}, max={train['Score'].max():.4f}"
)

# ============================================================
# 2. SPLIT FIRST (before any processing)
# ============================================================
target_col = "Score"
id_col = "id" if "id" in train.columns else None

y = train[target_col].values
X = train.drop(columns=[target_col])

if id_col:
    X = X.drop(columns=[id_col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")

# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in X_train.columns if X_train[c].dtype == bool]

for c in bool_cols:
    X_train[c] = X_train[c].astype(int)
    X_val[c] = X_val[c].astype(int)
    if c in cat_cols:
        cat_cols.remove(c)
    if c not in num_cols:
        num_cols.append(c)

print(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

# ============================================================
# 4. HANDLE MISSING VALUES (fit on train only)
# ============================================================
print("\nHandling missing values...")
for c in X_train.columns:
    miss_rate = X_train[c].isna().mean()
    if miss_rate > 0:
        print(f"  {c}: {miss_rate:.2%} missing")

for c in X_train.columns:
    miss_rate = X_train[c].isna().mean()
    if 0 < miss_rate < 0.95:
        X_train[f"{c}_missing"] = X_train[c].isna().astype(int)
        X_val[f"{c}_missing"] = X_val[c].isna().astype(int)

num_cols_updated = [
    c
    for c in X_train.columns
    if X_train[c].dtype in [np.float64, np.int64, np.int32, np.float32, bool]
    and X_train[c].nunique() > 2
]
cat_cols_updated = [
    c for c in X_train.columns if X_train[c].dtype in ["object", "category"]
]

num_imputer = SimpleImputer(strategy="median")
X_train[num_cols_updated] = num_imputer.fit_transform(X_train[num_cols_updated])
X_val[num_cols_updated] = num_imputer.transform(X_val[num_cols_updated])

for c in cat_cols_updated:
    X_train[c] = X_train[c].fillna("missing")
    X_val[c] = X_val[c].fillna("missing")

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
print("\nEngineering features...")

# Log transform for skewed numeric features
skewed_cols = []
for c in num_cols_updated:
    if c in X_train.columns:
        skew = X_train[c].skew()
        if abs(skew) > 1.0 and X_train[c].min() > -1:
            skewed_cols.append(c)
            X_train[c] = np.log1p(X_train[c].clip(lower=0))
            X_val[c] = np.log1p(X_val[c].clip(lower=0))

print(f"  Log-transformed {len(skewed_cols)} skewed features")

# Ratio features
top_num = [c for c in num_cols_updated if c in X_train.columns][:15]
ratio_features = []
for i in range(len(top_num)):
    for j in range(i + 1, min(i + 5, len(top_num))):
        c1, c2 = top_num[i], top_num[j]
        if X_train[c2].abs().sum() > 0:
            ratio_name = f"{c1}_div_{c2}"
            X_train[ratio_name] = X_train[c1] / (X_train[c2].abs() + 1e-8)
            X_val[ratio_name] = X_val[c1] / (X_val[c2].abs() + 1e-8)
            ratio_features.append(ratio_name)

print(f"  Created {len(ratio_features)} ratio features")

# Interaction features
interaction_features = []
for i in range(len(top_num)):
    for j in range(i + 1, min(i + 3, len(top_num))):
        c1, c2 = top_num[i], top_num[j]
        inter_name = f"{c1}_x_{c2}"
        X_train[inter_name] = X_train[c1] * X_train[c2]
        X_val[inter_name] = X_val[c1] * X_val[c2]
        interaction_features.append(inter_name)

print(f"  Created {len(interaction_features)} interaction features")

# Aggregate features
if len(num_cols_updated) > 2:
    active_num = [c for c in num_cols_updated if c in X_train.columns]
    X_train["num_mean"] = X_train[active_num].mean(axis=1)
    X_val["num_mean"] = X_val[active_num].mean(axis=1)
    X_train["num_std"] = X_train[active_num].std(axis=1)
    X_val["num_std"] = X_val[active_num].std(axis=1)
    X_train["num_max"] = X_train[active_num].max(axis=1)
    X_val["num_max"] = X_val[active_num].max(axis=1)
    X_train["num_min"] = X_train[active_num].min(axis=1)
    X_val["num_min"] = X_val[active_num].min(axis=1)
    X_train["num_range"] = X_train["num_max"] - X_train["num_min"]
    X_val["num_range"] = X_val["num_max"] - X_val["num_min"]
    print("  Created aggregate features (mean, std, max, min, range)")

# Frequency encoding
for c in cat_cols_updated:
    if c in X_train.columns:
        freq_map = X_train[c].value_counts(normalize=True)
        X_train[f"{c}_freq"] = X_train[c].map(freq_map).fillna(0)
        X_val[f"{c}_freq"] = X_val[c].map(freq_map).fillna(0)

# Target encoding (smoothed, fit on train only)
for c in cat_cols_updated:
    if c in X_train.columns:
        global_mean = y_train.mean()
        agg = pd.DataFrame({"cat": X_train[c], "target": y_train}).groupby("cat")[
            "target"
        ]
        counts = agg.count()
        means = agg.mean()
        smoothing = 10
        smooth_enc = (counts * means + smoothing * global_mean) / (counts + smoothing)
        X_train[f"{c}_target_enc"] = X_train[c].map(smooth_enc).fillna(global_mean)
        X_val[f"{c}_target_enc"] = X_val[c].map(smooth_enc).fillna(global_mean)

print(
    f"  Created frequency and target encodings for {len(cat_cols_updated)} categorical features"
)

# Categorical interactions
if len(cat_cols_updated) > 1:
    cat_pairs = list(zip(cat_cols_updated[:5], cat_cols_updated[1:6]))
    for c1, c2 in cat_pairs:
        if c1 in X_train.columns and c2 in X_train.columns:
            X_train[f"{c1}_x_{c2}"] = (
                X_train[c1].astype(str) + "_" + X_train[c2].astype(str)
            )
            X_val[f"{c1}_x_{c2}"] = X_val[c1].astype(str) + "_" + X_val[c2].astype(str)
    print(f"  Created {len(cat_pairs)} categorical interaction features")

# ============================================================
# 6. ENCODE CATEGORICALS
# ============================================================
final_cat_cols = [
    c for c in X_train.columns if X_train[c].dtype in ["object", "category"]
]
final_num_cols = [c for c in X_train.columns if c not in final_cat_cols]

print(
    f"\nFinal: {len(final_num_cols)} numeric, {len(final_cat_cols)} categorical features"
)

label_encoders = {}
for c in final_cat_cols:
    le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[[c]] = le.fit_transform(X_train[[c]].astype(str))
    X_val[[c]] = le.transform(X_val[[c]].astype(str))
    label_encoders[c] = le
    final_num_cols.append(c)

final_cat_cols = []

# ============================================================
# 7. SCALE FEATURES (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[final_num_cols]),
    columns=final_num_cols,
    index=X_train.index,
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val[final_num_cols]),
    columns=final_num_cols,
    index=X_val.index,
)

X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val_scaled = X_val_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Scaled features: {X_train_scaled.shape[1]}")

# ============================================================
# 8. PROCESS TEST DATA
# ============================================================
print("\nProcessing test data...")
X_test = test.drop(columns=[target_col] if target_col in test.columns else [])
if id_col and id_col in X_test.columns:
    X_test = X_test.drop(columns=[id_col])

# Missing indicators
for c in X_train.columns:
    if f"{c}_missing" in X_train.columns:
        if c in X_test.columns:
            X_test[f"{c}_missing"] = X_test[c].isna().astype(int)
        else:
            X_test[f"{c}_missing"] = 0

# Impute
test_num_cols = [c for c in X_test.columns if c in num_imputer.feature_names_in_]
if len(test_num_cols) > 0:
    X_test[test_num_cols] = num_imputer.transform(X_test[test_num_cols])

for c in cat_cols_updated:
    if c in X_test.columns:
        X_test[c] = X_test[c].fillna("missing")

# Log transform
for c in skewed_cols:
    if c in X_test.columns:
        X_test[c] = np.log1p(X_test[c].clip(lower=0))

# Ratio features
for i in range(len(top_num)):
    for j in range(i + 1, min(i + 5, len(top_num))):
        c1, c2 = top_num[i], top_num[j]
        if c1 in X_test.columns and c2 in X_test.columns:
            X_test[f"{c1}_div_{c2}"] = X_test[c1] / (X_test[c2].abs() + 1e-8)

# Interaction features
for i in range(len(top_num)):
    for j in range(i + 1, min(i + 3, len(top_num))):
        c1, c2 = top_num[i], top_num[j]
        if c1 in X_test.columns and c2 in X_test.columns:
            X_test[f"{c1}_x_{c2}"] = X_test[c1] * X_test[c2]

# Aggregate features
if len(num_cols_updated) > 2:
    active_num_test = [c for c in num_cols_updated if c in X_test.columns]
    if len(active_num_test) > 0:
        X_test["num_mean"] = X_test[active_num_test].mean(axis=1)
        X_test["num_std"] = X_test[active_num_test].std(axis=1)
        X_test["num_max"] = X_test[active_num_test].max(axis=1)
        X_test["num_min"] = X_test[active_num_test].min(axis=1)
        X_test["num_range"] = X_test["num_max"] - X_test["num_min"]

# Frequency encoding
for c in cat_cols_updated:
    if c in X_test.columns:
        freq_map = X_train[c].value_counts(normalize=True)
        X_test[f"{c}_freq"] = X_test[c].map(freq_map).fillna(0)

# Target encoding
for c in cat_cols_updated:
    if c in X_test.columns:
        global_mean = y_train.mean()
        agg = pd.DataFrame({"cat": X_train[c], "target": y_train}).groupby("cat")[
            "target"
        ]
        counts = agg.count()
        means = agg.mean()
        smoothing = 10
        smooth_enc = (counts * means + smoothing * global_mean) / (counts + smoothing)
        X_test[f"{c}_target_enc"] = X_test[c].map(smooth_enc).fillna(global_mean)

# Categorical interactions
if len(cat_cols_updated) > 1:
    cat_pairs_test = list(zip(cat_cols_updated[:5], cat_cols_updated[1:6]))
    for c1, c2 in cat_pairs_test:
        if c1 in X_test.columns and c2 in X_test.columns:
            X_test[f"{c1}_x_{c2}"] = (
                X_test[c1].astype(str) + "_" + X_test[c2].astype(str)
            )

# Label encode test categoricals
test_final_cat_cols = [
    c for c in X_test.columns if X_test[c].dtype in ["object", "category"]
]
for c in test_final_cat_cols:
    if c in label_encoders:
        X_test[[c]] = label_encoders[c].transform(X_test[[c]].astype(str))
    else:
        X_test[c] = -1

# Scale test
X_test = X_test[final_num_cols]
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=final_num_cols, index=X_test.index
)
X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Test processed: {X_test_scaled.shape}")


# ============================================================
# 9. MODEL ARCHITECTURE: Residual MLP with LayerNorm
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.net(x)) + x


class ClinicianScorerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_blocks=4, dropout_rate=0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x).squeeze(-1)


# ============================================================
# 10. TRAINING CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
N_FOLDS = 5
MLP_EPOCHS = 400
MLP_PATIENCE = 40
LGB_EARLY_STOP = 100

n_features = X_train_scaled.shape[1]
print(
    f"\nFeatures: {n_features}, Train: {X_train_scaled.shape[0]}, Val: {X_val_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}"
)

# ============================================================
# 11. K-FOLD TRAINING FOR MLP
# ============================================================
print("\n=== Training MLP with K-Fold CV ===")

X_train_np = X_train_scaled.values.astype(np.float32)
X_val_np = X_val_scaled.values.astype(np.float32)
X_test_np = X_test_scaled.values.astype(np.float32)

mlp_oof_train = np.zeros(len(X_train_np))
mlp_val_preds_folds = []
mlp_test_preds_folds = []

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
    X_tr = torch.FloatTensor(X_train_np[train_idx]).to(device)
    y_tr = torch.FloatTensor(y_train[train_idx]).to(device)
    X_fv = torch.FloatTensor(X_train_np[val_idx]).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)

    model = ClinicianScorerMLP(
        input_dim=n_features, hidden_dim=128, n_blocks=4, dropout_rate=0.25
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MLP_EPOCHS, eta_min=1e-5
    )

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MLP_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            fv_pred = model(X_fv).cpu().numpy()
            fv_rmse = root_mean_squared_error(y_train[val_idx], fv_pred)

        if fv_rmse < best_val_rmse:
            best_val_rmse = fv_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        mlp_oof_train[val_idx] = model(X_fv).cpu().numpy()
        mlp_val_fold_pred = model(torch.FloatTensor(X_val_np).to(device)).cpu().numpy()
        mlp_test_fold_pred = (
            model(torch.FloatTensor(X_test_np).to(device)).cpu().numpy()
        )

    mlp_val_preds_folds.append(mlp_val_fold_pred)
    mlp_test_preds_folds.append(mlp_test_fold_pred)

    print(f"  Fold {fold+1}/{N_FOLDS}: best_val_rmse={best_val_rmse:.5f}")

mlp_val_pred = np.mean(mlp_val_preds_folds, axis=0)
mlp_test_pred = np.mean(mlp_test_preds_folds, axis=0)
mlp_val_rmse = root_mean_squared_error(y_val, mlp_val_pred)
print(f"  MLP ensemble val RMSE: {mlp_val_rmse:.5f}")

# ============================================================
# 12. K-FOLD TRAINING FOR LIGHTGBM
# ============================================================
print("\n=== Training LightGBM with K-Fold CV ===")

lgb_oof_train = np.zeros(len(X_train_np))
lgb_val_preds_folds = []
lgb_test_preds_folds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
    X_tr = X_train_scaled.iloc[train_idx]
    y_tr = y_train[train_idx]
    X_fv = X_train_scaled.iloc[val_idx]
    y_fv = y_train[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=23,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42 + fold,
        verbose=-1,
    )

    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_fv, y_fv)],
        callbacks=[
            lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    lgb_oof_train[val_idx] = lgb_model.predict(X_fv)
    lgb_val_fold_pred = lgb_model.predict(X_val_scaled)
    lgb_test_fold_pred = lgb_model.predict(X_test_scaled)

    lgb_val_preds_folds.append(lgb_val_fold_pred)
    lgb_test_preds_folds.append(lgb_test_fold_pred)

    fold_rmse = root_mean_squared_error(y_fv, lgb_oof_train[val_idx])
    print(
        f"  Fold {fold+1}/{N_FOLDS}: fold_rmse={fold_rmse:.5f}, best_iter={lgb_model.best_iteration_}"
    )

lgb_val_pred = np.mean(lgb_val_preds_folds, axis=0)
lgb_test_pred = np.mean(lgb_test_preds_folds, axis=0)
lgb_val_rmse = root_mean_squared_error(y_val, lgb_val_pred)
print(f"  LightGBM ensemble val RMSE: {lgb_val_rmse:.5f}")

# ============================================================
# 13. META-LEARNER: Ridge on OOF predictions
# ============================================================
print("\n=== Fitting Meta-Learner ===")

meta_train = np.column_stack(
    [
        lgb_oof_train,
        mlp_oof_train,
        lgb_oof_train**2,
        mlp_oof_train**2,
        lgb_oof_train * mlp_oof_train,
    ]
)

meta_val = np.column_stack(
    [
        lgb_val_pred,
        mlp_val_pred,
        lgb_val_pred**2,
        mlp_val_pred**2,
        lgb_val_pred * mlp_val_pred,
    ]
)

meta_test = np.column_stack(
    [
        lgb_test_pred,
        mlp_test_pred,
        lgb_test_pred**2,
        mlp_test_pred**2,
        lgb_test_pred * mlp_test_pred,
    ]
)

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y_train)

meta_val_pred = meta_model.predict(meta_val)
meta_val_rmse = root_mean_squared_error(y_val, meta_val_pred)

w_avg = (lgb_val_pred + mlp_val_pred) / 2
w_avg_rmse = root_mean_squared_error(y_val, w_avg)

print(f"  Meta-learner val RMSE: {meta_val_rmse:.5f}")
print(f"  Simple average val RMSE: {w_avg_rmse:.5f}")
print(
    f"  Meta weights: {np.round(meta_model.coef_, 4)}, intercept: {meta_model.intercept_:.4f}"
)

# ============================================================
# 14. SELECT BEST STRATEGY & GENERATE TEST PREDICTIONS
# ============================================================
print("\n=== Selecting Best Strategy ===")

candidates = {
    "lgb_only": (lgb_val_rmse, lgb_test_pred),
    "mlp_only": (mlp_val_rmse, mlp_test_pred),
    "meta_learner": (meta_val_rmse, meta_model.predict(meta_test)),
    "simple_avg": (w_avg_rmse, (lgb_test_pred + mlp_test_pred) / 2),
}

best_name = min(candidates, key=lambda k: candidates[k][0])
best_val_score = candidates[best_name][0]
test_pred = candidates[best_name][1]

print(f"  Best strategy: {best_name} (val RMSE = {best_val_score:.5f})")
for name, (rmse, _) in candidates.items():
    marker = " ✓" if name == best_name else ""
    print(f"    {name}: {rmse:.5f}{marker}")

# ============================================================
# 15. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "Score": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Test prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, min={test_pred.min():.4f}, max={test_pred.max():.4f}"
)

# ============================================================
# 16. SAVE ARTIFACTS
# ============================================================
X_train_scaled.to_parquet("./working/X_train_scaled.parquet")
X_val_scaled.to_parquet("./working/X_val_scaled.parquet")
X_test_scaled.to_parquet("./working/X_test_scaled.parquet")
pd.DataFrame({"y_train": y_train}).to_parquet("./working/y_train.parquet")
pd.DataFrame({"y_val": y_val}).to_parquet("./working/y_val.parquet")

with open("./working/feature_names.json", "w") as f:
    json.dump(final_num_cols, f)

print(f"Processed data saved to ./working/")
print(f"Feature count: {len(final_num_cols)}")

# ============================================================
# FINAL OUTPUT
# ============================================================
score = best_val_score
print(f"Final Validation Score: {score}")
