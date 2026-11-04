import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['class'].value_counts()}")

# ============================================================
# 2. STRATIFIED SPLIT (before any transformation)
# ============================================================
target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df[target_col]
)

# Convert string labels to numeric: 'h' (hadron) -> 0, 'g' (gamma) -> 1
label_map = {'h': 0, 'g': 1}
y_train = train_split[target_col].map(label_map).values
y_val = val_split[target_col].map(label_map).values
test_ids = np.arange(len(test_df))

print(f"Train split: {train_split.shape}, Val split: {val_split.shape}")
print(
    f"Train class balance: {train_split[target_col].value_counts(normalize=True).to_dict()}"
)
print(
    f"Val class balance: {val_split[target_col].value_counts(normalize=True).to_dict()}"
)


# ============================================================
# 3. FEATURE ENGINEERING (physics-motivated)
# ============================================================
def engineer_features(df):
    """
    Physics-motivated feature engineering for MAGIC Gamma Telescope data.
    Gamma events: compact, aligned with source (low fAlpha, low fDist)
    Hadron events: diffuse, random orientation (high fAlpha, high fDist)
    """
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # 1. Elongation
    X["elongation"] = X["fLength"] / (X["fWidth"] + 1e-8)
    X["elongation_log"] = np.log1p(X["elongation"])

    # 2. Angular encoding (fAlpha is circular)
    X["fAlpha_sin"] = np.sin(X["fAlpha"])
    X["fAlpha_cos"] = np.cos(X["fAlpha"])
    X["fAlpha_sin2"] = np.sin(2 * X["fAlpha"])
    X["fAlpha_cos2"] = np.cos(2 * X["fAlpha"])
    X["fAlpha_abs"] = np.abs(X["fAlpha"])
    X["fAlpha_sin_abs"] = np.abs(np.sin(X["fAlpha"]))

    # 3. Distance features
    X["fDist_log"] = np.log1p(X["fDist"])
    X["fDist_sq"] = X["fDist"] ** 2
    X["fDist_cubed"] = X["fDist"] ** 3

    # 4. Alignment x proximity interactions
    X["alpha_dist"] = X["fAlpha"] * X["fDist"]
    X["alpha_dist_abs"] = np.abs(X["fAlpha"]) * X["fDist"]
    X["alpha_sin_dist"] = np.sin(X["fAlpha"]) * X["fDist"]
    X["alpha_cos_dist"] = np.cos(X["fAlpha"]) * X["fDist"]

    # 5. Concentration features
    X["conc_ratio"] = X["fConc"] / (X["fConc1"] + 1e-8)
    X["conc_diff"] = X["fConc"] - X["fConc1"]
    X["conc_product"] = X["fConc"] * X["fConc1"]
    X["fConc_log"] = np.log1p(X["fConc"])
    X["fConc1_log"] = np.log1p(X["fConc1"])

    # 6. Size features
    X["fSize_log"] = np.log1p(X["fSize"])
    X["fSize_sq"] = X["fSize"] ** 2

    # 7. Moment features
    X["m3_ratio"] = X["fM3Long"] / (X["fM3Trans"] + 1e-8)
    X["m3_diff"] = X["fM3Long"] - X["fM3Trans"]
    X["m3_product"] = X["fM3Long"] * X["fM3Trans"]
    X["m3_long_log"] = np.log1p(np.abs(X["fM3Long"]))
    X["m3_trans_log"] = np.log1p(np.abs(X["fM3Trans"]))
    X["m3_long_sign"] = np.sign(X["fM3Long"])
    X["m3_trans_sign"] = np.sign(X["fM3Trans"])

    # 8. Asymmetry features
    X["fAsym_log"] = np.log1p(np.abs(X["fAsym"]))
    X["fAsym_sign"] = np.sign(X["fAsym"])
    X["asym_length_ratio"] = X["fAsym"] / (X["fLength"] + 1e-8)

    # 9. Combined shape-orientation
    X["length_width_dist"] = X["fLength"] * X["fWidth"] * X["fDist"]
    X["size_dist"] = X["fSize"] * X["fDist"]
    X["size_alpha"] = X["fSize"] * X["fAlpha"]
    X["size_alpha_sin"] = X["fSize"] * np.sin(X["fAlpha"])

    # 10. Higher-order interactions
    X["alpha2_dist"] = X["fAlpha"] ** 2 * X["fDist"]
    X["alpha2_dist2"] = X["fAlpha"] ** 2 * X["fDist"] ** 2
    X["elong_alpha"] = X["elongation"] * X["fAlpha"]
    X["elong_dist"] = X["elongation"] * X["fDist"]
    X["conc_alpha"] = X["fConc"] * X["fAlpha"]
    X["conc_dist"] = X["fConc"] * X["fDist"]

    # 11. Width-related
    X["fWidth_log"] = np.log1p(X["fWidth"])
    X["fLength_log"] = np.log1p(X["fLength"])
    X["length_width_product"] = X["fLength"] * X["fWidth"]
    X["length_width_sum"] = X["fLength"] + X["fWidth"]
    X["length_width_diff"] = X["fLength"] - X["fWidth"]

    # 12. Normalized distance
    X["dist_size_ratio"] = X["fDist"] / (X["fSize"] + 1e-8)
    X["dist_length_ratio"] = X["fDist"] / (X["fLength"] + 1e-8)
    X["dist_width_ratio"] = X["fDist"] / (X["fWidth"] + 1e-8)

    # 13. Square root transforms
    X["fDist_sqrt"] = np.sqrt(np.abs(X["fDist"]))
    X["fAlpha_sqrt"] = np.sqrt(np.abs(X["fAlpha"]))

    # 14. Bilinear interactions (top-5 pairs)
    top5 = ["fAlpha", "fDist", "fLength", "fWidth", "fConc"]
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            col_name = f"{top5[i]}_x_{top5[j]}"
            X[col_name] = X[top5[i]] * X[top5[j]]

    # 15. Squared top discriminators
    X["fAlpha_sq"] = X["fAlpha"] ** 2

    # 16. Physics-motivated interaction features (gamma/hadron distinction)
    # (1) Alignment-proximity-size coupling
    X["alpha_dist_length"] = X["fAlpha"] * X["fDist"] * X["fLength"]
    # (2) Normalized elongation ratio
    X["elong_norm"] = (X["fLength"] - X["fWidth"]) / (X["fLength"] + X["fWidth"] + 1e-8)
    # (3) Concentration-alignment coupling
    X["conc1_alpha"] = X["fConc1"] * X["fAlpha"]
    # (4) Moment product capturing shape symmetry
    X["m3_long_trans_product"] = X["fM3Long"] * X["fM3Trans"]
    # (5) Density-like feature
    X["size_density"] = X["fSize"] / (X["fLength"] * X["fWidth"] + 1e-8)

    # Handle NaN with median imputation
    X = X.fillna(X.median())

    return X


X_train = engineer_features(train_split).values
X_val = engineer_features(val_split).values
X_test = engineer_features(test_df).values

print(f"Engineered features: {X_train.shape[1]}")

# ============================================================
# 4. NORMALIZE (fit on train only)
# ============================================================
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_val_scaled = np.clip(X_val_scaled, -10, 10)
X_test_scaled = np.clip(X_test_scaled, -10, 10)

n_features = X_train_scaled.shape[1]
print(f"Number of features after scaling: {n_features}")


# ============================================================
# 5. TRAIN 5-MODEL ENSEMBLE
# ============================================================
# Model 1: XGBoost with current params
xgb_model1 = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

xgb_model1.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False,
)

print(f"XGBoost Model 1 best iteration: {xgb_model1.best_iteration}")

# Model 2: LightGBM with tuned params
lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    num_leaves=63,
    learning_rate=0.03,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

print(f"LightGBM Model 2 best iteration: {lgb_model.best_iteration_}")

# Model 3: XGBoost with different params (deeper trees, different regularization)
xgb_model3 = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=1.0,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

xgb_model3.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False,
)

print(f"XGBoost Model 3 best iteration: {xgb_model3.best_iteration}")

# Model 4: CatBoost with ordered boosting
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    subsample=0.8,
    eval_metric="AUC",
    early_stopping_rounds=100,
    random_seed=42,
    verbose=False,
)

cat_model.fit(
    X_train_scaled,
    y_train,
    eval_set=(X_val_scaled, y_val),
    verbose=False,
)

print(f"CatBoost Model 4 best iteration: {cat_model.get_best_iteration()}")

# Model 5: Small MLP Neural Network
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare torch tensors
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val_scaled).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

mlp_model = MLPClassifier(n_features).to(device)
optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

best_val_auc_mlp = 0.0
best_mlp_state = None
patience_counter = 0
patience = 20

for epoch in range(200):
    mlp_model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = mlp_model(batch_X)
        loss = nn.BCELoss()(preds, batch_y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    mlp_model.eval()
    with torch.no_grad():
        val_preds = mlp_model(X_val_t).cpu().numpy()
        val_auc_mlp = roc_auc_score(y_val, val_preds)

    if val_auc_mlp > best_val_auc_mlp:
        best_val_auc_mlp = val_auc_mlp
        best_mlp_state = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break

mlp_model.load_state_dict(best_mlp_state)
mlp_model.to(device)
print(f"MLP Model 5 best val AUC: {best_val_auc_mlp:.6f} (stopped at epoch {epoch + 1})")

# ============================================================
# 6. VALIDATION (official metric: ROC-AUC) - Ensemble
# ============================================================
val_probs_1 = xgb_model1.predict_proba(X_val_scaled)[:, 1]
val_probs_2 = lgb_model.predict_proba(X_val_scaled)[:, 1]
val_probs_3 = xgb_model3.predict_proba(X_val_scaled)[:, 1]
val_probs_4 = cat_model.predict_proba(X_val_scaled)[:, 1]

mlp_model.eval()
with torch.no_grad():
    val_probs_5 = mlp_model(X_val_t).cpu().numpy()

val_probs = (val_probs_1 + val_probs_2 + val_probs_3 + val_probs_4 + val_probs_5) / 5.0
val_auc = roc_auc_score(y_val, val_probs)
print(f"\nEnsemble Validation ROC-AUC: {val_auc:.6f}")
print(f"  Model 1 (XGB depth=5) AUC: {roc_auc_score(y_val, val_probs_1):.6f}")
print(f"  Model 2 (LGBM) AUC:        {roc_auc_score(y_val, val_probs_2):.6f}")
print(f"  Model 3 (XGB depth=7) AUC: {roc_auc_score(y_val, val_probs_3):.6f}")
print(f"  Model 4 (CatBoost) AUC:    {roc_auc_score(y_val, val_probs_4):.6f}")
print(f"  Model 5 (MLP) AUC:         {roc_auc_score(y_val, val_probs_5):.6f}")

# ============================================================
# 7. TEST PREDICTIONS (identical inference logic)
# ============================================================
test_probs_1 = xgb_model1.predict_proba(X_test_scaled)[:, 1]
test_probs_2 = lgb_model.predict_proba(X_test_scaled)[:, 1]
test_probs_3 = xgb_model3.predict_proba(X_test_scaled)[:, 1]
test_probs_4 = cat_model.predict_proba(X_test_scaled)[:, 1]

X_test_t = torch.FloatTensor(X_test_scaled).to(device)
mlp_model.eval()
with torch.no_grad():
    test_probs_5 = mlp_model(X_test_t).cpu().numpy()

test_probs = (test_probs_1 + test_probs_2 + test_probs_3 + test_probs_4 + test_probs_5) / 5.0

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids.astype(int), "class": test_probs})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 9. FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")
