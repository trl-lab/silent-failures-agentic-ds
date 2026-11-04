import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -------------------------------------------------------------------------
# Step 1: Data Processing & Feature Engineering
# -------------------------------------------------------------------------

# 1. Load raw data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Ensure 'id' column exists in test set
if "id" not in test_df.columns:
    test_ids = np.arange(len(test_df))
else:
    test_ids = test_df["id"].values

# Feature and target definition
X_raw = train_df.drop(columns=["quality"])
y = train_df["quality"]


# PyTorch Multi-Layer Perceptron Definition
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Domain feature engineering function
def engineer_domain_features(df):
    df = df.copy()

    # Binary color encoding
    if "color" in df.columns:
        df["is_white"] = (df["color"].astype(str).str.lower() == "white").astype(int)
        df["is_red"] = (df["color"].astype(str).str.lower() == "red").astype(int)

    # Active molecular SO2
    df["active_molecular_so2"] = df["free_sulfur_dioxide"] / (
        1.0 + 10.0 ** (df["pH"] - 1.81)
    )

    # Sugar-free extract estimate
    df["sugar_free_extract"] = (df["density"] - 1.0) * 1000.0 - df["residual_sugar"]

    # Density residual based on alcohol-sugar-density equilibrium
    df["density_residual"] = df["density"] - (
        0.9918 + 0.0038 * df["residual_sugar"] - 0.0011 * df["alcohol"]
    )

    # Color-interacted chemical metrics
    df["volatile_acidity_pH"] = df["volatile_acidity"] * df["pH"]
    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
    if "is_white" in df.columns:
        df["volatile_acidity_pH_white"] = df["volatile_acidity_pH"] * df["is_white"]
        df["alcohol_sulphates_white"] = df["alcohol_sulphates"] * df["is_white"]

    # Sulfur dioxide dynamics
    df["bound_sulfur_dioxide"] = df["total_sulfur_dioxide"] - df["free_sulfur_dioxide"]
    df["free_to_total_so2_ratio"] = df["free_sulfur_dioxide"] / (
        df["total_sulfur_dioxide"] + 1e-5
    )
    df["bound_to_total_so2_ratio"] = df["bound_sulfur_dioxide"] / (
        df["total_sulfur_dioxide"] + 1e-5
    )

    # Acid balance
    df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"]
    df["volatile_to_fixed_acidity_ratio"] = df["volatile_acidity"] / (
        df["fixed_acidity"] + 1e-5
    )
    df["citric_to_fixed_acidity_ratio"] = df["citric_acid"] / (
        df["fixed_acidity"] + 1e-5
    )
    df["citric_to_total_acidity_ratio"] = df["citric_acid"] / (
        df["total_acidity"] + 1e-5
    )
    df["pH_total_acidity_product"] = df["pH"] * df["total_acidity"]

    # Sugar, Density, Alcohol interaction
    df["sugar_to_alcohol_ratio"] = df["residual_sugar"] / (df["alcohol"] + 1e-5)
    df["alcohol_density_product"] = df["alcohol"] * df["density"]
    df["sugar_density_ratio"] = df["residual_sugar"] / (df["density"] + 1e-5)

    # Preservative / Salt relationships
    df["chlorides_to_sulphates_ratio"] = df["chlorides"] / (df["sulphates"] + 1e-5)
    df["sulphates_to_acidity_ratio"] = df["sulphates"] / (df["volatile_acidity"] + 1e-5)

    # Log transforms for skewed continuous features
    skewed_cols = [
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "sulphates",
    ]
    for col in skewed_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(np.maximum(0, df[col]))

    return df


# Create domain features for train and test
train_fe = engineer_domain_features(X_raw)
test_fe = engineer_domain_features(test_df)

num_cols = [
    "alcohol",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "fixed_acidity",
]

num_cols = [c for c in num_cols if c in X_raw.columns]

if "color" in X_raw.columns:
    color_stats = X_raw.groupby("color")[num_cols].agg(["mean", "std"]).to_dict()
    for col in num_cols:
        for split_df in [train_fe, test_fe]:
            if "color" in split_df.columns:
                mean_s = split_df["color"].map(color_stats[(col, "mean")])
                std_s = split_df["color"].map(color_stats[(col, "std")]).replace(0, 1.0)
                split_df[f"{col}_z_by_color"] = (split_df[col] - mean_s) / (std_s + 1e-8)
else:
    global_means = X_raw[num_cols].mean().to_dict()
    global_stds = X_raw[num_cols].std().replace(0, 1.0).to_dict()
    for col in num_cols:
        for split_df in [train_fe, test_fe]:
            split_df[f"{col}_z_by_color"] = (split_df[col] - global_means[col]) / (global_stds[col] + 1e-8)

# Drop non-numeric categorical column
for split_df in [train_fe, test_fe]:
    split_df.drop(columns=["color"], errors="ignore", inplace=True)

# Assemble processed dataframes and compute imputations strictly on training set
train_processed = train_fe.copy()
train_processed["quality"] = y.values

test_processed = test_fe.copy()

# Handle potential missing values and infs using train medians
train_processed = train_processed.replace([np.inf, -np.inf], np.nan)
test_processed = test_processed.replace([np.inf, -np.inf], np.nan)

train_medians = train_processed.drop(
    columns=["quality", "id"], errors="ignore"
).median()
train_processed = train_processed.fillna(train_medians)
test_processed = test_processed.fillna(train_medians)

# Save processed datasets for auditing / persistence
train_processed.to_parquet("./working/train_processed.parquet", index=False)
test_processed.to_parquet("./working/test_processed.parquet", index=False)

# Prepare feature matrices
target_col = "quality"
drop_cols = [target_col, "id"]
feature_cols = [c for c in train_processed.columns if c not in drop_cols]

X = train_processed[feature_cols].values
y = train_processed[target_col].values
X_test = test_processed[feature_cols].values


# Helper metric function
def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


# Helper training function for PyTorch MLP
def train_pytorch_mlp(X_tr, y_tr, X_val, y_val, X_te, fold, epochs=80, batch_size=64, lr=1e-3):
    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    X_te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)

    model = PyTorchMLP(X_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_val_preds = None
    best_test_preds = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_tensor)
            val_loss = criterion(
                val_out, torch.tensor(y_val, dtype=torch.float32).to(device)
            ).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_preds = val_out.cpu().numpy()
                best_test_preds = model(X_te_tensor).cpu().numpy()

    return best_val_preds, best_test_preds


# -------------------------------------------------------------------------
# Step 2: 5-Fold Stratified K-Fold Training & Model Design
# -------------------------------------------------------------------------

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
stratify_target = np.round(y).astype(int)

n_models = 8
oof_preds = np.zeros((len(X), n_models))
test_preds = np.zeros((len(X_test), n_models))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_target)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_v, y_v = X[val_idx], y[val_idx]

    # --- Model 0: LightGBM Regressor ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds[val_idx, 0] = lgb_model.predict(X_v)
    test_preds[:, 0] += lgb_model.predict(X_test) / n_splits

    # --- Model 1: XGBoost Regressor ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    oof_preds[val_idx, 1] = xgb_model.predict(X_v)
    test_preds[:, 1] += xgb_model.predict(X_test) / n_splits

    # --- Model 2: CatBoost Regressor ---
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_v, y_v))
    oof_preds[val_idx, 2] = cb_model.predict(X_v)
    test_preds[:, 2] += cb_model.predict(X_test) / n_splits

    # --- Model 3: ExtraTrees Regressor ---
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_preds[val_idx, 3] = et_model.predict(X_v)
    test_preds[:, 3] += et_model.predict(X_test) / n_splits

    # --- Model 4: PyTorch MLP ---
    mlp_val, mlp_te = train_pytorch_mlp(
        X_tr, y_tr, X_v, y_v, X_test, fold=fold, epochs=80, batch_size=64, lr=1e-3
    )
    oof_preds[val_idx, 4] = mlp_val
    test_preds[:, 4] += mlp_te / n_splits

    # --- Model 5: LightGBM Classifier (Expected Value) ---
    lgb_cls = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        objective="multiclass",
        verbose=-1,
    )
    lgb_cls.fit(
        X_tr,
        y_tr.astype(int),
        eval_set=[(X_v, y_v.astype(int))],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_cls_classes = lgb_cls.classes_.astype(float)
    oof_preds[val_idx, 5] = lgb_cls.predict_proba(X_v) @ lgb_cls_classes
    test_preds[:, 5] += (lgb_cls.predict_proba(X_test) @ lgb_cls_classes) / n_splits

    # --- Model 6: XGBoost Classifier (Expected Value) ---
    unique_classes_tr, y_tr_enc = np.unique(y_tr.astype(int), return_inverse=True)
    y_v_enc = np.array(
        [
            np.where(unique_classes_tr == val)[0][0]
            if val in unique_classes_tr
            else 0
            for val in y_v.astype(int)
        ]
    )
    xgb_cls = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    xgb_cls.fit(X_tr, y_tr_enc, eval_set=[(X_v, y_v_enc)], verbose=False)
    oof_preds[val_idx, 6] = xgb_cls.predict_proba(X_v) @ unique_classes_tr.astype(float)
    test_preds[:, 6] += (xgb_cls.predict_proba(X_test) @ unique_classes_tr.astype(float)) / n_splits

    # --- Model 7: CatBoost Classifier (Expected Value) ---
    cb_cls = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        loss_function="MultiClass",
        verbose=0,
    )
    cb_cls.fit(X_tr, y_tr.astype(int), eval_set=(X_v, y_v.astype(int)))
    cb_classes = cb_cls.classes_.astype(float)
    oof_preds[val_idx, 7] = cb_cls.predict_proba(X_v) @ cb_classes
    test_preds[:, 7] += (cb_cls.predict_proba(X_test) @ cb_classes) / n_splits

# -------------------------------------------------------------------------
# Step 3: Optimization & Submission Generation
# -------------------------------------------------------------------------


# Constrained weight optimization for optimal OOF RMSE blending
def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = oof_preds @ w
    return compute_rmse(y, blend_oof)


init_weights = [1.0 / n_models] * n_models
bounds = [(0, 1)] * n_models
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = res.x / np.sum(res.x)

# Generate final ensemble predictions
final_oof_pred = oof_preds @ best_weights
final_test_pred = test_preds @ best_weights

# Physical bounds clipping (quality scores strictly between 0 and 10)
final_oof_pred = np.clip(final_oof_pred, 0.0, 10.0)
final_test_pred = np.clip(final_test_pred, 0.0, 10.0)

# Calculate final validation metric on complete OOF
final_val_score = compute_rmse(y, final_oof_pred)

# Save predictions to required submission format
submission_df = pd.DataFrame({"id": test_ids, "quality": final_test_pred})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")