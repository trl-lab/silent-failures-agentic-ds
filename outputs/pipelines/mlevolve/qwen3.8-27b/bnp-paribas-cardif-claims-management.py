import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import os

# Create output directory
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts(normalize=True)}")

# Separate target
y = train_df["target"].values
X = train_df.drop(columns=["target"])

# Identify feature columns
feature_cols = [c for c in X.columns if c.startswith("v")]
print(f"Number of feature columns: {len(feature_cols)}")

# ============================================================
# 2. INITIAL CLEANING & ANALYSIS
# ============================================================
missing_train = X[feature_cols].isna().sum()
print(f"Features with missing values in train: {(missing_train > 0).sum()}")

categorical_cols = (
    X[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
)
numeric_cols = X[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical features: {len(categorical_cols)}")
print(f"Numeric features: {len(numeric_cols)}")

# ============================================================
# 3. STRATIFIED SPLIT (BEFORE any feature engineering)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(X)), test_size=0.15, random_state=42, stratify=y
)

X_train = X.iloc[train_idx][feature_cols].copy()
X_val = X.iloc[val_idx][feature_cols].copy()
y_train = y[train_idx]
y_val = y[val_idx]

print(f"Train split: {X_train.shape[0]} samples")
print(f"Val split: {X_val.shape[0]} samples")
print(f"Val target mean: {y_val.mean():.4f}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(
    X_train, X_val, X_test, feature_cols, y_train, categorical_cols, numeric_cols
):
    """Apply feature engineering. All statistics computed ONLY on X_train."""

    # --- 4a. Missing Value Indicators ---
    missing_cols = [c for c in feature_cols if X_train[c].isna().sum() > 0]
    print(f"Creating missing indicators for {len(missing_cols)} features")

    for c in missing_cols:
        X_train[f"{c}_missing"] = X_train[c].isna().astype(np.int8)
        X_val[f"{c}_missing"] = X_val[c].isna().astype(np.int8)
        X_test[f"{c}_missing"] = X_test[c].isna().astype(np.int8)

    # Total missing count per row
    X_train["n_missing"] = X_train[missing_cols].isna().sum(axis=1).astype(np.int16)
    X_val["n_missing"] = X_val[missing_cols].isna().sum(axis=1).astype(np.int16)
    X_test["n_missing"] = X_test[missing_cols].isna().sum(axis=1).astype(np.int16)

    # --- 4b. Impute missing values (fit on train only) ---
    numeric_with_missing = [c for c in numeric_cols if X_train[c].isna().sum() > 0]
    for c in numeric_with_missing:
        median_val = X_train[c].median()
        X_train[c] = X_train[c].fillna(median_val)
        X_val[c] = X_val[c].fillna(median_val)
        X_test[c] = X_test[c].fillna(median_val)

    cat_with_missing = [c for c in categorical_cols if X_train[c].isna().sum() > 0]
    for c in cat_with_missing:
        mode_val = X_train[c].mode()[0] if not X_train[c].mode().empty else "UNKNOWN"
        X_train[c] = X_train[c].fillna(mode_val)
        X_val[c] = X_val[c].fillna(mode_val)
        X_test[c] = X_test[c].fillna(mode_val)

    # --- 4c. Encode categorical features ---
    cat_cols_to_encode = categorical_cols.copy()
    if not cat_cols_to_encode:
        for c in numeric_cols:
            nunique = X_train[c].nunique()
            if nunique <= 50:
                cat_cols_to_encode.append(c)

    print(f"Encoding {len(cat_cols_to_encode)} categorical features")

    for c in cat_cols_to_encode:
        freq = X_train[c].value_counts()
        cat_to_num = {cat: i for i, cat in enumerate(freq.index)}

        X_train[c] = X_train[c].map(cat_to_num).fillna(-1).astype(np.int16)
        X_val[c] = X_val[c].map(cat_to_num).fillna(-1).astype(np.int16)
        X_test[c] = X_test[c].map(cat_to_num).fillna(-1).astype(np.int16)

        freq_map = freq / len(X_train)
        # Build a mapping from encoded value -> frequency
        encoded_to_freq = {}
        for k, v in cat_to_num.items():
            encoded_to_freq[v] = freq_map.get(k, 0)

        X_train[f"{c}_freq"] = (
            X_train[c].map(encoded_to_freq).fillna(0).astype(np.float32)
        )
        X_val[f"{c}_freq"] = X_val[c].map(encoded_to_freq).fillna(0).astype(np.float32)
        X_test[f"{c}_freq"] = (
            X_test[c].map(encoded_to_freq).fillna(0).astype(np.float32)
        )

    # --- 4d. Identify top features by univariate correlation with target ---
    numeric_features_after = [
        c for c in X_train.columns if np.issubdtype(X_train[c].dtype, np.number)
    ]

    correlations = {}
    for c in numeric_features_after:
        if X_train[c].std() > 0:
            corr = np.corrcoef(X_train[c].values, y_train)[0, 1]
            correlations[c] = abs(corr)

    top_features = sorted(correlations, key=correlations.get, reverse=True)[:15]
    print(f"Top 15 features by |corr| with target:")
    for i, f in enumerate(top_features):
        print(f"  {i+1}. {f}: {correlations[f]:.4f}")

    # --- 4e. Pairwise interactions for top features ---
    interaction_count = 0
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            fi, fj = top_features[i], top_features[j]
            X_train[f"{fi}_x_{fj}"] = X_train[fi].values * X_train[fj].values
            X_val[f"{fi}_x_{fj}"] = X_val[fi].values * X_val[fj].values
            X_test[f"{fi}_x_{fj}"] = X_test[fi].values * X_test[fj].values
            interaction_count += 1

    print(f"Created {interaction_count} interaction features")

    # --- 4f. Ratio features for top features ---
    for i in range(len(top_features)):
        for j in range(i + 1, min(i + 3, len(top_features))):
            fi, fj = top_features[i], top_features[j]
            denom_train = X_train[fj].values.copy()
            denom_train[np.abs(denom_train) < 1e-8] = 1e-8
            X_train[f"{fi}_div_{fj}"] = X_train[fi].values / denom_train

            denom_val = X_val[fj].values.copy()
            denom_val[np.abs(denom_val) < 1e-8] = 1e-8
            X_val[f"{fi}_div_{fj}"] = X_val[fi].values / denom_val

            denom_test = X_test[fj].values.copy()
            denom_test[np.abs(denom_test) < 1e-8] = 1e-8
            X_test[f"{fi}_div_{fj}"] = X_test[fi].values / denom_test

    # --- 4g. Row-level aggregate statistics ---
    all_numeric = [
        c for c in X_train.columns if np.issubdtype(X_train[c].dtype, np.number)
    ]

    X_train["row_mean"] = X_train[all_numeric].mean(axis=1).astype(np.float32)
    X_val["row_mean"] = X_val[all_numeric].mean(axis=1).astype(np.float32)
    X_test["row_mean"] = X_test[all_numeric].mean(axis=1).astype(np.float32)

    X_train["row_std"] = X_train[all_numeric].std(axis=1).astype(np.float32)
    X_val["row_std"] = X_val[all_numeric].std(axis=1).astype(np.float32)
    X_test["row_std"] = X_test[all_numeric].std(axis=1).astype(np.float32)

    X_train["row_max"] = X_train[all_numeric].max(axis=1).astype(np.float32)
    X_val["row_max"] = X_val[all_numeric].max(axis=1).astype(np.float32)
    X_test["row_max"] = X_test[all_numeric].max(axis=1).astype(np.float32)

    X_train["row_min"] = X_train[all_numeric].min(axis=1).astype(np.float32)
    X_val["row_min"] = X_val[all_numeric].min(axis=1).astype(np.float32)
    X_test["row_min"] = X_test[all_numeric].min(axis=1).astype(np.float32)

    X_train["row_range"] = (X_train["row_max"] - X_train["row_min"]).astype(np.float32)
    X_val["row_range"] = (X_val["row_max"] - X_val["row_min"]).astype(np.float32)
    X_test["row_range"] = (X_test["row_max"] - X_test["row_min"]).astype(np.float32)

    # --- 4h. Binned features for top numeric features ---
    for c in top_features[:10]:
        if np.issubdtype(X_train[c].dtype, np.number):
            quantiles = np.quantile(X_train[c].values, [0.2, 0.4, 0.6, 0.8])
            bins = [-np.inf] + quantiles.tolist() + [np.inf]
            X_train[f"{c}_bin"] = np.digitize(X_train[c].values, bins).astype(np.int8)
            X_val[f"{c}_bin"] = np.digitize(X_val[c].values, bins).astype(np.int8)
            X_test[f"{c}_bin"] = np.digitize(X_test[c].values, bins).astype(np.int8)

    # --- 4i. Polynomial features for top 5 features ---
    for c in top_features[:5]:
        X_train[f"{c}_sq"] = (X_train[c].values ** 2).astype(np.float32)
        X_val[f"{c}_sq"] = (X_val[c].values ** 2).astype(np.float32)
        X_test[f"{c}_sq"] = (X_test[c].values ** 2).astype(np.float32)

    return X_train, X_val, X_test


# Apply feature engineering
X_test_raw = test_df[feature_cols].copy()
X_train, X_val, X_test = engineer_features(
    X_train, X_val, X_test_raw, feature_cols, y_train, categorical_cols, numeric_cols
)

# ============================================================
# 5. FINAL VERIFICATION & CLEANUP
# ============================================================
print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Handle any remaining NaNs
nan_train = X_train.isna().sum().sum()
if nan_train > 0:
    cols_with_nan = X_train.columns[X_train.isna().sum() > 0]
    for c in cols_with_nan:
        fill_val = (
            X_train[c].median() if np.issubdtype(X_train[c].dtype, np.number) else 0
        )
        X_train[c] = X_train[c].fillna(fill_val)
        X_val[c] = X_val[c].fillna(fill_val)
        X_test[c] = X_test[c].fillna(fill_val)
    print("Filled remaining NaNs")

# Verify column consistency
assert set(X_train.columns) == set(
    X_val.columns
), "Column mismatch between train and val!"
assert set(X_train.columns) == set(
    X_test.columns
), "Column mismatch between train and test!"

# Convert to numpy
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# Safety: handle any remaining NaNs/Infs
X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

n_features = X_train_np.shape[1]
print(f"Number of features: {n_features}")

# ============================================================
# 6. TRAIN LIGHTGBM MODEL
# ============================================================
lgb_train_data = lgb.Dataset(X_train_np, label=y_train)
lgb_val_data = lgb.Dataset(X_val_np, label=y_val, reference=lgb_train_data)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    num_boost_round=5000,
    valid_sets=[lgb_val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=1000),
    ],
)

lgb_val_pred = lgb_model.predict(X_val_np, num_iteration=lgb_model.best_iteration)
lgb_test_pred = lgb_model.predict(X_test_np, num_iteration=lgb_model.best_iteration)

lgb_val_score = log_loss(y_val, np.clip(lgb_val_pred, 1e-7, 1 - 1e-7))
print(
    f"LightGBM val log_loss: {lgb_val_score:.6f} (best_iter={lgb_model.best_iteration})"
)


# ============================================================
# 7. TRAIN NEURAL NETWORK (Simple robust MLP)
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_tensor = torch.FloatTensor(X_train_np).to(device)
val_tensor = torch.FloatTensor(X_val_np).to(device)
test_tensor = torch.FloatTensor(X_test_np).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)

train_dataset = TensorDataset(train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=2)

nn_model = SimpleMLP(n_features).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(nn_model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

best_val_loss = float("inf")
best_model_state = None
patience = 10
patience_counter = 0

for epoch in range(60):
    nn_model.train()
    train_loss = 0.0
    n_batches = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = nn_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    scheduler.step()

    # Validation
    nn_model.eval()
    with torch.no_grad():
        val_pred = nn_model(val_tensor).cpu().numpy()
    val_logloss = log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7))

    print(
        f"Epoch {epoch+1:3d} | Train: {train_loss/n_batches:.6f} | Val: {val_logloss:.6f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_model_state = {
            k: v.cpu().clone() for k, v in nn_model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model and get predictions
nn_model.load_state_dict(best_model_state)
nn_model.to(device)
nn_model.eval()

with torch.no_grad():
    nn_val_pred = nn_model(val_tensor).cpu().numpy()
    nn_test_pred = nn_model(test_tensor).cpu().numpy()

nn_val_score = log_loss(y_val, np.clip(nn_val_pred, 1e-7, 1 - 1e-7))
print(f"NN val log_loss: {nn_val_score:.6f}")

# ============================================================
# 8. ENSEMBLE - Find optimal weight on validation
# ============================================================
best_weight = 0.5
best_ensemble_score = float("inf")

for w in np.arange(0.0, 1.01, 0.02):
    ensemble_pred = w * lgb_val_pred + (1 - w) * nn_val_pred
    score = log_loss(y_val, np.clip(ensemble_pred, 1e-7, 1 - 1e-7))
    if score < best_ensemble_score:
        best_ensemble_score = score
        best_weight = w

print(f"Best ensemble weight (LGB): {best_weight:.2f}")
print(f"Best ensemble val log_loss: {best_ensemble_score:.6f}")

# ============================================================
# 9. GENERATE FINAL PREDICTIONS & SUBMISSION
# ============================================================
final_test_pred = best_weight * lgb_test_pred + (1 - best_weight) * nn_test_pred
final_test_pred = np.clip(final_test_pred, 1e-7, 1 - 1e-7)

# Final validation score (official metric: log_loss)
final_val_pred = best_weight * lgb_val_pred + (1 - best_weight) * nn_val_pred
final_val_pred = np.clip(final_val_pred, 1e-7, 1 - 1e-7)
score = log_loss(y_val, final_val_pred)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": range(len(final_test_pred)), "target": final_test_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")
print(f"Prediction range: [{final_test_pred.min():.6f}, {final_test_pred.max():.6f}]")
print(f"Prediction mean: {final_test_pred.mean():.6f}")

print(f"Final Validation Score: {score}")
