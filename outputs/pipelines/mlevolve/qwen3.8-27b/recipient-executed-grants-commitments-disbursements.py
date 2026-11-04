import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "Disbursements USD (Trust funds)"
y_train_full = train[target_col].copy()
train = train.drop(columns=[target_col])

# Identify column types
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
datetime_cols = [
    c
    for c in train.columns
    if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()
]

id_cols = [c for c in train.columns if c.lower() in ["id", "index", "row_id"]]
numeric_cols = [c for c in numeric_cols if c not in id_cols]
categorical_cols = [c for c in categorical_cols if c not in id_cols]

print(
    f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}, Date-like: {datetime_cols}"
)

# ============================================================
# 2. TRAIN/VALIDATION SPLIT
# ============================================================
train_indices, val_indices = train_test_split(
    np.arange(len(train)), test_size=0.15, random_state=42
)

train_split = train.iloc[train_indices].copy()
val_split = train.iloc[val_indices].copy()
y_train_split = y_train_full.iloc[train_indices].copy()
y_val_split = y_train_full.iloc[val_indices].copy()

print(f"Split: train={len(train_split)}, val={len(val_split)}, test={len(test)}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(
    train_df, val_df, test_df, y_train, numeric_cols, categorical_cols, datetime_cols
):
    """Engineer features. All stateful transforms fitted ONLY on train_df."""

    # --- Handle missing values ---
    for col in numeric_cols:
        if col in train_df.columns:
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            if col in val_df.columns:
                val_df[col] = val_df[col].fillna(med)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(med)

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("MISSING").astype(str)
        if col in val_df.columns:
            val_df[col] = val_df[col].fillna("MISSING").astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("MISSING").astype(str)

    # --- Temporal features ---
    for col in datetime_cols:
        if col in train_df.columns:
            try:
                parsed = pd.to_datetime(train_df[col], errors="coerce")
                if parsed.notna().sum() > len(train_df) * 0.5:
                    for df in [train_df, val_df, test_df]:
                        if col in df.columns:
                            p = pd.to_datetime(df[col], errors="coerce")
                            df[f"{col}_year"] = p.dt.year
                            df[f"{col}_month"] = p.dt.month
                            df[f"{col}_quarter"] = p.dt.quarter
                            df[f"{col}_dayofyear"] = p.dt.dayofyear
                            df[f"{col}_is_q1"] = (p.dt.quarter == 1).astype(int)
                            df[f"{col}_is_q4"] = (p.dt.quarter == 4).astype(int)
            except:
                pass

    # Cyclical year features
    for col in numeric_cols:
        if "year" in col.lower() and col in train_df.columns:
            for df in [train_df, val_df, test_df]:
                if col in df.columns:
                    df[f"{col}_cyc_sin"] = np.sin(2 * np.pi * df[col] / 40)
                    df[f"{col}_cyc_cos"] = np.cos(2 * np.pi * df[col] / 40)

    # --- Frequency encoding ---
    for col in categorical_cols:
        if col in train_df.columns:
            freq_map = train_df[col].value_counts(normalize=True)
            train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
            val_df[f"{col}_freq"] = val_df[col].map(freq_map).fillna(0)
            test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

    # --- Categorical interactions ---
    cat_cardinalities = [
        (col, train_df[col].nunique())
        for col in categorical_cols
        if col in train_df.columns
    ]
    cat_cardinalities.sort(key=lambda x: x[1], reverse=True)
    top_cats = [c for c, _ in cat_cardinalities[:6]]

    interaction_cols = []
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            col_name = f"{top_cats[i]}__{top_cats[j]}"
            train_df[col_name] = (
                train_df[top_cats[i]].astype(str)
                + "_"
                + train_df[top_cats[j]].astype(str)
            )
            val_df[col_name] = (
                val_df[top_cats[i]].astype(str) + "_" + val_df[top_cats[j]].astype(str)
            )
            test_df[col_name] = (
                test_df[top_cats[i]].astype(str)
                + "_"
                + test_df[top_cats[j]].astype(str)
            )
            interaction_cols.append(col_name)

    for col in interaction_cols:
        freq_map = train_df[col].value_counts(normalize=True)
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
        val_df[f"{col}_freq"] = val_df[col].map(freq_map).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

    # --- Log transform skewed numeric columns ---
    for col in numeric_cols:
        if col in train_df.columns and train_df[col].min() >= 0:
            if train_df[col].skew() > 1:
                train_df[f"{col}_log1p"] = np.log1p(train_df[col].clip(lower=0))
                val_df[f"{col}_log1p"] = np.log1p(val_df[col].clip(lower=0))
                test_df[f"{col}_log1p"] = np.log1p(test_df[col].clip(lower=0))

    # --- Ratio features ---
    num_cols_in_data = [c for c in numeric_cols if c in train_df.columns]
    for i in range(min(len(num_cols_in_data), 8)):
        for j in range(i + 1, min(len(num_cols_in_data), 8)):
            col_a, col_b = num_cols_in_data[i], num_cols_in_data[j]
            ratio_name = f"{col_a}_div_{col_b}"
            for df in [train_df, val_df, test_df]:
                if col_a in df.columns and col_b in df.columns:
                    df[ratio_name] = df[col_a] / (df[col_b].abs() + 1e-8)

    # --- K-Fold Target encoding with smoothing (leakage-free) ---
    n_te_folds = 5
    te_kf = KFold(n_splits=n_te_folds, shuffle=True, random_state=42)
    global_mean = y_train.mean()
    smoothing = 10

    for col in categorical_cols:
        if col in train_df.columns:
            # For train: use OOF target encoding
            train_te = np.full(len(train_df), global_mean)
            for te_fold in range(n_te_folds):
                te_tr_idx, te_va_idx = list(te_kf.split(train_df))[te_fold]
                agg = y_train.iloc[te_tr_idx].groupby(train_df[col].iloc[te_tr_idx]).agg(
                    ["mean", "count"]
                )
                te_enc = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
                    agg["count"] + smoothing
                )
                train_te[te_va_idx] = (
                    train_df[col].iloc[te_va_idx].map(te_enc).fillna(global_mean).values
                )
            train_df[f"{col}_target_enc"] = train_te

            # For val and test: use full train stats
            agg_full = y_train.groupby(train_df[col]).agg(["mean", "count"])
            target_enc_full = (
                agg_full["mean"] * agg_full["count"] + global_mean * smoothing
            ) / (agg_full["count"] + smoothing)
            val_df[f"{col}_target_enc"] = (
                val_df[col].map(target_enc_full).fillna(global_mean)
            )
            test_df[f"{col}_target_enc"] = (
                test_df[col].map(target_enc_full).fillna(global_mean)
            )

    # --- Polynomial interaction features (numeric x top categorical freq) ---
    num_cols_for_poly = [c for c in numeric_cols if c in train_df.columns][:2]
    top_cat_freq_cols = [f"{c}_freq" for c in top_cats[:4]]

    for nc in num_cols_for_poly:
        for cf in top_cat_freq_cols:
            if nc in train_df.columns and cf in train_df.columns:
                poly_name = f"{nc}_x_{cf}"
                train_df[poly_name] = train_df[nc] * train_df[cf]
                val_df[poly_name] = val_df[nc] * val_df[cf]
                test_df[poly_name] = test_df[nc] * test_df[cf]

    # --- Fiscal Year difference feature ---
    fy_col = "Fiscal Year"
    if fy_col in train_df.columns and "Grant Agreement Date" in train_df.columns:
        try:
            ga_year = pd.to_datetime(
                train_df["Grant Agreement Date"], errors="coerce"
            ).dt.year
            train_df["fy_diff"] = train_df[fy_col] - ga_year
            ga_year_val = pd.to_datetime(
                val_df["Grant Agreement Date"], errors="coerce"
            ).dt.year
            val_df["fy_diff"] = val_df[fy_col] - ga_year_val
            ga_year_test = pd.to_datetime(
                test_df["Grant Agreement Date"], errors="coerce"
            ).dt.year
            test_df["fy_diff"] = test_df[fy_col] - ga_year_test
        except:
            pass

    # --- Aggregate features (group-level stats, OOF for train to avoid leakage) ---
    grp_kf = KFold(n_splits=5, shuffle=True, random_state=99)
    for col in top_cats[:4]:
        if col in train_df.columns:
            # OOF group stats for train
            train_grp_mean = np.full(len(train_df), y_train.mean())
            train_grp_std = np.zeros(len(train_df))
            train_grp_count = np.zeros(len(train_df))
            for tr_idx, va_idx in grp_kf.split(train_df):
                agg = y_train.iloc[tr_idx].groupby(train_df[col].iloc[tr_idx]).agg(["mean", "std", "count"])
                train_grp_mean[va_idx] = (
                    train_df[col].iloc[va_idx].map(agg["mean"]).fillna(y_train.mean()).values
                )
                train_grp_std[va_idx] = (
                    train_df[col].iloc[va_idx].map(agg["std"]).fillna(0).values
                )
                train_grp_count[va_idx] = (
                    train_df[col].iloc[va_idx].map(agg["count"]).fillna(0).values
                )
            train_df[f"{col}_grp_mean"] = train_grp_mean
            train_df[f"{col}_grp_std"] = train_grp_std
            train_df[f"{col}_grp_count"] = train_grp_count

            # Full train stats for val and test
            agg_full = y_train.groupby(train_df[col]).agg(["mean", "std", "count"])
            val_df[f"{col}_grp_mean"] = val_df[col].map(agg_full["mean"]).fillna(y_train.mean())
            val_df[f"{col}_grp_std"] = val_df[col].map(agg_full["std"]).fillna(0)
            val_df[f"{col}_grp_count"] = val_df[col].map(agg_full["count"]).fillna(0)
            test_df[f"{col}_grp_mean"] = test_df[col].map(agg_full["mean"]).fillna(y_train.mean())
            test_df[f"{col}_grp_std"] = test_df[col].map(agg_full["std"]).fillna(0)
            test_df[f"{col}_grp_count"] = test_df[col].map(agg_full["count"]).fillna(0)

    return train_df, val_df, test_df


train_processed, val_processed, test_processed = engineer_features(
    train_split,
    val_split,
    test,
    y_train_split,
    numeric_cols,
    categorical_cols,
    datetime_cols,
)

print(
    f"After feature engineering: train={train_processed.shape}, val={val_processed.shape}, test={test_processed.shape}"
)

# ============================================================
# 4. PREPARE FEATURE MATRICES
# ============================================================
feature_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()

for col in feature_cols:
    if col not in val_processed.columns:
        val_processed[col] = 0
    if col not in test_processed.columns:
        test_processed[col] = 0

X_train = train_processed[feature_cols].values.astype(np.float32)
X_val = val_processed[feature_cols].values.astype(np.float32)
X_test = test_processed[feature_cols].values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.clip(X_train_scaled, -10, 10).astype(np.float32)
X_val_scaled = np.clip(X_val_scaled, -10, 10).astype(np.float32)
X_test_scaled = np.clip(X_test_scaled, -10, 10).astype(np.float32)

# Log1p target
y_train = y_train_split.values
y_val = y_val_split.values
y_train_log = np.log1p(np.clip(y_train, 0, None))
y_val_log = np.log1p(np.clip(y_val, 0, None))

print(f"Features: {len(feature_cols)}")
print(f"Target - train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"Target - val:   mean={y_val.mean():.2f}, std={y_val.std():.2f}")

# ============================================================
# 5. TRAIN LIGHTGBM (5-fold CV)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM (5-fold CV)")
print("=" * 60)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 15,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 5000,
    "random_state": 42,
}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

lgb_val_preds = np.zeros(len(X_val))
lgb_test_preds = np.zeros(len(X_test))

splits = list(kf.split(X_train))
for fold in range(n_folds):
    tr_idx, va_idx = splits[fold]

    # Use internal holdout for early stopping (prevents val set leakage into model selection)
    inner_tr, inner_va = train_test_split(tr_idx, test_size=0.15, random_state=42 + fold)

    dtrain = lgb.Dataset(X_train[inner_tr], label=y_train_log[inner_tr])
    dval_inner = lgb.Dataset(X_train[inner_va], label=y_train_log[inner_va], reference=dtrain)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dval_inner],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
    )

    # Retrain on full fold-train with best_iteration for final predictions
    dtrain_full = lgb.Dataset(X_train[tr_idx], label=y_train_log[tr_idx])
    model_lgb_final = lgb.train(
        lgb_params,
        dtrain_full,
        num_boost_round=model_lgb.best_iteration,
        callbacks=[lgb.log_evaluation(0)],
    )

    val_pred_log = model_lgb_final.predict(X_val)
    test_pred_log = model_lgb_final.predict(X_test)

    lgb_val_preds += np.expm1(val_pred_log) / n_folds
    lgb_test_preds += np.expm1(test_pred_log) / n_folds

    print(f"  Fold {fold+1}/{n_folds}: best_iter={model_lgb.best_iteration}")

lgb_val_rmse = root_mean_squared_error(y_val, lgb_val_preds)
print(f"LightGBM CV RMSE (original scale): {lgb_val_rmse:.4f}")

# ============================================================
# 6. TRAIN NEURAL MLP
# ============================================================
print("\n" + "=" * 60)
print("TRAINING NEURAL MLP")
print("=" * 60)


class DisbursementMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super(DisbursementMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
            )
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.projections.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            else:
                self.projections.append(nn.Identity())

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for i, layer in enumerate(self.layers):
            residual = self.projections[i](x)
            x = layer(x) + residual
        return self.output_layer(x).squeeze(-1)


def train_mlp(
    X_tr, y_tr, X_v, y_v, X_te,
    epochs=600, batch_size=256, lr=5e-4, weight_decay=1e-5,
    hidden_dims=[256, 128, 64], seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_tr.shape[1]
    model = DisbursementMLP(input_dim, hidden_dims=hidden_dims, dropout_rate=0.2)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_tensor = torch.FloatTensor(X_v)
    val_target = torch.FloatTensor(y_v)

    best_val_loss = float("inf")
    best_model_state = None
    patience = 80
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor)
            val_loss = criterion(val_pred, val_target).item()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1}/{epochs}: train_loss={epoch_loss/n_batches:.6f}, val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        val_pred_log = model(val_tensor).numpy()
        test_pred_log = model(torch.FloatTensor(X_te)).numpy()

    return model, val_pred_log, test_pred_log, best_val_loss


# Multi-seed, multi-architecture MLP ensemble
mlp_seeds = [42, 123, 456, 789, 1024]
mlp_architectures = [
    [256, 128, 64],
    [512, 256, 128],
]

mlp_val_preds_all = []
mlp_test_preds_all = []

for arch_idx, hidden_dims in enumerate(mlp_architectures):
    for seed in mlp_seeds:
        print(f"\n  Training MLP: seed={seed}, hidden_dims={hidden_dims}")
        _, val_pred_log, test_pred_log, best_loss = train_mlp(
            X_train_scaled,
            y_train_log,
            X_val_scaled,
            y_val_log,
            X_test_scaled,
            epochs=600,
            batch_size=256,
            lr=5e-4,
            weight_decay=1e-5,
            hidden_dims=hidden_dims,
            seed=seed,
        )
        mlp_val_preds_all.append(np.expm1(val_pred_log))
        mlp_test_preds_all.append(np.expm1(test_pred_log))
        print(f"    best_loss={best_loss:.6f}, val_rmse={root_mean_squared_error(y_val, np.expm1(val_pred_log)):.4f}")

# Average all MLP predictions
mlp_val_preds = np.mean(mlp_val_preds_all, axis=0)
mlp_test_preds = np.mean(mlp_test_preds_all, axis=0)
mlp_val_rmse = root_mean_squared_error(y_val, mlp_val_preds)
print(f"\nMLP Ensemble ({len(mlp_val_preds_all)} models) RMSE (original scale): {mlp_val_rmse:.4f}")

# ============================================================
# 7. ENSEMBLE OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE OPTIMIZATION")
print("=" * 60)

best_w = 0.5
best_rmse = float("inf")

for w in np.arange(0.0, 1.001, 0.01):
    blend_val = w * lgb_val_preds + (1 - w) * mlp_val_preds
    rmse = root_mean_squared_error(y_val, blend_val)
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

print(f"Optimal LGB weight: {best_w:.2f}")
print(f"Ensemble RMSE (pre-post-processing): {best_rmse:.4f}")

ensemble_val_preds = best_w * lgb_val_preds + (1 - best_w) * mlp_val_preds
ensemble_test_preds = best_w * lgb_test_preds + (1 - best_w) * mlp_test_preds

# ============================================================
# 8. POST-PROCESSING: Clip to observed range
# ============================================================
target_min = 0.0
target_max = np.percentile(y_train, 99.5)

ensemble_val_preds = np.clip(ensemble_val_preds, target_min, target_max)
ensemble_test_preds = np.clip(ensemble_test_preds, target_min, target_max)

final_val_rmse = root_mean_squared_error(y_val, ensemble_val_preds)
print(f"Post-processing bounds: [{target_min:.2f}, {target_max:.2f}]")
print(f"Final Ensemble RMSE (original scale): {final_val_rmse:.4f}")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {
        "id": np.arange(len(ensemble_test_preds)),
        "Disbursements USD (Trust funds)": ensemble_test_preds,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())
print(
    f"Prediction stats: mean={ensemble_test_preds.mean():.2f}, std={ensemble_test_preds.std():.2f}, "
    f"min={ensemble_test_preds.min():.2f}, max={ensemble_test_preds.max():.2f}"
)

# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  LightGBM RMSE:       {lgb_val_rmse:.4f}")
print(f"  MLP RMSE:            {mlp_val_rmse:.4f}")
print(f"  Ensemble RMSE:       {final_val_rmse:.4f}")
print(f"  LGB weight:          {best_w:.2f}")
print(f"  Post-processing:     clip to [0, {target_max:.2f}]")
print(f"  Metric:              RMSE on original scale")
print(f"  Validation split:    15% hold-out (random_state=42)")

score = final_val_rmse
print(f"Final Validation Score: {score}")