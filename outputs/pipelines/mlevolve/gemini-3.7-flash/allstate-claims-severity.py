import gc
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

# Set random seed for full reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
print("Loading train and test datasets...", flush=True)
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

id_col = "id"
target_col = "loss"

cat_cols = sorted([c for c in train_df.columns if c.startswith("cat")])
cont_cols = sorted([c for c in train_df.columns if c.startswith("cont")])

n_train = len(train_df)
n_test = len(test_df)
print(
    f"Train shape: {train_df.shape}, Test shape: {test_df.shape}",
    flush=True,
)
print(
    f"Categorical features: {len(cat_cols)}, Continuous features: {len(cont_cols)}",
    flush=True,
)

# Target transformation: log(loss + 200) optimal shift for Allstate claims MAE
SHIFT = 200.0
y_train_orig = train_df[target_col].values.astype(np.float64)
y_train_log = np.log(y_train_orig + SHIFT).astype(np.float32)


def to_loss(log_preds):
    clipped = np.clip(log_preds, 0.0, 15.0)
    return np.maximum(0.0, np.exp(clipped) - SHIFT)


# 2. Continuous Feature Engineering (Constructed via NumPy arrays to prevent DataFrame fragmentation)
print("Engineering continuous feature transformations...", flush=True)
cont_train_raw = train_df[cont_cols].values.astype(np.float32)
cont_test_raw = test_df[cont_cols].values.astype(np.float32)


def compute_cont_features(cont_mat, df):
    # Summary statistics across continuous features per sample
    mean_val = cont_mat.mean(axis=1, keepdims=True)
    std_val = cont_mat.std(axis=1, keepdims=True)
    min_val = cont_mat.min(axis=1, keepdims=True)
    max_val = cont_mat.max(axis=1, keepdims=True)
    sum_val = cont_mat.sum(axis=1, keepdims=True)
    skew_val = (((cont_mat - mean_val) ** 3).mean(axis=1, keepdims=True)) / (
        std_val**3 + 1e-6
    )

    # Non-linear transformations
    log_feat = np.log1p(cont_mat)
    sqrt_feat = np.sqrt(np.maximum(0.0, cont_mat))

    # Known correlated continuous feature interactions
    c1 = df["cont1"].values[:, None]
    c2 = df["cont2"].values[:, None]
    c6 = df["cont6"].values[:, None]
    c7 = df["cont7"].values[:, None]
    c9 = df["cont9"].values[:, None]
    c10 = df["cont10"].values[:, None]
    c11 = df["cont11"].values[:, None]
    c12 = df["cont12"].values[:, None]

    interactions = np.hstack(
        [
            c1 * c2,
            c1 / (c2 + 1e-4),
            c6 * c7,
            c6 / (c7 + 1e-4),
            c9 * c10,
            c9 / (c10 + 1e-4),
            c11 * c12,
            c11 / (c12 + 1e-4),
        ]
    ).astype(np.float32)

    all_cont = np.hstack(
        [
            cont_mat,
            mean_val,
            std_val,
            min_val,
            max_val,
            sum_val,
            skew_val,
            log_feat,
            sqrt_feat,
            interactions,
        ]
    )
    return all_cont.astype(np.float32)


X_cont_train = compute_cont_features(cont_train_raw, train_df)
X_cont_test = compute_cont_features(cont_test_raw, test_df)

# Category-Grouped Continuous Aggregations
print("Computing category-grouped continuous aggregations...", flush=True)
dom_cats = ["cat1", "cat80", "cat79", "cat101", "cat12", "cat87", "cat57", "cat72"]
agg_conts = ["cont1", "cont2", "cont7", "cont8", "cont9", "cont11", "cont14"]

cat_cont_train_list = []
cat_cont_test_list = []

for c_col in dom_cats:
    for num_col in agg_conts:
        grouped = train_df.groupby(c_col)[num_col]
        mean_map = grouped.mean().to_dict()
        std_map = grouped.std().fillna(0.0).to_dict()
        global_m = float(train_df[num_col].mean())
        global_s = float(train_df[num_col].std())

        tr_mean = (
            train_df[c_col]
            .map(mean_map)
            .fillna(global_m)
            .values.astype(np.float32)[:, None]
        )
        te_mean = (
            test_df[c_col]
            .map(mean_map)
            .fillna(global_m)
            .values.astype(np.float32)[:, None]
        )
        tr_std = (
            train_df[c_col]
            .map(std_map)
            .fillna(global_s)
            .values.astype(np.float32)[:, None]
        )
        te_std = (
            test_df[c_col]
            .map(std_map)
            .fillna(global_s)
            .values.astype(np.float32)[:, None]
        )

        cat_cont_train_list.extend([tr_mean, tr_std])
        cat_cont_test_list.extend([te_mean, te_std])

X_cat_cont_train = np.hstack(cat_cont_train_list)
X_cat_cont_test = np.hstack(cat_cont_test_list)

# Pairwise Categorical Combinations with Frequency Encoding
print("Computing pairwise categorical combinations and frequency encodings...", flush=True)
cat_pairs = [
    ("cat80", "cat87"),
    ("cat79", "cat80"),
    ("cat1", "cat2"),
    ("cat101", "cat79"),
    ("cat12", "cat80"),
    ("cat57", "cat80"),
    ("cat1", "cat80"),
]
pair_freq_train_list = []
pair_freq_test_list = []

for c1, c2 in cat_pairs:
    tr_comb = train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
    te_comb = test_df[c1].astype(str) + "_" + test_df[c2].astype(str)
    vc = tr_comb.value_counts(normalize=True).to_dict()
    tr_pf = np.array([vc.get(v, 0.0) for v in tr_comb], dtype=np.float32)[:, None]
    te_pf = np.array([vc.get(v, 0.0) for v in te_comb], dtype=np.float32)[:, None]
    pair_freq_train_list.append(tr_pf)
    pair_freq_test_list.append(te_pf)

X_pair_freq_train = np.hstack(pair_freq_train_list)
X_pair_freq_test = np.hstack(pair_freq_test_list)

# 3. Categorical Feature Encoding (Ordinal & Frequency Encoding)
print("Encoding categorical features (Ordinal + Frequency)...", flush=True)
cat_ord_train_list = []
cat_ord_test_list = []
cat_freq_train_list = []
cat_freq_test_list = []
high_card_cols = []

for col in cat_cols:
    tr_vals = train_df[col].astype(str).values
    te_vals = test_df[col].astype(str).values

    unique_cats = np.unique(np.concatenate([tr_vals, te_vals]))
    cat_to_int = {val: idx for idx, val in enumerate(unique_cats)}

    tr_ord = np.array([cat_to_int[v] for v in tr_vals], dtype=np.int32)
    te_ord = np.array([cat_to_int[v] for v in te_vals], dtype=np.int32)
    cat_ord_train_list.append(tr_ord[:, None])
    cat_ord_test_list.append(te_ord[:, None])

    # Normalized frequency encoding based on training set distribution
    val_counts = train_df[col].value_counts(normalize=True).to_dict()
    tr_freq = np.array([val_counts.get(v, 0.0) for v in tr_vals], dtype=np.float32)
    te_freq = np.array([val_counts.get(v, 0.0) for v in te_vals], dtype=np.float32)
    cat_freq_train_list.append(tr_freq[:, None])
    cat_freq_test_list.append(te_freq[:, None])

    if len(unique_cats) > 4:
        high_card_cols.append(col)

X_cat_ord_train = np.hstack(cat_ord_train_list)
X_cat_ord_test = np.hstack(cat_ord_test_list)
X_cat_freq_train = np.hstack(cat_freq_train_list)
X_cat_freq_test = np.hstack(cat_freq_test_list)

# 4. Out-of-Fold Leak-Free Target Encoding for High-Cardinality Categoricals
print(
    f"Calculating leak-free out-of-fold target encodings for {len(high_card_cols)} features...",
    flush=True,
)
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = np.zeros(n_train, dtype=np.int32)
for f, (_, val_idx) in enumerate(kf.split(train_df)):
    folds[val_idx] = f

global_mean = float(y_train_log.mean())
SMOOTHING = 15.0

X_te_train = np.zeros((n_train, len(high_card_cols)), dtype=np.float32)
X_te_test = np.zeros((n_test, len(high_card_cols)), dtype=np.float32)

for c_idx, col in enumerate(high_card_cols):
    # Out-of-fold computation on training folds
    for f in range(N_SPLITS):
        tr_mask = folds != f
        va_mask = folds == f

        sub_df = pd.DataFrame(
            {"cat": train_df.loc[tr_mask, col], "target": y_train_log[tr_mask]}
        )
        stats = sub_df.groupby("cat")["target"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + SMOOTHING * global_mean) / (
            stats["count"] + SMOOTHING
        )
        smoothed_dict = smoothed.to_dict()

        va_vals = train_df.loc[va_mask, col].values
        X_te_train[va_mask, c_idx] = [
            smoothed_dict.get(v, global_mean) for v in va_vals
        ]

    # Full train target encoding for test set
    full_df = pd.DataFrame({"cat": train_df[col], "target": y_train_log})
    stats_full = full_df.groupby("cat")["target"].agg(["count", "mean"])
    smoothed_full = (
        stats_full["count"] * stats_full["mean"] + SMOOTHING * global_mean
    ) / (stats_full["count"] + SMOOTHING)
    smoothed_full_dict = smoothed_full.to_dict()

    test_vals = test_df[col].values
    X_te_test[:, c_idx] = [smoothed_full_dict.get(v, global_mean) for v in test_vals]

# 5. Assemble Full Feature Matrices
print("Stacking finalized feature matrices...", flush=True)
X_train_full = np.hstack(
    [
        X_cont_train,
        X_cat_cont_train,
        X_pair_freq_train,
        X_cat_ord_train,
        X_cat_freq_train,
        X_te_train,
    ]
).astype(np.float32)
X_test_full = np.hstack(
    [
        X_cont_test,
        X_cat_cont_test,
        X_pair_freq_test,
        X_cat_ord_test,
        X_cat_freq_test,
        X_te_test,
    ]
).astype(np.float32)

print(
    f"Total features: {X_train_full.shape[1]} | Train Matrix: {X_train_full.shape} | Test Matrix: {X_test_full.shape}",
    flush=True,
)

# Clean up memory
del (
    X_cont_train,
    X_cont_test,
    X_cat_cont_train,
    X_cat_cont_test,
    X_pair_freq_train,
    X_pair_freq_test,
    X_cat_ord_train,
    X_cat_ord_test,
    X_cat_freq_train,
    X_cat_freq_test,
    X_te_train,
    X_te_test,
)
gc.collect()

# 6. Model Training - 5-Fold Cross-Validation Ensemble
# Model 1: LightGBM with L1 (MAE) Objective
lgb_params_l1 = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "n_jobs": 4,
    "random_state": 42,
    "verbose": -1,
}

# Model 2: LightGBM with Huber Objective (Robust Regression)
lgb_params_huber = {
    "objective": "huber",
    "huber_delta": 1.2,
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 95,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "n_jobs": 4,
    "random_state": 2024,
    "verbose": -1,
}

# Model 3: XGBoost with L1 (MAE) Objective
xgb_params = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "learning_rate": 0.04,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.5,
    "reg_lambda": 2.0,
    "n_estimators": 2500,
    "early_stopping_rounds": 40,
    "random_state": 101,
    "n_jobs": 4,
}

# Model 4: PyTorch Tabular Deep Neural Network
class TabularNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


oof_m1 = np.zeros(n_train, dtype=np.float64)
oof_m2 = np.zeros(n_train, dtype=np.float64)
oof_m3 = np.zeros(n_train, dtype=np.float64)
oof_m4 = np.zeros(n_train, dtype=np.float64)

test_m1 = np.zeros(n_test, dtype=np.float64)
test_m2 = np.zeros(n_test, dtype=np.float64)
test_m3 = np.zeros(n_test, dtype=np.float64)
test_m4 = np.zeros(n_test, dtype=np.float64)

print(f"\n--- Commencing {N_SPLITS}-Fold Ensemble Training ---", flush=True)

for fold in range(N_SPLITS):
    tr_idx = folds != fold
    va_idx = folds == fold

    X_tr, y_tr = X_train_full[tr_idx], y_train_log[tr_idx]
    X_va, y_va = X_train_full[va_idx], y_train_log[va_idx]
    y_va_orig_fold = y_train_orig[va_idx]

    # --- Train Model 1 (LightGBM L1) ---
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    callbacks_lgb = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    bst1 = lgb.train(
        lgb_params_l1,
        dtr,
        num_boost_round=2500,
        valid_sets=[dva],
        callbacks=callbacks_lgb,
    )
    pred_va_m1 = bst1.predict(X_va, num_iteration=bst1.best_iteration)
    oof_m1[va_idx] = pred_va_m1
    test_m1 += bst1.predict(X_test_full, num_iteration=bst1.best_iteration) / N_SPLITS

    # --- Train Model 2 (LightGBM Huber) ---
    bst2 = lgb.train(
        lgb_params_huber,
        dtr,
        num_boost_round=2500,
        valid_sets=[dva],
        callbacks=callbacks_lgb,
    )
    pred_va_m2 = bst2.predict(X_va, num_iteration=bst2.best_iteration)
    oof_m2[va_idx] = pred_va_m2
    test_m2 += bst2.predict(X_test_full, num_iteration=bst2.best_iteration) / N_SPLITS

    # --- Train Model 3 (XGBoost L1 MAE) ---
    xgb_reg = xgb.XGBRegressor(**xgb_params)
    xgb_reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_va_m3 = np.clip(xgb_reg.predict(X_va), 0.0, 15.0)
    oof_m3[va_idx] = pred_va_m3
    test_m3 += np.clip(xgb_reg.predict(X_test_full), 0.0, 15.0) / N_SPLITS

    # --- Train Model 4 (PyTorch Tabular Deep NN) ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test_full)

    train_ds = TabularDataset(X_tr_scaled, y_tr)
    val_ds = TabularDataset(X_va_scaled, y_va)
    test_ds = TabularDataset(X_te_scaled)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    nn_model = TabularNN(X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22, eta_min=1e-5)

    best_nn_val_mae = float("inf")
    best_nn_preds = None
    best_nn_state = None

    for epoch in range(22):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = nn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_preds_list.append(nn_model(batch_x).cpu().numpy())
        val_preds_arr = np.concatenate(val_preds_list)
        val_mae_epoch = mean_absolute_error(
            y_va_orig_fold, to_loss(val_preds_arr)
        )
        if val_mae_epoch < best_nn_val_mae:
            best_nn_val_mae = val_mae_epoch
            best_nn_preds = val_preds_arr
            best_nn_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_state.items()})
    oof_m4[va_idx] = best_nn_preds

    nn_model.eval()
    test_preds_list = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            test_preds_list.append(nn_model(batch_x).cpu().numpy())
    test_m4 += np.concatenate(test_preds_list) / N_SPLITS

    # Compute fold individual MAEs
    mae_m1 = mean_absolute_error(y_va_orig_fold, to_loss(pred_va_m1))
    mae_m2 = mean_absolute_error(y_va_orig_fold, to_loss(pred_va_m2))
    mae_m3 = mean_absolute_error(y_va_orig_fold, to_loss(pred_va_m3))
    mae_m4 = mean_absolute_error(y_va_orig_fold, to_loss(best_nn_preds))

    print(
        f"Fold {fold+1}/{N_SPLITS} | LGB-L1 MAE: {mae_m1:.3f} | LGB-Huber MAE: {mae_m2:.3f} | XGB-Huber MAE: {mae_m3:.3f} | NN MAE: {mae_m4:.3f}",
        flush=True,
    )
    gc.collect()

# 7. Model Ensembling & Optimal Simplex Weights
print("\nOptimizing ensemble weights on Out-of-Fold predictions...", flush=True)


def ensemble_objective(weights):
    w = np.maximum(0.0, weights)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 1e9
    w = w / w_sum
    blend_log = (
        w[0] * oof_m1 + w[1] * oof_m2 + w[2] * oof_m3 + w[3] * oof_m4
    )
    blend_orig = to_loss(blend_log)
    return mean_absolute_error(y_train_orig, blend_orig)


opt_res = minimize(
    ensemble_objective,
    [0.35, 0.35, 0.15, 0.15],
    method="Nelder-Mead",
    options={"maxiter": 500},
)
opt_w = np.maximum(0.0, opt_res.x)
opt_w = opt_w / np.sum(opt_w)
print(
    f"Optimal Ensemble Weights: LGB-L1={opt_w[0]:.3f}, LGB-Huber={opt_w[1]:.3f}, XGB-Huber={opt_w[2]:.3f}, TabularNN={opt_w[3]:.3f}",
    flush=True,
)

oof_blend_log = (
    opt_w[0] * oof_m1
    + opt_w[1] * oof_m2
    + opt_w[2] * oof_m3
    + opt_w[3] * oof_m4
)
oof_blend_orig = to_loss(oof_blend_log)


# Scale Calibration
def scale_objective(scale):
    return mean_absolute_error(y_train_orig, oof_blend_orig * scale)


scale_res = minimize_scalar(scale_objective, bounds=(0.95, 1.05), method="bounded")
opt_scale = scale_res.x
print(f"Optimal Scaling Multiplier: {opt_scale:.5f}", flush=True)

final_oof_preds = oof_blend_orig * opt_scale
final_oof_mae = mean_absolute_error(y_train_orig, final_oof_preds)
print(f"Ensemble Out-of-Fold MAE: {final_oof_mae:.4f}", flush=True)

# 8. Test Inference & Submission Generation
print("Generating final test predictions...", flush=True)
test_blend_log = (
    opt_w[0] * test_m1
    + opt_w[1] * test_m2
    + opt_w[2] * test_m3
    + opt_w[3] * test_m4
)
test_loss_preds = (to_loss(test_blend_log) * opt_scale).astype(
    np.float64
)

submission_df = pd.DataFrame({"id": test_df[id_col].values, "loss": test_loss_preds})

assert not submission_df.isna().any().any(), "Submission file contains null values!"
assert len(submission_df) == n_test, "Submission row count mismatch!"

sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)
print(
    f"Submission saved successfully to {sub_path} ({len(submission_df)} rows).",
    flush=True,
)

# 9. Required Validation Score Output
print(f"Final Validation Score: {final_oof_mae}")
