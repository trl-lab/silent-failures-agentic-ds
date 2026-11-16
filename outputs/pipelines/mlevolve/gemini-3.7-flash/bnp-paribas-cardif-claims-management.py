import gc
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# -----------------------------------------------------------------------------
# Configuration & Reproducibility Setup
# -----------------------------------------------------------------------------
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 1. Tabular Residual Neural Network Architecture (PyTorch)
# -----------------------------------------------------------------------------
class TabularResNetBlock(nn.Module):
    """Residual Block for tabular representations with LayerNorm, Dropout, and Swish."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.norm2(out)
        out = self.linear2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return out + residual


class TabularResNet(nn.Module):
    """Deep Tabular Network designed for binary classification with log_loss objective."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)

        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(
                TabularResNetBlock(curr_dim, hidden_dim, dropout_rate=dropout_rate)
            )
            curr_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head_norm = nn.LayerNorm(curr_dim)
        self.classifier = nn.Linear(curr_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        features = self.backbone(x)
        features = self.head_norm(features)
        logits = self.classifier(features)
        return logits.squeeze(-1)


# -----------------------------------------------------------------------------
# 2. Hyperparameter Helper Functions
# -----------------------------------------------------------------------------
def get_lgbm_params(seed: int = 42) -> dict:
    """LightGBM hyperparameter configuration tuned for binary log_loss."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": 7,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "lambda_l1": 1.5,
        "lambda_l2": 3.0,
        "random_state": seed,
        "n_estimators": 2500,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    """XGBoost hyperparameter configuration tuned for binary log_loss."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 4,
        "colsample_bytree": 0.70,
        "subsample": 0.85,
        "reg_alpha": 1.2,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_estimators": 2500,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> dict:
    """CatBoost hyperparameter configuration tuned for binary log_loss."""
    return {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "Logloss",
        "random_seed": seed,
        "verbose": False,
        "early_stopping_rounds": 50,
        "thread_count": -1,
    }


def compute_log_loss(y_true, y_pred, eps: float = 1e-15) -> float:
    """Official competition metric evaluation with safe bounding."""
    clipped_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, clipped_pred))


# -----------------------------------------------------------------------------
# 3. Data Loading and Feature Engineering
# -----------------------------------------------------------------------------
print("Loading train and test parquet datasets...")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

train_ids = (
    train_df["ID"].values
    if "ID" in train_df.columns
    else (
        train_df["id"].values if "id" in train_df.columns else np.arange(len(train_df))
    )
)
test_ids = (
    test_df["ID"].values
    if "ID" in test_df.columns
    else (test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df)))
)
y_train = train_df["target"].values.astype(np.int32)

feature_cols = [c for c in train_df.columns if c not in ["ID", "id", "target"]]
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

del train_df, test_df
gc.collect()

cat_cols = [
    c
    for c in feature_cols
    if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("category")
]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Missingness indicators & Summary statistics across continuous features
X_train["null_count"] = X_train.isnull().sum(axis=1).astype(np.float32)
X_test["null_count"] = X_test.isnull().sum(axis=1).astype(np.float32)

X_train["num_mean"] = X_train[num_cols].mean(axis=1).astype(np.float32)
X_test["num_mean"] = X_test[num_cols].mean(axis=1).astype(np.float32)

X_train["num_std"] = X_train[num_cols].std(axis=1).astype(np.float32)
X_test["num_std"] = X_test[num_cols].std(axis=1).astype(np.float32)

X_train["num_median"] = X_train[num_cols].median(axis=1).astype(np.float32)
X_test["num_median"] = X_test[num_cols].median(axis=1).astype(np.float32)

X_train["num_skew"] = X_train[num_cols].skew(axis=1).astype(np.float32)
X_test["num_skew"] = X_test[num_cols].skew(axis=1).astype(np.float32)

X_train["num_min"] = X_train[num_cols].min(axis=1).astype(np.float32)
X_test["num_min"] = X_test[num_cols].min(axis=1).astype(np.float32)

X_train["num_max"] = X_train[num_cols].max(axis=1).astype(np.float32)
X_test["num_max"] = X_test[num_cols].max(axis=1).astype(np.float32)

# Continuous Pairwise Ratios and Differences
top_num_pairs = [
    ("v50", "v10"),
    ("v50", "v12"),
    ("v50", "v14"),
    ("v10", "v12"),
    ("v10", "v14"),
    ("v12", "v14"),
    ("v34", "v40"),
    ("v50", "v34"),
    ("v10", "v40"),
    ("v114", "v40"),
    ("v114", "v21"),
]
for n1, n2 in top_num_pairs:
    if n1 in num_cols and n2 in num_cols:
        diff_col = f"{n1}_minus_{n2}"
        ratio_col = f"{n1}_ratio_{n2}"
        X_train[diff_col] = (X_train[n1] - X_train[n2]).astype(np.float32)
        X_test[diff_col] = (X_test[n1] - X_test[n2]).astype(np.float32)
        X_train[ratio_col] = ((X_train[n1] + 1e-4) / (X_train[n2] + 1e-4)).astype(np.float32)
        X_test[ratio_col] = ((X_test[n1] + 1e-4) / (X_test[n2] + 1e-4)).astype(np.float32)

# Categorical Interactions
cat_pairs = [
    ("v24", "v30"),
    ("v47", "v110"),
    ("v31", "v66"),
    ("v71", "v75"),
    ("v22", "v56"),
    ("v66", "v110"),
    ("v3", "v31"),
    ("v24", "v31"),
    ("v30", "v110"),
]
for c1, c2 in cat_pairs:
    if c1 in cat_cols and c2 in cat_cols:
        interaction_col = f"{c1}_{c2}"
        X_train[interaction_col] = (
            X_train[c1].astype(str) + "_" + X_train[c2].astype(str)
        )
        X_test[interaction_col] = (
            X_test[c1].astype(str) + "_" + X_test[c2].astype(str)
        )
        cat_cols.append(interaction_col)

# Frequency Encoding for Categoricals
for col in cat_cols:
    freq_col = f"{col}_freq"
    combined_series = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
    freq_map = combined_series.value_counts(dropna=False, normalize=True).to_dict()
    X_train[freq_col] = X_train[col].astype(str).map(freq_map).fillna(0).astype(np.float32)
    X_test[freq_col] = X_test[col].astype(str).map(freq_map).fillna(0).astype(np.float32)

# Setup Stratified Folds
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = np.zeros(len(y_train), dtype=np.int32)
for fold, (_, val_idx) in enumerate(skf.split(X_train, y_train)):
    folds[val_idx] = fold

# Out-of-Fold (OOF) Smoothed Target Encoding (Leak-Free)
target_enc_cols = list(cat_cols)
smoothing = 10.0
global_mean = float(np.mean(y_train))

for col in target_enc_cols:
    te_col = f"{col}_te"
    X_train[te_col] = np.zeros(len(X_train), dtype=np.float32)
    train_str = X_train[col].astype(str)
    test_str = X_test[col].astype(str)

    for fold in range(N_SPLITS):
        val_mask = folds == fold
        trn_mask = ~val_mask

        trn_cat = train_str[trn_mask]
        trn_y = y_train[trn_mask]

        stats = (
            pd.DataFrame({"cat": trn_cat, "y": trn_y})
            .groupby("cat", observed=False)["y"]
            .agg(["count", "mean"])
        )
        smoothed_val = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        te_map = smoothed_val.to_dict()

        val_cat = train_str[val_mask]
        X_train.loc[val_mask, te_col] = (
            val_cat.map(te_map).fillna(global_mean).astype(np.float32).values
        )

    # Full train encoding for test set
    stats_full = (
        pd.DataFrame({"cat": train_str, "y": y_train})
        .groupby("cat", observed=False)["y"]
        .agg(["count", "mean"])
    )
    smoothed_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    te_map_full = smoothed_full.to_dict()
    X_test[te_col] = test_str.map(te_map_full).fillna(global_mean).astype(np.float32).values

# Label factorize categoricals for tree compatibility
for col in cat_cols:
    train_vals = X_train[col].astype(str)
    test_vals = X_test[col].astype(str)

    unique_cats = {val: idx for idx, val in enumerate(train_vals.unique())}
    X_train[col] = train_vals.map(unique_cats).fillna(-1).astype(np.int32)
    X_test[col] = test_vals.map(unique_cats).fillna(-1).astype(np.int32)

# Ensure consistent float32 representations
for col in X_train.columns:
    if X_train[col].dtype == np.float64:
        X_train[col] = X_train[col].astype(np.float32)
        X_test[col] = X_test[col].astype(np.float32)

n_samples, n_features = X_train.shape
n_test = len(X_test)
print(
    f"Engineered dataset: Samples={n_samples}, Features={n_features}, Test Samples={n_test}"
)

# -----------------------------------------------------------------------------
# 4. Multi-Model Cross-Validation Training Loop
# -----------------------------------------------------------------------------
oof_lgb = np.zeros(n_samples, dtype=np.float64)
oof_xgb = np.zeros(n_samples, dtype=np.float64)
oof_cat = np.zeros(n_samples, dtype=np.float64)
oof_nn = np.zeros(n_samples, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cat = np.zeros(n_test, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

for fold in range(N_SPLITS):
    trn_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_trn_df, y_trn = X_train.iloc[trn_idx], y_train[trn_idx]
    X_val_df, y_val = X_train.iloc[val_idx], y_train[val_idx]

    # --- 4A. Train LightGBM ---
    lgb_params = get_lgbm_params(seed=RANDOM_STATE + fold)
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(
        X_trn_df,
        y_trn,
        eval_set=[(X_val_df, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_p_lgb = clf_lgb.predict_proba(X_val_df)[:, 1]
    oof_lgb[val_idx] = val_p_lgb
    test_preds_lgb += clf_lgb.predict_proba(X_test)[:, 1] / N_SPLITS
    lgb_score = compute_log_loss(y_val, val_p_lgb)

    # --- 4B. Train XGBoost ---
    xgb_params = get_xgb_params(seed=RANDOM_STATE + fold)
    clf_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    clf_xgb.fit(X_trn_df, y_trn, eval_set=[(X_val_df, y_val)], verbose=False)

    val_p_xgb = clf_xgb.predict_proba(X_val_df)[:, 1]
    oof_xgb[val_idx] = val_p_xgb
    test_preds_xgb += clf_xgb.predict_proba(X_test)[:, 1] / N_SPLITS
    xgb_score = compute_log_loss(y_val, val_p_xgb)

    # --- 4C. Train CatBoost ---
    cat_params = get_catboost_params(seed=RANDOM_STATE + fold)
    clf_cat = CatBoostClassifier(**cat_params)
    clf_cat.fit(X_trn_df, y_trn, eval_set=(X_val_df, y_val), verbose=False)

    val_p_cat = clf_cat.predict_proba(X_val_df)[:, 1]
    oof_cat[val_idx] = val_p_cat
    test_preds_cat += clf_cat.predict_proba(X_test)[:, 1] / N_SPLITS
    cat_score = compute_log_loss(y_val, val_p_cat)

    # --- 4D. Train TabularResNet (PyTorch) ---
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_trn_nn = scaler.fit_transform(imputer.fit_transform(X_trn_df))
    X_val_nn = scaler.transform(imputer.transform(X_val_df))
    X_test_nn = scaler.transform(imputer.transform(X_test))

    train_dataset = TensorDataset(
        torch.tensor(X_trn_nn, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_nn, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1024, shuffle=False, drop_last=False
    )

    nn_model = TabularResNet(
        input_dim=n_features,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=12, eta_min=1e-5
    )

    best_nn_val_loss = float("inf")
    best_nn_preds = None
    best_model_state = None

    epochs = 10
    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                probs = torch.sigmoid(nn_model(batch_x)).cpu().numpy()
                val_preds_list.append(probs)
        current_val_preds = np.concatenate(val_preds_list)
        current_val_loss = compute_log_loss(y_val, current_val_preds)

        if current_val_loss < best_nn_val_loss:
            best_nn_val_loss = current_val_loss
            best_nn_preds = current_val_preds
            best_model_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    oof_nn[val_idx] = best_nn_preds

    # Infer test set using best neural weights for this fold
    nn_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    nn_model.eval()
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_nn, dtype=torch.float32)),
        batch_size=1024,
        shuffle=False,
    )
    nn_test_preds_list = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            probs = torch.sigmoid(nn_model(batch_x)).cpu().numpy()
            nn_test_preds_list.append(probs)
    test_preds_nn += np.concatenate(nn_test_preds_list) / N_SPLITS

    print(
        f"Fold {fold+1}/{N_SPLITS} | LGB: {lgb_score:.5f} | XGB: {xgb_score:.5f} | CAT: {cat_score:.5f} | NN: {best_nn_val_loss:.5f}"
    )

    del (
        clf_lgb,
        clf_xgb,
        clf_cat,
        nn_model,
        train_dataset,
        val_dataset,
        X_trn_nn,
        X_val_nn,
        X_test_nn,
    )
    gc.collect()

# -----------------------------------------------------------------------------
# 5. Out-of-Fold Model Blending Optimization (Log-Odds Space with Intercept)
# -----------------------------------------------------------------------------
print(f"Overall OOF - LightGBM: {compute_log_loss(y_train, oof_lgb):.5f}")
print(f"Overall OOF - XGBoost:  {compute_log_loss(y_train, oof_xgb):.5f}")
print(f"Overall OOF - CatBoost: {compute_log_loss(y_train, oof_cat):.5f}")
print(f"Overall OOF - TabResNet:{compute_log_loss(y_train, oof_nn):.5f}")

eps_logit = 1e-6
oof_mat = np.column_stack([
    logit(np.clip(oof_lgb, eps_logit, 1.0 - eps_logit)),
    logit(np.clip(oof_xgb, eps_logit, 1.0 - eps_logit)),
    logit(np.clip(oof_cat, eps_logit, 1.0 - eps_logit)),
    logit(np.clip(oof_nn, eps_logit, 1.0 - eps_logit)),
])

test_mat = np.column_stack([
    logit(np.clip(test_preds_lgb, eps_logit, 1.0 - eps_logit)),
    logit(np.clip(test_preds_xgb, eps_logit, 1.0 - eps_logit)),
    logit(np.clip(test_preds_cat, eps_logit, 1.0 - eps_logit)),
    logit(np.clip(test_preds_nn, eps_logit, 1.0 - eps_logit)),
])


def logit_ensemble_loss(params: np.ndarray) -> float:
    intercept = params[0]
    weights = params[1:]
    blended_logits = intercept + np.dot(oof_mat, weights)
    blended_probs = expit(blended_logits)
    return compute_log_loss(y_train, blended_probs)


init_params = np.array([0.0, 0.35, 0.30, 0.25, 0.10])
opt_res = minimize(
    logit_ensemble_loss,
    init_params,
    method="Nelder-Mead",
    options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-6},
)

opt_params = opt_res.x
print(
    f"Optimal logit weights: Bias={opt_params[0]:.4f}, LGB={opt_params[1]:.4f}, "
    f"XGB={opt_params[2]:.4f}, CAT={opt_params[3]:.4f}, NN={opt_params[4]:.4f}"
)

oof_final = expit(opt_params[0] + np.dot(oof_mat, opt_params[1:]))
oof_final = np.clip(oof_final, 1e-15, 1.0 - 1e-15)
final_val_score = compute_log_loss(y_train, oof_final)

# -----------------------------------------------------------------------------
# 6. Generate & Save Submission
# -----------------------------------------------------------------------------
test_final = expit(opt_params[0] + np.dot(test_mat, opt_params[1:]))
test_final = np.clip(test_final, 1e-15, 1.0 - 1e-15)

submission_df = pd.DataFrame({"id": test_ids, "target": test_final})

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission_df.shape}")

print(f"Final Validation Score: {final_val_score}")
