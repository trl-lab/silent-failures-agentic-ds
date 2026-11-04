import os
import warnings
from typing import Any, Dict, Optional, Tuple

from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Setup Directories and Load Raw Data
# ---------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Standardize target variable
target_col = "outcome"
if target_col not in train_df.columns:
    potential_targets = [
        c
        for c in train_df.columns
        if c.lower() in ["outcome", "target", "label", "class"]
    ]
    if potential_targets:
        target_col = potential_targets[0]


def standardize_target(y_series: pd.Series) -> pd.Series:
    if y_series.dtype == "object" or pd.api.types.is_categorical_dtype(y_series):
        mapping = {
            "0": 0,
            "1": 1,
            "0.0": 0,
            "1.0": 1,
            "failure": 0,
            "success": 1,
            "fail": 0,
            "succeed": 1,
            "crash": 0,
            "no_crash": 1,
            "false": 0,
            "true": 1,
            "False": 0,
            "True": 1,
        }
        y_clean = (
            y_series.astype(str)
            .str.strip()
            .str.lower()
            .map(
                lambda x: mapping.get(
                    x,
                    int(x) if x.isdigit() else (1 if "succ" in x or "pass" in x else 0),
                )
            )
        )
        return y_clean.astype(int)
    return y_series.astype(int)


y_train = standardize_target(train_df[target_col])

# Extract test IDs and filter non-feature metadata columns
ignored_cols = [
    target_col,
    "id",
    "Id",
    "ID",
    "index",
    "study",
    "run",
    "study_number",
    "run_number",
    "batch",
]
raw_feature_cols = [
    c
    for c in train_df.columns
    if c not in ignored_cols and np.issubdtype(train_df[c].dtype, np.number)
]

if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "ID" in test_df.columns:
    test_ids = test_df["ID"].values
else:
    test_ids = np.arange(len(test_df))


# ---------------------------------------------------------------------------
# 2. Physics-Informed Feature Engineering Pipeline (<50 compact features)
# ---------------------------------------------------------------------------
# Feature engineering pipeline
raw_pairs = list(combinations(raw_feature_cols, 2))[:15]
selected_interaction_pairs = raw_pairs


def generate_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    feat_df = pd.DataFrame(index=df.index)
    X = df[feature_cols].values.astype(np.float64)

    # 1. Base continuous parameters (18)
    for col in feature_cols:
        feat_df[col] = df[col].astype(np.float64)

    # 2. Bounded coordinate geometry: hypercube boundary proximities
    boundary_dist = np.minimum(X, 1.0 - X)
    feat_df["min_boundary_distance"] = boundary_dist.min(axis=1)
    feat_df["mean_boundary_distance"] = boundary_dist.mean(axis=1)
    feat_df["boundary_distance_std"] = boundary_dist.std(axis=1)
    feat_df["num_extreme_low"] = (X < 0.05).astype(float).sum(axis=1)
    feat_df["num_extreme_high"] = (X > 0.95).astype(float).sum(axis=1)
    feat_df["total_extreme_params"] = feat_df["num_extreme_low"] + feat_df["num_extreme_high"]

    # 3. Geometric distances to hypercube center (0.5)
    centered_X = X - 0.5
    feat_df["dist_center_l2"] = np.sqrt(np.maximum(np.sum(centered_X**2, axis=1), 0.0))
    feat_df["dist_center_l1"] = np.abs(centered_X).sum(axis=1)
    feat_df["dist_center_linf"] = np.abs(centered_X).max(axis=1)

    # 4. Interaction pairs
    for col_a, col_b in selected_interaction_pairs:
        feat_df[f"{col_a}_x_{col_b}"] = df[col_a].values.astype(np.float64) * df[col_b].values.astype(np.float64)

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat_df


X_train_df = generate_features(train_df, raw_feature_cols)
X_test_df = generate_features(test_df, raw_feature_cols)

feature_cols = list(X_train_df.columns)
X_train_full = X_train_df.values.astype(np.float32)
y_train_full = y_train.values.astype(np.int64)
X_test_full = X_test_df.values.astype(np.float32)

n_samples = len(X_train_full)
n_test = len(X_test_full)
input_dim = len(feature_cols)


# ---------------------------------------------------------------------------
# 3. Model Architectures: PyTorch Residual MLP & Ensemble Learners
# ---------------------------------------------------------------------------
class TabularResidualMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


class PyTorchTabularClassifier:
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        epochs: int = 70,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = TabularResidualMLP(
            self.in_features, self.hidden_dim, self.dropout
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - probs, probs])


def build_simulation_models(input_dim: int, random_state: int = 42) -> Dict[str, Any]:
    models = {
        "xgb": xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric="auc",
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        ),
        "lgb": lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            num_leaves=4,
            min_child_samples=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=3,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=250,
            max_depth=3,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svc_rbf": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=random_state),
            method="sigmoid",
            cv=3,
        ),
        "svc_poly": CalibratedClassifierCV(
            SVC(kernel="poly", degree=2, C=1.0, probability=True, random_state=random_state),
            method="sigmoid",
            cv=3,
        ),
        "logreg_elastic": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.5,
            max_iter=1500,
            random_state=random_state,
        ),
        "mlp": PyTorchTabularClassifier(
            in_features=input_dim,
            hidden_dim=64,
            dropout=0.2,
            epochs=75,
            batch_size=32,
            random_state=random_state,
        ),
    }

    if HAS_CATBOOST:
        models["cb"] = cb.CatBoostClassifier(
            iterations=250,
            learning_rate=0.03,
            depth=2,
            l2_leaf_reg=5.0,
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
            thread_count=-1,
        )

    return models


# ---------------------------------------------------------------------------
# 4. Repeated Stratified 10x5 Cross-Validation Training & Inference
# ---------------------------------------------------------------------------
n_splits = 10
n_repeats = 5
total_folds = n_splits * n_repeats
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

temp_models = build_simulation_models(input_dim=input_dim, random_state=42)
model_names = list(temp_models.keys())
del temp_models

oof_preds = {m: np.zeros(n_samples, dtype=np.float64) for m in model_names}
test_preds = {m: np.zeros(n_test, dtype=np.float64) for m in model_names}

for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_train_full, y_train_full)):
    X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    X_te = X_test_full.copy()

    # Leak-free feature scaling strictly fit on training fold
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_te)

    # Quantile transformer for regularized kernel SVMs
    n_quantiles = min(100, len(X_tr))
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=42 + fold_idx)
    X_tr_qt = qt.fit_transform(X_tr)
    X_val_qt = qt.transform(X_val)
    X_te_qt = qt.transform(X_te)

    fold_models = build_simulation_models(input_dim=input_dim, random_state=42 + fold_idx)

    for m_name, model in fold_models.items():
        if m_name in {"svc_rbf", "svc_poly"}:
            model.fit(X_tr_qt, y_tr)
            val_p = model.predict_proba(X_val_qt)[:, 1]
            te_p = model.predict_proba(X_te_qt)[:, 1]
        elif m_name in {"mlp", "logreg_elastic"}:
            model.fit(X_tr_scaled, y_tr)
            val_p = model.predict_proba(X_val_scaled)[:, 1]
            te_p = model.predict_proba(X_te_scaled)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            val_p = model.predict_proba(X_val)[:, 1]
            te_p = model.predict_proba(X_te)[:, 1]

        oof_preds[m_name][val_idx] += val_p / n_repeats
        test_preds[m_name] += te_p / total_folds

    if (fold_idx + 1) % n_splits == 0:
        rep = (fold_idx + 1) // n_splits
        print(f"Repeat {rep}/{n_repeats} complete ({fold_idx + 1}/{total_folds} folds)")

# ---------------------------------------------------------------------------
# 5. Quality Filtering & Constrained SLSQP Probability Optimization
# ---------------------------------------------------------------------------
print("\n--- Out-of-Fold Model Performance ---")
oof_scores = {}
for m in model_names:
    score = roc_auc_score(y_train_full, oof_preds[m])
    oof_scores[m] = score
    print(f"OOF AUC - {m:15s}: {score:.5f}")

# Filter out low-performing models (<0.92 AUC) to prevent ensemble degradation
threshold = 0.92
active_models = [m for m in model_names if oof_scores[m] >= threshold]
if len(active_models) < 2:
    active_models = sorted(model_names, key=lambda m: oof_scores[m], reverse=True)[:3]

print(f"\nActive models selected for ensembling (AUC >= {threshold}): {active_models}")

oof_prob_matrix = np.column_stack([oof_preds[m] for m in active_models])
test_prob_matrix = np.column_stack([test_preds[m] for m in active_models])
n_active = len(active_models)

# Smooth Wilcoxon-Mann-Whitney loss surrogate for gradient-based SLSQP
pos_idx = np.where(y_train_full == 1)[0]
neg_idx = np.where(y_train_full == 0)[0]


def smooth_auc_loss(w: np.ndarray, tau: float = 0.05) -> float:
    w = np.maximum(w, 0.0)
    w_sum = w.sum()
    if w_sum < 1e-8:
        return 0.0
    w = w / w_sum
    pred = np.dot(oof_prob_matrix, w)
    diff = pred[pos_idx, None] - pred[neg_idx]
    smooth_auc = torch.sigmoid(torch.tensor(diff / tau, dtype=torch.float64)).mean().item()
    return -smooth_auc


def exact_auc_loss(w: np.ndarray) -> float:
    w = np.maximum(w, 0.0)
    w_sum = w.sum()
    if w_sum < 1e-8:
        return 0.0
    w = w / w_sum
    pred = np.dot(oof_prob_matrix, w)
    return -roc_auc_score(y_train_full, pred)


init_weights = np.ones(n_active, dtype=np.float64) / n_active
bounds = [(0.0, 1.0) for _ in range(n_active)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

res_slsqp = minimize(
    smooth_auc_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 300, "ftol": 1e-6},
)

init_for_nm = res_slsqp.x if res_slsqp.success else init_weights
res_nm = minimize(
    exact_auc_loss,
    init_for_nm,
    method="Nelder-Mead",
    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
)

opt_w_slsqp = np.maximum(res_slsqp.x, 0.0)
opt_w_slsqp /= opt_w_slsqp.sum()

opt_w_nm = np.maximum(res_nm.x, 0.0)
opt_w_nm /= opt_w_nm.sum()

candidates = [
    ("SLSQP Smooth AUC", opt_w_slsqp),
    ("Nelder-Mead Fine-Tuned", opt_w_nm),
    ("Equal Weights", init_weights),
]
for idx, m in enumerate(active_models):
    single_w = np.zeros(n_active)
    single_w[idx] = 1.0
    candidates.append((f"Single Model ({m})", single_w))

best_name, best_weights, best_score = None, None, -1.0
for name, w in candidates:
    pred = np.dot(oof_prob_matrix, w)
    score = roc_auc_score(y_train_full, pred)
    if score > best_score:
        best_score = score
        best_name = name
        best_weights = w

print(f"\nBest Blending Scheme: {best_name} with OOF AUC: {best_score:.5f}")
optimal_weights = best_weights

print("Final Ensemble Weights:")
for m_name, w in zip(active_models, optimal_weights):
    print(f"  {m_name:15s}: {w:.4f}")

# Calculate final validation score
final_oof_predictions = np.dot(oof_prob_matrix, optimal_weights)
final_validation_score = float(roc_auc_score(y_train_full, final_oof_predictions))

# ---------------------------------------------------------------------------
# 6. Generate Submission File
# ---------------------------------------------------------------------------
final_test_predictions = np.dot(test_prob_matrix, optimal_weights)
final_test_predictions = np.clip(final_test_predictions, 0.0, 1.0)

submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "outcome": final_test_predictions}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_score}")