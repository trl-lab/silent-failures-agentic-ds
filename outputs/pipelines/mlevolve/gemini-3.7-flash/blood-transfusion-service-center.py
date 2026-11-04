import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================
# 0. Global Setup and Reproducibility
# =========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Target Standardization
# =========================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class"
if target_col not in train_df.columns:
    candidate_targets = [
        c
        for c in train_df.columns
        if c not in test_df.columns or "class" in c.lower() or "target" in c.lower()
    ]
    target_col = candidate_targets[0] if candidate_targets else train_df.columns[-1]

raw_target = train_df[target_col].copy()
if set(raw_target.unique()).issubset({1, 2, "1", "2"}):
    y_train = (raw_target.astype(int) == 2).astype(int)
elif set(raw_target.unique()).issubset({0, 1, "0", "1"}):
    y_train = raw_target.astype(int)
else:
    unique_vals = sorted(raw_target.unique())
    val_map = {unique_vals[0]: 0, unique_vals[-1]: 1}
    y_train = raw_target.map(val_map).astype(int)

train_features_df = train_df.drop(
    columns=[c for c in [target_col, "target"] if c in train_df.columns]
)
test_features_df = test_df.copy()


# =========================================================================
# 2. Domain Feature Engineering Pipeline
# =========================================================================
def resolve_columns(df):
    mapping = {}
    for c in df.columns:
        c_lower = c.lower()
        if "recency" in c_lower or c == "V1":
            mapping["recency"] = c
        elif "frequency" in c_lower or c == "V2":
            mapping["frequency"] = c
        elif "monetary" in c_lower or c == "V3":
            mapping["monetary"] = c
        elif "time" in c_lower or c == "V4":
            mapping["time"] = c

    cols = list(df.columns)
    if "recency" not in mapping and len(cols) >= 1:
        mapping["recency"] = cols[0]
    if "frequency" not in mapping and len(cols) >= 2:
        mapping["frequency"] = cols[1]
    if "monetary" not in mapping and len(cols) >= 3:
        mapping["monetary"] = cols[2]
    if "time" not in mapping and len(cols) >= 4:
        mapping["time"] = cols[3]
    return mapping


col_map = resolve_columns(train_features_df)


def engineer_features(df, col_map):
    feat = pd.DataFrame(index=df.index)

    rec = df[col_map["recency"]].astype(float)
    freq = df[col_map["frequency"]].astype(float)
    time = df[col_map["time"]].astype(float)

    # Core RFM dimensions (pruning collinear monetary = 250 * freq)
    feat["recency"] = rec
    feat["frequency"] = freq
    feat["time"] = time

    # Domain RFM features: inter-donation intervals, cadence, and momentum
    active_span = np.maximum(time - rec, 0.0)
    feat["active_span"] = active_span

    # True donor inter-donation interval: average time between repeat donations
    inter_donation_interval = (time - rec) / np.maximum(freq - 1.0, 1.0)
    feat["inter_donation_interval"] = inter_donation_interval

    # Average donation cadence across entire history
    cadence_interval = time / np.maximum(freq, 1.0)
    feat["cadence_interval"] = cadence_interval

    # Donation cadence and overdue ratios
    feat["cadence_overdue_ratio"] = rec / np.maximum(cadence_interval, 1.0)
    feat["interval_overdue_ratio"] = rec / np.maximum(inter_donation_interval, 1.0)
    feat["active_frequency"] = np.maximum(freq - 1.0, 0.0) / (active_span + 1.0)
    feat["donation_velocity"] = freq / (time + 1.0)
    feat["span_ratio"] = active_span / (time + 1.0)
    feat["recency_ratio"] = rec / (time + 1.0)
    feat["recency_to_span"] = rec / np.maximum(active_span, 1.0)
    feat["quarterly_time"] = time / 3.0
    feat["quarterly_recency"] = rec / 3.0
    feat["donations_per_year"] = freq / np.maximum(time / 12.0, 1.0)

    # Robust non-linear log and power basis expansions
    feat["log_recency"] = np.log1p(np.maximum(rec, 0.0))
    feat["log_frequency"] = np.log1p(np.maximum(freq, 0.0))
    feat["log_time"] = np.log1p(np.maximum(time, 0.0))
    feat["log_cadence"] = np.log1p(np.maximum(cadence_interval, 0.0))
    feat["log_interval"] = np.log1p(np.maximum(inter_donation_interval, 0.0))
    feat["sqrt_recency"] = np.sqrt(np.maximum(rec, 0.0))
    feat["sqrt_frequency"] = np.sqrt(np.maximum(freq, 0.0))
    feat["sqrt_time"] = np.sqrt(np.maximum(time, 0.0))

    # Binary domain indicators
    feat["is_one_time_donor"] = (freq <= 1.0).astype(float)
    feat["is_recent_donor"] = (rec <= 3.0).astype(float)
    feat["is_dormant_donor"] = (rec >= 12.0).astype(float)
    feat["is_overdue"] = (rec > cadence_interval).astype(float)
    feat["is_interval_overdue"] = (rec > inter_donation_interval).astype(float)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


X_train_engineered = engineer_features(train_features_df, col_map)
X_test_engineered = engineer_features(test_features_df, col_map)

# Setup 5-Fold Stratified K-Fold column for interface compatibility
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
fold_indices = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_engineered, y_train)):
    fold_indices[val_idx] = fold

X_train_engineered["fold"] = fold_indices
X_train_engineered["target"] = y_train.values

X_train_engineered.to_parquet("./working/train_engineered.parquet", index=False)
X_test_engineered.to_parquet("./working/test_engineered.parquet", index=False)


# =========================================================================
# 3. Model Architectures & Factories
# =========================================================================
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def get_torch_resnet_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    pos_weight: float = None,
    device: torch.device = None,
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=48, num_blocks=2, dropout_rate=0.2
    )
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            [pos_weight], dtype=torch.float32, device=device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    if device is not None:
        criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


def get_spline_lr_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "spline",
                SplineTransformer(
                    n_knots=5, degree=3, include_bias=False, extrapolation="linear"
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    C=0.3,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_svc_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_hist_gbm_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=12,
        min_samples_leaf=15,
        l2_regularization=1.5,
        learning_rate=0.03,
        class_weight="balanced",
        random_state=random_state,
    )


def get_shallow_lgb_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=8,
        max_depth=3,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=random_state,
        verbose=-1,
    )


def get_shallow_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_extra_trees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


# =========================================================================
# 4. Multi-Paradigm Repeated Stratified K-Fold Training & OOF Prediction
# =========================================================================
feature_cols = [c for c in X_train_engineered.columns if c not in ["fold", "target"]]
y_train_all = X_train_engineered["target"].values
X_train_raw_all = X_train_engineered[feature_cols].values
X_test_raw_all = X_test_engineered[feature_cols].values

n_train = len(X_train_engineered)
n_test = len(X_test_engineered)
n_splits = 5
n_repeats = 5
total_splits = n_splits * n_repeats

model_names = [
    "spline_lr",
    "rbf_svc",
    "hist_gbm",
    "shallow_lgb",
    "shallow_xgb",
    "extra_trees",
    "tabular_resnet",
]

oof_predictions = {name: np.zeros(n_train, dtype=np.float64) for name in model_names}
test_predictions = {name: np.zeros(n_test, dtype=np.float64) for name in model_names}

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train_raw_all, y_train_all)):
    X_tr_raw = X_train_raw_all[train_idx]
    y_tr = y_train_all[train_idx]
    X_va_raw = X_train_raw_all[val_idx]
    y_va = y_train_all[val_idx]
    X_te_raw = X_test_raw_all

    # Fold-isolated standard scaling for linear and neural pipelines
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_va_scaled = scaler.transform(X_va_raw)
    X_te_scaled = scaler.transform(X_te_raw)

    # 1. Spline Logistic Regression
    slr_model = get_spline_lr_model(random_state=SEED + split_idx)
    slr_model.fit(X_tr_raw, y_tr)
    oof_predictions["spline_lr"][val_idx] += slr_model.predict_proba(X_va_raw)[:, 1] / n_repeats
    test_predictions["spline_lr"] += slr_model.predict_proba(X_te_raw)[:, 1] / total_splits

    # 2. Probability-calibrated RBF SVC
    svc_model = get_svc_model(random_state=SEED + split_idx)
    svc_model.fit(X_tr_raw, y_tr)
    oof_predictions["rbf_svc"][val_idx] += svc_model.predict_proba(X_va_raw)[:, 1] / n_repeats
    test_predictions["rbf_svc"] += svc_model.predict_proba(X_te_raw)[:, 1] / total_splits

    # 3. HistGradientBoosting
    hgb_model = get_hist_gbm_model(random_state=SEED + split_idx)
    hgb_model.fit(X_tr_raw, y_tr)
    oof_predictions["hist_gbm"][val_idx] += hgb_model.predict_proba(X_va_raw)[:, 1] / n_repeats
    test_predictions["hist_gbm"] += hgb_model.predict_proba(X_te_raw)[:, 1] / total_splits

    # 4. Shallow LightGBM
    lgb_model = get_shallow_lgb_model(random_state=SEED + split_idx)
    lgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_va_raw, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_predictions["shallow_lgb"][val_idx] += lgb_model.predict_proba(X_va_raw)[:, 1] / n_repeats
    test_predictions["shallow_lgb"] += lgb_model.predict_proba(X_te_raw)[:, 1] / total_splits

    # 5. Shallow XGBoost
    xgb_model = get_shallow_xgb_model(random_state=SEED + split_idx)
    xgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_va_raw, y_va)],
        verbose=False,
    )
    oof_predictions["shallow_xgb"][val_idx] += xgb_model.predict_proba(X_va_raw)[:, 1] / n_repeats
    test_predictions["shallow_xgb"] += xgb_model.predict_proba(X_te_raw)[:, 1] / total_splits

    # 6. Extra Trees
    et_model = get_extra_trees_model(random_state=SEED + split_idx)
    et_model.fit(X_tr_raw, y_tr)
    oof_predictions["extra_trees"][val_idx] += et_model.predict_proba(X_va_raw)[:, 1] / n_repeats
    test_predictions["extra_trees"] += et_model.predict_proba(X_te_raw)[:, 1] / total_splits

    # 7. Tabular ResNet (PyTorch)
    pos_ratio = (len(y_tr) - np.sum(y_tr)) / (np.sum(y_tr) + 1e-5)
    resnet_model, criterion, optimizer, scheduler = get_torch_resnet_components(
        input_dim=len(feature_cols),
        lr=1e-3,
        weight_decay=1e-3,
        pos_weight=pos_ratio,
        device=device,
    )
    resnet_model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(50):
        resnet_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = resnet_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        resnet_model.eval()
        with torch.no_grad():
            va_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            va_logits = resnet_model(va_t)
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
            curr_auc = roc_auc_score(y_va, va_probs)
            if curr_auc > best_val_auc:
                best_val_auc = curr_auc
                best_state = {
                    k: v.cpu().clone() for k, v in resnet_model.state_dict().items()
                }

    if best_state is not None:
        resnet_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    resnet_model.eval()
    with torch.no_grad():
        va_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        te_t = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        va_preds_res = torch.sigmoid(resnet_model(va_t)).cpu().numpy()
        te_preds_res = torch.sigmoid(resnet_model(te_t)).cpu().numpy()

    oof_predictions["tabular_resnet"][val_idx] += va_preds_res / n_repeats
    test_predictions["tabular_resnet"] += te_preds_res / total_splits

# =========================================================================
# 5. Optimal Ensemble Blending & Evaluation
# =========================================================================
for name in model_names:
    model_score = roc_auc_score(y_train_all, oof_predictions[name])
    print(f"Model OOF AUC [{name}]: {model_score:.5f}")

oof_mat = np.column_stack([oof_predictions[name] for name in model_names])
test_mat = np.column_stack([test_predictions[name] for name in model_names])


def ensemble_objective(weights):
    w = np.maximum(weights, 0)
    w_sum = np.sum(w)
    w_norm = np.ones(len(w)) / len(w) if w_sum == 0 else w / w_sum
    blend = oof_mat @ w_norm
    return -roc_auc_score(y_train_all, blend)


initial_weights = np.ones(len(model_names)) / len(model_names)
opt_res = minimize(
    ensemble_objective,
    initial_weights,
    method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5},
)

opt_weights = np.maximum(opt_res.x, 0)
if np.sum(opt_weights) > 0:
    opt_weights = opt_weights / np.sum(opt_weights)
else:
    opt_weights = initial_weights

final_oof_blend = oof_mat @ opt_weights
final_test_blend = test_mat @ opt_weights

final_score = roc_auc_score(y_train_all, final_oof_blend)

# =========================================================================
# 6. Submission Generation
# =========================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=int), "Class": np.clip(final_test_blend, 0.0, 1.0)}
)

submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")