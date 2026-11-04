import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


# ---------------------------------------------------------------------------
# Reproducibility Setup
# ---------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# ---------------------------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target_col = "defects"
if target_col not in train_df.columns:
    if "problems" in train_df.columns:
        target_col = "problems"
    else:
        target_col = [
            c
            for c in train_df.columns
            if c not in test_df.columns or c in ["defects", "problems", "target"]
        ][0]

# Standardize target values to binary 0/1 integers
if train_df[target_col].dtype == bool:
    train_df[target_col] = train_df[target_col].astype(int)
elif train_df[target_col].dtype == object or isinstance(
    train_df[target_col].iloc[0], str
):
    train_df[target_col] = (
        train_df[target_col]
        .map(
            {
                "true": 1,
                "True": 1,
                "t": 1,
                "T": 1,
                "1": 1,
                1: 1,
                "false": 0,
                "False": 0,
                "f": 0,
                "F": 0,
                "0": 0,
                0: 0,
            }
        )
        .fillna(0)
        .astype(int)
    )
else:
    train_df[target_col] = train_df[target_col].astype(int)


# ---------------------------------------------------------------------------
# 2. Domain-Specific Feature Engineering (McCabe & Halstead Metrics)
# ---------------------------------------------------------------------------
def engineer_software_metrics(df_input):
    df = df_input.copy()
    eps = 1e-6

    # Standardize expected column names
    col_map = {
        "v(g)": "vg",
        "ev(g)": "evg",
        "iv(g)": "ivg",
        "v_g": "vg",
        "ev_g": "evg",
        "iv_g": "ivg",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    loc = df["loc"] if "loc" in df.columns else df.get("lOCode", 1.0)
    vg = df["vg"] if "vg" in df.columns else 1.0
    evg = df["evg"] if "evg" in df.columns else 0.0
    ivg = df["ivg"] if "ivg" in df.columns else 0.0

    l_code = df["lOCode"] if "lOCode" in df.columns else loc
    l_comment = df["lOComment"] if "lOComment" in df.columns else 0.0
    l_blank = df["lOBlank"] if "lOBlank" in df.columns else 0.0

    uniq_op = df["uniq_Op"] if "uniq_Op" in df.columns else 1.0
    uniq_opnd = df["uniq_Opnd"] if "uniq_Opnd" in df.columns else 1.0
    tot_op = df["total_Op"] if "total_Op" in df.columns else 1.0
    tot_opnd = df["total_Opnd"] if "total_Opnd" in df.columns else 1.0

    h_v = df["v"] if "v" in df.columns else 0.0
    h_d = df["d"] if "d" in df.columns else 0.0
    h_e = df["e"] if "e" in df.columns else 0.0
    h_b = df["b"] if "b" in df.columns else 0.0
    h_t = df["t"] if "t" in df.columns else 0.0
    h_n = df["n"] if "n" in df.columns else (tot_op + tot_opnd)

    # 2.1 McCabe Complexity & Code Density Features
    df["cyclomatic_density"] = vg / (loc + eps)
    df["essential_complexity_ratio"] = evg / (vg + eps)
    df["design_complexity_ratio"] = ivg / (vg + eps)
    df["structuredness_index"] = (vg - evg) / (vg + eps)

    # 2.2 Code Composition & Documentation Ratios
    total_lines = loc + l_comment + l_blank + eps
    df["comment_ratio"] = l_comment / total_lines
    df["blank_ratio"] = l_blank / total_lines
    df["code_ratio"] = l_code / total_lines
    df["comment_to_code_ratio"] = l_comment / (l_code + eps)

    # 2.3 Halstead Vocabulary and Length Consistency Features
    computed_vocab = uniq_op + uniq_opnd
    computed_length = tot_op + tot_opnd
    df["computed_vocab_mu"] = computed_vocab
    df["computed_length_N"] = computed_length
    df["length_discrepancy"] = h_n - computed_length

    # Halstead theoretical metric discrepancy formulas
    n1_safe = np.maximum(uniq_op.astype(float), 1.0)
    n2_safe = np.maximum(uniq_opnd.astype(float), 1.0)
    n_hat = n1_safe * np.log2(n1_safe) + n2_safe * np.log2(n2_safe)
    df["halstead_n_hat"] = n_hat
    df["halstead_length_rel_error"] = np.abs(h_n - n_hat) / (h_n + eps)

    theoretical_volume = computed_length * np.log2(
        np.maximum(computed_vocab, 1.0) + eps
    )
    df["theoretical_volume"] = theoretical_volume
    df["volume_discrepancy"] = h_v - theoretical_volume
    df["volume_rel_error"] = np.abs(h_v - theoretical_volume) / (h_v + eps)

    theoretical_diff = (uniq_op / 2.0) * (tot_opnd / (uniq_opnd + eps))
    df["difficulty_discrepancy"] = h_d - theoretical_diff
    df["effort_per_volume"] = h_e / (h_v + eps)
    df["time_per_loc"] = h_t / (loc + eps)
    df["bugs_per_loc"] = h_b / (loc + eps)
    df["halstead_est_bugs_B"] = h_v / 3000.0
    df["halstead_intelligence_I"] = h_v / (h_d + eps)

    # 2.4 Operator / Operand Ratios
    df["operator_to_operand_ratio"] = tot_op / (tot_opnd + eps)
    df["uniq_operator_ratio"] = uniq_op / (tot_op + eps)
    df["uniq_operand_ratio"] = uniq_opnd / (tot_opnd + eps)
    df["operator_diversity"] = uniq_op / (computed_vocab + eps)

    # 2.5 Interaction, Risk Indices & Industry Thresholds
    df["mccabe_halstead_risk"] = df["cyclomatic_density"] * np.log1p(
        np.maximum(h_e, 0.0)
    )
    df["complexity_density_product"] = (vg * ivg) / (loc + eps)
    df["mccabe_risk_moderate"] = (vg > 10.0).astype(float)
    df["mccabe_risk_high"] = (vg > 20.0).astype(float)
    df["essential_risk_high"] = (evg > 4.0).astype(float)
    df["design_risk_high"] = (ivg > 7.0).astype(float)

    # 2.6 Log1p transforms for heavy-tailed metrics
    skewed_cols = [
        "loc",
        "v",
        "d",
        "e",
        "b",
        "t",
        "n",
        "lOCode",
        "lOComment",
        "lOBlank",
        "total_Op",
        "total_Opnd",
        "uniq_Op",
        "uniq_Opnd",
    ]
    for c in skewed_cols:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(np.maximum(df[c].astype(float), 0.0))

    return df


train_features_df = engineer_software_metrics(train_df.drop(columns=[target_col]))
test_features_df = engineer_software_metrics(test_df)

feature_cols = [c for c in train_features_df.columns if c in test_features_df.columns]
train_features_df = train_features_df[feature_cols]
test_features_df = test_features_df[feature_cols]

train_features_df = train_features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_features_df = test_features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

y_all = train_df[target_col].values
X_all = train_features_df.values
X_test_raw = test_features_df.values
n_test = len(test_df)


# ---------------------------------------------------------------------------
# 3. Model Architecture Definitions
# ---------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features * 2)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features * 2)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h_val, h_gate = self.linear1(h).chunk(2, dim=-1)
        h = h_val * torch.sigmoid(h_gate)
        h = self.dropout1(self.act1(h))

        h2 = self.norm2(h)
        h2_val, h2_gate = self.linear2(h2).chunk(2, dim=-1)
        h2 = h2_val * torch.sigmoid(h2_gate)
        h2 = self.dropout2(self.act2(h2))

        return h2 + residual


class GatedTabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 128, 64, 32],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                GatedResidualBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 16),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        feat = self.res_blocks(feat)
        logits = self.head(feat)
        return logits.squeeze(-1)


def build_lgb_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )


def build_extra_trees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def build_cb_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1000,
        depth=4,
        l2_leaf_reg=5.0,
        learning_rate=0.03,
        eval_metric="AUC",
        random_seed=random_state,
        early_stopping_rounds=50,
        verbose=False,
    )


def build_rf_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svc_model(random_state: int = 42) -> SVC:
    return SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# 4. 10-Fold Cross-Validation Training & Inference
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = {
    "lgb": np.zeros(len(train_df)),
    "xgb": np.zeros(len(train_df)),
    "cb": np.zeros(len(train_df)),
    "et": np.zeros(len(train_df)),
    "rf": np.zeros(len(train_df)),
    "svc": np.zeros(len(train_df)),
    "nn": np.zeros(len(train_df)),
}

test_preds = {
    "lgb": np.zeros(n_test),
    "xgb": np.zeros(n_test),
    "cb": np.zeros(n_test),
    "et": np.zeros(n_test),
    "rf": np.zeros(n_test),
    "svc": np.zeros(n_test),
    "nn": np.zeros(n_test),
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr_raw, y_tr = X_all[train_idx], y_all[train_idx]
    X_va_raw, y_va = X_all[val_idx], y_all[val_idx]
    X_te_raw = X_test_raw.copy()

    # PCA feature extraction on quantile-scaled representations fitted on training fold
    qt_pca = QuantileTransformer(
        n_quantiles=min(len(train_idx), 1000),
        output_distribution="normal",
        random_state=42 + fold,
    )
    X_tr_qt = qt_pca.fit_transform(X_tr_raw)
    X_va_qt = qt_pca.transform(X_va_raw)
    X_te_qt = qt_pca.transform(X_te_raw)

    pca = PCA(n_components=3, random_state=42 + fold)
    X_tr_pca = pca.fit_transform(X_tr_qt)
    X_va_pca = pca.transform(X_va_qt)
    X_te_pca = pca.transform(X_te_qt)

    X_tr = np.hstack([X_tr_raw, X_tr_pca])
    X_va = np.hstack([X_va_raw, X_va_pca])
    X_te = np.hstack([X_te_raw, X_te_pca])

    # --- 4.1 LightGBM ---
    lgb_clf = build_lgb_model(random_state=42 + fold)
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds["lgb"][val_idx] = lgb_clf.predict_proba(X_va)[:, 1]
    test_preds["lgb"] += lgb_clf.predict_proba(X_te)[:, 1] / n_folds

    # --- 4.2 XGBoost ---
    xgb_clf = build_xgb_model(random_state=42 + fold)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["xgb"][val_idx] = xgb_clf.predict_proba(X_va)[:, 1]
    test_preds["xgb"] += xgb_clf.predict_proba(X_te)[:, 1] / n_folds

    # --- 4.3 CatBoost ---
    cb_clf = build_cb_model(random_state=42 + fold)
    cb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["cb"][val_idx] = cb_clf.predict_proba(X_va)[:, 1]
    test_preds["cb"] += cb_clf.predict_proba(X_te)[:, 1] / n_folds

    # --- 4.4 ExtraTrees ---
    et_clf = build_extra_trees_model(random_state=42 + fold)
    et_clf.fit(X_tr, y_tr)
    oof_preds["et"][val_idx] = et_clf.predict_proba(X_va)[:, 1]
    test_preds["et"] += et_clf.predict_proba(X_te)[:, 1] / n_folds

    # --- 4.5 Random Forest ---
    rf_clf = build_rf_model(random_state=42 + fold)
    rf_clf.fit(X_tr, y_tr)
    oof_preds["rf"][val_idx] = rf_clf.predict_proba(X_va)[:, 1]
    test_preds["rf"] += rf_clf.predict_proba(X_te)[:, 1] / n_folds

    # --- 4.6 Support Vector Classifier (RBF Kernel on Quantile Data) ---
    svc_clf = build_svc_model(random_state=42 + fold)
    svc_clf.fit(X_tr_qt, y_tr)
    oof_preds["svc"][val_idx] = svc_clf.predict_proba(X_va_qt)[:, 1]
    test_preds["svc"] += svc_clf.predict_proba(X_te_qt)[:, 1] / n_folds

    # --- 4.7 GatedTabularResNet (Neural Network) ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_te)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    nn_model = GatedTabularResNet(
        input_dim=X_tr.shape[1], hidden_dims=[128, 128, 64, 32], dropout_rate=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(40):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        va_logits_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                va_logits_list.append(nn_model(bx).cpu().numpy())

        va_nn_probs = 1.0 / (1.0 + np.exp(-np.concatenate(va_logits_list)))
        epoch_auc = roc_auc_score(y_va, va_nn_probs)

        if epoch_auc > best_val_auc:
            best_val_auc = epoch_auc
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    if best_weights is not None:
        nn_model.load_state_dict(best_weights)
    nn_model.to(device)
    nn_model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        oof_preds["nn"][val_idx] = torch.sigmoid(nn_model(va_tensor)).cpu().numpy()

        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds["nn"] += torch.sigmoid(nn_model(test_tensor)).cpu().numpy() / n_folds


# ---------------------------------------------------------------------------
# 5. Rank-Based Ensemble Blending Optimization & Final Metric
# ---------------------------------------------------------------------------
model_names = ["lgb", "xgb", "cb", "et", "rf", "svc", "nn"]

oof_ranks = np.column_stack(
    [rankdata(oof_preds[m]) / len(train_df) for m in model_names]
)
test_ranks = np.column_stack(
    [rankdata(test_preds[m]) / n_test for m in model_names]
)


def loss_func(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    sum_w = np.sum(w)
    if sum_w == 0:
        return 0.0
    w = w / sum_w
    pred = np.dot(oof_ranks, w)
    return -roc_auc_score(y_all, pred)


init_weights = np.ones(len(model_names)) / len(model_names)

opt_res_nm = minimize(
    loss_func,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 1000, "disp": False},
)

opt_res_pw = minimize(
    loss_func,
    init_weights,
    method="Powell",
    options={"maxiter": 1000, "disp": False},
)

best_res = opt_res_pw if opt_res_pw.fun < opt_res_nm.fun else opt_res_nm

optimal_weights = np.maximum(best_res.x, 0)
if np.sum(optimal_weights) > 0:
    optimal_weights = optimal_weights / np.sum(optimal_weights)
else:
    optimal_weights = init_weights

for name, w in zip(model_names, optimal_weights):
    print(f"Model {name} optimal rank blend weight: {w:.4f}")

oof_ensemble = np.dot(oof_ranks, optimal_weights)
final_score = roc_auc_score(y_all, oof_ensemble)

final_test_preds = np.dot(test_ranks, optimal_weights)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "defects": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")