import os
import random
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# 0. Reproducibility & Environment Setup
# -------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


RANDOM_STATE = 42
N_SPLITS = 10
seed_everything(RANDOM_STATE)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# 1. Feature Engineering
# -------------------------------------------------------------------------
def create_grid_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Computes physics-informed features for the 4-node star smart grid system:

    - Node 1: Central power generator (p1 > 0)
    - Nodes 2, 3, 4: Distributed consumers (p2, p3, p4 < 0)
    - tau_i: Participant reaction delays
    - g_i: Participant price elasticity / damping constants
    """
    df = df_in.copy()

    # 1. Node-level delay and damping dynamics
    for i in [1, 2, 3, 4]:
        t_col = f"tau{i}"
        g_col = f"g{i}"
        p_col = f"p{i}"

        # Response inertia ratio and product
        df[f"tau_div_g_{i}"] = df[t_col] / (df[g_col] + 1e-6)
        df[f"g_div_tau_{i}"] = df[g_col] / (df[t_col] + 1e-6)
        df[f"tau_mul_g_{i}"] = df[t_col] * df[g_col]

        # Power-scaled dynamics
        df[f"p_tau_{i}"] = df[p_col] * df[t_col]
        df[f"p_g_{i}"] = df[p_col] * df[g_col]
        df[f"p_effective_damping_{i}"] = df[p_col] * (df[g_col] / (df[t_col] + 1e-6))

    # 2. Network-level power balance and aggregations
    df["p_sum_total"] = df["p1"] + df["p2"] + df["p3"] + df["p4"]
    df["p_consumers_sum"] = df["p2"] + df["p3"] + df["p4"]
    df["p_gen_cons_ratio"] = df["p1"] / (np.abs(df["p_consumers_sum"]) + 1e-6)

    # Consumer power dispersion
    cons_p = df[["p2", "p3", "p4"]]
    df["p_cons_mean"] = cons_p.mean(axis=1)
    df["p_cons_std"] = cons_p.std(axis=1)
    df["p_cons_max"] = cons_p.max(axis=1)
    df["p_cons_min"] = cons_p.min(axis=1)
    df["p_cons_range"] = df["p_cons_max"] - df["p_cons_min"]

    # 3. Delay interactions & system time-constants
    tau_cols = ["tau1", "tau2", "tau3", "tau4"]
    df["tau_sum"] = df[tau_cols].sum(axis=1)
    df["tau_mean"] = df[tau_cols].mean(axis=1)
    df["tau_std"] = df[tau_cols].std(axis=1)
    df["tau_max"] = df[tau_cols].max(axis=1)
    df["tau_min"] = df[tau_cols].min(axis=1)
    df["tau_range"] = df["tau_max"] - df["tau_min"]

    # Producer tau vs Consumer mean tau
    df["tau_cons_mean"] = df[["tau2", "tau3", "tau4"]].mean(axis=1)
    df["tau_gen_to_cons_ratio"] = df["tau1"] / (df["tau_cons_mean"] + 1e-6)

    # 4. Damping metrics & dissipation
    g_cols = ["g1", "g2", "g3", "g4"]
    df["g_sum"] = df[g_cols].sum(axis=1)
    df["g_mean"] = df[g_cols].mean(axis=1)
    df["g_std"] = df[g_cols].std(axis=1)
    df["g_harmonic_mean"] = 4.0 / (
        (1.0 / (df["g1"] + 1e-6))
        + (1.0 / (df["g2"] + 1e-6))
        + (1.0 / (df["g3"] + 1e-6))
        + (1.0 / (df["g4"] + 1e-6))
    )

    # Producer g vs Consumer mean g
    df["g_cons_mean"] = df[["g2", "g3", "g4"]].mean(axis=1)
    df["g_gen_to_cons_ratio"] = df["g1"] / (df["g_cons_mean"] + 1e-6)

    # 5. Composite stability index approximations
    df["delay_damping_system_ratio"] = df["tau_sum"] / (df["g_sum"] + 1e-6)
    df["kinetic_energy_proxy"] = (
        df["p1"] ** 2 * df["tau1"]
        + df["p2"] ** 2 * df["tau2"]
        + df["p3"] ** 2 * df["tau3"]
        + df["p4"] ** 2 * df["tau4"]
    )
    df["dissipation_proxy"] = (
        df["p1"] ** 2 * df["g1"]
        + df["p2"] ** 2 * df["g2"]
        + df["p3"] ** 2 * df["g3"]
        + df["p4"] ** 2 * df["g4"]
    )
    df["stability_energy_ratio"] = df["kinetic_energy_proxy"] / (
        df["dissipation_proxy"] + 1e-6
    )

    # 6. Advanced Physics & Dynamic Stability Features
    # Lyapunov dissipation exponents: sum(p_i^2 * tau_i^2 / g_i)
    for i in [1, 2, 3, 4]:
        df[f"lyapunov_term_{i}"] = (df[f"p{i}"] ** 2 * df[f"tau{i}"] ** 2) / (df[f"g{i}"] + 1e-6)
    df["lyapunov_dissipation_total"] = (
        df["lyapunov_term_1"] + df["lyapunov_term_2"] + df["lyapunov_term_3"] + df["lyapunov_term_4"]
    )

    # Generator-to-consumer momentum ratio: p1 * tau1 / sum(|p_j| * tau_j)
    cons_momentum = (
        np.abs(df["p2"]) * df["tau2"]
        + np.abs(df["p3"]) * df["tau3"]
        + np.abs(df["p4"]) * df["tau4"]
    )
    df["gen_to_cons_momentum_ratio"] = (df["p1"] * df["tau1"]) / (cons_momentum + 1e-6)

    # Consumer asymmetry deltas (|p_i - p_j|, |tau_i - tau_j|, |g_i - g_j|)
    pairs = [(2, 3), (2, 4), (3, 4)]
    for i, j in pairs:
        df[f"delta_p_{i}_{j}"] = np.abs(df[f"p{i}"] - df[f"p{j}"])
        df[f"delta_tau_{i}_{j}"] = np.abs(df[f"tau{i}"] - df[f"tau{j}"])
        df[f"delta_g_{i}_{j}"] = np.abs(df[f"g{i}"] - df[f"g{j}"])

    df["consumer_p_asymmetry_sum"] = df["delta_p_2_3"] + df["delta_p_2_4"] + df["delta_p_3_4"]
    df["consumer_tau_asymmetry_sum"] = df["delta_tau_2_3"] + df["delta_tau_2_4"] + df["delta_tau_3_4"]
    df["consumer_g_asymmetry_sum"] = df["delta_g_2_3"] + df["delta_g_2_4"] + df["delta_g_3_4"]

    # Normalized nodal impedance ratios (p_i / p1, tau_i / tau1, g_i / g1 for i in {2,3,4})
    for i in [2, 3, 4]:
        df[f"p_{i}_div_p1"] = df[f"p{i}"] / (df["p1"] + 1e-6)
        df[f"tau_{i}_div_tau1"] = df[f"tau{i}"] / (df["tau1"] + 1e-6)
        df[f"g_{i}_div_g1"] = df[f"g{i}"] / (df["g1"] + 1e-6)

    # Cross delay products
    df["tau_1_2"] = df["tau1"] * df["tau2"]
    df["tau_1_3"] = df["tau1"] * df["tau3"]
    df["tau_1_4"] = df["tau1"] * df["tau4"]

    # 7. Dimensionless grid dynamic features & star branch couplings
    p1_phase_vel = np.abs(df["p1"]) / (df["tau1"] + 1e-6)
    for j in [2, 3, 4]:
        pj_phase_vel = np.abs(df[f"p{j}"]) / (df[f"tau{j}"] + 1e-6)
        df[f"phase_vel_diff_1_{j}"] = p1_phase_vel - pj_phase_vel
        df[f"branch_impedance_ratio_1_{j}"] = (df["g1"] * df["tau1"]) / (
            df[f"g{j}"] * df[f"tau{j}"] + 1e-6
        )
        df[f"branch_p_tau_ratio_1_{j}"] = (df["p1"] / (df["tau1"] + 1e-6)) / (
            np.abs(df[f"p{j}"]) / (df[f"tau{j}"] + 1e-6) + 1e-6
        )

    # Power-weighted network delay and damping centers of mass
    total_abs_p = (
        np.abs(df["p1"])
        + np.abs(df["p2"])
        + np.abs(df["p3"])
        + np.abs(df["p4"])
        + 1e-6
    )
    df["tau_weighted_center"] = (
        np.abs(df["p1"]) * df["tau1"]
        + np.abs(df["p2"]) * df["tau2"]
        + np.abs(df["p3"]) * df["tau3"]
        + np.abs(df["p4"]) * df["tau4"]
    ) / total_abs_p
    df["g_weighted_center"] = (
        np.abs(df["p1"]) * df["g1"]
        + np.abs(df["p2"]) * df["g2"]
        + np.abs(df["p3"]) * df["g3"]
        + np.abs(df["p4"]) * df["g4"]
    ) / total_abs_p
    df["weighted_tau_div_g"] = df["tau_weighted_center"] / (
        df["g_weighted_center"] + 1e-6
    )

    # Characteristic swing frequency proxies
    for i in [1, 2, 3, 4]:
        df[f"swing_freq_proxy_{i}"] = np.sqrt(
            np.maximum(df[f"tau{i}"] * df[f"g{i}"], 0.0) + 1e-8
        )
    df["swing_freq_sum"] = (
        df["swing_freq_proxy_1"]
        + df["swing_freq_proxy_2"]
        + df["swing_freq_proxy_3"]
        + df["swing_freq_proxy_4"]
    )
    df["swing_freq_gen_to_cons_ratio"] = df["swing_freq_proxy_1"] / (
        (
            df["swing_freq_proxy_2"]
            + df["swing_freq_proxy_3"]
            + df["swing_freq_proxy_4"]
        )
        / 3.0
        + 1e-6
    )

    return df


# -------------------------------------------------------------------------
# 2. PyTorch Tabular ResNet Architecture
# -------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        clf_logits = self.clf_head(x).squeeze(-1)
        return clf_logits


def get_neural_network_setup(
    input_dim: int,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    ).to(device)
    clf_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    return model, clf_criterion, optimizer


# -------------------------------------------------------------------------
# 3. Estimator Configuration Factories
# -------------------------------------------------------------------------
def get_lgb_params(random_state: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(random_state: int = 42) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 2,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }


def get_catboost_params(random_state: int = 42) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": 50,
    }


def get_lgb_params2(random_state: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_samples": 30,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


# -------------------------------------------------------------------------
# 4. Data Loading, Processing & Target Encoding
# -------------------------------------------------------------------------
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "stabf"
if target_col in train_df.columns:
    if (
        train_df[target_col].dtype == object
        or isinstance(train_df[target_col].iloc[0], str)
        or train_df[target_col].dtype.name == "category"
    ):
        unique_vals = [str(u).lower() for u in train_df[target_col].unique()]
        if "stable" in unique_vals:
            train_df[target_col] = (
                train_df[target_col]
                .astype(str)
                .str.lower()
                .map({"stable": 1, "unstable": 0})
            )
        else:
            from sklearn.preprocessing import LabelEncoder

            train_df[target_col] = LabelEncoder().fit_transform(train_df[target_col])
    else:
        train_df[target_col] = train_df[target_col].astype(int)

train_featured = create_grid_features(train_df)
test_featured = create_grid_features(test_df)

feature_columns = [
    col
    for col in train_featured.columns
    if col not in ["id", "stab", "stabf", "__index_level_0__", "fold"]
]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_featured["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(
    skf.split(train_featured, train_featured[target_col])
):
    train_featured.loc[val_idx, "fold"] = fold_idx

# -------------------------------------------------------------------------
# 5. Model Training & Cross-Validation
# -------------------------------------------------------------------------
n_train = len(train_featured)
n_test = len(test_featured)

oof_lgb_clf1 = np.zeros(n_train)
oof_lgb_clf2 = np.zeros(n_train)
oof_xgb_clf = np.zeros(n_train)
oof_cb_clf = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_preds_lgb_clf1 = np.zeros(n_test)
test_preds_lgb_clf2 = np.zeros(n_test)
test_preds_xgb_clf = np.zeros(n_test)
test_preds_cb_clf = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

y_true = train_featured[target_col].values
X_test_raw = test_featured[feature_columns].values

for fold in range(N_SPLITS):
    train_idx = (train_featured["fold"] != fold).values
    val_idx = (train_featured["fold"] == fold).values

    X_train_f = train_featured.loc[train_idx, feature_columns].values
    y_train_f = train_featured.loc[train_idx, target_col].values

    X_val_f = train_featured.loc[val_idx, feature_columns].values
    y_val_f = train_featured.loc[val_idx, target_col].values

    # Strict fold-level standardization for PyTorch TabularResNet
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)
    X_test_scaled = scaler.transform(X_test_raw)

    # 1. LightGBM Classifier 1
    lgb_clf1 = lgb.LGBMClassifier(**get_lgb_params(RANDOM_STATE + fold))
    lgb_clf1.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb_clf1[val_idx] = lgb_clf1.predict_proba(X_val_f)[:, 1]
    test_preds_lgb_clf1 += lgb_clf1.predict_proba(X_test_raw)[:, 1] / N_SPLITS

    # 2. LightGBM Classifier 2 (shallow & regularized)
    lgb_clf2 = lgb.LGBMClassifier(**get_lgb_params2(RANDOM_STATE + fold))
    lgb_clf2.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb_clf2[val_idx] = lgb_clf2.predict_proba(X_val_f)[:, 1]
    test_preds_lgb_clf2 += lgb_clf2.predict_proba(X_test_raw)[:, 1] / N_SPLITS

    # 3. XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(**get_xgb_params(RANDOM_STATE + fold))
    xgb_clf.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    oof_xgb_clf[val_idx] = xgb_clf.predict_proba(X_val_f)[:, 1]
    test_preds_xgb_clf += xgb_clf.predict_proba(X_test_raw)[:, 1] / N_SPLITS

    # 4. CatBoost Classifier
    cb_clf = cb.CatBoostClassifier(**get_catboost_params(RANDOM_STATE + fold))
    cb_clf.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)
    oof_cb_clf[val_idx] = cb_clf.predict_proba(X_val_f)[:, 1]
    test_preds_cb_clf += cb_clf.predict_proba(X_test_raw)[:, 1] / N_SPLITS

    # 5. PyTorch TabularResNet Classifier
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_f, dtype=torch.float32),
    )
    val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )

    nn_model, clf_criterion, optimizer = get_neural_network_setup(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.15,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device=device,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_auc = 0.0
    best_nn_val_preds = None
    best_state_dict = None

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            clf_logits = nn_model(batch_x)
            loss = clf_criterion(clf_logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(val_tensor_x)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            epoch_auc = roc_auc_score(y_val_f, val_probs)

            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_nn_val_preds = val_probs
                best_state_dict = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

    oof_nn[val_idx] = best_nn_val_preds

    nn_model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    nn_model.eval()
    with torch.no_grad():
        test_logits = nn_model(test_tensor_x)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds_nn += test_probs / N_SPLITS

# -------------------------------------------------------------------------
# 6. Ensemble Optimization & Validation
# -------------------------------------------------------------------------
def rank_normalize(arr: np.ndarray) -> np.ndarray:
    return (rankdata(arr) - 0.5) / len(arr)


oof_matrix = np.column_stack([
    rank_normalize(oof_lgb_clf1),
    rank_normalize(oof_lgb_clf2),
    rank_normalize(oof_xgb_clf),
    rank_normalize(oof_cb_clf),
    rank_normalize(oof_nn),
])

test_matrix = np.column_stack([
    rank_normalize(test_preds_lgb_clf1),
    rank_normalize(test_preds_lgb_clf2),
    rank_normalize(test_preds_xgb_clf),
    rank_normalize(test_preds_cb_clf),
    rank_normalize(test_preds_nn),
])


def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = np.dot(oof_matrix, w)
    return -roc_auc_score(y_true, blend_oof)


init_weights = [1.0 / 5.0] * 5
bounds = [(0.0, 1.0) for _ in range(5)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

final_oof_preds = np.dot(oof_matrix, best_weights)
final_validation_score = float(roc_auc_score(y_true, final_oof_preds))

# -------------------------------------------------------------------------
# 7. Final Submission Generation
# -------------------------------------------------------------------------
final_test_preds = np.dot(test_matrix, best_weights)
final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

submission_id = (
    test_featured["id"].values
    if "id" in test_featured.columns
    else np.arange(len(test_featured))
)
submission_df = pd.DataFrame({"id": submission_id, "stabf": final_test_preds})

submission_file_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print(f"Final Validation Score: {final_validation_score}")
