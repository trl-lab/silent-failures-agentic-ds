import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. Data Loading & Schema Normalization
# =========================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


def normalize_col_name(c: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", c).lower()


target_col_candidates = [
    "contraceptivemethodused",
    "contraceptivemethod",
    "contraceptive",
    "target",
    "class",
]

target_col = None
for col in train_df.columns:
    if normalize_col_name(col) in target_col_candidates:
        target_col = col
        break

if target_col is None:
    target_col = train_df.columns[-1]

column_mapping = {}
for col in train_df.columns:
    norm = normalize_col_name(col)
    if "wife" in norm and "age" in norm:
        column_mapping["wife_age"] = col
    elif "wife" in norm and "edu" in norm:
        column_mapping["wife_edu"] = col
    elif "husband" in norm and "edu" in norm:
        column_mapping["husband_edu"] = col
    elif "child" in norm or "born" in norm or "parity" in norm:
        column_mapping["num_children"] = col
    elif "relig" in norm:
        column_mapping["wife_rel"] = col
    elif "work" in norm:
        column_mapping["wife_work"] = col
    elif "husband" in norm and "occ" in norm:
        column_mapping["husband_occ"] = col
    elif "living" in norm or "sol" in norm or "standard" in norm:
        column_mapping["sol_index"] = col
    elif "media" in norm:
        column_mapping["media_exp"] = col

rename_dict = {orig: std for std, orig in column_mapping.items()}
train_std = train_df.rename(columns=rename_dict).copy()
test_std = test_df.rename(columns=rename_dict).copy()

y_train_raw = train_std[target_col].astype(int).values
min_y = y_train_raw.min()
y_train = y_train_raw - min_y


# =========================================================================
# 2. Domain-Specific Feature Engineering
# =========================================================================
def create_domain_features(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    if ref_df is None:
        ref_df = df

    feat = pd.DataFrame(index=df.index)

    # Core Base Features
    feat["wife_age"] = df["wife_age"].astype(float)
    feat["wife_edu"] = df["wife_edu"].astype(float)
    feat["husband_edu"] = df["husband_edu"].astype(float)
    feat["num_children"] = df["num_children"].astype(float)
    feat["wife_rel"] = df["wife_rel"].astype(float)
    feat["wife_work"] = df["wife_work"].astype(float)
    feat["husband_occ"] = df["husband_occ"].astype(float)
    feat["sol_index"] = df["sol_index"].astype(float)
    feat["media_exp"] = df["media_exp"].astype(float)

    # Demographic Parity & Age Dynamics
    reproductive_span = np.maximum(feat["wife_age"] - 14.0, 1.0)
    feat["reproductive_density"] = feat["num_children"] / reproductive_span
    feat["est_age_at_first_birth"] = np.where(
        feat["num_children"] > 0,
        np.maximum(feat["wife_age"] - (2.0 * feat["num_children"]), 14.0),
        feat["wife_age"],
    )
    feat["age_per_child"] = feat["wife_age"] / (feat["num_children"] + 1.0)
    feat["children_per_decade"] = (feat["num_children"] / feat["wife_age"]) * 10.0
    feat["childless_indicator"] = (feat["num_children"] == 0).astype(float)
    feat["single_child"] = (feat["num_children"] == 1).astype(float)
    feat["small_family"] = ((feat["num_children"] >= 1) & (feat["num_children"] <= 2)).astype(float)
    feat["medium_family"] = ((feat["num_children"] >= 3) & (feat["num_children"] <= 4)).astype(float)
    feat["large_family"] = (feat["num_children"] >= 5).astype(float)
    feat["very_large_family"] = (feat["num_children"] >= 7).astype(float)

    # Life-cycle Stages
    feat["young_low_parity"] = ((feat["wife_age"] < 25) & (feat["num_children"] <= 1)).astype(float)
    feat["completed_family_stage"] = ((feat["wife_age"] >= 30) & (feat["num_children"] >= 3)).astype(float)
    feat["late_reproductive_age"] = (feat["wife_age"] >= 35).astype(float)
    feat["early_reproductive_age"] = (feat["wife_age"] <= 24).astype(float)
    feat["mid_reproductive_age"] = ((feat["wife_age"] >= 25) & (feat["wife_age"] <= 34)).astype(float)

    # Household Dynamics & Educational Disparities
    feat["edu_gap"] = feat["husband_edu"] - feat["wife_edu"]
    feat["edu_prod"] = feat["husband_edu"] * feat["wife_edu"]
    feat["edu_ratio"] = (feat["husband_edu"] + 1.0) / (feat["wife_edu"] + 1.0)
    feat["wife_edu_higher"] = (feat["wife_edu"] > feat["husband_edu"]).astype(float)
    feat["husband_edu_higher"] = (feat["husband_edu"] > feat["wife_edu"]).astype(float)
    feat["edu_equal"] = (feat["wife_edu"] == feat["husband_edu"]).astype(float)
    feat["both_high_edu"] = ((feat["wife_edu"] == 4) & (feat["husband_edu"] == 4)).astype(float)
    feat["both_low_edu"] = ((feat["wife_edu"] == 1) & (feat["husband_edu"] == 1)).astype(float)
    feat["wife_high_edu"] = (feat["wife_edu"] >= 3).astype(float)
    feat["total_household_edu"] = feat["wife_edu"] + feat["husband_edu"]
    feat["socio_economic_composite"] = feat["total_household_edu"] + feat["sol_index"]

    # Modernization Composite
    feat["modernization_score"] = (
        (feat["wife_work"] == 0).astype(float)
        + (feat["media_exp"] == 0).astype(float)
        + (feat["sol_index"] >= 3).astype(float)
        + (feat["wife_edu"] >= 3).astype(float)
    )
    feat["tradition_score"] = (
        (feat["wife_rel"] == 1).astype(float)
        + (feat["media_exp"] == 1).astype(float)
        + (feat["sol_index"] <= 2).astype(float)
        + (feat["wife_edu"] <= 2).astype(float)
    )
    feat["sol_x_wife_edu"] = feat["sol_index"] * feat["wife_edu"]
    feat["sol_x_husband_edu"] = feat["sol_index"] * feat["husband_edu"]
    feat["sol_x_num_children"] = feat["sol_index"] * feat["num_children"]
    feat["rel_x_wife_edu"] = feat["wife_rel"] * feat["wife_edu"]

    # Categorical Interaction Codes
    feat["rel_by_sol"] = feat["wife_rel"] * 10 + feat["sol_index"]
    feat["rel_by_wife_edu"] = feat["wife_rel"] * 10 + feat["wife_edu"]
    feat["occ_by_sol"] = feat["husband_occ"] * 10 + feat["sol_index"]
    feat["wife_edu_by_husband_edu"] = feat["wife_edu"] * 10 + feat["husband_edu"]
    feat["media_by_wife_edu"] = feat["media_exp"] * 10 + feat["wife_edu"]
    feat["media_by_sol"] = feat["media_exp"] * 10 + feat["sol_index"]
    feat["occ_by_wife_edu"] = feat["husband_occ"] * 10 + feat["wife_edu"]

    # Non-linear & Polynomial Transformations
    feat["log_num_children"] = np.log1p(feat["num_children"])
    feat["age_squared"] = (feat["wife_age"] / 10.0) ** 2
    feat["parity_squared"] = (feat["num_children"]) ** 2
    feat["children_x_age"] = feat["num_children"] * feat["wife_age"]

    # Peer-Group Demographic Comparative Norms
    cohort_definitions = [
        (["wife_edu", "sol_index"], "edu_sol"),
        (["wife_edu", "husband_occ"], "edu_occ"),
        (["sol_index", "husband_occ"], "sol_occ"),
        (["wife_edu", "husband_edu"], "edu_couple"),
        (["sol_index"], "sol"),
        (["wife_edu"], "edu"),
    ]

    ref_calc = ref_df.copy()
    ref_calc["num_children"] = ref_calc["num_children"].astype(float)
    ref_calc["wife_age"] = ref_calc["wife_age"].astype(float)

    for grp_cols, name in cohort_definitions:
        grp_child_mean = ref_calc.groupby(grp_cols)["num_children"].mean().to_dict()
        grp_child_std = ref_calc.groupby(grp_cols)["num_children"].std().fillna(1.0).to_dict()
        grp_age_mean = ref_calc.groupby(grp_cols)["wife_age"].mean().to_dict()

        if len(grp_cols) == 1:
            grp_key = list(df[grp_cols[0]])
        else:
            grp_key = list(zip(*[df[c] for c in grp_cols]))

        mean_child_s = pd.Series(grp_key, index=df.index).map(grp_child_mean).astype(float).fillna(float(feat["num_children"].mean()))
        std_child_s = pd.Series(grp_key, index=df.index).map(grp_child_std).astype(float).fillna(1.0)
        mean_age_s = pd.Series(grp_key, index=df.index).map(grp_age_mean).astype(float).fillna(float(feat["wife_age"].mean()))

        feat[f"diff_child_vs_{name}_mean"] = feat["num_children"] - mean_child_s
        feat[f"zscore_child_vs_{name}"] = (feat["num_children"] - mean_child_s) / (std_child_s + 1e-3)
        feat[f"ratio_child_vs_{name}_mean"] = (feat["num_children"] + 1.0) / (mean_child_s + 1.0)
        feat[f"diff_age_vs_{name}_mean"] = feat["wife_age"] - mean_age_s
        feat[f"ratio_age_vs_{name}_mean"] = (feat["wife_age"] + 1.0) / (mean_age_s + 1.0)

    return feat


def compute_multiclass_target_encoding(
    tr_df: pd.DataFrame,
    val_df: pd.DataFrame,
    te_df: pd.DataFrame,
    cat_cols: list,
    y_tr: np.ndarray,
    num_classes: int = 3,
    alpha: float = 10.0,
):
    priors = np.array([(y_tr == c).mean() for c in range(num_classes)], dtype=np.float32)
    tr_encoded, val_encoded, te_encoded = [], [], []

    for cat in cat_cols:
        tr_cat = tr_df[cat].values
        val_cat = val_df[cat].values
        te_cat = te_df[cat].values

        tr_enc = np.zeros((len(tr_df), num_classes), dtype=np.float32)
        val_enc = np.zeros((len(val_df), num_classes), dtype=np.float32)
        te_enc = np.zeros((len(te_df), num_classes), dtype=np.float32)

        cat_stats = {}
        for val in np.unique(tr_cat):
            mask = tr_cat == val
            cnt = np.sum(mask)
            class_cnts = np.array([np.sum(mask & (y_tr == c)) for c in range(num_classes)], dtype=np.float32)
            smoothed = (class_cnts + alpha * priors) / (cnt + alpha)
            cat_stats[val] = smoothed

        for i, val in enumerate(tr_cat):
            tr_enc[i] = cat_stats.get(val, priors)
        for i, val in enumerate(val_cat):
            val_enc[i] = cat_stats.get(val, priors)
        for i, val in enumerate(te_cat):
            te_enc[i] = cat_stats.get(val, priors)

        tr_encoded.append(tr_enc)
        val_encoded.append(val_enc)
        te_encoded.append(te_enc)

    return np.hstack(tr_encoded), np.hstack(val_encoded), np.hstack(te_encoded)


# Create base domain features without cross-split aggregate leakage
X_train_base = create_domain_features(train_std, train_std)
X_test_base = create_domain_features(test_std, train_std)

# Compute full domain dataset
train_processed = X_train_base.copy()
train_processed["target"] = y_train
train_processed["target_raw"] = y_train_raw
test_processed = X_test_base.copy()


# =========================================================================
# 3. Model Architectures & Builders
# =========================================================================
class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_preds = self.log_softmax(logits)
        loss = -log_preds.sum(dim=-1).mean() * self.smoothing / num_classes
        nll_loss = nn.functional.nll_loss(log_preds, targets, reduction="mean") * (
            1.0 - self.smoothing
        )
        return loss + nll_loss


def build_nn_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    steps_per_epoch: int = 1,
):
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=lr * 0.01
    )
    return optimizer, scheduler


def create_lgbm_model(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def create_xgb_model(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )


def create_catboost_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=600,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=4.0,
        random_seed=random_state,
        verbose=0,
        thread_count=-1,
    )


def create_extratrees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def create_rf_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def create_logreg_model(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=random_state,
    )


# =========================================================================
# 4. Stratified Cross-Validation Training Loop
# =========================================================================
num_train = len(train_processed)
num_test = len(test_processed)
num_classes = 3

y_tr = train_processed["target"].values.astype(int)
y_tr_raw = train_processed["target_raw"].values.astype(int)

model_names = ["lgb", "xgb", "cat", "et", "rf", "lr", "nn"]

oof_preds = {m: np.zeros((num_train, num_classes), dtype=np.float64) for m in model_names}
test_preds = {m: np.zeros((num_test, num_classes), dtype=np.float64) for m in model_names}

n_splits = 5
n_repeats = 2
total_folds = n_splits * n_repeats
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

target_enc_cats = ["wife_edu", "husband_edu", "husband_occ", "sol_index", "rel_by_sol", "occ_by_sol"]

for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(train_std, y_tr)):
    tr_raw_df, va_raw_df = train_std.iloc[tr_idx], train_std.iloc[va_idx]
    tr_df = create_domain_features(tr_raw_df, tr_raw_df)
    va_df = create_domain_features(va_raw_df, tr_raw_df)
    te_df = create_domain_features(test_std, tr_raw_df)
    y_tr_fold, y_va_fold = y_tr[tr_idx], y_tr[va_idx]

    # Multiclass out-of-fold target encoding
    tr_enc, va_enc, te_enc = compute_multiclass_target_encoding(
        tr_df, va_df, te_df, target_enc_cats, y_tr_fold, num_classes=num_classes, alpha=10.0
    )

    X_tr_fold = np.hstack([tr_df.values.astype(np.float32), tr_enc])
    X_va_fold = np.hstack([va_df.values.astype(np.float32), va_enc])
    X_te_fold = np.hstack([te_df.values.astype(np.float32), te_enc])

    # Scaled inputs for linear model and NN
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_fold).astype(np.float32)
    X_va_scaled = scaler.transform(X_va_fold).astype(np.float32)
    X_te_scaled = scaler.transform(X_te_fold).astype(np.float32)

    # --- 1. LightGBM ---
    lgb_model = create_lgbm_model(random_state=SEED + fold_idx)
    lgb_callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    lgb_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=[(X_va_fold, y_va_fold)],
        callbacks=lgb_callbacks,
    )
    oof_preds["lgb"][va_idx] += lgb_model.predict_proba(X_va_fold) / n_repeats
    test_preds["lgb"] += lgb_model.predict_proba(X_te_fold) / total_folds

    # --- 2. XGBoost ---
    xgb_model = create_xgb_model(random_state=SEED + fold_idx)
    xgb_model.fit(
        X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)], verbose=False
    )
    oof_preds["xgb"][va_idx] += xgb_model.predict_proba(X_va_fold) / n_repeats
    test_preds["xgb"] += xgb_model.predict_proba(X_te_fold) / total_folds

    # --- 3. CatBoost ---
    cat_model = create_catboost_model(random_state=SEED + fold_idx)
    cat_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=(X_va_fold, y_va_fold),
        early_stopping_rounds=50,
        verbose=0,
    )
    oof_preds["cat"][va_idx] += cat_model.predict_proba(X_va_fold) / n_repeats
    test_preds["cat"] += cat_model.predict_proba(X_te_fold) / total_folds

    # --- 4. ExtraTrees ---
    et_model = create_extratrees_model(random_state=SEED + fold_idx)
    et_model.fit(X_tr_fold, y_tr_fold)
    oof_preds["et"][va_idx] += et_model.predict_proba(X_va_fold) / n_repeats
    test_preds["et"] += et_model.predict_proba(X_te_fold) / total_folds

    # --- 5. Random Forest ---
    rf_model = create_rf_model(random_state=SEED + fold_idx)
    rf_model.fit(X_tr_fold, y_tr_fold)
    oof_preds["rf"][va_idx] += rf_model.predict_proba(X_va_fold) / n_repeats
    test_preds["rf"] += rf_model.predict_proba(X_te_fold) / total_folds

    # --- 6. Regularized Logistic Regression ---
    lr_model = create_logreg_model(random_state=SEED + fold_idx)
    lr_model.fit(X_tr_scaled, y_tr_fold)
    oof_preds["lr"][va_idx] += lr_model.predict_proba(X_va_scaled) / n_repeats
    test_preds["lr"] += lr_model.predict_proba(X_te_scaled) / total_folds

    # --- 7. Tabular ResNet ---
    nn_model = TabularResNet(
        input_dim=X_tr_scaled.shape[1],
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.2,
    ).to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    epochs = 40
    batch_size = 32

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_fold, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va_fold, dtype=torch.long),
    )
    test_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer, scheduler = build_nn_optimizer_and_scheduler(
        nn_model,
        lr=2e-3,
        weight_decay=1e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_loss = float("inf")
    best_val_probs = None
    best_test_probs = None

    for epoch in range(epochs):
        nn_model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = nn_model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        nn_model.eval()
        val_loss_total = 0.0
        val_logits_list = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                l_v = nn_model(x_v)
                val_loss_total += nn.functional.cross_entropy(
                    l_v, y_v, reduction="sum"
                ).item()
                val_logits_list.append(l_v.cpu())

        val_loss_epoch = val_loss_total / len(y_va_fold)
        val_probs_epoch = torch.softmax(
            torch.cat(val_logits_list, dim=0), dim=-1
        ).numpy()

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_val_probs = val_probs_epoch

            t_logits = []
            with torch.no_grad():
                for (x_t,) in test_loader:
                    x_t = x_t.to(device)
                    t_logits.append(nn_model(x_t).cpu())
            best_test_probs = torch.softmax(torch.cat(t_logits, dim=0), dim=-1).numpy()

    oof_preds["nn"][va_idx] += best_val_probs / n_repeats
    test_preds["nn"] += best_test_probs / total_folds

    fold_acc = accuracy_score(y_va_fold, np.argmax(cat_model.predict_proba(X_va_fold), axis=1))
    print(f"Fold {fold_idx + 1}/{total_folds} Complete | CatBoost Val Acc: {fold_acc:.4f}")


# =========================================================================
# 5. Ensemble Optimization & Submission
# =========================================================================
def ensemble_loss_fn(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blended_oof = sum(w * oof_preds[m] for w, m in zip(weights, model_names))
    blended_oof = np.clip(blended_oof, 1e-15, 1.0 - 1e-15)
    return log_loss(y_tr, blended_oof)


init_weights = [1.0 / len(model_names)] * len(model_names)
bounds = [(0.0, 1.0) for _ in model_names]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss_fn,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x / np.sum(opt_res.x)

blended_oof_probs = sum(w * oof_preds[m] for w, m in zip(opt_weights, model_names))
blended_test_probs = sum(w * test_preds[m] for w, m in zip(opt_weights, model_names))


# Accuracy-Oriented Threshold / Class Multiplier Optimization via Nelder-Mead
def accuracy_neg_obj(multipliers):
    w = np.array(multipliers)
    scaled_probs = blended_oof_probs * w
    preds = np.argmax(scaled_probs, axis=1)
    return -accuracy_score(y_tr, preds)


init_multipliers = [1.0, 1.0, 1.0]
mult_res = minimize(
    accuracy_neg_obj,
    init_multipliers,
    method="Nelder-Mead",
    options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4},
)
opt_multipliers = mult_res.x

final_oof_preds_0indexed = np.argmax(blended_oof_probs * opt_multipliers, axis=1)
final_oof_preds_raw = final_oof_preds_0indexed + min_y
final_score = accuracy_score(y_tr_raw, final_oof_preds_raw)

final_test_preds_0indexed = np.argmax(blended_test_probs * opt_multipliers, axis=1)
final_test_preds_raw = final_test_preds_0indexed + min_y

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), target_col: final_test_preds_raw}
)
submission_df.to_csv(submission_path, index=False)

print(f"Ensemble Model Weights: {dict(zip(model_names, np.round(opt_weights, 3)))}")
print(f"Optimal Class Multipliers: {np.round(opt_multipliers, 4)}")
print(f"Final Validation Score: {final_score}")