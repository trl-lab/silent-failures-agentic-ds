import os
import random
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 0. Reproducibility & Environment Setup
# ==========================================
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==========================================
# 1. Load Data & Domain Feature Engineering
# ==========================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"

# Ordinal Mappings reflecting domain hierarchical structure
ORDINAL_MAPS = {
    "buying": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "maint": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
    "persons": {"2": 2, "4": 4, "more": 6},
    "lug_boot": {"small": 1, "med": 2, "big": 3},
    "safety": {"low": 1, "med": 2, "high": 3},
}

RAW_CATEGORICAL_COLS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


def engineer_features(train_df, test_df):
    """
    Construct complete hierarchical concept features reflecting the deterministic DEX model:
    - PRICE = f(buying, maint)
    - COMFORT = f(doors, persons, lug_boot)
    - TECH = f(doors, persons, lug_boot, safety)
    - All 15 pairwise categorical interactions, frequency encodings, and one-hot encoding
    """
    combined = pd.concat([train_df, test_df], ignore_index=True)

    for col, mapping in ORDINAL_MAPS.items():
        if col in combined.columns:
            combined[f"{col}_ord"] = (
                combined[col].astype(str).map(mapping).fillna(0).astype(float)
            )

    # Concept 1: PRICE hierarchy
    combined["price_sum"] = combined["buying_ord"] + combined["maint_ord"]
    combined["price_mult"] = combined["buying_ord"] * combined["maint_ord"]
    combined["price_ratio"] = combined["buying_ord"] / (combined["maint_ord"] + 1e-5)

    # Concept 2: COMFORT hierarchy
    combined["comfort_sum"] = (
        combined["doors_ord"] + combined["persons_ord"] + combined["lug_boot_ord"]
    )
    combined["comfort_mult"] = combined["persons_ord"] * combined["lug_boot_ord"]
    combined["capacity_ratio"] = combined["persons_ord"] / (combined["doors_ord"] + 1e-5)

    # Concept 3: TECH hierarchy
    combined["tech_score"] = combined["comfort_sum"] * combined["safety_ord"]
    combined["safety_person_mult"] = combined["safety_ord"] * combined["persons_ord"]

    # Concept 4: Global Balance
    combined["quality_to_price_ratio"] = combined["tech_score"] / (
        combined["price_sum"] + 1e-5
    )
    combined["safety_to_price_ratio"] = combined["safety_ord"] / (
        combined["price_sum"] + 1e-5
    )

    # Critical Veto Condition Flags
    combined["is_safety_low"] = (combined["safety"].astype(str) == "low").astype(int)
    combined["is_persons_2"] = (combined["persons"].astype(str) == "2").astype(int)
    combined["is_veto_risk"] = (
        (combined["is_safety_low"] == 1) | (combined["is_persons_2"] == 1)
    ).astype(int)
    combined["is_price_extreme"] = (
        (combined["buying"].astype(str) == "vhigh")
        & (combined["maint"].astype(str) == "vhigh")
    ).astype(int)

    # Complete DEX Hierarchical Concept Tuples
    combined["price_tuple"] = (
        combined["buying"].astype(str) + "_" + combined["maint"].astype(str)
    )
    combined["comfort_tuple"] = (
        combined["doors"].astype(str)
        + "_"
        + combined["persons"].astype(str)
        + "_"
        + combined["lug_boot"].astype(str)
    )
    combined["tech_tuple"] = (
        combined["doors"].astype(str)
        + "_"
        + combined["persons"].astype(str)
        + "_"
        + combined["lug_boot"].astype(str)
        + "_"
        + combined["safety"].astype(str)
    )

    # All 15 Pairwise Categorical Combinations
    for col1, col2 in itertools.combinations(RAW_CATEGORICAL_COLS, 2):
        combined[f"{col1}_{col2}_pair"] = (
            combined[col1].astype(str) + "_" + combined[col2].astype(str)
        )

    # Frequency encodings for categorical columns & tuples
    str_cols = [
        c for c in combined.columns if combined[c].dtype == "object" and c != target_col
    ]
    for c in str_cols:
        freq = combined[c].value_counts(normalize=True)
        combined[f"{c}_freq"] = combined[c].map(freq).astype(float)

    # One-hot encoding for raw categories and key concept tuples
    ohe_cols = RAW_CATEGORICAL_COLS + ["price_tuple", "comfort_tuple"]
    ohe_df = pd.get_dummies(
        combined[ohe_cols], prefix=ohe_cols, drop_first=False, dtype=float
    )
    combined = pd.concat([combined, ohe_df], axis=1)

    # Label encoding for all object columns
    for col in str_cols:
        le = LabelEncoder()
        combined[f"{col}_le"] = le.fit_transform(combined[col].astype(str))

    n_train = len(train_df)
    train_res = combined.iloc[:n_train].copy().reset_index(drop=True)
    test_res = combined.iloc[n_train:].copy().reset_index(drop=True)
    return train_res, test_res


train_processed, test_processed = engineer_features(train_df, test_df)

# Encode target
target_le = LabelEncoder()
train_processed["target"] = target_le.fit_transform(
    train_processed[target_col].astype(str)
)
classes = target_le.classes_
num_classes = len(classes)

# 5-Fold Stratified Split
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_processed["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, train_processed["target"])
):
    train_processed.loc[val_idx, "fold"] = fold

# Feature column selection
feature_cols = [
    c
    for c in train_processed.columns
    if c not in ["class", "target", "fold"]
    and pd.api.types.is_numeric_dtype(train_processed[c])
]

X_all = train_processed[feature_cols].copy()
y_all = train_processed["target"].values
X_test_df = test_processed[feature_cols].copy()
n_features = len(feature_cols)

# ==========================================
# 2. Model Architectures & Factories
# ==========================================


class TabularResNetBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [TabularResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def get_nn_model_and_optimizer(
    num_features: int,
    num_classes: int = 4,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.03,
    t_max: int = 60,
):
    model = TabularResNet(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    return model, criterion, optimizer, scheduler


def get_lgb_model(num_classes: int = 4, random_state: int = 42):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=random_state,
        n_estimators=1000,
        verbose=-1,
        n_jobs=-1,
    )


def get_xgb_model(num_classes: int = 4, random_state: int = 42):
    return XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(num_classes: int = 4, random_state: int = 42):
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        min_data_in_leaf=1,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def get_et_model(random_state: int = 42):
    return ExtraTreesClassifier(
        n_estimators=500,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )


def get_rf_model(random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=500,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )


# ==========================================
# 3. Stratified 5-Fold Multi-Model Training
# ==========================================
oof_lgb = np.zeros((len(train_processed), num_classes))
oof_xgb = np.zeros((len(train_processed), num_classes))
oof_cb = np.zeros((len(train_processed), num_classes))
oof_et = np.zeros((len(train_processed), num_classes))
oof_rf = np.zeros((len(train_processed), num_classes))
oof_nn = np.zeros((len(train_processed), num_classes))

test_preds_lgb = np.zeros((len(test_processed), num_classes))
test_preds_xgb = np.zeros((len(test_processed), num_classes))
test_preds_cb = np.zeros((len(test_processed), num_classes))
test_preds_et = np.zeros((len(test_processed), num_classes))
test_preds_rf = np.zeros((len(test_processed), num_classes))
test_preds_nn = np.zeros((len(test_processed), num_classes))

unacc_label_idx = target_le.transform(["unacc"])[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(N_SPLITS):
    train_mask = train_processed["fold"] != fold
    val_mask = train_processed["fold"] == fold

    X_train_fold = X_all[train_mask].values
    y_train_fold = y_all[train_mask]
    X_val_fold = X_all[val_mask].values
    y_val_fold = y_all[val_mask]
    X_test_arr = X_test_df.values

    val_indices = np.where(val_mask)[0]

    # --- LightGBM ---
    lgb_clf = get_lgb_model(num_classes=num_classes, random_state=SEED + fold * 10)
    lgb_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_indices] = lgb_clf.predict_proba(X_val_fold)
    test_preds_lgb += lgb_clf.predict_proba(X_test_arr) / N_SPLITS

    # --- XGBoost ---
    xgb_clf = get_xgb_model(num_classes=num_classes, random_state=SEED + fold * 10)
    xgb_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    oof_xgb[val_indices] = xgb_clf.predict_proba(X_val_fold)
    test_preds_xgb += xgb_clf.predict_proba(X_test_arr) / N_SPLITS

    # --- CatBoost ---
    cb_clf = get_catboost_model(num_classes=num_classes, random_state=SEED + fold * 10)
    cb_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_indices] = cb_clf.predict_proba(X_val_fold)
    test_preds_cb += cb_clf.predict_proba(X_test_arr) / N_SPLITS

    # --- Extra Trees ---
    et_clf = get_et_model(random_state=SEED + fold * 10)
    et_clf.fit(X_train_fold, y_train_fold)
    oof_et[val_indices] = et_clf.predict_proba(X_val_fold)
    test_preds_et += et_clf.predict_proba(X_test_arr) / N_SPLITS

    # --- Random Forest ---
    rf_clf = get_rf_model(random_state=SEED + fold * 10)
    rf_clf.fit(X_train_fold, y_train_fold)
    oof_rf[val_indices] = rf_clf.predict_proba(X_val_fold)
    test_preds_rf += rf_clf.predict_proba(X_test_arr) / N_SPLITS

    # --- Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train_fold)
    X_val_sc = scaler.transform(X_val_fold)
    X_te_sc = scaler.transform(X_test_arr)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_train_fold, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_sc, dtype=torch.float32),
        torch.tensor(y_val_fold, dtype=torch.long),
    )
    test_dataset = TensorDataset(torch.tensor(X_te_sc, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    nn_model, criterion, optimizer, scheduler = get_nn_model_and_optimizer(
        num_features=n_features,
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.15,
        lr=2e-3,
        weight_decay=1e-4,
        label_smoothing=0.03,
        t_max=60,
    )
    nn_model.to(device)

    best_val_loss = float("inf")
    best_nn_state = None

    for epoch in range(60):
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
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                val_loss += criterion(logits, batch_y).item() * len(batch_y)
                val_samples += len(batch_y)
        val_loss /= val_samples

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_state = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_nn_state)
    nn_model.eval()

    val_probs_nn = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(nn_model(batch_x), dim=1).cpu().numpy()
            val_probs_nn.append(probs)
    oof_nn[val_indices] = np.vstack(val_probs_nn)

    test_probs_nn_fold = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(nn_model(batch_x), dim=1).cpu().numpy()
            test_probs_nn_fold.append(probs)
    test_preds_nn += np.vstack(test_probs_nn_fold) / N_SPLITS

    fold_blend = (
        oof_lgb[val_indices]
        + oof_xgb[val_indices]
        + oof_cb[val_indices]
        + oof_et[val_indices]
        + oof_rf[val_indices]
        + oof_nn[val_indices]
    ) / 6.0
    fold_val_veto = (
        (train_processed.iloc[val_indices]["safety"].astype(str) == "low")
        | (train_processed.iloc[val_indices]["persons"].astype(str) == "2")
    ).values
    fold_blend[fold_val_veto, :] = 0.0
    fold_blend[fold_val_veto, unacc_label_idx] = 1.0
    fold_acc = accuracy_score(y_val_fold, np.argmax(fold_blend, axis=1))
    print(f"Fold {fold+1}/{N_SPLITS} - Validation Accuracy: {fold_acc:.4f}")

# ==========================================
# 4. Out-of-Fold Evaluation & Submission
# ==========================================
oof_list = [oof_lgb, oof_xgb, oof_cb, oof_et, oof_rf, oof_nn]
test_list = [
    test_preds_lgb,
    test_preds_xgb,
    test_preds_cb,
    test_preds_et,
    test_preds_rf,
    test_preds_nn,
]
n_models = len(oof_list)

train_veto_mask = (
    (train_processed["safety"].astype(str) == "low")
    | (train_processed["persons"].astype(str) == "2")
).values
test_veto_mask = (
    (test_processed["safety"].astype(str) == "low")
    | (test_processed["persons"].astype(str) == "2")
).values


def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = sum(w * oof for w, oof in zip(weights, oof_list))
    blend[train_veto_mask, :] = 0.0
    blend[train_veto_mask, unacc_label_idx] = 1.0
    blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
    blend = blend / np.sum(blend, axis=1, keepdims=True)
    return log_loss(y_all, blend)


initial_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-7},
)
best_weights = opt_res.x / np.sum(opt_res.x)
print(f"Optimal Blending Weights: {[round(float(w), 4) for w in best_weights]}")

oof_blend = sum(w * oof for w, oof in zip(best_weights, oof_list))
oof_blend[train_veto_mask, :] = 0.0
oof_blend[train_veto_mask, unacc_label_idx] = 1.0

final_val_score = float(accuracy_score(y_all, np.argmax(oof_blend, axis=1)))

test_blend = sum(w * t_pred for w, t_pred in zip(best_weights, test_list))
test_blend[test_veto_mask, :] = 0.0
test_blend[test_veto_mask, unacc_label_idx] = 1.0

test_predicted_classes = np.argmax(test_blend, axis=1)
test_predicted_labels = classes[test_predicted_classes]

id_col = test_df["id"] if "id" in test_df.columns else np.arange(len(test_processed))
submission_df = pd.DataFrame({"id": id_col, "class": test_predicted_labels})

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_score}")
