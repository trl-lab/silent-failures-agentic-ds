import os
import random
from typing import Any, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# =========================================================================
# 0. Global Setup and Reproducibility
# =========================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Domain-Specific Feature Engineering
# =========================================================================
def engineer_features(df: pd.DataFrame, target_col: str = "NObeyesdad") -> pd.DataFrame:
    df_feat = df.copy()

    # Drop id and target if present
    cols_to_drop = [c for c in ["id", target_col] if c in df_feat.columns]
    df_feat = df_feat.drop(columns=cols_to_drop)

    # Anthropometric Features
    df_feat["BMI"] = df_feat["Weight"] / (df_feat["Height"] ** 2)
    df_feat["Ponderal_Index"] = df_feat["Weight"] / (df_feat["Height"] ** 3)
    df_feat["Height_Weight_Ratio"] = df_feat["Height"] / (df_feat["Weight"] + 1e-6)
    df_feat["Weight_Height_Ratio"] = df_feat["Weight"] / (df_feat["Height"] + 1e-6)
    df_feat["Log_BMI"] = np.log1p(df_feat["BMI"])
    df_feat["Log_Weight"] = np.log1p(df_feat["Weight"])
    df_feat["Log_Age"] = np.log1p(df_feat["Age"])

    # Mosteller Body Surface Area (BSA)
    df_feat["BSA"] = np.sqrt((df_feat["Height"] * 100.0 * df_feat["Weight"]) / 3600.0)

    # Devine Ideal Body Weight (IBW) & Deviations
    height_in = df_feat["Height"] * 39.3700787
    if "Gender" in df_feat.columns:
        is_male = df_feat["Gender"].astype(str).str.lower() == "male"
        ibw_base = np.where(is_male, 50.0, 45.5)
    else:
        ibw_base = 47.75
    df_feat["IBW"] = ibw_base + 2.3 * (height_in - 60.0)
    df_feat["IBW_Deviation"] = df_feat["Weight"] - df_feat["IBW"]
    df_feat["IBW_Ratio"] = df_feat["Weight"] / (df_feat["IBW"] + 1e-6)

    # Oxford New BMI Formula (1.3 * Weight / Height^2.5)
    df_feat["Oxford_BMI"] = 1.3 * df_feat["Weight"] / (df_feat["Height"] ** 2.5)
    df_feat["BMI_Diff_Oxford"] = df_feat["BMI"] - df_feat["Oxford_BMI"]

    # Continuous Distance Metrics to Exact WHO Obesity Threshold Boundaries
    for cut in [18.5, 25.0, 30.0, 35.0, 40.0]:
        df_feat[f"Dist_WHO_{str(cut).replace('.', '_')}"] = df_feat["BMI"] - cut
        df_feat[f"AbsDist_WHO_{str(cut).replace('.', '_')}"] = np.abs(df_feat["BMI"] - cut)

    # Standard WHO BMI Categories
    df_feat["BMI_WHO_Category"] = pd.cut(
        df_feat["BMI"],
        bins=[-np.inf, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float)

    # Ordinal Mappings
    ordinal_map_4 = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    if "CAEC" in df_feat.columns:
        df_feat["CAEC_ord"] = df_feat["CAEC"].map(ordinal_map_4).fillna(1).astype(float)
    if "CALC" in df_feat.columns:
        df_feat["CALC_ord"] = df_feat["CALC"].map(ordinal_map_4).fillna(0).astype(float)

    # Binary Mappings
    binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    for b_col in binary_cols:
        if b_col in df_feat.columns:
            df_feat[b_col + "_num"] = (
                df_feat[b_col].astype(str).str.lower() == "yes"
            ).astype(float)

    if "Gender" in df_feat.columns:
        df_feat["Gender_Male"] = (
            df_feat["Gender"].astype(str).str.lower() == "male"
        ).astype(float)
        # Gender-body composition cross-terms
        df_feat["Gender_BMI"] = df_feat["Gender_Male"] * df_feat["BMI"]
        df_feat["Gender_Weight"] = df_feat["Gender_Male"] * df_feat["Weight"]
        df_feat["Gender_BSA"] = df_feat["Gender_Male"] * df_feat["BSA"]
        df_feat["Gender_Oxford_BMI"] = df_feat["Gender_Male"] * df_feat["Oxford_BMI"]

    # Hierarchical behavioral interaction terms
    favc_val = df_feat["FAVC_num"] if "FAVC_num" in df_feat.columns else 0.0
    scc_val = df_feat["SCC_num"] if "SCC_num" in df_feat.columns else 0.0
    caec_ord = df_feat["CAEC_ord"] if "CAEC_ord" in df_feat.columns else 1.0

    df_feat["FAVC_CAEC_Interaction"] = favc_val * caec_ord
    df_feat["SCC_FAVC_Interaction"] = scc_val * favc_val
    df_feat["FAF_Sedentary_Age_Ratio"] = df_feat["FAF"] / (
        df_feat["TUE"] + (df_feat["Age"] / 100.0) + 0.05
    )

    # Behavioral and Activity Interactions
    df_feat["Physical_Sedentary_Ratio"] = df_feat["FAF"] / (df_feat["TUE"] + 0.1)
    df_feat["Water_Per_Kg"] = df_feat["CH2O"] / (df_feat["Weight"] + 1e-6)
    df_feat["Vegetables_Per_Meal"] = df_feat["FCVC"] / (df_feat["NCP"] + 1e-6)
    df_feat["Activity_Per_Age"] = df_feat["FAF"] / (df_feat["Age"] + 1e-6)
    df_feat["Age_BMI_Interaction"] = df_feat["Age"] * df_feat["BMI"]
    df_feat["FAF_Weight_Interaction"] = df_feat["FAF"] * df_feat["Weight"]
    df_feat["FCVC_CH2O_Interaction"] = df_feat["FCVC"] * df_feat["CH2O"]

    # Age-stratified body metric z-scores & deviations
    age_bins = pd.cut(
        df_feat["Age"],
        bins=[-np.inf, 20, 25, 30, 40, np.inf],
        labels=[0, 1, 2, 3, 4],
    )
    for metric in ["BMI", "Weight", "Height"]:
        grp_mean = df_feat.groupby(age_bins, observed=False)[metric].transform("mean")
        grp_std = (
            df_feat.groupby(age_bins, observed=False)[metric]
            .transform("std")
            .fillna(1.0)
        )
        df_feat[f"{metric}_AgeGrp_ZScore"] = (df_feat[metric] - grp_mean) / (
            grp_std + 1e-6
        )
        df_feat[f"{metric}_AgeGrp_Diff"] = df_feat[metric] - grp_mean

    # Transportation Mode Categorization
    if "MTRANS" in df_feat.columns:
        df_feat["Active_Commute"] = (
            df_feat["MTRANS"].isin(["Walking", "Bike"]).astype(float)
        )
        df_feat["Motorized_Commute"] = (
            df_feat["MTRANS"].isin(["Automobile", "Motorbike"]).astype(float)
        )
        df_feat["Public_Commute"] = (
            df_feat["MTRANS"] == "Public_Transportation"
        ).astype(float)

    # Survey Artifact Indicators
    df_feat["FCVC_is_integer"] = (df_feat["FCVC"] % 1 == 0).astype(float)
    df_feat["NCP_is_integer"] = (df_feat["NCP"] % 1 == 0).astype(float)
    df_feat["CH2O_is_integer"] = (df_feat["CH2O"] % 1 == 0).astype(float)
    df_feat["FAF_is_integer"] = (df_feat["FAF"] % 1 == 0).astype(float)
    df_feat["TUE_is_integer"] = (df_feat["TUE"] % 1 == 0).astype(float)
    df_feat["Age_is_integer"] = (df_feat["Age"] % 1 == 0).astype(float)

    # Lifestyle and Energy Balance Composite Scores
    favc_val = df_feat["FAVC_num"] if "FAVC_num" in df_feat.columns else 0.0
    scc_val = df_feat["SCC_num"] if "SCC_num" in df_feat.columns else 0.0
    smoke_val = df_feat["SMOKE_num"] if "SMOKE_num" in df_feat.columns else 0.0
    caec_ord = df_feat["CAEC_ord"] if "CAEC_ord" in df_feat.columns else 1.0

    df_feat["Healthy_Habit_Score"] = (
        (df_feat["FCVC"] / 3.0)
        + (df_feat["CH2O"] / 3.0)
        + (df_feat["FAF"] / 3.0)
        + scc_val
        - favc_val
        - (caec_ord / 3.0)
        - smoke_val
    )

    # Energy Balance Indices
    df_feat["Energy_Intake_Proxy"] = (favc_val + 1.0) * df_feat["NCP"] * (caec_ord + 1.0)
    df_feat["Energy_Expenditure_Proxy"] = (df_feat["FAF"] + 0.5) / (df_feat["TUE"] + 1.0)
    df_feat["Energy_Balance_Index"] = df_feat["Energy_Intake_Proxy"] / (
        df_feat["Energy_Expenditure_Proxy"] + 1e-6
    )

    # Frequency Encoding for Categoricals
    cat_freq_cols = [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
    ]
    for col in cat_freq_cols:
        if col in df_feat.columns:
            freq_map = df_feat[col].value_counts(normalize=True)
            df_feat[col + "_freq"] = df_feat[col].map(freq_map).astype(float)

    # Categorical string formatting
    cat_columns = df_feat.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_columns:
        df_feat[col] = df_feat[col].astype(str)

    return df_feat


# Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NObeyesdad"
if "id" in train_df.columns:
    train_ids = train_df["id"].values
else:
    train_ids = np.arange(len(train_df))

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# Encode Multi-Class Target
target_encoder = LabelEncoder()
y_all = target_encoder.fit_transform(train_df[target_col].values)
num_classes = len(target_encoder.classes_)

# Apply Feature Engineering and One-Hot Encoding
X_train_raw = engineer_features(train_df, target_col=target_col)
X_test_raw = engineer_features(test_df, target_col=target_col)

categorical_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
X_combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
X_combined_encoded = pd.get_dummies(
    X_combined, columns=categorical_cols, drop_first=False
)

n_train = len(X_train_raw)
X_train_df = X_combined_encoded.iloc[:n_train].copy().reset_index(drop=True)
X_test_df = X_combined_encoded.iloc[n_train:].copy().reset_index(drop=True)

X_tr_all = X_train_df.values.astype(np.float32)
X_te_all = X_test_df.values.astype(np.float32)
n_samples = len(X_tr_all)
n_test_samples = len(X_te_all)


# =========================================================================
# 2. Model Architectures & Builders
# =========================================================================
def get_lgbm_model(num_classes: int = 7, random_state: int = 42) -> lgb.LGBMClassifier:
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMClassifier(**params)


def get_xgb_model(num_classes: int = 7, random_state: int = 42) -> xgb.XGBClassifier:
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    return xgb.XGBClassifier(**params)


def get_catboost_model(
    num_classes: int = 7, random_state: int = 42
) -> cb.CatBoostClassifier:
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "early_stopping_rounds": 50,
        "verbose": False,
        "thread_count": -1,
    }
    return cb.CatBoostClassifier(**params)


def get_extratrees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=350,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


class TabularResBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
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
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


def get_torch_model_and_optimizer(
    input_dim: int,
    num_classes: int = 7,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 45,
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, Any]:
    model = TabularMLP(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


# =========================================================================
# 3. Stratified K-Fold Training & Inference Loop (Multi-Seed & 5 Diverse Models)
# =========================================================================
N_SPLITS = 10
SEEDS = [42, 2024]
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_lgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_xgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_cb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_et = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_nn = np.zeros((n_samples, num_classes), dtype=np.float32)

test_preds_lgb = np.zeros((n_test_samples, num_classes), dtype=np.float32)
test_preds_xgb = np.zeros((n_test_samples, num_classes), dtype=np.float32)
test_preds_cb = np.zeros((n_test_samples, num_classes), dtype=np.float32)
test_preds_et = np.zeros((n_test_samples, num_classes), dtype=np.float32)
test_preds_nn = np.zeros((n_test_samples, num_classes), dtype=np.float32)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_tr_all, y_all)):
    X_tr_f, y_tr_f = X_tr_all[train_idx], y_all[train_idx]
    X_va_f, y_va_f = X_tr_all[val_idx], y_all[val_idx]

    # 1. LightGBM (Multi-seed)
    fold_oof_lgb = np.zeros((len(val_idx), num_classes), dtype=np.float32)
    for s in SEEDS:
        lgb_model = get_lgbm_model(
            num_classes=num_classes, random_state=s + fold_idx * 10
        )
        lgb_model.fit(
            X_tr_f,
            y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        fold_oof_lgb += lgb_model.predict_proba(X_va_f) / len(SEEDS)
        test_preds_lgb += (
            lgb_model.predict_proba(X_te_all) / (N_SPLITS * len(SEEDS))
        )
    oof_lgb[val_idx] = fold_oof_lgb

    # 2. XGBoost (Multi-seed)
    fold_oof_xgb = np.zeros((len(val_idx), num_classes), dtype=np.float32)
    for s in SEEDS:
        xgb_model = get_xgb_model(
            num_classes=num_classes, random_state=s + fold_idx * 10
        )
        xgb_model.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
        fold_oof_xgb += xgb_model.predict_proba(X_va_f) / len(SEEDS)
        test_preds_xgb += (
            xgb_model.predict_proba(X_te_all) / (N_SPLITS * len(SEEDS))
        )
    oof_xgb[val_idx] = fold_oof_xgb

    # 3. CatBoost (Multi-seed)
    fold_oof_cb = np.zeros((len(val_idx), num_classes), dtype=np.float32)
    for s in SEEDS:
        cb_model = get_catboost_model(
            num_classes=num_classes, random_state=s + fold_idx * 10
        )
        cb_model.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), verbose=False)
        fold_oof_cb += cb_model.predict_proba(X_va_f) / len(SEEDS)
        test_preds_cb += (
            cb_model.predict_proba(X_te_all) / (N_SPLITS * len(SEEDS))
        )
    oof_cb[val_idx] = fold_oof_cb

    # 4. ExtraTrees Classifier
    et_model = get_extratrees_model(random_state=RANDOM_SEED + fold_idx)
    et_model.fit(X_tr_f, y_tr_f)
    oof_et[val_idx] = et_model.predict_proba(X_va_f)
    test_preds_et += et_model.predict_proba(X_te_all) / N_SPLITS

    # 5. Tabular Neural Network
    scaler = StandardScaler()
    X_tr_f_scaled = scaler.fit_transform(X_tr_f)
    X_va_f_scaled = scaler.transform(X_va_f)
    X_te_scaled = scaler.transform(X_te_all)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_f_scaled, dtype=torch.float32),
        torch.tensor(y_tr_f, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )

    nn_epochs = 45
    nn_model, criterion, optimizer, scheduler = get_torch_model_and_optimizer(
        input_dim=X_tr_f.shape[1],
        num_classes=num_classes,
        learning_rate=1e-3,
        epochs=nn_epochs,
    )
    nn_model.to(device)

    best_val_loss = float("inf")
    best_val_probs = None
    best_test_probs = None

    va_tensor_x = torch.tensor(X_va_f_scaled, dtype=torch.float32).to(device)
    va_tensor_y = torch.tensor(y_va_f, dtype=torch.long).to(device)
    te_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

    for epoch in range(nn_epochs):
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
        with torch.no_grad():
            val_logits = nn_model(va_tensor_x)
            val_loss = criterion(val_logits, va_tensor_y).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()
                test_logits = nn_model(te_tensor_x)
                best_test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()

    oof_nn[val_idx] = best_val_probs
    test_preds_nn += best_test_probs / N_SPLITS


# =========================================================================
# 4. Out-of-Fold Evaluation, Stacking Meta-Learner & Nelder-Mead Calibration
# =========================================================================
# Combine all model OOF predictions as stacking features
oof_stack_features = np.hstack([oof_lgb, oof_xgb, oof_cb, oof_et, oof_nn])
test_stack_features = np.hstack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_et, test_preds_nn]
)

# 2-Stage Out-of-Fold Regularized Multinomial Logistic Regression Meta-Learner
meta_oof_preds = np.zeros((n_samples, num_classes), dtype=np.float32)
meta_test_preds = np.zeros((n_test_samples, num_classes), dtype=np.float32)

meta_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
for meta_tr_idx, meta_va_idx in meta_skf.split(oof_stack_features, y_all):
    meta_clf = LogisticRegression(
        C=0.2,
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    meta_clf.fit(oof_stack_features[meta_tr_idx], y_all[meta_tr_idx])
    meta_oof_preds[meta_va_idx] = meta_clf.predict_proba(
        oof_stack_features[meta_va_idx]
    )
    meta_test_preds += (
        meta_clf.predict_proba(test_stack_features) / N_SPLITS
    )

# Composite Probabilities: Weighted combination of Meta-Learner and Base Model Ensembles
oof_base_mean = (oof_lgb + oof_xgb + oof_cb + oof_et + oof_nn) / 5.0
test_base_mean = (
    test_preds_lgb + test_preds_xgb + test_preds_cb + test_preds_et + test_preds_nn
) / 5.0

oof_composite = 0.65 * meta_oof_preds + 0.35 * oof_base_mean
test_composite = 0.65 * meta_test_preds + 0.35 * test_base_mean

# Metric-Aligned Optimizer with Temperature / Multiplier Scaling to Directly Maximize Discrete Accuracy
def accuracy_loss_objective(weights: np.ndarray) -> float:
    scaled_probs = oof_composite * weights
    preds = np.argmax(scaled_probs, axis=1)
    return -accuracy_score(y_all, preds)

init_multipliers = np.ones(num_classes, dtype=np.float64)
opt_result = minimize(
    accuracy_loss_objective,
    init_multipliers,
    method="Nelder-Mead",
    options={"maxiter": 1000, "disp": False},
)

optimal_multipliers = opt_result.x
oof_calibrated = oof_composite * optimal_multipliers
oof_final_pred = np.argmax(oof_calibrated, axis=1)
final_acc = accuracy_score(y_all, oof_final_pred)


# =========================================================================
# 5. Submission Generation
# =========================================================================
test_calibrated = test_composite * optimal_multipliers
test_final_pred_indices = np.argmax(test_calibrated, axis=1)
predicted_labels = target_encoder.inverse_transform(test_final_pred_indices)

submission_df = pd.DataFrame({"id": test_ids, "NObeyesdad": predicted_labels})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_acc}")
