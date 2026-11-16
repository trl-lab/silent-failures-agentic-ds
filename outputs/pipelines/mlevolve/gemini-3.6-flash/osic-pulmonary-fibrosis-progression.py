import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize

# Ensure necessary directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# Set seeds for complete reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. Data Processing & Feature Engineering
# =====================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))


def extract_patient_baselines(df):
    """
    Extracts baseline visit information for each patient.
    """
    df_sorted = df.sort_values(["Patient", "Weeks"]).copy()

    baseline_records = []
    for patient, group in df_sorted.groupby("Patient"):
        valid_fvc = group[group["FVC"].notna()] if "FVC" in group.columns else group
        base_row = valid_fvc.iloc[0] if len(valid_fvc) > 0 else group.iloc[0]

        baseline_records.append(
            {
                "Patient": patient,
                "baseline_FVC": base_row["FVC"] if "FVC" in base_row else np.nan,
                "baseline_Percent": base_row["Percent"],
                "baseline_Weeks": base_row["Weeks"],
                "baseline_Age": base_row["Age"],
                "baseline_Sex": base_row["Sex"],
                "baseline_SmokingStatus": base_row["SmokingStatus"],
            }
        )

    return pd.DataFrame(baseline_records)


train_baselines = extract_patient_baselines(train_df)
test_baselines = extract_patient_baselines(test_df)

train_df = train_df.merge(train_baselines, on="Patient", how="left")

test_base_cols = [c for c in test_baselines.columns if c != "Patient"]
for c in test_base_cols:
    if c in test_df.columns:
        test_df.drop(columns=[c], inplace=True)
test_df = test_df.merge(test_baselines, on="Patient", how="left")


def engineer_features(df):
    df = df.copy()

    # Temporal Differential Features
    df["dt_weeks"] = df["Weeks"] - df["baseline_Weeks"]
    df["abs_dt_weeks"] = df["dt_weeks"].abs()
    df["dt_weeks_sq"] = df["dt_weeks"] ** 2

    # Categorical Mappings
    sex_map = {"Male": 1, "Female": 0}
    smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}

    df["Sex_code"] = df["baseline_Sex"].map(sex_map).fillna(0).astype(int)
    df["SmokingStatus_code"] = (
        df["baseline_SmokingStatus"].map(smoke_map).fillna(0).astype(int)
    )
    df["sex_smoke_comb"] = df["Sex_code"] * 3 + df["SmokingStatus_code"]

    # Physiological Proxies & Interactions
    df["normal_fvc_proxy"] = df["baseline_FVC"] / np.maximum(
        df["baseline_Percent"] / 100.0, 0.05
    )
    df["fvc_age_ratio"] = df["baseline_FVC"] / (df["baseline_Age"] + 1e-5)

    # Temporal-Physiological Interactions
    df["dt_fvc_interaction"] = df["dt_weeks"] * df["baseline_FVC"]
    df["dt_percent_interaction"] = df["dt_weeks"] * df["baseline_Percent"]
    df["dt_age_interaction"] = df["dt_weeks"] * df["baseline_Age"]
    df["dt_smoke_interaction"] = df["dt_weeks"] * df["SmokingStatus_code"]

    # Higher-order non-linear physiological interactions
    df["fvc_age_percent_ratio"] = df["baseline_FVC"] / ((df["baseline_Age"] + 1e-5) * (df["baseline_Percent"] / 100.0 + 1e-5))
    df["smoke_percent_interaction"] = df["SmokingStatus_code"] * df["baseline_Percent"]
    df["smoke_age_interaction"] = df["SmokingStatus_code"] * df["baseline_Age"]

    # Temporal acceleration proxies
    df["dt_weeks_sq_fvc_interaction"] = df["dt_weeks_sq"] * df["baseline_FVC"]
    df["dt_weeks_sq_percent_interaction"] = df["dt_weeks_sq"] * df["baseline_Percent"]
    df["dt_weeks_sq_smoke_interaction"] = df["dt_weeks_sq"] * df["SmokingStatus_code"]

    # Baseline expected linear decline trajectory anchor (-1.72 ml/week estimated mean decline)
    df["linear_fvc_decline_est"] = df["baseline_FVC"] + df["dt_weeks"] * (-1.72)

    return df


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

feature_cols = [
    "baseline_FVC",
    "baseline_Percent",
    "baseline_Weeks",
    "baseline_Age",
    "Weeks",
    "dt_weeks",
    "abs_dt_weeks",
    "dt_weeks_sq",
    "Sex_code",
    "SmokingStatus_code",
    "sex_smoke_comb",
    "normal_fvc_proxy",
    "fvc_age_ratio",
    "fvc_age_percent_ratio",
    "smoke_percent_interaction",
    "smoke_age_interaction",
    "dt_fvc_interaction",
    "dt_percent_interaction",
    "dt_age_interaction",
    "dt_smoke_interaction",
    "dt_weeks_sq_fvc_interaction",
    "dt_weeks_sq_percent_interaction",
    "dt_weeks_sq_smoke_interaction",
    "linear_fvc_decline_est",
]

cat_cols = ["Sex_code", "SmokingStatus_code", "sex_smoke_comb"]
num_cols = [c for c in feature_cols if c not in cat_cols]
cont_cols = [c for c in feature_cols if c not in cat_cols]
target_col = "FVC"
id_col = "id"
n_folds = 10

# Global missing values for test will be filled using overall train medians after fold calculations
train_medians = {col: train_df[col].median() for col in num_cols}
for col in num_cols:
    test_df[col] = test_df[col].fillna(train_medians[col])

# GroupKFold partitioning by Patient
gkf = GroupKFold(n_splits=n_folds)
train_df["fold"] = -1
patient_groups = train_df["Patient"].values
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=patient_groups)):
    train_df.loc[val_idx, "fold"] = fold

# Dynamic categorical cardinalities
cat_cardinalities = {}
for c in cat_cols:
    max_val = max(int(train_df[c].max()), int(test_df[c].max()))
    cat_cardinalities[c] = max_val + 2


# =====================================================================
# 2. PyTorch Model Architecture & Dataset Definition
# =====================================================================
class TabularResBlock(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.ln2 = nn.LayerNorm(in_features)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return residual + out


class OSICLungTrajectoryNet(nn.Module):

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: dict,
        emb_dims: dict = None,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()

        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        default_emb_dims = {"Sex_code": 4, "SmokingStatus_code": 4, "sex_smoke_comb": 8}
        if emb_dims is not None:
            default_emb_dims.update(emb_dims)

        for cat_col, cardinality in cat_cardinalities.items():
            emb_dim = default_emb_dims.get(cat_col, 4)
            self.embeddings[cat_col] = nn.Embedding(cardinality, emb_dim)
            total_emb_dim += emb_dim

        in_dim = num_continuous + total_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, hidden_dim * 2, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        self.slope_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.curvature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: dict,
        baseline_fvc: torch.Tensor,
        dt_weeks: torch.Tensor,
    ) -> torch.Tensor:
        emb_list = []
        for cat_col, emb_layer in self.embeddings.items():
            if cat_col in x_cat:
                emb_list.append(emb_layer(x_cat[cat_col]))

        if emb_list:
            x_cat_emb = torch.cat(emb_list, dim=1)
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)

        slope = self.slope_head(h).squeeze(-1)
        curvature = self.curvature_head(h).squeeze(-1)
        residual_offset = self.residual_head(h).squeeze(-1)

        base_fvc = baseline_fvc.view(-1)
        dt = dt_weeks.view(-1)
        dt_sq = dt ** 2

        fvc_pred = base_fvc + dt * slope + dt_sq * curvature + residual_offset

        return fvc_pred


class RobustMAELoss(nn.Module):

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred.view(-1), y_true.view(-1))


class OSICDataset(Dataset):

    def __init__(self, x_num, x_cat_dict, base_fvc, dt_w, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat_dict = {
            k: torch.tensor(v, dtype=torch.long) for k, v in x_cat_dict.items()
        }
        self.base_fvc = torch.tensor(base_fvc, dtype=torch.float32)
        self.dt_w = torch.tensor(dt_w, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        cat_sample = {k: v[idx] for k, v in self.x_cat_dict.items()}
        if self.y is not None:
            return (
                self.x_num[idx],
                cat_sample,
                self.base_fvc[idx],
                self.dt_w[idx],
                self.y[idx],
            )
        return self.x_num[idx], cat_sample, self.base_fvc[idx], self.dt_w[idx]


# =====================================================================
# 3. Model Parameters & Cross-Validation Pipeline
# =====================================================================
lgb_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_estimators": 1500,
    "n_jobs": -1,
}

oof_preds_lgb = np.zeros(len(train_df))
oof_preds_xgb = np.zeros(len(train_df))
oof_preds_nn = np.zeros(len(train_df))

test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))
test_preds_nn = np.zeros(len(test_df))

for fold in range(n_folds):
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    df_tr, df_va = train_df[train_mask].copy(), train_df[val_mask].copy()

    # Impute missing values using fold training statistics
    for col in num_cols:
        fold_median = df_tr[col].median()
        df_tr[col] = df_tr[col].fillna(fold_median)
        df_va[col] = df_va[col].fillna(fold_median)

    # --- LightGBM Model ---
    X_tr, y_tr = df_tr[feature_cols], df_tr[target_col].values
    X_va, y_va = df_va[feature_cols], df_va[target_col].values
    X_te = test_df[feature_cols]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds_lgb_fold = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    test_preds_lgb_fold = lgb_model.predict(
        X_te, num_iteration=lgb_model.best_iteration
    )

    oof_preds_lgb[val_mask] = val_preds_lgb_fold
    test_preds_lgb += test_preds_lgb_fold / n_folds

    # --- XGBoost Model ---
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    val_preds_xgb_fold = xgb_model.predict(X_va)
    test_preds_xgb_fold = xgb_model.predict(X_te)

    oof_preds_xgb[val_mask] = val_preds_xgb_fold
    test_preds_xgb += test_preds_xgb_fold / n_folds

    # --- PyTorch Trajectory Neural Network ---
    scaler = StandardScaler()
    scaler.fit(df_tr[cont_cols])

    X_num_tr = scaler.transform(df_tr[cont_cols])
    X_num_va = scaler.transform(df_va[cont_cols])
    X_num_te = scaler.transform(test_df[cont_cols])

    X_cat_tr = {c: df_tr[c].values for c in cat_cols}
    X_cat_va = {c: df_va[c].values for c in cat_cols}
    X_cat_te = {c: test_df[c].values for c in cat_cols}

    base_fvc_tr, dt_w_tr = df_tr["baseline_FVC"].values, df_tr["dt_weeks"].values
    base_fvc_va, dt_w_va = df_va["baseline_FVC"].values, df_va["dt_weeks"].values
    base_fvc_te, dt_w_te = test_df["baseline_FVC"].values, test_df["dt_weeks"].values

    train_ds = OSICDataset(X_num_tr, X_cat_tr, base_fvc_tr, dt_w_tr, y_tr)
    val_ds = OSICDataset(X_num_va, X_cat_va, base_fvc_va, dt_w_va, y_va)
    test_ds = OSICDataset(X_num_te, X_cat_te, base_fvc_te, dt_w_te)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    nn_model = OSICLungTrajectoryNet(
        num_continuous=len(cont_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=128,
        num_res_blocks=2,
        dropout_rate=0.15,
    ).to(device)

    criterion = RobustMAELoss(beta=1.0)
    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

    best_val_mae = float("inf")
    best_nn_preds = None

    for epoch in range(80):
        nn_model.train()
        for x_n, x_c, base, dt, y in train_loader:
            x_n, base, dt, y = (
                x_n.to(device),
                base.to(device),
                dt.to(device),
                y.to(device),
            )
            x_c = {k: v.to(device) for k, v in x_c.items()}

            optimizer.zero_grad()
            preds = nn_model(x_n, x_c, base, dt)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for x_n, x_c, base, dt, y in val_loader:
                x_n, base, dt = x_n.to(device), base.to(device), dt.to(device)
                x_c = {k: v.to(device) for k, v in x_c.items()}
                preds = nn_model(x_n, x_c, base, dt)
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(y.numpy())

        current_val_mae = mean_absolute_error(val_targets_list, val_preds_list)
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            best_nn_preds = np.array(val_preds_list)
            torch.save(nn_model.state_dict(), f"./working/best_nn_fold{fold}.pt")

    oof_preds_nn[val_mask] = best_nn_preds

    nn_model.load_state_dict(torch.load(f"./working/best_nn_fold{fold}.pt"))
    nn_model.eval()
    fold_test_preds_nn = []
    with torch.no_grad():
        for x_n, x_c, base, dt in test_loader:
            x_n, base, dt = x_n.to(device), base.to(device), dt.to(device)
            x_c = {k: v.to(device) for k, v in x_c.items()}
            preds = nn_model(x_n, x_c, base, dt)
            fold_test_preds_nn.extend(preds.cpu().numpy())

    test_preds_nn += np.array(fold_test_preds_nn) / n_folds

    fold_lgb_mae = mean_absolute_error(y_va, val_preds_lgb_fold)
    fold_xgb_mae = mean_absolute_error(y_va, val_preds_xgb_fold)
    print(f"Fold {fold} - LGB MAE: {fold_lgb_mae:.4f} | XGB MAE: {fold_xgb_mae:.4f} | NN MAE: {best_val_mae:.4f}")

# =====================================================================
# 4. Out-Of-Fold Ensemble Optimization & Submission
# =====================================================================
lgb_cv_mae = mean_absolute_error(train_df[target_col].values, oof_preds_lgb)
xgb_cv_mae = mean_absolute_error(train_df[target_col].values, oof_preds_xgb)
nn_cv_mae = mean_absolute_error(train_df[target_col].values, oof_preds_nn)

print(f"LGB CV MAE: {lgb_cv_mae:.4f}")
print(f"XGB CV MAE: {xgb_cv_mae:.4f}")
print(f"NN CV MAE:  {nn_cv_mae:.4f}")

y_true_all = train_df[target_col].values

def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = weights[0] * oof_preds_lgb + weights[1] * oof_preds_xgb + weights[2] * oof_preds_nn
    return mean_absolute_error(y_true_all, blend)

init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)}
)

opt_weights = res.x / np.sum(res.x)
best_ensemble_mae = res.fun

print(f"Optimized Ensemble Weights: LGB={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, NN={opt_weights[2]:.4f}")

final_test_preds = (
    opt_weights[0] * test_preds_lgb +
    opt_weights[1] * test_preds_xgb +
    opt_weights[2] * test_preds_nn
)
final_test_preds = np.clip(final_test_preds, 100.0, 8000.0)

submission_df = pd.DataFrame(
    {"id": test_df[id_col].astype(int).values, "FVC": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {best_ensemble_mae}")