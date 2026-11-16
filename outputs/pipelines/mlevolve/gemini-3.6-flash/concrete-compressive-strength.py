import os
import random
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Reproducibility & Environment Setup
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load Raw Datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)
target_col = "strength"


# 3. Domain-Specific Feature Engineering Function
def create_concrete_features(df):
    data = df.copy()

    # Raw component variables
    c = data["cement"]
    slag = data["blast_furnace_slag"]
    fly_ash = data["fly_ash"]
    water = data["water"]
    sp = data["superplasticizer"]
    coarse = data["coarse_aggregate"]
    fine = data["fine_aggregate"]
    age = data["age"]

    # 3.1 Chemical & Binder Aggregations
    data["total_binder"] = c + slag + fly_ash
    b = data["total_binder"]
    data["pozzolan"] = slag + fly_ash

    # 3.2 Abram's Law & Mix Proportions
    data["water_binder_ratio"] = water / (b + 1e-5)
    data["water_cement_ratio"] = water / (c + 1e-5)
    data["sp_binder_ratio"] = sp / (b + 1e-5)
    data["sp_water_ratio"] = sp / (water + 1e-5)

    data["slag_binder_ratio"] = slag / (b + 1e-5)
    data["flyash_binder_ratio"] = fly_ash / (b + 1e-5)
    data["pozzolan_binder_ratio"] = data["pozzolan"] / (b + 1e-5)
    data["cement_binder_ratio"] = c / (b + 1e-5)

    # 3.3 Aggregates & Volumetric Mass Fractions
    data["total_aggregate"] = coarse + fine
    agg = data["total_aggregate"]
    data["fine_coarse_ratio"] = fine / (coarse + 1e-5)
    data["aggregate_binder_ratio"] = agg / (b + 1e-5)
    data["fine_binder_ratio"] = fine / (b + 1e-5)
    data["coarse_binder_ratio"] = coarse / (b + 1e-5)

    data["total_mass"] = b + water + agg + sp
    data["binder_mass_fraction"] = b / (data["total_mass"] + 1e-5)
    data["water_mass_fraction"] = water / (data["total_mass"] + 1e-5)
    data["agg_mass_fraction"] = agg / (data["total_mass"] + 1e-5)

    # 3.4 Hydration Kinetics & Non-linear Age Curves
    data["log_age"] = np.log1p(age)
    data["sqrt_age"] = np.sqrt(age)
    data["age_cubed_root"] = np.cbrt(age)
    data["hydration_decay"] = 1.0 - np.exp(-0.035 * age)

    # Abram's Law inverse power transform & Compound maturity
    data["abram_inverse_wb"] = 1.0 / (data["water_binder_ratio"] + 1e-5)
    data["compound_maturity"] = (b * data["log_age"]) / (data["water_binder_ratio"] + 1e-5)

    # Age Interaction Terms
    data["cement_log_age"] = c * data["log_age"]
    data["binder_log_age"] = b * data["log_age"]
    data["wb_ratio_log_age"] = data["water_binder_ratio"] / (data["log_age"] + 1e-5)

    # 3.5 Gel-Space Ratio Proxy & Effective Water
    effective_water = np.maximum(water - 4.5 * sp, 40.0)
    data["effective_wb_ratio"] = effective_water / (b + 1e-5)
    data["gel_space_ratio_proxy"] = (0.68 * b) / (0.32 * b + water + 1e-5)

    # Empirical Strength Models & Hydration Physics Formulas
    data["fib_age_maturation_20"] = np.exp(0.20 * (1.0 - np.sqrt(28.0 / (age + 1e-5))))
    data["fib_age_maturation_25"] = np.exp(0.25 * (1.0 - np.sqrt(28.0 / (age + 1e-5))))
    data["fib_age_maturation_38"] = np.exp(0.38 * (1.0 - np.sqrt(28.0 / (age + 1e-5))))
    data["bolomey_binder_water"] = (b / (water + 1e-5)) - 0.5
    data["bolomey_cement_water"] = (c / (water + 1e-5)) - 0.5
    data["gel_space_ratio_cubic"] = data["gel_space_ratio_proxy"] ** 3
    data["hyperbolic_age_maturation"] = age / (4.0 + 0.85 * age)
    data["log_water_binder_ratio"] = np.log(data["water_binder_ratio"] + 1e-5)
    data["log_water_cement_ratio"] = np.log(data["water_cement_ratio"] + 1e-5)

    # Standard curing age binary flags
    for ca in [3, 7, 14, 28, 56, 90, 180, 365]:
        data[f"age_equals_{ca}"] = (age == ca).astype(np.float32)

    return data


def compute_age_group_stats(df_tr):
    wb_means = df_tr.groupby("age")["water_binder_ratio"].mean()
    tb_means = df_tr.groupby("age")["total_binder"].mean()
    global_wb = df_tr["water_binder_ratio"].mean()
    global_tb = df_tr["total_binder"].mean()
    return wb_means, tb_means, global_wb, global_tb


def add_age_group_features(df, stats):
    wb_means, tb_means, global_wb, global_tb = stats
    df_out = df.copy()
    age_wb_mean = df_out["age"].map(wb_means).fillna(global_wb)
    age_tb_mean = df_out["age"].map(tb_means).fillna(global_tb)

    df_out["wb_diff_age_mean"] = df_out["water_binder_ratio"] - age_wb_mean
    df_out["wb_ratio_age_mean"] = df_out["water_binder_ratio"] / (age_wb_mean + 1e-5)
    df_out["tb_diff_age_mean"] = df_out["total_binder"] - age_tb_mean
    df_out["tb_ratio_age_mean"] = df_out["total_binder"] / (age_tb_mean + 1e-5)
    return df_out


# Apply Feature Engineering
train_fe = create_concrete_features(train_df)
test_fe = create_concrete_features(test_df)

feature_cols = [c for c in train_fe.columns if c not in [target_col, "id"]]

X = train_fe[feature_cols].copy()
y = train_fe[target_col].values
X_test = test_fe[feature_cols].copy()

n_samples = len(X)
n_test = len(X_test)
n_splits = 10
SEEDS = [42, 100, 2024]


# 4. PyTorch Tabular ResNet Architecture
class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.10):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = x + self.act(block(x))
        return self.head(x)


def train_nn_fold(X_tr, y_tr, X_va, y_va, X_te, fold_num, epochs=100):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    tr_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    tr_loader = DataLoader(
        tr_ds, batch_size=32, shuffle=True, drop_last=False, num_workers=0
    )

    model = TabularResNet(input_dim=X_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state = None

    X_va_tensor = torch.tensor(X_va_s, dtype=torch.float32).to(device)
    y_va_tensor = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1).to(device)

    for ep in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_tensor)
            val_loss = criterion(val_preds, y_va_tensor).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, f"./working/models/nn_fold_{fold_num}.pt")

    model.eval()
    with torch.no_grad():
        val_oof = model(X_va_tensor).cpu().numpy().ravel()
        X_te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        te_preds = model(X_te_tensor).cpu().numpy().ravel()

    return np.clip(val_oof, 0, None), np.clip(te_preds, 0, None)


# 5. Out-of-Fold Prediction Arrays Initialization
oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_et = np.zeros(n_samples)
oof_ridge = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_et = np.zeros(n_test)
test_ridge = np.zeros(n_test)
test_nn = np.zeros(n_test)

# 6. Multi-Seed 10-Fold Cross-Validation Training Loop
for s_idx, s_val in enumerate(SEEDS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=s_val)
    seed_oof_lgb = np.zeros(n_samples)
    seed_oof_xgb = np.zeros(n_samples)
    seed_oof_cat = np.zeros(n_samples)
    seed_oof_et = np.zeros(n_samples)
    seed_oof_ridge = np.zeros(n_samples)
    seed_oof_nn = np.zeros(n_samples)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr_raw, y_tr = X.iloc[train_idx], y[train_idx]
        X_va_raw, y_va = X.iloc[val_idx], y[val_idx]
        X_te_raw = X_test.copy()

        # Compute fold-level age-group mix statistics strictly on training fold
        age_stats = compute_age_group_stats(X_tr_raw)
        X_tr = add_age_group_features(X_tr_raw, age_stats)
        X_va = add_age_group_features(X_va_raw, age_stats)
        X_te = add_age_group_features(X_te_raw, age_stats)

        # Model 1: LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=s_val + fold,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        seed_oof_lgb[val_idx] = np.clip(lgb_model.predict(X_va), 0, None)
        test_lgb += np.clip(lgb_model.predict(X_te), 0, None) / (n_splits * len(SEEDS))

        # Model 2: XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.025,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=s_val + fold,
            early_stopping_rounds=50,
            eval_metric="rmse",
            n_jobs=-1,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        seed_oof_xgb[val_idx] = np.clip(xgb_model.predict(X_va), 0, None)
        test_xgb += np.clip(xgb_model.predict(X_te), 0, None) / (n_splits * len(SEEDS))

        # Model 3: CatBoost Regressor
        cat_model = cb.CatBoostRegressor(
            iterations=1500,
            learning_rate=0.025,
            depth=6,
            random_seed=s_val + fold,
            early_stopping_rounds=50,
            eval_metric="RMSE",
            verbose=0,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        seed_oof_cat[val_idx] = np.clip(cat_model.predict(X_va), 0, None)
        test_cat += np.clip(cat_model.predict(X_te), 0, None) / (n_splits * len(SEEDS))

        # Model 4: ExtraTrees Regressor
        et_model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=s_val + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        seed_oof_et[val_idx] = np.clip(et_model.predict(X_va), 0, None)
        test_et += np.clip(et_model.predict(X_te), 0, None) / (n_splits * len(SEEDS))

        # Model 5: Ridge Regression
        ridge_scaler = StandardScaler()
        X_tr_ridge = ridge_scaler.fit_transform(X_tr)
        X_va_ridge = ridge_scaler.transform(X_va)
        X_te_ridge = ridge_scaler.transform(X_te)
        ridge_model = Ridge(alpha=10.0, random_state=s_val + fold)
        ridge_model.fit(X_tr_ridge, y_tr)
        seed_oof_ridge[val_idx] = np.clip(ridge_model.predict(X_va_ridge), 0, None)
        test_ridge += np.clip(ridge_model.predict(X_te_ridge), 0, None) / (n_splits * len(SEEDS))

        # Model 6: PyTorch Tabular ResNet
        nn_oof_fold, nn_test_fold = train_nn_fold(
            X_tr.values, y_tr, X_va.values, y_va, X_te.values, fold_num=s_idx * n_splits + fold
        )
        seed_oof_nn[val_idx] = nn_oof_fold
        test_nn += nn_test_fold / (n_splits * len(SEEDS))

    oof_lgb += seed_oof_lgb / len(SEEDS)
    oof_xgb += seed_oof_xgb / len(SEEDS)
    oof_cat += seed_oof_cat / len(SEEDS)
    oof_et += seed_oof_et / len(SEEDS)
    oof_ridge += seed_oof_ridge / len(SEEDS)
    oof_nn += seed_oof_nn / len(SEEDS)

# 7. SLSQP Optimization for Model Ensembling
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_ridge, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et, test_ridge, test_nn])


def ensemble_objective(weights):
    w = weights / np.sum(weights)
    blend_oof = oof_matrix @ w
    return root_mean_squared_error(y, blend_oof)


init_weights = np.ones(6) / 6.0
bounds = [(0, 1)] * 6
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500},
)

final_weights = opt_res.x / np.sum(opt_res.x)

oof_final = oof_matrix @ final_weights
test_final = np.clip(test_matrix @ final_weights, 0, None)

# 8. Generate Submission File
test_id = test_df["id"] if "id" in test_df.columns else range(n_test)
submission = pd.DataFrame({"id": test_id, "strength": test_final})
submission.to_csv("./submission/submission.csv", index=False)

# 9. Compute and Output Final Validation Score
final_cv_rmse = root_mean_squared_error(y, oof_final)
print(f"Final Validation Score: {final_cv_rmse}")
