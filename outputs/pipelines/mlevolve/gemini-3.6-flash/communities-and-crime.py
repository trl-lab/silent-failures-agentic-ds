import os
import random
import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


# Reproducibility helper
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

if "id" not in train_df.columns:
    train_df["id"] = train_df.index
if "id" not in test_df.columns:
    test_df["id"] = test_df.index

target_col = "ViolentCrimesPerPop"
drop_cols = ["communityname", "fold", "id", target_col]

# Clean object string columns ('?' values)
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")

feature_cols = [
    c for c in train_df.columns if c not in drop_cols and c in test_df.columns
]


# 2. Domain-Specific Feature Engineering
def create_domain_features(df, high_missing_cols):
    data = df.copy()

    # Missingness Indicators for sparse police/LEMAS features
    for col in high_missing_cols:
        if col in data.columns:
            data[f"{col}_is_na"] = data[col].isna().astype(float)

    # Concentrated Disadvantage Index
    poverty = data.get("PctPopUnderPov", 0)
    unemp = data.get("PctUnemployed", 0)
    pub_asst = data.get("pctWPubAsst", 0)
    no_hs = data.get("PctNotHSGrad", 0)
    med_inc = data.get("medIncome", 0)
    data["DisadvantageIndex"] = (poverty + unemp + pub_asst + no_hs) - med_inc

    # Family Stability Index
    two_par = data.get("PctFam2Par", 0)
    kids_2par = data.get("PctKids2Par", 0)
    illeg = data.get("PctIlleg", 0)
    divorced = data.get("TotalPctDiv", 0)
    data["FamilyStabilityIndex"] = (two_par + kids_2par) - (illeg + divorced)

    # Housing Distress Index
    vacant_boarded = data.get("PctVacantBoarded", 0)
    vacant_6mos = data.get("PctVacMore6Mos", 0)
    data["HousingDistress"] = vacant_boarded * vacant_6mos

    # Economic Disparity & Non-Linear Domain Ratios
    per_cap = data.get("perCapInc", 0)
    data["IncomeDisparityRatio"] = per_cap / (med_inc + 1e-5)

    white_cap = data.get("whitePerCap", 0)
    black_cap = data.get("blackPerCap", 0)
    data["RacialIncomeGap"] = white_cap / (black_cap + 1e-5)

    pop = data.get("population", 0)
    num_pov = data.get("NumUnderPov", 0)
    num_vacant = data.get("HousVacant", 0)
    num_illeg = data.get("NumIlleg", 0)

    data["PovertyRatio"] = num_pov / (pop + 1e-5)
    data["VacancyRatio"] = num_vacant / (pop + 1e-5)
    data["IllegRatio"] = num_illeg / (pop + 1e-5)
    data["PovertyToIncomeRatio"] = poverty / (med_inc + 1e-5)
    data["FamilyInstabilityRatio"] = illeg / (two_par + 1e-5)

    # Logarithmic transformations for skewed count variables
    skewed_counts = [
        "population", "NumUnderPov", "NumIlleg", "HousVacant",
        "numbUrban", "numKids2Par", "numServices"
    ]
    for c in skewed_counts:
        if c in data.columns:
            data[f"{c}_log1p"] = np.log1p(np.maximum(0, data[c].fillna(0)))

    # Interaction Signals
    black_pct = data.get("racepctblack", 0)
    data["RacePovertyInteraction"] = black_pct * poverty
    data["FamilyUnemployInteraction"] = (1.0 - two_par) * unemp

    # Housing Cost Burden & Domain Ratios
    med_rent = data.get("MedRent", 0)
    data["HousingCostBurden"] = med_rent / (med_inc + 1e-5)

    offic_pop = data.get("officPerPop", 0)
    if "officPerPop" not in data.columns:
        offic_pop = data.get("PolicPerPop", 0)
    data["PoliceToPopRatio"] = offic_pop / (pop + 1e-5)

    data["FamEconVulnerability"] = (1.0 - two_par) * poverty

    # High-impact socio-economic interaction features
    hi_quart = data.get("OwnOccHiQuart", 0)
    low_quart = data.get("OwnOccLowQuart", 0)
    med_val = data.get("OwnOccMedVal", 0)
    data["HousingWealthDispersion"] = (hi_quart - low_quart) / (med_val + 1e-5)

    shelter = data.get("NumInShelters", 0)
    street = data.get("NumStreet", 0)
    data["HomelessnessDensity"] = (shelter + street) / (pop + 1e-5)

    age_12_21 = data.get("agePct12t21", 0)
    data["YouthRiskIndex"] = age_12_21 / (two_par + 1e-5)

    return data


def add_state_aggregates(tr_df, te_df):
    tr_df = tr_df.copy()
    te_df = te_df.copy()
    state_cols = ["medIncome", "PctPopUnderPov", "PctFam2Par", "PctUnemployed"]
    if "state" in tr_df.columns:
        for col in state_cols:
            if col in tr_df.columns:
                st_stats = tr_df.groupby("state")[col].agg(["mean", "std"]).reset_index()
                st_stats.columns = ["state", f"{col}_state_mean", f"{col}_state_std"]
                st_stats[f"{col}_state_std"] = st_stats[f"{col}_state_std"].fillna(0)

                tr_df = tr_df.merge(st_stats, on="state", how="left")
                tr_df[f"{col}_state_diff"] = tr_df[col] - tr_df[f"{col}_state_mean"]

                te_df = te_df.merge(st_stats, on="state", how="left")
                te_df[f"{col}_state_mean"] = te_df[f"{col}_state_mean"].fillna(tr_df[f"{col}_state_mean"].mean())
                te_df[f"{col}_state_std"] = te_df[f"{col}_state_std"].fillna(0)
                te_df[f"{col}_state_diff"] = te_df[col] - te_df[f"{col}_state_mean"]
    return tr_df, te_df


train_df, test_df = add_state_aggregates(train_df, test_df)

feature_cols = [
    c for c in train_df.columns if c not in drop_cols and c in test_df.columns
]

missing_ratios = train_df[feature_cols].isna().mean()
high_missing_cols = missing_ratios[missing_ratios > 0.20].index.tolist()

train_fe = create_domain_features(train_df, high_missing_cols)
test_fe = create_domain_features(test_df, high_missing_cols)

updated_features = [
    c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns
]


# 3. Model Architecture Definitions
class GatedLinearUnit(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.linear(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.glu(x)
        out = self.dropout(out)
        return self.norm(residual + out)


class GatedResidualTabularNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Bounded strictly to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 4. Out-of-Fold Cross Validation Setup
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
target_bins = pd.qcut(train_fe[target_col], q=10, labels=False, duplicates="drop")

train_fe["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(skf.split(train_fe, target_bins)):
    train_fe.loc[val_idx, "fold"] = fold_idx

num_samples = len(train_fe)
num_test = len(test_fe)

oof_ridge = np.zeros(num_samples)
oof_huber = np.zeros(num_samples)
oof_cb = np.zeros(num_samples)
oof_lgb = np.zeros(num_samples)
oof_xgb = np.zeros(num_samples)
oof_et = np.zeros(num_samples)
oof_nn = np.zeros(num_samples)

test_preds_ridge = np.zeros(num_test)
test_preds_huber = np.zeros(num_test)
test_preds_cb = np.zeros(num_test)
test_preds_lgb = np.zeros(num_test)
test_preds_xgb = np.zeros(num_test)
test_preds_et = np.zeros(num_test)
test_preds_nn = np.zeros(num_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Cross-Validation Training Loop
for fold in range(num_folds):
    tr_mask = train_fe["fold"] != fold
    val_mask = train_fe["fold"] == fold

    val_indices = train_fe[val_mask].index.values

    # Fold-isolated target encoding for state with m-estimate smoothing
    if "state" in train_fe.columns:
        global_mean_tr = train_fe.loc[tr_mask, target_col].mean()
        m_param = 10.0
        state_stats = train_fe.loc[tr_mask].groupby("state")[target_col].agg(["count", "mean"])
        state_te_map = (state_stats["count"] * state_stats["mean"] + m_param * global_mean_tr) / (
            state_stats["count"] + m_param
        )

        tr_state_te = (
            train_fe.loc[tr_mask, "state"]
            .map(state_te_map)
            .fillna(global_mean_tr)
            .values.reshape(-1, 1)
        )
        val_state_te = (
            train_fe.loc[val_mask, "state"]
            .map(state_te_map)
            .fillna(global_mean_tr)
            .values.reshape(-1, 1)
        )
        test_state_te = (
            test_fe["state"].map(state_te_map).fillna(global_mean_tr).values.reshape(-1, 1)
        )
    else:
        tr_state_te = np.zeros((tr_mask.sum(), 1))
        val_state_te = np.zeros((val_mask.sum(), 1))
        test_state_te = np.zeros((len(test_fe), 1))

    X_tr_raw = np.hstack([train_fe.loc[tr_mask, updated_features].values, tr_state_te])
    y_tr = train_fe.loc[tr_mask, target_col].values

    X_val_raw = np.hstack([train_fe.loc[val_mask, updated_features].values, val_state_te])
    y_val = train_fe.loc[val_mask, target_col].values

    X_test_raw = np.hstack([test_fe[updated_features].values, test_state_te])

    # Strict Fold-level Imputation, Scaling, PCA & Quantile Transformation
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_tr = scaler.fit_transform(X_tr_imp)
    X_val = scaler.transform(X_val_imp)
    X_test = scaler.transform(X_test_imp)

    # PCA component extraction fit strictly on training fold
    pca = PCA(n_components=5, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    X_tr_ext = np.hstack([X_tr, X_tr_pca])
    X_val_ext = np.hstack([X_val, X_val_pca])
    X_test_ext = np.hstack([X_test, X_test_pca])

    # Quantile transformation for Linear models
    qt = QuantileTransformer(n_quantiles=min(100, len(X_tr)), output_distribution="normal", random_state=42)
    X_tr_qt = qt.fit_transform(X_tr_imp)
    X_val_qt = qt.transform(X_val_imp)
    X_test_qt = qt.transform(X_test_imp)

    # --- Model 1: Ridge Regressor ---
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_tr_qt, y_tr)

    val_ridge = np.clip(ridge_model.predict(X_val_qt), 0.0, 1.0)
    test_ridge = np.clip(ridge_model.predict(X_test_qt), 0.0, 1.0)

    oof_ridge[val_indices] = val_ridge
    test_preds_ridge += test_ridge / num_folds

    # --- Model 2: Huber Regressor ---
    huber_model = HuberRegressor(max_iter=1000, alpha=1.0)
    huber_model.fit(X_tr_qt, y_tr)

    val_huber = np.clip(huber_model.predict(X_val_qt), 0.0, 1.0)
    test_huber = np.clip(huber_model.predict(X_test_qt), 0.0, 1.0)

    oof_huber[val_indices] = val_huber
    test_preds_huber += test_huber / num_folds

    # --- Model 3: CatBoost (raw unimputed features) ---
    cb_model = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.025,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.7,
        rsm=0.6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr_raw, y_tr, eval_set=(X_val_raw, y_val), early_stopping_rounds=50, verbose=0
    )

    val_cb = np.clip(cb_model.predict(X_val_raw), 0.0, 1.0)
    test_cb = np.clip(cb_model.predict(X_test_raw), 0.0, 1.0)

    oof_cb[val_indices] = val_cb
    test_preds_cb += test_cb / num_folds

    # --- Model 4: LightGBM (raw unimputed features) ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_val_raw, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_lgb = np.clip(lgb_model.predict(X_val_raw), 0.0, 1.0)
    test_lgb = np.clip(lgb_model.predict(X_test_raw), 0.0, 1.0)

    oof_lgb[val_indices] = val_lgb
    test_preds_lgb += test_lgb / num_folds

    # --- Model 5: XGBoost (raw unimputed features) ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.025,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_val_raw, y_val)],
        verbose=False,
    )

    val_xgb = np.clip(xgb_model.predict(X_val_raw), 0.0, 1.0)
    test_xgb = np.clip(xgb_model.predict(X_test_raw), 0.0, 1.0)

    oof_xgb[val_indices] = val_xgb
    test_preds_xgb += test_xgb / num_folds

    # --- Model 6: ExtraTrees ---
    et_model = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr_ext, y_tr)

    val_et = np.clip(et_model.predict(X_val_ext), 0.0, 1.0)
    test_et = np.clip(et_model.predict(X_test_ext), 0.0, 1.0)

    oof_et[val_indices] = val_et
    test_preds_et += test_et / num_folds

    # --- Model 7: PyTorch GRTN ---
    train_dataset = TensorDataset(
        torch.tensor(X_tr_ext, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_ext, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    nn_model = GatedResidualTabularNet(
        input_dim=X_tr_ext.shape[1], hidden_dim=128, num_blocks=3, dropout_rate=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_weights = None

    for epoch in range(100):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = nn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = nn_model(batch_x)
                val_loss += criterion(out, batch_y).item() * len(batch_y)
                val_count += len(batch_y)

        val_loss /= val_count
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = nn_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        nn_model.load_state_dict(best_weights)

    nn_model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_ext, dtype=torch.float32).to(device)
        val_nn = nn_model(val_tensor).cpu().numpy()

        test_tensor = torch.tensor(X_test_ext, dtype=torch.float32).to(device)
        test_nn = nn_model(test_tensor).cpu().numpy()

    val_nn = np.clip(val_nn, 0.0, 1.0)
    test_nn = np.clip(test_nn, 0.0, 1.0)

    oof_nn[val_indices] = val_nn
    test_preds_nn += test_nn / num_folds

# 6. Constrained Non-Negative Meta-Learner Stacking
y_true = train_fe[target_col].values
oof_matrix = np.column_stack(
    [oof_ridge, oof_huber, oof_cb, oof_lgb, oof_xgb, oof_et, oof_nn]
)
test_matrix = np.column_stack(
    [
        test_preds_ridge,
        test_preds_huber,
        test_preds_cb,
        test_preds_lgb,
        test_preds_xgb,
        test_preds_et,
        test_preds_nn,
    ]
)


def optimize_weights(OOF, y):
    num_models = OOF.shape[1]

    def loss_func(weights):
        pred = OOF @ weights
        return np.sqrt(mean_squared_error(y, pred))

    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x


weights = optimize_weights(oof_matrix, y_true)

# Evaluate Final Validation Score
final_oof_blend = np.clip(oof_matrix @ weights, 0.0, 1.0)
final_score = np.sqrt(mean_squared_error(y_true, final_oof_blend))

# 7. Generate Submission
final_test_preds = np.clip(test_matrix @ weights, 0.0, 1.0)

submission_df = pd.DataFrame(
    {"id": test_fe["id"].astype(int), "ViolentCrimesPerPop": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")