import copy
import json
import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42):
    """Set random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories():
    """Ensure working and submission directories exist."""
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)


# ==============================================================================
# 1. Data Loading & Domain-Specific Feature Engineering
# ==============================================================================
def load_raw_data():
    """Load raw train and test data from parquet files."""
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
    return train_df, test_df


def identify_patient_baselines(df, is_train=True):
    """
    Extract patient baseline characteristics:
    Finds visit closest to week 0 for each patient where baseline metrics were recorded.
    """
    patient_col = "Patient" if "Patient" in df.columns else None

    if patient_col is None:
        df_base = df.copy()
        df_base["baseline_Week"] = df_base.get("Weeks", 0)
        df_base["baseline_Percent"] = df_base.get("Percent", 100.0)
        df_base["baseline_Age"] = df_base.get("Age", 65.0)
        if "FVC" in df_base.columns:
            df_base["baseline_FVC"] = df_base["FVC"]
        return df_base

    df_copy = df.copy()
    if "Weeks" in df_copy.columns:
        df_copy["abs_week"] = df_copy["Weeks"].abs()
        sorted_df = df_copy.sort_values(by=[patient_col, "abs_week", "Weeks"])
    else:
        sorted_df = df_copy

    baseline_records = {}
    for pid, group in sorted_df.groupby(patient_col):
        base_row = group.iloc[0]
        base_dict = {
            "baseline_Week": base_row["Weeks"] if "Weeks" in base_row else 0,
            "baseline_Percent": base_row["Percent"] if "Percent" in base_row else 100.0,
            "baseline_Age": base_row["Age"] if "Age" in base_row else 65.0,
            "baseline_Sex": base_row["Sex"] if "Sex" in base_row else "Male",
            "baseline_SmokingStatus": (
                base_row["SmokingStatus"]
                if "SmokingStatus" in base_row
                else "Ex-smoker"
            ),
        }
        if "FVC" in base_row and pd.notnull(base_row["FVC"]):
            base_dict["baseline_FVC"] = base_row["FVC"]
        baseline_records[pid] = base_dict

    return baseline_records


def engineer_features(df, baseline_records, is_train=True):
    """
    Engineer domain-specific clinical, temporal, and interaction features.
    """
    data = df.copy()
    patient_col = "Patient" if "Patient" in data.columns else None

    # Map baseline features per patient
    if patient_col is not None and baseline_records:
        data["baseline_Week"] = data[patient_col].map(
            lambda x: baseline_records.get(x, {}).get("baseline_Week", 0)
        )
        data["baseline_Percent"] = data[patient_col].map(
            lambda x: baseline_records.get(x, {}).get("baseline_Percent", 100.0)
        )
        data["baseline_Age"] = data[patient_col].map(
            lambda x: baseline_records.get(x, {}).get("baseline_Age", 65.0)
        )
        data["baseline_FVC"] = data[patient_col].map(
            lambda x: baseline_records.get(x, {}).get("baseline_FVC", np.nan)
        )
    else:
        if "baseline_Week" not in data.columns:
            data["baseline_Week"] = data.get("Weeks", 0)
        if "baseline_Percent" not in data.columns:
            data["baseline_Percent"] = data.get("Percent", 100.0)
        if "baseline_Age" not in data.columns:
            data["baseline_Age"] = data.get("Age", 65.0)
        if "baseline_FVC" not in data.columns and "FVC" in data.columns:
            data["baseline_FVC"] = data["FVC"]

    if "baseline_FVC" in data.columns and data["baseline_FVC"].isnull().any():
        if "FVC" in data.columns:
            data["baseline_FVC"] = data["baseline_FVC"].fillna(data["FVC"])
        male_est = np.where(
            data.get("Sex", pd.Series("male", index=data.index)).astype(str).str.lower() == "male",
            3800.0,
            2800.0,
        ) * (data["baseline_Percent"].clip(lower=1.0) / 100.0)
        data["baseline_FVC"] = data["baseline_FVC"].fillna(pd.Series(male_est, index=data.index)).fillna(2600.0)

    # Calculate target delta_FVC if FVC is available
    if "FVC" in data.columns:
        data["delta_FVC"] = data["FVC"] - data["baseline_FVC"]

    # 1. Temporal / Trajectory Features
    weeks = data["Weeks"] if "Weeks" in data.columns else pd.Series(0, index=data.index)
    base_weeks = data["baseline_Week"]
    data["delta_week"] = weeks - base_weeks
    data["abs_delta_week"] = data["delta_week"].abs()
    data["delta_week_sq"] = (data["delta_week"] / 10.0) ** 2
    data["delta_week_sign"] = np.sign(data["delta_week"])
    data["delta_week_log"] = np.sign(data["delta_week"]) * np.log1p(
        np.abs(data["delta_week"])
    )
    data["is_baseline_visit"] = (data["delta_week"] == 0).astype(int)

    # 2. Demographic & Physiological Encodings
    sex_series = (
        data["Sex"].astype(str).str.lower()
        if "Sex" in data.columns
        else pd.Series("male", index=data.index)
    )
    data["is_male"] = (sex_series == "male").astype(float)

    smoke_series = (
        data["SmokingStatus"].astype(str).str.lower()
        if "SmokingStatus" in data.columns
        else pd.Series("ex-smoker", index=data.index)
    )
    data["smoke_never"] = (smoke_series.str.contains("never")).astype(float)
    data["smoke_ex"] = (smoke_series.str.contains("ex")).astype(float)
    data["smoke_current"] = (
        smoke_series.str.contains("currently") | (smoke_series == "current")
    ).astype(float)
    data["smoke_severity"] = data["smoke_ex"] * 1.0 + data["smoke_current"] * 2.0

    age = data["Age"] if "Age" in data.columns else pd.Series(65.0, index=data.index)
    data["age_decade"] = age / 10.0
    data["age_squared"] = (age / 10.0) ** 2

    # Theoretical Normal FVC: Percent = (FVC / Normal_FVC) * 100
    base_percent = data["baseline_Percent"].clip(lower=1.0)
    if "baseline_FVC" in data.columns:
        data["estimated_normal_fvc"] = (data["baseline_FVC"] / base_percent) * 100.0
        data["baseline_fvc_per_age"] = data["baseline_FVC"] / age.clip(lower=1.0)
    else:
        data["estimated_normal_fvc"] = 3500.0
        data["baseline_fvc_per_age"] = 3500.0 / age.clip(lower=1.0)

    if "Percent" in data.columns:
        percent = data["Percent"]
        data["percent_to_baseline_ratio"] = percent / base_percent
        data["percent_delta"] = percent - base_percent
    else:
        percent = base_percent
        data["percent_to_baseline_ratio"] = 1.0
        data["percent_delta"] = 0.0

    # 3. Trajectory & Physiological Interactions
    data["time_x_baseline_percent"] = (data["delta_week"] / 10.0) * (
        base_percent / 100.0
    )
    data["time_x_age"] = (data["delta_week"] / 10.0) * (age / 10.0)
    data["time_x_smoke"] = (data["delta_week"] / 10.0) * data["smoke_severity"]
    data["time_x_male"] = (data["delta_week"] / 10.0) * data["is_male"]

    if "baseline_FVC" in data.columns:
        data["time_x_baseline_fvc"] = (data["delta_week"] / 10.0) * (
            data["baseline_FVC"] / 1000.0
        )
        data["expected_baseline_decay"] = data["baseline_FVC"] * (
            1.0 - 0.0025 * data["delta_week"]
        )

    return data


def select_feature_columns(train_features):
    """Identify numeric feature columns excluding targets and non-numeric identifiers."""
    exclude_cols = {
        "id",
        "Patient",
        "FVC",
        "delta_FVC",
        "fold",
        "abs_week",
        "Sex",
        "SmokingStatus",
    }
    feature_cols = [c for c in train_features.columns if c not in exclude_cols]
    return sorted(feature_cols)


# ==============================================================================
# 2. Neural Architecture Definition
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual Tabular block with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super(TabularResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class OSICClinicalMLP(nn.Module):
    """
    Deep Tabular ResNet with LayerNorm, GELU, and residual skip connections
    designed for regularized clinical trajectory regression.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super(OSICClinicalMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


# ==============================================================================
# 3. Model Training Functions (MAE / L1 Aligned)
# ==============================================================================
def train_lgb_model(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM regressor using regression_l1 objective with early stopping."""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 12,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

    val_preds = model.predict(X_val)
    return model, val_preds


def train_xgb_model(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost regressor with absolute error objective and early stopping."""
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 1200,
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_preds = model.predict(X_val)
    return model, val_preds


def train_hgb_model(X_train, y_train, X_val, y_val, seed=42):
    """Train HistGradientBoosting regressor with absolute error loss."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.5,
        early_stopping=True,
        n_iter_no_change=40,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    return model, val_preds


def train_mlp_model(
    X_train_np, y_train_np, X_val_np, y_val_np, epochs=150, lr=1e-3, seed=42, fold=0
):
    """
    Train PyTorch Tabular ResNet with L1Loss, AdamW, and CosineAnnealing.
    Standardizes both features and regression targets strictly on the fold training split.
    """
    set_seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature and Target standardization
    scaler_x = StandardScaler()
    X_tr_scaled = scaler_x.fit_transform(X_train_np)
    X_v_scaled = scaler_x.transform(X_val_np)

    scaler_y = StandardScaler()
    y_tr_scaled = scaler_y.fit_transform(y_train_np.reshape(-1, 1))

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_scaled, dtype=torch.float32),
    )
    val_tensor = torch.tensor(X_v_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=False
    )

    model = OSICClinicalMLP(input_dim=X_tr_scaled.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_mae = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out_scaled = model(val_tensor).cpu().numpy()
            val_out = scaler_y.inverse_transform(val_out_scaled).ravel()
            val_mae = mean_absolute_error(y_val_np, val_out)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        best_val_scaled = model(val_tensor).cpu().numpy()
        best_val_preds = scaler_y.inverse_transform(best_val_scaled).ravel()

    return model, scaler_x, scaler_y, best_val_preds


# ==============================================================================
# 4. Main Execution Pipeline
# ==============================================================================
def main():
    set_seed(42)
    create_directories()

    # Step 1: Load raw data
    train_raw, test_raw = load_raw_data()

    # Step 2: Extract Patient-level baselines
    train_baselines = identify_patient_baselines(train_raw, is_train=True)
    test_baselines = identify_patient_baselines(test_raw, is_train=False)
    combined_baselines = {**train_baselines, **test_baselines}

    # Step 3: Feature Engineering
    train_processed = engineer_features(train_raw, combined_baselines, is_train=True)
    test_processed = engineer_features(test_raw, combined_baselines, is_train=False)

    feature_cols = select_feature_columns(train_processed)
    print(f"Engineered {len(feature_cols)} features: {feature_cols}")

    # Step 4: Strict Patient Grouped 5-Fold Cross-Validation
    n_splits = 5
    train_processed["fold"] = -1

    if "Patient" in train_processed.columns:
        gkf = GroupKFold(n_splits=n_splits)
        groups = train_processed["Patient"].values
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(train_processed, groups=groups)
        ):
            train_processed.loc[val_idx, "fold"] = fold
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
            train_processed.loc[val_idx, "fold"] = fold

    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0.0

    # Step 5: Save feature metadata and caches
    metadata = {
        "feature_cols": feature_cols,
        "n_splits": n_splits,
        "n_train_samples": len(train_processed),
        "n_test_samples": len(test_processed),
        "target_col": "FVC",
    }
    with open("./working/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 6: Multi-Model 5-Fold Out-of-Fold Training and Evaluation
    oof_lgb = np.zeros(len(train_processed))
    oof_xgb = np.zeros(len(train_processed))
    oof_hgb = np.zeros(len(train_processed))
    oof_mlp = np.zeros(len(train_processed))

    test_preds_lgb = np.zeros(len(test_processed))
    test_preds_xgb = np.zeros(len(test_processed))
    test_preds_hgb = np.zeros(len(test_processed))
    test_preds_mlp = np.zeros(len(test_processed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true_all = train_processed["FVC"].values

    for fold in range(n_splits):
        train_idx = np.where(train_processed["fold"] != fold)[0]
        val_idx = np.where(train_processed["fold"] == fold)[0]

        X_tr = train_processed.iloc[train_idx][feature_cols].copy()
        y_tr = train_processed.iloc[train_idx]["FVC"].values

        X_va = train_processed.iloc[val_idx][feature_cols].copy()
        y_va = train_processed.iloc[val_idx]["FVC"].values

        X_te = test_processed[feature_cols].copy()

        # Fold-specific leak-free median imputation
        fold_medians = X_tr.median()
        X_tr = X_tr.fillna(fold_medians)
        X_va = X_va.fillna(fold_medians)
        X_te = X_te.fillna(fold_medians)

        # 1. LightGBM
        lgb_mod, lgb_val_p = train_lgb_model(
            X_tr, y_tr, X_va, y_va, seed=42 + fold * 10
        )
        oof_lgb[val_idx] = lgb_val_p
        test_preds_lgb += lgb_mod.predict(X_te) / n_splits

        # 2. XGBoost
        xgb_mod, xgb_val_p = train_xgb_model(
            X_tr, y_tr, X_va, y_va, seed=42 + fold * 10
        )
        oof_xgb[val_idx] = xgb_val_p
        test_preds_xgb += xgb_mod.predict(X_te) / n_splits

        # 3. HistGradientBoosting
        hgb_mod, hgb_val_p = train_hgb_model(
            X_tr, y_tr, X_va, y_va, seed=42 + fold * 10
        )
        oof_hgb[val_idx] = hgb_val_p
        test_preds_hgb += hgb_mod.predict(X_te) / n_splits

        # 4. PyTorch Tabular ResNet
        mlp_mod, mlp_sc_x, mlp_sc_y, mlp_val_p = train_mlp_model(
            X_tr.values,
            y_tr,
            X_va.values,
            y_va,
            epochs=150,
            lr=1e-3,
            seed=42,
            fold=fold,
        )
        oof_mlp[val_idx] = mlp_val_p

        mlp_mod.eval()
        with torch.no_grad():
            X_te_scaled = mlp_sc_x.transform(X_te.values)
            test_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            mlp_te_scaled = mlp_mod(test_tensor).cpu().numpy()
            mlp_te_preds = mlp_sc_y.inverse_transform(mlp_te_scaled).ravel()
            test_preds_mlp += mlp_te_preds / n_splits

        print(
            f"Fold {fold} MAE -> LGB: {mean_absolute_error(y_va, lgb_val_p):.2f}, "
            f"XGB: {mean_absolute_error(y_va, xgb_val_p):.2f}, "
            f"HGB: {mean_absolute_error(y_va, hgb_val_p):.2f}, "
            f"MLP: {mean_absolute_error(y_va, mlp_val_p):.2f}"
        )

    # Step 7: Optimization of Ensemble Weights on Out-of-Fold Matrix
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_hgb, oof_mlp])
    print(f"Overall OOF LightGBM MAE: {mean_absolute_error(y_true_all, oof_lgb):.4f}")
    print(f"Overall OOF XGBoost MAE:  {mean_absolute_error(y_true_all, oof_xgb):.4f}")
    print(f"Overall OOF HistGB MAE:   {mean_absolute_error(y_true_all, oof_hgb):.4f}")
    print(f"Overall OOF PyTorch MAE:  {mean_absolute_error(y_true_all, oof_mlp):.4f}")

    def loss_func(weights):
        w = weights / np.sum(weights)
        pred = oof_matrix @ w
        return mean_absolute_error(y_true_all, pred)

    init_weights = np.ones(4) / 4.0
    bounds = [(0.0, 1.0) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
    opt_res = minimize(
        loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )
    best_weights = opt_res.x / np.sum(opt_res.x)
    print(f"Optimized Ensemble Weights [LGB, XGB, HGB, MLP]: {np.round(best_weights, 4)}")

    oof_ensemble_preds = oof_matrix @ best_weights
    final_val_mae = float(mean_absolute_error(y_true_all, oof_ensemble_preds))
    print(f"Final 5-Fold OOF Ensemble MAE: {final_val_mae:.4f}")

    # Step 8: Final Test Inference Blending
    test_preds_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_hgb, test_preds_mlp]
    )
    final_test_preds = test_preds_matrix @ best_weights

    # Step 9: Format and Save Submission File
    if "id" in test_raw.columns:
        test_ids = test_raw["id"].values
    else:
        test_ids = np.arange(len(final_test_preds))

    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "FVC": final_test_preds,
        }
    )

    sub_path = "./submission/submission.csv"
    submission_df.to_csv(sub_path, index=False)
    print(
        f"Saved submission to {sub_path} with shape {submission_df.shape} and head:\n{submission_df.head(3)}"
    )

    # Step 10: Print Final Validation Score
    print(f"Final Validation Score: {final_val_mae}")


if __name__ == "__main__":
    main()