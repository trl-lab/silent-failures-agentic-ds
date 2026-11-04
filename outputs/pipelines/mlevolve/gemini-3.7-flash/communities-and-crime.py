import os
import re
from typing import Any, Dict, List, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

# Reproducibility settings
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-specific composite socio-economic, housing,
    and demographic features from Census and LEMAS indicators.
    """
    df = df.copy()

    # Identify LEMAS law enforcement columns and record missingness signatures
    lemas_cols = [
        c
        for c in df.columns
        if c.lower().startswith("lemas") or c.lower().startswith("polic")
    ]
    df["lemas_missing_count"] = df[lemas_cols].isna().sum(axis=1)
    df["lemas_missing_ratio"] = df["lemas_missing_count"] / max(1, len(lemas_cols))
    df["total_missing_count"] = df.isna().sum(axis=1)

    # 1. Concentrated Disadvantage & Poverty Index
    disadv_terms = [
        "PctPopUnderPov",
        "PctUnemployed",
        "PctIlleg",
        "PctWPubAsst",
        "PctLess9thGrade",
    ]
    available_disadv = [c for c in disadv_terms if c in df.columns]
    if available_disadv:
        df["concentrated_disadvantage_idx"] = df[available_disadv].mean(axis=1)
        if "PctFam2Par" in df.columns:
            df["concentrated_disadvantage_idx"] = (
                df["concentrated_disadvantage_idx"] + (1.0 - df["PctFam2Par"])
            ) / 2.0

    # 2. Economic Affluence / Advantage Index
    affluence_terms = [
        "medIncome",
        "medFamInc",
        "perCapInc",
        "PctBSorMore",
        "PctOccupMgmtProf",
        "PctWInvInc",
    ]
    available_affl = [c for c in affluence_terms if c in df.columns]
    if available_affl:
        df["economic_affluence_idx"] = df[available_affl].mean(axis=1)

    # 3. Family Structure & Stability Ratios
    if "TotalPctDiv" in df.columns and "PctKids2Par" in df.columns:
        df["family_instability_idx"] = (
            df["TotalPctDiv"] + (1.0 - df["PctKids2Par"])
        ) / 2.0
    if "PctFam2Par" in df.columns and "PctIlleg" in df.columns:
        df["parental_stability_ratio"] = df["PctFam2Par"] / (df["PctIlleg"] + 1e-4)

    # 4. Youth Demographic Concentration
    youth_terms = ["agePct12t21", "agePct12t29", "agePct16t24"]
    available_youth = [c for c in youth_terms if c in df.columns]
    if available_youth:
        df["youth_demographic_pressure"] = df[available_youth].mean(axis=1)
        if "agePct65up" in df.columns:
            df["youth_to_elderly_ratio"] = df["youth_demographic_pressure"] / (
                df["agePct65up"] + 1e-4
            )

    # 5. Housing Burden & Residential Stability
    housing_terms = [
        "MedRentPctHousInc",
        "MedOwnCostPctInc",
        "PctPersDenseHous",
        "PctHousLess3BR",
    ]
    available_housing = [c for c in housing_terms if c in df.columns]
    if available_housing:
        df["housing_stress_idx"] = df[available_housing].mean(axis=1)
    if "PctVacantBoarded" in df.columns and "HousVacant" in df.columns:
        df["vacant_boarded_intensity"] = df["PctVacantBoarded"] / (
            df["HousVacant"] + 1e-4
        )
    if "PctSameHouse85" in df.columns and "PctSameCity85" in df.columns:
        df["residential_mobility_idx"] = 2.0 - (
            df["PctSameHouse85"] + df["PctSameCity85"]
        )

    # 6. Economic Strain Interactions
    if "PctUnemployed" in df.columns and "PctEmploy" in df.columns:
        df["unemploy_to_employ_ratio"] = df["PctUnemployed"] / (df["PctEmploy"] + 1e-4)
    if "PctPopUnderPov" in df.columns and "medIncome" in df.columns:
        df["poverty_income_interaction"] = df["PctPopUnderPov"] * (
            1.0 - df["medIncome"]
        )

    # 7. Categorical & Identifier Cleanup
    if "communityname" in df.columns:
        df["cleaned_comm_name"] = (
            df["communityname"]
            .astype(str)
            .str.lower()
            .apply(
                lambda x: re.sub(
                    r"(city|town|village|borough|township|cdp)$", "", x
                ).strip()
            )
        )
    else:
        df["cleaned_comm_name"] = "unknown"

    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by converting '?' and string representations of missing values
    to np.nan and ensuring numerical columns are properly typed as float.
    """
    df = df.copy()
    df = df.replace("?", np.nan)
    for col in df.columns:
        if col not in ["communityname"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_cross_validation_folds(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits: int = 5
) -> Dict[str, Any]:
    """
    Transforms raw datasets and builds strictly isolated cross-validation folds.
    """
    target_col = "ViolentCrimesPerPop"
    train_clean = clean_raw_data(train_df)
    test_clean = clean_raw_data(test_df)

    train_eng = engineer_domain_features(train_clean)
    test_eng = engineer_domain_features(test_clean)

    train_eng = train_eng.replace([np.inf, -np.inf], np.nan)
    test_eng = test_eng.replace([np.inf, -np.inf], np.nan)

    ignored_cols = [target_col, "fold", "communityname", "cleaned_comm_name"]
    base_feature_cols = [c for c in train_eng.columns if c not in ignored_cols]

    missing_cols = [
        c
        for c in base_feature_cols
        if train_eng[c].isna().any() or test_eng[c].isna().any()
    ]
    for col in missing_cols:
        train_eng[f"{col}_isna"] = train_eng[col].isna().astype(float)
        test_eng[f"{col}_isna"] = test_eng[col].isna().astype(float)

    indicator_cols = [f"{col}_isna" for col in missing_cols]
    all_numerical_cols = base_feature_cols + indicator_cols

    y_train = train_eng[target_col].values
    binned_y = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_eng, binned_y)):
        fold_train = train_eng.iloc[train_idx].copy()
        fold_val = train_eng.iloc[val_idx].copy()
        fold_test = test_eng.copy()

        # 1. Impute numerical features using training fold median
        imputer = SimpleImputer(strategy="median")
        imputer.fit(fold_train[all_numerical_cols])

        fold_train[all_numerical_cols] = imputer.transform(
            fold_train[all_numerical_cols]
        )
        fold_val[all_numerical_cols] = imputer.transform(fold_val[all_numerical_cols])
        fold_test[all_numerical_cols] = imputer.transform(fold_test[all_numerical_cols])

        # 2. Out-of-fold frequency and target aggregations
        if "state" in fold_train.columns:
            state_freq = fold_train["state"].value_counts(normalize=True).to_dict()
            state_target_mean = fold_train.groupby("state")[target_col].mean().to_dict()
            global_target_mean = fold_train[target_col].mean()

            for df_curr in [fold_train, fold_val, fold_test]:
                df_curr["state_freq"] = df_curr["state"].map(state_freq).fillna(0.0)
                df_curr["state_target_enc"] = (
                    df_curr["state"].map(state_target_mean).fillna(global_target_mean)
                )

        if "cleaned_comm_name" in fold_train.columns:
            comm_freq = fold_train["cleaned_comm_name"].value_counts().to_dict()
            for df_curr in [fold_train, fold_val, fold_test]:
                df_curr["comm_name_freq"] = (
                    df_curr["cleaned_comm_name"].map(comm_freq).fillna(0.0)
                )

        engineered_feature_cols = [
            c for c in fold_train.columns if c not in ignored_cols
        ]

        # 3. Fit standard scalers strictly on training fold
        scaler = RobustScaler()
        scaler.fit(fold_train[engineered_feature_cols])

        X_tr_scaled = pd.DataFrame(
            scaler.transform(fold_train[engineered_feature_cols]),
            columns=engineered_feature_cols,
        )
        X_va_scaled = pd.DataFrame(
            scaler.transform(fold_val[engineered_feature_cols]),
            columns=engineered_feature_cols,
        )
        X_te_scaled = pd.DataFrame(
            scaler.transform(fold_test[engineered_feature_cols]),
            columns=engineered_feature_cols,
        )

        folds_data.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "X_train_raw": fold_train[engineered_feature_cols].reset_index(
                    drop=True
                ),
                "y_train": fold_train[target_col].reset_index(drop=True).values,
                "X_val_raw": fold_val[engineered_feature_cols].reset_index(drop=True),
                "y_val": fold_val[target_col].reset_index(drop=True).values,
                "X_test_raw": fold_test[engineered_feature_cols].reset_index(drop=True),
                "X_train_scaled": X_tr_scaled,
                "X_val_scaled": X_va_scaled,
                "X_test_scaled": X_te_scaled,
                "feature_names": engineered_feature_cols,
            }
        )

    return {
        "folds_data": folds_data,
        "feature_names": folds_data[0]["feature_names"],
        "n_splits": n_splits,
        "y_train_full": y_train,
        "test_len": len(test_df),
    }


class ResidualBlock(nn.Module):
    """
    Residual Block with LayerNorm, SiLU activation, and Dropout for tabular features.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """
    Deep Tabular ResNet architecture designed for continuous regression on tabular socio-economic features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def build_neural_model_and_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.2,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-6
    )
    return model, criterion, optimizer, scheduler


def build_tree_and_linear_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "lgbm": LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_samples=15,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1200,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=2.0,
            early_stopping_rounds=50,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=4.0,
            subsample=0.8,
            random_seed=random_state,
            thread_count=-1,
            verbose=False,
        ),
        "ridge": Ridge(alpha=3.0, random_state=random_state),
        "elastic_net": ElasticNet(
            alpha=0.01, l1_ratio=0.3, max_iter=2000, random_state=random_state
        ),
    }


def train_neural_network(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 80,
    patience: int = 15,
) -> nn.Module:
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    model.to(device)

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(y_batch)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    return model


def predict_neural_network(model: nn.Module, X_data: pd.DataFrame) -> np.ndarray:
    model.to(device)
    model.eval()
    tensor_x = torch.tensor(X_data.values, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    preds_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds_list.append(pred.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # 2. Build isolated CV folds with feature transformations
    pipeline_data = prepare_cross_validation_folds(train_df, test_df, n_splits=5)
    folds_data = pipeline_data["folds_data"]
    y_train_full = pipeline_data["y_train_full"]
    num_samples = len(y_train_full)
    num_test = pipeline_data["test_len"]
    n_splits = pipeline_data["n_splits"]

    model_names = ["lgbm", "xgb", "catboost", "ridge", "elastic_net", "neural_net"]
    oof_predictions = {name: np.zeros(num_samples) for name in model_names}
    test_predictions = {name: np.zeros(num_test) for name in model_names}

    # 3. Model Training across CV folds
    for fold_info in folds_data:
        fold_idx = fold_info["fold"]
        val_idx = fold_info["val_idx"]
        X_tr_raw = fold_info["X_train_raw"]
        y_tr = fold_info["y_train"]
        X_va_raw = fold_info["X_val_raw"]
        y_va = fold_info["y_val"]
        X_te_raw = fold_info["X_test_raw"]

        X_tr_sc = fold_info["X_train_scaled"]
        X_va_sc = fold_info["X_val_scaled"]
        X_te_sc = fold_info["X_test_scaled"]

        tree_models = build_tree_and_linear_models(random_state=42 + fold_idx)

        # 3.1 LightGBM
        lgbm = tree_models["lgbm"]
        lgbm.fit(
            X_tr_raw,
            y_tr,
            eval_set=[(X_va_raw, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_predictions["lgbm"][val_idx] = lgbm.predict(X_va_raw)
        test_predictions["lgbm"] += lgbm.predict(X_te_raw) / n_splits

        # 3.2 XGBoost
        xgb = tree_models["xgb"]
        xgb.fit(X_tr_raw, y_tr, eval_set=[(X_va_raw, y_va)], verbose=False)
        oof_predictions["xgb"][val_idx] = xgb.predict(X_va_raw)
        test_predictions["xgb"] += xgb.predict(X_te_raw) / n_splits

        # 3.3 CatBoost
        cb = tree_models["catboost"]
        cb.fit(
            X_tr_raw,
            y_tr,
            eval_set=(X_va_raw, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_predictions["catboost"][val_idx] = cb.predict(X_va_raw)
        test_predictions["catboost"] += cb.predict(X_te_raw) / n_splits

        # 3.4 Ridge Regression
        ridge = tree_models["ridge"]
        ridge.fit(X_tr_sc, y_tr)
        oof_predictions["ridge"][val_idx] = ridge.predict(X_va_sc)
        test_predictions["ridge"] += ridge.predict(X_te_sc) / n_splits

        # 3.5 ElasticNet
        enet = tree_models["elastic_net"]
        enet.fit(X_tr_sc, y_tr)
        oof_predictions["elastic_net"][val_idx] = enet.predict(X_va_sc)
        test_predictions["elastic_net"] += enet.predict(X_te_sc) / n_splits

        # 3.6 TabularResNet PyTorch Model
        input_dim = X_tr_sc.shape[1]
        nn_model, criterion, optimizer, scheduler = build_neural_model_and_optimizer(
            input_dim=input_dim,
            lr=1e-3,
            weight_decay=1e-4,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.2,
        )

        train_ds = TensorDataset(
            torch.tensor(X_tr_sc.values, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va_sc.values, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        trained_nn = train_neural_network(
            model=nn_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=80,
            patience=15,
        )

        oof_predictions["neural_net"][val_idx] = predict_neural_network(
            trained_nn, X_va_sc
        )
        test_predictions["neural_net"] += (
            predict_neural_network(trained_nn, X_te_sc) / n_splits
        )

    # 4. Ensemble Weight Optimization (SLSQP)
    num_models = len(model_names)
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])
    test_matrix = np.column_stack([test_predictions[name] for name in model_names])

    def rmse_objective(weights: np.ndarray) -> float:
        blended = oof_matrix @ weights
        clipped = np.clip(blended, 0.0, 1.0)
        return root_mean_squared_error(y_train_full, clipped)

    initial_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        rmse_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = np.maximum(0.0, opt_result.x)
    optimal_weights /= np.sum(optimal_weights)

    # 5. Evaluate final validation score and compute test predictions
    final_oof_preds = np.clip(oof_matrix @ optimal_weights, 0.0, 1.0)
    final_test_preds = np.clip(test_matrix @ optimal_weights, 0.0, 1.0)
    final_validation_score = float(
        root_mean_squared_error(y_train_full, final_oof_preds)
    )

    # 6. Generate submission file
    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_preds), dtype=int),
            "ViolentCrimesPerPop": final_test_preds,
        }
    )

    assert (
        len(submission_df) == num_test
    ), f"Expected {num_test} rows, found {len(submission_df)}"
    assert list(submission_df.columns) == [
        "id",
        "ViolentCrimesPerPop",
    ], f"Invalid columns: {submission_df.columns}"
    assert (
        not submission_df["ViolentCrimesPerPop"].isna().any()
    ), "Submission contains NaN values"

    submission_df.to_csv(submission_path, index=False)

    # Output required final score line
    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()
