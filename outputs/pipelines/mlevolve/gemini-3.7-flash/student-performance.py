import os
import random
import warnings
import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =========================================================================
# 0. Reproducibility and System Setup
# =========================================================================


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Feature Engineering and Data Preprocessing
# =========================================================================


def load_and_preprocess_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "G3"
    y_train = train_df[target_col].values.astype(np.float32)

    df_train_feats = train_df.drop(columns=[target_col]).copy()
    df_test_feats = test_df.copy()
    if target_col in df_test_feats.columns:
        df_test_feats = df_test_feats.drop(columns=[target_col])

    n_train = len(df_train_feats)
    combined = pd.concat([df_train_feats, df_test_feats], axis=0).reset_index(drop=True)

    # Binary Feature Encodings
    binary_mapping = {
        "yes": 1,
        "no": 0,
        "GP": 1,
        "MS": 0,
        "F": 1,
        "M": 0,
        "U": 1,
        "R": 0,
        "LE3": 1,
        "GT3": 0,
        "T": 1,
        "A": 0,
    }

    binary_cols = [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
    ]

    for col in binary_cols:
        if col in combined.columns:
            combined[col] = combined[col].map(binary_mapping).fillna(0).astype(int)

    # Domain-Specific Feature Engineering
    if "Medu" in combined.columns and "Fedu" in combined.columns:
        combined["parent_edu_sum"] = combined["Medu"] + combined["Fedu"]
        combined["parent_edu_max"] = np.maximum(combined["Medu"], combined["Fedu"])
        combined["parent_edu_min"] = np.minimum(combined["Medu"], combined["Fedu"])
        combined["parent_edu_diff"] = np.abs(combined["Medu"] - combined["Fedu"])
        combined["parent_edu_prod"] = combined["Medu"] * combined["Fedu"]

    if "Dalc" in combined.columns and "Walc" in combined.columns:
        combined["total_alcohol"] = combined["Dalc"] + combined["Walc"]
        combined["alcohol_ratio"] = combined["Walc"] / (combined["Dalc"] + 0.5)
        combined["heavy_drinking"] = (
            (combined["Dalc"] >= 3) | (combined["Walc"] >= 4)
        ).astype(int)

    if (
        "studytime" in combined.columns
        and "freetime" in combined.columns
        and "goout" in combined.columns
    ):
        combined["study_to_social_ratio"] = combined["studytime"] / (
            combined["freetime"] + combined["goout"] + 1.0
        )
        combined["social_sum"] = combined["freetime"] + combined["goout"]

    if "studytime" in combined.columns and "traveltime" in combined.columns:
        combined["study_to_travel_ratio"] = combined["studytime"] / (
            combined["traveltime"] + 0.5
        )

    if "studytime" in combined.columns and "age" in combined.columns:
        combined["study_per_age"] = combined["studytime"] / (combined["age"] - 14.0)

    if "absences" in combined.columns:
        combined["log_absences"] = np.log1p(np.maximum(0, combined["absences"]))
        combined["high_absence_flag"] = (combined["absences"] > 10).astype(int)

    if "failures" in combined.columns and "absences" in combined.columns:
        combined["failure_absence_risk"] = combined["failures"] * np.log1p(
            np.maximum(0, combined["absences"])
        )

    if "failures" in combined.columns and "studytime" in combined.columns:
        combined["failure_to_study"] = combined["failures"] / (
            combined["studytime"] + 0.5
        )

    # Composite student risk index
    if (
        "failures" in combined.columns
        and "absences" in combined.columns
        and "total_alcohol" in combined.columns
        and "studytime" in combined.columns
    ):
        combined["composite_student_risk"] = (
            (combined["failures"] + 0.1)
            * (np.log1p(np.maximum(0, combined["absences"])) + 0.1)
            * (combined["total_alcohol"] + 0.1)
        ) / (combined["studytime"] + 0.5)

    support_components = [
        col
        for col in ["schoolsup", "famsup", "paid", "higher", "internet"]
        if col in combined.columns
    ]
    if support_components:
        combined["academic_support_score"] = combined[support_components].sum(axis=1)

    if "famrel" in combined.columns and "health" in combined.columns:
        combined["family_health_wellbeing"] = combined["famrel"] + combined["health"]

    # Categorical interaction combinations
    if "school" in combined.columns and "sex" in combined.columns:
        combined["school_sex"] = (
            combined["school"].astype(str) + "_" + combined["sex"].astype(str)
        )
    if "Medu" in combined.columns and "Fedu" in combined.columns:
        combined["Medu_Fedu_level"] = (
            combined["Medu"].astype(str) + "_" + combined["Fedu"].astype(str)
        )

    # Historical grade dynamics if present
    has_g1 = "G1" in combined.columns
    has_g2 = "G2" in combined.columns

    if has_g1 and has_g2:
        combined["G1"] = pd.to_numeric(combined["G1"], errors="coerce").fillna(0)
        combined["G2"] = pd.to_numeric(combined["G2"], errors="coerce").fillna(0)
        combined["G_avg"] = (combined["G1"] + combined["G2"]) / 2.0
        combined["G_diff"] = combined["G2"] - combined["G1"]
        combined["G_growth_rate"] = (combined["G2"] + 1.0) / (combined["G1"] + 1.0)
        combined["G1_times_G2"] = combined["G1"] * combined["G2"]
        combined["G2_squared"] = combined["G2"] ** 2
        combined["G1_squared"] = combined["G1"] ** 2
        combined["is_pass_G1"] = (combined["G1"] >= 10).astype(int)
        combined["is_pass_G2"] = (combined["G2"] >= 10).astype(int)
        combined["both_passed"] = (
            (combined["G1"] >= 10) & (combined["G2"] >= 10)
        ).astype(int)
        combined["G1_zero"] = (combined["G1"] == 0).astype(int)
        combined["G2_zero"] = (combined["G2"] == 0).astype(int)
        combined["both_grades_zero"] = (
            (combined["G1"] == 0) & (combined["G2"] == 0)
        ).astype(int)
        combined["improving_grade"] = (combined["G2"] > combined["G1"]).astype(int)
        combined["declining_grade"] = (combined["G2"] < combined["G1"]).astype(int)
    elif has_g1:
        combined["G1"] = pd.to_numeric(combined["G1"], errors="coerce").fillna(0)
        combined["G1_zero"] = (combined["G1"] == 0).astype(int)
        combined["is_pass_G1"] = (combined["G1"] >= 10).astype(int)

    nominal_cols = [
        col
        for col in ["Mjob", "Fjob", "reason", "guardian", "school_sex", "Medu_Fedu_level"]
        if col in combined.columns
    ]

    for col in nominal_cols:
        combined[col] = combined[col].astype(str).fillna("missing")
        freq_map = combined[col].value_counts(normalize=True).to_dict()
        combined[f"{col}_freq"] = combined[col].map(freq_map).astype(np.float32)

    if nominal_cols:
        combined = pd.get_dummies(
            combined, columns=nominal_cols, drop_first=False, dtype=np.float32
        )

    for col in combined.columns:
        if combined[col].dtype == "object":
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    X_train_df = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test_df = combined.iloc[n_train:].copy().reset_index(drop=True)

    n_splits = 10
    binned_y = pd.qcut(
        y_train,
        q=min(10, len(np.unique(y_train))),
        labels=False,
        duplicates="drop",
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X_train_df, binned_y))

    return X_train_df, X_test_df, y_train, folds


# =========================================================================
# 2. PyTorch Deep Tabular Architecture & Loss
# =========================================================================


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.block(x) + x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


class RMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def create_torch_model_and_optimizer(
    input_dim: int,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_blocks: int = 2,
    dropout_rate: float = 0.25,
    epochs: int = 120,
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = RMSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


# =========================================================================
# 3. Model Factories
# =========================================================================


def get_lgb_model(random_state: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=15,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_extratrees_model(random_state: int = 42) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_hist_gb_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=random_state,
    )


def get_gbr_model(random_state: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=random_state,
    )


def get_svr_model(C: float = 2.0, epsilon: float = 0.1) -> SVR:
    return SVR(C=C, epsilon=epsilon, kernel="rbf")


def get_ridge_model(alpha: float = 10.0, random_state: int = 42) -> Ridge:
    return Ridge(alpha=alpha, random_state=random_state)


# =========================================================================
# 4. Cross-Validation Training and Evaluation Pipeline
# =========================================================================


def main():
    X_train_df, X_test_df, y_train, folds = load_and_preprocess_data()

    n_train = len(X_train_df)
    n_test = len(X_test_df)
    n_folds = len(folds)

    model_names = [
        "lgb",
        "xgb",
        "extratrees",
        "hist_gb",
        "gbr",
        "ridge",
        "svr",
        "tabular_resnet",
    ]
    oof_predictions = {m: np.zeros(n_train, dtype=np.float32) for m in model_names}
    test_predictions = {m: np.zeros(n_test, dtype=np.float32) for m in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr_df, y_tr = X_train_df.iloc[train_idx], y_train[train_idx]
        X_va_df, y_va = X_train_df.iloc[val_idx], y_train[val_idx]

        # Strict out-of-fold scaling for linear/kernel/neural models
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_tr_df).astype(np.float32)
        X_va_scaled = fold_scaler.transform(X_va_df).astype(np.float32)
        X_te_scaled = fold_scaler.transform(X_test_df).astype(np.float32)

        # 1. LightGBM
        lgb_model = get_lgb_model(random_state=42 + fold_idx)
        lgb_callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        lgb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=[(X_va_df, y_va)],
            callbacks=lgb_callbacks,
        )
        oof_predictions["lgb"][val_idx] = lgb_model.predict(X_va_df)
        test_predictions["lgb"] += lgb_model.predict(X_test_df) / n_folds

        # 2. XGBoost
        xgb_model = get_xgb_model(random_state=42 + fold_idx)
        xgb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=[(X_va_df, y_va)],
            verbose=False,
        )
        oof_predictions["xgb"][val_idx] = xgb_model.predict(X_va_df)
        test_predictions["xgb"] += xgb_model.predict(X_test_df) / n_folds

        # 3. ExtraTrees
        et_model = get_extratrees_model(random_state=42 + fold_idx)
        et_model.fit(X_tr_df, y_tr)
        oof_predictions["extratrees"][val_idx] = et_model.predict(X_va_df)
        test_predictions["extratrees"] += et_model.predict(X_test_df) / n_folds

        # 4. HistGradientBoosting
        hist_gb_model = get_hist_gb_model(random_state=42 + fold_idx)
        hist_gb_model.fit(X_tr_df, y_tr)
        oof_predictions["hist_gb"][val_idx] = hist_gb_model.predict(X_va_df)
        test_predictions["hist_gb"] += hist_gb_model.predict(X_test_df) / n_folds

        # 5. GradientBoosting
        gbr_model = get_gbr_model(random_state=42 + fold_idx)
        gbr_model.fit(X_tr_df, y_tr)
        oof_predictions["gbr"][val_idx] = gbr_model.predict(X_va_df)
        test_predictions["gbr"] += gbr_model.predict(X_test_df) / n_folds

        # 6. Ridge Regressor
        ridge_model = get_ridge_model(alpha=10.0, random_state=42 + fold_idx)
        ridge_model.fit(X_tr_scaled, y_tr)
        oof_predictions["ridge"][val_idx] = ridge_model.predict(X_va_scaled)
        test_predictions["ridge"] += ridge_model.predict(X_te_scaled) / n_folds

        # 7. Support Vector Regression (SVR)
        svr_model = get_svr_model(C=2.0, epsilon=0.1)
        svr_model.fit(X_tr_scaled, y_tr)
        oof_predictions["svr"][val_idx] = svr_model.predict(X_va_scaled)
        test_predictions["svr"] += svr_model.predict(X_te_scaled) / n_folds

        # 8. PyTorch TabularResNet
        input_dim = X_tr_scaled.shape[1]
        net, criterion, optimizer, scheduler = create_torch_model_and_optimizer(
            input_dim=input_dim,
            lr=3e-3,
            weight_decay=1e-4,
            hidden_dim=128,
            num_blocks=2,
            dropout_rate=0.25,
            epochs=120,
        )
        net = net.to(device)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        batch_size = min(32, len(train_dataset))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        best_val_rmse = float("inf")
        best_net_weights = None
        patience = 25
        patience_counter = 0

        for epoch in range(120):
            net.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = net(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                optimizer.step()

            scheduler.step()

            net.eval()
            with torch.no_grad():
                vx = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
                vy = torch.tensor(y_va, dtype=torch.float32).to(device)
                val_preds = net(vx)
                val_rmse = criterion(val_preds, vy).item()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_net_weights = {
                    k: v.cpu().clone() for k, v in net.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        net.load_state_dict(best_net_weights)
        net.eval()
        with torch.no_grad():
            oof_nn_val = (
                net(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            test_nn_preds = (
                net(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )

        oof_predictions["tabular_resnet"][val_idx] = oof_nn_val
        test_predictions["tabular_resnet"] += test_nn_preds / n_folds

    # Metric-Aligned Ensemble Optimization with L2 Regularization
    oof_matrix = np.column_stack([oof_predictions[m] for m in model_names])
    test_matrix = np.column_stack([test_predictions[m] for m in model_names])

    l2_reg = 1e-4

    def ensemble_loss_fn(weights):
        w = np.array(weights)
        w_norm = w / (np.sum(w) + 1e-8)
        pred = np.clip(np.dot(oof_matrix, w_norm), 0.0, 20.0)
        return root_mean_squared_error(y_train, pred) + l2_reg * np.sum(w_norm**2)

    initial_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0.0, 1.0) for _ in range(len(model_names))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        ensemble_loss_fn,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x / np.sum(opt_res.x)

    oof_final = np.clip(np.dot(oof_matrix, best_weights), 0.0, 20.0)
    test_final = np.clip(np.dot(test_matrix, best_weights), 0.0, 20.0)

    final_score = root_mean_squared_error(y_train, oof_final)

    # Save submission file with required structure: id, G3
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_final), dtype=int),
            "G3": test_final.astype(np.float32),
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()