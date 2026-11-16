import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_qwk(y_true, y_pred):
    """
    Computes Quadratic Weighted Kappa between true integer labels and predicted integer labels.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class OptimizedRounder:
    """
    Optimizes threshold boundaries to convert continuous regression predictions
    into ordinal discrete classes (0, 1, 2, 3) maximizing Quadratic Weighted Kappa.
    """

    def __init__(self, n_classes=4):
        self.n_classes = n_classes
        self.coef_ = [0.5, 1.5, 2.5]

    def _loss(self, coef, X, y):
        X_p = np.digitize(X, sorted(coef))
        return -compute_qwk(y, X_p)

    def fit(self, X, y):
        initial_coef = [0.5, 1.5, 2.5]
        res = minimize(self._loss, initial_coef, args=(X, y), method="Powell")
        self.coef_ = sorted(res.x)

    def predict(self, X):
        return np.digitize(X, self.coef_)


class ResidualBlock(nn.Module):
    """
    Dense residual block for tabular data with LayerNorm and Dropout regularization.
    """

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    """
    Deep PyTorch Neural Network architecture for continuous tabular regression.
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[0], dropout=dropout) for _ in range(2)]
        )

        self.neck = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
        )

        self.head = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.neck(out)
        return self.head(out).squeeze(-1)


def build_pytorch_model_and_optimizer(input_dim, lr=1e-3, weight_decay=1e-4):
    model = TabularResNet(input_dim=input_dim, hidden_dims=[256, 128], dropout=0.25)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def process_and_engineer_features():
    print("Processing and engineering features...")
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Filter out training samples without target 'sii'
    train_df = train_df.dropna(subset=["sii"]).reset_index(drop=True)
    train_df["sii"] = train_df["sii"].astype(int)

    y_train = train_df["sii"].values
    n_train = len(train_df)

    df_all = pd.concat(
        [
            train_df.drop(columns=["sii"], errors="ignore"),
            test_df.drop(columns=["sii"], errors="ignore"),
        ],
        axis=0,
        ignore_index=True,
    )

    modules = [
        "Basic_Demos",
        "CGAS",
        "Physical",
        "Fitness_Endurance",
        "FCT",
        "BIA",
        "PAQ_A",
        "PAQ_C",
        "SDS",
        "PreInt_Demos",
    ]

    # Remove all PCIAT columns (unavailable in test set, causes target leakage)
    pciat_cols = [c for c in df_all.columns if "PCIAT" in c]
    if pciat_cols:
        df_all = df_all.drop(columns=pciat_cols)

    # 1. Missingness features
    feature_cols = [c for c in df_all.columns if c != "id"]
    df_all["missing_count_total"] = df_all[feature_cols].isnull().sum(axis=1)
    df_all["missing_ratio_total"] = df_all["missing_count_total"] / len(feature_cols)

    for mod in modules:
        mod_cols = [c for c in feature_cols if c.startswith(mod)]
        if mod_cols:
            df_all[f"missing_count_{mod}"] = df_all[mod_cols].isnull().sum(axis=1)
            df_all[f"missing_ratio_{mod}"] = df_all[f"missing_count_{mod}"] / len(
                mod_cols
            )

    # 2. Physical domain ratios and indicators
    if (
        "Physical-Systolic_BP" in df_all.columns
        and "Physical-Diastolic_BP" in df_all.columns
    ):
        sys = df_all["Physical-Systolic_BP"]
        dia = df_all["Physical-Diastolic_BP"]
        df_all["Physical-MAP"] = (2 * dia + sys) / 3.0
        df_all["Physical-Pulse_Pressure"] = sys - dia

    if (
        "Physical-Waist_Circumference" in df_all.columns
        and "Physical-Height" in df_all.columns
    ):
        df_all["Physical-Waist_to_Height"] = df_all["Physical-Waist_Circumference"] / (
            df_all["Physical-Height"] + 1e-5
        )

    if "Physical-Weight" in df_all.columns and "Physical-Height" in df_all.columns:
        height_m = df_all["Physical-Height"] / 100.0
        df_all["Physical-BMI_calc"] = df_all["Physical-Weight"] / (height_m**2 + 1e-5)
        if "Physical-BMI" in df_all.columns:
            df_all["Physical-BMI_diff"] = (
                df_all["Physical-BMI"] - df_all["Physical-BMI_calc"]
            )

    if "BIA-BIA_Fat" in df_all.columns and "BIA-BIA_FFM" in df_all.columns:
        df_all["BIA-Fat_to_FFM_ratio"] = df_all["BIA-BIA_Fat"] / (
            df_all["BIA-BIA_FFM"] + 1e-5
        )

    if "BIA-BIA_TBW" in df_all.columns and "Physical-Weight" in df_all.columns:
        df_all["BIA-TBW_to_Weight_ratio"] = df_all["BIA-BIA_TBW"] / (
            df_all["Physical-Weight"] + 1e-5
        )

    # 3. Questionnaire aggregations
    sds_cols = [c for c in feature_cols if c.startswith("SDS-")]
    if sds_cols:
        numeric_sds = [
            c
            for c in sds_cols
            if pd.api.types.is_numeric_dtype(df_all[c]) or df_all[c].dtype == object
        ]
        if numeric_sds:
            sds_num = df_all[numeric_sds].apply(pd.to_numeric, errors="coerce")
            df_all["SDS_mean"] = sds_num.mean(axis=1)
            df_all["SDS_std"] = sds_num.std(axis=1)
            df_all["SDS_sum"] = sds_num.sum(axis=1)

    paq_cols = [
        c for c in feature_cols if c.startswith("PAQ_A-") or c.startswith("PAQ_C-")
    ]
    if paq_cols:
        paq_num = df_all[paq_cols].apply(pd.to_numeric, errors="coerce")
        df_all["PAQ_mean"] = paq_num.mean(axis=1)
        df_all["PAQ_std"] = paq_num.std(axis=1)

    # 4. Handle Categorical / Object columns safely
    categorical_cols = df_all.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "id"]

    for col in categorical_cols:
        df_all[col] = df_all[col].astype(str).fillna("Missing")

    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_cat = df_all.iloc[:n_train][categorical_cols]
        encoder.fit(train_cat)
        df_all[categorical_cols] = encoder.transform(df_all[categorical_cols])

    # Separate processed train and test sets
    train_processed = df_all.iloc[:n_train].copy().reset_index(drop=True)
    test_processed = df_all.iloc[n_train:].copy().reset_index(drop=True)

    train_processed["sii"] = y_train

    # 5. Create Stratified 5-Fold split mapping
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_processed["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        skf.split(train_processed, train_processed["sii"])
    ):
        train_processed.loc[val_idx, "fold"] = fold

    os.makedirs("./working", exist_ok=True)
    train_processed.to_parquet("./working/train_processed.parquet", index=False)
    test_processed.to_parquet("./working/test_processed.parquet", index=False)

    return train_processed, test_processed


def main():
    seed_everything(42)
    train_df, test_df = process_and_engineer_features()

    ignore_cols = ["id", "sii", "fold"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X = train_df[feature_cols]
    y = train_df["sii"].values
    folds = train_df["fold"].values
    X_test = test_df[feature_cols]

    n_splits = 5
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(n_splits):
        val_idx = folds == fold
        train_idx = ~val_idx

        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        # 1. LightGBM Regressor
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "n_estimators": 1000,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "verbose": -1,
            "n_jobs": -1,
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        lgb_val = lgb_model.predict(X_va)
        lgb_test = lgb_model.predict(X_test)

        # 2. XGBoost Regressor
        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.03,
            "n_estimators": 1000,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "early_stopping_rounds": 50,
            "n_jobs": -1,
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_val = xgb_model.predict(X_va)
        xgb_test = xgb_model.predict(X_test)

        # 3. CatBoost Regressor
        cat_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "learning_rate": 0.03,
            "iterations": 1000,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42 + fold,
            "verbose": 0,
        }
        cat_model = cb.CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        cat_val = cat_model.predict(X_va)
        cat_test = cat_model.predict(X_test)

        # 4. PyTorch TabularResNet
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr_imp = imputer.fit_transform(X_tr)
        X_va_imp = imputer.transform(X_va)
        X_te_imp = imputer.transform(X_test)

        X_tr_imp = np.nan_to_num(X_tr_imp, nan=0.0)
        X_va_imp = np.nan_to_num(X_va_imp, nan=0.0)
        X_te_imp = np.nan_to_num(X_te_imp, nan=0.0)

        X_tr_imp = scaler.fit_transform(X_tr_imp)
        X_va_imp = scaler.transform(X_va_imp)
        X_te_imp = scaler.transform(X_te_imp)

        X_tr_imp = np.nan_to_num(X_tr_imp, nan=0.0)
        X_va_imp = np.nan_to_num(X_va_imp, nan=0.0)
        X_te_imp = np.nan_to_num(X_te_imp, nan=0.0)

        model, criterion, optimizer = build_pytorch_model_and_optimizer(
            input_dim=X_tr_imp.shape[1], lr=1e-3, weight_decay=1e-4
        )
        model.to(device)

        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_tr_imp, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_va_imp, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=64, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)

        best_val_loss = float("inf")
        best_weights = None
        patience = 15
        patience_counter = 0

        for epoch in range(80):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_loss += criterion(out, by).item() * len(by)
            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)

        model.eval()
        with torch.no_grad():
            nn_val = (
                model(torch.tensor(X_va_imp, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            nn_test = (
                model(torch.tensor(X_te_imp, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )

        # Ensemble continuous predictions
        fold_val_pred = 0.35 * lgb_val + 0.30 * xgb_val + 0.25 * cat_val + 0.10 * nn_val
        fold_test_pred = (
            0.35 * lgb_test + 0.30 * xgb_test + 0.25 * cat_test + 0.10 * nn_test
        )

        oof_preds[val_idx] = fold_val_pred
        test_preds += fold_test_pred / n_splits

    # Global Out-Of-Fold Threshold Optimization
    global_rounder = OptimizedRounder()
    global_rounder.fit(oof_preds, y)
    oof_discrete_preds = global_rounder.predict(oof_preds)
    final_score = compute_qwk(y, oof_discrete_preds)

    # Predict discrete class targets for test dataset
    test_discrete_preds = global_rounder.predict(test_preds)

    # Save final submission file
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission = pd.DataFrame({"id": test_df["id"], "sii": test_discrete_preds.astype(int)})
    submission.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score:.5f}")


if __name__ == "__main__":
    main()
