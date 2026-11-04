import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# Set random seed for exact reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# PyTorch Deep Tabular Residual Neural Network with LayerNorm & SiLU
class TabularResidualMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
        )
        self.act = nn.SiLU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x + self.res_block(x))
        return self.head(x)


def preprocess_data(train_path, test_path, target_col):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if target_col not in train.columns:
        raise KeyError(f"Target column '{target_col}' not found in train data.")

    # Ensure integer 0-based ID column exists
    if "id" in test.columns:
        test_id = test["id"].values
    else:
        test_id = np.arange(len(test))

    if "id" in train.columns:
        train_id = train["id"].values
    else:
        train_id = np.arange(len(train))

    y_train_raw = train[target_col].values

    # Drop target and id for feature processing
    train_df = train.drop(columns=[target_col, "id"], errors="ignore").copy()
    test_df = test.drop(columns=["id"], errors="ignore").copy()

    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Detect column types
    datetime_cols = []
    categorical_cols = []
    numerical_cols = []
    text_cols = []

    for col in combined.columns:
        if pd.api.types.is_datetime64_any_dtype(combined[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(combined[col]):
            numerical_cols.append(col)
        else:
            sample_val = combined[col].dropna().astype(str)
            if len(sample_val) > 0:
                avg_len = sample_val.str.len().mean()
                num_unique = combined[col].nunique()
                is_date_col = any(
                    keyword in col.lower()
                    for keyword in [
                        "date",
                        "year",
                        "dt",
                        "approval",
                        "board",
                        "closing",
                    ]
                )
                if is_date_col:
                    try:
                        combined[col] = pd.to_datetime(combined[col], errors="coerce")
                        datetime_cols.append(col)
                        continue
                    except Exception:
                        pass
                if avg_len > 100 or (num_unique > 500 and avg_len > 40):
                    text_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)

    engineered_df = pd.DataFrame(index=combined.index)

    # 1. Extract Datetime features
    for col in datetime_cols:
        dt_series = pd.to_datetime(combined[col], errors="coerce")
        engineered_df[f"{col}_year"] = dt_series.dt.year
        engineered_df[f"{col}_month"] = dt_series.dt.month
        engineered_df[f"{col}_day"] = dt_series.dt.day
        engineered_df[f"{col}_quarter"] = dt_series.dt.quarter
        engineered_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        engineered_df[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(float)

        min_date = dt_series.min()
        if pd.notna(min_date):
            engineered_df[f"{col}_days_from_start"] = (dt_series - min_date).dt.days

    # 2. Process Numerical features & Missingness
    for col in numerical_cols:
        col_vals = combined[col].astype(float)
        engineered_df[col] = col_vals
        if col_vals.isna().sum() > 0:
            engineered_df[f"{col}_isna"] = col_vals.isna().astype(float)
            median_val = col_vals.iloc[:n_train].median()
            engineered_df[col].fillna(
                median_val if pd.notna(median_val) else 0.0, inplace=True
            )

    # 3. Categorical Frequency Encoding & Factorization
    for col in categorical_cols:
        cat_str = combined[col].astype(str).fillna("missing")
        freq_map = cat_str.iloc[:n_train].value_counts(normalize=True).to_dict()
        engineered_df[f"{col}_freq"] = cat_str.map(freq_map).fillna(0.0).astype(float)

        fact_codes, _ = pd.factorize(cat_str)
        engineered_df[f"{col}_cat"] = fact_codes

    # 4. Process Text Features
    for col in text_cols:
        text_series = combined[col].fillna("").astype(str)
        engineered_df[f"{col}_len"] = text_series.str.len()
        engineered_df[f"{col}_word_count"] = text_series.str.split().str.len()

        try:
            tfidf = TfidfVectorizer(max_features=50, stop_words="english")
            svd = TruncatedSVD(n_components=5, random_state=42)
            tfidf_mat_train = tfidf.fit_transform(text_series.iloc[:n_train])
            svd.fit(tfidf_mat_train)
            tfidf_mat_full = tfidf.transform(text_series)
            svd_mat = svd.transform(tfidf_mat_full)
            for k in range(svd_mat.shape[1]):
                engineered_df[f"{col}_svd_{k}"] = svd_mat[:, k]
        except Exception:
            pass

    engineered_df.fillna(0.0, inplace=True)

    X_train_full = engineered_df.iloc[:n_train].reset_index(drop=True)
    X_test_full = engineered_df.iloc[n_train:].reset_index(drop=True)

    # 5. Out-Of-Fold Target Encoding for Categorical Features (No Leakage)
    y_train_log = np.log1p(np.maximum(0, y_train_raw))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in categorical_cols[:5]:
        cat_train_str = train_df[col].astype(str).fillna("missing")
        cat_test_str = test_df[col].astype(str).fillna("missing")

        te_train = pd.Series(0.0, index=X_train_full.index)
        global_mean = y_train_log.mean()

        for tr_idx, va_idx in kf.split(X_train_full):
            fold_cat = cat_train_str.iloc[tr_idx]
            fold_target = y_train_log[tr_idx]
            stats = (
                pd.DataFrame({"cat": fold_cat, "target": fold_target})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            smooth = 10
            stats["smooth_te"] = (
                stats["count"] * stats["mean"] + smooth * global_mean
            ) / (stats["count"] + smooth)
            te_map = stats["smooth_te"].to_dict()
            te_train.iloc[va_idx] = (
                cat_train_str.iloc[va_idx].map(te_map).fillna(global_mean)
            )

        stats_full = (
            pd.DataFrame({"cat": cat_train_str, "target": y_train_log})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        stats_full["smooth_te"] = (
            stats_full["count"] * stats_full["mean"] + 10 * global_mean
        ) / (stats_full["count"] + 10)
        te_test = cat_test_str.map(stats_full["smooth_te"].to_dict()).fillna(
            global_mean
        )

        X_train_full[f"{col}_target_enc"] = te_train.values
        X_test_full[f"{col}_target_enc"] = te_test.values

    # Save processed features to working directory
    X_train_full["id"] = train_id
    X_train_full[target_col] = y_train_raw
    X_test_full["id"] = test_id

    X_train_full.to_parquet("./working/processed_train.parquet", index=False)
    X_test_full.to_parquet("./working/processed_test.parquet", index=False)

    X_tr_np = X_train_full.drop(columns=["id", target_col]).values
    X_te_np = X_test_full.drop(columns=["id"]).values

    return X_tr_np, y_train_raw, X_te_np, train_id, test_id


def train_pytorch_model(
    X_train_np,
    y_train_np,
    X_val_np,
    y_val_np,
    X_test_np,
    epochs=30,
    batch_size=256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_np)
    X_va_s = scaler.transform(X_val_np)
    X_te_s = scaler.transform(X_test_np)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = TabularResidualMLP(input_dim=X_train_np.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_val_preds = None
    best_test_preds = None

    val_tensor = torch.tensor(X_va_s, dtype=torch.float32).to(device)
    test_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
    val_target = torch.tensor(y_val_np.reshape(-1, 1), dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_out = model(val_tensor)
            v_loss = criterion(v_out, val_target).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_val_preds = v_out.cpu().numpy().squeeze()
                best_test_preds = model(test_tensor).cpu().numpy().squeeze()

    return best_val_preds, best_test_preds


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    target_col = "New Commitments (US$)"

    # Preprocess & Feature Engineering Pipeline
    X, y_raw, X_test, train_id, test_id = preprocess_data(
        train_path, test_path, target_col
    )
    y_log = np.log1p(np.maximum(0, y_raw))

    n_samples = len(X)
    n_test = len(X_test)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_models = 3
    oof_preds_log = np.zeros((n_samples, n_models))
    test_preds_log = np.zeros((n_test, n_models))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": 42,
        "n_estimators": 1000,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
        "verbosity": 0,
    }

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, y_tr_log = X[tr_idx], y_log[tr_idx]
        X_va, y_va_log = X[va_idx], y_log[va_idx]

        # Model 0: LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds_log[va_idx, 0] = model_lgb.predict(X_va)
        test_preds_log[:, 0] += model_lgb.predict(X_test) / float(n_splits)

        # Model 1: XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
        oof_preds_log[va_idx, 1] = model_xgb.predict(X_va)
        test_preds_log[:, 1] += model_xgb.predict(X_test) / float(n_splits)

        # Model 2: CatBoost or HistGradientBoosting
        if HAS_CATBOOST:
            model_cb = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                random_seed=42,
                thread_count=-1,
                verbose=0,
            )
            model_cb.fit(
                X_tr,
                y_tr_log,
                eval_set=(X_va, y_va_log),
                early_stopping_rounds=50,
            )
            oof_preds_log[va_idx, 2] = model_cb.predict(X_va)
            test_preds_log[:, 2] += model_cb.predict(X_test) / float(n_splits)
        else:
            model_hgb = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.03, random_state=42
            )
            model_hgb.fit(X_tr, y_tr_log)
            oof_preds_log[va_idx, 2] = model_hgb.predict(X_va)
            test_preds_log[:, 2] += model_hgb.predict(X_test) / float(n_splits)

    # Revert log1p predictions to raw dollar scale
    oof_preds_raw = np.expm1(np.maximum(0, oof_preds_log))
    test_preds_raw = np.expm1(np.maximum(0, test_preds_log))

    # Constrained Ensemble Optimization (Weights + Scale Multiplier on Raw RMSE)
    def ensemble_objective(params):
        weights = params[:-1]
        scale = params[-1]
        w_norm = weights / (np.sum(weights) + 1e-8)
        blended_oof = scale * np.dot(oof_preds_raw, w_norm)
        blended_oof = np.maximum(0, blended_oof)
        return np.sqrt(mean_squared_error(y_raw, blended_oof))

    init_params = np.array([1.0 / n_models] * n_models + [1.0])
    bounds = [(0.0, 1.0)] * n_models + [(0.5, 2.0)]
    res = minimize(ensemble_objective, init_params, bounds=bounds, method="L-BFGS-B")

    opt_weights = res.x[:-1] / (np.sum(res.x[:-1]) + 1e-8)
    opt_scale = res.x[-1]

    oof_final = np.maximum(0, opt_scale * np.dot(oof_preds_raw, opt_weights))
    test_final = np.maximum(0, opt_scale * np.dot(test_preds_raw, opt_weights))

    final_rmse = np.sqrt(mean_squared_error(y_raw, oof_final))

    # Save official submission file
    submission = pd.DataFrame({"id": test_id, target_col: test_final})
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()