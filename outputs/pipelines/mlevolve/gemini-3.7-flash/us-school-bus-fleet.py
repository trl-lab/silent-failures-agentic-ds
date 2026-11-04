import os
import random
import re
import warnings
from typing import Tuple

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 1. REPRODUCIBILITY & METRIC DEFINITIONS
# =========================================================================
def set_seed(seed: int = 42):
    """Set global random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, clip_min: float = 0.0
) -> float:
    """Compute Root Mean Squared Error with non-negative lower bound clipping."""
    preds_clipped = np.clip(y_pred, clip_min, None)
    return float(np.sqrt(mean_squared_error(y_true, preds_clipped)))


# =========================================================================
# 2. FEATURE ENGINEERING & PREPROCESSING PIPELINE
# =========================================================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names while preserving semantic meaning."""
    new_cols = {}
    for col in df.columns:
        clean_col = re.sub(r"[^\w\s]", "_", str(col)).strip()
        clean_col = re.sub(r"\s+", "_", clean_col)
        new_cols[col] = clean_col
    return df.rename(columns=new_cols)


def identify_column_types(df: pd.DataFrame, target_col: str):
    """Identify numeric, categorical, and datetime columns dynamically."""
    ignore_cols = {"id", "ID", "Id", target_col}
    candidate_cols = [c for c in df.columns if c not in ignore_cols]

    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in candidate_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if unique_count <= 5 and df[col].dtype == "object":
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            sample_non_null = df[col].dropna().astype(str).head(20)
            is_date = False
            if len(sample_non_null) > 0 and any(
                kw in col.lower() for kw in ["date", "time", "year", "month", "day"]
            ):
                try:
                    pd.to_datetime(sample_non_null, errors="raise")
                    is_date = True
                except (ValueError, TypeError, UserWarning):
                    is_date = False
            if is_date:
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols


def compute_out_of_fold_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
):
    """Compute leak-free Out-Of-Fold Target Encoding."""
    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_encoded = pd.DataFrame(index=train_df.index)
    test_encoded = pd.DataFrame(index=test_df.index)

    for col in cat_cols:
        encoded_col_name = f"{col}_te"
        train_encoded[encoded_col_name] = np.nan

        for train_idx, val_idx in kf.split(train_df):
            tr_slice = train_df.iloc[train_idx]
            stats = tr_slice.groupby(col)[target_col].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)

            val_cats = train_df.iloc[val_idx][col]
            train_encoded.iloc[
                val_idx, train_encoded.columns.get_loc(encoded_col_name)
            ] = val_cats.map(smoothed).fillna(global_mean)

        full_stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
        full_counts = full_stats["count"]
        full_means = full_stats["mean"]
        full_smoothed = (full_counts * full_means + smoothing * global_mean) / (
            full_counts + smoothing
        )
        test_encoded[encoded_col_name] = (
            test_df[col].map(full_smoothed).fillna(global_mean)
        )

    return train_encoded, test_encoded


def extract_datetime_features(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    """Extract temporal features from datetime columns."""
    dt_df = pd.DataFrame(index=df.index)
    for col in datetime_cols:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        dt_df[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(np.float32)
        dt_df[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(np.float32)
        dt_df[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(np.float32)
        dt_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(np.float32)
        dt_df[f"{col}_is_na"] = dt_series.isna().astype(np.float32)
    return dt_df


def process_features(
    train_path: str = "./input/train.parquet",
    test_path: str = "./input/test.parquet",
    target_col_raw: str = "Bus age",
    n_splits: int = 10,
    random_state: int = 42,
):
    """Execute end-to-end feature extraction and preprocessing."""
    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    train_df = clean_column_names(train_raw)
    test_df = clean_column_names(test_raw)

    target_candidates = [
        c
        for c in train_df.columns
        if c.lower() == target_col_raw.lower().replace(" ", "_") or c == target_col_raw
    ]
    if target_candidates:
        target_col = target_candidates[0]
    else:
        target_col = [c for c in train_df.columns if "age" in c.lower()][0]

    id_col = None
    for cand in ["id", "ID", "Id", "index"]:
        if cand in test_df.columns:
            id_col = cand
            break

    y_train = train_df[target_col].astype(np.float32).values

    numeric_cols, categorical_cols, datetime_cols = identify_column_types(
        train_df, target_col
    )
    numeric_cols = [c for c in numeric_cols if c in test_df.columns]
    categorical_cols = [c for c in categorical_cols if c in test_df.columns]
    datetime_cols = [c for c in datetime_cols if c in test_df.columns]

    # 1. Datetime Features
    train_dt_feat = extract_datetime_features(train_df, datetime_cols)
    test_dt_feat = extract_datetime_features(test_df, datetime_cols)

    # 2. Categorical Encodings
    train_cat_feat = pd.DataFrame(index=train_df.index)
    test_cat_feat = pd.DataFrame(index=test_df.index)

    for col in categorical_cols:
        tr_s = train_df[col].astype(str).fillna("__MISSING__")
        te_s = test_df[col].astype(str).fillna("__MISSING__")

        freq_map = tr_s.value_counts(normalize=True).to_dict()
        train_cat_feat[f"{col}_freq"] = (
            tr_s.map(freq_map).fillna(0.0).astype(np.float32)
        )
        test_cat_feat[f"{col}_freq"] = te_s.map(freq_map).fillna(0.0).astype(np.float32)

        all_cats = sorted(list(set(tr_s.unique()).union(set(te_s.unique()))))
        cat_map = {cat: idx for idx, cat in enumerate(all_cats)}
        train_cat_feat[f"{col}_code"] = tr_s.map(cat_map).fillna(-1).astype(np.float32)
        test_cat_feat[f"{col}_code"] = te_s.map(cat_map).fillna(-1).astype(np.float32)

    # 3. Out-Of-Fold Target Encodings
    train_te_feat, test_te_feat = compute_out_of_fold_target_encoding(
        train_df,
        test_df,
        categorical_cols,
        target_col,
        n_splits=n_splits,
        smoothing=10.0,
        random_state=random_state,
    )

    # 4. Numeric Transformations
    train_num_feat = pd.DataFrame(index=train_df.index)
    test_num_feat = pd.DataFrame(index=test_df.index)

    for col in numeric_cols:
        tr_val = pd.to_numeric(train_df[col], errors="coerce")
        te_val = pd.to_numeric(test_df[col], errors="coerce")

        if tr_val.isna().sum() > 0 or te_val.isna().sum() > 0:
            train_num_feat[f"{col}_isna"] = tr_val.isna().astype(np.float32)
            test_num_feat[f"{col}_isna"] = te_val.isna().astype(np.float32)

        med_val = tr_val.median()
        if pd.isna(med_val):
            med_val = 0.0

        tr_clean = tr_val.fillna(med_val).astype(np.float32)
        te_clean = te_val.fillna(med_val).astype(np.float32)

        train_num_feat[col] = tr_clean
        test_num_feat[col] = te_clean

        if (tr_clean >= 0).all() and (te_clean >= 0).all():
            skew_val = tr_clean.skew()
            if skew_val > 1.5:
                train_num_feat[f"{col}_log1p"] = np.log1p(
                    np.clip(tr_clean, 0, None)
                ).astype(np.float32)
                test_num_feat[f"{col}_log1p"] = np.log1p(
                    np.clip(te_clean, 0, None)
                ).astype(np.float32)

    # 5. Domain Interaction Ratios and Products
    key_fleet_keywords = [
        "fleet",
        "bus",
        "student",
        "enroll",
        "expend",
        "revenue",
        "income",
        "poverty",
        "mile",
        "route",
    ]
    interact_candidates = [
        c for c in numeric_cols if any(kw in c.lower() for kw in key_fleet_keywords)
    ]

    if len(interact_candidates) >= 2:
        for i in range(min(len(interact_candidates), 8)):
            for j in range(i + 1, min(len(interact_candidates), 8)):
                col1 = interact_candidates[i]
                col2 = interact_candidates[j]
                ratio_name = f"ratio_{col1}_div_{col2}"
                train_num_feat[ratio_name] = (
                    train_num_feat[col1] / (train_num_feat[col2].abs() + 1e-4)
                ).astype(np.float32)
                test_num_feat[ratio_name] = (
                    test_num_feat[col1] / (test_num_feat[col2].abs() + 1e-4)
                ).astype(np.float32)

                mult_name = f"mult_{col1}_x_{col2}"
                train_num_feat[mult_name] = (
                    train_num_feat[col1] * train_num_feat[col2]
                ).astype(np.float32)
                test_num_feat[mult_name] = (
                    test_num_feat[col1] * test_num_feat[col2]
                ).astype(np.float32)

    # 6. Group Aggregation Features (mean, std, min, max, diff, z-score)
    for cat_col in categorical_cols[:5]:
        if 1 < train_df[cat_col].nunique() < 500:
            for num_col in numeric_cols[:8]:
                grp_stats = (
                    train_df.groupby(cat_col)[num_col]
                    .agg(["mean", "std", "min", "max"])
                    .reset_index()
                )
                grp_stats.columns = [
                    cat_col,
                    f"grp_{cat_col}_{num_col}_mean",
                    f"grp_{cat_col}_{num_col}_std",
                    f"grp_{cat_col}_{num_col}_min",
                    f"grp_{cat_col}_{num_col}_max",
                ]

                train_merged = train_df[[cat_col]].merge(
                    grp_stats, on=cat_col, how="left"
                )
                test_merged = test_df[[cat_col]].merge(
                    grp_stats, on=cat_col, how="left"
                )

                mean_col = f"grp_{cat_col}_{num_col}_mean"
                std_col = f"grp_{cat_col}_{num_col}_std"
                min_col = f"grp_{cat_col}_{num_col}_min"
                max_col = f"grp_{cat_col}_{num_col}_max"

                for stat_col in [mean_col, std_col, min_col, max_col]:
                    fill_val = train_merged[stat_col].mean()
                    if pd.isna(fill_val):
                        fill_val = 0.0
                    train_num_feat[stat_col] = (
                        train_merged[stat_col].fillna(fill_val).values.astype(np.float32)
                    )
                    test_num_feat[stat_col] = (
                        test_merged[stat_col].fillna(fill_val).values.astype(np.float32)
                    )

                diff_col = f"grp_{cat_col}_{num_col}_diff"
                zscore_col = f"grp_{cat_col}_{num_col}_zscore"

                train_diff = train_num_feat[num_col] - train_num_feat[mean_col]
                test_diff = test_num_feat[num_col] - test_num_feat[mean_col]
                train_num_feat[diff_col] = train_diff.astype(np.float32)
                test_num_feat[diff_col] = test_diff.astype(np.float32)

                train_z = train_diff / (train_num_feat[std_col].abs() + 1e-4)
                test_z = test_diff / (test_num_feat[std_col].abs() + 1e-4)
                train_num_feat[zscore_col] = train_z.astype(np.float32)
                test_num_feat[zscore_col] = test_z.astype(np.float32)

    # Concatenate all features
    X_train_df = pd.concat(
        [train_num_feat, train_cat_feat, train_dt_feat, train_te_feat], axis=1
    )
    X_test_df = pd.concat(
        [test_num_feat, test_cat_feat, test_dt_feat, test_te_feat], axis=1
    )

    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
    X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)

    col_medians = X_train_df.median()
    X_train_df = X_train_df.fillna(col_medians).fillna(0.0)
    X_test_df = X_test_df.fillna(col_medians).fillna(0.0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(train_df)):
        folds[val_idx] = fold

    test_ids = (
        test_df[id_col].values
        if id_col is not None
        else np.arange(len(test_df), dtype=int)
    )

    return (
        X_train_df.values.astype(np.float32),
        y_train,
        folds,
        X_test_df.values.astype(np.float32),
        test_ids,
        X_train_df.columns.tolist(),
    )


# =========================================================================
# 3. MODEL ARCHITECTURE & ENSEMBLE DEFINITIONS
# =========================================================================
class TabularResMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, dropout_rate: float = 0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(num_blocks)
        ])
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = self.act(h + block(h))
        return self.head(h).squeeze(-1)


# =========================================================================
# 4. TRAINING & ENSEMBLE EVALUATION
# =========================================================================
def run_pipeline(
    train_path: str = "./input/train.parquet",
    test_path: str = "./input/test.parquet",
    submission_path: str = "./submission/submission.csv",
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on compute device: {device}")

    # Process Features
    X_train, y_train, folds, X_test, test_ids, feature_names = process_features(
        train_path=train_path,
        test_path=test_path,
        target_col_raw="Bus age",
        n_splits=5,
        random_state=seed,
    )

    n_samples, n_features = X_train.shape
    n_splits = len(np.unique(folds))
    print(
        f"Prepared data: {n_samples} training samples, {len(X_test)} test samples, {n_features} features."
    )

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)
    oof_nn = np.zeros(n_samples, dtype=np.float32)

    test_preds_lgb = np.zeros(len(X_test), dtype=np.float32)
    test_preds_xgb = np.zeros(len(X_test), dtype=np.float32)
    test_preds_nn = np.zeros(len(X_test), dtype=np.float32)

    # Cross-Validation
    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # ---------------- LightGBM ----------------
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_pred_lgb = lgb_model.predict(X_va)
        oof_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += lgb_model.predict(X_test) / n_splits

        # ---------------- XGBoost ----------------
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_pred_xgb = xgb_model.predict(X_va)
        oof_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += xgb_model.predict(X_test) / n_splits

        # ---------------- Tabular Residual MLP ----------------
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        nn_model = TabularResMLP(
            input_dim=n_features, hidden_dim=128, num_blocks=3, dropout_rate=0.15
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=1e-5
        )

        tr_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr_s, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
            ),
            batch_size=64,
            shuffle=True,
        )
        va_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_va_s, dtype=torch.float32),
                torch.tensor(y_va, dtype=torch.float32),
            ),
            batch_size=64,
            shuffle=False,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(40):
            nn_model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = nn_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = nn_model(bx)
                    val_loss += criterion(pred, by).item() * len(by)
            val_loss /= len(y_va)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        nn_model.load_state_dict(best_state)
        nn_model.to(device)
        nn_model.eval()

        with torch.no_grad():
            nn_val_pred = (
                nn_model(torch.tensor(X_va_s, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            nn_test_pred = (
                nn_model(torch.tensor(X_te_s, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )

        oof_nn[val_idx] = nn_val_pred
        test_preds_nn += nn_test_pred / n_splits

        fold_rmse_lgb = compute_rmse(y_va, val_pred_lgb)
        fold_rmse_xgb = compute_rmse(y_va, val_pred_xgb)
        fold_rmse_nn = compute_rmse(y_va, nn_val_pred)
        print(
            f"Fold {fold+1}/{n_splits} - LGB: {fold_rmse_lgb:.4f} | XGB: {fold_rmse_xgb:.4f} | NN: {fold_rmse_nn:.4f}"
        )

    # ---------------- Blend Optimization ----------------
    def blend_objective(weights):
        w1, w2, w3 = weights
        blend_oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_nn
        return compute_rmse(y_train, blend_oof)

    opt_res = minimize(
        blend_objective,
        x0=[0.45, 0.45, 0.10],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
        method="SLSQP",
    )

    opt_weights = (
        opt_res.x if opt_res.success else np.array([0.45, 0.45, 0.10], dtype=np.float32)
    )
    opt_weights = opt_weights / np.sum(opt_weights)
    print(
        f"Optimized Ensemble Weights: LGB={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, NN={opt_weights[2]:.4f}"
    )

    final_oof_preds = (
        opt_weights[0] * oof_lgb + opt_weights[1] * oof_xgb + opt_weights[2] * oof_nn
    )
    final_validation_rmse = compute_rmse(y_train, final_oof_preds)

    # ---------------- Final Test Inference & Export ----------------
    final_test_preds = (
        opt_weights[0] * test_preds_lgb
        + opt_weights[1] * test_preds_xgb
        + opt_weights[2] * test_preds_nn
    )
    final_test_preds_clipped = np.clip(final_test_preds, 0.0, None)

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            "Bus age": final_test_preds_clipped.astype(np.float32),
        }
    )
    submission_df.to_csv(submission_path, index=False)

    print(
        f"Submission generated successfully at '{submission_path}' with {len(submission_df)} rows."
    )
    print(f"Final Validation Score: {final_validation_rmse}")


if __name__ == "__main__":
    run_pipeline(
        train_path="./input/train.parquet",
        test_path="./input/test.parquet",
        submission_path="./submission/submission.csv",
        seed=42,
    )