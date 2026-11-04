import json
import os
import lightgbm as lgb
import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TabularResBlock(nn.Module):
    """Residual block for tabular deep learning with LayerNorm and Dropout."""

    def __init__(
        self, in_features: int, hidden_features: int, dropout_rate: float = 0.2
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, in_features),
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet architecture for continuous regression tasks."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(
                    hidden_dim,
                    hidden_features=hidden_dim * 2,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


def get_lgb_model_params(extra_params: dict = None) -> dict:
    """Returns baseline parameters for LightGBM aligned with MAE optimization."""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if extra_params:
        params.update(extra_params)
    return params


def get_xgb_model_params(extra_params: dict = None) -> dict:
    """Returns baseline parameters for XGBoost aligned with MAE optimization."""
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    if extra_params:
        params.update(extra_params)
    return params


def get_nn_components(
    in_features: int,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """Instantiates PyTorch Tabular model, SmoothL1Loss criterion, and AdamW optimizer."""
    model = TabularResNet(in_features=in_features, hidden_dim=hidden_dim)
    criterion = nn.SmoothL1Loss(beta=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    print("Loading datasets...")
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    # Target transformation
    target_raw = train["loss"].values.astype(np.float32)
    train["loss_log"] = np.log1p(target_raw)

    # Construct explicit 2-way categorical interaction features prior to encoding
    print("Constructing 2-way categorical interaction features...")
    top_cats = ["cat80", "cat87", "cat100", "cat101", "cat112"]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            inter_col = f"{c1}_{c2}"
            train[inter_col] = (
                train[c1].astype(str) + "_" + train[c2].astype(str)
            )
            test[inter_col] = test[c1].astype(str) + "_" + test[c2].astype(str)

    # Identify categorical and continuous columns
    cat_cols = [c for c in train.columns if c.startswith("cat")]
    cont_cols = [c for c in train.columns if c.startswith("cont")]

    # Create 5-Fold split on train
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train)):
        train.loc[val_idx, "fold"] = fold

    # Combine categorical data for global frequency & ordinal encoding
    full_cat_df = pd.concat([train[cat_cols], test[cat_cols]], axis=0).reset_index(
        drop=True
    )

    # 2. Categorical Frequency & Ordinal Encoding
    print("Performing categorical frequency & ordinal encoding...")
    for col in cat_cols:
        freq = full_cat_df[col].value_counts()
        train[f"{col}_freq"] = train[col].map(freq).astype(np.float32)
        test[f"{col}_freq"] = test[col].map(freq).astype(np.float32)

        unique_cats = sorted(full_cat_df[col].dropna().unique())
        cat_type = pd.CategoricalDtype(categories=unique_cats)
        train[col] = train[col].astype(cat_type).cat.codes.astype(np.int32)
        test[col] = test[col].astype(cat_type).cat.codes.astype(np.int32)

    # 3. Out-Of-Fold Target Encoding for High-Cardinality Categoricals
    print("Performing leak-free out-of-fold target encoding...")
    high_card_cats = [col for col in cat_cols if full_cat_df[col].nunique() > 5]
    global_mean_log = train["loss_log"].mean()
    smoothing = 10.0

    for col in high_card_cats:
        target_enc_col = f"{col}_te"
        train[target_enc_col] = global_mean_log
        test[target_enc_col] = global_mean_log

        for fold in range(5):
            tr_mask = train["fold"] != fold
            val_mask = train["fold"] == fold

            stats = train[tr_mask].groupby(col)["loss_log"].agg(["count", "mean"])
            smooth_val = (
                stats["count"] * stats["mean"] + smoothing * global_mean_log
            ) / (stats["count"] + smoothing)

            train.loc[val_mask, target_enc_col] = (
                train.loc[val_mask, col]
                .map(smooth_val)
                .fillna(global_mean_log)
                .astype(np.float32)
            )

        stats_full = train.groupby(col)["loss_log"].agg(["count", "mean"])
        smooth_full = (
            stats_full["count"] * stats_full["mean"] + smoothing * global_mean_log
        ) / (stats_full["count"] + smoothing)
        test[target_enc_col] = (
            test[col].map(smooth_full).fillna(global_mean_log).astype(np.float32)
        )

    # 4. Continuous Row-wise Statistics
    print("Generating continuous summary features...")
    train["cont_mean"] = train[cont_cols].mean(axis=1).astype(np.float32)
    test["cont_mean"] = test[cont_cols].mean(axis=1).astype(np.float32)

    train["cont_std"] = train[cont_cols].std(axis=1).astype(np.float32)
    test["cont_std"] = test[cont_cols].std(axis=1).astype(np.float32)

    train["cont_min"] = train[cont_cols].min(axis=1).astype(np.float32)
    test["cont_min"] = test[cont_cols].min(axis=1).astype(np.float32)

    train["cont_max"] = train[cont_cols].max(axis=1).astype(np.float32)
    test["cont_max"] = test[cont_cols].max(axis=1).astype(np.float32)

    train["cont_sum"] = train[cont_cols].sum(axis=1).astype(np.float32)
    test["cont_sum"] = test[cont_cols].sum(axis=1).astype(np.float32)

    # 5. Continuous Pairwise Interaction Features
    correlated_pairs = [
        ("cont1", "cont9"),
        ("cont11", "cont12"),
        ("cont6", "cont7"),
        ("cont2", "cont3"),
        ("cont4", "cont5"),
    ]

    for c1, c2 in correlated_pairs:
        train[f"{c1}_{c2}_diff"] = (train[c1] - train[c2]).astype(np.float32)
        test[f"{c1}_{c2}_diff"] = (test[c1] - test[c2]).astype(np.float32)

        train[f"{c1}_{c2}_ratio"] = (train[c1] / (train[c2] + 1e-5)).astype(np.float32)
        test[f"{c1}_{c2}_ratio"] = (test[c1] / (test[c2] + 1e-5)).astype(np.float32)

        train[f"{c1}_{c2}_prod"] = (train[c1] * train[c2]).astype(np.float32)
        test[f"{c1}_{c2}_prod"] = (test[c1] * test[c2]).astype(np.float32)

    # 6. Group Aggregations of Key Continuous Features
    group_cats = ["cat1", "cat80", "cat100", "cat112"]
    group_conts = ["cont1", "cont7", "cont14"]

    for cat_col in group_cats:
        for cont_col in group_conts:
            grp_mean = train.groupby(cat_col)[cont_col].mean().astype(np.float32)
            grp_std = (
                train.groupby(cat_col)[cont_col].std().fillna(0).astype(np.float32)
            )

            train[f"grp_{cat_col}_{cont_col}_mean"] = (
                train[cat_col].map(grp_mean).astype(np.float32)
            )
            test[f"grp_{cat_col}_{cont_col}_mean"] = (
                test[cat_col].map(grp_mean).astype(np.float32)
            )

            train[f"grp_{cat_col}_{cont_col}_std"] = (
                train[cat_col].map(grp_std).astype(np.float32)
            )
            test[f"grp_{cat_col}_{cont_col}_std"] = (
                test[cat_col].map(grp_std).astype(np.float32)
            )

    # Clean missing/infinite values if any
    train = train.fillna(0)
    test = test.fillna(0)

    feature_cols = [
        c for c in train.columns if c not in ["id", "loss", "loss_log", "fold"]
    ]

    X = train[feature_cols].values.astype(np.float32)
    y_log = train["loss_log"].values.astype(np.float32)
    y_raw = train["loss"].values.astype(np.float32)
    folds = train["fold"].values.astype(int)

    X_test = test[feature_cols].values.astype(np.float32)

    n_samples = len(train)
    n_test = len(test)

    # Out-of-fold & Test prediction accumulators
    oof_lgb_log = np.zeros(n_samples, dtype=np.float32)
    test_lgb_log = np.zeros(n_test, dtype=np.float32)

    oof_xgb_log = np.zeros(n_samples, dtype=np.float32)
    test_xgb_log = np.zeros(n_test, dtype=np.float32)

    oof_nn_log = np.zeros(n_samples, dtype=np.float32)
    test_nn_log = np.zeros(n_test, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Pipeline initialized: {n_samples} train samples, {n_test} test samples, {len(feature_cols)} features. Device: {device}"
    )

    # 7. Train LightGBM (5-Fold CV)
    print("\n=== Training LightGBM Models ===")
    lgb_params = get_lgb_model_params()

    for fold in range(5):
        tr_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr_log = X[tr_idx], y_log[tr_idx]
        X_va, y_va_log = X[val_idx], y_log[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr_log)
        dval = lgb.Dataset(X_va, label=y_va_log, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        val_preds_log = model.predict(X_va)
        oof_lgb_log[val_idx] = val_preds_log
        test_lgb_log += model.predict(X_test) / 5.0

        fold_mae = mean_absolute_error(y_raw[val_idx], np.expm1(val_preds_log))
        print(f"LGBM Fold {fold + 1}/5 MAE: {fold_mae:.4f}")

    lgb_oof_mae = mean_absolute_error(y_raw, np.expm1(oof_lgb_log))
    print(f"LGBM Overall OOF MAE: {lgb_oof_mae:.4f}")

    # 8. Train PyTorch TabularResNet (5-Fold CV)
    print("\n=== Training PyTorch TabularResNet Models ===")
    batch_size = 512
    epochs = 20

    for fold in range(5):
        tr_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X[tr_idx])
        X_va_sc = scaler.transform(X[val_idx])
        X_test_sc = scaler.transform(X_test)

        train_ds = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_log[tr_idx], dtype=torch.float32).unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va_sc, dtype=torch.float32),
            torch.tensor(y_log[val_idx], dtype=torch.float32).unsqueeze(1),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model, criterion, optimizer = get_nn_components(in_features=len(feature_cols))
        model.to(device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_val_loss = float("inf")
        best_preds = None

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds_list = []
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss_sum += loss.item() * len(bx)
                    val_count += len(bx)
                    val_preds_list.append(out.cpu().numpy())

            val_loss = val_loss_sum / val_count
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = np.vstack(val_preds_list).squeeze()

                test_ds = TensorDataset(torch.tensor(X_test_sc, dtype=torch.float32))
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                test_preds_list = []
                with torch.no_grad():
                    for (bx,) in test_loader:
                        bx = bx.to(device)
                        out = model(bx)
                        test_preds_list.append(out.cpu().numpy())
                best_test_preds = np.vstack(test_preds_list).squeeze()

        oof_nn_log[val_idx] = best_preds
        test_nn_log += best_test_preds / 5.0

        fold_nn_mae = mean_absolute_error(y_raw[val_idx], np.expm1(best_preds))
        print(f"ResNet Fold {fold + 1}/5 MAE: {fold_nn_mae:.4f}")

    nn_oof_mae = mean_absolute_error(y_raw, np.expm1(oof_nn_log))
    print(f"ResNet Overall OOF MAE: {nn_oof_mae:.4f}")

    # 10. Ensemble & Post-Processing Optimization
    print("\n=== Optimizing Ensemble Weights & Log Shift ===")

    def obj_func(params):
        w1, w2, s = params
        weights = np.exp([w1, w2])
        weights /= np.sum(weights)

        blend_log = (
            weights[0] * oof_lgb_log
            + weights[1] * oof_nn_log
            + s
        )
        blend_raw = np.clip(np.expm1(blend_log), a_min=0, a_max=None)
        return mean_absolute_error(y_raw, blend_raw)

    res = minimize(obj_func, x0=[1.0, 1.0, 0.002], method="Nelder-Mead")
    weights = np.exp(res.x[:2])
    weights /= np.sum(weights)
    best_w_lgb, best_w_nn = weights[0], weights[1]
    best_s = float(res.x[2])

    print(
        f"Optimal Weights - LGBM: {best_w_lgb:.4f}, ResNet: {best_w_nn:.4f}, Log Shift: {best_s:.6f}"
    )

    final_oof_log = (
        best_w_lgb * oof_lgb_log
        + best_w_nn * oof_nn_log
        + best_s
    )
    final_oof_preds = np.clip(np.expm1(final_oof_log), a_min=0, a_max=None)
    final_mae = mean_absolute_error(y_raw, final_oof_preds)

    # 11. Generate Test Predictions and Output Submission
    final_test_log = (
        best_w_lgb * test_lgb_log
        + best_w_nn * test_nn_log
        + best_s
    )
    final_test_preds = np.clip(np.expm1(final_test_log), a_min=0, a_max=None)

    sub = pd.DataFrame({"id": test["id"].values, "loss": final_test_preds})
    sub_path = "./submission/submission.csv"
    sub.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path} with shape {sub.shape}.")
    print(f"Final Validation Score: {final_mae:.5f}")


if __name__ == "__main__":
    main()