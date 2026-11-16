import os
import pickle
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ==============================================================================
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# ==============================================================================
def prepare_data_and_features():
    print("Starting data processing and feature engineering...")

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_test["date"] = pd.to_datetime(df_test["date"])

    df_train["is_test"] = 0
    df_test["is_test"] = 1
    df_test["sales"] = np.nan

    # Target log-transform to match RMSLE metric optimization
    df_train["log_sales"] = np.log1p(df_train["sales"].clip(lower=0))
    df_test["log_sales"] = np.nan

    # Combine for seamless time series lag and rolling computation
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df.sort_values(by=["store_nbr", "family", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Date and Calendar Features
    print("Generating calendar and cyclical features...")
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(np.int16)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(np.int8)
    df["is_quincena"] = (
        (df["day"] == 15) | (df["date"].dt.is_month_end)
    ).astype(np.int8)
    df["payday_offset"] = np.minimum(
        np.abs(df["day"] - 15), df["date"].dt.days_in_month - df["day"]
    ).astype(np.int8)

    # Cyclical encoding
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    # Categorical Encodings
    print("Encoding categorical variables...")
    family_categories = {cat: idx for idx, cat in enumerate(df["family"].unique())}
    df["family_code"] = df["family"].map(family_categories).astype("category")
    df["store_nbr"] = df["store_nbr"].astype("category")

    # Promotion Features
    print("Generating promotion interaction features...")
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(np.float32)

    grp_promo = df.groupby(["store_nbr", "family"])["onpromotion"]
    df["promo_lag_1"] = grp_promo.shift(1).fillna(0).astype(np.float32)
    df["promo_lag_7"] = grp_promo.shift(7).fillna(0).astype(np.float32)
    df["promo_roll_mean_7"] = (
        grp_promo.transform(lambda x: x.rolling(7, min_periods=1).mean())
        .fillna(0)
        .astype(np.float32)
    )
    df["promo_roll_mean_14"] = (
        grp_promo.transform(lambda x: x.rolling(14, min_periods=1).mean())
        .fillna(0)
        .astype(np.float32)
    )
    df["promo_roll_mean_28"] = (
        grp_promo.transform(lambda x: x.rolling(28, min_periods=1).mean())
        .fillna(0)
        .astype(np.float32)
    )
    df["promo_trend_ratio_7_28"] = (
        (df["promo_roll_mean_7"] + 1e-4) / (df["promo_roll_mean_28"] + 1e-4)
    ).astype(np.float32)

    # Target Lags (Starting at lag 16 for 16-day test horizon)
    print("Generating lag and rolling window features for target...")
    grp_sales = df.groupby(["store_nbr", "family"])["log_sales"]

    lags = [16, 17, 18, 19, 20, 21, 28, 35, 42, 56]
    for l in lags:
        df[f"sales_lag_{l}"] = grp_sales.shift(l).astype(np.float32)

    roll_windows = [7, 14, 28, 56]
    for w in roll_windows:
        df[f"sales_roll_mean_{w}"] = (
            df.groupby(["store_nbr", "family"])["sales_lag_16"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
            .astype(np.float32)
        )
        df[f"sales_roll_std_{w}"] = (
            df.groupby(["store_nbr", "family"])["sales_lag_16"]
            .transform(lambda x: x.rolling(w, min_periods=1).std())
            .fillna(0)
            .astype(np.float32)
        )

    for w in [14, 28]:
        df[f"sales_roll_min_{w}"] = (
            df.groupby(["store_nbr", "family"])["sales_lag_16"]
            .transform(lambda x: x.rolling(w, min_periods=1).min())
            .astype(np.float32)
        )
        df[f"sales_roll_max_{w}"] = (
            df.groupby(["store_nbr", "family"])["sales_lag_16"]
            .transform(lambda x: x.rolling(w, min_periods=1).max())
            .astype(np.float32)
        )

    df["sales_ewm_alpha01"] = (
        df.groupby(["store_nbr", "family"])["sales_lag_16"]
        .transform(lambda x: x.ewm(alpha=0.1, min_periods=1).mean())
        .astype(np.float32)
    )
    df["sales_ewm_alpha03"] = (
        df.groupby(["store_nbr", "family"])["sales_lag_16"]
        .transform(lambda x: x.ewm(alpha=0.3, min_periods=1).mean())
        .astype(np.float32)
    )

    # Dynamic trend ratio feature comparing 7-day to 28-day rolling mean
    df["sales_trend_ratio_7_28"] = (
        (df["sales_roll_mean_7"] + 1e-4) / (df["sales_roll_mean_28"] + 1e-4)
    ).astype(np.float32)

    # Historical Group Target Statistics
    print("Computing historical group target statistics...")
    max_train_date = df_train["date"].max()
    val_start_date = max_train_date - pd.Timedelta(days=15)

    hist_train_mask = df["date"] < val_start_date
    hist_df = df[hist_train_mask]

    sf_stats = (
        hist_df.groupby(["store_nbr", "family"])["log_sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    sf_stats.columns = ["store_nbr", "family", "sf_sales_mean", "sf_sales_std"]
    df = df.merge(sf_stats, on=["store_nbr", "family"], how="left")
    df["sf_sales_mean"] = df["sf_sales_mean"].fillna(0).astype(np.float32)
    df["sf_sales_std"] = df["sf_sales_std"].fillna(0).astype(np.float32)

    f_stats = hist_df.groupby("family")["log_sales"].agg(["mean", "std"]).reset_index()
    f_stats.columns = ["family", "f_sales_mean", "f_sales_std"]
    df = df.merge(f_stats, on="family", how="left")
    df["f_sales_mean"] = df["f_sales_mean"].fillna(0).astype(np.float32)
    df["f_sales_std"] = df["f_sales_std"].fillna(0).astype(np.float32)

    # Store-Family-DayOfWeek Target Statistics
    sfd_stats = (
        hist_df.groupby(["store_nbr", "family", "dayofweek"])["log_sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    sfd_stats.columns = [
        "store_nbr",
        "family",
        "dayofweek",
        "sfd_sales_mean",
        "sfd_sales_std",
    ]
    df = df.merge(sfd_stats, on=["store_nbr", "family", "dayofweek"], how="left")
    df["sfd_sales_mean"] = df["sfd_sales_mean"].fillna(0).astype(np.float32)
    df["sfd_sales_std"] = df["sfd_sales_std"].fillna(0).astype(np.float32)

    # Temporal Split
    print("Splitting into Train, Validation, and Test sets...")
    test_mask = df["is_test"] == 1
    val_mask = (~test_mask) & (df["date"] >= val_start_date)
    train_mask = (~test_mask) & (df["date"] < val_start_date)

    min_date = df_train["date"].min() + pd.Timedelta(days=60)
    train_mask = train_mask & (df["date"] >= min_date)

    feature_cols = [
        "store_nbr",
        "family_code",
        "onpromotion",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "is_weekend",
        "weekofyear",
        "is_quincena",
        "sin_dayofweek",
        "cos_dayofweek",
        "sin_month",
        "cos_month",
        "promo_lag_1",
        "promo_lag_7",
        "promo_roll_mean_7",
        "promo_roll_mean_14",
        "promo_roll_mean_28",
        "promo_trend_ratio_7_28",
        "sales_trend_ratio_7_28",
        "sf_sales_mean",
        "sf_sales_std",
        "sfd_sales_mean",
        "sfd_sales_std",
        "f_sales_mean",
        "f_sales_std",
        "sales_ewm_alpha01",
        "sales_ewm_alpha03",
    ] + [f"sales_lag_{l}" for l in lags]

    for w in roll_windows:
        feature_cols.extend([f"sales_roll_mean_{w}", f"sales_roll_std_{w}"])
    for w in [14, 28]:
        feature_cols.extend([f"sales_roll_min_{w}", f"sales_roll_max_{w}"])

    df_train_split = df[train_mask].copy().reset_index(drop=True)
    df_val_split = df[val_mask].copy().reset_index(drop=True)
    df_test_split = df[test_mask].copy().reset_index(drop=True)

    print(
        f"Train shape: {df_train_split.shape}, Val shape: {df_val_split.shape}, Test shape: {df_test_split.shape}"
    )

    dataset_bundle = {
        "X_train": df_train_split[feature_cols],
        "y_train": df_train_split["log_sales"].values,
        "X_val": df_val_split[feature_cols],
        "y_val": df_val_split["log_sales"].values,
        "val_sales_raw": df_val_split["sales"].values,
        "X_test": df_test_split[feature_cols],
        "test_ids": df_test_split["id"].values,
        "feature_names": feature_cols,
    }

    os.makedirs("./working", exist_ok=True)
    with open("./working/processed_data.pkl", "wb") as f:
        pickle.dump(dataset_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_bundle


# ==============================================================================
# STEP 2: MODEL ARCHITECTURE DEFINITIONS
# ==============================================================================
class TabularSalesResNet(nn.Module):
    """Residual Neural Network for Tabular Store Sales Forecasting."""

    def __init__(
        self,
        num_continuous_features,
        cat_cardinalities=None,
        hidden_dim=256,
        dropout_rate=0.3,
    ):
        super().__init__()
        if cat_cardinalities is None:
            cat_cardinalities = {
                "store_nbr": (55, 16),
                "family_code": (34, 12),
                "month": (13, 4),
                "dayofweek": (7, 4),
            }

        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(card + 1, dim)
                for col, (card, dim) in cat_cardinalities.items()
            }
        )

        total_emb_dim = sum(dim for _, dim in cat_cardinalities.values())

        self.num_proj = nn.Sequential(
            nn.Linear(num_continuous_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        in_dim = hidden_dim + total_emb_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn1 = nn.LayerNorm(hidden_dim)
        self.res1_act = nn.SiLU()
        self.res1_drop = nn.Dropout(dropout_rate)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn2 = nn.LayerNorm(hidden_dim)

        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res2_bn1 = nn.LayerNorm(hidden_dim // 2)
        self.res2_act = nn.SiLU()
        self.res2_drop = nn.Dropout(dropout_rate)
        self.res2_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.res2_bn2 = nn.LayerNorm(hidden_dim // 2)
        self.res2_shortcut = nn.Linear(hidden_dim, hidden_dim // 2)

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num, x_cat):
        cat_keys = ["store_nbr", "family_code", "month", "dayofweek"]
        emb_outs = [
            self.embeddings[key](x_cat[:, idx].long())
            for idx, key in enumerate(cat_keys)
        ]
        emb_concat = torch.cat(emb_outs, dim=1)
        num_proj = self.num_proj(x_num)

        x = torch.cat([num_proj, emb_concat], dim=1)
        x = self.input_layer(x)

        res1 = self.res1_act(self.res1_bn1(self.res1_fc1(x)))
        res1 = self.res1_drop(res1)
        res1 = self.res1_bn2(self.res1_fc2(res1))
        x = nn.functional.silu(x + res1)

        res2 = self.res2_act(self.res2_bn1(self.res2_fc1(x)))
        res2 = self.res2_drop(res2)
        res2 = self.res2_bn2(self.res2_fc2(res2))
        x = nn.functional.silu(self.res2_shortcut(x) + res2)

        out = self.head(x)
        return out.squeeze(-1)


def get_lightgbm_params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": True,
    }


def get_catboost_params():
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }


# ==============================================================================
# STEP 3: TRAINING, EVALUATION, AND ENSEMBLE BLENDING
# ==============================================================================
def run_training_and_evaluation(data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    val_sales_raw = data["val_sales_raw"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    feature_names = data["feature_names"]

    print(
        f"Train: {len(X_train)} rows | Val: {len(X_val)} rows | Test: {len(X_test)} rows"
    )

    # 1. LightGBM Model
    print("Training LightGBM Model...")
    lgb_params = get_lightgbm_params()
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    pred_val_lgb = lgb_model.predict(X_val)
    pred_test_lgb = lgb_model.predict(X_test)

    # 2. XGBoost Model
    print("Training XGBoost Model...")
    xgb_params = get_xgboost_params()
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_val_xgb = xgb_model.predict(X_val)
    pred_test_xgb = xgb_model.predict(X_test)

    # 3. CatBoost Model
    print("Training CatBoost Model...")
    cb_params = get_catboost_params()
    cb_model = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=["store_nbr", "family_code"],
        verbose=False,
    )
    pred_val_cb = cb_model.predict(X_val)
    pred_test_cb = cb_model.predict(X_test)

    # 4. PyTorch Tabular ResNet Model
    print("Training PyTorch Tabular ResNet Model...")
    cat_cols = ["store_nbr", "family_code", "month", "dayofweek"]
    num_cols = [col for col in feature_names if col not in cat_cols]

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols].fillna(0).values)
    X_val_num = scaler.transform(X_val[num_cols].fillna(0).values)
    X_test_num = scaler.transform(X_test[num_cols].fillna(0).values)

    X_train_cat = X_train[cat_cols].values.astype(np.int64)
    X_val_cat = X_val[cat_cols].values.astype(np.int64)
    X_test_cat = X_test[cat_cols].values.astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_num, dtype=torch.float32),
        torch.tensor(X_val_cat, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4096, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0)

    nn_model = TabularSalesResNet(num_continuous_features=len(num_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 4
    patience_counter = 0

    for epoch in range(1, 16):
        nn_model.train()
        total_train_loss = 0.0
        for num_b, cat_b, target_b in train_loader:
            num_b, cat_b, target_b = (
                num_b.to(device),
                cat_b.to(device),
                target_b.to(device),
            )
            optimizer.zero_grad()
            output = nn_model(num_b, cat_b)
            loss = criterion(output, target_b)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * num_b.size(0)

        train_loss = total_train_loss / len(train_dataset)

        nn_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for num_b, cat_b, target_b in val_loader:
                num_b, cat_b, target_b = (
                    num_b.to(device),
                    cat_b.to(device),
                    target_b.to(device),
                )
                output = nn_model(num_b, cat_b)
                loss = criterion(output, target_b)
                total_val_loss += loss.item() * num_b.size(0)

        val_loss = total_val_loss / len(val_dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    val_preds_nn = []
    with torch.no_grad():
        for num_b, cat_b, _ in val_loader:
            num_b, cat_b = num_b.to(device), cat_b.to(device)
            out = nn_model(num_b, cat_b)
            val_preds_nn.append(out.cpu().numpy())
    pred_val_nn = np.concatenate(val_preds_nn)

    test_dataset = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4096, shuffle=False, num_workers=0
    )

    test_preds_nn = []
    with torch.no_grad():
        for num_b, cat_b in test_loader:
            num_b, cat_b = num_b.to(device), cat_b.to(device)
            out = nn_model(num_b, cat_b)
            test_preds_nn.append(out.cpu().numpy())
    pred_test_nn = np.concatenate(test_preds_nn)

    # Metric Evaluation Function (RMSLE)
    def calc_rmsle(pred_log, target_sales):
        pred_sales = np.expm1(np.maximum(0, pred_log))
        return np.sqrt(
            np.mean(
                (
                    np.log1p(np.maximum(0, pred_sales))
                    - np.log1p(np.maximum(0, target_sales))
                )
                ** 2
            )
        )

    rmsle_lgb = calc_rmsle(pred_val_lgb, val_sales_raw)
    rmsle_xgb = calc_rmsle(pred_val_xgb, val_sales_raw)
    rmsle_cb = calc_rmsle(pred_val_cb, val_sales_raw)
    rmsle_nn = calc_rmsle(pred_val_nn, val_sales_raw)

    print(f"Validation RMSLE - LightGBM: {rmsle_lgb:.5f}")
    print(f"Validation RMSLE - XGBoost: {rmsle_xgb:.5f}")
    print(f"Validation RMSLE - CatBoost: {rmsle_cb:.5f}")
    print(f"Validation RMSLE - PyTorch ResNet: {rmsle_nn:.5f}")

    # Optimization of Blending Weights
    val_matrix = np.column_stack([pred_val_lgb, pred_val_xgb, pred_val_cb, pred_val_nn])
    test_matrix = np.column_stack(
        [pred_test_lgb, pred_test_xgb, pred_test_cb, pred_test_nn]
    )

    def loss_func(w):
        w_norm = w / np.sum(w)
        blend_log = np.dot(val_matrix, w_norm)
        return calc_rmsle(blend_log, val_sales_raw)

    res = opt.minimize(
        loss_func,
        x0=[0.25, 0.25, 0.25, 0.25],
        method="SLSQP",
        bounds=[(0, 1)] * 4,
        constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
    )

    opt_w = res.x / np.sum(res.x)
    print(
        f"Optimal Weights: LGBM={opt_w[0]:.4f}, XGB={opt_w[1]:.4f}, CB={opt_w[2]:.4f}, NN={opt_w[3]:.4f}"
    )

    final_val_log = np.dot(val_matrix, opt_w)
    final_score = calc_rmsle(final_val_log, val_sales_raw)

    # Predictions & Submission
    final_test_log = np.dot(test_matrix, opt_w)
    final_test_sales = np.expm1(np.maximum(0, final_test_log))

    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"id": test_ids.astype(int), "sales": final_test_sales})
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission file to {sub_path} with {len(sub_df)} rows.")

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    data_bundle = prepare_data_and_features()
    run_training_and_evaluation(data_bundle)