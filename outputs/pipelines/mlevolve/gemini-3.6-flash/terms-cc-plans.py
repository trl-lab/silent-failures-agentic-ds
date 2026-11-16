import copy
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================================
# 1. DATA PROCESSING AND FEATURE ENGINEERING
# ==============================================================================
def process_data_and_engineer_features(input_dir="./input", output_dir="./working"):
    print("=== Starting Data Processing and Feature Engineering ===")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Loaded Train shape: {train_df.shape}")
    print(f"Loaded Test shape: {test_df.shape}")

    target_col = "APR"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training dataset.")

    y_train = train_df[target_col].values
    train_df_features = train_df.drop(columns=[target_col]).copy()
    test_df_features = test_df.copy()

    if "id" in test_df_features.columns:
        test_ids = test_df_features["id"].values
    else:
        test_ids = np.arange(len(test_df_features))
        test_df_features["id"] = test_ids
        if "id" not in train_df_features.columns:
            train_df_features["id"] = np.arange(len(train_df_features))

    train_len = len(train_df_features)
    combined = pd.concat(
        [train_df_features, test_df_features], axis=0, ignore_index=True
    )

    non_feature_cols = ["id"]
    candidate_cols = [c for c in combined.columns if c not in non_feature_cols]

    num_cols = []
    cat_cols = []
    datetime_cols = []

    for col in candidate_cols:
        if pd.api.types.is_datetime64_any_dtype(combined[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(combined[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    print(
        f"Feature breakdown - Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}, Datetime: {len(datetime_cols)}"
    )

    # Dictionary accumulation to eliminate DataFrame fragmentation
    new_features = {}

    # Datetime features
    for col in datetime_cols:
        new_features[f"{col}_year"] = combined[col].dt.year
        new_features[f"{col}_month"] = combined[col].dt.month
        new_features[f"{col}_day"] = combined[col].dt.day
        new_features[f"{col}_dayofweek"] = combined[col].dt.dayofweek
        num_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])

    if datetime_cols:
        combined.drop(columns=datetime_cols, inplace=True)

    # Missing value handling
    for col in num_cols:
        if combined[col].isnull().sum() > 0:
            new_features[f"{col}_isna"] = combined[col].isnull().astype(int)
            train_median = combined.iloc[:train_len][col].median()
            combined[col] = combined[col].fillna(
                train_median if not pd.isna(train_median) else 0
            )

    for col in cat_cols:
        combined[col] = combined[col].astype(str).fillna("missing")

    # Non-linear numeric transformations
    for col in num_cols:
        vals = combined[col].values
        new_features[f"{col}_log1p"] = np.log1p(np.maximum(0, vals))
        new_features[f"{col}_square"] = vals ** 2
        new_features[f"{col}_sqrt"] = np.sqrt(np.maximum(0, vals))

    # KMeans spatial cluster distance features
    num_vals_train = combined.iloc[:train_len][num_cols].values
    num_vals_all = combined[num_cols].values
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(num_vals_train)
    cluster_dists = kmeans.transform(num_vals_all)
    for k in range(5):
        new_features[f"kmeans_dist_cluster_{k}"] = cluster_dists[:, k]

    # Numerical quantile rank transformations (computed on train set without leakage)
    for col in num_cols:
        train_vals = np.sort(combined.iloc[:train_len][col].values)
        vals = combined[col].values
        new_features[f"{col}_rank"] = np.searchsorted(train_vals, vals, side="right") / float(train_len)

    # Numerical pairwise transformations: differences, sums, products, ratios
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1, col2 = num_cols[i], num_cols[j]
            v1 = combined[col1].values
            v2 = combined[col2].values

            new_features[f"{col1}_diff_{col2}"] = v1 - v2
            new_features[f"{col1}_sum_{col2}"] = v1 + v2
            new_features[f"{col1}_mult_{col2}"] = v1 * v2
            new_features[f"{col1}_ratio_{col2}"] = v1 / (np.abs(v2) + 1e-5)

    # Pairwise categorical interactions
    cat_interaction_cols = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            cat1, cat2 = cat_cols[i], cat_cols[j]
            pair_col = f"{cat1}_X_{cat2}"
            new_features[pair_col] = combined[cat1].astype(str) + "_" + combined[cat2].astype(str)
            cat_interaction_cols.append(pair_col)

    if new_features:
        combined = pd.concat([combined, pd.DataFrame(new_features, index=combined.index)], axis=1)
        new_features = {}

    # Group aggregations (mean, std, min, max, median) calculated on train set only to prevent leakage
    all_cats_for_grp = cat_cols + cat_interaction_cols
    train_combined = combined.iloc[:train_len]

    grp_features = {}
    for cat_col in all_cats_for_grp:
        for num_col in num_cols:
            grp_stats = train_combined.groupby(cat_col)[num_col].agg(["mean", "std", "min", "max", "median"])
            for stat in ["mean", "std", "min", "max", "median"]:
                stat_map = grp_stats[stat].to_dict()
                grp_features[f"grp_{cat_col}_{num_col}_{stat}"] = combined[cat_col].map(stat_map).fillna(0).values

    # Frequency encoding
    for col in all_cats_for_grp:
        freq = combined.iloc[:train_len][col].value_counts(normalize=True).to_dict()
        grp_features[f"{col}_freq"] = combined[col].map(freq).fillna(0).values

    if grp_features:
        combined = pd.concat([combined, pd.DataFrame(grp_features, index=combined.index)], axis=1)

    train_processed = combined.iloc[:train_len].copy().reset_index(drop=True)
    test_processed = combined.iloc[train_len:].copy().reset_index(drop=True)

    train_processed[target_col] = y_train

    # Leak-free 5-fold OOF multi-statistic Bayesian target encoding (mean, std, min, max)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_processed["fold"] = -1

    te_cols = cat_cols + cat_interaction_cols
    m_factor = 10.0  # Bayesian smoothing m-factor

    stats_list = ["mean", "std", "min", "max"]
    te_train_dict = {f"{cat}_te_{stat}": np.zeros(len(train_processed), dtype=np.float32) for cat in te_cols for stat in stats_list}
    te_test_dict = {f"{cat}_te_{stat}": np.zeros(len(test_processed), dtype=np.float32) for cat in te_cols for stat in stats_list}

    global_target_mean = float(train_processed[target_col].mean())
    global_target_std = float(train_processed[target_col].std())
    global_target_min = float(train_processed[target_col].min())
    global_target_max = float(train_processed[target_col].max())

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
        train_processed.loc[val_idx, "fold"] = fold
        fold_train = train_processed.iloc[train_idx]

        for cat in te_cols:
            grouped = fold_train.groupby(cat)[target_col].agg(["count", "mean", "std", "min", "max"])
            smoothed_mean = (grouped["count"] * grouped["mean"] + m_factor * global_target_mean) / (grouped["count"] + m_factor)

            val_cats = train_processed.loc[val_idx, cat]

            te_train_dict[f"{cat}_te_mean"][val_idx] = val_cats.map(smoothed_mean).fillna(global_target_mean).values
            te_train_dict[f"{cat}_te_std"][val_idx] = val_cats.map(grouped["std"]).fillna(global_target_std).values
            te_train_dict[f"{cat}_te_min"][val_idx] = val_cats.map(grouped["min"]).fillna(global_target_min).values
            te_train_dict[f"{cat}_te_max"][val_idx] = val_cats.map(grouped["max"]).fillna(global_target_max).values

    for cat in te_cols:
        grouped = train_processed.groupby(cat)[target_col].agg(["count", "mean", "std", "min", "max"])
        smoothed_mean = (grouped["count"] * grouped["mean"] + m_factor * global_target_mean) / (grouped["count"] + m_factor)

        te_test_dict[f"{cat}_te_mean"] = test_processed[cat].map(smoothed_mean).fillna(global_target_mean).values
        te_test_dict[f"{cat}_te_std"] = test_processed[cat].map(grouped["std"]).fillna(global_target_std).values
        te_test_dict[f"{cat}_te_min"] = test_processed[cat].map(grouped["min"]).fillna(global_target_min).values
        te_test_dict[f"{cat}_te_max"] = test_processed[cat].map(grouped["max"]).fillna(global_target_max).values

    train_processed = pd.concat([train_processed, pd.DataFrame(te_train_dict, index=train_processed.index)], axis=1)
    test_processed = pd.concat([test_processed, pd.DataFrame(te_test_dict, index=test_processed.index)], axis=1)

    # Variance Filtering & LightGBM-based Feature Importance Selection
    print("=== Performing Variance Filtering and Feature Importance Selection ===")
    non_feature_cols = [target_col, "fold", "id"]
    candidate_cols = [
        c for c in train_processed.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(train_processed[c])
    ]

    # Variance Filter
    variances = train_processed[candidate_cols].var()
    valid_cols = variances[variances > 1e-6].index.tolist()
    print(f"Features passing variance filter: {len(valid_cols)} / {len(candidate_cols)}")

    # LightGBM Feature Importance
    X_selection = train_processed[valid_cols].fillna(0).values
    y_selection = train_processed[target_col].values

    sel_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    sel_lgb.fit(X_selection, y_selection)
    importances = sel_lgb.feature_importances_

    k_selected = min(180, len(valid_cols))
    top_indices = np.argsort(importances)[::-1][:k_selected]
    selected_features = [valid_cols[i] for i in top_indices]

    print(f"Retained top {len(selected_features)} features based on feature importance.")

    train_cols_keep = ["id", target_col, "fold"] + selected_features
    test_cols_keep = ["id"] + selected_features

    train_processed = train_processed[[c for c in train_cols_keep if c in train_processed.columns]]
    test_processed = test_processed[[c for c in test_cols_keep if c in test_processed.columns]]

    train_out_path = os.path.join(output_dir, "train_processed.parquet")
    test_out_path = os.path.join(output_dir, "test_processed.parquet")

    train_processed.to_parquet(train_out_path, index=False)
    test_processed.to_parquet(test_out_path, index=False)

    print("=== Feature Engineering Completed Successfully ===")


# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeAndExcitation, self).__init__()
        reduced_dim = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights


class TabularResBlock(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_rate=0.2):
        super(TabularResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.se = SqueezeAndExcitation(in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.act(self.fc1(out))
        out = self.norm2(out)
        out = self.dropout(self.fc2(out))
        out = self.se(out)
        return residual + out


class TabularFeatureTokenizerSE(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.25,
        out_features=1,
    ):
        super(TabularFeatureTokenizerSE, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(
                    in_features=hidden_dim,
                    hidden_features=hidden_dim * 2,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, out_features),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def create_torch_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-4, T_0=20):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )
    return optimizer, scheduler


def get_lgbm_model(random_state=42):
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2500,
        learning_rate=0.015,
        num_leaves=63,
        max_depth=7,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model(random_state=42):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2500,
        learning_rate=0.015,
        max_depth=6,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(random_state=42):
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.015,
        depth=7,
        l2_leaf_reg=5.0,
        subsample=0.75,
        random_seed=random_state,
        verbose=0,
    )


def get_histgbm_model(random_state=42):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=1000,
        learning_rate=0.03,
        max_leaf_nodes=63,
        max_depth=8,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )


def get_extratrees_model(random_state=42):
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


# ==============================================================================
# 3. TRAINING AND EVALUATION PIPELINE
# ==============================================================================
def run_training_and_evaluation():
    print("=== Starting 5-Fold Training and Evaluation Pipeline ===")

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(working_dir, "train_processed.parquet")
    test_path = os.path.join(working_dir, "test_processed.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "APR"
    fold_col = "fold"
    id_col = "id"

    ignore_cols = [target_col, fold_col, id_col]
    feature_cols = [
        c
        for c in train_df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    print(
        f"Training set size: {len(train_df)}, Test set size: {len(test_df)}, Features count: {len(feature_cols)}"
    )

    y = train_df[target_col].values.astype(np.float32)
    folds = train_df[fold_col].values
    test_ids = (
        test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))
    )

    X_train_full = train_df[feature_cols].fillna(0).values.astype(np.float32)
    X_test_full = test_df[feature_cols].fillna(0).values.astype(np.float32)

    n_samples = len(train_df)
    n_test = len(test_df)
    n_folds = int(np.max(folds) + 1)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_nn = np.zeros(n_samples, dtype=np.float32)
    oof_hist = np.zeros(n_samples, dtype=np.float32)
    oof_et = np.zeros(n_samples, dtype=np.float32)

    test_preds_lgb = np.zeros(n_test, dtype=np.float32)
    test_preds_xgb = np.zeros(n_test, dtype=np.float32)
    test_preds_cat = np.zeros(n_test, dtype=np.float32)
    test_preds_nn = np.zeros(n_test, dtype=np.float32)
    test_preds_hist = np.zeros(n_test, dtype=np.float32)
    test_preds_et = np.zeros(n_test, dtype=np.float32)

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = X_train_full[train_idx], y[train_idx]
        X_va, y_va = X_train_full[val_idx], y[val_idx]
        X_te = X_test_full.copy()

        # 1. LightGBM
        lgb_model = get_lgbm_model(random_state=seed + fold)
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
        test_preds_lgb += lgb_model.predict(X_te) / n_folds

        # 2. XGBoost
        xgb_model = get_xgboost_model(random_state=seed + fold)
        xgb_model.set_params(early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_pred_xgb = xgb_model.predict(X_va)
        oof_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += xgb_model.predict(X_te) / n_folds

        # 3. CatBoost
        cat_model = get_catboost_model(random_state=seed + fold)
        cat_model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
        )
        val_pred_cat = cat_model.predict(X_va)
        oof_cat[val_idx] = val_pred_cat
        test_preds_cat += cat_model.predict(X_te) / n_folds

        # 4. HistGradientBoosting
        hist_model = get_histgbm_model(random_state=seed + fold)
        hist_model.fit(X_tr, y_tr)
        val_pred_hist = hist_model.predict(X_va)
        oof_hist[val_idx] = val_pred_hist
        test_preds_hist += hist_model.predict(X_te) / n_folds

        # 5. ExtraTrees
        et_model = get_extratrees_model(random_state=seed + fold)
        et_model.fit(X_tr, y_tr)
        val_pred_et = et_model.predict(X_va)
        oof_et[val_idx] = val_pred_et
        test_preds_et += et_model.predict(X_te) / n_folds

        # 6. PyTorch TabularFeatureTokenizerSE
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=512, shuffle=False, num_workers=2
        )

        nn_model = TabularFeatureTokenizerSE(
            in_features=X_tr.shape[1],
            hidden_dim=256,
            num_blocks=3,
            dropout_rate=0.2,
        ).to(device)

        criterion = RMSELoss()
        optimizer, scheduler = create_torch_optimizer_and_scheduler(
            nn_model, lr=7e-4, weight_decay=1e-4, T_0=20
        )

        best_val_loss = float("inf")
        best_model_weights = copy.deepcopy(nn_model.state_dict())
        patience, max_patience = 0, 15
        epochs = 80

        for epoch in range(epochs):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = nn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = nn_model(bx)
                    val_loss += ((out - by) ** 2).sum().item()
                    val_samples += len(by)

            val_rmse = np.sqrt(val_loss / val_samples)

            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_model_weights = copy.deepcopy(nn_model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break

        nn_model.load_state_dict(best_model_weights)
        nn_model.eval()

        with torch.no_grad():
            va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            val_pred_nn = nn_model(va_tensor).cpu().numpy()
            oof_nn[val_idx] = val_pred_nn

            te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            test_pred_nn = nn_model(te_tensor).cpu().numpy()
            test_preds_nn += test_pred_nn / n_folds

        fold_rmse_lgb = np.sqrt(mean_squared_error(y_va, val_pred_lgb))
        fold_rmse_xgb = np.sqrt(mean_squared_error(y_va, val_pred_xgb))
        fold_rmse_cat = np.sqrt(mean_squared_error(y_va, val_pred_cat))
        fold_rmse_nn = np.sqrt(mean_squared_error(y_va, val_pred_nn))
        fold_rmse_hist = np.sqrt(mean_squared_error(y_va, val_pred_hist))
        fold_rmse_et = np.sqrt(mean_squared_error(y_va, val_pred_et))

        print(
            f"Fold {fold+1}/{n_folds} RMSE - LGB: {fold_rmse_lgb:.4f} | XGB: {fold_rmse_xgb:.4f} | CAT: {fold_rmse_cat:.4f} | NN: {fold_rmse_nn:.4f} | HIST: {fold_rmse_hist:.4f} | ET: {fold_rmse_et:.4f}"
        )

    rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
    rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
    rmse_cat = np.sqrt(mean_squared_error(y, oof_cat))
    rmse_nn = np.sqrt(mean_squared_error(y, oof_nn))
    rmse_hist = np.sqrt(mean_squared_error(y, oof_hist))
    rmse_et = np.sqrt(mean_squared_error(y, oof_et))

    print(
        f"Overall OOF RMSE -> LGB: {rmse_lgb:.4f}, XGB: {rmse_xgb:.4f}, CAT: {rmse_cat:.4f}, NN: {rmse_nn:.4f}, HIST: {rmse_hist:.4f}, ET: {rmse_et:.4f}"
    )

    # Ensembling via SLSQP constrained optimization
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn, oof_hist, oof_et])
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn, test_preds_hist, test_preds_et]
    )

    def blend_loss(weights):
        w = weights / np.sum(weights)
        pred = np.dot(oof_matrix, w)
        return np.sqrt(mean_squared_error(y, pred))

    init_weights = np.ones(6) / 6.0
    bounds = [(0, 1)] * 6
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        blend_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = res.x / np.sum(res.x)

    print(f"Optimal Blend Weights: {optimal_weights}")

    oof_blend = np.dot(oof_matrix, optimal_weights)
    test_blend = np.dot(test_matrix, optimal_weights)

    final_score = np.sqrt(mean_squared_error(y, oof_blend))

    # Save final submission
    sub_df = pd.DataFrame({"id": test_ids, "APR": test_blend})
    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with shape {sub_df.shape}")

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    process_data_and_engineer_features()
    run_training_and_evaluation()