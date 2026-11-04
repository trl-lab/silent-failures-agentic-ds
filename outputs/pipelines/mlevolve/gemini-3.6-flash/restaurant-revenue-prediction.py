import copy
import os
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target scaling factor for numerical stability (revenue in millions)
TARGET_SCALE = 1e6


# 1. Feature Engineering & Preprocessing
def preprocess_and_engineer_features():
    os.makedirs("./working", exist_ok=True)

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns = [c.strip() for c in test_df.columns]

    target_col = [c for c in train_df.columns if c.lower() == "revenue"][0]
    id_col = [c for c in train_df.columns if c.lower() in ["id", "row_id"]][0]

    y_train = train_df[target_col].values
    train_ids = train_df[id_col].values
    test_ids = test_df[id_col].values

    date_col = [
        c for c in train_df.columns if "date" in c.lower() or "open" in c.lower()
    ][0]

    def process_dates(df, ref_date=None):
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if ref_date is None:
            ref_date = dates.max()
        df_feat = pd.DataFrame(index=df.index)
        df_feat["open_year"] = dates.dt.year
        df_feat["open_month"] = dates.dt.month
        df_feat["open_day"] = dates.dt.day
        df_feat["open_dayofweek"] = dates.dt.dayofweek
        df_feat["open_quarter"] = dates.dt.quarter
        df_feat["restaurant_age_days"] = (ref_date - dates).dt.days
        df_feat["restaurant_age_years"] = df_feat["restaurant_age_days"] / 365.25
        return df_feat, ref_date

    train_date_feats, ref_date = process_dates(train_df)
    test_date_feats, _ = process_dates(test_df, ref_date=ref_date)

    p_cols = sorted(
        [c for c in train_df.columns if c.startswith("P") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )

    def extract_p_stats(df):
        p_data = df[p_cols].astype(float)
        feats = pd.DataFrame(index=df.index)

        feats["P_mean"] = p_data.mean(axis=1)
        feats["P_std"] = p_data.std(axis=1)
        feats["P_min"] = p_data.min(axis=1)
        feats["P_max"] = p_data.max(axis=1)
        feats["P_sum"] = p_data.sum(axis=1)
        feats["P_median"] = p_data.median(axis=1)
        feats["P_skew"] = p_data.skew(axis=1)
        feats["P_nonzero"] = (p_data > 0).sum(axis=1)

        p_group1 = [c for c in p_cols if 1 <= int(c[1:]) <= 11]
        p_group2 = [c for c in p_cols if 12 <= int(c[1:]) <= 23]
        p_group3 = [c for c in p_cols if 24 <= int(c[1:]) <= 37]

        if p_group1:
            feats["P_grp1_mean"] = p_data[p_group1].mean(axis=1)
            feats["P_grp1_sum"] = p_data[p_group1].sum(axis=1)
            feats["P_grp1_std"] = p_data[p_group1].std(axis=1)
        if p_group2:
            feats["P_grp2_mean"] = p_data[p_group2].mean(axis=1)
            feats["P_grp2_sum"] = p_data[p_group2].sum(axis=1)
            feats["P_grp2_std"] = p_data[p_group2].std(axis=1)
        if p_group3:
            feats["P_grp3_mean"] = p_data[p_group3].mean(axis=1)
            feats["P_grp3_sum"] = p_data[p_group3].sum(axis=1)
            feats["P_grp3_std"] = p_data[p_group3].std(axis=1)

        for col in p_cols:
            feats[col] = p_data[col]

        return feats

    train_p_feats = extract_p_stats(train_df)
    test_p_feats = extract_p_stats(test_df)

    cat_cols = [c for c in ["City", "City Group", "Type"] if c in train_df.columns]
    combined_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)
    combined_ohe = pd.get_dummies(
        combined_cat, columns=cat_cols, drop_first=False, dtype=float
    )

    train_ohe = combined_ohe.iloc[: len(train_df)].reset_index(drop=True)
    test_ohe = combined_ohe.iloc[len(train_df) :].reset_index(drop=True)

    if "City" in train_df.columns:
        city_counts = train_df["City"].value_counts().to_dict()
        train_ohe["City_freq"] = train_df["City"].map(city_counts).fillna(0).values
        test_ohe["City_freq"] = test_df["City"].map(city_counts).fillna(0).values

    X_train = pd.concat([train_date_feats, train_p_feats, train_ohe], axis=1)
    X_test = pd.concat([test_date_feats, test_p_feats, test_ohe], axis=1)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(X_train, y_train))

    return X_train, X_test, y_train, test_ids, folds


# 2. Model Architectures & Loss Definitions
class TabularResNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h + self.res_block(h)
        out = self.head(h)
        return out


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def get_tabular_models(random_state: int = 42):
    return {
        "lgb": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=10.0,
            random_state=random_state,
            verbose=-1,
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=10.0,
            random_state=random_state,
            verbosity=0,
        ),
        "ridge": Ridge(alpha=25.0, random_state=random_state),
        "huber": HuberRegressor(alpha=10.0, max_iter=1000),
        "elasticnet": ElasticNet(alpha=1.5, l1_ratio=0.5, random_state=random_state),
    }


def train_pytorch_model(X_tr, y_tr, X_va, y_va, X_te, epochs=250):
    input_dim = X_tr.shape[1]
    model = TabularResNet(input_dim=input_dim, hidden_dim=64, dropout_rate=0.4).to(
        device
    )
    criterion = RMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for b_x, b_y in loader:
            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_t)
            val_loss = criterion(val_preds, y_va_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 35:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_va_t).cpu().numpy().flatten()
        test_pred = model(X_te_t).cpu().numpy().flatten()

    return val_pred, test_pred


# 3. Main Training, Blending, and Submission Execution
def main():
    os.makedirs("./submission", exist_ok=True)

    X_train, X_test, y_train, test_ids, folds = preprocess_and_engineer_features()

    n_samples = len(X_train)
    n_test = len(X_test)
    n_folds = len(folds)

    model_names = ["lgb", "xgb", "ridge", "huber", "elasticnet", "nn"]

    oof_preds = {m: np.zeros(n_samples) for m in model_names}
    test_preds = {m: np.zeros(n_test) for m in model_names}

    y_train_scaled = y_train / TARGET_SCALE

    print(f"Starting {n_folds}-Fold Cross-Validation on {n_samples} samples...")
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr_raw, y_tr_s = (
            X_train.iloc[train_idx].values,
            y_train_scaled[train_idx],
        )
        X_va_raw, y_va_s = (
            X_train.iloc[val_idx].values,
            y_train_scaled[val_idx],
        )
        X_te_raw = X_test.values

        p_cols_idx = [
            i
            for i, c in enumerate(X_train.columns)
            if c.startswith("P") and c[1:].isdigit()
        ]
        pca = PCA(n_components=min(5, len(p_cols_idx)), random_state=42)
        pca_tr = pca.fit_transform(X_tr_raw[:, p_cols_idx])
        pca_va = pca.transform(X_va_raw[:, p_cols_idx])
        pca_te = pca.transform(X_te_raw[:, p_cols_idx])

        X_tr_raw_ext = np.hstack([X_tr_raw, pca_tr])
        X_va_raw_ext = np.hstack([X_va_raw, pca_va])
        X_te_raw_ext = np.hstack([X_te_raw, pca_te])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw_ext)
        X_va = scaler.transform(X_va_raw_ext)
        X_te = scaler.transform(X_te_raw_ext)

        models = get_tabular_models(random_state=42 + fold)
        for name, model in models.items():
            model.fit(X_tr, y_tr_s)
            val_p = model.predict(X_va)
            test_p = model.predict(X_te)

            oof_preds[name][val_idx] = val_p * TARGET_SCALE
            test_preds[name] += (test_p * TARGET_SCALE) / n_folds

        nn_val_p, nn_test_p = train_pytorch_model(X_tr, y_tr_s, X_va, y_va_s, X_te)
        oof_preds["nn"][val_idx] = nn_val_p * TARGET_SCALE
        test_preds["nn"] += (nn_test_p * TARGET_SCALE) / n_folds

        fold_rmse_lgb = np.sqrt(
            mean_squared_error(y_train[val_idx], oof_preds["lgb"][val_idx])
        )
        print(f"Fold {fold+1}/{n_folds} - LGB RMSE: {fold_rmse_lgb:.2f}")

    print("\n--- Out-of-Fold Model Performance ---")
    for name in model_names:
        score = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"Model [{name.upper():10s}] OOF RMSE: {score:.2f}")

    oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    def blend_objective(weights):
        blend_oof = np.dot(oof_matrix, weights)
        return np.sqrt(mean_squared_error(y_train, blend_oof))

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = opt_res.x

    print("\n--- Optimal Ensemble Weights ---")
    for name, w in zip(model_names, opt_weights):
        print(f"Weight [{name.upper():10s}]: {w:.4f}")

    final_oof_preds = np.dot(oof_matrix, opt_weights)
    final_score = np.sqrt(mean_squared_error(y_train, final_oof_preds))

    final_test_preds = np.dot(test_matrix, opt_weights)
    final_test_preds = np.clip(final_test_preds, a_min=y_train.min() * 0.5, a_max=None)

    sub_df = pd.DataFrame({"id": test_ids, "revenue": final_test_preds})
    sub_df["id"] = sub_df["id"].astype(int)
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path} with shape {sub_df.shape}")

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()