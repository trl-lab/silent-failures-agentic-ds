import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# ==========================================
def process_and_engineer_features():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    print("Loading raw dataset...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Federal Funds"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    # Ensure id column exists
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    # Log1p transformation for right-skewed target
    train_df["target_raw"] = train_df[target_col].copy()
    train_df["target_log"] = np.log1p(np.maximum(0, train_df[target_col].values))

    # Set up 5-fold CV split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, "fold"] = fold

    # Categorize columns
    ignore_cols = {target_col, "target_raw", "target_log", "fold", "id"}
    feature_candidates = [c for c in train_df.columns if c not in ignore_cols]

    text_cols = []
    date_cols = []
    numeric_cols = []
    categorical_cols = []

    for col in feature_candidates:
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            sample_str = train_df[col].dropna().astype(str)
            avg_len = sample_str.str.len().mean() if len(sample_str) > 0 else 0
            if avg_len > 40 or any(
                k in col.lower()
                for k in ["title", "desc", "abstract", "summary", "text", "narrative"]
            ):
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    # Convert string date candidates if present
    for col in list(categorical_cols):
        if any(k in col.lower() for k in ["date", "year", "time"]):
            try:
                converted_train = pd.to_datetime(train_df[col], errors="coerce")
                if converted_train.notnull().sum() > 0.5 * len(train_df):
                    train_df[col] = converted_train
                    test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
                    date_cols.append(col)
                    categorical_cols.remove(col)
            except Exception:
                pass

    print(
        f"Detected {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
        f"{len(text_cols)} text, and {len(date_cols)} date columns."
    )

    # 1. Feature Engineering: Date Features
    for col in date_cols:
        for df in [train_df, test_df]:
            df[f"{col}_year"] = df[col].dt.year.fillna(-1)
            df[f"{col}_month"] = df[col].dt.month.fillna(-1)
            df[f"{col}_quarter"] = df[col].dt.quarter.fillna(-1)
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek.fillna(-1)

    # 2. Feature Engineering: Text Features (Word & Char TF-IDF + 128 TruncatedSVD)
    if text_cols:
        print("Extracting Word & Char TF-IDF and 128 TruncatedSVD text features...")
        train_text_combined = (
            train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        test_text_combined = (
            test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )

        for col in text_cols:
            for df in [train_df, test_df]:
                s = df[col].fillna("").astype(str)
                df[f"{col}_char_len"] = s.str.len()
                df[f"{col}_word_count"] = s.str.split().str.len()

        word_tfidf = TfidfVectorizer(
            max_features=15000, ngram_range=(1, 3), stop_words="english"
        )
        char_tfidf = TfidfVectorizer(
            max_features=15000, ngram_range=(3, 6), analyzer="char"
        )

        train_word = word_tfidf.fit_transform(train_text_combined)
        test_word = word_tfidf.transform(test_text_combined)

        train_char = char_tfidf.fit_transform(train_text_combined)
        test_char = char_tfidf.transform(test_text_combined)

        train_tfidf = hstack([train_word, train_char])
        test_tfidf = hstack([test_word, test_char])

        n_svd = min(128, train_tfidf.shape[1] - 1) if train_tfidf.shape[1] > 1 else 1
        if n_svd > 0:
            svd = TruncatedSVD(n_components=n_svd, random_state=42)
            train_svd = svd.fit_transform(train_tfidf)
            test_svd = svd.transform(test_tfidf)

            for i in range(n_svd):
                train_df[f"text_svd_{i}"] = train_svd[:, i]
                test_df[f"text_svd_{i}"] = test_svd[:, i]

    # 3. Feature Engineering: Numeric Features & Numerical Aggregations
    for col in numeric_cols:
        train_df[f"{col}_isnull"] = train_df[col].isnull().astype(int)
        test_df[f"{col}_isnull"] = test_df[col].isnull().astype(int)

        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

        if (train_df[col] >= 0).all():
            skewness = train_df[col].skew()
            if skewness > 1.5:
                train_df[f"{col}_log1p"] = np.log1p(train_df[col])
                test_df[f"{col}_log1p"] = np.log1p(test_df[col])

    if len(numeric_cols) >= 2:
        for i in range(min(5, len(numeric_cols))):
            for j in range(i + 1, min(5, len(numeric_cols))):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                train_df[f"{c1}_ratio_{c2}"] = train_df[c1] / (train_df[c2] + 1e-5)
                test_df[f"{c1}_ratio_{c2}"] = test_df[c1] / (test_df[c2] + 1e-5)

    # Numerical aggregation statistics grouped by categorical features
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            mean_map = train_df.groupby(cat_col)[num_col].mean().to_dict()
            std_map = train_df.groupby(cat_col)[num_col].std().fillna(0).to_dict()
            q25_map = train_df.groupby(cat_col)[num_col].quantile(0.25).to_dict()
            q75_map = train_df.groupby(cat_col)[num_col].quantile(0.75).to_dict()

            train_df[f"{cat_col}_{num_col}_mean"] = train_df[cat_col].map(mean_map).fillna(0)
            train_df[f"{cat_col}_{num_col}_std"] = train_df[cat_col].map(std_map).fillna(0)
            train_df[f"{cat_col}_{num_col}_q25"] = train_df[cat_col].map(q25_map).fillna(0)
            train_df[f"{cat_col}_{num_col}_q75"] = train_df[cat_col].map(q75_map).fillna(0)

            test_df[f"{cat_col}_{num_col}_mean"] = test_df[cat_col].map(mean_map).fillna(0)
            test_df[f"{cat_col}_{num_col}_std"] = test_df[cat_col].map(std_map).fillna(0)
            test_df[f"{cat_col}_{num_col}_q25"] = test_df[cat_col].map(q25_map).fillna(0)
            test_df[f"{cat_col}_{num_col}_q75"] = test_df[cat_col].map(q75_map).fillna(0)

    # 4. Feature Engineering: Categorical Features
    for col in categorical_cols:
        freq = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

        train_df[f"{col}_te"] = np.nan
        global_target_mean = train_df["target_log"].mean()

        for fold in range(5):
            idx_tr = train_df["fold"] != fold
            idx_va = train_df["fold"] == fold
            fold_te_map = train_df[idx_tr].groupby(col)["target_log"].mean().to_dict()
            train_df.loc[idx_va, f"{col}_te"] = train_df.loc[idx_va, col].map(
                fold_te_map
            )

        train_df[f"{col}_te"] = train_df[f"{col}_te"].fillna(global_target_mean)
        full_te_map = train_df.groupby(col)["target_log"].mean().to_dict()
        test_df[f"{col}_te"] = test_df[col].map(full_te_map).fillna(global_target_mean)

        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[f"{col}_cat"] = oe.fit_transform(train_df[[col]])
        test_df[f"{col}_cat"] = oe.transform(test_df[[col]])

    exclude_all = (
        {target_col, "target_raw", "target_log", "fold", "id"}
        | set(date_cols)
        | set(text_cols)
        | set(categorical_cols)
    )
    final_features = [c for c in train_df.columns if c not in exclude_all]

    for c in final_features:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0)
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce").fillna(0)

    print(f"Data processing complete. Engineered {len(final_features)} features.")
    return train_df, test_df, final_features


# ==========================================
# 2. MODEL DESIGN
# ==========================================
class GatedLinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)

        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.norm2(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        return x + res


class GrantTabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                GatedLinearBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        out = self.head(x)
        return out.squeeze(-1)


def build_model_criterion_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dims: list = None,
):
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    model = GrantTabularResNet(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=0.15
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return model, criterion, optimizer


def create_gbdt_models():
    lgb_log = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_raw = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    xgb_log = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_raw = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    cb_log = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.015,
        depth=7,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_raw = CatBoostRegressor(
        loss_function="Huber:delta=1.0",
        iterations=1500,
        learning_rate=0.015,
        depth=7,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    return lgb_log, lgb_raw, xgb_log, xgb_raw, cb_log, cb_raw


# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================
def main():
    train_df, test_df, feature_cols = process_and_engineer_features()

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training device: {device} | Total Features: {len(feature_cols)} | Samples: {len(train_df)}"
    )

    # Array to hold 7 model predictions: lgb_log, lgb_raw, xgb_log, xgb_raw, cb_log, cb_raw, nn_log
    num_models = 7
    oof_matrix = np.zeros((len(train_df), num_models), dtype=np.float64)
    test_matrix = np.zeros((len(test_df), num_models), dtype=np.float64)

    X_test_raw = test_df[feature_cols].values.astype(np.float32)

    num_folds = train_df["fold"].nunique()
    epochs = 40
    batch_size = 128
    patience = 10

    for fold in range(num_folds):
        print(f"--- Training Fold {fold + 1}/{num_folds} ---")

        tr_idx = train_df["fold"] != fold
        va_idx = train_df["fold"] == fold

        X_tr_raw = train_df.loc[tr_idx, feature_cols].values.astype(np.float32)
        y_tr_log = train_df.loc[tr_idx, "target_log"].values.astype(np.float32)
        y_tr_raw = train_df.loc[tr_idx, "target_raw"].values.astype(np.float32)

        X_va_raw = train_df.loc[va_idx, feature_cols].values.astype(np.float32)
        y_va_log = train_df.loc[va_idx, "target_log"].values.astype(np.float32)
        y_va_raw = train_df.loc[va_idx, "target_raw"].values

        # 1. Train LightGBM, XGBoost, and CatBoost models on dual targets (log and raw)
        lgb_log, lgb_raw, xgb_log, xgb_raw, cb_log, cb_raw = create_gbdt_models()

        print("  Training LightGBM (log & raw)...")
        lgb_log.fit(
            X_tr_raw,
            y_tr_log,
            eval_set=[(X_va_raw, y_va_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_log_va = np.expm1(np.maximum(0, lgb_log.predict(X_va_raw)))
        lgb_log_te = np.expm1(np.maximum(0, lgb_log.predict(X_test_raw)))

        lgb_raw.fit(
            X_tr_raw,
            y_tr_raw,
            eval_set=[(X_va_raw, y_va_raw)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_raw_va = np.maximum(0, lgb_raw.predict(X_va_raw))
        lgb_raw_te = np.maximum(0, lgb_raw.predict(X_test_raw))

        print("  Training XGBoost (log & raw)...")
        xgb_log.fit(
            X_tr_raw,
            y_tr_log,
            eval_set=[(X_va_raw, y_va_log)],
            verbose=False,
        )
        xgb_log_va = np.expm1(np.maximum(0, xgb_log.predict(X_va_raw)))
        xgb_log_te = np.expm1(np.maximum(0, xgb_log.predict(X_test_raw)))

        xgb_raw.fit(
            X_tr_raw,
            y_tr_raw,
            eval_set=[(X_va_raw, y_va_raw)],
            verbose=False,
        )
        xgb_raw_va = np.maximum(0, xgb_raw.predict(X_va_raw))
        xgb_raw_te = np.maximum(0, xgb_raw.predict(X_test_raw))

        print("  Training CatBoost (log & raw)...")
        cb_log.fit(
            X_tr_raw,
            y_tr_log,
            eval_set=(X_va_raw, y_va_log),
            verbose=0,
        )
        cb_log_va = np.expm1(np.maximum(0, cb_log.predict(X_va_raw)))
        cb_log_te = np.expm1(np.maximum(0, cb_log.predict(X_test_raw)))

        cb_raw.fit(
            X_tr_raw,
            y_tr_raw,
            eval_set=(X_va_raw, y_va_raw),
            verbose=0,
        )
        cb_raw_va = np.maximum(0, cb_raw.predict(X_va_raw))
        cb_raw_te = np.maximum(0, cb_raw.predict(X_test_raw))

        # 2. Train Tabular ResNet Neural Network
        print("  Training Tabular ResNet...")
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        X_te_scaled = scaler.transform(X_test_raw)

        tr_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr_log, dtype=torch.float32),
        )
        va_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va_log, dtype=torch.float32),
        )

        tr_loader = DataLoader(
            tr_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        va_loader = DataLoader(
            va_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        model, criterion, optimizer = build_model_criterion_optimizer(
            input_dim=len(feature_cols),
            lr=1e-3,
            weight_decay=1e-4,
        )
        model.to(device)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_rmse = float("inf")
        best_model_weights = None
        no_improve_epochs = 0

        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            total_samples = 0

            for x_b, y_b in tr_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * len(y_b)
                total_samples += len(y_b)

            scheduler.step()

            model.eval()
            val_preds_log_list = []
            with torch.no_grad():
                for x_b, _ in va_loader:
                    x_b = x_b.to(device)
                    out = model(x_b)
                    val_preds_log_list.append(out.cpu().numpy())

            val_preds_log = np.concatenate(val_preds_log_list)
            val_preds_raw = np.expm1(np.maximum(0, val_preds_log))
            val_rmse = np.sqrt(mean_squared_error(y_va_raw, val_preds_raw))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                break

        model.load_state_dict(best_model_weights)
        model.to(device)
        model.eval()

        with torch.no_grad():
            x_va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            nn_oof_log = model(x_va_tensor).cpu().numpy()
            nn_va_preds = np.expm1(np.maximum(0, nn_oof_log))

            x_te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            nn_test_log = model(x_te_tensor).cpu().numpy()
            nn_te_preds = np.expm1(np.maximum(0, nn_test_log))

        # Store individual model predictions for ensemble weight optimization
        fold_va_preds = np.column_stack([
            lgb_log_va, lgb_raw_va,
            xgb_log_va, xgb_raw_va,
            cb_log_va, cb_raw_va,
            nn_va_preds
        ])
        fold_te_preds = np.column_stack([
            lgb_log_te, lgb_raw_te,
            xgb_log_te, xgb_raw_te,
            cb_log_te, cb_raw_te,
            nn_te_preds
        ])

        oof_matrix[va_idx] = fold_va_preds
        test_matrix += fold_te_preds / num_folds

        fold_mean_blend = np.mean(fold_va_preds, axis=1)
        fold_rmse = np.sqrt(mean_squared_error(y_va_raw, fold_mean_blend))
        print(f"Fold {fold + 1} Mean Model Val RMSE ($): {fold_rmse:.4f}")

    # Optimize weights on OOF predictions using SLSQP constrained optimization to minimize raw RMSE
    print("Optimizing ensemble weights with SciPy SLSQP...")
    y_true_raw = train_df["target_raw"].values

    def loss_func(weights):
        pred = oof_matrix @ weights
        return np.sqrt(mean_squared_error(y_true_raw, pred))

    init_weights = np.ones(num_models) / float(num_models)
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = ({"type": "eq", "fun": lambda w: 1.0 - np.sum(w)})

    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_weights = res.x
    print(f"Optimal weights across 7 models: {np.round(opt_weights, 4)}")

    oof_preds = oof_matrix @ opt_weights
    test_preds = test_matrix @ opt_weights

    final_oof_rmse = np.sqrt(mean_squared_error(y_true_raw, oof_preds))

    submission_path = os.path.join(submission_dir, "submission.csv")
    sub = pd.DataFrame(
        {
            "id": test_df["id"].astype(int),
            "Federal Funds": np.maximum(0.0, test_preds),
        }
    )
    sub.to_csv(submission_path, index=False)

    print(f"Submission saved successfully to {submission_path} with shape {sub.shape}.")
    print(f"Final Validation Score: {final_oof_rmse}")


if __name__ == "__main__":
    main()
