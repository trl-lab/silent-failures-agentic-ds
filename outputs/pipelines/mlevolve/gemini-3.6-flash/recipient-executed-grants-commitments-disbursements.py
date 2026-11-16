import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==========================================
# 1. PyTorch Neural Network Architecture
# ==========================================
class TabularGatedResidualBlock(nn.Module):
    """Gated Residual Block for Tabular Feature Representations."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.norm2(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        g = self.gate(x)
        return residual + g * out


class TabularDeepFeatureNet(nn.Module):
    """Deep Neural Network with LayerNorm, Gated Residual Blocks, and Multi-stage Reduction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_block1 = TabularGatedResidualBlock(
            hidden_dims[0], hidden_dims[0], dropout_rate
        )

        self.transition = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_block2 = TabularGatedResidualBlock(
            hidden_dims[1], hidden_dims[1], dropout_rate
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.transition(x)
        x = self.res_block2(x)
        out = self.head(x)
        return out.squeeze(-1)


# ==========================================
# 2. Data Processing & Feature Engineering
# ==========================================
def process_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Disbursements USD (Trust funds)"
    if target_col not in train_df.columns:
        possible_targets = [
            c
            for c in train_df.columns
            if "disbursement" in c.lower() or "usd" in c.lower()
        ]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            target_col = train_df.columns[-1]

    y_train = train_df[target_col].values
    train_features = train_df.drop(columns=[target_col], errors="ignore").copy()
    test_features = test_df.drop(columns=[target_col], errors="ignore").copy()

    if "id" in train_features.columns:
        test_ids = test_features["id"].values
    else:
        test_ids = np.arange(len(test_features))

    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    datetime_cols = []
    numeric_cols = []
    categorical_cols = []
    text_cols = []

    for col in combined.columns:
        if col.lower() == "id":
            continue
        dtype = combined[col].dtype
        col_lower = col.lower()
        if (
            np.issubdtype(dtype, np.datetime64)
            or "date" in col_lower
            or "dt" in col_lower
        ):
            datetime_cols.append(col)
        elif np.issubdtype(dtype, np.number):
            numeric_cols.append(col)
        else:
            str_lengths = combined[col].astype(str).str.len().mean()
            num_unique = combined[col].nunique()
            if str_lengths > 50 and num_unique > 50:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    dt_features = pd.DataFrame(index=combined.index)
    parsed_date_cols = []
    for col in datetime_cols:
        try:
            dt_series = pd.to_datetime(combined[col], errors="coerce")
            dt_features[f"{col}_year"] = dt_series.dt.year
            dt_features[f"{col}_month"] = dt_series.dt.month
            dt_features[f"{col}_quarter"] = dt_series.dt.quarter
            parsed_date_cols.append(dt_series)
        except Exception:
            pass

    if len(parsed_date_cols) >= 2:
        for i in range(len(parsed_date_cols)):
            for j in range(i + 1, len(parsed_date_cols)):
                diff_days = (parsed_date_cols[j] - parsed_date_cols[i]).dt.days
                dt_features[f"date_diff_{i}_{j}_days"] = diff_days

    for col in dt_features.columns:
        median_val = dt_features.iloc[:n_train][col].median()
        dt_features[col] = dt_features[col].fillna(
            median_val if not np.isnan(median_val) else -1
        )

    num_df = pd.DataFrame(index=combined.index)
    for col in numeric_cols:
        val = combined[col].astype(float)
        num_df[col] = val
        num_df[f"{col}_isna"] = combined[col].isna().astype(float)
        if (val.dropna() >= 0).all() and val.max() > 100:
            num_df[f"{col}_log1p"] = np.log1p(np.maximum(0, val))

    num_cols_list = list(numeric_cols)
    if len(num_cols_list) >= 2:
        for i in range(min(5, len(num_cols_list))):
            for j in range(i + 1, min(5, len(num_cols_list))):
                col_i = num_cols_list[i]
                col_j = num_cols_list[j]
                v_i = num_df[col_i].replace(0, np.nan)
                v_j = num_df[col_j].replace(0, np.nan)
                num_df[f"ratio_{col_i}_vs_{col_j}"] = v_i / v_j
                num_df[f"diff_{col_i}_vs_{col_j}"] = num_df[col_i] - num_df[col_j]

    for col in num_df.columns:
        num_df[col] = num_df[col].replace([np.inf, -np.inf], np.nan)
        med = num_df.iloc[:n_train][col].median()
        num_df[col] = num_df[col].fillna(med if not np.isnan(med) else 0)

    group_df = pd.DataFrame(index=combined.index)
    top_categoricals = [c for c in categorical_cols if combined[c].nunique() > 1][:5]
    top_numerics = [c for c in numeric_cols][:5]

    for cat_col in top_categoricals:
        train_cat = combined.iloc[:n_train][[cat_col] + top_numerics]
        for num_col in top_numerics:
            stats = (
                train_cat.groupby(cat_col)[num_col]
                .agg(["mean", "std", "min", "max", "sum"])
                .reset_index()
            )
            stats.columns = [
                cat_col,
                f"{cat_col}_{num_col}_mean",
                f"{cat_col}_{num_col}_std",
                f"{cat_col}_{num_col}_min",
                f"{cat_col}_{num_col}_max",
                f"{cat_col}_{num_col}_sum",
            ]
            merged = combined[[cat_col]].merge(stats, on=cat_col, how="left")
            for stat_col in stats.columns:
                if stat_col != cat_col:
                    group_df[stat_col] = merged[stat_col].values
                    med = group_df.iloc[:n_train][stat_col].median()
                    group_df[stat_col] = group_df[stat_col].fillna(
                        med if not np.isnan(med) else 0
                    )

    kf_te = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_log_te = np.log1p(np.maximum(0, y_train))
    smoothing = 10.0

    te_df = pd.DataFrame(index=combined.index)
    for col in categorical_cols:
        cat_series_train = train_features[col].astype(str).fillna("missing")
        cat_series_test = test_features[col].astype(str).fillna("missing")

        oof_te = np.zeros(n_train)
        test_te = np.zeros(len(test_features))

        for tr_idx, val_idx in kf_te.split(train_features):
            tr_cats = cat_series_train.iloc[tr_idx]
            tr_y = y_train_log_te[tr_idx]

            global_mean = tr_y.mean()

            cat_df_tr = pd.DataFrame({"cat": tr_cats, "y": tr_y})
            grouped = cat_df_tr.groupby("cat")["y"].agg(["count", "mean"])

            smooth_map = (
                grouped["count"] * grouped["mean"] + smoothing * global_mean
            ) / (grouped["count"] + smoothing)
            smooth_dict = smooth_map.to_dict()

            oof_te[val_idx] = (
                cat_series_train.iloc[val_idx]
                .map(smooth_dict)
                .fillna(global_mean)
                .values
            )

            test_te += (
                cat_series_test.map(smooth_dict).fillna(global_mean).values
                / kf_te.n_splits
            )

        te_df[f"{col}_target_enc"] = np.concatenate([oof_te, test_te])

    cat_df = pd.DataFrame(index=combined.index)
    for col in categorical_cols:
        str_vals = combined[col].astype(str).fillna("missing")
        freq_map = str_vals.iloc[:n_train].value_counts().to_dict()
        cat_df[f"{col}_freq"] = str_vals.map(freq_map).fillna(0)

        le = LabelEncoder()
        cat_df[f"{col}_encoded"] = le.fit_transform(str_vals)

    text_df = pd.DataFrame(index=combined.index)
    for col in text_cols:
        text_data = combined[col].astype(str).fillna("")
        tfidf_word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=1000, stop_words="english")
        tfidf_char = TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb", sublinear_tf=True, max_features=1000)

        tfidf_word.fit(text_data.iloc[:n_train])
        word_mat = tfidf_word.transform(text_data)

        tfidf_char.fit(text_data.iloc[:n_train])
        char_mat = tfidf_char.transform(text_data)

        tfidf_mat = sp.hstack([word_mat, char_mat]).tocsr()

        n_comp = min(16, tfidf_mat.shape[1] - 1)
        if n_comp > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(tfidf_mat[:n_train])
            svd_feats = svd.transform(tfidf_mat)
            for k in range(n_comp):
                text_df[f"{col}_svd_{k}"] = svd_feats[:, k]

    X_all = pd.concat([num_df, dt_features, group_df, cat_df, te_df, text_df], axis=1)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    X_final = X_all
    clean_cols = [re.sub(r"[^\w_]", "_", str(col)) for col in X_final.columns]
    seen = {}
    cols = []
    for col in clean_cols:
        if col in seen:
            seen[col] += 1
            cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            cols.append(col)
    X_final.columns = cols

    X_tr_proc = X_final.iloc[:n_train].values
    X_te_proc = X_final.iloc[n_train:].values

    folds = list(kf.split(X_tr_proc))

    return X_tr_proc, y_train, X_te_proc, test_ids, folds


# ==========================================
# 3. Model Training & Evaluation Pipeline
# ==========================================
def run_pipeline():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    X_train, y_train, X_test, test_ids, folds = process_data()

    n_samples = len(X_train)
    n_test = len(X_test)
    n_folds = len(folds)

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cb = np.zeros(n_samples)
    oof_nn = np.zeros(n_samples)

    test_preds_lgb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)
    test_preds_cb = np.zeros(n_test)
    test_preds_nn = np.zeros(n_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        y_tr_log = np.log1p(np.maximum(0, y_tr))
        y_val_log = np.log1p(np.maximum(0, y_val))

        # 1. LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold_idx,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ],
        )
        oof_lgb[val_idx] = np.maximum(0, np.expm1(lgb_model.predict(X_val)))
        test_preds_lgb += np.maximum(0, np.expm1(lgb_model.predict(X_test))) / n_folds

        # 2. XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold_idx,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
        oof_xgb[val_idx] = np.maximum(0, np.expm1(xgb_model.predict(X_val)))
        test_preds_xgb += np.maximum(0, np.expm1(xgb_model.predict(X_test))) / n_folds

        # 3. CatBoost Regressor
        cb_model = cb.CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold_idx,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), verbose=False)
        oof_cb[val_idx] = np.maximum(0, np.expm1(cb_model.predict(X_val)))
        test_preds_cb += np.maximum(0, np.expm1(cb_model.predict(X_test))) / n_folds

        # 4. PyTorch Tabular Deep Feature Net
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_scaled), torch.FloatTensor(y_tr_log)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_log)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=512, shuffle=False, drop_last=False
        )

        nn_model = TabularDeepFeatureNet(
            input_dim=X_tr.shape[1], hidden_dims=[256, 128], dropout_rate=0.2
        ).to(device)
        criterion = nn.SmoothL1Loss(beta=1.0)
        optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        best_nn_weights = None
        patience = 15
        patience_counter = 0

        for epoch in range(100):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = nn_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            nn_model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    out = nn_model(batch_x)
                    val_loss += nn.MSELoss()(out, batch_y).item() * len(batch_y)
                    val_count += len(batch_y)

            val_rmse = np.sqrt(val_loss / val_count)
            scheduler.step(val_rmse)

            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_nn_weights is not None:
            nn_model.load_state_dict(
                {k: v.to(device) for k, v in best_nn_weights.items()}
            )
        nn_model.eval()

        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            oof_nn[val_idx] = np.maximum(0, np.expm1(nn_model(val_tensor).cpu().numpy()))

            test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            test_preds_nn += np.maximum(0, np.expm1(nn_model(test_tensor).cpu().numpy())) / n_folds

    # Post-process predictions to non-negative monetary amounts
    oof_lgb = np.maximum(0, oof_lgb)
    oof_xgb = np.maximum(0, oof_xgb)
    oof_cb = np.maximum(0, oof_cb)
    oof_nn = np.maximum(0, oof_nn)

    # SLSQP Weight Optimization
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])

    def loss_func(weights):
        blend = np.dot(oof_matrix, weights)
        return np.sqrt(mean_squared_error(y_train, blend))

    init_weights = np.ones(4) / 4.0
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    opt_res = opt.minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x

    oof_blend = np.maximum(0, np.dot(oof_matrix, best_weights))
    final_score = np.sqrt(mean_squared_error(y_train, oof_blend))

    test_preds_lgb = np.maximum(0, test_preds_lgb)
    test_preds_xgb = np.maximum(0, test_preds_xgb)
    test_preds_cb = np.maximum(0, test_preds_cb)
    test_preds_nn = np.maximum(0, test_preds_nn)

    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_nn]
    )
    final_test_preds = np.maximum(0, np.dot(test_matrix, best_weights))

    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "Disbursements USD (Trust funds)": final_test_preds,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    run_pipeline()