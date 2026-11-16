import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ---------------------------------------------------------
# Global Configuration and Reproducibility
# ---------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data and Stratified Split Setup
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path).copy()
test_df = pd.read_parquet(test_path).copy()

target_col = "Amount in USD"

if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df), dtype=np.int64)

test_ids = test_df["id"].values

y_train = train_df[target_col].values.astype(np.float64)
y_log = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))

n_splits = 5
bin_edges = np.percentile(y_log, np.linspace(0, 100, 11))
bin_edges[-1] += 1e-5
y_binned = np.digitize(y_log, bin_edges) - 1
y_binned = np.clip(y_binned, 0, 9)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
train_df["fold"] = -1
for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(train_df, y_binned)):
    train_df.loc[val_idx, "fold"] = fold_idx

# ---------------------------------------------------------
# 2. Identify Column Modalities
# ---------------------------------------------------------
feature_cols = [c for c in train_df.columns if c not in [target_col, "fold", "id"]]

datetime_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train_df[col]):
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    else:
        sample_non_null = train_df[col].dropna()
        is_date = False
        if len(sample_non_null) > 0:
            sample_val = str(sample_non_null.iloc[0])
            if (
                len(sample_val) == 10 and sample_val[4] == "-" and sample_val[7] == "-"
            ) or (
                len(sample_val) == 10 and sample_val[2] == "/" and sample_val[5] == "/"
            ):
                try:
                    pd.to_datetime(sample_non_null.iloc[:50])
                    is_date = True
                except Exception:
                    is_date = False
        if is_date:
            datetime_cols.append(col)
        else:
            avg_len = (
                sample_non_null.astype(str).map(len).mean()
                if len(sample_non_null) > 0
                else 0
            )
            n_unique = sample_non_null.nunique()
            if avg_len > 40 or n_unique > len(sample_non_null) * 0.7:
                text_cols.append(col)
            else:
                cat_cols.append(col)


# ---------------------------------------------------------
# 3. Feature Extraction Functions
# ---------------------------------------------------------
def extract_datetime_features(df_train, df_test, dt_cols):
    train_feats = pd.DataFrame(index=df_train.index)
    test_feats = pd.DataFrame(index=df_test.index)

    for col in dt_cols:
        t_tr = pd.to_datetime(df_train[col], errors="coerce")
        t_te = pd.to_datetime(df_test[col], errors="coerce")

        min_date = t_tr.min()
        if pd.isna(min_date):
            min_date = pd.Timestamp("2000-01-01")

        for prefix, s_series in [("tr", t_tr), ("te", t_te)]:
            feats = pd.DataFrame(index=s_series.index)
            feats[f"{col}_year"] = s_series.dt.year.fillna(-1).astype(np.int32)
            feats[f"{col}_month"] = s_series.dt.month.fillna(-1).astype(np.int32)
            feats[f"{col}_day"] = s_series.dt.day.fillna(-1).astype(np.int32)
            feats[f"{col}_quarter"] = s_series.dt.quarter.fillna(-1).astype(np.int32)
            feats[f"{col}_dayofweek"] = s_series.dt.dayofweek.fillna(-1).astype(
                np.int32
            )
            feats[f"{col}_dayofyear"] = s_series.dt.dayofyear.fillna(-1).astype(
                np.int32
            )
            feats[f"{col}_is_month_end"] = s_series.dt.is_month_end.fillna(
                False
            ).astype(np.int8)
            feats[f"{col}_is_quarter_end"] = s_series.dt.is_quarter_end.fillna(
                False
            ).astype(np.int8)
            feats[f"{col}_fiscal_year"] = np.where(
                s_series.dt.month >= 7, s_series.dt.year + 1, s_series.dt.year
            )
            feats[f"{col}_fiscal_year"] = (
                feats[f"{col}_fiscal_year"].fillna(-1).astype(np.int32)
            )
            feats[f"{col}_days_elapsed"] = (
                (s_series - min_date).dt.days.fillna(-1).astype(np.float32)
            )

            if prefix == "tr":
                for c in feats.columns:
                    train_feats[c] = feats[c]
            else:
                for c in feats.columns:
                    test_feats[c] = feats[c]

    if len(dt_cols) >= 2:
        for i in range(len(dt_cols)):
            for j in range(i + 1, len(dt_cols)):
                col1, col2 = dt_cols[i], dt_cols[j]
                t_tr1 = pd.to_datetime(df_train[col1], errors="coerce")
                t_tr2 = pd.to_datetime(df_train[col2], errors="coerce")
                t_te1 = pd.to_datetime(df_test[col1], errors="coerce")
                t_te2 = pd.to_datetime(df_test[col2], errors="coerce")

                diff_tr = (t_tr1 - t_tr2).dt.days.fillna(0).astype(np.float32)
                diff_te = (t_te1 - t_te2).dt.days.fillna(0).astype(np.float32)

                train_feats[f"diff_days_{col1}_vs_{col2}"] = diff_tr
                test_feats[f"diff_days_{col1}_vs_{col2}"] = diff_te
                train_feats[f"abs_diff_days_{col1}_vs_{col2}"] = np.abs(diff_tr)
                test_feats[f"abs_diff_days_{col1}_vs_{col2}"] = np.abs(diff_te)

    return train_feats, test_feats


def extract_text_features(df_train, df_test, t_cols):
    train_feats = pd.DataFrame(index=df_train.index)
    test_feats = pd.DataFrame(index=df_test.index)

    for col in t_cols:
        s_tr = df_train[col].fillna("").astype(str)
        s_te = df_test[col].fillna("").astype(str)

        train_feats[f"{col}_char_len"] = s_tr.map(len).astype(np.float32)
        test_feats[f"{col}_char_len"] = s_te.map(len).astype(np.float32)
        train_feats[f"{col}_word_count"] = s_tr.map(lambda x: len(x.split())).astype(
            np.float32
        )
        test_feats[f"{col}_word_count"] = s_te.map(lambda x: len(x.split())).astype(
            np.float32
        )
        train_feats[f"{col}_cap_count"] = s_tr.map(
            lambda x: sum(1 for c in x if c.isupper())
        ).astype(np.float32)
        test_feats[f"{col}_cap_count"] = s_te.map(
            lambda x: sum(1 for c in x if c.isupper())
        ).astype(np.float32)
        train_feats[f"{col}_digit_count"] = s_tr.map(
            lambda x: sum(1 for c in x if c.isdigit())
        ).astype(np.float32)
        test_feats[f"{col}_digit_count"] = s_te.map(
            lambda x: sum(1 for c in x if c.isdigit())
        ).astype(np.float32)

        try:
            tfidf = TfidfVectorizer(
                max_features=100, stop_words="english", token_pattern=r"(?u)\b\w+\b"
            )
            svd = TruncatedSVD(
                n_components=min(5, len(df_train)), random_state=RANDOM_STATE
            )

            tr_vec = tfidf.fit_transform(s_tr)
            te_vec = tfidf.transform(s_te)

            tr_svd = svd.fit_transform(tr_vec)
            te_svd = svd.transform(te_vec)

            for k in range(tr_svd.shape[1]):
                train_feats[f"{col}_svd_{k}"] = tr_svd[:, k].astype(np.float32)
                test_feats[f"{col}_svd_{k}"] = te_svd[:, k].astype(np.float32)
        except Exception:
            pass

    return train_feats, test_feats


def extract_categorical_features(
    df_train_raw, df_test_raw, c_cols, y_target, folds, n_folds=5
):
    df_train = df_train_raw.copy()
    df_test = df_test_raw.copy()

    train_feats = pd.DataFrame(index=df_train.index)
    test_feats = pd.DataFrame(index=df_test.index)

    interaction_pairs = []
    if len(c_cols) >= 2:
        for i in range(len(c_cols)):
            for j in range(i + 1, min(i + 3, len(c_cols))):
                pair_name = f"{c_cols[i]}_x_{c_cols[j]}"
                interaction_pairs.append((c_cols[i], c_cols[j], pair_name))
                df_train[pair_name] = (
                    df_train[c_cols[i]].astype(str)
                    + "___"
                    + df_train[c_cols[j]].astype(str)
                )
                df_test[pair_name] = (
                    df_test[c_cols[i]].astype(str)
                    + "___"
                    + df_test[c_cols[j]].astype(str)
                )

    all_cats = list(c_cols) + [p[2] for p in interaction_pairs]

    global_target_mean = float(np.mean(y_target))
    log_target = np.log1p(np.clip(y_target, a_min=0.0, a_max=None))
    global_log_mean = float(np.mean(log_target))
    smooth_weight = 10.0

    for col in all_cats:
        s_tr = df_train[col].fillna("MISSING").astype(str)
        s_te = df_test[col].fillna("MISSING").astype(str)

        freq_map = s_tr.value_counts().to_dict()
        train_feats[f"{col}_freq"] = s_tr.map(freq_map).fillna(0).astype(np.float32)
        test_feats[f"{col}_freq"] = s_te.map(freq_map).fillna(0).astype(np.float32)

        train_len = len(df_train)
        train_feats[f"{col}_freq_ratio"] = (
            train_feats[f"{col}_freq"] / train_len
        ).astype(np.float32)
        test_feats[f"{col}_freq_ratio"] = (
            test_feats[f"{col}_freq"] / train_len
        ).astype(np.float32)

        categories = sorted(list(set(s_tr.unique()).union(set(s_te.unique()))))
        cat_to_code = {cat: idx for idx, cat in enumerate(categories)}
        train_feats[f"{col}_code"] = s_tr.map(cat_to_code).fillna(-1).astype(np.int32)
        test_feats[f"{col}_code"] = s_te.map(cat_to_code).fillna(-1).astype(np.int32)

        oof_te_mean = np.full(len(df_train), global_target_mean, dtype=np.float32)
        oof_te_log_mean = np.full(len(df_train), global_log_mean, dtype=np.float32)
        oof_te_std = np.full(len(df_train), 0.0, dtype=np.float32)

        for f_idx in range(n_folds):
            trn_idx = folds != f_idx
            val_idx = folds == f_idx

            trn_series = s_tr[trn_idx]
            trn_y = y_target[trn_idx]
            trn_log_y = log_target[trn_idx]

            stats = (
                pd.DataFrame({"cat": trn_series, "y": trn_y, "log_y": trn_log_y})
                .groupby("cat")
                .agg(
                    count=("y", "count"), sum_y=("y", "sum"), sum_log_y=("log_y", "sum")
                )
            )

            stats["smooth_mean"] = (
                stats["sum_y"] + smooth_weight * global_target_mean
            ) / (stats["count"] + smooth_weight)
            stats["smooth_log_mean"] = (
                stats["sum_log_y"] + smooth_weight * global_log_mean
            ) / (stats["count"] + smooth_weight)

            mean_dict = stats["smooth_mean"].to_dict()
            log_mean_dict = stats["smooth_log_mean"].to_dict()
            std_dict = (
                pd.DataFrame({"cat": trn_series, "y": trn_y})
                .groupby("cat")["y"]
                .std()
                .fillna(0.0)
                .to_dict()
            )

            val_cats = s_tr[val_idx]
            oof_te_mean[val_idx] = (
                val_cats.map(mean_dict).fillna(global_target_mean).values
            )
            oof_te_log_mean[val_idx] = (
                val_cats.map(log_mean_dict).fillna(global_log_mean).values
            )
            oof_te_std[val_idx] = val_cats.map(std_dict).fillna(0.0).values

        train_feats[f"{col}_te_mean"] = oof_te_mean
        train_feats[f"{col}_te_log_mean"] = oof_te_log_mean
        train_feats[f"{col}_te_std"] = oof_te_std

        full_stats = (
            pd.DataFrame({"cat": s_tr, "y": y_target, "log_y": log_target})
            .groupby("cat")
            .agg(count=("y", "count"), sum_y=("y", "sum"), sum_log_y=("log_y", "sum"))
        )
        full_stats["smooth_mean"] = (
            full_stats["sum_y"] + smooth_weight * global_target_mean
        ) / (full_stats["count"] + smooth_weight)
        full_stats["smooth_log_mean"] = (
            full_stats["sum_log_y"] + smooth_weight * global_log_mean
        ) / (full_stats["count"] + smooth_weight)

        full_mean_dict = full_stats["smooth_mean"].to_dict()
        full_log_mean_dict = full_stats["smooth_log_mean"].to_dict()
        full_std_dict = (
            pd.DataFrame({"cat": s_tr, "y": y_target})
            .groupby("cat")["y"]
            .std()
            .fillna(0.0)
            .to_dict()
        )

        test_feats[f"{col}_te_mean"] = (
            s_te.map(full_mean_dict).fillna(global_target_mean).astype(np.float32)
        )
        test_feats[f"{col}_te_log_mean"] = (
            s_te.map(full_log_mean_dict).fillna(global_log_mean).astype(np.float32)
        )
        test_feats[f"{col}_te_std"] = (
            s_te.map(full_std_dict).fillna(0.0).astype(np.float32)
        )

    return train_feats, test_feats


def extract_numeric_features(df_train, df_test, n_cols):
    train_feats = pd.DataFrame(index=df_train.index)
    test_feats = pd.DataFrame(index=df_test.index)

    for col in n_cols:
        median_val = df_train[col].median()
        train_feats[f"{col}_imputed"] = (
            df_train[col].fillna(median_val).astype(np.float32)
        )
        test_feats[f"{col}_imputed"] = (
            df_test[col].fillna(median_val).astype(np.float32)
        )
        train_feats[f"{col}_isna"] = df_train[col].isna().astype(np.int8)
        test_feats[f"{col}_isna"] = df_test[col].isna().astype(np.int8)

    return train_feats, test_feats


# ---------------------------------------------------------
# 4. Feature Extraction Pipeline Execution
# ---------------------------------------------------------
dt_train, dt_test = extract_datetime_features(train_df, test_df, datetime_cols)
text_train, text_test = extract_text_features(train_df, test_df, text_cols)
cat_train, cat_test = extract_categorical_features(
    train_df, test_df, cat_cols, y_train, train_df["fold"].values, n_splits
)
num_train, num_test = extract_numeric_features(train_df, test_df, num_cols)

X_train_df = pd.concat([dt_train, text_train, cat_train, num_train], axis=1)
X_test_df = pd.concat([dt_test, text_test, cat_test, num_test], axis=1)

common_cols = [c for c in X_train_df.columns if c in X_test_df.columns]
X_train_df = X_train_df[common_cols]
X_test_df = X_test_df[common_cols]

# Replace residual non-finites and convert to numpy
X = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
X_test = (
    X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
)
y = y_train.copy()
folds = train_df["fold"].values


# ---------------------------------------------------------
# 5. Model Architecture Definitions
# ---------------------------------------------------------
def create_lgb_regressor():
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMRegressor(**params)


def create_xgb_regressor():
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }
    return xgb.XGBRegressor(**params)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dropout1(self.act1(self.ln1(self.linear1(x))))
        out = self.dropout2(self.act2(self.ln2(self.linear2(out))))
        return out + residual


class TabularResidualRegressor(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.15,
        output_dim=1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                ResidualBlock(
                    hidden_dims[i], hidden_dims[i + 1], dropout_rate=dropout_rate
                )
            )
        self.residual_stack = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_stack(x)
        out = self.head(x)
        return out.squeeze(-1)


class RMSELoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


# ---------------------------------------------------------
# 6. Model Training & Cross-Validation
# ---------------------------------------------------------
oof_preds_lgb = np.zeros(len(train_df), dtype=np.float64)
test_preds_lgb_folds = np.zeros((n_splits, len(test_df)), dtype=np.float64)

for fold_idx in range(n_splits):
    trn_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_trn, y_trn = X[trn_mask], y[trn_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    model = create_lgb_regressor()
    model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_preds = model.predict(X_val)
    oof_preds_lgb[val_mask] = val_preds
    test_preds_lgb_folds[fold_idx] = model.predict(X_test)

oof_preds_xgb = np.zeros(len(train_df), dtype=np.float64)
test_preds_xgb_folds = np.zeros((n_splits, len(test_df)), dtype=np.float64)

for fold_idx in range(n_splits):
    trn_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_trn, y_trn = X[trn_mask], y[trn_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    model_xgb = create_xgb_regressor()
    model_xgb.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = model_xgb.predict(X_val)
    oof_preds_xgb[val_mask] = val_preds
    test_preds_xgb_folds[fold_idx] = model_xgb.predict(X_test)

oof_preds_nn = np.zeros(len(train_df), dtype=np.float64)
test_preds_nn_folds = np.zeros((n_splits, len(test_df)), dtype=np.float64)
nn_epochs = 70
batch_size = 64

for fold_idx in range(n_splits):
    trn_mask = folds != fold_idx
    val_mask = folds == fold_idx

    scaler = StandardScaler()
    X_trn_scaled = scaler.fit_transform(X[trn_mask]).astype(np.float32)
    X_val_scaled = scaler.transform(X[val_mask]).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    y_trn = y[trn_mask].astype(np.float32)
    y_val = y[val_mask].astype(np.float32)

    trn_dataset = TensorDataset(
        torch.tensor(X_trn_scaled), torch.tensor(y_trn, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled), torch.tensor(y_val, dtype=torch.float32)
    )

    trn_loader = DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    nn_model = TabularResidualRegressor(
        in_features=X.shape[1], hidden_dims=[256, 128, 64], dropout_rate=0.15
    ).to(device)
    criterion = RMSELoss()
    optimizer = optim.AdamW(
        nn_model.parameters(), lr=3e-3, weight_decay=1e-5, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nn_epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(nn_epochs):
        nn_model.train()
        for batch_x, batch_y in trn_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_preds_list.append(nn_model(batch_x).cpu().numpy())

        epoch_val_preds = np.concatenate(val_preds_list)
        epoch_val_loss = root_mean_squared_error(y_val, epoch_val_preds)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    nn_model.eval()

    with torch.no_grad():
        val_preds_fold = nn_model(torch.tensor(X_val_scaled).to(device)).cpu().numpy()
        test_preds_fold = nn_model(torch.tensor(X_test_scaled).to(device)).cpu().numpy()

    oof_preds_nn[val_mask] = val_preds_fold
    test_preds_nn_folds[fold_idx] = test_preds_fold

# ---------------------------------------------------------
# 7. Out-Of-Fold Ensembling and Submission Generation
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_nn])
test_lgb_avg = np.mean(test_preds_lgb_folds, axis=0)
test_xgb_avg = np.mean(test_preds_xgb_folds, axis=0)
test_nn_avg = np.mean(test_preds_nn_folds, axis=0)
test_matrix = np.column_stack([test_lgb_avg, test_xgb_avg, test_nn_avg])

def ensemble_loss(weights):
    weights = np.array(weights)
    pred = np.dot(oof_matrix, weights)
    pred_clipped = np.clip(pred, 0.0, None)
    return root_mean_squared_error(y, pred_clipped)

init_weights = np.ones(3) / 3.0
bounds = [(0.0, 1.0) for _ in range(3)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-9},
)

if opt_res.success and np.isfinite(opt_res.fun):
    opt_weights = opt_res.x / np.sum(opt_res.x)
else:
    opt_weights = init_weights

final_oof_preds = np.clip(np.dot(oof_matrix, opt_weights), 0.0, None)
final_score = root_mean_squared_error(y, final_oof_preds)

final_test_preds = np.clip(np.dot(test_matrix, opt_weights), 0.0, None)

submission_df = pd.DataFrame({"id": test_ids, "Amount in USD": final_test_preds})
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")
