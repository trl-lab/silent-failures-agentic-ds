import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Ensure working directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data & Initial Setup
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)

target_col = "Amount in USD"
if target_col not in train_raw.columns:
    for col in train_raw.columns:
        if "amount" in col.lower() and "usd" in col.lower():
            target_col = col
            break

y_train_raw = train_raw[target_col].values.astype(np.float64)
train_df = train_raw.drop(columns=[target_col]).copy()
test_df = test_raw.copy()

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

if "id" in train_df.columns:
    train_df = train_df.drop(columns=["id"])
if "id" in test_df.columns:
    test_df = test_df.drop(columns=["id"])

n_train = len(train_df)
df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 2. Datetime Feature Extraction
datetime_cols = []
for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        datetime_cols.append(col)
    elif df_all[col].dtype == "object":
        sample_vals = df_all[col].dropna().astype(str).head(20)
        if len(sample_vals) > 0 and all(
            re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", v) for v in sample_vals
        ):
            try:
                df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
                datetime_cols.append(col)
            except Exception:
                pass

for col in datetime_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    min_date = df_all[col].min()
    if pd.notna(min_date):
        df_all[f"{col}_elapsed_days"] = (df_all[col] - min_date).dt.total_seconds() / (
            24 * 3600
        )
    df_all = df_all.drop(columns=[col])

# 3. Categorical, Text, & Numerical Feature Processing
text_cols = []
cat_cols = []
num_cols = []

for col in df_all.columns:
    if df_all[col].dtype == "object" or isinstance(
        df_all[col].dtype, pd.CategoricalDtype
    ):
        avg_len = (
            df_all[col].astype(str).str.len().mean() if len(df_all[col]) > 0 else 0
        )
        if avg_len > 30 and df_all[col].nunique() > 100:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# Separate Train and Test Data first to prevent leakage
X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

# Text Processing (TF-IDF + TruncatedSVD) fitted on Train only
if text_cols:
    train_text = X_train[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    test_text = X_test[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_tr = tfidf.fit_transform(train_text)
    tfidf_te = tfidf.transform(test_text)
    n_comp = min(10, tfidf_tr.shape[1])
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_tr = svd.fit_transform(tfidf_tr)
        svd_te = svd.transform(tfidf_te)
        for i in range(n_comp):
            col_name = f"text_svd_{i}"
            X_train[col_name] = svd_tr[:, i]
            X_test[col_name] = svd_te[:, i]
            num_cols.append(col_name)
    X_train = X_train.drop(columns=text_cols)
    X_test = X_test.drop(columns=text_cols)

# Frequency Encoding for Categoricals fitted on Train only
for col in cat_cols:
    freq = X_train[col].value_counts(normalize=True).to_dict()
    X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)
    num_cols.append(f"{col}_freq")

# Missingness Indicators & Numerical Imputation fitted on Train only
for col in num_cols:
    if X_train[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
        X_train[f"{col}_isna"] = X_train[col].isnull().astype(int)
        X_test[f"{col}_isna"] = X_test[col].isnull().astype(int)
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

# Group Statistics Features fitted on Train only
for c_col in cat_cols:
    if 2 <= X_train[c_col].nunique() <= 200:
        for n_col in num_cols[:3]:
            if not n_col.endswith("_isna") and not n_col.endswith("_freq"):
                grp = X_train.groupby(c_col)[n_col].agg(["mean", "std"])
                grp.columns = [
                    f"{c_col}_{n_col}_grp_mean",
                    f"{c_col}_{n_col}_grp_std",
                ]
                X_train = X_train.join(grp, on=c_col)
                X_test = X_test.join(grp, on=c_col)
                X_train[f"{c_col}_{n_col}_grp_mean"] = X_train[f"{c_col}_{n_col}_grp_mean"].fillna(0)
                X_train[f"{c_col}_{n_col}_grp_std"] = X_train[f"{c_col}_{n_col}_grp_std"].fillna(0)
                X_test[f"{c_col}_{n_col}_grp_mean"] = X_test[f"{c_col}_{n_col}_grp_mean"].fillna(0)
                X_test[f"{c_col}_{n_col}_grp_std"] = X_test[f"{c_col}_{n_col}_grp_std"].fillna(0)

# Encode String/Categorical columns consistently across Train & Test
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[c], X_test[c]], axis=0).astype(str)
    le.fit(combined)
    X_train[c] = le.transform(X_train[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

y_train_log = np.log1p(np.maximum(0, y_train_raw))


# 4. PyTorch Tabular ResNet Neural Network Architecture
class TabularResNet(nn.Module):

    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)

        self.res_fc1 = nn.Linear(256, 256)
        self.res_bn1 = nn.BatchNorm1d(256)
        self.res_act = nn.SiLU()
        self.res_drop = nn.Dropout(dropout)
        self.res_fc2 = nn.Linear(256, 256)
        self.res_bn2 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        res = x
        x = self.res_fc1(x)
        x = self.res_bn1(x)
        x = self.res_act(x)
        x = self.res_drop(x)
        x = self.res_fc2(x)
        x = self.res_bn2(x)
        x = self.act1(x + res)
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.out(x)


# 5. Validation Setup & Stratified Cross-Validation Loop
num_bins = min(10, len(np.unique(y_train_log)))
target_bins = pd.qcut(y_train_log, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_test = len(X_test)
num_models = 5

oof_preds_matrix = np.zeros((n_train, num_models))
test_preds_matrix = np.zeros((n_test, num_models))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, target_bins)):
    X_tr = X_train.iloc[train_idx].copy()
    y_tr_log = y_train_log[train_idx]
    y_tr_raw = y_train_raw[train_idx]

    X_va = X_train.iloc[val_idx].copy()
    y_va_log = y_train_log[val_idx]
    y_va_raw = y_train_raw[val_idx]

    X_te = X_test.copy()

    # Dynamic Out-of-fold Target Encoding
    for c_col in cat_cols:
        target_map = (
            X_tr.assign(target=y_tr_log).groupby(c_col)["target"].mean().to_dict()
        )
        global_mean = y_tr_log.mean()
        X_tr[f"{c_col}_te"] = X_tr[c_col].map(target_map).fillna(global_mean)
        X_va[f"{c_col}_te"] = X_va[c_col].map(target_map).fillna(global_mean)
        X_te[f"{c_col}_te"] = X_te[c_col].map(target_map).fillna(global_mean)

    # Scale-weighting factors aligned with raw dollar RMSE gradients
    w_heavy = (1.0 + np.maximum(0, y_tr_raw)) ** 0.5
    w_heavy = w_heavy / np.mean(w_heavy)

    w_moderate = (1.0 + np.maximum(0, y_tr_raw)) ** 0.35
    w_moderate = w_moderate / np.mean(w_moderate)

    w_light = (1.0 + np.maximum(0, y_tr_raw)) ** 0.25
    w_light = w_light / np.mean(w_light)

    w_nn = (1.0 + np.maximum(0, y_tr_raw)) ** 0.3
    w_nn = w_nn / np.mean(w_nn)

    # Model 0: LightGBM Heavy Scale-Weighted
    lgb_heavy = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=42 + fold,
        n_estimators=1000,
        verbose=-1,
    )
    lgb_heavy.fit(
        X_tr,
        y_tr_log,
        sample_weight=w_heavy,
        eval_set=[(X_va, y_va_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_preds_matrix[val_idx, 0] = np.expm1(np.maximum(0, lgb_heavy.predict(X_va)))
    test_preds_matrix[:, 0] += (
        np.expm1(np.maximum(0, lgb_heavy.predict(X_te))) / skf.n_splits
    )

    # Model 1: LightGBM Standard Unweighted
    lgb_std = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=100 + fold,
        n_estimators=1000,
        verbose=-1,
    )
    lgb_std.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_preds_matrix[val_idx, 1] = np.expm1(np.maximum(0, lgb_std.predict(X_va)))
    test_preds_matrix[:, 1] += (
        np.expm1(np.maximum(0, lgb_std.predict(X_te))) / skf.n_splits
    )

    # Model 2: XGBoost Moderate Scale-Weighted
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr,
        y_tr_log,
        sample_weight=w_moderate,
        eval_set=[(X_va, y_va_log)],
        verbose=False,
    )
    oof_preds_matrix[val_idx, 2] = np.expm1(np.maximum(0, xgb_model.predict(X_va)))
    test_preds_matrix[:, 2] += (
        np.expm1(np.maximum(0, xgb_model.predict(X_te))) / skf.n_splits
    )

    # Model 3: CatBoost Light Scale-Weighted
    cat_model = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(
        X_tr,
        y_tr_log,
        sample_weight=w_light,
        eval_set=(X_va, y_va_log),
        verbose=False,
    )
    oof_preds_matrix[val_idx, 3] = np.expm1(np.maximum(0, cat_model.predict(X_va)))
    test_preds_matrix[:, 3] += (
        np.expm1(np.maximum(0, cat_model.predict(X_te))) / skf.n_splits
    )

    # Model 4: PyTorch Tabular ResNet with Weighted Loss
    scaler = StandardScaler()
    X_tr_scaled = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
    X_va_scaled = np.nan_to_num(scaler.transform(X_va), nan=0.0)
    X_te_scaled = np.nan_to_num(scaler.transform(X_te), nan=0.0)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1),
        torch.tensor(w_nn, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=min(64, len(X_tr_scaled)), shuffle=True
    )

    nn_model = TabularResNet(input_dim=X_tr_scaled.shape[1], dropout=0.2).to(device)
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(80):
        nn_model.train()
        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = torch.mean(bw * nn.functional.huber_loss(out, by, reduction="none"))
            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            vy = torch.tensor(y_va_log, dtype=torch.float32).unsqueeze(1).to(device)
            val_out = nn_model(vx)
            val_loss = torch.mean(
                nn.functional.huber_loss(val_out, vy, reduction="none")
            ).item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = nn_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 12:
                break

    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
    nn_model.eval()
    with torch.no_grad():
        val_preds_nn = (
            nn_model(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .squeeze()
        )
        test_preds_nn = (
            nn_model(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .squeeze()
        )

    oof_preds_matrix[val_idx, 4] = np.expm1(np.maximum(0, val_preds_nn))
    test_preds_matrix[:, 4] += np.expm1(np.maximum(0, test_preds_nn)) / skf.n_splits


# 6. Level-2 Meta Feature Stacking Framework
def build_meta_features(preds_mat):
    raw_preds = np.maximum(0, preds_mat)
    log_preds = np.log1p(raw_preds)
    mean_log = np.mean(log_preds, axis=1, keepdims=True)
    std_log = np.std(log_preds, axis=1, keepdims=True)
    max_raw = np.max(raw_preds, axis=1, keepdims=True)
    min_raw = np.min(raw_preds, axis=1, keepdims=True)
    return np.hstack([raw_preds, log_preds, mean_log, std_log, max_raw, min_raw])


meta_X_train = build_meta_features(oof_preds_matrix)
meta_X_test = build_meta_features(test_preds_matrix)

meta_learner = RidgeCV(alphas=np.logspace(-3, 5, 50))
meta_learner.fit(meta_X_train, y_train_raw)

oof_stacked = np.maximum(0, meta_learner.predict(meta_X_train))
test_stacked = np.maximum(0, meta_learner.predict(meta_X_test))


# 7. Level-3 Affine Post-Processing Scale Alignment
def calibration_obj(params):
    a, b = params
    calibrated = np.maximum(0, a * oof_stacked + b)
    return root_mean_squared_error(y_train_raw, calibrated)


res_cal = minimize(
    calibration_obj,
    [1.0, 0.0],
    method="Nelder-Mead",
    bounds=[(0.5, 2.0), (-1e6, 1e6)],
)
best_a, best_b = res_cal.x

final_oof_preds = np.maximum(0, best_a * oof_stacked + best_b)
final_test_preds = np.maximum(0, best_a * test_stacked + best_b)

# 8. Output Submission and Print Final Validation Score
sub = pd.DataFrame({"id": test_ids, "Amount in USD": final_test_preds})
sub.to_csv("./submission/submission.csv", index=False)

final_score = root_mean_squared_error(y_train_raw, final_oof_preds)
print(f"Final Validation Score: {final_score}")