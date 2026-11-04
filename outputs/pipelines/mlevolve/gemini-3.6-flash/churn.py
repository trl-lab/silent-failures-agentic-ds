import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# 1. Reproducibility & Setup
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


seed_everything(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading & Feature Engineering
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

raw_train_df = pd.read_parquet(train_path)
raw_test_df = pd.read_parquet(test_path)

# Standardize column names
raw_train_df.columns = (
    raw_train_df.columns.str.strip().str.lower().str.replace(" ", "_")
)
raw_test_df.columns = raw_test_df.columns.str.strip().str.lower().str.replace(" ", "_")

target_col = "class"
if raw_train_df[target_col].dtype == object:
    y = (
        raw_train_df[target_col]
        .astype(str)
        .str.lower()
        .isin(["true", "true.", "1", "yes", "churn"])
        .astype(np.float32)
        .values
    )
else:
    y = raw_train_df[target_col].astype(np.float32).values

raw_train_df = raw_train_df.drop(columns=[target_col])
if target_col in raw_test_df.columns:
    raw_test_df = raw_test_df.drop(columns=[target_col])


def process_data(df, is_train=True, train_medians=None, train_freqs=None):
    df_feat = df.copy()
    cols = df_feat.columns

    for col in cols:
        if "international_plan" in col or "intl_plan" in col:
            df_feat[col] = (
                df_feat[col]
                .astype(str)
                .str.lower()
                .isin(["yes", "true", "1", "t"])
                .astype(float)
            )
        elif "voice_mail_plan" in col or "vmail_plan" in col:
            df_feat[col] = (
                df_feat[col]
                .astype(str)
                .str.lower()
                .isin(["yes", "true", "1", "t"])
                .astype(float)
            )

    def get_col(patterns):
        for p in patterns:
            for c in cols:
                if p in c:
                    return c
        return None

    c_acc_len = get_col(["account_length", "acc_len"])
    c_vmail_msg = get_col(["number_vmail_messages", "vmail_messages", "vmail_msg"])
    c_day_min = get_col(["total_day_minutes", "day_minutes", "day_min"])
    c_day_calls = get_col(["total_day_calls", "day_calls"])
    c_day_charge = get_col(["total_day_charge", "day_charge"])

    c_eve_min = get_col(["total_eve_minutes", "eve_minutes", "eve_min"])
    c_eve_calls = get_col(["total_eve_calls", "eve_calls"])
    c_eve_charge = get_col(["total_eve_charge", "eve_charge"])

    c_night_min = get_col(["total_night_minutes", "night_minutes", "night_min"])
    c_night_calls = get_col(["total_night_calls", "night_calls"])
    c_night_charge = get_col(["total_night_charge", "night_charge"])

    c_intl_min = get_col(["total_intl_minutes", "intl_minutes", "intl_min"])
    c_intl_calls = get_col(["total_intl_calls", "intl_calls"])
    c_intl_charge = get_col(["total_intl_charge", "intl_charge"])

    c_cs_calls = get_col(
        ["customer_service_calls", "number_customer_service_calls", "cs_calls"]
    )
    c_intl_plan = get_col(["international_plan", "intl_plan"])
    c_vmail_plan = get_col(["voice_mail_plan", "vmail_plan"])

    num_cols_to_convert = [
        c_acc_len,
        c_vmail_msg,
        c_day_min,
        c_day_calls,
        c_day_charge,
        c_eve_min,
        c_eve_calls,
        c_eve_charge,
        c_night_min,
        c_night_calls,
        c_night_charge,
        c_intl_min,
        c_intl_calls,
        c_intl_charge,
        c_cs_calls,
    ]
    for col in num_cols_to_convert:
        if col is not None and col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce").astype(
                np.float32
            )

    # Aggregations & Usage Metrics
    df_feat["total_charge"] = (
        df_feat[c_day_charge]
        + df_feat[c_eve_charge]
        + df_feat[c_night_charge]
        + df_feat[c_intl_charge]
    )
    df_feat["total_minutes"] = (
        df_feat[c_day_min]
        + df_feat[c_eve_min]
        + df_feat[c_night_min]
        + df_feat[c_intl_min]
    )
    df_feat["total_calls"] = (
        df_feat[c_day_calls]
        + df_feat[c_eve_calls]
        + df_feat[c_night_calls]
        + df_feat[c_intl_calls]
    )

    df_feat["day_cost_per_min"] = df_feat[c_day_charge] / (df_feat[c_day_min] + 1e-5)
    df_feat["eve_cost_per_min"] = df_feat[c_eve_charge] / (df_feat[c_eve_min] + 1e-5)
    df_feat["night_cost_per_min"] = df_feat[c_night_charge] / (
        df_feat[c_night_min] + 1e-5
    )
    df_feat["intl_cost_per_min"] = df_feat[c_intl_charge] / (df_feat[c_intl_min] + 1e-5)
    df_feat["overall_cost_per_min"] = df_feat["total_charge"] / (
        df_feat["total_minutes"] + 1e-5
    )

    df_feat["day_min_per_call"] = df_feat[c_day_min] / (df_feat[c_day_calls] + 1e-5)
    df_feat["eve_min_per_call"] = df_feat[c_eve_min] / (df_feat[c_eve_calls] + 1e-5)
    df_feat["night_min_per_call"] = df_feat[c_night_min] / (
        df_feat[c_night_calls] + 1e-5
    )
    df_feat["intl_min_per_call"] = df_feat[c_intl_min] / (df_feat[c_intl_calls] + 1e-5)
    df_feat["overall_min_per_call"] = df_feat["total_minutes"] / (
        df_feat["total_calls"] + 1e-5
    )
    df_feat["overall_charge_per_call"] = df_feat["total_charge"] / (
        df_feat["total_calls"] + 1e-5
    )

    df_feat["day_min_ratio"] = df_feat[c_day_min] / (df_feat["total_minutes"] + 1e-5)
    df_feat["eve_min_ratio"] = df_feat[c_eve_min] / (df_feat["total_minutes"] + 1e-5)
    df_feat["night_min_ratio"] = df_feat[c_night_min] / (
        df_feat["total_minutes"] + 1e-5
    )
    df_feat["intl_min_ratio"] = df_feat[c_intl_min] / (df_feat["total_minutes"] + 1e-5)

    df_feat["day_charge_ratio"] = df_feat[c_day_charge] / (
        df_feat["total_charge"] + 1e-5
    )
    df_feat["eve_charge_ratio"] = df_feat[c_eve_charge] / (
        df_feat["total_charge"] + 1e-5
    )
    df_feat["night_charge_ratio"] = df_feat[c_night_charge] / (
        df_feat["total_charge"] + 1e-5
    )
    df_feat["intl_charge_ratio"] = df_feat[c_intl_charge] / (
        df_feat["total_charge"] + 1e-5
    )

    df_feat["day_eve_min_ratio"] = df_feat[c_day_min] / (df_feat[c_eve_min] + 1e-5)
    df_feat["day_night_min_ratio"] = df_feat[c_day_min] / (df_feat[c_night_min] + 1e-5)

    # CS Friction & Plan Interactions
    df_feat["cs_calls_per_acc_len"] = df_feat[c_cs_calls] / (df_feat[c_acc_len] + 1.0)
    df_feat["high_cs_calls_flag"] = (df_feat[c_cs_calls] >= 4).astype(float)
    df_feat["cs_calls_x_total_charge"] = df_feat[c_cs_calls] * df_feat["total_charge"]
    df_feat["charge_per_cs_call"] = df_feat["total_charge"] / (
        df_feat[c_cs_calls] + 1.0
    )

    if c_vmail_msg and c_acc_len:
        df_feat["vmail_msg_per_acc_len"] = df_feat[c_vmail_msg] / (
            df_feat[c_acc_len] + 1.0
        )
    if c_vmail_plan and c_vmail_msg:
        df_feat["vmail_plan_zero_msg"] = (
            (df_feat[c_vmail_plan] == 1) & (df_feat[c_vmail_msg] == 0)
        ).astype(float)
    if c_intl_plan and c_intl_calls:
        df_feat["intl_plan_zero_calls"] = (
            (df_feat[c_intl_plan] == 1) & (df_feat[c_intl_calls] == 0)
        ).astype(float)
    if c_intl_plan and c_intl_min:
        df_feat["high_intl_no_plan"] = (
            (df_feat[c_intl_plan] == 0) & (df_feat[c_intl_min] > 10.0)
        ).astype(float)

    # Frequency Encoding
    cat_cols = ["state", "area_code", "phone_number"]
    cat_cols = [c for c in cat_cols if c in df_feat.columns]

    if is_train:
        train_freqs = {}
        for c in cat_cols:
            df_feat[c] = df_feat[c].astype(str)
            freq = df_feat[c].value_counts(normalize=True).to_dict()
            train_freqs[c] = freq
            df_feat[f"{c}_freq"] = df_feat[c].map(freq).fillna(0.0)
    else:
        for c in cat_cols:
            df_feat[c] = df_feat[c].astype(str)
            freq = train_freqs.get(c, {})
            df_feat[f"{c}_freq"] = df_feat[c].map(freq).fillna(0.0)

    df_feat = df_feat.drop(columns=[c for c in cat_cols if c in df_feat.columns])

    # Missing Value Handling
    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    if is_train:
        train_medians = df_feat[num_cols].median().to_dict()
        df_feat[num_cols] = df_feat[num_cols].fillna(train_medians)
    else:
        for c in num_cols:
            if c in train_medians:
                df_feat[c] = df_feat[c].fillna(train_medians[c])
        df_feat = df_feat.fillna(0.0)

    return df_feat, train_medians, train_freqs


n_samples = len(raw_train_df)
n_test = len(raw_test_df)
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_cb = np.zeros(n_samples, dtype=np.float32)
oof_nn = np.zeros(n_samples, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)
test_cb = np.zeros(n_test, dtype=np.float32)
test_nn = np.zeros(n_test, dtype=np.float32)


# 3. Tabular ResNet Architecture
class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# 4. Stratified 5-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(raw_train_df, y)):
    df_tr_fold, medians, freqs = process_data(raw_train_df.iloc[train_idx], is_train=True)
    df_va_fold, _, _ = process_data(raw_train_df.iloc[val_idx], is_train=False, train_medians=medians, train_freqs=freqs)
    df_te_fold, _, _ = process_data(raw_test_df, is_train=False, train_medians=medians, train_freqs=freqs)

    feat_cols = list(df_tr_fold.columns)
    df_va_fold = df_va_fold[feat_cols]
    df_te_fold = df_te_fold[feat_cols]

    X_tr_raw = df_tr_fold.values.astype(np.float32)
    X_va_raw = df_va_fold.values.astype(np.float32)
    X_te_raw = df_te_fold.values.astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)
    X_test = scaler.transform(X_te_raw)
    y_tr, y_va = y[train_idx], y[val_idx]

    # --- LightGBM ---
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": 42 + fold,
        "verbose": -1,
    }
    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # --- XGBoost ---
    model_xgb = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    # --- CatBoost / ExtraTrees Fallback ---
    try:
        from catboost import CatBoostClassifier

        model_cb = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.025,
            depth=5,
            eval_metric="AUC",
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_idx] = model_cb.predict_proba(X_va)[:, 1]
        test_cb += model_cb.predict_proba(X_test)[:, 1] / n_splits
    except Exception:
        from sklearn.ensemble import ExtraTreesClassifier

        model_et = ExtraTreesClassifier(
            n_estimators=400, max_depth=12, random_state=42 + fold, n_jobs=-1
        )
        model_et.fit(X_tr, y_tr)
        oof_cb[val_idx] = model_et.predict_proba(X_va)[:, 1]
        test_cb += model_et.predict_proba(X_test)[:, 1] / n_splits

    # --- PyTorch Tabular ResNet ---
    ds_tr = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    ds_va = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
    )

    loader_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, drop_last=False)
    loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)

    model_nn = TabularResNet(
        input_dim=X_tr.shape[1], hidden_dim=128, num_blocks=2, dropout=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )

    best_val_auc_nn = 0.0
    best_nn_va_preds = None
    best_nn_test_preds = None

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    for epoch in range(30):
        model_nn.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model_nn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model_nn.eval()
        va_preds_epoch = []
        with torch.no_grad():
            for bx, _ in loader_va:
                bx = bx.to(device)
                logits = model_nn(bx)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                va_preds_epoch.extend(probs)

            test_logits = model_nn(X_test_tensor)
            test_probs_epoch = torch.sigmoid(test_logits).cpu().numpy().flatten()

        va_preds_epoch = np.array(va_preds_epoch)
        auc_epoch = roc_auc_score(y_va, va_preds_epoch)

        if auc_epoch > best_val_auc_nn:
            best_val_auc_nn = auc_epoch
            best_nn_va_preds = va_preds_epoch
            best_nn_test_preds = test_probs_epoch

    oof_nn[val_idx] = best_nn_va_preds
    test_nn += best_nn_test_preds / n_splits


# 5. Optimization & Ensemble Blending
oof_list = [oof_lgb, oof_xgb, oof_cb, oof_nn]
test_list = [test_lgb, test_xgb, test_cb, test_nn]


def oof_objective(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    else:
        w = np.ones_like(w) / len(w)

    blend = np.zeros(n_samples, dtype=np.float32)
    for weight, pred in zip(w, oof_list):
        blend += weight * pred
    return -roc_auc_score(y, blend)


init_weights = [0.25, 0.25, 0.25, 0.25]
res = minimize(
    oof_objective, init_weights, method="Nelder-Mead", options={"maxiter": 500}
)

optimal_weights = np.maximum(res.x, 0)
optimal_weights = optimal_weights / np.sum(optimal_weights)

final_oof_preds = np.zeros(n_samples, dtype=np.float32)
final_test_preds = np.zeros(n_test, dtype=np.float32)

for w, oof_p, test_p in zip(optimal_weights, oof_list, test_list):
    final_oof_preds += w * oof_p
    final_test_preds += w * test_p

final_validation_auc = roc_auc_score(y, final_oof_preds)

# 6. Submission File & Metric Output
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "class": final_test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_validation_auc}")