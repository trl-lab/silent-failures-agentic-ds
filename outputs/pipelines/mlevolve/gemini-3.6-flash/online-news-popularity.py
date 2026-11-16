import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Set random seeds for deterministic reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Raw Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


# 2. Feature Engineering Function
def process_and_engineer_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    drop_cols = ["url", "shares", "id"]

    # Outlier / Anomaly flags & Capping
    if "n_unique_tokens" in df.columns:
        df["is_anomalous_unique_tokens"] = (df["n_unique_tokens"] > 1.0).astype(np.float32)
        df["n_unique_tokens_clipped"] = df["n_unique_tokens"].clip(0, 1)
    if "n_non_stop_unique_tokens" in df.columns:
        df["n_non_stop_unique_tokens_clipped"] = df["n_non_stop_unique_tokens"].clip(0, 1)
    if "n_non_stop_words" in df.columns:
        df["n_non_stop_words_clipped"] = df["n_non_stop_words"].clip(0, 1)

    # Text Ratios
    if "n_tokens_title" in df.columns and "n_tokens_content" in df.columns:
        df["title_to_content_ratio"] = df["n_tokens_title"] / (df["n_tokens_content"] + 1.0)
    if "n_tokens_content" in df.columns and "num_keywords" in df.columns:
        df["words_per_keyword"] = df["n_tokens_content"] / (df["num_keywords"] + 1.0)
    if "n_non_stop_words" in df.columns and "n_tokens_content" in df.columns:
        df["non_stop_ratio"] = df["n_non_stop_words"] * df["n_tokens_content"]

    # Media & Visual Features
    if "num_imgs" in df.columns and "num_videos" in df.columns:
        df["total_media"] = df["num_imgs"] + df["num_videos"]
        df["img_to_video_ratio"] = df["num_imgs"] / (df["num_videos"] + 1.0)
    if "num_imgs" in df.columns:
        if "n_tokens_content" in df.columns:
            df["img_density"] = df["num_imgs"] / (df["n_tokens_content"] + 1.0)
        df["has_imgs"] = (df["num_imgs"] > 0).astype(np.float32)
    if "num_videos" in df.columns:
        if "n_tokens_content" in df.columns:
            df["video_density"] = df["num_videos"] / (df["n_tokens_content"] + 1.0)
        df["has_videos"] = (df["num_videos"] > 0).astype(np.float32)
    if "total_media" in df.columns and "n_tokens_content" in df.columns:
        df["media_density"] = df["total_media"] / (df["n_tokens_content"] + 1.0)

    # Hyperlinks & Self-Reference Features
    if "num_self_hrefs" in df.columns and "num_hrefs" in df.columns:
        df["self_link_ratio"] = df["num_self_hrefs"] / (df["num_hrefs"] + 1.0)
        df["external_hrefs"] = np.maximum(0, df["num_hrefs"] - df["num_self_hrefs"])
    if "num_hrefs" in df.columns and "n_tokens_content" in df.columns:
        df["href_density"] = df["num_hrefs"] / (df["n_tokens_content"] + 1.0)

    # Channel Categorical Encoding
    channel_cols = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
    ]
    present_channels = [c for c in channel_cols if c in df.columns]
    if present_channels:
        channel_matrix = df[present_channels].values
        df["channel_code"] = np.where(
            channel_matrix.sum(axis=1) > 0, np.argmax(channel_matrix, axis=1), len(present_channels)
        ).astype(np.int32)
        df["is_channel_missing"] = (channel_matrix.sum(axis=1) == 0).astype(np.float32)

    # Day of Week Encoding & Timedelta
    weekday_cols = [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
    ]
    present_weekdays = [c for c in weekday_cols if c in df.columns]
    if present_weekdays:
        weekday_matrix = df[present_weekdays].values
        df["day_of_week"] = np.argmax(weekday_matrix, axis=1).astype(np.int32)

    if "timedelta" in df.columns:
        df["timedelta_weeks"] = df["timedelta"] / 7.0
        df["timedelta_day_in_week"] = df["timedelta"] % 7.0

    # Keyword Statistics & Dynamics
    if "kw_min_min" in df.columns:
        df["has_kw_min_min"] = (df["kw_min_min"] >= 0).astype(np.float32)
    for col in ["kw_avg_avg", "kw_max_avg", "kw_min_avg", "kw_avg_max"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(np.maximum(0, df[col]))

    if "kw_max_avg" in df.columns and "kw_min_avg" in df.columns:
        df["kw_avg_spread"] = df["kw_max_avg"] - df["kw_min_avg"]
    if "kw_max_max" in df.columns and "kw_min_max" in df.columns:
        df["kw_max_spread"] = df["kw_max_max"] - df["kw_min_max"]
    if "kw_max_min" in df.columns and "kw_min_min" in df.columns:
        df["kw_min_spread"] = df["kw_max_min"] - df["kw_min_min"]
    if "kw_avg_avg" in df.columns and "kw_max_avg" in df.columns:
        df["kw_avg_ratio"] = df["kw_avg_avg"] / (df["kw_max_avg"] + 1.0)

    kw_avg_cols = [c for c in ["kw_avg_min", "kw_avg_max", "kw_avg_avg"] if c in df.columns]
    if kw_avg_cols:
        df["kw_avg_stats_mean"] = df[kw_avg_cols].mean(axis=1)
        df["kw_avg_stats_std"] = df[kw_avg_cols].std(axis=1)
        df["kw_avg_stats_max"] = df[kw_avg_cols].max(axis=1)
        df["kw_avg_stats_min"] = df[kw_avg_cols].min(axis=1)

    # Self-Reference Shares Features
    if "self_reference_min_shares" in df.columns:
        df["has_self_ref"] = (df["self_reference_min_shares"] >= 0).astype(np.float32)
        df["self_ref_min_log"] = np.log1p(np.maximum(0, df["self_reference_min_shares"]))
    if "self_reference_avg_sharess" in df.columns:
        df["self_ref_avg_log"] = np.log1p(np.maximum(0, df["self_reference_avg_sharess"]))
    if "self_reference_max_shares" in df.columns:
        df["self_ref_max_log"] = np.log1p(np.maximum(0, df["self_reference_max_shares"]))

    if "self_reference_max_shares" in df.columns and "self_reference_min_shares" in df.columns:
        df["self_ref_spread"] = (
            df["self_reference_max_shares"] - df["self_reference_min_shares"]
        )
        df["self_ref_ratio"] = df["self_reference_min_shares"] / (
            df["self_reference_max_shares"] + 1.0
        )

    # Topic (LDA) Features
    lda_cols = [c for c in ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"] if c in df.columns]
    if len(lda_cols) > 0:
        lda_vals = df[lda_cols].values
        df["primary_topic"] = np.argmax(lda_vals, axis=1).astype(np.int32)
        df["primary_topic_prob"] = np.max(lda_vals, axis=1)
        eps = 1e-8
        df["topic_entropy"] = -np.sum(lda_vals * np.log(lda_vals + eps), axis=1)

    # Sentiment & Subjectivity Interactions
    if "global_rate_positive_words" in df.columns and "global_rate_negative_words" in df.columns:
        df["net_sentiment_rate"] = (
            df["global_rate_positive_words"] - df["global_rate_negative_words"]
        )
        df["total_sentiment_rate"] = (
            df["global_rate_positive_words"] + df["global_rate_negative_words"]
        )
    if "rate_positive_words" in df.columns and "rate_negative_words" in df.columns:
        df["pos_neg_word_ratio"] = df["rate_positive_words"] / (
            df["rate_negative_words"] + 1e-5
        )

    if "max_positive_polarity" in df.columns and "min_positive_polarity" in df.columns:
        df["polarity_spread_pos"] = (
            df["max_positive_polarity"] - df["min_positive_polarity"]
        )
    if "max_negative_polarity" in df.columns and "min_negative_polarity" in df.columns:
        df["polarity_spread_neg"] = (
            df["max_negative_polarity"] - df["min_negative_polarity"]
        )

    if "global_subjectivity" in df.columns and "global_sentiment_polarity" in df.columns:
        df["subjectivity_polarity_interaction"] = df["global_subjectivity"] * np.abs(
            df["global_sentiment_polarity"]
        )
    if "title_subjectivity" in df.columns and "title_sentiment_polarity" in df.columns:
        df["title_subjectivity_polarity_interaction"] = df["title_subjectivity"] * np.abs(
            df["title_sentiment_polarity"]
        )

    if "title_sentiment_polarity" in df.columns and "global_sentiment_polarity" in df.columns:
        df["title_vs_content_sentiment_diff"] = (
            df["title_sentiment_polarity"] - df["global_sentiment_polarity"]
        )
    if "title_subjectivity" in df.columns and "global_subjectivity" in df.columns:
        df["title_vs_content_subj_diff"] = (
            df["title_subjectivity"] - df["global_subjectivity"]
        )

    # Clean feature columns
    feature_cols = [c for c in df.columns if c not in drop_cols]
    res_df = df[feature_cols].copy()
    res_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    res_df.fillna(0, inplace=True)

    return res_df


# Process Train and Test Features
X_train_df = process_and_engineer_features(train_df)
X_test_df = process_and_engineer_features(test_df)
y_train = train_df["shares"].values

log_y = np.log1p(y_train)
y_bins = pd.qcut(log_y, q=10, labels=False, duplicates="drop")

X_train_np = X_train_df.values.astype(np.float32)
X_test_np = X_test_df.values.astype(np.float32)


# 3. PyTorch Tabular Residual Neural Network Architecture
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.dropout(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 4. Stratified 5-Fold Setup and Out-of-Fold Arrays
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(y_train), dtype=np.float64)
oof_xgb = np.zeros(len(y_train), dtype=np.float64)
oof_cb = np.zeros(len(y_train), dtype=np.float64) if HAS_CATBOOST else None
oof_nn = np.zeros(len(y_train), dtype=np.float64)

test_lgb = np.zeros(len(X_test_df), dtype=np.float64)
test_xgb = np.zeros(len(X_test_df), dtype=np.float64)
test_cb = np.zeros(len(X_test_df), dtype=np.float64) if HAS_CATBOOST else None
test_nn = np.zeros(len(X_test_df), dtype=np.float64)

# Model Hyperparameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 25,
    "random_state": SEED,
    "n_estimators": 1200,
    "verbosity": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "random_state": SEED,
    "n_estimators": 1200,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

# 5. Model Training across Stratified Folds
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_bins)):
    X_tr_df, y_tr_log = X_train_df.iloc[train_idx], log_y[train_idx]
    X_va_df, y_va_log = X_train_df.iloc[val_idx], log_y[val_idx]

    # --- LightGBM ---
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_df,
        y_tr_log,
        eval_set=[(X_va_df, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = np.expm1(np.clip(model_lgb.predict(X_va_df), 0, 15))
    test_lgb += np.expm1(np.clip(model_lgb.predict(X_test_df), 0, 15)) / skf.n_splits

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr_df,
        y_tr_log,
        eval_set=[(X_va_df, y_va_log)],
        verbose=False,
    )
    oof_xgb[val_idx] = np.expm1(np.clip(model_xgb.predict(X_va_df), 0, 15))
    test_xgb += np.expm1(np.clip(model_xgb.predict(X_test_df), 0, 15)) / skf.n_splits

    # --- CatBoost ---
    if HAS_CATBOOST:
        model_cb = cb.CatBoostRegressor(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=SEED,
            early_stopping_rounds=50,
            verbose=False,
        )
        model_cb.fit(
            X_tr_df,
            y_tr_log,
            eval_set=(X_va_df, y_va_log),
            verbose=False,
        )
        oof_cb[val_idx] = np.expm1(np.clip(model_cb.predict(X_va_df), 0, 15))
        test_cb += np.expm1(np.clip(model_cb.predict(X_test_df), 0, 15)) / skf.n_splits

    # --- PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_np[train_idx])
    X_va_scaled = scaler.transform(X_train_np[val_idx])
    X_te_scaled = scaler.transform(X_test_np)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va_log, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    nn_model = TabularResNet(
        input_dim=X_train_np.shape[1], hidden_dim=128, num_blocks=2, dropout_rate=0.2
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(35):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(by)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        va_pred_log = np.clip(nn_model(va_tensor).cpu().numpy(), 0, 15)
        te_pred_log = np.clip(nn_model(te_tensor).cpu().numpy(), 0, 15)

    oof_nn[val_idx] = np.expm1(va_pred_log)
    test_nn += np.expm1(te_pred_log) / skf.n_splits

    print(f"Fold {fold + 1}/5 completed.")

# Lower and upper target sanity clipping
max_train_val = float(np.max(y_train)) * 2.0
oof_lgb = np.clip(oof_lgb, 1.0, max_train_val)
oof_xgb = np.clip(oof_xgb, 1.0, max_train_val)
oof_nn = np.clip(oof_nn, 1.0, max_train_val)

test_lgb = np.clip(test_lgb, 1.0, max_train_val)
test_xgb = np.clip(test_xgb, 1.0, max_train_val)
test_nn = np.clip(test_nn, 1.0, max_train_val)

oof_list = [oof_lgb, oof_xgb, oof_nn]
test_list = [test_lgb, test_xgb, test_nn]

if HAS_CATBOOST:
    oof_cb = np.clip(oof_cb, 1.0, max_train_val)
    test_cb = np.clip(test_cb, 1.0, max_train_val)
    oof_list.append(oof_cb)
    test_list.append(test_cb)


# 6. Constrained Blend Optimization on OOF RMSE
def blend_objective(weights):
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return 1e9
    weights = weights / w_sum
    pred = sum(w * o for w, o in zip(weights, oof_list))
    pred = np.clip(pred, 1.0, max_train_val)
    return root_mean_squared_error(y_train, pred)


init_w = np.ones(len(oof_list)) / len(oof_list)
bounds = [(0, 1)] * len(oof_list)
res = minimize(blend_objective, init_w, method="SLSQP", bounds=bounds)
opt_w = res.x / np.sum(res.x)

raw_oof_blend = sum(w * o for w, o in zip(opt_w, oof_list))
raw_test_blend = sum(w * t for w, t in zip(opt_w, test_list))


# 7. Post-Processing Affine Calibration
def post_proc_objective(params):
    scale, shift = params
    pred = np.clip(scale * raw_oof_blend + shift, 1.0, max_train_val)
    return root_mean_squared_error(y_train, pred)


calib_res = minimize(
    post_proc_objective,
    x0=[1.0, 0.0],
    method="Nelder-Mead",
    options={"maxiter": 500},
)
opt_scale, opt_shift = calib_res.x

final_oof_preds = np.clip(opt_scale * raw_oof_blend + opt_shift, 1.0, max_train_val)
final_test_preds = np.clip(opt_scale * raw_test_blend + opt_shift, 1.0, max_train_val)

final_rmse = root_mean_squared_error(y_train, final_oof_preds)

# 8. Save Output Submission File
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "shares": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

# Final Output Requirement
print(f"Final Validation Score: {final_rmse}")
