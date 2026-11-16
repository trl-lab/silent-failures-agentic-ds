import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# 2. IDENTIFY AND SEPARATE COLUMNS
# ============================================================
target_col = "shares"
id_col = "url"

train_df = train_df.drop(columns=[id_col], errors="ignore")
test_df = test_df.drop(columns=[id_col], errors="ignore")

# Drop timedelta - marked as "Other" role, potential leakage
drop_cols = ["timedelta"]
train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Separate target
y_train = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Apply feature engineering to dataframe. All transformations are deterministic."""
    df = df.copy()

    # --- LDA Topic Features ---
    lda_cols = [c for c in df.columns if c.startswith("LDA_")]
    if lda_cols:
        df["lda_max_topic"] = df[lda_cols].max(axis=1)
        df["lda_min_topic"] = df[lda_cols].min(axis=1)
        df["lda_range"] = df["lda_max_topic"] - df["lda_min_topic"]
        df["lda_entropy"] = -(df[lda_cols] * np.log1p(df[lda_cols])).sum(axis=1)
        df["lda_top1_share"] = df["lda_max_topic"] / (df[lda_cols].sum(axis=1) + 1e-8)
        df["lda_top2_sum"] = df[lda_cols].apply(
            lambda x: np.sort(x.values)[-2:].sum(), axis=1
        )
        df["lda_concentration"] = df["lda_top2_sum"] / (df[lda_cols].sum(axis=1) + 1e-8)

    # --- Keyword Features ---
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    if kw_cols:
        kw_min_cols = [c for c in kw_cols if "_min" in c]
        kw_max_cols = [c for c in kw_cols if "_max" in c]
        kw_avg_cols = [c for c in kw_cols if "_avg" in c]

        if kw_min_cols and "kw_min_min" in df.columns:
            df["kw_min_range"] = (
                df[[c for c in kw_min_cols if c != "kw_min_min"]].max(axis=1)
                - df["kw_min_min"]
            )
        if kw_max_cols and "kw_max_max" in df.columns:
            df["kw_max_range"] = (
                df[[c for c in kw_max_cols if c != "kw_max_max"]].max(axis=1)
                - df["kw_max_max"]
            )
        if "kw_avg_max" in df.columns and "kw_avg_min" in df.columns:
            df["kw_avg_spread"] = df["kw_avg_max"] - df["kw_avg_min"]
        if "kw_avg_avg" in df.columns and "kw_avg_min" in df.columns:
            df["kw_importance_ratio"] = df["kw_avg_avg"] / (df["kw_avg_min"] + 1e-8)
        df["kw_total_weight"] = df[kw_cols].sum(axis=1)

    # --- Sentiment Features ---
    if "global_sentiment_polarity" in df.columns:
        df["sentiment_abs"] = df["global_sentiment_polarity"].abs()
        df["sentiment_sign"] = np.sign(df["global_sentiment_polarity"])
        df["sentiment_x_subjectivity"] = (
            df["global_sentiment_polarity"] * df["global_subjectivity"]
        )

    if "avg_positive_polarity" in df.columns and "avg_negative_polarity" in df.columns:
        df["polarity_balance"] = (
            df["avg_positive_polarity"] - df["avg_negative_polarity"]
        )
        df["polarity_ratio"] = df["avg_positive_polarity"] / (
            df["avg_negative_polarity"] + 1e-8
        )

    if "rate_positive_words" in df.columns and "rate_negative_words" in df.columns:
        df["sentiment_word_ratio"] = df["rate_positive_words"] / (
            df["rate_negative_words"] + 1e-8
        )
        df["sentiment_word_diff"] = (
            df["rate_positive_words"] - df["rate_negative_words"]
        )
        df["sentiment_word_total"] = (
            df["rate_positive_words"] + df["rate_negative_words"]
        )

    # --- Title Sentiment Features ---
    if "title_sentiment_polarity" in df.columns:
        df["title_sentiment_abs"] = df["title_sentiment_polarity"].abs()
        if "global_sentiment_polarity" in df.columns:
            df["title_vs_global_sentiment"] = (
                df["title_sentiment_polarity"] - df["global_sentiment_polarity"]
            )

    # --- Content Structure Features ---
    if "n_tokens_title" in df.columns and "n_tokens_content" in df.columns:
        df["title_content_ratio"] = df["n_tokens_title"] / (
            df["n_tokens_content"] + 1e-8
        )
        df["content_length"] = df["n_tokens_content"]

    if "n_unique_tokens" in df.columns and "n_tokens_content" in df.columns:
        df["lexical_diversity"] = df["n_unique_tokens"] / (
            df["n_tokens_content"] + 1e-8
        )

    if "n_non_stop_words" in df.columns and "n_tokens_content" in df.columns:
        df["content_word_ratio"] = df["n_non_stop_words"] / (
            df["n_tokens_content"] + 1e-8
        )

    if "n_non_stop_unique_tokens" in df.columns and "n_non_stop_words" in df.columns:
        df["nonstop_unique_ratio"] = df["n_non_stop_unique_tokens"] / (
            df["n_non_stop_words"] + 1e-8
        )

    # --- Media Features ---
    if "num_hrefs" in df.columns:
        df["href_density"] = df["num_hrefs"] / (df["n_tokens_content"] + 1e-8)
        if "num_self_hrefs" in df.columns:
            df["self_href_ratio"] = df["num_self_hrefs"] / (df["num_hrefs"] + 1e-8)

    if "num_imgs" in df.columns and "num_videos" in df.columns:
        df["total_media"] = df["num_imgs"] + df["num_videos"]
        df["media_ratio"] = df["total_media"] / (df["n_tokens_content"] + 1e-8)

    # --- Channel Features ---
    channel_cols = [c for c in df.columns if c.startswith("data_channel_is_")]
    if channel_cols:
        df["num_channels"] = df[channel_cols].sum(axis=1)
        df["is_multi_channel"] = (df["num_channels"] > 1).astype(int)
        if "data_channel_is_tech" in df.columns and "num_keywords" in df.columns:
            df["tech_x_keywords"] = df["data_channel_is_tech"] * df["num_keywords"]
        if "data_channel_is_entertainment" in df.columns and "num_imgs" in df.columns:
            df["ent_x_imgs"] = df["data_channel_is_entertainment"] * df["num_imgs"]
        if (
            "data_channel_is_world" in df.columns
            and "global_sentiment_polarity" in df.columns
        ):
            df["world_x_sentiment"] = (
                df["data_channel_is_world"] * df["global_sentiment_polarity"]
            )
        if "data_channel_is_lifestyle" in df.columns and "num_videos" in df.columns:
            df["lifestyle_x_videos"] = (
                df["data_channel_is_lifestyle"] * df["num_videos"]
            )

    # --- Weekday Features ---
    weekday_cols = [c for c in df.columns if c.startswith("weekday_is_")]
    if weekday_cols:
        df["weekday_num"] = df[weekday_cols].sum(axis=1)
        if "is_weekend" in df.columns and "num_imgs" in df.columns:
            df["weekend_x_imgs"] = df["is_weekend"] * df["num_imgs"]
        if "is_weekend" in df.columns and "num_videos" in df.columns:
            df["weekend_x_videos"] = df["is_weekend"] * df["num_videos"]
        if "is_weekend" in df.columns and "data_channel_is_entertainment" in df.columns:
            df["weekend_x_ent"] = df["is_weekend"] * df["data_channel_is_entertainment"]

    # --- Self-Reference Features ---
    self_ref_cols = [c for c in df.columns if c.startswith("self_reference_")]
    if self_ref_cols:
        df["self_ref_range"] = (
            df["self_reference_max_shares"] - df["self_reference_min_shares"]
        )
        df["self_ref_spread_ratio"] = df["self_ref_range"] / (
            df["self_reference_max_shares"] + 1e-8
        )
        df["self_ref_log_max"] = np.log1p(df["self_reference_max_shares"])
        df["self_ref_log_avg"] = np.log1p(df["self_reference_avg_sharess"])
        if "n_tokens_content" in df.columns:
            df["self_ref_per_token"] = df["self_reference_max_shares"] / (
                df["n_tokens_content"] + 1e-8
            )

    # --- LDA × Sentiment Interactions ---
    if lda_cols and "global_sentiment_polarity" in df.columns:
        for i in range(5):
            col = f"LDA_{i:02d}"
            if col in df.columns:
                df[f"lda{i}_x_sentiment"] = df[col] * df["global_sentiment_polarity"]

    # --- LDA × Channel Interactions ---
    if lda_cols and "data_channel_is_tech" in df.columns:
        df["lda0_x_tech"] = df["LDA_00"] * df["data_channel_is_tech"]
        df["lda1_x_tech"] = df["LDA_01"] * df["data_channel_is_tech"]

    # --- Token Length Features ---
    if "average_token_length" in df.columns:
        df["token_length_x_content"] = df["average_token_length"] * np.log1p(
            df["n_tokens_content"]
        )

    # --- Keyword × Content Interactions ---
    if "num_keywords" in df.columns:
        df["keywords_per_token"] = df["num_keywords"] / (df["n_tokens_content"] + 1e-8)
        if "LDA_00" in df.columns:
            df["keywords_x_lda0"] = df["num_keywords"] * df["LDA_00"]

    # --- Polarity Range Features ---
    if "min_positive_polarity" in df.columns and "max_positive_polarity" in df.columns:
        df["pos_polarity_range"] = (
            df["max_positive_polarity"] - df["min_positive_polarity"]
        )
    if "min_negative_polarity" in df.columns and "max_negative_polarity" in df.columns:
        df["neg_polarity_range"] = (
            df["max_negative_polarity"] - df["min_negative_polarity"]
        )

    # --- Title Features ---
    if "title_subjectivity" in df.columns and "title_sentiment_polarity" in df.columns:
        df["title_subj_x_sentiment"] = (
            df["title_subjectivity"] * df["title_sentiment_polarity"]
        )

    return df


# Apply feature engineering
print("Engineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# ============================================================
# 4. HANDLE MISSING VALUES AND OUTLIERS
# ============================================================
train_medians = train_df.median(numeric_only=True)
train_df = train_df.fillna(train_medians)
test_df = test_df.fillna(train_medians)

Q1 = train_df.quantile(0.01, numeric_only=True)
Q3 = train_df.quantile(0.99, numeric_only=True)
IQR = Q3 - Q1
lower_bound = Q1 - 10 * IQR
upper_bound = Q3 + 10 * IQR

train_df = train_df.clip(lower=lower_bound, upper=upper_bound, axis=1)
test_df = test_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# ============================================================
# 5. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
y_log = np.log1p(y_train)
strat_bins = pd.qcut(y_log, q=10, duplicates="drop")

train_sub_df, val_df, y_train_sub, y_val, _, _ = train_test_split(
    train_df, y_train, strat_bins, test_size=0.15, random_state=42, stratify=strat_bins
)

print(f"Training set: {train_sub_df.shape[0]} samples")
print(f"Validation set: {val_df.shape[0]} samples")
print(f"Test set: {test_df.shape[0]} samples")

# ============================================================
# 6. PREPARE TENSORS
# ============================================================
feature_columns = train_sub_df.columns.tolist()
print(f"Number of features: {len(feature_columns)}")

X_train = train_sub_df[feature_columns].values.astype(np.float32)
X_val = val_df[feature_columns].values.astype(np.float32)
X_test = test_df[feature_columns].values.astype(np.float32)

# Log-transform target for training
y_train_log = np.log1p(y_train_sub).astype(np.float32)
y_val_log = np.log1p(y_val).astype(np.float32)

# Standardize features (fit on train only)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Create tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_log_t = torch.tensor(y_val_log, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)


# ============================================================
# 7. DEFINE MODEL ARCHITECTURE
# ============================================================
class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.gelu(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return residual + out


class NewsPopularityNet(nn.Module):
    """MLP with residual connections for predicting news article shares."""

    def __init__(self, input_dim, hidden_dim=128, n_blocks=4, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_head(x)
        return x.squeeze(-1)


# ============================================================
# 8. SETUP TRAINING
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = len(feature_columns)
model = NewsPopularityNet(
    input_dim=input_dim, hidden_dim=128, n_blocks=4, dropout=0.15
).to(device)

criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
max_epochs = 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epochs, eta_min=1e-5
)

# ============================================================
# 9. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
batch_size = 512
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

best_val_rmse = float("inf")
best_model_state = None
patience = 20
patience_counter = 0

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    # Validation: compute RMSE on original scale (official metric)
    model.eval()
    with torch.no_grad():
        val_pred_log = model(X_val_t.to(device)).cpu().numpy()
        val_pred = np.expm1(val_pred_log)
        val_pred = np.clip(val_pred, 0, None)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    if (epoch + 1) % 10 == 0 or val_rmse < best_val_rmse:
        print(
            f"Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.5f} | Val RMSE: {val_rmse:.4f}"
        )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ============================================================
# 10. LOAD BEST MODEL AND GENERATE FINAL PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Validation predictions (same pipeline as test)
with torch.no_grad():
    val_pred_log = model(X_val_t.to(device)).cpu().numpy()
    val_pred = np.expm1(val_pred_log)
    val_pred = np.clip(val_pred, 0, None)

final_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Best Validation RMSE: {final_rmse:.4f}")

# Test predictions (identical processing to validation)
with torch.no_grad():
    test_pred_log = model(X_test_t.to(device)).cpu().numpy()
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0, None)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "shares": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# ============================================================
# 12. FINAL SCORE
# ============================================================
score = final_rmse
print(f"Final Validation Score: {score}")
