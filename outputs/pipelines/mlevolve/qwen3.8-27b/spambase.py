import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
print(f"Target distribution:\n{train_df['class'].value_counts(normalize=True)}")

# Separate target (convert categorical to plain int numpy array)
y_train_full = train_df["class"].astype(int).values
X_train_full = train_df.drop(columns=["class"])
X_test = test_df.copy()

# Fill any missing values with 0 (safe for frequency features)
X_train_full = X_train_full.fillna(0)
X_test = X_test.fillna(0)

# ============================================================
# 2. CREATE TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\nTrain split: {X_train.shape}, Val split: {X_val.shape}")
print(f"Train class balance: {y_train.mean():.4f}")
print(f"Val class balance: {y_val.mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

word_freq_cols = [c for c in X_train.columns if c.startswith("word_freq_")]
char_freq_cols = [c for c in X_train.columns if c.startswith("char_freq_")]
capital_cols = [c for c in X_train.columns if c.startswith("capital_run_length")]

print(
    f"\nFeature groups: word_freq={len(word_freq_cols)}, char_freq={len(char_freq_cols)}, capital={len(capital_cols)}"
)


def engineer_features(X):
    """Apply deterministic feature engineering (no learned parameters)."""
    X = X.copy()

    # --- Group-level aggregate features ---
    X["word_freq_mean"] = X[word_freq_cols].mean(axis=1)
    X["word_freq_max"] = X[word_freq_cols].max(axis=1)
    X["word_freq_std"] = X[word_freq_cols].std(axis=1)
    X["word_freq_min"] = X[word_freq_cols].min(axis=1)
    X["word_freq_median"] = X[word_freq_cols].median(axis=1)
    X["word_freq_skew"] = X[word_freq_cols].skew(axis=1)
    X["word_freq_sum"] = X[word_freq_cols].sum(axis=1)
    X["word_freq_range"] = X["word_freq_max"] - X["word_freq_min"]

    X["char_freq_mean"] = X[char_freq_cols].mean(axis=1)
    X["char_freq_max"] = X[char_freq_cols].max(axis=1)
    X["char_freq_std"] = X[char_freq_cols].std(axis=1)
    X["char_freq_min"] = X[char_freq_cols].min(axis=1)
    X["char_freq_median"] = X[char_freq_cols].median(axis=1)
    X["char_freq_skew"] = X[char_freq_cols].skew(axis=1)
    X["char_freq_sum"] = X[char_freq_cols].sum(axis=1)
    X["char_freq_range"] = X["char_freq_max"] - X["char_freq_min"]

    # --- Cross-group ratio features ---
    X["word_to_char_ratio"] = X["word_freq_mean"] / (X["char_freq_mean"] + 1e-8)
    X["word_std_to_char_std"] = X["word_freq_std"] / (X["char_freq_std"] + 1e-8)
    X["word_max_to_char_max"] = X["word_freq_max"] / (X["char_freq_max"] + 1e-8)

    # --- Capital run length derived features ---
    if len(capital_cols) >= 3:
        avg_col = [c for c in capital_cols if "average" in c][0]
        longest_col = [c for c in capital_cols if "longest" in c][0]
        total_col = [c for c in capital_cols if "total" in c][0]

        X["capital_avg"] = X[avg_col]
        X["capital_longest"] = X[longest_col]
        X["capital_total"] = X[total_col]

        X["capital_longest_to_avg"] = X[longest_col] / (X[avg_col] + 1e-8)
        X["capital_total_to_avg"] = X[total_col] / (X[avg_col] + 1e-8)
        X["capital_density"] = X[total_col] / (X[total_col] + 1e-8)
        X["capital_burst_ratio"] = X[longest_col] / (X[total_col] + 1e-8)

    # --- Top-K word frequency features ---
    word_vals = X[word_freq_cols].values.copy()
    np.sort(word_vals, axis=1)
    X["word_freq_top1"] = word_vals[:, -1]
    X["word_freq_top2"] = word_vals[:, -2]
    X["word_freq_top3"] = word_vals[:, -3]
    X["word_freq_top5"] = word_vals[:, -5]
    X["word_freq_top10"] = word_vals[:, -10]
    X["word_freq_top10_sum"] = word_vals[:, -10:].sum(axis=1)
    X["word_freq_top10_mean"] = word_vals[:, -10:].mean(axis=1)

    # --- Concentration features ---
    X["word_freq_concentration"] = X["word_freq_top1"] / (X["word_freq_sum"] + 1e-8)
    X["word_freq_gini_approx"] = X["word_freq_top10_sum"] / (X["word_freq_sum"] + 1e-8)

    # --- Char frequency concentration ---
    char_vals = X[char_freq_cols].values.copy()
    np.sort(char_vals, axis=1)
    X["char_freq_top1"] = char_vals[:, -1]
    X["char_freq_top3"] = char_vals[:, -3]
    X["char_freq_top5"] = char_vals[:, -5]
    X["char_freq_concentration"] = X["char_freq_top1"] / (X["char_freq_sum"] + 1e-8)

    # --- Interaction: capital intensity with word/char patterns ---
    if len(capital_cols) >= 1:
        total_col = [c for c in capital_cols if "total" in c][0]
        X["capital_per_word"] = X[total_col] / (X["word_freq_sum"] + 1e-8)
        X["capital_per_char"] = X[total_col] / (X["char_freq_sum"] + 1e-8)

    # --- Overall text complexity proxy ---
    X["text_complexity"] = X["word_freq_std"] * X["char_freq_std"]
    X["text_uniformity"] = 1.0 / (1.0 + X["word_freq_std"] + X["char_freq_std"])

    return X


# Apply feature engineering (deterministic, no learned parameters)
X_train_eng = engineer_features(X_train)
X_val_eng = engineer_features(X_val)
X_test_eng = engineer_features(X_test)

print(f"\nAfter feature engineering:")
print(f"  Train: {X_train_eng.shape}")
print(f"  Val: {X_val_eng.shape}")
print(f"  Test: {X_test_eng.shape}")

# ============================================================
# 4. DEFINE MODELS
# ============================================================

# --- Model A: Regularized LightGBM ---
lgb_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_split_gain=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# --- Model B: 1D CNN ---
class SpambaseCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)


input_dim = X_train_eng.shape[1]
cnn_model = SpambaseCNN(input_dim=input_dim, hidden_dim=128, dropout=0.3)
cnn_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=200)
cnn_criterion = nn.BCEWithLogitsLoss()

# ============================================================
# 5. TRAIN LIGHTGBM
# ============================================================
print("\n--- Training LightGBM ---")
lgb_model.fit(
    X_train_eng,
    y_train,
    eval_set=[(X_val_eng, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

val_pred_lgb = lgb_model.predict_proba(X_val_eng)[:, 1]
val_auc_lgb = roc_auc_score(y_val, val_pred_lgb)
print(
    f"LightGBM - Best iteration: {lgb_model.best_iteration_}, Val AUC: {val_auc_lgb:.6f}"
)

# ============================================================
# 6. TRAIN CNN
# ============================================================
print("\n--- Training CNN ---")

# Fit normalization on TRAIN ONLY (prevent leakage)
X_train_np = X_train_eng.values.astype(np.float32)
X_val_np = X_val_eng.values.astype(np.float32)
X_test_np = X_test_eng.values.astype(np.float32)

train_mean = X_train_np.mean(axis=0)
train_std = X_train_np.std(axis=0) + 1e-8

X_train_norm = torch.tensor((X_train_np - train_mean) / train_std, dtype=torch.float32)
X_val_norm = torch.tensor((X_val_np - train_mean) / train_std, dtype=torch.float32)
X_test_norm = torch.tensor((X_test_np - train_mean) / train_std, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32)

cnn_model.train()
best_val_auc_cnn = 0.0
patience_counter = 0
patience = 20
best_cnn_state = None

for epoch in range(200):
    perm = torch.randperm(X_train_norm.shape[0])
    X_shuffled = X_train_norm[perm]
    y_shuffled = y_train_t[perm]

    batch_size = 64
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]

        cnn_optimizer.zero_grad()
        logits = cnn_model(X_batch)
        loss = cnn_criterion(logits, y_batch)
        loss.backward()
        cnn_optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    cnn_scheduler.step()

    # Validation
    cnn_model.eval()
    with torch.no_grad():
        val_logits = cnn_model(X_val_norm)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_auc_cnn = roc_auc_score(y_val, val_probs)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.4f} | Val AUC: {val_auc_cnn:.6f}"
        )

    if val_auc_cnn > best_val_auc_cnn:
        best_val_auc_cnn = val_auc_cnn
        patience_counter = 0
        best_cnn_state = {k: v.clone() for k, v in cnn_model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1}, best AUC: {best_val_auc_cnn:.6f}"
            )
            break

# Load best CNN state
cnn_model.load_state_dict(best_cnn_state)
cnn_model.eval()

with torch.no_grad():
    val_logits_cnn = cnn_model(X_val_norm)
    val_pred_cnn = torch.sigmoid(val_logits_cnn).numpy()

val_auc_cnn_final = roc_auc_score(y_val, val_pred_cnn)
print(f"CNN - Best Val AUC: {val_auc_cnn_final:.6f}")

# ============================================================
# 7. SELECT BEST MODEL AND GENERATE TEST PREDICTIONS
# ============================================================
print(f"\n{'='*50}")
print(f"Model Comparison:")
print(f"  LightGBM Val AUC: {val_auc_lgb:.6f}")
print(f"  CNN Val AUC:      {val_auc_cnn_final:.6f}")
print(f"{'='*50}")

if val_auc_lgb >= val_auc_cnn_final:
    print("Selected model: LightGBM")
    test_pred = lgb_model.predict_proba(X_test_eng)[:, 1]
    final_auc = val_auc_lgb
else:
    print("Selected model: CNN")
    with torch.no_grad():
        test_logits = cnn_model(X_test_norm)
        test_pred = torch.sigmoid(test_logits).numpy()
    final_auc = val_auc_cnn_final

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "class": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred.mean():.4f}, min={test_pred.min():.4f}, max={test_pred.max():.4f}"
)

# ============================================================
# 9. FINAL VALIDATION SCORE
# ============================================================
print(f"\nFinal Validation Score: {final_auc:.6f}")
