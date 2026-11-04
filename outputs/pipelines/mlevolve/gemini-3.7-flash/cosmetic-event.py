import os
import re
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 0. Global Setup and Seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Parquet Datasets
print("Loading raw parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df.columns = [str(c).strip() for c in train_df.columns]
test_df.columns = [str(c).strip() for c in test_df.columns]

target_col = "outcomes"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in train.parquet")

y_raw = train_df[target_col].copy()
X_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw.astype(str))
num_classes = len(label_encoder.classes_)
print(
    f"Dataset loaded: {len(X_raw)} train samples, {len(X_test_raw)} test samples, {num_classes} outcome classes."
)

# 2. Identify Modalities (Date, Text, Categorical, Numerical)
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in X_raw.columns:
    if col.lower() in ["id", "index"]:
        continue
    if any(
        k in col.lower() for k in ["date", "time", "created", "reported", "year_month"]
    ):
        date_cols.append(col)
        continue
    if pd.api.types.is_numeric_dtype(X_raw[col]):
        num_cols.append(col)
    else:
        sample_str = X_raw[col].dropna().astype(str)
        if len(sample_str) > 0 and sample_str.str.len().mean() > 40:
            text_cols.append(col)
        else:
            cat_cols.append(col)

print(
    f"Identified features -> Dates: {date_cols}, Categorical: {cat_cols}, Text: {text_cols}, Numerical: {num_cols}"
)

# 3. Create Stratified Validation Split (80/20 train/val)
train_idx, val_idx = train_test_split(
    np.arange(len(X_raw)), test_size=0.20, random_state=42, stratify=y_encoded
)


# 4. Feature Engineering Functions
def extract_date_features(df_subset, cols):
    features = pd.DataFrame(index=df_subset.index)
    date_objs = {}
    for col in cols:
        try:
            parsed = pd.to_datetime(df_subset[col], errors="coerce")
            date_objs[col] = parsed
            features[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(np.float32)
            features[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(np.float32)
            features[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(np.float32)
            features[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(
                np.float32
            )
            features[f"{col}_dayofyear"] = parsed.dt.dayofyear.fillna(-1).astype(
                np.float32
            )
            features[f"{col}_quarter"] = parsed.dt.quarter.fillna(-1).astype(np.float32)
            features[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(
                np.float32
            )
        except Exception:
            continue

    date_keys = list(date_objs.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            d1, d2 = date_keys[i], date_keys[j]
            diff_days = (date_objs[d1] - date_objs[d2]).dt.total_seconds() / (24 * 3600)
            features[f"diff_days_{d1}_{d2}"] = diff_days.fillna(-999).astype(np.float32)

    return features


def extract_num_features(df_subset, cols, train_medians=None):
    features = pd.DataFrame(index=df_subset.index)
    medians = {} if train_medians is None else train_medians
    for col in cols:
        series = pd.to_numeric(df_subset[col], errors="coerce")
        features[f"{col}_isna"] = series.isna().astype(np.float32)
        if train_medians is None:
            med = series.median()
            medians[col] = 0.0 if np.isnan(med) else med
        features[col] = series.fillna(medians[col]).astype(np.float32)
        features[f"{col}_log1p"] = np.log1p(np.maximum(0, features[col]))
    return features, medians


# Extract date features
train_date_feat = extract_date_features(X_raw.iloc[train_idx], date_cols)
val_date_feat = extract_date_features(X_raw.iloc[val_idx], date_cols)
test_date_feat = extract_date_features(X_test_raw, date_cols)

# Extract numerical features
train_num_feat, num_medians = extract_num_features(X_raw.iloc[train_idx], num_cols)
val_num_feat, _ = extract_num_features(
    X_raw.iloc[val_idx], num_cols, train_medians=num_medians
)
test_num_feat, _ = extract_num_features(X_test_raw, num_cols, train_medians=num_medians)

# Frequency & Ordinal Encodings for Categorical Features
train_cat_feat = pd.DataFrame(index=X_raw.iloc[train_idx].index)
val_cat_feat = pd.DataFrame(index=X_raw.iloc[val_idx].index)
test_cat_feat = pd.DataFrame(index=X_test_raw.index)

cat_maps = {}
freq_maps = {}

for col in cat_cols:
    train_series = X_raw.iloc[train_idx][col].fillna("__MISSING__").astype(str)
    val_series = X_raw.iloc[val_idx][col].fillna("__MISSING__").astype(str)
    test_series = X_test_raw[col].fillna("__MISSING__").astype(str)

    freq = train_series.value_counts(normalize=True).to_dict()
    freq_maps[col] = freq
    train_cat_feat[f"{col}_freq"] = (
        train_series.map(freq).fillna(0.0).astype(np.float32)
    )
    val_cat_feat[f"{col}_freq"] = val_series.map(freq).fillna(0.0).astype(np.float32)
    test_cat_feat[f"{col}_freq"] = test_series.map(freq).fillna(0.0).astype(np.float32)

    unique_vals = list(train_series.unique())
    val_to_idx = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    cat_maps[col] = val_to_idx
    train_cat_feat[f"{col}_ord"] = (
        train_series.map(val_to_idx).fillna(0).astype(np.float32)
    )
    val_cat_feat[f"{col}_ord"] = val_series.map(val_to_idx).fillna(0).astype(np.float32)
    test_cat_feat[f"{col}_ord"] = (
        test_series.map(val_to_idx).fillna(0).astype(np.float32)
    )

# Text Feature Processing: TF-IDF + SVD Dimensionality Reduction
train_text_dfs = []
val_text_dfs = []
test_text_dfs = []

for col in text_cols:
    train_text = X_raw.iloc[train_idx][col].fillna("").astype(str).tolist()
    val_text = X_raw.iloc[val_idx][col].fillna("").astype(str).tolist()
    test_text = X_test_raw[col].fillna("").astype(str).tolist()

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    svd_components = min(16, max(2, len(train_text) // 50))
    svd = TruncatedSVD(n_components=svd_components, random_state=42)

    train_tfidf = tfidf.fit_transform(train_text)
    train_svd = svd.fit_transform(train_tfidf)

    val_tfidf = tfidf.transform(val_text)
    val_svd = svd.transform(val_tfidf)

    test_tfidf = tfidf.transform(test_text)
    test_svd = svd.transform(test_tfidf)

    cols_svd = [f"{col}_svd_{k}" for k in range(svd_components)]
    train_text_dfs.append(
        pd.DataFrame(
            train_svd,
            index=X_raw.iloc[train_idx].index,
            columns=cols_svd,
            dtype=np.float32,
        )
    )
    val_text_dfs.append(
        pd.DataFrame(
            val_svd, index=X_raw.iloc[val_idx].index, columns=cols_svd, dtype=np.float32
        )
    )
    test_text_dfs.append(
        pd.DataFrame(
            test_svd, index=X_test_raw.index, columns=cols_svd, dtype=np.float32
        )
    )

train_text_feat = (
    pd.concat(train_text_dfs, axis=1)
    if train_text_dfs
    else pd.DataFrame(index=X_raw.iloc[train_idx].index)
)
val_text_feat = (
    pd.concat(val_text_dfs, axis=1)
    if val_text_dfs
    else pd.DataFrame(index=X_raw.iloc[val_idx].index)
)
test_text_feat = (
    pd.concat(test_text_dfs, axis=1)
    if test_text_dfs
    else pd.DataFrame(index=X_test_raw.index)
)

# Assemble Full Feature Matrices
X_train_processed = pd.concat(
    [train_num_feat, train_cat_feat, train_date_feat, train_text_feat], axis=1
)
X_val_processed = pd.concat(
    [val_num_feat, val_cat_feat, val_date_feat, val_text_feat], axis=1
)
X_test_processed = pd.concat(
    [test_num_feat, test_cat_feat, test_date_feat, test_text_feat], axis=1
)

# Ensure unique and safe column names for GBDT models
raw_cols = [
    re.sub(r'[\[\]<>,:\s"]+', "_", str(col)) for col in X_train_processed.columns
]
seen = {}
clean_cols = []
for col in raw_cols:
    if col in seen:
        seen[col] += 1
        clean_cols.append(f"{col}_{seen[col]}")
    else:
        seen[col] = 0
        clean_cols.append(col)

X_train_processed.columns = clean_cols
X_val_processed.columns = clean_cols
X_test_processed.columns = clean_cols

y_train = y_encoded[train_idx]
y_val = y_encoded[val_idx]

# Feature Scaling for Neural Tabular Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed.fillna(0)).astype(np.float32)
X_val_scaled = scaler.transform(X_val_processed.fillna(0)).astype(np.float32)
X_test_scaled = scaler.transform(X_test_processed.fillna(0)).astype(np.float32)

num_features = X_train_processed.shape[1]
print(f"Feature engineering complete. Total features: {num_features}")


# 5. Model Definitions
def build_lightgbm_model(num_classes, random_state=42):
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 1000,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMClassifier(**params)


def build_xgboost_model(num_classes, random_state=42):
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": random_state,
        "n_estimators": 1000,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    return xgb.XGBClassifier(**params)


class ResidualTabularBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self, in_features, num_classes, hidden_dim=256, num_blocks=3, dropout_rate=0.25
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualTabularBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


# 6. Train LightGBM
print("Training LightGBM Classifier...")
lgb_model = build_lightgbm_model(num_classes=num_classes)
lgb_callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(-1),
]
lgb_model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    callbacks=lgb_callbacks,
)
lgb_val_probs = lgb_model.predict_proba(X_val_processed)
lgb_val_acc = accuracy_score(y_val, np.argmax(lgb_val_probs, axis=1))
print(f"LightGBM Validation Accuracy: {lgb_val_acc:.5f}")

# 7. Train XGBoost
print("Training XGBoost Classifier...")
xgb_model = build_xgboost_model(num_classes=num_classes)
xgb_model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    verbose=False,
)
xgb_val_probs = xgb_model.predict_proba(X_val_processed)
xgb_val_acc = accuracy_score(y_val, np.argmax(xgb_val_probs, axis=1))
print(f"XGBoost Validation Accuracy: {xgb_val_acc:.5f}")

# 8. Train PyTorch TabularResNet
print("Training PyTorch TabularResNet...")
nn_model = TabularResNet(in_features=num_features, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)

num_epochs = 35
best_val_loss = float("inf")
best_nn_weights_path = "./working/best_tabular_resnet.pt"

for epoch in range(1, num_epochs + 1):
    nn_model.train()
    total_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = nn_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item() * len(batch_y)

    scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataset)

    nn_model.eval()
    total_val_loss = 0.0
    correct_val = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = nn_model(batch_x)
            val_loss = criterion(logits, batch_y)
            total_val_loss += val_loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct_val += (preds == batch_y).sum().item()

    avg_val_loss = total_val_loss / len(val_dataset)
    epoch_val_acc = correct_val / len(val_dataset)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(nn_model.state_dict(), best_nn_weights_path)

    if epoch % 5 == 0 or epoch == num_epochs:
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {epoch_val_acc:.5f}"
        )

nn_model.load_state_dict(
    torch.load(best_nn_weights_path, map_location=device, weights_only=True)
)
nn_model.eval()


def get_nn_probabilities(model, data_array, batch_size=256):
    loader = DataLoader(
        TensorDataset(torch.tensor(data_array, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    probs_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


nn_val_probs = get_nn_probabilities(nn_model, X_val_scaled)
nn_val_acc = accuracy_score(y_val, np.argmax(nn_val_probs, axis=1))
print(f"TabularResNet Validation Accuracy: {nn_val_acc:.5f}")

# 9. Ensemble Blending and Out-of-Fold Evaluation
weight_lgb = 0.45
weight_xgb = 0.35
weight_nn = 0.20

blend_val_probs = (
    weight_lgb * lgb_val_probs + weight_xgb * xgb_val_probs + weight_nn * nn_val_probs
)
final_val_preds = np.argmax(blend_val_probs, axis=1)
score = float(accuracy_score(y_val, final_val_preds))
print(
    f"Ensemble Model Weights: [LGB={weight_lgb}, XGB={weight_xgb}, TabularResNet={weight_nn}]"
)

# 10. Test Inference and Submission Generation
print("Generating full test inference with ensemble models...")
lgb_test_probs = lgb_model.predict_proba(X_test_processed)
xgb_test_probs = xgb_model.predict_proba(X_test_processed)
nn_test_probs = get_nn_probabilities(nn_model, X_test_scaled)

blend_test_probs = (
    weight_lgb * lgb_test_probs
    + weight_xgb * xgb_test_probs
    + weight_nn * nn_test_probs
)
final_test_preds = np.argmax(blend_test_probs, axis=1)
test_labels = label_encoder.inverse_transform(final_test_preds)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_labels), dtype=int), "outcomes": test_labels}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission file saved successfully to '{submission_path}' ({len(submission_df)} rows)."
)

# 11. Final Validation Score
print(f"Final Validation Score: {score}")