import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Load Datasets
print("Loading datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "awardFor"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

# Process target label mapping
y_raw = train_df[target_col].astype(str)
unique_labels = sorted(y_raw.unique())
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
idx_to_label = {i: label for i, label in enumerate(unique_labels)}

y = y_raw.map(label_to_idx).values
train_df = train_df.drop(columns=[target_col])

# Extract sample IDs
train_ids = (
    train_df["id"].values if "id" in train_df.columns else np.arange(len(train_df))
)
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

n_train = len(train_df)
n_test = len(test_df)
n_classes = len(unique_labels)

# Concatenate for feature engineering pipeline
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
features_df = pd.DataFrame(index=df.index)

# 2. Extract Existing Numeric Features
num_cols = [
    c for c in df.columns if df[c].dtype in ["int64", "float64", "int32", "float32"]
]
num_cols = [
    c
    for c in num_cols
    if c.lower() not in ["id", "latitude", "lat", "longitude", "lon", "lng", "x", "y"]
]
for nc in num_cols:
    features_df[nc] = pd.to_numeric(df[nc], errors="coerce").values

# 3. Price Features Parsing
price_cols = [c for c in df.columns if "price" in c.lower()]
if price_cols:
    price_col = price_cols[0]
    raw_price = df[price_col].astype(str).fillna("")
    features_df["price_char_len"] = raw_price.apply(len)
    features_df["price_level"] = raw_price.apply(
        lambda s: (
            len(re.findall(r"[€$¥£₩]", s))
            if len(re.findall(r"[€$¥£₩]", s)) > 0
            else (
                float(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else np.nan
            )
        )
    )
    median_price = features_df["price_level"].iloc[:n_train].median()
    features_df["price_level"] = features_df["price_level"].fillna(
        median_price if not np.isnan(median_price) else 1
    )

# 4. Spatial Geometry & Distances to Culinary Capitals
lat_cols = [c for c in df.columns if c.lower() in ["latitude", "lat", "y"]]
lon_cols = [c for c in df.columns if c.lower() in ["longitude", "lon", "lng", "x"]]

if lat_cols and lon_cols:
    lat_col, lon_col = lat_cols[0], lon_cols[0]
    lat_vals = pd.to_numeric(df[lat_col], errors="coerce")
    lon_vals = pd.to_numeric(df[lon_col], errors="coerce")

    lat_med = lat_vals.iloc[:n_train].median()
    lon_med = lon_vals.iloc[:n_train].median()

    lat = lat_vals.fillna(lat_med).values
    lon = lon_vals.fillna(lon_med).values

    features_df["latitude"] = lat
    features_df["longitude"] = lon

    # Key global culinary centers
    capitals = {
        "paris": (48.8566, 2.3522),
        "tokyo": (35.6762, 139.6503),
        "ny": (40.7128, -74.0060),
        "london": (51.5074, -0.1278),
        "kyoto": (35.0116, 135.7681),
        "sf": (37.7749, -122.4194),
        "hk": (22.3193, 114.1694),
        "rome": (41.9028, 12.4964),
        "madrid": (40.4168, -3.7038),
    }

    for cap_name, (cap_lat, cap_lon) in capitals.items():
        dlat = np.radians(lat - cap_lat)
        dlon = np.radians(lon - cap_lon)
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(np.radians(cap_lat))
            * np.cos(np.radians(lat))
            * np.sin(dlon / 2.0) ** 2
        )
        features_df[f"dist_{cap_name}"] = (
            6371.0
            * 2.0
            * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1 - a, 0, 1)))
        )

    # Spatial KMeans Clustering
    coords = np.column_stack([lat, lon])
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    kmeans.fit(coords[:n_train])
    features_df["geo_cluster"] = kmeans.predict(coords)

    cluster_dists = kmeans.transform(coords)
    for k in range(12):
        features_df[f"dist_cluster_{k}"] = cluster_dists[:, k]

# 5. Multi-valued Categoricals (Cuisine & Facilities)
cuisine_cols = [c for c in df.columns if "cuisine" in c.lower()]
if cuisine_cols:
    cuisines_raw = df[cuisine_cols[0]].fillna("").astype(str)
    features_df["num_cuisines"] = cuisines_raw.apply(
        lambda s: len(re.split(r"[,/|;]", s)) if s.strip() else 0
    )

    tfidf_cuisine = TfidfVectorizer(max_features=25, token_pattern=r"(?u)\b\w+\b")
    tfidf_cuisine.fit(cuisines_raw.iloc[:n_train])
    all_cuisine_vecs = tfidf_cuisine.transform(cuisines_raw).toarray()
    for idx, feature_name in enumerate(tfidf_cuisine.get_feature_names_out()):
        features_df[f"cuisine_{feature_name}"] = all_cuisine_vecs[:, idx]

facility_cols = [
    c for c in df.columns if "facility" in c.lower() or "service" in c.lower()
]
if facility_cols:
    facilities_raw = df[facility_cols[0]].fillna("").astype(str)
    features_df["num_facilities"] = facilities_raw.apply(
        lambda s: len(re.split(r"[,/|;]", s)) if s.strip() else 0
    )

    tfidf_fac = TfidfVectorizer(max_features=25, token_pattern=r"(?u)\b\w+\b")
    tfidf_fac.fit(facilities_raw.iloc[:n_train])
    all_fac_vecs = tfidf_fac.transform(facilities_raw).toarray()
    for idx, feature_name in enumerate(tfidf_fac.get_feature_names_out()):
        features_df[f"facility_{feature_name}"] = all_fac_vecs[:, idx]

# 6. Text Semantics & SVD Dimensionality Reduction
text_cols = [
    c
    for c in df.columns
    if c.lower() in ["description", "name", "address", "summary", "review"]
]
combined_text = pd.Series("", index=df.index)

for tc in text_cols:
    col_str = df[tc].fillna("").astype(str)
    features_df[f"len_{tc}"] = col_str.apply(len)
    features_df[f"words_{tc}"] = col_str.apply(lambda s: len(s.split()))
    combined_text = combined_text + " " + col_str

tfidf_text = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), stop_words="english"
)
tfidf_train = tfidf_text.fit_transform(combined_text.iloc[:n_train])
tfidf_all = tfidf_text.transform(combined_text)

svd = TruncatedSVD(n_components=20, random_state=42)
svd.fit(tfidf_train)
svd_all = svd.transform(tfidf_all)

for k in range(20):
    features_df[f"text_svd_{k}"] = svd_all[:, k]

# 7. Categorical Frequency & Out-Of-Fold Multiclass Target Encoding
cat_cols = [
    c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"
]
cat_cols = [c for c in cat_cols if c.lower() not in text_cols + [target_col, "id"]]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    val_counts = df[col].iloc[:n_train].value_counts().to_dict()
    features_df[f"freq_{col}"] = df[col].map(val_counts).fillna(0)

    for c_idx in range(n_classes):
        target_binary = (y == c_idx).astype(float)
        oof_te = np.zeros(n_train)

        for train_idx, val_idx in skf.split(df.iloc[:n_train], y):
            fold_series = pd.Series(target_binary[train_idx], index=df.index[train_idx])
            means = fold_series.groupby(df.iloc[train_idx][col]).mean().to_dict()
            global_mean = float(target_binary[train_idx].mean())
            oof_te[val_idx] = df.iloc[val_idx][col].map(means).fillna(global_mean)

        features_df.loc[df.index[:n_train], f"te_{col}_class_{c_idx}"] = oof_te

        full_series = pd.Series(target_binary, index=df.index[:n_train])
        full_means = full_series.groupby(df.iloc[:n_train][col]).mean().to_dict()
        global_mean = float(target_binary.mean())
        features_df.loc[df.index[n_train:], f"te_{col}_class_{c_idx}"] = (
            df.iloc[n_train:][col].map(full_means).fillna(global_mean)
        )

# 8. Clean & Scale Features
X_train_df = features_df.iloc[:n_train].copy()
X_test_df = features_df.iloc[n_train:].copy()

med_fill = X_train_df.median()
X_train_df = X_train_df.fillna(med_fill).fillna(0)
X_test_df = X_test_df.fillna(med_fill).fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

X_train = pd.DataFrame(X_train_scaled, columns=features_df.columns)
X_test = pd.DataFrame(X_test_scaled, columns=features_df.columns)

input_dim = X_train.shape[1]
print(f"Feature engineering completed. Total features: {input_dim}")


# 9. Define Neural Network Architecture
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        return self.head(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Probability prediction containers
oof_lgb = np.zeros((n_train, n_classes))
oof_xgb = np.zeros((n_train, n_classes))
oof_nn = np.zeros((n_train, n_classes))

test_lgb = np.zeros((n_test, n_classes))
test_xgb = np.zeros((n_test, n_classes))
test_nn = np.zeros((n_test, n_classes))

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

# 10. Stratified K-Fold Cross Validation Training
print("Starting 5-fold cross-validation training...")
fold_num = 0

for train_idx, val_idx in skf.split(X_train, y):
    fold_num += 1
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # --- LightGBM Model ---
    lgb_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "random_state": 42 + fold_num,
        "n_jobs": -1,
    }

    model_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)
    test_lgb += model_lgb.predict_proba(X_test) / 5.0

    # --- XGBoost Model ---
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42 + fold_num,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=1000)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
    test_xgb += model_xgb.predict_proba(X_test) / 5.0

    # --- PyTorch TabularResNet ---
    nn_model = TabularResNet(input_dim=input_dim, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_ds = TensorDataset(
        torch.tensor(X_tr.values, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    X_va_tensor = torch.tensor(X_va.values, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_nn_probs = None
    best_test_nn_probs = None

    for epoch in range(30):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(X_va_tensor)
            val_loss = criterion(
                val_logits, torch.tensor(y_va, dtype=torch.long).to(device)
            ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
                test_logits = nn_model(X_test_tensor)
                best_test_nn_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

    oof_nn[val_idx] = best_nn_probs
    test_nn += best_test_nn_probs / 5.0

    fold_blend = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_nn[val_idx]) / 3.0
    fold_acc = accuracy_score(y_va, np.argmax(fold_blend, axis=1))
    print(f"Fold {fold_num}/5 - Blended Accuracy: {fold_acc:.5f}")

# 11. Ensemble Blending and Out-Of-Fold Evaluation
oof_ensemble = 0.40 * oof_lgb + 0.40 * oof_xgb + 0.20 * oof_nn
oof_preds = np.argmax(oof_ensemble, axis=1)
score = accuracy_score(y, oof_preds)

# 12. Generate Final Test Predictions and Save Submission
test_ensemble = 0.40 * test_lgb + 0.40 * test_xgb + 0.20 * test_nn
test_preds = np.argmax(test_ensemble, axis=1)
test_labels = [idx_to_label[idx] for idx in test_preds]

submission_df = pd.DataFrame({"id": test_ids, "awardFor": test_labels})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
