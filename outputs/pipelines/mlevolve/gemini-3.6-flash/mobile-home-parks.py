import os
import re
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import xgboost as xgb

warnings.filterwarnings("ignore")

# 0. Global Setup & Seed Preservation
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

TARGET_COL = "SIZE"
if TARGET_COL not in train_df.columns:
    cols_lower = {col.lower(): col for col in train_df.columns}
    if "size" in cols_lower:
        TARGET_COL = cols_lower["size"]

train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_df[TARGET_COL].astype(str))
num_classes = len(target_encoder.classes_)

X_train_raw = train_df.drop(columns=[TARGET_COL]).copy()
X_test_raw = test_df.drop(columns=[TARGET_COL], errors="ignore").copy()

n_train = len(X_train_raw)
n_test = len(X_test_raw)

combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# 2. Preprocessing & Feature Engineering
sentinels = [
    -999,
    -9999,
    -999.0,
    "NOT AVAILABLE",
    "UNKNOWN",
    "UNAVAILABLE",
    "N/A",
    "NONE",
    "",
]
combined_df = combined_df.replace(sentinels, np.nan)

lat_col = next(
    (col for col in combined_df.columns if col.upper() in ["LATITUDE", "LAT", "Y"]),
    None,
)
lon_col = next(
    (
        col
        for col in combined_df.columns
        if col.upper() in ["LONGITUDE", "LON", "LONG", "X"]
    ),
    None,
)
name_col = next(
    (
        col
        for col in combined_df.columns
        if col.upper() in ["NAME", "PARK_NAME", "FACILITY_NAME", "TITLE"]
    ),
    None,
)

features_df = pd.DataFrame(index=combined_df.index)

# Spatial Features
if lat_col and lon_col:
    lat = pd.to_numeric(combined_df[lat_col], errors="coerce").fillna(0)
    lon = pd.to_numeric(combined_df[lon_col], errors="coerce").fillna(0)

    features_df["lat"] = lat
    features_df["lon"] = lon
    features_df["lat_lon_sum"] = lat + lon
    features_df["lat_lon_diff"] = lat - lon
    features_df["lat_lon_ratio"] = lat / (lon.replace(0, 1e-5))

    angles = [15, 30, 45, 60, 75]
    for angle in angles:
        rad = np.radians(angle)
        features_df[f"rot_{angle}_x"] = lat * np.cos(rad) - lon * np.sin(rad)
        features_df[f"rot_{angle}_y"] = lat * np.sin(rad) + lon * np.cos(rad)

    centroid_lat, centroid_lon = 39.8283, -98.5795
    dlat = lat - centroid_lat
    dlon = lon - centroid_lon
    features_df["dist_us_center"] = np.sqrt(dlat**2 + dlon**2)
    features_df["angle_us_center"] = np.arctan2(dlat, dlon)

    train_spatial = np.column_stack([lat[:n_train], lon[:n_train]])
    all_spatial = np.column_stack([lat, lon])
    for n_clusters in [10, 25, 50]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        kmeans.fit(train_spatial)
        features_df[f"spatial_cluster_{n_clusters}"] = kmeans.predict(all_spatial)

    # Spatial K-NN distance features fitted exclusively on train coordinates
    nn_model = NearestNeighbors(n_neighbors=11, algorithm="kd_tree")
    nn_model.fit(train_spatial)
    train_dist, _ = nn_model.kneighbors(train_spatial)
    test_dist, _ = nn_model.kneighbors(all_spatial[n_train:])

    k_pairs = [(1, 0, "knn_dist_1st"), (3, 2, "knn_dist_3rd"), (5, 4, "knn_dist_5th"), (10, 9, "knn_dist_10th")]
    for train_k, test_k, col_name in k_pairs:
        col_vals = np.zeros(len(all_spatial))
        col_vals[:n_train] = train_dist[:, train_k]
        col_vals[n_train:] = test_dist[:, test_k]
        features_df[col_name] = col_vals

    # Multi-radius spatial density counts using KDTree
    spatial_tree = KDTree(train_spatial)
    for r in [0.02, 0.05, 0.1, 0.25, 0.5]:
        densities = spatial_tree.query_ball_point(all_spatial, r, return_length=True)
        if isinstance(densities, np.ndarray):
            features_df[f"spatial_density_r_{str(r).replace('.', '_')}"] = densities
        else:
            features_df[f"spatial_density_r_{str(r).replace('.', '_')}"] = [len(x) for x in densities]

# Administrative State/County Features
admin_cols = [
    col for col in combined_df.columns
    if col.upper() in ["STATE", "ST", "STATE_NAME", "STATE_ABBR", "COUNTY", "COUNTY_NAME", "COUNTYFIPS", "CNTY", "CITY"]
]

for col in admin_cols:
    val_str = combined_df[col].fillna("MISSING").astype(str)
    freq_map = val_str[:n_train].value_counts().to_dict()
    features_df[f"{col}_admin_count"] = val_str.map(freq_map).fillna(0)

    if lat_col and lon_col:
        lat_vals = pd.to_numeric(combined_df[lat_col], errors="coerce")
        lon_vals = pd.to_numeric(combined_df[lon_col], errors="coerce")
        lat_mean_map = lat_vals[:n_train].groupby(val_str[:n_train]).mean().to_dict()
        lat_std_map = lat_vals[:n_train].groupby(val_str[:n_train]).std().fillna(0).to_dict()
        lon_mean_map = lon_vals[:n_train].groupby(val_str[:n_train]).mean().to_dict()
        lon_std_map = lon_vals[:n_train].groupby(val_str[:n_train]).std().fillna(0).to_dict()
        features_df[f"{col}_lat_mean"] = val_str.map(lat_mean_map).fillna(0)
        features_df[f"{col}_lat_std"] = val_str.map(lat_std_map).fillna(0)
        features_df[f"{col}_lon_mean"] = val_str.map(lon_mean_map).fillna(0)
        features_df[f"{col}_lon_std"] = val_str.map(lon_std_map).fillna(0)

# Text & Name Semantics Features
if name_col:
    name_text = combined_df[name_col].fillna("").astype(str).str.upper()

    features_df["name_char_len"] = name_text.str.len()
    features_df["name_word_count"] = name_text.str.split().str.len()
    features_df["name_has_digits"] = name_text.str.contains(r"\d").astype(int)

    # Embedded numerical quantities in park names
    def extract_name_num_stats(text_series):
        max_nums, min_nums, count_nums, total_digits = [], [], [], []
        for text in text_series:
            nums = [float(n) for n in re.findall(r"\d+", text)]
            digits_count = sum(c.isdigit() for c in text)
            total_digits.append(digits_count)
            if nums:
                max_nums.append(max(nums))
                min_nums.append(min(nums))
                count_nums.append(len(nums))
            else:
                max_nums.append(-1.0)
                min_nums.append(-1.0)
                count_nums.append(0)
        return max_nums, min_nums, count_nums, total_digits

    max_n, min_n, count_n, tot_d = extract_name_num_stats(name_text)
    features_df["name_max_number"] = max_n
    features_df["name_min_number"] = min_n
    features_df["name_count_numbers"] = count_n
    features_df["name_digit_count"] = tot_d

    domain_keywords = [
        "ESTATES",
        "VILLAGE",
        "PARK",
        "COMMUNITY",
        "MOBILE",
        "RESORT",
        "COURT",
        "MHP",
        "GARDENS",
        "TRAILER",
        "CLUB",
        "BAY",
        "LAKE",
        "RIVER",
        "PALM",
        "SUN",
        "OAK",
        "PINES",
        "SENIOR",
        "55+",
        "MANOR",
    ]
    for kw in domain_keywords:
        features_df[f"kw_{kw.lower()}"] = name_text.str.contains(
            kw, regex=False
        ).astype(int)

    tfidf_word = TfidfVectorizer(
        max_features=500, stop_words="english", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_word.fit(name_text[:n_train])
    tfidf_word_mat = tfidf_word.transform(name_text)

    tfidf_char = TfidfVectorizer(
        max_features=500, analyzer="char_wb", ngram_range=(2, 4)
    )
    tfidf_char.fit(name_text[:n_train])
    tfidf_char_mat = tfidf_char.transform(name_text)

    tfidf_mat = hstack([tfidf_word_mat, tfidf_char_mat])

    n_components = min(20, tfidf_mat.shape[1] - 1)
    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        svd.fit(tfidf_mat.tocsr()[:n_train])
        svd_cols = svd.transform(tfidf_mat)
        for i in range(n_components):
            features_df[f"tfidf_lsa_{i}"] = svd_cols[:, i]

# Categorical & Frequency Encoding
cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()
if name_col in cat_cols:
    cat_cols.remove(name_col)

for col in cat_cols:
    val_str = combined_df[col].fillna("MISSING").astype(str)
    features_df[f"{col}_isna"] = combined_df[col].isna().astype(int)

    freq_map = val_str[:n_train].value_counts().to_dict()
    features_df[f"{col}_freq"] = val_str.map(freq_map).fillna(0)
    features_df[f"{col}_code"] = LabelEncoder().fit_transform(val_str)

# Numerical Features Processing
num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if col in [lat_col, lon_col]:
        continue
    vals = pd.to_numeric(combined_df[col], errors="coerce")
    med_val = vals[:n_train].median()
    if pd.isna(med_val):
        med_val = 0.0
    features_df[f"{col}_val"] = vals.fillna(med_val)
    features_df[f"{col}_isna"] = combined_df[col].isna().astype(int)
    if vals[:n_train].min() >= 0:
        features_df[f"{col}_log"] = np.log1p(np.maximum(0, vals.fillna(0)))

# Sanitize Column Names
clean_cols = []
seen = set()
for c in features_df.columns:
    clean_c = re.sub(r"[^\w]", "_", str(c))
    base_c = clean_c
    cnt = 1
    while clean_c in seen:
        clean_c = f"{base_c}_{cnt}"
        cnt += 1
    seen.add(clean_c)
    clean_cols.append(clean_c)
features_df.columns = clean_cols

X_train = features_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = features_df.iloc[n_train:].copy().reset_index(drop=True)

# Out-Of-Fold Multi-Class Target Encoding for Administrative Categorical Variables
if admin_cols:
    alpha = 10.0
    te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for col in admin_cols:
        val_str = combined_df[col].fillna("MISSING").astype(str)
        col_train = val_str.iloc[:n_train].reset_index(drop=True)
        col_test = val_str.iloc[n_train:].reset_index(drop=True)

        for c in range(num_classes):
            te_col_name = f"{col}_te_class_{c}"
            X_train[te_col_name] = 0.0
            X_test[te_col_name] = 0.0

            prior = np.mean(y == c)

            for tr_idx, va_idx in te_skf.split(col_train, y):
                tr_vals = col_train.iloc[tr_idx]
                tr_y = y[tr_idx]

                cat_counts = tr_vals.value_counts()
                class_counts = tr_vals[tr_y == c].value_counts()

                te_map = (class_counts + alpha * prior) / (cat_counts + alpha)
                va_vals = col_train.iloc[va_idx]
                X_train.loc[va_idx, te_col_name] = va_vals.map(te_map).fillna(prior).values

            full_cat_counts = col_train.value_counts()
            full_class_counts = col_train[y == c].value_counts()
            full_te_map = (full_class_counts + alpha * prior) / (full_cat_counts + alpha)
            X_test[te_col_name] = col_test.map(full_te_map).fillna(prior).values

input_dim = X_train.shape[1]


# 3. Neural Architecture Definition
class TabularResBlock(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.fc1(self.act(self.norm1(x)))
        out = self.drop(out)
        out = self.fc2(self.act(self.norm2(out)))
        out = self.drop(out)
        return residual + out


class TabularResNet(nn.Module):

    def __init__(
        self, input_dim, num_classes, hidden_dim=256, num_blocks=3, dropout=0.25
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head_norm = nn.BatchNorm1d(hidden_dim)
        self.head_act = nn.SiLU()
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_drop(self.head_act(self.head_norm(h)))
        return self.head(h)


# 4. Model Configurations
lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
}

catboost_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "learning_rate": 0.04,
    "depth": 6,
    "iterations": 700,
    "random_seed": SEED,
    "verbose": False,
}

et_params = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": SEED,
    "n_jobs": -1,
}

hgb_params = {
    "max_iter": 300,
    "learning_rate": 0.04,
    "max_depth": 6,
    "random_state": SEED,
}

# 5. Stratified K-Fold Training Loop
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_preds_lgb = np.zeros((n_train, num_classes))
oof_preds_xgb = np.zeros((n_train, num_classes))
oof_preds_cb = np.zeros((n_train, num_classes))
oof_preds_et = np.zeros((n_train, num_classes))
oof_preds_hgb = np.zeros((n_train, num_classes))
oof_preds_nn = np.zeros((n_train, num_classes))

test_preds_lgb = np.zeros((n_test, num_classes))
test_preds_xgb = np.zeros((n_test, num_classes))
test_preds_cb = np.zeros((n_test, num_classes))
test_preds_et = np.zeros((n_test, num_classes))
test_preds_hgb = np.zeros((n_test, num_classes))
test_preds_nn = np.zeros((n_test, num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    train_data_lgb = lgb.Dataset(X_tr, label=y_tr)
    val_data_lgb = lgb.Dataset(X_va, label=y_va, reference=train_data_lgb)
    model_lgb = lgb.train(
        lgb_params,
        train_data_lgb,
        num_boost_round=600,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(
        X_va, num_iteration=model_lgb.best_iteration
    )
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        **xgb_params, n_estimators=600, early_stopping_rounds=50
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_va)
    test_preds_xgb += model_xgb.predict_proba(X_test) / n_splits

    # CatBoost
    model_cb = CatBoostClassifier(**catboost_params, early_stopping_rounds=50)
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_preds_cb[val_idx] = model_cb.predict_proba(X_va)
    test_preds_cb += model_cb.predict_proba(X_test) / n_splits

    # Imputed arrays for ExtraTrees and HistGradientBoosting
    X_tr_imp = np.nan_to_num(X_tr.values, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_imp = np.nan_to_num(X_va.values, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_imp = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0)

    # ExtraTrees
    model_et = ExtraTreesClassifier(**et_params)
    model_et.fit(X_tr_imp, y_tr)
    oof_preds_et[val_idx] = model_et.predict_proba(X_va_imp)
    test_preds_et += model_et.predict_proba(X_te_imp) / n_splits

    # HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(**hgb_params)
    model_hgb.fit(X_tr_imp, y_tr)
    oof_preds_hgb[val_idx] = model_hgb.predict_proba(X_va_imp)
    test_preds_hgb += model_hgb.predict_proba(X_te_imp) / n_splits

    # PyTorch TabularResNet
    scaler = StandardScaler()
    X_tr_sc = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0, posinf=0.0, neginf=0.0)
    X_va_sc = np.nan_to_num(scaler.transform(X_va), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_sc = np.nan_to_num(scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

    tr_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=128, shuffle=True)

    nn_model = TabularResNet(
        input_dim=input_dim, num_classes=num_classes, hidden_dim=256, num_blocks=3
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(40):
        nn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            va_x = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            va_logits = nn_model(va_x)
            va_loss = criterion(
                va_logits, torch.tensor(y_va, dtype=torch.long).to(device)
            ).item()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_weights)
    nn_model.to(device)
    nn_model.eval()
    with torch.no_grad():
        va_x = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        te_x = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        oof_preds_nn[val_idx] = torch.softmax(nn_model(va_x), dim=1).cpu().numpy()
        test_preds_nn += torch.softmax(nn_model(te_x), dim=1).cpu().numpy() / n_splits

# 6. Ensemble Weighting & Validation Assessment
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = (
        weights[0] * oof_preds_lgb
        + weights[1] * oof_preds_xgb
        + weights[2] * oof_preds_cb
        + weights[3] * oof_preds_et
        + weights[4] * oof_preds_hgb
        + weights[5] * oof_preds_nn
    )
    blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
    return log_loss(y, blend)

init_weights = [1.0 / 6.0] * 6
bounds = [(0.0, 1.0) for _ in range(6)]
constraints = ({"type": "eq", "fun": lambda w: 1.0 - np.sum(w)})

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-7},
)

opt_weights = res.x / np.sum(res.x)

oof_ensemble = (
    opt_weights[0] * oof_preds_lgb
    + opt_weights[1] * oof_preds_xgb
    + opt_weights[2] * oof_preds_cb
    + opt_weights[3] * oof_preds_et
    + opt_weights[4] * oof_preds_hgb
    + opt_weights[5] * oof_preds_nn
)
test_ensemble = (
    opt_weights[0] * test_preds_lgb
    + opt_weights[1] * test_preds_xgb
    + opt_weights[2] * test_preds_cb
    + opt_weights[3] * test_preds_et
    + opt_weights[4] * test_preds_hgb
    + opt_weights[5] * test_preds_nn
)

# Optimize class probability threshold multipliers directly for multi-class accuracy
def acc_objective(multipliers):
    scaled_probs = oof_ensemble * multipliers
    preds = np.argmax(scaled_probs, axis=1)
    return -accuracy_score(y, preds)

init_multipliers = np.ones(num_classes)
res_nm = minimize(
    acc_objective,
    init_multipliers,
    method="Nelder-Mead",
    options={"maxiter": 1000}
)
opt_multipliers = res_nm.x

oof_predictions = np.argmax(oof_ensemble * opt_multipliers, axis=1)
score = accuracy_score(y, oof_predictions)

# 7. Generate Submission File
test_predictions = np.argmax(test_ensemble * opt_multipliers, axis=1)
predicted_labels = target_encoder.inverse_transform(test_predictions)

if "id" in test_df.columns:
    submission_ids = test_df["id"]
elif "ID" in test_df.columns:
    submission_ids = test_df["ID"]
else:
    submission_ids = np.arange(n_test)

submission_df = pd.DataFrame({"id": submission_ids, "SIZE": predicted_labels})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")