import gc
import json
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# --- Step 1: Environment Setup & Directory Creation ---
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("--- Step 1: Loading Raw Parquet Datasets ---")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Loaded Train shape: {train_df.shape}")
print(f"Loaded Test shape: {test_df.shape}")

target_col = "outcomes"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in train data.")

# Encode Target Column
le_target = LabelEncoder()
y_raw = train_df[target_col].astype(str)
y = le_target.fit_transform(y_raw)
train_df["target"] = y

joblib.dump(le_target, "./working/target_label_encoder.pkl")
with open("./working/class_names.json", "w") as f:
    json.dump(list(le_target.classes_), f)

num_classes = len(le_target.classes_)
print(f"Target distribution over {num_classes} classes encoded.")

# Handle Test IDs
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))
    test_df["id"] = test_ids

np.save("./working/test_ids.npy", test_ids)

# Separate features from target in train
X_train_raw = train_df.drop(columns=[target_col, "target"], errors="ignore")
X_test_raw = test_df.drop(columns=["id", target_col], errors="ignore")

# --- Step 2: Strict Train / Validation Split to Prevent Leakage ---
X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
    X_train_raw, y, test_size=0.20, random_state=42, stratify=y
)

print(
    f"Train split size: {X_tr_raw.shape[0]}, Validation split size: {X_va_raw.shape[0]}"
)

np.save("./working/y_train.npy", y_tr)
np.save("./working/y_val.npy", y_va)


# --- Step 3: Feature Engineering Pipeline ---
def extract_datetime_features(df_input):
    df = df_input.copy()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    new_features = pd.DataFrame(index=df.index)

    for c in date_cols:
        parsed_date = pd.to_datetime(df[c], errors="coerce")
        if parsed_date.notnull().sum() > 0:
            new_features[f"{c}_year"] = parsed_date.dt.year.fillna(-1)
            new_features[f"{c}_month"] = parsed_date.dt.month.fillna(-1)
            new_features[f"{c}_day"] = parsed_date.dt.day.fillna(-1)
            new_features[f"{c}_dayofweek"] = parsed_date.dt.dayofweek.fillna(-1)
            new_features[f"{c}_dayofyear"] = parsed_date.dt.dayofyear.fillna(-1)
            new_features[f"{c}_quarter"] = parsed_date.dt.quarter.fillna(-1)
            new_features[f"{c}_is_month_end"] = parsed_date.dt.is_month_end.astype(int)

    if len(date_cols) >= 2:
        for i in range(len(date_cols)):
            for j in range(i + 1, len(date_cols)):
                d1 = pd.to_datetime(df[date_cols[i]], errors="coerce")
                d2 = pd.to_datetime(df[date_cols[j]], errors="coerce")
                delta = (d1 - d2).dt.days
                if delta.notnull().sum() > 0:
                    new_features[f"delta_{date_cols[i]}_{date_cols[j]}"] = delta.fillna(
                        -999
                    )

    return new_features


def standardize_age(df_input):
    df = df_input.copy()
    age_cols = [c for c in df.columns if "age" in c.lower()]
    unit_cols = [c for c in df.columns if "unit" in c.lower()]

    new_features = pd.DataFrame(index=df.index)
    if age_cols:
        for ac in age_cols:
            numeric_age = pd.to_numeric(df[ac], errors="coerce")
            new_features[f"{ac}_numeric"] = numeric_age
            new_features[f"{ac}_isna"] = numeric_age.isna().astype(int)

            if unit_cols:
                for uc in unit_cols:
                    unit_str = df[uc].astype(str).str.lower()
                    age_years = numeric_age.copy()

                    month_mask = unit_str.str.contains("month", na=False)
                    day_mask = unit_str.str.contains("day", na=False)
                    week_mask = unit_str.str.contains("week", na=False)

                    age_years[month_mask] = age_years[month_mask] / 12.0
                    age_years[day_mask] = age_years[day_mask] / 365.25
                    age_years[week_mask] = age_years[week_mask] / 52.18

                    new_features[f"{ac}_years_std"] = age_years
    return new_features


# 1. Process Datetime Features
tr_dt = extract_datetime_features(X_tr_raw)
va_dt = extract_datetime_features(X_va_raw)
te_dt = extract_datetime_features(X_test_raw)

# 2. Process Age / Demographic Features
tr_age = standardize_age(X_tr_raw)
va_age = standardize_age(X_va_raw)
te_age = standardize_age(X_test_raw)

# Categorize input columns
non_dt_cols = [
    c for c in X_tr_raw.columns if "date" not in c.lower() and "time" not in c.lower()
]
cat_cols = []
text_cols = []
num_cols = []

for c in non_dt_cols:
    if X_tr_raw[c].dtype == "object" or X_tr_raw[c].dtype.name == "category":
        mean_len = X_tr_raw[c].astype(str).str.len().mean()
        nunique = X_tr_raw[c].nunique()
        if mean_len > 40 or nunique > 500:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        num_cols.append(c)

print(
    f"Identified {len(cat_cols)} Categorical, {len(text_cols)} Text, and {len(num_cols)} Numeric features."
)

# Impute Numeric Features strictly based on Training medians
tr_num = pd.DataFrame(index=X_tr_raw.index)
va_num = pd.DataFrame(index=X_va_raw.index)
te_num = pd.DataFrame(index=X_test_raw.index)

for col in num_cols:
    median_val = pd.to_numeric(X_tr_raw[col], errors="coerce").median()
    if pd.isna(median_val):
        median_val = 0.0

    tr_num[col] = pd.to_numeric(X_tr_raw[col], errors="coerce").fillna(median_val)
    va_num[col] = pd.to_numeric(X_va_raw[col], errors="coerce").fillna(median_val)
    te_num[col] = pd.to_numeric(X_test_raw[col], errors="coerce").fillna(median_val)

    tr_num[f"{col}_isna"] = X_tr_raw[col].isna().astype(int)
    va_num[f"{col}_isna"] = X_va_raw[col].isna().astype(int)
    te_num[f"{col}_isna"] = X_test_raw[col].isna().astype(int)

# Frequency & Target Encoding for Categorical Columns
tr_cat = pd.DataFrame(index=X_tr_raw.index)
va_cat = pd.DataFrame(index=X_va_raw.index)
te_cat = pd.DataFrame(index=X_test_raw.index)

for col in cat_cols:
    freq_map = X_tr_raw[col].value_counts(normalize=True).to_dict()

    tr_cat[f"{col}_freq"] = X_tr_raw[col].map(freq_map).fillna(0)
    va_cat[f"{col}_freq"] = X_va_raw[col].map(freq_map).fillna(0)
    te_cat[f"{col}_freq"] = X_test_raw[col].map(freq_map).fillna(0)

    tr_str = X_tr_raw[col].astype(str)
    va_str = X_va_raw[col].astype(str)
    te_str = X_test_raw[col].astype(str)

    le = LabelEncoder()
    le.fit(tr_str)

    tr_cat[f"{col}_le"] = le.transform(tr_str)
    va_cat[f"{col}_le"] = va_str.map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    te_cat[f"{col}_le"] = te_str.map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# Out-of-fold Target Encoding
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y_tr.mean()

for col in cat_cols:
    tr_cat[f"{col}_te"] = global_mean
    va_cat[f"{col}_te"] = global_mean
    te_cat[f"{col}_te"] = global_mean

    for tr_idx, val_idx in skf.split(X_tr_raw, y_tr):
        fold_tr_x = X_tr_raw.iloc[tr_idx][col].astype(str)
        fold_tr_y = pd.Series(y_tr[tr_idx], index=fold_tr_x.index)

        means = fold_tr_y.groupby(fold_tr_x).mean()
        counts = fold_tr_y.groupby(fold_tr_x).count()
        smooth = 10
        smoothed_means = (counts * means + smooth * global_mean) / (counts + smooth)

        tr_cat.iloc[val_idx, tr_cat.columns.get_loc(f"{col}_te")] = (
            X_tr_raw.iloc[val_idx][col]
            .astype(str)
            .map(smoothed_means)
            .fillna(global_mean)
            .values
        )

    full_y = pd.Series(y_tr, index=X_tr_raw.index)
    full_means = full_y.groupby(X_tr_raw[col].astype(str)).mean()
    full_counts = full_y.groupby(X_tr_raw[col].astype(str)).count()
    full_smoothed = (full_counts * full_means + 10 * global_mean) / (full_counts + 10)

    va_cat[f"{col}_te"] = (
        X_va_raw[col].astype(str).map(full_smoothed).fillna(global_mean)
    )
    te_cat[f"{col}_te"] = (
        X_test_raw[col].astype(str).map(full_smoothed).fillna(global_mean)
    )

# Text Processing: TF-IDF + TruncatedSVD
tr_text_dfs = []
va_text_dfs = []
te_text_dfs = []

if text_cols:
    for col in text_cols:
        tr_txt = X_tr_raw[col].fillna("").astype(str)
        va_txt = X_va_raw[col].fillna("").astype(str)
        te_txt = X_test_raw[col].fillna("").astype(str)

        tr_stats = pd.DataFrame(index=X_tr_raw.index)
        va_stats = pd.DataFrame(index=X_va_raw.index)
        te_stats = pd.DataFrame(index=X_test_raw.index)

        for name, txt_series, stat_df in zip(
            ["tr", "va", "te"], [tr_txt, va_txt, te_txt], [tr_stats, va_stats, te_stats]
        ):
            stat_df[f"{col}_char_len"] = txt_series.str.len()
            stat_df[f"{col}_word_count"] = txt_series.str.split().str.len()
            stat_df[f"{col}_upper_count"] = txt_series.str.findall(r"[A-Z]").str.len()

        tr_text_dfs.append(tr_stats)
        va_text_dfs.append(va_stats)
        te_text_dfs.append(te_stats)

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), max_features=300, sublinear_tf=True, stop_words="english"
        )
        tfidf_tr = tfidf.fit_transform(tr_txt).toarray()
        tfidf_va = tfidf.transform(va_txt).toarray()
        tfidf_te = tfidf.transform(te_txt).toarray()

        tfidf_cols = [f"{col}_tfidf_{i}" for i in range(tfidf_tr.shape[1])]
        tr_text_dfs.append(
            pd.DataFrame(tfidf_tr, index=X_tr_raw.index, columns=tfidf_cols)
        )
        va_text_dfs.append(
            pd.DataFrame(tfidf_va, index=X_va_raw.index, columns=tfidf_cols)
        )
        te_text_dfs.append(
            pd.DataFrame(tfidf_te, index=X_test_raw.index, columns=tfidf_cols)
        )

# Combine engineered feature subsets
X_train_proc = pd.concat([tr_num, tr_cat, tr_dt, tr_age] + tr_text_dfs, axis=1)
X_val_proc = pd.concat([va_num, va_cat, va_dt, va_age] + va_text_dfs, axis=1)
X_test_proc = pd.concat([te_num, te_cat, te_dt, te_age] + te_text_dfs, axis=1)

X_train_proc = X_train_proc.loc[:, ~X_train_proc.columns.duplicated()]
X_val_proc = X_val_proc.loc[:, ~X_val_proc.columns.duplicated()]
X_test_proc = X_test_proc.loc[:, ~X_test_proc.columns.duplicated()]

common_cols = [
    c
    for c in X_train_proc.columns
    if c in X_val_proc.columns and c in X_test_proc.columns
]
X_train_proc = X_train_proc[common_cols]
X_val_proc = X_val_proc[common_cols]
X_test_proc = X_test_proc[common_cols]

# Scale features strictly fit on Train set
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train_proc.fillna(0))
X_va_scaled = scaler.transform(X_val_proc.fillna(0))
X_te_scaled = scaler.transform(X_test_proc.fillna(0))

joblib.dump(scaler, "./working/scaler.pkl")
np.save("./working/X_train.npy", X_tr_scaled)
np.save("./working/X_val.npy", X_va_scaled)
np.save("./working/X_test.npy", X_te_scaled)

input_dim = X_tr_scaled.shape[1]
print(
    f"--- Feature Engineering Complete: Train {X_tr_scaled.shape}, Val {X_va_scaled.shape}, Test {X_te_scaled.shape} ---"
)


# --- Step 4: Model Architecture Definitions ---
class TabularResNet(nn.Module):

    def __init__(
        self, input_dim, num_classes, hidden_dim=256, num_blocks=2, dropout=0.3
    ):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )
        self.act = nn.SiLU()

        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = x + self.act(block(x))
        out = self.head(x)
        return out


# --- Step 5: PyTorch Neural Network Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(
    torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_va_scaled, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)
)
test_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

nn_model = TabularResNet(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=25, eta_min=1e-5
)

best_val_loss = float("inf")
best_nn_weights = None
patience = 7
patience_counter = 0

print(f"Training TabularResNet on device: {device}")
for epoch in range(1, 30):
    nn_model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = nn_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * bx.size(0)

    train_loss /= len(train_dataset)
    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = nn_model(bx)
            loss = criterion(logits, by)
            val_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == by).sum().item()

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    print(
        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_nn_weights = nn_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("NN Early Stopping Triggered.")
            break

if best_nn_weights is not None:
    nn_model.load_state_dict(best_nn_weights)

# NN Inference
nn_model.eval()
val_nn_probs = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device)
        probs = torch.softmax(nn_model(bx), dim=1).cpu().numpy()
        val_nn_probs.append(probs)
val_nn_probs = np.vstack(val_nn_probs)

test_nn_probs = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device)
        probs = torch.softmax(nn_model(bx), dim=1).cpu().numpy()
        test_nn_probs.append(probs)
test_nn_probs = np.vstack(test_nn_probs)


# --- Step 6: Gradient Boosted Trees Training ---
print("Training LightGBM model...")
lgb_params = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "multiclass",
    "num_class": num_classes,
    "n_jobs": -1,
    "verbose": -1,
}
lgb_model = LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_tr_scaled,
    y_tr,
    eval_set=[(X_va_scaled, y_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
val_lgb_probs = lgb_model.predict_proba(X_va_scaled)
test_lgb_probs = lgb_model.predict_proba(X_te_scaled)

print("Training XGBoost model...")
xgb_params = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "multi:softprob",
    "num_class": num_classes,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_tr_scaled, y_tr, eval_set=[(X_va_scaled, y_va)], verbose=False)
val_xgb_probs = xgb_model.predict_proba(X_va_scaled)
test_xgb_probs = xgb_model.predict_proba(X_te_scaled)

print("Training CatBoost model...")
cat_params = {
    "iterations": 600,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "MultiClass",
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "early_stopping_rounds": 50,
}
cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(X_tr_scaled, y_tr, eval_set=(X_va_scaled, y_va), verbose=False)
val_cat_probs = cat_model.predict_proba(X_va_scaled)
test_cat_probs = cat_model.predict_proba(X_te_scaled)


# --- Step 7: Multi-Model Ensemble Blending ---
val_probs_list = [val_nn_probs, val_lgb_probs, val_xgb_probs, val_cat_probs]
test_probs_list = [test_nn_probs, test_lgb_probs, test_xgb_probs, test_cat_probs]

models_names = ["TabularResNet", "LightGBM", "XGBoost", "CatBoost"]
for name, v_prob in zip(models_names, val_probs_list):
    acc = accuracy_score(y_va, v_prob.argmax(axis=1))
    print(f"Validation Accuracy ({name}): {acc:.5f}")


def loss_func(weights):
    weights = np.array(weights)
    weights = np.maximum(0, weights)
    if weights.sum() == 0:
        return 0.0
    weights /= weights.sum()

    blended_prob = sum(w * p for w, p in zip(weights, val_probs_list))
    preds = blended_prob.argmax(axis=1)
    acc = accuracy_score(y_va, preds)
    return -acc


init_weights = [0.25, 0.25, 0.25, 0.25]

opt_res = opt.minimize(
    loss_func, init_weights, method="Nelder-Mead"
)

best_weights = opt_res.x
best_weights = np.maximum(0, best_weights)
if best_weights.sum() > 0:
    best_weights /= best_weights.sum()
else:
    best_weights = np.array(init_weights)

print(
    f"Optimized Blend Weights: ResNet={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}, XGB={best_weights[2]:.3f}, CAT={best_weights[3]:.3f}"
)

val_blended_probs = sum(w * p for w, p in zip(best_weights, val_probs_list))
final_val_acc = accuracy_score(y_va, val_blended_probs.argmax(axis=1))


# --- Step 8: Submission File Generation ---
test_blended_probs = sum(w * p for w, p in zip(best_weights, test_probs_list))
test_preds_idx = test_blended_probs.argmax(axis=1)

# Map predicted target indices back to original string labels
test_outcomes = le_target.inverse_transform(test_preds_idx)

submission_df = pd.DataFrame({"id": test_ids.astype(int), "outcomes": test_outcomes})

sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)

print(
    f"Submission saved successfully to {sub_path} with total rows: {len(submission_df)}"
)

# Final Required Output Line
print(f"Final Validation Score: {final_val_acc}")