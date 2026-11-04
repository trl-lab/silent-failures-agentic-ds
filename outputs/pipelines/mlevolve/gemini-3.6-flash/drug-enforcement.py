import os
import pickle
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)

# Ensure working and submission directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data & Target Encoding
# ---------------------------------------------------------
print("Loading dataset...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "classification"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "class" in c.lower() or "target" in c.lower()
    ]
    target_col = possible_targets[0] if possible_targets else train_df.columns[-1]

print(
    f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Target: {target_col}"
)

train_ids = train_df["id"].values if "id" in train_df.columns else np.arange(len(train_df))
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

y_raw = train_df[target_col].astype(str).values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

with open("./working/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

drop_cols_train = [c for c in [target_col, "id"] if c in train_df.columns]
drop_cols_test = [c for c in [target_col, "id"] if c in test_df.columns]
features_train_df = train_df.drop(columns=drop_cols_train).copy()
features_test_df = test_df.drop(columns=drop_cols_test).copy()

# ---------------------------------------------------------
# 2. Text Feature Extraction & Domain Keywords
# ---------------------------------------------------------
text_cols = [
    col for col in features_train_df.columns
    if features_train_df[col].dtype == "object" or isinstance(
        features_train_df[col].dtype, pd.CategoricalDtype
    )
]

def combine_text_fields(df, cols):
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)

train_text = combine_text_fields(features_train_df, text_cols)
test_text = combine_text_fields(features_test_df, text_cols)

text_stats_train = pd.DataFrame(index=features_train_df.index)
text_stats_test = pd.DataFrame(index=features_test_df.index)

text_stats_train["text_char_len"] = train_text.str.len()
text_stats_test["text_char_len"] = test_text.str.len()

text_stats_train["text_word_count"] = train_text.str.split().str.len()
text_stats_test["text_word_count"] = test_text.str.split().str.len()

text_stats_train["text_avg_word_len"] = text_stats_train["text_char_len"] / (
    text_stats_train["text_word_count"] + 1e-5
)
text_stats_test["text_avg_word_len"] = text_stats_test["text_char_len"] / (
    text_stats_test["text_word_count"] + 1e-5
)

risk_keywords = [
    "hazard", "death", "serious", "adverse", "contam", "potency", "steril",
    "particulate", "subpoten", "superpoten", "label", "mislabel", "foreign",
    "microbiol", "fungal", "bacteri", "leak", "defect", "expir", "dosage",
    "purity", "allergen", "recall", "class i", "class ii", "class iii",
    "life-threatening", "injury", "temporary", "medication", "tablet", "injection"
]

for kw in risk_keywords:
    kw_col = f"kw_{kw.replace(' ', '_').replace('-', '_')}"
    text_stats_train[kw_col] = train_text.str.lower().str.count(kw).astype(np.float32)
    text_stats_test[kw_col] = test_text.str.lower().str.count(kw).astype(np.float32)

train_indices_init, val_indices_init = train_test_split(
    np.arange(len(features_train_df)),
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

word_tfidf = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    stop_words="english",
)
char_tfidf = TfidfVectorizer(
    max_features=25000,
    analyzer="char_wb",
    ngram_range=(2, 5),
    sublinear_tf=True,
)

word_tfidf.fit(train_text.iloc[train_indices_init])
char_tfidf.fit(train_text.iloc[train_indices_init])

X_tfidf_word_tr = word_tfidf.transform(train_text)
X_tfidf_char_tr = char_tfidf.transform(train_text)
X_tfidf_train = hstack([X_tfidf_word_tr, X_tfidf_char_tr]).tocsr()

X_tfidf_word_te = word_tfidf.transform(test_text)
X_tfidf_char_te = char_tfidf.transform(test_text)
X_tfidf_test = hstack([X_tfidf_word_te, X_tfidf_char_te]).tocsr()

n_svd = min(100, X_tfidf_train.shape[1] - 1) if X_tfidf_train.shape[1] > 1 else 1
svd = TruncatedSVD(n_components=n_svd, random_state=42)
svd.fit(X_tfidf_train[train_indices_init])
svd_train = svd.transform(X_tfidf_train)
svd_test = svd.transform(X_tfidf_test)

svd_cols = [f"svd_{i}" for i in range(n_svd)]
svd_train_df = pd.DataFrame(svd_train, columns=svd_cols, index=features_train_df.index)
svd_test_df = pd.DataFrame(svd_test, columns=svd_cols, index=features_test_df.index)

# Out-Of-Fold Logistic Regression Meta-Features on full sparse TF-IDF
lr_meta_train = np.zeros((len(features_train_df), num_classes), dtype=np.float32)
lr_meta_test = np.zeros((len(features_test_df), num_classes), dtype=np.float32)

skf_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in skf_lr.split(train_indices_init, y_encoded[train_indices_init]):
    real_tr_idx = train_indices_init[tr_idx]
    real_val_idx = train_indices_init[val_idx]
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tfidf_train[real_tr_idx], y_encoded[real_tr_idx])
    lr_meta_train[real_val_idx] = clf.predict_proba(X_tfidf_train[real_val_idx])

clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_full.fit(X_tfidf_train[train_indices_init], y_encoded[train_indices_init])
lr_meta_train[val_indices_init] = clf_full.predict_proba(X_tfidf_train[val_indices_init])
lr_meta_test = clf_full.predict_proba(X_tfidf_test)

lr_cols = [f"tfidf_lr_prob_class_{c}" for c in range(num_classes)]
lr_train_df = pd.DataFrame(lr_meta_train, columns=lr_cols, index=features_train_df.index)
lr_test_df = pd.DataFrame(lr_meta_test, columns=lr_cols, index=features_test_df.index)

with open("./working/word_tfidf.pkl", "wb") as f:
    pickle.dump(word_tfidf, f)
with open("./working/char_tfidf.pkl", "wb") as f:
    pickle.dump(char_tfidf, f)
with open("./working/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
with open("./working/lr_meta_model.pkl", "wb") as f:
    pickle.dump(clf_full, f)

# ---------------------------------------------------------
# 3. Categorical, Date, and Numeric Feature Engineering
# ---------------------------------------------------------
non_text_cols = [c for c in features_train_df.columns if c not in text_cols]
cat_cols = []
date_cols = []
num_cols = []

for col in non_text_cols:
    if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(
        features_train_df[col]
    ):
        date_cols.append(col)
    elif (
        features_train_df[col].dtype == "object"
        or features_train_df[col].dtype.name == "category"
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

date_feats_train = pd.DataFrame(index=features_train_df.index)
date_feats_test = pd.DataFrame(index=features_test_df.index)

for col in date_cols:
    d_tr = pd.to_datetime(features_train_df[col], errors="coerce")
    d_te = pd.to_datetime(features_test_df[col], errors="coerce")

    date_feats_train[f"{col}_year"] = d_tr.dt.year.fillna(-1)
    date_feats_test[f"{col}_year"] = d_te.dt.year.fillna(-1)

    date_feats_train[f"{col}_month"] = d_tr.dt.month.fillna(-1)
    date_feats_test[f"{col}_month"] = d_te.dt.month.fillna(-1)

    date_feats_train[f"{col}_day"] = d_tr.dt.day.fillna(-1)
    date_feats_test[f"{col}_day"] = d_te.dt.day.fillna(-1)

    date_feats_train[f"{col}_dayofweek"] = d_tr.dt.dayofweek.fillna(-1)
    date_feats_test[f"{col}_dayofweek"] = d_te.dt.dayofweek.fillna(-1)

cat_feats_train = pd.DataFrame(index=features_train_df.index)
cat_feats_test = pd.DataFrame(index=features_test_df.index)

for col in cat_cols:
    freq = (
        features_train_df[col].fillna("Missing").value_counts(normalize=True).to_dict()
    )
    cat_feats_train[f"{col}_freq"] = (
        features_train_df[col].fillna("Missing").map(freq).fillna(0)
    )
    cat_feats_test[f"{col}_freq"] = (
        features_test_df[col].fillna("Missing").map(freq).fillna(0)
    )

    le = LabelEncoder()
    tr_v = features_train_df[col].fillna("Missing").astype(str)
    te_v = features_test_df[col].fillna("Missing").astype(str)
    all_v = pd.concat([tr_v, te_v], axis=0)
    le.fit(all_v)
    cat_feats_train[f"{col}_le"] = le.transform(tr_v)
    cat_feats_test[f"{col}_le"] = le.transform(te_v)

num_feats_train = pd.DataFrame(index=features_train_df.index)
num_feats_test = pd.DataFrame(index=features_test_df.index)

for col in num_cols:
    med = features_train_df[col].median()
    num_feats_train[col] = features_train_df[col].fillna(med)
    num_feats_test[col] = features_test_df[col].fillna(med)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_feats_train = pd.DataFrame(index=features_train_df.index)
te_feats_test = pd.DataFrame(index=features_test_df.index)

for col in cat_cols:
    for c_idx in range(num_classes):
        target_binary = (y_encoded == c_idx).astype(float)
        col_te_train = np.zeros(len(features_train_df))

        for tr_idx, val_idx in skf.split(features_train_df, y_encoded):
            means = (
                pd.DataFrame(
                    {
                        "cat": features_train_df[col].iloc[tr_idx].fillna("Missing"),
                        "target": target_binary[tr_idx],
                    }
                )
                .groupby("cat")["target"]
                .mean()
                .to_dict()
            )
            col_te_train[val_idx] = (
                features_train_df[col]
                .iloc[val_idx]
                .fillna("Missing")
                .map(means)
                .fillna(target_binary[tr_idx].mean())
            )

        global_means = (
            pd.DataFrame(
                {
                    "cat": features_train_df[col].fillna("Missing"),
                    "target": target_binary,
                }
            )
            .groupby("cat")["target"]
            .mean()
            .to_dict()
        )
        col_te_test = (
            features_test_df[col]
            .fillna("Missing")
            .map(global_means)
            .fillna(target_binary.mean())
        )

        te_feats_train[f"{col}_te_class_{c_idx}"] = col_te_train
        te_feats_test[f"{col}_te_class_{c_idx}"] = col_te_test

# ---------------------------------------------------------
# Split raw features before fitting transformers to avoid data leakage
# ---------------------------------------------------------
train_indices, val_indices = train_test_split(
    np.arange(len(features_train_df)),
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

# Target encoding computed cleanly without leakage
te_feats_train = pd.DataFrame(index=features_train_df.index)
te_feats_test = pd.DataFrame(index=features_test_df.index)
for col in cat_cols:
    for c_idx in range(num_classes):
        target_binary = (y_encoded == c_idx).astype(float)
        global_means_tr = (
            pd.DataFrame(
                {
                    "cat": features_train_df[col].iloc[train_indices].fillna("Missing"),
                    "target": target_binary[train_indices],
                }
            )
            .groupby("cat")["target"]
            .mean()
            .to_dict()
        )
        te_feats_train[f"{col}_te_class_{c_idx}"] = (
            features_train_df[col]
            .fillna("Missing")
            .map(global_means_tr)
            .fillna(target_binary[train_indices].mean())
        )
        te_feats_test[f"{col}_te_class_{c_idx}"] = (
            features_test_df[col]
            .fillna("Missing")
            .map(global_means_tr)
            .fillna(target_binary[train_indices].mean())
        )

# Concatenate all tabular engineered features
X_all_train = pd.concat(
    [
        num_feats_train,
        cat_feats_train,
        date_feats_train,
        text_stats_train,
        svd_train_df,
        lr_train_df,
        te_feats_train,
    ],
    axis=1,
)

X_all_test = pd.concat(
    [
        num_feats_test,
        cat_feats_test,
        date_feats_test,
        text_stats_test,
        svd_test_df,
        lr_test_df,
        te_feats_test,
    ],
    axis=1,
)

X_all_train = X_all_train.loc[:, ~X_all_train.columns.duplicated()]
X_all_test = X_all_test.loc[:, ~X_all_test.columns.duplicated()]
X_all_test = X_all_test[X_all_train.columns]

X_tr_df = X_all_train.iloc[train_indices].copy().fillna(0)
X_val_df = X_all_train.iloc[val_indices].copy().fillna(0)
X_test_df = X_all_test.copy().fillna(0)

y_tr = y_encoded[train_indices]
y_val = y_encoded[val_indices]

# Scaled data specifically prepared for Neural Network
scaler = StandardScaler()
X_tr_scaled = np.nan_to_num(scaler.fit_transform(X_tr_df), nan=0.0)
X_val_scaled = np.nan_to_num(scaler.transform(X_val_df), nan=0.0)
X_test_scaled = np.nan_to_num(scaler.transform(X_test_df), nan=0.0)

input_dim = X_tr_df.shape[1]
print(
    f"Feature engineering complete! Features: {input_dim}. Train: {len(X_tr_df)}, Val: {len(X_val_df)}, Test: {len(X_test_df)}"
)

# ---------------------------------------------------------
# 5. Train LightGBM Model
# ---------------------------------------------------------
print("--- Training LightGBM Model ---")
lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

dtrain = lgb.Dataset(X_tr_df, label=y_tr)
dval = lgb.Dataset(X_val_df, label=y_val, reference=dtrain)

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

lgb_val_probs = lgb_model.predict(X_val_df)
lgb_test_probs = lgb_model.predict(X_test_df)
lgb_val_acc = accuracy_score(y_val, np.argmax(lgb_val_probs, axis=1))
print(f"LightGBM Validation Accuracy: {lgb_val_acc:.5f}")

# ---------------------------------------------------------
# 5b. Train XGBoost Model
# ---------------------------------------------------------
print("--- Training XGBoost Model ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    n_jobs=-1,
)
xgb_model.fit(
    X_tr_df, y_tr,
    eval_set=[(X_tr_df, y_tr), (X_val_df, y_val)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(X_val_df)
xgb_test_probs = xgb_model.predict_proba(X_test_df)
xgb_val_acc = accuracy_score(y_val, np.argmax(xgb_val_probs, axis=1))
print(f"XGBoost Validation Accuracy: {xgb_val_acc:.5f}")

# ---------------------------------------------------------
# 5c. Train CatBoost Model
# ---------------------------------------------------------
print("--- Training CatBoost Model ---")
cb_model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    early_stopping_rounds=50,
    verbose=False,
    thread_count=-1,
)
cb_model.fit(
    X_tr_df, y_tr,
    eval_set=(X_val_df, y_val),
    verbose=False,
)

cb_val_probs = cb_model.predict_proba(X_val_df)
cb_test_probs = cb_model.predict_proba(X_test_df)
cb_val_acc = accuracy_score(y_val, np.argmax(cb_val_probs, axis=1))
print(f"CatBoost Validation Accuracy: {cb_val_acc:.5f}")


# ---------------------------------------------------------
# 6. Train PyTorch Tabular ResNet Architecture
# ---------------------------------------------------------
print("--- Training PyTorch Tabular ResNet ---")


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act1 = nn.SiLU()

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
        )
        self.shortcut2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.SiLU()

        self.head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.act1(h + self.block1(h))
        h = self.act2(self.shortcut2(h) + self.block2(h))
        return self.head(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = TabularResNet(
    input_dim=input_dim, num_classes=num_classes, hidden_dim=256, dropout_rate=0.3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

epochs = 60
best_val_loss = float("inf")
patience = 12
patience_counter = 0

for epoch in range(1, epochs + 1):
    nn_model.train()
    total_train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = nn_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch_x.size(0)

    train_loss = total_train_loss / len(train_dataset)

    nn_model.eval()
    total_val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            total_val_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()

    val_loss = total_val_loss / len(val_dataset)
    val_acc = correct / len(val_dataset)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(nn_model.state_dict(), "./working/best_nn_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

nn_model.load_state_dict(torch.load("./working/best_nn_model.pth"))
nn_model.eval()


def predict_nn_probs(model, tensor_data, device):
    loader = DataLoader(TensorDataset(tensor_data), batch_size=256, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    return np.vstack(probs_list)


nn_val_probs = predict_nn_probs(nn_model, X_val_tensor, device)
nn_test_probs = predict_nn_probs(nn_model, X_test_tensor, device)
nn_val_acc = accuracy_score(y_val, np.argmax(nn_val_probs, axis=1))
print(f"TabularResNet Validation Accuracy: {nn_val_acc:.5f}")

# ---------------------------------------------------------
# 7. Model Ensembling & Optimal Blend Optimization
# ---------------------------------------------------------
best_weight = 0.5
best_ensemble_acc = 0.0

for w in np.linspace(0.0, 1.0, 101):
    blend_val_probs = w * lgb_val_probs + (1.0 - w) * nn_val_probs
    blend_val_preds = np.argmax(blend_val_probs, axis=1)
    blend_acc = accuracy_score(y_val, blend_val_preds)
    if blend_acc > best_ensemble_acc:
        best_ensemble_acc = blend_acc
        best_weight = w

print(
    f"Optimal LightGBM Blend Weight: {best_weight:.2f} (NN Weight: {1.0 - best_weight:.2f})"
)

final_val_probs = best_weight * lgb_val_probs + (1.0 - best_weight) * nn_val_probs
final_val_preds = np.argmax(final_val_probs, axis=1)
final_val_score = accuracy_score(y_val, final_val_preds)

# ---------------------------------------------------------
# 8. Test Set Inference & Submission
# ---------------------------------------------------------
final_test_probs = best_weight * lgb_test_probs + (1.0 - best_weight) * nn_test_probs
final_test_preds = np.argmax(final_test_probs, axis=1)

test_pred_labels = label_encoder.inverse_transform(final_test_preds)

submission_df = pd.DataFrame({"id": test_ids, "classification": test_pred_labels})

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}."
)

print(f"Final Validation Score: {final_val_score}")