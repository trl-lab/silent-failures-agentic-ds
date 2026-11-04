import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Datasets
print("Loading datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "dea_schedule"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "schedule" in c.lower() or "dea" in c.lower()
    ]
    target_col = possible_targets[0] if possible_targets else train_df.columns[-1]

# Clean Target
train_df[target_col] = (
    train_df[target_col].fillna("UNSCHEDULED").astype(str).str.strip().str.upper()
)
train_df[target_col] = train_df[target_col].replace(
    {"": "UNSCHEDULED", "NONE": "UNSCHEDULED", "NAN": "UNSCHEDULED"}
)

lbl_enc = LabelEncoder()
y_all = lbl_enc.fit_transform(train_df[target_col])
label_classes = lbl_enc.classes_
num_classes = len(label_classes)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

df_train = train_df.drop(columns=[target_col], errors="ignore").copy()
df_test = test_df.drop(columns=[target_col], errors="ignore").copy()

# 2. Extract Base Features
CONTROLLED_KEYWORDS = {
    "kw_opioid": r"morphine|codeine|fentanyl|oxycodone|hydrocodone|buprenorphine|methadone|hydromorphone|oxymorphone|tapentadol|tramadol|meperidine|opium|sufentanil|alfentanil|remifentanil",
    "kw_benzo": r"alprazolam|diazepam|lorazepam|clonazepam|midazolam|triazolam|temazepam|oxazepam|chlordiazepoxide|clobazam|flurazepam",
    "kw_stimulant": r"amphetamine|methylphenidate|lisdexamfetamine|dexmethylphenidate|dextroamphetamine|modafinil|armodafinil|phentermine",
    "kw_barbiturate": r"phenobarbital|secobarbital|pentobarbital|amobarbital|butalbital",
    "kw_sedative": r"zolpidem|eszopiclone|zaleplon|carisoprodol|suvorexant|ramelteon",
    "kw_steroid": r"testosterone|nandrolone|oxandrolone|stanozolol|oxymetholone|fluoxymesterone",
    "kw_cannabinoid": r"dronabinol|cannabidiol|nabilone|thc",
    "kw_hallucinogen": r"ketamine|phencyclidine|psilocybin|mescaline|lysergic",
    "kw_gabapentinoid": r"pregabalin|gabapentin",
    "kw_stems": r"codone|fentanyl|pam|lam|barbital|phetamine|zolam|dione|gest",
}


def extract_domain_features(df):
    feats = pd.DataFrame(index=df.index)
    text_cols = [
        c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"
    ]
    combined_text = (
        df[text_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
    )

    for kw_name, pattern in CONTROLLED_KEYWORDS.items():
        feats[kw_name] = combined_text.str.contains(pattern, regex=True).astype(int)

    feats["char_count"] = combined_text.str.len()
    feats["word_count"] = combined_text.str.split().str.len()
    feats["ingredient_count"] = combined_text.str.count(",") + 1

    strengths = combined_text.str.extractall(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|%)")
    if not strengths.empty:
        strengths_numeric = (
            pd.to_numeric(strengths[0], errors="coerce").groupby(level=0).max()
        )
        feats["max_extracted_strength"] = df.index.map(strengths_numeric).fillna(0)
    else:
        feats["max_extracted_strength"] = 0.0

    return feats, text_cols, combined_text


train_domain, text_cols, train_text = extract_domain_features(df_train)
test_domain, _, test_text = extract_domain_features(df_test)

date_cols = [c for c in df_train.columns if "date" in c.lower()]


def extract_date_features(df, cols):
    d_feats = pd.DataFrame(index=df.index)
    for c in cols:
        dt = pd.to_datetime(df[c], errors="coerce")
        d_feats[f"{c}_year"] = dt.dt.year.fillna(-1).astype(int)
        d_feats[f"{c}_month"] = dt.dt.month.fillna(-1).astype(int)
        d_feats[f"{c}_is_null"] = dt.isna().astype(int)
    return d_feats


train_dates = extract_date_features(df_train, date_cols)
test_dates = extract_date_features(df_test, date_cols)

cat_cols = [c for c in text_cols if c not in date_cols]

# Stratified Split for Validation
train_idx, val_idx = train_test_split(
    np.arange(len(y_all)), test_size=0.2, random_state=42, stratify=y_all
)

# Frequency Encoding fitted ONLY on train split
train_freq = pd.DataFrame(index=df_train.index)
test_freq = pd.DataFrame(index=df_test.index)
for c in cat_cols:
    freq_map = (
        df_train.iloc[train_idx][c].astype(str).value_counts(normalize=True).to_dict()
    )
    train_freq[f"{c}_freq"] = df_train[c].astype(str).map(freq_map).fillna(0)
    test_freq[f"{c}_freq"] = df_test[c].astype(str).map(freq_map).fillna(0)

# Text Vectorization & SVD fitted ONLY on train split
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)
svd = TruncatedSVD(n_components=64, random_state=42)

tfidf_train_text = tfidf.fit_transform(train_text.iloc[train_idx])
svd.fit(tfidf_train_text)


def get_text_svd_features(text_series):
    vecs = tfidf.transform(text_series)
    svd_vecs = svd.transform(vecs)
    return pd.DataFrame(
        svd_vecs, index=text_series.index, columns=[f"svd_{i}" for i in range(64)]
    )


train_svd = get_text_svd_features(train_text)
test_svd = get_text_svd_features(test_text)

# Assemble feature matrices
X_all_df = pd.concat([train_domain, train_dates, train_freq, train_svd], axis=1)
X_test_df = pd.concat([test_domain, test_dates, test_freq, test_svd], axis=1)
X_all_df, X_test_df = X_all_df.align(X_test_df, join="left", axis=1, fill_value=0)

X_train_raw = X_all_df.iloc[train_idx].values
y_train = y_all[train_idx]

X_val_raw = X_all_df.iloc[val_idx].values
y_val = y_all[val_idx]

X_test_raw = X_test_df.values

# Standard scaling fitted strictly on train set
scaler = StandardScaler()
X_train = scaler.fit_transform(np.nan_to_num(X_train_raw)).astype(np.float32)
X_val = scaler.transform(np.nan_to_num(X_val_raw)).astype(np.float32)
X_test = scaler.transform(np.nan_to_num(X_test_raw)).astype(np.float32)

input_dim = X_train.shape[1]


# 3. Neural Network Architecture (TabularResNet)
class TabularResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.drop(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)


class TabularResNet(nn.Module):
    def __init__(
        self, input_dim, num_classes, hidden_dim=256, num_blocks=2, dropout=0.2
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# Setup PyTorch Training
class_weights_vals = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.tensor(class_weights_vals, dtype=torch.float32).to(device)

train_dataset = TensorDataset(
    torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

test_dataset = TensorDataset(torch.tensor(X_test))
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

nn_model = TabularResNet(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_loss = float("inf")
best_model_path = "./working/best_tabular_resnet.pt"
epochs = 30

for epoch in range(epochs):
    nn_model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = nn_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bx.size(0)
    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = nn_model(bx)
            loss = criterion(logits, by)
            val_loss += loss.item() * bx.size(0)

    val_loss /= len(val_dataset)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(nn_model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {running_loss/len(train_dataset):.4f} | Val Loss: {val_loss:.4f}"
    )

nn_model.load_state_dict(torch.load(best_model_path))
nn_model.eval()


def predict_nn_probs(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            bx = batch[0].to(device)
            probs = torch.softmax(model(bx), dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


val_probs_nn = predict_nn_probs(nn_model, val_loader)
test_probs_nn = predict_nn_probs(nn_model, test_loader)

# 4. LightGBM Model
print("Training LightGBM Model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbosity=-1,
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(-1),
    ],
)

val_probs_lgb = lgb_model.predict_proba(X_val)
test_probs_lgb = lgb_model.predict_proba(X_test)

# 5. XGBoost Model
print("Training XGBoost Model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=num_classes,
    random_state=42,
    early_stopping_rounds=40,
    eval_metric="mlogloss",
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_probs_xgb = xgb_model.predict_proba(X_val)
test_probs_xgb = xgb_model.predict_proba(X_test)

# 6. Optimization of Ensemble Weights
best_acc = -1.0
best_weights = (0.33, 0.33, 0.34)

for w_nn in np.linspace(0, 1, 21):
    for w_lgb in np.linspace(0, 1 - w_nn, 21):
        w_xgb = max(0.0, 1.0 - w_nn - w_lgb)
        blend_val_probs = (
            w_nn * val_probs_nn + w_lgb * val_probs_lgb + w_xgb * val_probs_xgb
        )
        val_preds_tmp = np.argmax(blend_val_probs, axis=1)
        acc_tmp = accuracy_score(y_val, val_preds_tmp)

        if acc_tmp > best_acc:
            best_acc = acc_tmp
            best_weights = (w_nn, w_lgb, w_xgb)

w_nn, w_lgb, w_xgb = best_weights
val_probs_ensemble = w_nn * val_probs_nn + w_lgb * val_probs_lgb + w_xgb * val_probs_xgb
test_probs_ensemble = (
    w_nn * test_probs_nn + w_lgb * test_probs_lgb + w_xgb * test_probs_xgb
)

val_preds = np.argmax(val_probs_ensemble, axis=1)
score = accuracy_score(y_val, val_preds)

# 7. Generate Submission
test_preds = np.argmax(test_probs_ensemble, axis=1)
test_schedules = label_classes[test_preds]

submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "dea_schedule": test_schedules}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")