import gc
import json
import os
import random
import re
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from xgboost import XGBClassifier


# 0. Reproducibility & Environment Setup
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 1. Load Raw Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Standardize IDs
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "Review #" in test_df.columns:
    test_ids = test_df["Review #"].values
else:
    test_ids = np.arange(len(test_df))
test_df["id"] = test_ids

if "id" not in train_df.columns:
    if "Review #" in train_df.columns:
        train_df["id"] = train_df["Review #"]
    else:
        train_df["id"] = np.arange(len(train_df))

# 2. Target Preprocessing & Class Label Mapping
target_col = "Stars"
train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

unique_stars = sorted(train_df[target_col].unique().tolist())
label_to_star = {i: float(star) for i, star in enumerate(unique_stars)}
star_to_label = {float(star): i for i, star in enumerate(unique_stars)}

train_df["target_label"] = train_df[target_col].map(star_to_label)
num_classes = len(unique_stars)

# 3. Categorical Processing & Interaction Features
cat_cols = ["Brand", "Style", "Country"]
for col in cat_cols:
    if col not in train_df.columns:
        train_df[col] = "Unknown"
    if col not in test_df.columns:
        test_df[col] = "Unknown"

    train_df[col] = train_df[col].fillna("Unknown").astype(str).str.strip()
    test_df[col] = test_df[col].fillna("Unknown").astype(str).str.strip()

train_df["Brand_Country"] = train_df["Brand"] + "_" + train_df["Country"]
test_df["Brand_Country"] = test_df["Brand"] + "_" + test_df["Country"]

train_df["Brand_Style"] = train_df["Brand"] + "_" + train_df["Style"]
test_df["Brand_Style"] = test_df["Brand"] + "_" + test_df["Style"]

train_df["Country_Style"] = train_df["Country"] + "_" + train_df["Style"]
test_df["Country_Style"] = test_df["Country"] + "_" + test_df["Style"]

all_cat_cols = [
    "Brand",
    "Style",
    "Country",
    "Brand_Country",
    "Brand_Style",
    "Country_Style",
]

# Frequency Encodings
for col in all_cat_cols:
    freq = train_df[col].value_counts(normalize=True).to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0.0)
    test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0.0)

# Initialize Target Encoding placeholder columns
target_enc_cols = ["Brand", "Country", "Brand_Country"]
for col in target_enc_cols:
    train_df[f"{col}_target_enc"] = 0.0
    test_df[f"{col}_target_enc"] = 0.0

# 4. Text Feature Extraction & SVD Embeddings
if "Variety" not in train_df.columns:
    train_df["Variety"] = ""
if "Variety" not in test_df.columns:
    test_df["Variety"] = ""

train_df["Variety"] = train_df["Variety"].fillna("").astype(str)
test_df["Variety"] = test_df["Variety"].fillna("").astype(str)

train_df["full_text"] = (
    "Brand: "
    + train_df["Brand"]
    + ". Country: "
    + train_df["Country"]
    + ". Style: "
    + train_df["Style"]
    + ". Variety: "
    + train_df["Variety"]
    + "."
)
test_df["full_text"] = (
    "Brand: "
    + test_df["Brand"]
    + ". Country: "
    + test_df["Country"]
    + ". Style: "
    + test_df["Style"]
    + ". Variety: "
    + test_df["Variety"]
    + "."
)

for df in [train_df, test_df]:
    df["variety_len"] = df["Variety"].apply(len)
    df["variety_word_count"] = df["Variety"].apply(lambda x: len(x.split()))
    df["variety_avg_word_len"] = df["variety_len"] / (df["variety_word_count"] + 1e-5)
    df["variety_has_digits"] = (
        df["Variety"].apply(lambda x: bool(re.search(r"\d", x))).astype(int)
    )
    df["variety_uppercase_count"] = df["Variety"].apply(
        lambda x: sum(1 for c in x if c.isupper())
    )

keywords = [
    "spicy",
    "hot",
    "chili",
    "chilli",
    "chicken",
    "beef",
    "pork",
    "seafood",
    "shrimp",
    "curry",
    "tom yum",
    "tonkotsu",
    "ramen",
    "miso",
    "shoyu",
    "laksa",
    "kimchi",
    "cheese",
    "veggie",
    "vegetable",
    "sauce",
    "stew",
    "dry",
    "soup",
    "instant",
    "fried",
    "black",
    "flavour",
    "flavor",
    "noodle",
]

for kw in keywords:
    kw_col = f'kw_{kw.replace(" ", "_")}'
    train_df[kw_col] = (
        train_df["full_text"].str.lower().str.contains(kw, regex=False).astype(int)
    )
    test_df[kw_col] = (
        test_df["full_text"].str.lower().str.contains(kw, regex=False).astype(int)
    )

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_embeddings = embedder.encode(
        train_df["full_text"].tolist(), show_progress_bar=False, convert_to_numpy=True
    )
    test_embeddings = embedder.encode(
        test_df["full_text"].tolist(), show_progress_bar=False, convert_to_numpy=True
    )
except Exception:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def get_embeddings(texts, batch_size=64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**encoded)
                mask = (
                    encoded["attention_mask"]
                    .unsqueeze(-1)
                    .expand(out.last_hidden_state.size())
                    .float()
                )
                sum_emb = torch.sum(out.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                mean_emb = (sum_emb / sum_mask).cpu().numpy()
                all_embs.append(mean_emb)
        return np.vstack(all_embs)

    train_embeddings = get_embeddings(train_df["full_text"].tolist())
    test_embeddings = get_embeddings(test_df["full_text"].tolist())

emb_cols = [f"emb_{i}" for i in range(train_embeddings.shape[1])]
emb_train_df = pd.DataFrame(train_embeddings, columns=emb_cols, index=train_df.index)
emb_test_df = pd.DataFrame(test_embeddings, columns=emb_cols, index=test_df.index)

train_df = pd.concat([train_df, emb_train_df], axis=1)
test_df = pd.concat([test_df, emb_test_df], axis=1)

for col in all_cat_cols:
    cat_type = train_df[col].astype("category").cat.categories
    cat_map = {cat: idx for idx, cat in enumerate(cat_type)}
    train_df[f"{col}_code"] = train_df[col].map(cat_map).fillna(-1).astype(int)
    test_df[f"{col}_code"] = test_df[col].map(cat_map).fillna(-1).astype(int)

# 5. Define Feature Set & Multi-GBDT Cross-Validation
ignore_cols = [
    "id",
    "Stars",
    "target_label",
    "Variety",
    "Brand",
    "Style",
    "Country",
    "Brand_Country",
    "Brand_Style",
    "Country_Style",
    "full_text",
]
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

X = train_df[feature_cols].fillna(0.0).values
y = train_df["target_label"].values.astype(np.int64)
X_test = test_df[feature_cols].fillna(0.0).values

n_splits = 5
skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(train_df), num_classes))
oof_xgb = np.zeros((len(train_df), num_classes))
oof_cat = np.zeros((len(train_df), num_classes))
oof_deberta = np.zeros((len(train_df), num_classes))

test_preds_lgb = np.zeros((len(test_df), num_classes))
test_preds_xgb = np.zeros((len(test_df), num_classes))
test_preds_cat = np.zeros((len(test_df), num_classes))
test_preds_deberta = np.zeros((len(test_df), num_classes))

deberta_model_name = "microsoft/deberta-v3-base"
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

for fold, (train_idx, val_idx) in enumerate(skf_cv.split(train_df, y)):
    # In-fold target encoding to avoid leakage
    tr_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]
    global_mean = tr_data["target_label"].mean()

    for col in target_enc_cols:
        enc_map = tr_data.groupby(col)["target_label"].mean().to_dict()
        train_df.loc[train_df.index[train_idx], f"{col}_target_enc"] = tr_data[col].map(enc_map).fillna(global_mean)
        train_df.loc[train_df.index[val_idx], f"{col}_target_enc"] = val_data[col].map(enc_map).fillna(global_mean)
        test_df[f"{col}_target_enc"] = test_df[col].map(enc_map).fillna(global_mean)

    X_tr = train_df.iloc[train_idx][feature_cols].fillna(0.0).values
    X_va = train_df.iloc[val_idx][feature_cols].fillna(0.0).values
    y_tr = y[train_idx]
    y_va = y[val_idx]
    X_test_fold = test_df[feature_cols].fillna(0.0).values

    # 1. LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)
    test_preds_lgb += lgb_model.predict_proba(X_test_fold) / n_splits

    # 2. XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
    test_preds_xgb += xgb_model.predict_proba(X_test_fold) / n_splits

    # 3. CatBoost Classifier
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_va)
    test_preds_cat += cat_model.predict_proba(X_test_fold) / n_splits

    # 4. DeBERTa-v3 Fine-Tuning
    tr_texts = train_df.iloc[train_idx]["full_text"].tolist()
    va_texts = train_df.iloc[val_idx]["full_text"].tolist()
    te_texts = test_df["full_text"].tolist()

    tr_ds = TextDataset(tr_texts, y_tr, tokenizer=deberta_tokenizer, max_len=128)
    va_ds = TextDataset(va_texts, y_va, tokenizer=deberta_tokenizer, max_len=128)
    te_ds = TextDataset(te_texts, None, tokenizer=deberta_tokenizer, max_len=128)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)

    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        deberta_model_name,
        num_labels=num_classes,
    )
    deberta_model.to(device)

    epochs = 4
    total_steps = len(tr_loader) * epochs
    optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        deberta_model.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = deberta_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    deberta_model.eval()
    val_probs = []
    with torch.no_grad():
        for batch in va_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            val_probs.append(probs)
    oof_deberta[val_idx] = np.vstack(val_probs)

    test_probs = []
    with torch.no_grad():
        for batch in te_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            test_probs.append(probs)
    test_preds_deberta += np.vstack(test_probs) / n_splits

    del deberta_model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

# Ensemble OOF and Test Predictions across models
oof_ensemble = (oof_lgb + oof_xgb + oof_cat + oof_deberta) / 4.0
test_ensemble = (test_preds_lgb + test_preds_xgb + test_preds_cat + test_preds_deberta) / 4.0

oof_preds = np.argmax(oof_ensemble, axis=1)
val_accuracy = accuracy_score(y, oof_preds)

test_pred_labels = np.argmax(test_ensemble, axis=1)
test_pred_stars = [label_to_star[lbl] for lbl in test_pred_labels]

submission = pd.DataFrame({"id": np.arange(len(test_df)), "Stars": test_pred_stars})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_accuracy}")