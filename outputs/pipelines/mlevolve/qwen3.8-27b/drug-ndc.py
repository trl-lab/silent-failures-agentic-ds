import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix
import pickle

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"\nTarget distribution:\n{train_df['dea_schedule'].value_counts()}")

# ============================================================
# 2. IDENTIFY COLUMNS AND DATA TYPES
# ============================================================
target_col = "dea_schedule"
y_train_full = train_df[target_col].values
X_train_full = train_df.drop(columns=[target_col])

numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
object_cols = X_train_full.select_dtypes(include=["object"]).columns.tolist()
print(f"\nNumeric columns: {numeric_cols}")
print(f"Object columns: {object_cols}")

# Find NDC column
ndc_col = None
for col in object_cols:
    if "ndc" in col.lower():
        ndc_col = col
        break

if ndc_col is None:
    for col in X_train_full.columns:
        if "ndc" in col.lower() or "product_code" in col.lower():
            ndc_col = col
            break

print(f"NDC column: {ndc_col}")

# ============================================================
# 3. STRATIFIED TRAIN/VAL SPLIT
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)
print(f"Train split: {X_train.shape[0]} samples")
print(f"Validation split: {X_val.shape[0]} samples")
print(f"Test set: {test_df.shape[0]} samples")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================


# --- 4a. NDC Code Decomposition ---
def decompose_ndc(ndc_str):
    if pd.isna(ndc_str) or not isinstance(ndc_str, str):
        return np.nan, np.nan, np.nan
    ndc_clean = str(ndc_str).replace("-", "").replace(" ", "")
    if len(ndc_clean) == 9 and ndc_clean.isdigit():
        return int(ndc_clean[:3]), int(ndc_clean[3:7]), int(ndc_clean[7:9])
    elif len(ndc_clean) == 10 and ndc_clean.isdigit():
        return int(ndc_clean[:4]), int(ndc_clean[4:8]), int(ndc_clean[8:10])
    else:
        parts = str(ndc_str).split("-")
        if len(parts) == 3:
            try:
                return int(parts[0]), int(parts[1]), int(parts[2])
            except:
                return np.nan, np.nan, np.nan
        return np.nan, np.nan, np.nan


test_proc = (
    test_df.drop(columns=[target_col])
    if target_col in test_df.columns
    else test_df.copy()
)

if ndc_col is not None:
    print(f"Sample NDC values: {X_train[ndc_col].dropna().head(5).tolist()}")

    for df_name, df in [("train", X_train), ("val", X_val), ("test", test_proc)]:
        decomposed = df[ndc_col].apply(lambda x: pd.Series(decompose_ndc(x)))
        decomposed.columns = ["ndc_manufacturer", "ndc_product", "ndc_package"]
        if df_name == "train":
            X_train = pd.concat([X_train, decomposed], axis=1)
        elif df_name == "val":
            X_val = pd.concat([X_val, decomposed], axis=1)
        else:
            test_proc = pd.concat([test_proc, decomposed], axis=1)

# --- 4b. Text Feature Engineering ---
all_text_cols = [c for c in object_cols if c != ndc_col]
print(f"Text columns: {all_text_cols}")

vectorizer = CountVectorizer(
    analyzer="char_wb", ngram_range=(2, 5), max_features=5000, min_df=2, max_df=0.95
)

train_texts = (
    X_train[all_text_cols].fillna("").apply(lambda row: " | ".join(row.values), axis=1)
)
X_train_text = vectorizer.fit_transform(train_texts)

val_texts = (
    X_val[all_text_cols].fillna("").apply(lambda row: " | ".join(row.values), axis=1)
)
X_val_text = vectorizer.transform(val_texts)

test_texts = (
    test_proc[all_text_cols]
    .fillna("")
    .apply(lambda row: " | ".join(row.values), axis=1)
)
X_test_text = vectorizer.transform(test_texts)

print(f"Text feature matrix (train): {X_train_text.shape}")

# --- 4c. Categorical Encoding ---
cat_cols = [c for c in object_cols if c not in all_text_cols and c != ndc_col]
if ndc_col:
    cat_cols = [c for c in cat_cols if c != ndc_col]

print(f"Categorical columns: {cat_cols}")

if cat_cols:
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("__MISSING__")
        X_val[col] = X_val[col].fillna("__MISSING__")
        test_proc[col] = test_proc[col].fillna("__MISSING__")

    ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_cat = ordinal_enc.fit_transform(X_train[cat_cols])
    X_val_cat = ordinal_enc.transform(X_val[cat_cols])
    X_test_cat = ordinal_enc.transform(test_proc[cat_cols])

    X_train_cat = pd.DataFrame(
        X_train_cat, columns=[f"cat_{c}" for c in cat_cols], index=X_train.index
    )
    X_val_cat = pd.DataFrame(
        X_val_cat, columns=[f"cat_{c}" for c in cat_cols], index=X_val.index
    )
    X_test_cat = pd.DataFrame(
        X_test_cat, columns=[f"cat_{c}" for c in cat_cols], index=test_proc.index
    )
else:
    X_train_cat = pd.DataFrame(index=X_train.index)
    X_val_cat = pd.DataFrame(index=X_val.index)
    X_test_cat = pd.DataFrame(index=test_proc.index)

# --- 4d. Numeric Features ---
numeric_feature_cols = numeric_cols.copy()
if ndc_col is not None:
    numeric_feature_cols.extend(["ndc_manufacturer", "ndc_product", "ndc_package"])

print(f"Numeric feature columns: {numeric_feature_cols}")

numeric_medians = {}
for col in numeric_feature_cols:
    if col in X_train.columns:
        numeric_medians[col] = X_train[col].median()
        X_train[col] = X_train[col].fillna(numeric_medians[col])
        X_val[col] = X_val[col].fillna(numeric_medians[col])
        if col in test_proc.columns:
            test_proc[col] = test_proc[col].fillna(numeric_medians[col])

X_train_num = X_train[numeric_feature_cols].values.astype(np.float64)
X_val_num = X_val[numeric_feature_cols].values.astype(np.float64)
X_test_num = test_proc[numeric_feature_cols].values.astype(np.float64)

# --- 4e. Additional Features ---
if ndc_col is not None and "ndc_manufacturer" in X_train.columns:
    mfr_freq = X_train["ndc_manufacturer"].value_counts()
    X_train["ndc_mfr_freq"] = X_train["ndc_manufacturer"].map(mfr_freq).fillna(0)
    X_val["ndc_mfr_freq"] = X_val["ndc_manufacturer"].map(mfr_freq).fillna(0)
    test_proc["ndc_mfr_freq"] = test_proc["ndc_manufacturer"].map(mfr_freq).fillna(0)

    X_train_num = np.column_stack([X_train_num, X_train["ndc_mfr_freq"].values])
    X_val_num = np.column_stack([X_val_num, X_val["ndc_mfr_freq"].values])
    X_test_num = np.column_stack([X_test_num, test_proc["ndc_mfr_freq"].values])

train_text_len = train_texts.str.len().values.reshape(-1, 1)
val_text_len = val_texts.str.len().values.reshape(-1, 1)
test_text_len = test_texts.str.len().values.reshape(-1, 1)

train_word_count = train_texts.str.split().str.len().values.reshape(-1, 1)
val_word_count = val_texts.str.split().str.len().values.reshape(-1, 1)
test_word_count = test_texts.str.split().str.len().values.reshape(-1, 1)

# ============================================================
# 5. COMBINE ALL FEATURES
# ============================================================
X_train_final = hstack(
    [
        csr_matrix(X_train_num),
        csr_matrix(X_train_cat.values),
        X_train_text,
        csr_matrix(train_text_len),
        csr_matrix(train_word_count),
    ]
)

X_val_final = hstack(
    [
        csr_matrix(X_val_num),
        csr_matrix(X_val_cat.values),
        X_val_text,
        csr_matrix(val_text_len),
        csr_matrix(val_word_count),
    ]
)

X_test_final = hstack(
    [
        csr_matrix(X_test_num),
        csr_matrix(X_test_cat.values),
        X_test_text,
        csr_matrix(test_text_len),
        csr_matrix(test_word_count),
    ]
)

print(f"\nFinal feature matrix shapes:")
print(f"  Train: {X_train_final.shape}")
print(f"  Val:   {X_val_final.shape}")
print(f"  Test:  {X_test_final.shape}")

# ============================================================
# 6. LABEL ENCODING
# ============================================================
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_val_enc = label_enc.transform(y_val)

n_classes = len(label_enc.classes_)
print(f"\nLabel encoding: {dict(zip(label_enc.classes_, range(n_classes)))}")
print(f"Number of classes: {n_classes}")


# ============================================================
# 7. MODEL ARCHITECTURE
# ============================================================
class DrugScheduleNet(nn.Module):
    def __init__(self, input_dim, n_classes, hidden1=256, hidden2=128, dropout=0.3):
        super(DrugScheduleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 8. SETUP TRAINING
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

n_features = X_train_final.shape[1]

# Convert to dense tensors
X_train_dense = torch.FloatTensor(X_train_final.toarray())
X_val_dense = torch.FloatTensor(X_val_final.toarray())
X_test_dense = torch.FloatTensor(X_test_final.toarray())

y_train_tensor = torch.LongTensor(y_train_enc)
y_val_tensor = torch.LongTensor(y_val_enc)

train_dataset = TensorDataset(X_train_dense, y_train_tensor)
val_dataset = TensorDataset(X_val_dense, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)

# Class weights
class_counts = np.bincount(y_train_enc, minlength=n_classes)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * n_classes
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print(f"Class weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

model = DrugScheduleNet(
    input_dim=n_features, n_classes=n_classes, hidden1=256, hidden2=128, dropout=0.3
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Architecture: {n_features} -> 256 -> 128 -> {n_classes}")

# ============================================================
# 9. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_val_acc = 0.0
best_model_state = None
patience_counter = 0
max_patience = 20
max_epochs = 300

print(
    f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>8} | {'LR':>8}"
)
print("-" * 60)

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(batch_y).sum().item()
        train_total += batch_X.size(0)

    train_acc = train_correct / train_total
    train_loss /= train_total

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(batch_y).sum().item()
            val_total += batch_X.size(0)

    val_acc = val_correct / val_total
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"{epoch+1:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_acc:>8.4f} | {current_lr:>8.6f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
        break

# ============================================================
# 10. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Validation predictions
with torch.no_grad():
    X_val_device = X_val_dense.to(device)
    val_outputs = model(X_val_device)
    val_preds_enc = val_outputs.argmax(dim=1).cpu().numpy()

val_accuracy = accuracy_score(y_val_enc, val_preds_enc)
print(f"\nValidation Accuracy (official metric): {val_accuracy:.4f}")

# Test predictions
with torch.no_grad():
    X_test_device = X_test_dense.to(device)
    test_outputs = model(X_test_device)
    test_preds_enc = test_outputs.argmax(dim=1).cpu().numpy()

test_preds = label_enc.inverse_transform(test_preds_enc)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "dea_schedule": test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: ./submission/submission.csv")
print(f"  Shape: {submission.shape}")
print(f"  Prediction distribution:")
print(submission["dea_schedule"].value_counts().to_string())

torch.save(best_model_state, "./working/best_model_state.pth")

# ============================================================
# 12. FINAL SCORE
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score:.4f}")
