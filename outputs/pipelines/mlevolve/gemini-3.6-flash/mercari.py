import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, SGDRegressor
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

# 1. Setup Environment and Reproducibility
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Raw Parquet Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Filter non-positive target rows
train_df = train_df[train_df["price"] > 0].reset_index(drop=True)

# Determine test set IDs
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "test_id" in test_df.columns:
    test_ids = test_df["test_id"].values
else:
    test_ids = np.arange(len(test_df))

# 3. Create Train / Validation Split FIRST (Prevents Data Leakage)
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)


# Helper function to preprocess text series
def preprocess_text(series, fill_value=""):
    return series.fillna(fill_value).astype(str).str.lower()


# Extract textual series
train_name = preprocess_text(train_data["name"])
val_name = preprocess_text(val_data["name"])
test_name = preprocess_text(test_df["name"])

train_desc = preprocess_text(
    train_data["item_description"], fill_value="no description yet"
)
val_desc = preprocess_text(
    val_data["item_description"], fill_value="no description yet"
)
test_desc = preprocess_text(
    test_df["item_description"], fill_value="no description yet"
)

def extract_subcategories(series):
    s = preprocess_text(series, fill_value="missing")
    split_df = s.str.split("/", n=2, expand=True)
    c0 = split_df[0].fillna("missing")
    c1 = split_df[1].fillna("missing") if 1 in split_df.columns else pd.Series(["missing"] * len(series))
    c2 = split_df[2].fillna("missing") if 2 in split_df.columns else pd.Series(["missing"] * len(series))
    return c0, c1, c2

train_cat0, train_cat1, train_cat2 = extract_subcategories(train_data["category_name"])
val_cat0, val_cat1, val_cat2 = extract_subcategories(val_data["category_name"])
test_cat0, test_cat1, test_cat2 = extract_subcategories(test_df["category_name"])

train_brand = preprocess_text(train_data["brand_name"], fill_value="missing")
val_brand = preprocess_text(val_data["brand_name"], fill_value="missing")
test_brand = preprocess_text(test_df["brand_name"], fill_value="missing")

train_cat_raw = preprocess_text(train_data["category_name"], fill_value="missing")
val_cat_raw = preprocess_text(val_data["category_name"], fill_value="missing")
test_cat_raw = preprocess_text(test_df["category_name"], fill_value="missing")

train_brand_cat = train_brand + " " + train_cat_raw
val_brand_cat = val_brand + " " + val_cat_raw
test_brand_cat = test_brand + " " + test_cat_raw


# Frequency maps for high-cardinality categorical fields calculated strictly on training data
brand_freq_map = train_brand.value_counts(normalize=True).to_dict()
cat_freq_map = train_cat_raw.value_counts(normalize=True).to_dict()


# Helper to extract dense metadata features and frequency encodings
def extract_dense_features(df, name_s, desc_s, brand_s, cat_s):
    name_len = name_s.str.len().values.reshape(-1, 1)
    name_words = name_s.str.split().str.len().values.reshape(-1, 1)
    desc_len = desc_s.str.len().values.reshape(-1, 1)
    desc_words = desc_s.str.split().str.len().values.reshape(-1, 1)
    len_ratio = (name_len / (desc_len + 1.0)).reshape(-1, 1)
    word_ratio = (name_words / (desc_words + 1.0)).reshape(-1, 1)
    is_desc_missing = (desc_s == "no description yet").astype(int).values.reshape(-1, 1)
    is_brand_missing = (brand_s == "missing").astype(int).values.reshape(-1, 1)
    brand_freq = brand_s.map(brand_freq_map).fillna(0).values.reshape(-1, 1)
    cat_freq = cat_s.map(cat_freq_map).fillna(0).values.reshape(-1, 1)
    return np.hstack(
        [
            name_len,
            name_words,
            desc_len,
            desc_words,
            len_ratio,
            word_ratio,
            is_desc_missing,
            is_brand_missing,
            brand_freq,
            cat_freq,
        ]
    )


train_dense = extract_dense_features(
    train_data, train_name, train_desc, train_brand, train_cat_raw
)
val_dense = extract_dense_features(
    val_data, val_name, val_desc, val_brand, val_cat_raw
)
test_dense = extract_dense_features(
    test_df, test_name, test_desc, test_brand, test_cat_raw
)

# 4. Fit Feature Transformers STRICTLY on Training Data
tfidf_name = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=35000,
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
tfidf_name_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 6),
    max_features=30000,
    min_df=3,
)
tfidf_desc = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=45000,
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
tfidf_desc_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 6),
    max_features=25000,
    min_df=3,
)
tfidf_brand_cat = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[\w\s&-]+\b",
)
count_cat0 = CountVectorizer(token_pattern=r"(?u)\b[\w\s&-]+\b", max_features=5000)
count_cat1 = CountVectorizer(token_pattern=r"(?u)\b[\w\s&-]+\b", max_features=5000)
count_cat2 = CountVectorizer(token_pattern=r"(?u)\b[\w\s&-]+\b", max_features=5000)
count_brand = CountVectorizer(token_pattern=r"(?u)\b[\w\s&-]+\b", max_features=3000)

scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown="ignore")

# Fit transformers
tfidf_name.fit(train_name)
tfidf_name_char.fit(train_name)
tfidf_desc.fit(train_desc)
tfidf_desc_char.fit(train_desc)
tfidf_brand_cat.fit(train_brand_cat)
count_cat0.fit(train_cat0)
count_cat1.fit(train_cat1)
count_cat2.fit(train_cat2)
count_brand.fit(train_brand)
scaler.fit(train_dense)

# Categorical column preparation
train_cat_cols = train_data[["item_condition_id", "shipping"]].fillna(0).astype(int)
val_cat_cols = val_data[["item_condition_id", "shipping"]].fillna(0).astype(int)
test_cat_cols = test_df[["item_condition_id", "shipping"]].fillna(0).astype(int)
ohe.fit(train_cat_cols)


# Transform matrices helper
def build_feature_matrix(name_s, desc_s, c0_s, c1_s, c2_s, brand_s, brand_cat_s, cat_cols, dense_arr):
    X_name = tfidf_name.transform(name_s)
    X_name_char = tfidf_name_char.transform(name_s)
    X_desc = tfidf_desc.transform(desc_s)
    X_desc_char = tfidf_desc_char.transform(desc_s)
    X_brand_cat = tfidf_brand_cat.transform(brand_cat_s)
    X_cat0 = count_cat0.transform(c0_s)
    X_cat1 = count_cat1.transform(c1_s)
    X_cat2 = count_cat2.transform(c2_s)
    X_brand = count_brand.transform(brand_s)
    X_ohe = ohe.transform(cat_cols)
    X_dense = scaler.transform(dense_arr)
    return hstack([X_name, X_name_char, X_desc, X_desc_char, X_brand_cat, X_cat0, X_cat1, X_cat2, X_brand, X_ohe, X_dense]).tocsr()


X_train = build_feature_matrix(
    train_name, train_desc, train_cat0, train_cat1, train_cat2, train_brand, train_brand_cat, train_cat_cols, train_dense
)
X_val = build_feature_matrix(
    val_name, val_desc, val_cat0, val_cat1, val_cat2, val_brand, val_brand_cat, val_cat_cols, val_dense
)
X_test = build_feature_matrix(
    test_name, test_desc, test_cat0, test_cat1, test_cat2, test_brand, test_brand_cat, test_cat_cols, test_dense
)

# Targets in log1p scale
y_train = np.log1p(train_data["price"].values)
y_val = np.log1p(val_data["price"].values)
val_price_raw = val_data["price"].values


# 5. Define Model Architecture (SparseResNet)
class SparseResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.drop1(self.act1(self.ln1(self.fc_in(x))))
        res1 = h
        h = self.drop2(self.act2(self.ln2(self.res_fc1(h))))
        h = h + res1
        res2 = h
        h = self.drop3(self.act3(self.ln3(self.res_fc2(h))))
        h = h + res2
        out = self.fc_out(h)
        return out


# Helper for sparse CSR mini-batch handling
def get_batch(X_sparse, y_array, indices, device):
    X_batch = X_sparse[indices].toarray()
    X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
    if y_array is not None:
        y_batch = y_array[indices]
        y_tensor = torch.tensor(y_batch, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        return X_tensor, y_tensor
    return X_tensor


# 6. Instantiate PyTorch Components
input_dim = X_train.shape[1]
model = SparseResNet(input_dim=input_dim, hidden_dim=256, dropout=0.2).to(device)

criterion = nn.SmoothL1Loss(beta=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

epochs = 8
batch_size = 4096
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def predict_sparse(model, X_sparse, device, batch_size=4096):
    model.eval()
    n = X_sparse.shape[0]
    preds = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_idx = np.arange(start_idx, end_idx)
            X_b = get_batch(X_sparse, None, batch_idx, device)
            out = model(X_b)
            preds[start_idx:end_idx] = out.cpu().numpy().ravel()
    return preds


def evaluate_loss(model, X_sparse, y_array, criterion, device, batch_size=4096):
    model.eval()
    n = X_sparse.shape[0]
    total_loss = 0.0
    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_idx = np.arange(start_idx, end_idx)
            X_b, y_b = get_batch(X_sparse, y_array, batch_idx, device)
            preds = model(X_b)
            loss = criterion(preds, y_b)
            total_loss += loss.item() * len(batch_idx)
    return total_loss / n


# 7. Train Neural Network with Early Stopping
best_val_loss = float("inf")
patience = 3
patience_counter = 0
best_model_path = "./working/best_sparse_resnet.pt"

n_samples = X_train.shape[0]
for epoch in range(epochs):
    model.train()
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    total_train_loss = 0.0

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        X_b, y_b = get_batch(X_train, y_train, batch_idx, device)

        optimizer.zero_grad()
        preds = model(X_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * len(batch_idx)

    scheduler.step()
    train_loss = total_train_loss / n_samples
    val_loss = evaluate_loss(
        model, X_val, y_val, criterion, device, batch_size=batch_size
    )

    print(
        f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# Load Best NN Checkpoint
model.load_state_dict(torch.load(best_model_path))

# Model Inferences
val_preds_nn = predict_sparse(model, X_val, device)
test_preds_nn = predict_sparse(model, X_test, device)

# 8. Train Complementary Linear (Ridge, SGD) and LightGBM Models
ridge_model = Ridge(alpha=2.5, solver="sag", random_state=42)
ridge_model.fit(X_train, y_train)

val_preds_ridge = ridge_model.predict(X_val)
test_preds_ridge = ridge_model.predict(X_test)

sgd_model = SGDRegressor(loss="squared_error", penalty="l2", max_iter=1000, random_state=42)
sgd_model.fit(X_train, y_train)

val_preds_sgd = sgd_model.predict(X_val)
test_preds_sgd = sgd_model.predict(X_test)

lgb_model = LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.08,
    random_state=42,
    n_jobs=-1,
)
lgb_model.fit(X_train, y_train)

val_preds_lgb = lgb_model.predict(X_val)
test_preds_lgb = lgb_model.predict(X_test)

# 9. Dynamic Bounded Optimization for Ensemble Weights
val_preds_matrix = np.column_stack([val_preds_ridge, val_preds_sgd, val_preds_nn, val_preds_lgb])
test_preds_matrix = np.column_stack([test_preds_ridge, test_preds_sgd, test_preds_nn, test_preds_lgb])


def objective(weights):
    weights = weights / np.sum(weights)
    blend_log = val_preds_matrix @ weights
    blend_price = np.clip(np.expm1(blend_log), 0, None)
    return np.sqrt(mean_squared_error(val_price_raw, blend_price))


init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
constraints = ({"type": "eq", "fun": lambda w: 1.0 - np.sum(w)})

opt_res = minimize(objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
opt_weights = opt_res.x / np.sum(opt_res.x)

val_preds_log = val_preds_matrix @ opt_weights
test_preds_log = test_preds_matrix @ opt_weights

val_preds_price = np.clip(np.expm1(val_preds_log), 0, None)
test_preds_price = np.clip(np.expm1(test_preds_log), 0, None)

# 10. Compute Validation Score & Export Predictions
val_rmse = np.sqrt(mean_squared_error(val_price_raw, val_preds_price))

submission = pd.DataFrame({"id": test_ids, "price": test_preds_price})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_rmse:.5f}")
