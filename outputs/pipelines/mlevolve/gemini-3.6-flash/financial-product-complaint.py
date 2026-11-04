import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureGating(nn.Module):
    """Dynamically scales feature dimensions using a squeeze-and-excitation inspired module."""

    def __init__(self, input_dim: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(16, input_dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    """Residual dense block with LayerNorm, GELU activation, and Dropout regularization."""

    def __init__(self, dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GatedResMLP(nn.Module):
    """Deep Gated Residual MLP architecture optimized for hybrid tabular and SVD text features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.feature_gating = FeatureGating(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gated = self.feature_gating(x)
        h = self.input_proj(x_gated)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for c in date_cols:
        dt = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"] = dt.dt.year.fillna(-1).astype(np.int16)
        df[f"{c}_month"] = dt.dt.month.fillna(-1).astype(np.int8)
        df[f"{c}_day"] = dt.dt.day.fillna(-1).astype(np.int8)
        df[f"{c}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.int8)

    rec_col = next(
        (c for c in date_cols if "received" in c.lower() or "rec" in c.lower()), None
    )
    sent_col = next((c for c in date_cols if "sent" in c.lower()), None)
    if rec_col and sent_col:
        rec_dt = pd.to_datetime(df[rec_col], errors="coerce")
        sent_dt = pd.to_datetime(df[sent_col], errors="coerce")
        df["days_rec_to_sent"] = (sent_dt - rec_dt).dt.days.fillna(-1)

    return df


def main():
    set_seed(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "Company response to consumer"
    assert target_col in train_df.columns, f"Target column '{target_col}' missing."

    # Encode Target Variable
    label_encoder = LabelEncoder()
    train_df["target"] = label_encoder.fit_transform(train_df[target_col].astype(str))
    num_classes = len(label_encoder.classes_)

    # Leakage Prevention: Stratified Split FIRST
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.20,
        random_state=42,
        stratify=train_df["target"],
    )

    train_split = train_split.copy()
    val_split = val_split.copy()
    test_df = test_df.copy()

    y_train = train_split["target"].values
    y_val = val_split["target"].values

    # 1. Date Features
    train_split = extract_date_features(train_split)
    val_split = extract_date_features(val_split)
    test_df = extract_date_features(test_df)

    # 2. Text Features & SVD
    text_cols = [
        c
        for c in train_split.columns
        if "narrative" in c.lower() or "issue" in c.lower() or "response" in c.lower()
    ]
    if "Consumer complaint narrative" in train_split.columns:
        narrative_col = "Consumer complaint narrative"
    elif len(text_cols) > 0:
        narrative_col = text_cols[0]
    else:
        narrative_col = None

    if narrative_col:
        for df in [train_split, val_split, test_df]:
            raw_text = df[narrative_col].fillna("").astype(str)
            df["text_len"] = raw_text.str.len()
            df["text_word_count"] = raw_text.str.split().str.len()
            df["text_has_narrative"] = (df["text_len"] > 0).astype(int)

        tfidf_word = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        svd_word = TruncatedSVD(n_components=30, random_state=42)

        train_text = train_split[narrative_col].fillna("").astype(str)
        val_text = val_split[narrative_col].fillna("").astype(str)
        test_text = test_df[narrative_col].fillna("").astype(str)

        train_tfidf = tfidf_word.fit_transform(train_text)
        val_tfidf = tfidf_word.transform(val_text)
        test_tfidf = tfidf_word.transform(test_text)

        train_svd = svd_word.fit_transform(train_tfidf)
        val_svd = svd_word.transform(val_tfidf)
        test_svd = svd_word.transform(test_tfidf)

        for i in range(30):
            train_split[f"svd_word_{i}"] = train_svd[:, i]
            val_split[f"svd_word_{i}"] = val_svd[:, i]
            test_df[f"svd_word_{i}"] = test_svd[:, i]

    # 3. Categorical Encodings
    cat_cols = [
        c
        for c in train_split.columns
        if train_split[c].dtype == "object"
        and c not in [target_col, narrative_col]
        and not c.startswith("svd_")
    ]

    if "Product" in cat_cols and "Submitted via" in cat_cols:
        for df in [train_split, val_split, test_df]:
            df["Prod_x_Submitted"] = (
                df["Product"].astype(str) + "_" + df["Submitted via"].astype(str)
            )
        cat_cols.append("Prod_x_Submitted")

    for col in cat_cols:
        freq_map = train_split[col].astype(str).value_counts(normalize=True).to_dict()
        train_split[f"{col}_freq"] = (
            train_split[col].astype(str).map(freq_map).fillna(0.0)
        )
        val_split[f"{col}_freq"] = val_split[col].astype(str).map(freq_map).fillna(0.0)
        test_df[f"{col}_freq"] = test_df[col].astype(str).map(freq_map).fillna(0.0)

        cat_encoder = LabelEncoder()
        all_vals = pd.concat(
            [
                train_split[col].astype(str),
                val_split[col].astype(str),
                test_df[col].astype(str),
            ]
        ).unique()
        cat_encoder.fit(all_vals)

        train_split[f"{col}_cat"] = cat_encoder.transform(train_split[col].astype(str))
        val_split[f"{col}_cat"] = cat_encoder.transform(val_split[col].astype(str))
        test_df[f"{col}_cat"] = cat_encoder.transform(test_df[col].astype(str))

    # Assemble Feature Matrices
    cols_to_drop = (
        [target_col, "target"]
        + cat_cols
        + [c for c in train_split.columns if train_split[c].dtype == "datetime64[ns]"]
    )
    if narrative_col and narrative_col in train_split.columns:
        cols_to_drop.append(narrative_col)

    feature_cols = [c for c in train_split.columns if c not in cols_to_drop]

    X_train_df = train_split[feature_cols].fillna(-999)
    X_val_df = val_split[feature_cols].fillna(-999)
    X_test_df = test_df[feature_cols].fillna(-999)

    input_dim = X_train_df.shape[1]
    print(f"Features: {input_dim}, Classes: {num_classes}")

    # Scale features for PyTorch GatedResMLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_val_scaled = scaler.transform(X_val_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train GatedResMLP
    print("--- Training Model 1: GatedResMLP ---")
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model_mlp = GatedResMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.25,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model_mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_mlp_path = "./working/best_gated_res_mlp.pt"

    for epoch in range(1, epochs + 1):
        model_mlp.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model_mlp(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)

        scheduler.step()
        train_loss /= len(train_dataset)

        model_mlp.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model_mlp(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_mlp.state_dict(), best_mlp_path)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    model_mlp.load_state_dict(torch.load(best_mlp_path))
    model_mlp.eval()

    def predict_mlp_probs(X_scaled_np):
        loader = DataLoader(
            TensorDataset(torch.tensor(X_scaled_np, dtype=torch.float32)),
            batch_size=512,
            shuffle=False,
            num_workers=2,
        )
        probs_list = []
        with torch.no_grad():
            for (x_b,) in loader:
                x_b = x_b.to(device)
                logits = model_mlp(x_b)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        return np.vstack(probs_list)

    p_mlp_val = predict_mlp_probs(X_val_scaled)
    p_mlp_test = predict_mlp_probs(X_test_scaled)

    # 2. Train LightGBM
    print("--- Training Model 2: LightGBM ---")
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_val_df, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    p_lgb_val = lgb_model.predict_proba(X_val_df)
    p_lgb_test = lgb_model.predict_proba(X_test_df)

    # 3. Train XGBoost
    print("--- Training Model 3: XGBoost ---")
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)

    p_xgb_val = xgb_model.predict_proba(X_val_df)
    p_xgb_test = xgb_model.predict_proba(X_test_df)

    # 4. Ensemble Weight Optimization
    print("--- Optimizing Ensemble Weights ---")
    best_acc = -1.0
    best_weights = (0.33, 0.33, 0.34)

    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1, 21):
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-5:
                w3 = max(0.0, w3)
                total = w1 + w2 + w3
                if total > 0:
                    w1_n, w2_n, w3_n = w1 / total, w2 / total, w3 / total
                    p_blend = w1_n * p_mlp_val + w2_n * p_lgb_val + w3_n * p_xgb_val
                    acc = accuracy_score(y_val, np.argmax(p_blend, axis=1))
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = (w1_n, w2_n, w3_n)

    w1, w2, w3 = best_weights
    print(f"Optimal Ensemble Weights (MLP, LGB, XGB): ({w1:.2f}, {w2:.2f}, {w3:.2f})")

    # 5. Generate Submission
    p_test_blend = w1 * p_mlp_test + w2 * p_lgb_test + w3 * p_xgb_test
    test_preds = np.argmax(p_test_blend, axis=1)
    test_labels = label_encoder.inverse_transform(test_preds)

    sub_df = pd.DataFrame(
        {
            "id": np.arange(len(test_labels)),
            "Company response to consumer": test_labels,
        }
    )
    sub_df.to_csv("./submission/submission.csv", index=False)
    print("Submission generated at ./submission/submission.csv")

    print(f"Final Validation Score: {best_acc}")


if __name__ == "__main__":
    main()
