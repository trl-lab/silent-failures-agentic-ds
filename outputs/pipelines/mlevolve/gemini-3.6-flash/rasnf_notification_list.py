import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Reproducibility & Output Setup
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Data Loading
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "num_countries_affected"
id_col = "id" if "id" in test_df.columns else None

def parse_target(series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() == 2:
            vals = sorted(series.unique())
            return (series == vals[1]).astype(int).values
        else:
            return (series > 1).astype(int).values

    try:
        num_s = pd.to_numeric(series)
        if num_s.nunique() == 2:
            vals = sorted(num_s.unique())
            return (num_s == vals[1]).astype(int).values
        else:
            return (num_s > 1).astype(int).values
    except Exception:
        pass

    s_str = series.astype(str).str.lower().str.strip()
    unique_vals = s_str.unique()

    if len(unique_vals) == 2:
        val0, val1 = unique_vals[0], unique_vals[1]
        is_val0_one = any(k in val0 for k in ["one", "single", "1"]) and not any(
            k in val0 for k in ["more", "multi", ">", "plus"]
        )
        is_val1_one = any(k in val1 for k in ["one", "single", "1"]) and not any(
            k in val1 for k in ["more", "multi", ">", "plus"]
        )

        if is_val0_one and not is_val1_one:
            return (s_str == val1).astype(int).values
        elif is_val1_one and not is_val0_one:
            return (s_str == val0).astype(int).values
        else:
            codes, _ = pd.factorize(s_str)
            return codes
    else:
        is_one = s_str.str.contains("one|single|^1$", regex=True) & ~s_str.str.contains(
            "more|multi|>"
        )
        return (~is_one).astype(int).values


y = parse_target(train_df[target_col])
train_df["target"] = y

if id_col is None:
    test_ids = np.arange(len(test_df))
else:
    test_ids = test_df[id_col].values

# 3. Identify Feature Columns
drop_cols = [target_col, "target", id_col] if id_col else [target_col, "target"]
candidate_cols = [c for c in train_df.columns if c not in drop_cols]

datetime_cols = []
cat_cols = []
text_cols = []
num_cols = []

for c in candidate_cols:
    if (
        pd.api.types.is_datetime64_any_dtype(train_df[c])
        or "date" in c.lower()
        or "time" in c.lower()
    ):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train_df[c]):
        if train_df[c].nunique() < 10:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        avg_len = (
            train_df[c].dropna().astype(str).str.len().mean()
            if len(train_df[c].dropna()) > 0
            else 0
        )
        if avg_len > 35 or train_df[c].nunique() > 1000:
            text_cols.append(c)
        else:
            cat_cols.append(c)


# 4. Feature Engineering
def process_data(train_data, test_data, y):
    X_tr = pd.DataFrame(index=train_data.index)
    X_te = pd.DataFrame(index=test_data.index)

    # Missingness Indicator Flags & Row-level Missingness Count
    sparse_cols = [c for c in candidate_cols if train_data[c].isna().any() or test_data[c].isna().any()]
    for c in sparse_cols:
        X_tr[f"{c}_isna"] = train_data[c].isna().astype(float)
        X_te[f"{c}_isna"] = test_data[c].isna().astype(float)

    X_tr["row_isna_count"] = train_data[candidate_cols].isna().sum(axis=1).astype(float)
    X_te["row_isna_count"] = test_data[candidate_cols].isna().sum(axis=1).astype(float)

    # A. Datetime Features
    for c in datetime_cols:
        for df, X in [(train_data, X_tr), (test_data, X_te)]:
            dt_series = pd.to_datetime(df[c], errors="coerce")
            X[f"{c}_year"] = dt_series.dt.year.fillna(-1)
            X[f"{c}_month"] = dt_series.dt.month.fillna(-1)
            X[f"{c}_day"] = dt_series.dt.day.fillna(-1)
            X[f"{c}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1)
            X[f"{c}_sin_month"] = np.sin(2 * np.pi * dt_series.dt.month / 12.0).fillna(
                0
            )
            X[f"{c}_cos_month"] = np.cos(2 * np.pi * dt_series.dt.month / 12.0).fillna(
                0
            )

    # B. Categorical Features & Frequency Encoding
    for c in cat_cols:
        train_str = train_data[c].fillna("missing").astype(str)
        test_str = test_data[c].fillna("missing").astype(str)

        freq = pd.concat([train_str, test_str]).value_counts()
        X_tr[f"{c}_freq"] = train_str.map(freq).astype(float)
        X_te[f"{c}_freq"] = test_str.map(freq).astype(float)

        cat_type = pd.CategoricalDtype(categories=freq.index, ordered=False)
        X_tr[c] = train_str.astype(cat_type).cat.codes
        X_te[c] = test_str.astype(cat_type).cat.codes

    # C. Pairwise Categorical Interactions
    if len(cat_cols) >= 2:
        top_cats = cat_cols[: min(4, len(cat_cols))]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                c1, c2 = top_cats[i], top_cats[j]
                inter_train = (
                    train_data[c1].fillna("").astype(str)
                    + "_"
                    + train_data[c2].fillna("").astype(str)
                )
                inter_test = (
                    test_data[c1].fillna("").astype(str)
                    + "_"
                    + test_data[c2].fillna("").astype(str)
                )

                freq_inter = pd.concat([inter_train, inter_test]).value_counts()
                X_tr[f"{c1}_{c2}_freq"] = inter_train.map(freq_inter).astype(float)
                X_te[f"{c1}_{c2}_freq"] = inter_test.map(freq_inter).astype(float)

    # D. Numerical Features & Group-by Aggregations
    for c in num_cols:
        X_tr[c] = train_data[c].fillna(train_data[c].median()).astype(float)
        X_te[c] = test_data[c].fillna(train_data[c].median()).astype(float)

    if len(cat_cols) > 0 and len(num_cols) > 0:
        target_cats = cat_cols[: min(4, len(cat_cols))]
        target_nums = num_cols[: min(5, len(num_cols))]
        for cat_c in target_cats:
            for num_c in target_nums:
                grouped = train_data.groupby(cat_c)[num_c]
                for agg in ["mean", "std", "min", "max"]:
                    agg_map = grouped.agg(agg)
                    X_tr[f"{cat_c}_{num_c}_{agg}"] = train_data[cat_c].map(agg_map).astype(float)
                    X_te[f"{cat_c}_{num_c}_{agg}"] = test_data[cat_c].map(agg_map).astype(float)

    # E. Text Features (Per-Column TF-IDF + 32 SVD + Stacked OOF Ridge + Length & Word Count)
    for col in text_cols:
        col_train_text = train_data[col].fillna("").astype(str)
        col_test_text = test_data[col].fillna("").astype(str)

        tfidf = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        svd = TruncatedSVD(n_components=32, random_state=42)

        train_tfidf = tfidf.fit_transform(col_train_text)
        test_tfidf = tfidf.transform(col_test_text)

        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)

        for k in range(32):
            X_tr[f"{col}_svd_{k}"] = train_svd[:, k]
            X_te[f"{col}_svd_{k}"] = test_svd[:, k]

        # Stacked Out-Of-Fold RidgeClassifier on Word and Char n-grams
        word_vec = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        char_vec = TfidfVectorizer(
            max_features=10000, analyzer="char", ngram_range=(2, 4)
        )

        X_word_tr = word_vec.fit_transform(col_train_text)
        X_word_te = word_vec.transform(col_test_text)
        X_char_tr = char_vec.fit_transform(col_train_text)
        X_char_te = char_vec.transform(col_test_text)

        X_tfidf_tr = hstack([X_word_tr, X_char_tr]).tocsr()
        X_tfidf_te = hstack([X_word_te, X_char_te]).tocsr()

        skf_text = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_ridge = np.zeros(len(train_data))
        test_ridge = np.zeros(len(test_data))

        for tr_idx, val_idx in skf_text.split(X_tfidf_tr, y):
            ridge = RidgeClassifier(alpha=1.0, random_state=42)
            ridge.fit(X_tfidf_tr[tr_idx], y[tr_idx])
            oof_ridge[val_idx] = ridge.decision_function(X_tfidf_tr[val_idx])
            test_ridge += ridge.decision_function(X_tfidf_te) / 5.0

        X_tr[f"{col}_ridge_oof"] = oof_ridge
        X_te[f"{col}_ridge_oof"] = test_ridge

        X_tr[f"{col}_len"] = col_train_text.str.len().astype(float)
        X_te[f"{col}_len"] = col_test_text.str.len().astype(float)
        X_tr[f"{col}_word_count"] = (
            col_train_text.str.split().str.len().astype(float)
        )
        X_te[f"{col}_word_count"] = (
            col_test_text.str.split().str.len().astype(float)
        )

    return X_tr, X_te


X_tr, X_te = process_data(train_df, test_df, y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_enc_cols = cat_cols

X_tr = X_tr.astype(np.float32)
X_te = X_te.astype(np.float32)


# 5. Model Architecture Definitions
class TabularResBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        return out + res


class DeepTabularNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                TabularResBlock(
                    hidden_dims[i], hidden_dims[i + 1], dropout_rate=dropout_rate
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        logits = self.head(x)
        return logits


def build_pytorch_criterion():
    return nn.BCEWithLogitsLoss()


def build_pytorch_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_pytorch_scheduler(optimizer, T_max: int = 50, eta_min: float = 1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )


def get_lgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_samples=20,
        reg_alpha=1.0,
        reg_lambda=3.0,
        n_estimators=1000,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_alpha=1.0,
        reg_lambda=3.0,
        n_estimators=1000,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )


def get_catboost_model(random_state: int = 42) -> cb.CatBoostClassifier:
    return cb.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        rsm=0.6,
        iterations=1000,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def get_hist_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_leaf_nodes=31,
        max_depth=6,
        l2_regularization=3.0,
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=50,
        scoring="roc_auc",
    )


# 6. PyTorch Training Procedure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pytorch_model(
    X_train, y_train, X_val, y_val, X_test, epochs=30, batch_size=256
):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model = DeepTabularNet(
        input_dim=X_train_s.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
    ).to(device)

    criterion = build_pytorch_criterion()
    optimizer = build_pytorch_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = build_pytorch_scheduler(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                val_preds.extend(probs)

        val_auc = roc_auc_score(y_val, val_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    val_preds = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            probs = torch.sigmoid(model(bx)).cpu().numpy().ravel()
            val_preds.extend(probs)

    test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_preds = torch.sigmoid(model(test_tensor)).cpu().numpy().ravel()

    return np.array(val_preds), test_preds


# 7. Stratified 5-Fold Training & Ensemble Blending
oof_lgb = np.zeros(len(X_tr))
oof_xgb = np.zeros(len(X_tr))
oof_cat = np.zeros(len(X_tr))
oof_hist = np.zeros(len(X_tr))
oof_nn = np.zeros(len(X_tr))

test_lgb = np.zeros(len(X_te))
test_xgb = np.zeros(len(X_te))
test_cat = np.zeros(len(X_te))
test_hist = np.zeros(len(X_te))
test_nn = np.zeros(len(X_te))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_train_f, y_train_f = X_tr.iloc[train_idx].copy(), y[train_idx]
    X_val_f, y_val_f = X_tr.iloc[val_idx].copy(), y[val_idx]
    X_te_f = X_te.copy()

    # Target Encoding computed strictly on training fold
    smoothing = 10.0
    global_mean = y_train_f.mean()
    for c in target_enc_cols:
        stats = pd.DataFrame({"cat": X_train_f[c], "target": y_train_f}).groupby("cat")["target"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        X_train_f[f"{c}_te"] = X_train_f[c].map(smooth).fillna(global_mean)
        X_val_f[f"{c}_te"] = X_val_f[c].map(smooth).fillna(global_mean)
        X_te_f[f"{c}_te"] = X_te_f[c].map(smooth).fillna(global_mean)

    feature_medians = X_train_f.median()
    X_train_f.fillna(feature_medians, inplace=True)
    X_val_f.fillna(feature_medians, inplace=True)
    X_te_f.fillna(feature_medians, inplace=True)

    # LightGBM
    lgb_model = get_lgbm_model(random_state=42 + fold)
    lgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val_f)[:, 1]
    test_lgb += lgb_model.predict_proba(X_te_f)[:, 1] / skf.n_splits

    # XGBoost
    xgb_model = get_xgb_model(random_state=42 + fold)
    xgb_model.set_params(early_stopping_rounds=50)
    xgb_model.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val_f)[:, 1]
    test_xgb += xgb_model.predict_proba(X_te_f)[:, 1] / skf.n_splits

    # CatBoost
    cat_model = get_catboost_model(random_state=42 + fold)
    cat_model.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_val_f)[:, 1]
    test_cat += cat_model.predict_proba(X_te_f)[:, 1] / skf.n_splits

    # HistGradientBoosting
    hist_model = get_hist_model(random_state=42 + fold)
    hist_model.fit(X_train_f, y_train_f)
    oof_hist[val_idx] = hist_model.predict_proba(X_val_f)[:, 1]
    test_hist += hist_model.predict_proba(X_te_f)[:, 1] / skf.n_splits

    # PyTorch Neural Net
    nn_val_pred, nn_test_pred = train_pytorch_model(
        X_train_f.values,
        y_train_f,
        X_val_f.values,
        y_val_f,
        X_te_f.values,
        epochs=30,
        batch_size=256,
    )
    oof_nn[val_idx] = nn_val_pred
    test_nn += nn_test_pred / skf.n_splits

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hist, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_hist, test_nn])

def loss_func(w):
    w_norm = w / np.sum(w)
    pred = np.dot(oof_matrix, w_norm)
    return -roc_auc_score(y, pred)

init_weights = np.ones(5) / 5.0
bounds = [(0, 1)] * 5
constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = res.x / np.sum(res.x)

oof_blend = np.dot(oof_matrix, opt_weights)
test_blend = np.dot(test_matrix, opt_weights)

final_cv_score = roc_auc_score(y, oof_blend)

submission = pd.DataFrame({"id": test_ids, "num_countries_affected": test_blend})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_cv_score}")