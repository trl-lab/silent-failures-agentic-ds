import os
import random
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# 1. Global Setup & Reproducibility
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Data Loading
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

id_col = None
for col in ["id", "ID", "Id", "index"]:
    if col in train_raw.columns and col in test_raw.columns:
        id_col = col
        break

if id_col is None:
    train_ids = np.arange(len(train_raw))
    test_ids = np.arange(len(test_raw))
else:
    train_ids = train_raw[id_col].values
    test_ids = test_raw[id_col].values
    train_raw = train_raw.drop(columns=[id_col])
    test_raw = test_raw.drop(columns=[id_col])

target_col = "Status"
assert (
    target_col in train_raw.columns
), f"Target column '{target_col}' missing in train dataset."

y_raw = train_raw[target_col].copy()
if y_raw.dtype == "object" or isinstance(y_raw.dtype, pd.CategoricalDtype):
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
else:
    y = y_raw.values.astype(int)

train_df = train_raw.drop(columns=[target_col])
test_df = test_raw.copy()


# 3. Feature Engineering Pipeline
def engineer_features(tr_data, te_data, y_train=None):
    tr = tr_data.copy()
    te = te_data.copy()
    n_train = len(tr)
    combined = pd.concat([tr, te], axis=0, ignore_index=True)

    feature_dfs_tr = []
    feature_dfs_te = []

    missing_df_tr = pd.DataFrame({"missing_count": tr.isnull().sum(axis=1)})
    missing_df_te = pd.DataFrame({"missing_count": te.isnull().sum(axis=1)})
    feature_dfs_tr.append(missing_df_tr.reset_index(drop=True))
    feature_dfs_te.append(missing_df_te.reset_index(drop=True))

    for col in tr.columns:
        is_date = False
        if pd.api.types.is_datetime64_any_dtype(combined[col]):
            is_date = True
            dt_series = combined[col]
        else:
            if combined[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(combined[col], errors="coerce")
                    if parsed.notnull().mean() > 0.7:
                        is_date = True
                        dt_series = parsed
                except Exception:
                    pass

        if is_date:
            dt_df = pd.DataFrame()
            dt_df[f"{col}_year"] = dt_series.dt.year
            dt_df[f"{col}_month"] = dt_series.dt.month
            dt_df[f"{col}_day"] = dt_series.dt.day
            dt_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
            dt_df[f"{col}_quarter"] = dt_series.dt.quarter
            min_date = dt_series.min()
            dt_df[f"{col}_days_since_start"] = (dt_series - min_date).dt.days

            feature_dfs_tr.append(dt_df.iloc[:n_train].reset_index(drop=True))
            feature_dfs_te.append(dt_df.iloc[n_train:].reset_index(drop=True))
            continue

        if combined[col].dtype == "object" or isinstance(
            combined[col].dtype, pd.CategoricalDtype
        ):
            str_series = combined[col].fillna("").astype(str)
            avg_word_count = str_series.apply(lambda x: len(x.split())).mean()

            if avg_word_count >= 3:
                str_tr = tr[col].fillna("").astype(str)
                str_te = te[col].fillna("").astype(str)

                txt_df_tr = pd.DataFrame()
                txt_df_te = pd.DataFrame()

                txt_df_tr[f"{col}_char_len"] = str_tr.apply(len)
                txt_df_tr[f"{col}_word_count"] = str_tr.apply(lambda x: len(x.split()))
                txt_df_tr[f"{col}_digit_count"] = str_tr.apply(lambda x: sum(c.isdigit() for c in x))
                txt_df_tr[f"{col}_upper_count"] = str_tr.apply(lambda x: sum(c.isupper() for c in x))
                txt_df_tr[f"{col}_avg_word_len"] = txt_df_tr[f"{col}_char_len"] / (txt_df_tr[f"{col}_word_count"] + 1e-5)

                txt_df_te[f"{col}_char_len"] = str_te.apply(len)
                txt_df_te[f"{col}_word_count"] = str_te.apply(lambda x: len(x.split()))
                txt_df_te[f"{col}_digit_count"] = str_te.apply(lambda x: sum(c.isdigit() for c in x))
                txt_df_te[f"{col}_upper_count"] = str_te.apply(lambda x: sum(c.isupper() for c in x))
                txt_df_te[f"{col}_avg_word_len"] = txt_df_te[f"{col}_char_len"] / (txt_df_te[f"{col}_word_count"] + 1e-5)

                tfidf = TfidfVectorizer(
                    ngram_range=(1, 2), max_features=5000, stop_words="english"
                )
                tfidf_tr = tfidf.fit_transform(str_tr)
                tfidf_te = tfidf.transform(str_te)

                if y_train is not None:
                    oof_ridge = np.zeros(len(str_tr))
                    skf_txt = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                    for tr_i, va_i in skf_txt.split(tfidf_tr, y_train):
                        clf = RidgeClassifier(alpha=1.0, random_state=SEED)
                        clf.fit(tfidf_tr[tr_i], y_train[tr_i])
                        oof_ridge[va_i] = clf.decision_function(tfidf_tr[va_i])

                    clf_full = RidgeClassifier(alpha=1.0, random_state=SEED)
                    clf_full.fit(tfidf_tr, y_train)
                    test_ridge = clf_full.decision_function(tfidf_te)

                    txt_df_tr[f"{col}_ridge_oof"] = oof_ridge
                    txt_df_te[f"{col}_ridge_oof"] = test_ridge

                n_comp = (
                    min(30, tfidf_tr.shape[1] - 1) if tfidf_tr.shape[1] > 1 else 1
                )
                if n_comp > 0:
                    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
                    svd_tr = svd.fit_transform(tfidf_tr)
                    svd_te = svd.transform(tfidf_te)
                    for i in range(n_comp):
                        txt_df_tr[f"{col}_svd_{i}"] = svd_tr[:, i]
                        txt_df_te[f"{col}_svd_{i}"] = svd_te[:, i]

                feature_dfs_tr.append(txt_df_tr.reset_index(drop=True))
                feature_dfs_te.append(txt_df_te.reset_index(drop=True))
                continue
            else:
                cat_df = pd.DataFrame()
                freq_map = tr[col].value_counts(normalize=True).to_dict()
                cat_df[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

                le = LabelEncoder()
                cat_df[f"{col}_encoded"] = le.fit_transform(
                    combined[col].fillna("Missing").astype(str)
                )

                feature_dfs_tr.append(cat_df.iloc[:n_train].reset_index(drop=True))
                feature_dfs_te.append(cat_df.iloc[n_train:].reset_index(drop=True))
                continue

        if pd.api.types.is_numeric_dtype(combined[col]):
            num_df = pd.DataFrame()
            num_series = combined[col]
            tr_median = tr[col].median()
            median_val = tr_median if not pd.isna(tr_median) else 0
            num_df[col] = num_series.fillna(median_val)

            skew = num_series.skew()
            if abs(skew) > 1.0 and (num_df[col] >= 0).all():
                num_df[f"{col}_log1p"] = np.log1p(num_df[col])

            feature_dfs_tr.append(num_df.iloc[:n_train].reset_index(drop=True))
            feature_dfs_te.append(num_df.iloc[n_train:].reset_index(drop=True))

    X_tr = pd.concat(feature_dfs_tr, axis=1)
    X_te = pd.concat(feature_dfs_te, axis=1)

    X_tr.columns = [f"feat_{i}_{c}" for i, c in enumerate(X_tr.columns)]
    X_te.columns = [f"feat_{i}_{c}" for i, c in enumerate(X_te.columns)]

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_tr, X_te


X_train, X_test = engineer_features(train_df, test_df, y)

# Out-Of-Fold Target Encoding with M-Estimate Smoothing (m=10)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cat_cols = [
    c
    for c in train_df.columns
    if train_df[c].dtype == "object"
    or isinstance(train_df[c].dtype, pd.CategoricalDtype)
]

m_param = 10.0

for c in cat_cols:
    te_col_name = f"te_{c}"
    X_train[te_col_name] = 0.0
    X_test[te_col_name] = 0.0

    global_mean = y.mean()
    cats_tr = train_df[c].fillna("Missing")
    cat_counts = cats_tr.value_counts()
    cat_sums = pd.Series(y).groupby(cats_tr).sum()
    cat_m_estimate = ((cat_sums + m_param * global_mean) / (cat_counts + m_param)).to_dict()
    X_test[te_col_name] = (
        test_df[c].fillna("Missing").map(cat_m_estimate).fillna(global_mean)
    )

    for tr_idx, va_idx in skf.split(X_train, y):
        tr_cat = train_df[c].iloc[tr_idx].fillna("Missing")
        tr_y = y[tr_idx]
        va_cat = train_df[c].iloc[va_idx].fillna("Missing")

        fold_global_mean = tr_y.mean()
        fold_counts = tr_cat.value_counts()
        fold_sums = pd.Series(tr_y, index=tr_cat.index).groupby(tr_cat).sum()
        fold_m_estimate = ((fold_sums + m_param * fold_global_mean) / (fold_counts + m_param)).to_dict()

        X_train.iloc[va_idx, X_train.columns.get_loc(te_col_name)] = (
            va_cat.map(fold_m_estimate).fillna(fold_global_mean).values
        )

X = X_train.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)


# 4. Neural Network Architecture Definition
class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = h + self.res_block1(h)
        h = h + self.res_block2(h)
        return self.out_head(h)


def train_pytorch_fold(X_tr_scaled, y_tr, X_va_scaled, y_va, X_te_scaled, device, fold):
    input_dim = X_tr_scaled.shape[1]
    model = TabularResNet(input_dim=input_dim, hidden_dim=256, dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=35, eta_min=1e-5
    )

    tr_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
    )

    tr_loader = DataLoader(
        tr_ds, batch_size=64, shuffle=True, drop_last=False, num_workers=0
    )
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)

    best_val_auc = 0.0
    best_va_preds = None
    best_weights = None

    for epoch in range(35):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        va_preds_list = []
        with torch.no_grad():
            for bx, _ in va_loader:
                bx = bx.to(device)
                preds = torch.sigmoid(model(bx)).cpu().numpy()
                va_preds_list.append(preds)

        va_preds = np.vstack(va_preds_list).ravel()
        val_auc = roc_auc_score(y_va, va_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_va_preds = va_preds
            import copy
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"./working/nn_fold_{fold}.pth")

    model.eval()
    with torch.no_grad():
        te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        te_preds = torch.sigmoid(model(te_tensor)).cpu().numpy().ravel()

    return best_va_preds, te_preds


# 5. Ensemble Training Setup across 5 Folds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_nn = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test_arr))
test_preds_xgb = np.zeros(len(X_test_arr))
test_preds_cat = np.zeros(len(X_test_arr))
test_preds_nn = np.zeros(len(X_test_arr))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test_arr)

    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=SEED + fold,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    model_lgb.booster_.save_model(f"./working/lgb_fold_{fold}.txt")
    oof_lgb[va_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test_arr)[:, 1] / skf.n_splits

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=SEED + fold,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    model_xgb.save_model(f"./working/xgb_fold_{fold}.json")
    oof_xgb[va_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test_arr)[:, 1] / skf.n_splits

    # CatBoost
    model_cat = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        random_seed=SEED + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    model_cat.save_model(f"./working/cat_fold_{fold}.cbm")
    oof_cat[va_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test_arr)[:, 1] / skf.n_splits

    # PyTorch Tabular ResNet
    va_nn_p, te_nn_p = train_pytorch_fold(
        X_tr_scaled, y_tr, X_va_scaled, y_va, X_te_scaled, device, fold
    )
    oof_nn[va_idx] = va_nn_p
    test_preds_nn += te_nn_p / skf.n_splits

    print(
        f"Fold {fold+1} ROC-AUC | LGB: {roc_auc_score(y_va, oof_lgb[va_idx]):.4f} | XGB: {roc_auc_score(y_va, oof_xgb[va_idx]):.4f} | CAT: {roc_auc_score(y_va, oof_cat[va_idx]):.4f} | NN: {roc_auc_score(y_va, oof_nn[va_idx]):.4f}"
    )

# 6. Rank Blending Optimization
oof_ranks = np.vstack(
    [
        rankdata(oof_lgb) / len(oof_lgb),
        rankdata(oof_xgb) / len(oof_xgb),
        rankdata(oof_cat) / len(oof_cat),
        rankdata(oof_nn) / len(oof_nn),
    ]
)

test_ranks = np.vstack(
    [
        rankdata(test_preds_lgb) / len(test_preds_lgb),
        rankdata(test_preds_xgb) / len(test_preds_xgb),
        rankdata(test_preds_cat) / len(test_preds_cat),
        rankdata(test_preds_nn) / len(test_preds_nn),
    ]
)


def loss_func(weights):
    w = np.maximum(0, np.array(weights))
    if w.sum() == 0:
        return 0.0
    w /= w.sum()
    blend = np.dot(w, oof_ranks)
    return -roc_auc_score(y, blend)


init_weights = [0.35, 0.25, 0.25, 0.15]
res = minimize(loss_func, init_weights, method="Nelder-Mead")

best_weights = np.maximum(0, res.x)
best_weights /= best_weights.sum()

oof_final_blend = np.dot(best_weights, oof_ranks)
cv_score = roc_auc_score(y, oof_final_blend)

# 7. Generate Submission Output
test_final_blend = np.dot(best_weights, test_ranks)

submission = pd.DataFrame({"id": test_ids, "Status": test_final_blend})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {cv_score}")