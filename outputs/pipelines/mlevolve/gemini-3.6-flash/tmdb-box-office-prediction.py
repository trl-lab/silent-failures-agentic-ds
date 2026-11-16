import ast
import json
import os
import re

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import xgboost as xgb

# ---------------------------------------------------------
# Directories Setup
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

test_ids = test_df["id"].values
y_train = np.log1p(np.maximum(0, train_df["revenue"].values))


# ---------------------------------------------------------
# 2. Preprocessing & Feature Extraction Utilities
# ---------------------------------------------------------
def safe_parse_json(val):
    if pd.isna(val) or val is None or val == "":
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return []


def get_names_list(json_data):
    parsed = safe_parse_json(json_data)
    if isinstance(parsed, list):
        return [
            item["name"] for item in parsed if isinstance(item, dict) and "name" in item
        ]
    return []


def get_crew_job_count(json_data, target_jobs):
    parsed = safe_parse_json(json_data)
    if isinstance(parsed, list):
        return sum(
            1
            for item in parsed
            if isinstance(item, dict) and item.get("job") in target_jobs
        )
    return 0


def extract_collection_name(val):
    parsed = safe_parse_json(val)
    if isinstance(parsed, dict) and "name" in parsed:
        return parsed["name"]
    elif (
        isinstance(parsed, list)
        and len(parsed) > 0
        and isinstance(parsed[0], dict)
        and "name" in parsed[0]
    ):
        return parsed[0]["name"]
    return None


def extract_primary_company(val):
    names = get_names_list(val)
    return names[0] if len(names) > 0 else None


def extract_director(val):
    parsed = safe_parse_json(val)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("job") == "Director":
                return item.get("name")
    return None


# Global mappings for frequency features
full_raw_df = pd.concat([train_df, test_df], ignore_index=True)

train_df["collection_name"] = train_df["belongs_to_collection"].apply(extract_collection_name)
test_df["collection_name"] = test_df["belongs_to_collection"].apply(extract_collection_name)
full_collections = pd.concat([train_df["collection_name"], test_df["collection_name"]])
collection_freq_map = full_collections.value_counts().to_dict()

train_df["primary_company"] = train_df["production_companies"].apply(extract_primary_company)
test_df["primary_company"] = test_df["production_companies"].apply(extract_primary_company)

train_df["director_name"] = train_df["crew"].apply(extract_director)
test_df["director_name"] = test_df["crew"].apply(extract_director)
full_directors = pd.concat([train_df["director_name"], test_df["director_name"]])
director_freq_map = full_directors.value_counts().to_dict()

def extract_top_actor(val):
    names = get_names_list(val)
    return names[0] if len(names) > 0 else None


def extract_top_keyword(val):
    names = get_names_list(val)
    return names[0] if len(names) > 0 else None


train_df["top_actor"] = train_df["cast"].apply(extract_top_actor)
test_df["top_actor"] = test_df["cast"].apply(extract_top_actor)

train_df["top_keyword"] = train_df["Keywords"].apply(extract_top_keyword)
test_df["top_keyword"] = test_df["Keywords"].apply(extract_top_keyword)

def parse_dates_raw(dates_series):
    def fix_year(dt_str):
        if pd.isna(dt_str) or str(dt_str).strip() == "":
            return np.nan
        parts = str(dt_str).strip().split("/")
        if len(parts) == 3:
            m, d, y = parts
            if len(y) == 2:
                year = int(y)
                year = year + 1900 if year > 20 else year + 2000
                return f"{m}/{d}/{year}"
        return dt_str

    fixed = dates_series.apply(fix_year)
    return pd.to_datetime(fixed, format="mixed", errors="coerce")


full_dates = parse_dates_raw(full_raw_df["release_date"])
full_years = full_dates.dt.year.fillna(2005)
full_decades = (full_years // 10) * 10
full_budgets = full_raw_df["budget"].fillna(0).astype(float)

nonzero_mask = full_budgets > 0
overall_nonzero_budget_median = (
    full_budgets[nonzero_mask].median() if nonzero_mask.any() else 1000000.0
)
budget_df = pd.DataFrame({"decade": full_decades, "budget": full_budgets})
decade_budget_medians = (
    budget_df[budget_df["budget"] > 0]
    .groupby("decade")["budget"]
    .median()
    .to_dict()
)

full_cast = pd.concat([train_df["cast"], test_df["cast"]]).apply(lambda x: get_names_list(x)[:5])
actor_counts = {}
for cast_list in full_cast:
    for actor in cast_list:
        actor_counts[actor] = actor_counts.get(actor, 0) + 1

all_genres = sorted(list(set(g for genres in full_raw_df["genres"].apply(get_names_list) for g in genres)))


def parse_dates(df):
    def fix_year(dt_str):
        if pd.isna(dt_str) or str(dt_str).strip() == "":
            return np.nan
        parts = str(dt_str).strip().split("/")
        if len(parts) == 3:
            m, d, y = parts
            if len(y) == 2:
                year = int(y)
                year = year + 1900 if year > 20 else year + 2000
                return f"{m}/{d}/{year}"
        return dt_str

    dates = df["release_date"].apply(fix_year)
    return pd.to_datetime(dates, format="mixed", errors="coerce")


def extract_tabular_features(df):
    feats = pd.DataFrame(index=df.index)

    # Budget with Decade Group-Median Imputation and Inflation Adjustment
    dt_temp = parse_dates(df)
    rel_years = dt_temp.dt.year.fillna(2005)
    rel_decades = (rel_years // 10) * 10

    raw_budget = df["budget"].fillna(0).astype(float).values
    popularity = df["popularity"].fillna(0).astype(float).values
    runtime = df["runtime"].fillna(df["runtime"].median()).astype(float).values

    base_median_budget = decade_budget_medians.get(2010, overall_nonzero_budget_median)

    imputed_budget = np.zeros(len(df))
    adjusted_budget = np.zeros(len(df))

    for idx in range(len(df)):
        b = raw_budget[idx]
        dec = rel_decades.iloc[idx]
        dec_med = decade_budget_medians.get(dec, overall_nonzero_budget_median)
        if b <= 0:
            b_imp = dec_med
        else:
            b_imp = b

        # Inflation factor relative to 2010s
        inflation_factor = base_median_budget / (dec_med if dec_med > 0 else base_median_budget)
        imputed_budget[idx] = b_imp
        adjusted_budget[idx] = b_imp * inflation_factor

    feats["budget_raw"] = raw_budget
    feats["budget_is_zero"] = (raw_budget == 0).astype(int)
    feats["budget_imputed"] = imputed_budget
    feats["budget_adjusted"] = adjusted_budget
    feats["log1p_budget"] = np.log1p(np.maximum(0, adjusted_budget))

    feats["popularity_raw"] = popularity
    feats["log1p_popularity"] = np.log1p(np.maximum(0, popularity))

    feats["runtime"] = runtime
    feats["runtime_is_na"] = df["runtime"].isna().astype(int)
    feats["log1p_runtime"] = np.log1p(np.maximum(0, runtime))

    # Domain Ratios and Interactions
    feats["budget_to_pop_ratio"] = feats["log1p_budget"] / (
        feats["log1p_popularity"] + 1e-5
    )
    feats["budget_per_runtime"] = feats["log1p_budget"] / (runtime + 1.0)
    feats["pop_per_runtime"] = feats["log1p_popularity"] / (runtime + 1.0)

    # Indicator / Metadata Presence Flags
    feats["has_homepage"] = df["homepage"].notna().astype(int)
    feats["has_tagline"] = df["tagline"].notna().astype(int)
    feats["has_overview"] = df["overview"].notna().astype(int)
    feats["has_poster"] = df["poster_path"].notna().astype(int)
    feats["has_collection"] = df["belongs_to_collection"].notna().astype(int)

    # Text Lengths
    feats["title_len"] = df["title"].fillna("").apply(len)
    feats["title_word_cnt"] = df["title"].fillna("").apply(lambda x: len(x.split()))
    feats["tagline_len"] = df["tagline"].fillna("").apply(len)
    feats["tagline_word_cnt"] = df["tagline"].fillna("").apply(lambda x: len(x.split()))
    feats["overview_len"] = df["overview"].fillna("").apply(len)
    feats["overview_word_cnt"] = (
        df["overview"].fillna("").apply(lambda x: len(x.split()))
    )

    # JSON Metadata Counts
    feats["num_genres"] = df["genres"].apply(lambda x: len(get_names_list(x)))
    feats["num_production_companies"] = df["production_companies"].apply(
        lambda x: len(get_names_list(x))
    )
    feats["num_production_countries"] = df["production_countries"].apply(
        lambda x: len(get_names_list(x))
    )
    feats["num_spoken_languages"] = df["spoken_languages"].apply(
        lambda x: len(get_names_list(x))
    )
    feats["num_keywords"] = df["Keywords"].apply(lambda x: len(get_names_list(x)))
    feats["num_cast"] = df["cast"].apply(lambda x: len(get_names_list(x)))
    feats["num_crew"] = df["crew"].apply(lambda x: len(get_names_list(x)))

    # Crew Specific Roles
    feats["num_directors"] = df["crew"].apply(
        lambda x: get_crew_job_count(x, ["Director"])
    )
    feats["num_producers"] = df["crew"].apply(
        lambda x: get_crew_job_count(x, ["Producer"])
    )
    feats["num_writers"] = df["crew"].apply(
        lambda x: get_crew_job_count(x, ["Writer", "Screenplay", "Author", "Story"])
    )
    feats["num_exec_producers"] = df["crew"].apply(
        lambda x: get_crew_job_count(x, ["Executive Producer"])
    )

    # Domain specific multi-hot genre flags & frequency scores
    for g in all_genres:
        feats[f"genre_{g}"] = df["genres"].apply(
            lambda x: 1 if g in get_names_list(x) else 0
        )

    feats["collection_freq"] = (
        df["belongs_to_collection"]
        .apply(extract_collection_name)
        .map(collection_freq_map)
        .fillna(0)
    )

    feats["director_freq"] = (
        df["crew"].apply(extract_director).map(director_freq_map).fillna(0)
    )

    cast_lists = df["cast"].apply(lambda x: get_names_list(x)[:5])
    feats["max_actor_freq"] = cast_lists.apply(
        lambda cast: max([actor_counts.get(a, 0) for a in cast]) if cast else 0
    )
    feats["mean_actor_freq"] = cast_lists.apply(
        lambda cast: (
            float(np.mean([actor_counts.get(a, 0) for a in cast])) if cast else 0.0
        )
    )

    # Top Language Flag
    lang = df["original_language"].fillna("en").astype(str)
    feats["is_english"] = (lang == "en").astype(int)

    # Temporal Features
    dt = parse_dates(df)
    median_year = dt.dt.year.median() if not dt.dt.year.isna().all() else 2005
    release_year = dt.dt.year.fillna(median_year)
    release_month = dt.dt.month.fillna(6)

    feats["release_year"] = release_year
    feats["release_month"] = release_month
    feats["release_day"] = dt.dt.day.fillna(15)
    feats["release_dayofweek"] = dt.dt.dayofweek.fillna(3)
    feats["release_quarter"] = dt.dt.quarter.fillna(2)
    feats["release_is_weekend"] = (dt.dt.dayofweek.fillna(3).isin([4, 5, 6])).astype(
        int
    )
    feats["movie_age"] = 2024.0 - release_year

    # Cyclical Month Encoding
    feats["sin_month"] = np.sin(2 * np.pi * release_month / 12.0)
    feats["cos_month"] = np.cos(2 * np.pi * release_month / 12.0)

    return feats


# Extract Tabular Datasets
X_train_tab = extract_tabular_features(train_df)
X_test_tab = extract_tabular_features(test_df)

# Align medians for missing values strictly from Training
train_medians = X_train_tab.median()
X_train_tab = X_train_tab.fillna(train_medians)
X_test_tab = X_test_tab.fillna(train_medians)


# Build Natural Text Fields for TF-IDF + TruncatedSVD
def build_combined_text(df):
    text_col = []
    for idx, row in df.iterrows():
        title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
        tagline = str(row.get("tagline", "")) if pd.notna(row.get("tagline")) else ""
        overview = str(row.get("overview", "")) if pd.notna(row.get("overview")) else ""

        genres = " ".join(get_names_list(row.get("genres")))
        keywords = " ".join(get_names_list(row.get("Keywords")))
        cast = " ".join(get_names_list(row.get("cast"))[:5])
        crew = " ".join(get_names_list(row.get("crew"))[:5])

        combined = (
            f"{title} {tagline} {overview} {genres} {keywords} {cast} {crew}".strip()
        )
        text_col.append(combined)
    return text_col


train_text = build_combined_text(train_df)
test_text = build_combined_text(test_df)

# Fit Text Vectors strictly on Training Set
tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=50, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

svd_cols = [f"svd_text_{i}" for i in range(50)]
train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=X_train_tab.index)
test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=X_test_tab.index)

def calc_oof_target_encodings(train_df, test_df, y_train, cat_cols, kf, m=5.0):
    train_te = pd.DataFrame(index=train_df.index)
    test_te = pd.DataFrame(index=test_df.index)
    global_mean = y_train.mean()

    for col in cat_cols:
        col_te_train = pd.Series(index=train_df.index, dtype=float)

        for train_idx, val_idx in kf.split(train_df, y_train):
            tr_cat = train_df[col].iloc[train_idx]
            tr_y = y_train[train_idx]

            stats = pd.DataFrame({"cat": tr_cat, "y": tr_y}).groupby("cat")["y"].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

            val_cat = train_df[col].iloc[val_idx]
            col_te_train.iloc[val_idx] = val_cat.map(smoothed).fillna(global_mean)

        train_te[f"{col}_te"] = col_te_train

        full_stats = pd.DataFrame({"cat": train_df[col], "y": y_train}).groupby("cat")["y"].agg(["count", "mean"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + m * global_mean) / (full_stats["count"] + m)
        test_te[f"{col}_te"] = test_df[col].map(full_smoothed).fillna(global_mean)

    return train_te, test_te


kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
cat_cols = ["collection_name", "primary_company", "director_name", "top_actor", "top_keyword"]
train_te, test_te = calc_oof_target_encodings(train_df, test_df, y_train, cat_cols, kf_te, m=5.0)

# Combine Tabular, Target Encoded, and Text Embedding Features
X_train = pd.concat([X_train_tab, train_te, train_svd_df], axis=1).astype(np.float64)
X_test = pd.concat([X_test_tab, test_te, test_svd_df], axis=1).astype(np.float64)

n_samples, in_features = X_train.shape
n_test = X_test.shape[0]


# ---------------------------------------------------------
# 3. Model Architecture (PyTorch TabularResNet)
# ---------------------------------------------------------
class ResBlock(nn.Module):

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=256, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res1 = ResBlock(hidden_dim, dropout_rate=dropout_rate)

        self.middle_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res2 = ResBlock(hidden_dim // 2, dropout_rate=dropout_rate)

        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.middle_layer(x)
        x = self.res2(x)
        out = self.head(x)
        return out.squeeze(-1)


# ---------------------------------------------------------
# 4. Cross-Validation & Ensemble Training Loop
# ---------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_preds_lgb += lgb_model.predict(X_test) / kf.n_splits

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        tree_method="hist",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_preds_xgb += xgb_model.predict(X_test) / kf.n_splits

    # --- CatBoost ---
    cat_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=False,
        thread_count=-1,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = cat_model.predict(X_va)
    test_preds_cat += cat_model.predict(X_test) / kf.n_splits

    # --- PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    net = TabularResNet(in_features=in_features, hidden_dim=256, dropout_rate=0.2).to(
        device
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True
    )

    best_val_loss = float("inf")
    best_weights = None
    patience, patience_counter = 15, 0

    for epoch in range(100):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = net(bx)
                val_loss += criterion(preds, by).item() * len(by)
        val_loss /= len(y_va)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        net.load_state_dict(best_weights)

    net.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        oof_nn[val_idx] = net(va_tensor).cpu().numpy()

        te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        test_preds_nn += net(te_tensor).cpu().numpy() / kf.n_splits


# ---------------------------------------------------------
# 5. Out-Of-Fold Evaluation & Predictions Export
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn])

def loss_func(weights):
    pred = np.maximum(0, oof_matrix @ weights)
    return root_mean_squared_error(y_train, pred)

init_weights = np.array([0.35, 0.30, 0.25, 0.10])
bounds = [(0, 1) for _ in range(4)]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = res.x
print(f"Optimized Blending Weights: {opt_weights}")

oof_blend = np.maximum(0, oof_matrix @ opt_weights)
final_score = root_mean_squared_error(y_train, oof_blend)

test_blend = np.maximum(0, test_matrix @ opt_weights)
final_test_revenue = np.expm1(test_blend)

submission_df = pd.DataFrame({"id": test_ids, "revenue": final_test_revenue})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")