import ast
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# 1. Feature Engineering & Preprocessing Helpers
# ============================================================


def parse_json_field(val):
    """Safely parse JSON or stringified python list/dict structures."""
    if pd.isna(val) or val == "" or val == "[]" or val is None:
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val.replace("'", '"'))
        except Exception:
            return []


def parse_release_date(date_series):
    """Parse release dates handling two-digit year anomalies and missing dates."""
    dates = pd.to_datetime(date_series, errors="coerce")
    # TMDB date anomaly: 2-digit years parsed > 2024 are historically from the 1900s
    years = dates.dt.year.apply(lambda y: y - 100 if y > 2024 else y)

    median_year = int(years.median()) if not years.dropna().empty else 2000
    years = years.fillna(median_year)
    months = dates.dt.month.fillna(1).astype(int)
    days = dates.dt.day.fillna(1).astype(int)
    dayofweek = dates.dt.dayofweek.fillna(4).astype(int)
    quarter = dates.dt.quarter.fillna(1).astype(int)
    weekofyear = dates.dt.isocalendar().week.fillna(1).astype(int)

    return pd.DataFrame(
        {
            "release_year": years.astype(int),
            "release_month": months,
            "release_day": days,
            "release_dayofweek": dayofweek,
            "release_quarter": quarter,
            "release_weekofyear": weekofyear,
            "is_weekend_release": dayofweek.isin([4, 5, 6]).astype(int),
            "is_summer_release": months.isin([5, 6, 7]).astype(int),
            "is_holiday_release": months.isin([11, 12]).astype(int),
            "movie_age": 2024 - years.astype(int),
        },
        index=date_series.index,
    )


def extract_top_vocab(series, key="name", top_n=30):
    """Extract most frequent entities from training list-of-dict column."""
    counts = {}
    for items in series:
        parsed = parse_json_field(items)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and key in item:
                    name = item[key]
                    if name:
                        counts[name] = counts.get(name, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_items[:top_n]]


def extract_top_crew_roles(series, role="Director", top_n=25):
    """Extract top directors/producers from crew list."""
    counts = {}
    for items in series:
        parsed = parse_json_field(items)
        if isinstance(parsed, list):
            for member in parsed:
                if isinstance(member, dict) and member.get("job") == role:
                    name = member.get("name")
                    if name:
                        counts[name] = counts.get(name, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_items[:top_n]]


def engineer_features(df, is_train=True, top_vocabs=None):
    """Extract comprehensive engineered tabular features from raw TMDB dataframe."""
    out = pd.DataFrame(index=df.index)

    # 1. Numerical & Log Transformations
    budget = pd.to_numeric(df.get("budget", 0), errors="coerce").fillna(0)
    popularity = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0)
    runtime = pd.to_numeric(df.get("runtime", np.nan), errors="coerce")

    median_runtime = (
        top_vocabs.get("median_runtime", 100.0) if top_vocabs else runtime.median()
    )
    runtime = runtime.fillna(median_runtime)

    out["budget"] = budget
    out["log_budget"] = np.log1p(budget)
    out["is_budget_zero"] = (budget == 0).astype(int)

    out["popularity"] = popularity
    out["log_popularity"] = np.log1p(popularity)

    out["runtime"] = runtime
    out["is_runtime_zero"] = (runtime == 0).astype(int)
    out["runtime_to_median"] = runtime / (median_runtime + 1e-5)

    # Interaction Ratios
    out["budget_to_popularity"] = budget / (popularity + 1.0)
    out["budget_to_runtime"] = budget / (runtime + 1.0)
    out["popularity_to_runtime"] = popularity / (runtime + 1.0)
    out["budget_popularity_mult"] = out["log_budget"] * out["log_popularity"]

    # 2. Temporal Features
    date_series = (
        df["release_date"]
        if "release_date" in df.columns
        else pd.Series(index=df.index)
    )
    date_features = parse_release_date(date_series)
    for col in date_features.columns:
        out[col] = date_features[col]

    # 3. Collection Features
    collection_col = df.get("belongs_to_collection", pd.Series(index=df.index))
    has_collection = (
        collection_col.notna()
        & (collection_col != "")
        & (collection_col != "[]")
        & (collection_col != "None")
    )
    out["has_collection"] = has_collection.astype(int)

    # 4. Text Length and Presence Signals
    homepage_col = df.get("homepage", pd.Series(index=df.index))
    tagline_col = df.get("tagline", pd.Series(index=df.index))
    overview_col = df.get("overview", pd.Series(index=df.index))
    lang_col = df.get("original_language", pd.Series(index=df.index))
    title_col = df.get("title", pd.Series(index=df.index))
    orig_title_col = df.get("original_title", pd.Series(index=df.index))

    out["has_homepage"] = (
        homepage_col.notna() & (homepage_col != "") & (homepage_col != "None")
    ).astype(int)
    out["has_tagline"] = (
        tagline_col.notna() & (tagline_col != "") & (tagline_col != "None")
    ).astype(int)
    out["has_overview"] = (
        overview_col.notna() & (overview_col != "") & (overview_col != "None")
    ).astype(int)
    out["is_english"] = (lang_col == "en").astype(int)

    out["title_len"] = title_col.fillna("").astype(str).apply(len)
    out["title_word_count"] = (
        title_col.fillna("").astype(str).apply(lambda x: len(x.split()))
    )
    out["tagline_len"] = tagline_col.fillna("").astype(str).apply(len)
    out["tagline_word_count"] = (
        tagline_col.fillna("").astype(str).apply(lambda x: len(x.split()))
    )
    out["overview_len"] = overview_col.fillna("").astype(str).apply(len)
    out["overview_word_count"] = (
        overview_col.fillna("").astype(str).apply(lambda x: len(x.split()))
    )
    out["is_title_same_as_orig"] = (
        title_col.fillna("") == orig_title_col.fillna("")
    ).astype(int)

    # 5. JSON Parsed Counts and Top Vocabulary Indicators
    # Genres
    genres_col = df.get("genres", pd.Series(index=df.index))
    parsed_genres = genres_col.apply(parse_json_field)
    out["genres_count"] = parsed_genres.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if top_vocabs and "genres" in top_vocabs:
        for genre in top_vocabs["genres"]:
            col_name = f"genre_{genre.lower().replace(' ', '_')}"
            out[col_name] = parsed_genres.apply(
                lambda items: int(
                    any(isinstance(i, dict) and i.get("name") == genre for i in items)
                )
            )

    # Production Companies
    comp_col = df.get("production_companies", pd.Series(index=df.index))
    parsed_companies = comp_col.apply(parse_json_field)
    out["companies_count"] = parsed_companies.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if top_vocabs and "companies" in top_vocabs:
        for company in top_vocabs["companies"]:
            clean_comp = (
                company[:15].lower().replace(" ", "_").replace(".", "").replace("-", "")
            )
            col_name = f"company_{clean_comp}"
            out[col_name] = parsed_companies.apply(
                lambda items: int(
                    any(isinstance(i, dict) and i.get("name") == company for i in items)
                )
            )

    # Production Countries
    country_col = df.get("production_countries", pd.Series(index=df.index))
    parsed_countries = country_col.apply(parse_json_field)
    out["countries_count"] = parsed_countries.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if top_vocabs and "countries" in top_vocabs:
        for country in top_vocabs["countries"]:
            col_name = f"country_{country[:15].lower().replace(' ', '_')}"
            out[col_name] = parsed_countries.apply(
                lambda items: int(
                    any(isinstance(i, dict) and i.get("name") == country for i in items)
                )
            )

    # Spoken Languages
    lang_json_col = df.get("spoken_languages", pd.Series(index=df.index))
    parsed_languages = lang_json_col.apply(parse_json_field)
    out["spoken_languages_count"] = parsed_languages.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if top_vocabs and "languages" in top_vocabs:
        for lang in top_vocabs["languages"]:
            col_name = f"lang_{lang.lower()}"
            out[col_name] = parsed_languages.apply(
                lambda items: int(
                    any(
                        isinstance(i, dict) and i.get("iso_639_1") == lang
                        for i in items
                    )
                )
            )

    # Keywords
    kw_col = (
        df["Keywords"]
        if "Keywords" in df.columns
        else df.get("keywords", pd.Series(index=df.index))
    )
    parsed_keywords = kw_col.apply(parse_json_field)
    out["keywords_count"] = parsed_keywords.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if top_vocabs and "keywords" in top_vocabs:
        for kw in top_vocabs["keywords"]:
            col_name = f"kw_{kw[:15].lower().replace(' ', '_')}"
            out[col_name] = parsed_keywords.apply(
                lambda items: int(
                    any(isinstance(i, dict) and i.get("name") == kw for i in items)
                )
            )

    # Cast Features
    cast_col = df.get("cast", pd.Series(index=df.index))
    parsed_cast = cast_col.apply(parse_json_field)
    out["cast_count"] = parsed_cast.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    out["cast_female_count"] = parsed_cast.apply(
        lambda items: sum(
            1 for i in items if isinstance(i, dict) and i.get("gender") == 1
        )
    )
    out["cast_male_count"] = parsed_cast.apply(
        lambda items: sum(
            1 for i in items if isinstance(i, dict) and i.get("gender") == 2
        )
    )
    if top_vocabs and "actors" in top_vocabs:
        for actor in top_vocabs["actors"]:
            col_name = f"actor_{actor[:15].lower().replace(' ', '_')}"
            out[col_name] = parsed_cast.apply(
                lambda items: int(
                    any(isinstance(i, dict) and i.get("name") == actor for i in items)
                )
            )

    # Crew Features
    crew_col = df.get("crew", pd.Series(index=df.index))
    parsed_crew = crew_col.apply(parse_json_field)
    out["crew_count"] = parsed_crew.apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    out["crew_female_count"] = parsed_crew.apply(
        lambda items: sum(
            1 for i in items if isinstance(i, dict) and i.get("gender") == 1
        )
    )
    out["crew_male_count"] = parsed_crew.apply(
        lambda items: sum(
            1 for i in items if isinstance(i, dict) and i.get("gender") == 2
        )
    )
    out["director_count"] = parsed_crew.apply(
        lambda items: sum(
            1 for i in items if isinstance(i, dict) and i.get("job") == "Director"
        )
    )
    out["producer_count"] = parsed_crew.apply(
        lambda items: sum(
            1 for i in items if isinstance(i, dict) and i.get("job") == "Producer"
        )
    )
    out["writer_count"] = parsed_crew.apply(
        lambda items: sum(
            1
            for i in items
            if isinstance(i, dict) and i.get("job") in ["Writer", "Screenplay"]
        )
    )
    if top_vocabs and "directors" in top_vocabs:
        for director in top_vocabs["directors"]:
            col_name = f"dir_{director[:15].lower().replace(' ', '_')}"
            out[col_name] = parsed_crew.apply(
                lambda items: int(
                    any(
                        isinstance(i, dict)
                        and i.get("job") == "Director"
                        and i.get("name") == director
                        for i in items
                    )
                )
            )

    return out


# ============================================================
# 2. Metric & Model Architectures
# ============================================================


def compute_rmsle(y_true, y_pred):
    """Compute Root Mean Squared Logarithmic Error (RMSLE) on original revenue scale."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    y_true_clipped = np.clip(y_true, 0, None)
    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)
    return root_mean_squared_error(log_true, log_pred)


class TabularMLP(nn.Module):
    """Deep Tabular Multi-Layer Perceptron with Residual Blocks, LayerNorm, and Dropout."""

    def __init__(
        self, input_dim: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.25
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            shortcut = (
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )
            self.blocks.append(nn.ModuleDict({"layer": block, "shortcut": shortcut}))

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for b in self.blocks:
            h = b["layer"](h) + b["shortcut"](h)
        return self.head(h)


def get_lightgbm_regressor(params=None):
    """Instantiate LightGBM Regressor configured for log-revenue regression."""
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def get_xgboost_regressor(params=None):
    """Instantiate XGBoost Regressor configured for log-revenue regression."""
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)


def get_neural_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dims: list = [256, 128, 64],
    dropout: float = 0.25,
):
    """Build and initialize PyTorch model, criterion, and AdamW optimizer."""
    model = TabularMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


# ============================================================
# 3. Main Pipeline Execution
# ============================================================


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # 2. Extract Vocabulary and Statistics strictly from training set
    kw_train_col = (
        train_df["Keywords"] if "Keywords" in train_df.columns else train_df["keywords"]
    )
    top_vocabs = {
        "median_runtime": pd.to_numeric(train_df["runtime"], errors="coerce").median(),
        "genres": extract_top_vocab(train_df["genres"], key="name", top_n=20),
        "companies": extract_top_vocab(
            train_df["production_companies"], key="name", top_n=30
        ),
        "countries": extract_top_vocab(
            train_df["production_countries"], key="name", top_n=15
        ),
        "languages": extract_top_vocab(
            train_df["spoken_languages"], key="iso_639_1", top_n=10
        ),
        "keywords": extract_top_vocab(kw_train_col, key="name", top_n=30),
        "actors": extract_top_vocab(train_df["cast"], key="name", top_n=40),
        "directors": extract_top_crew_roles(
            train_df["crew"], role="Director", top_n=30
        ),
    }

    # 3. Engineer Features
    X_train = engineer_features(train_df, is_train=True, top_vocabs=top_vocabs)
    X_test = engineer_features(test_df, is_train=False, top_vocabs=top_vocabs)

    y_train_raw = train_df["revenue"].values
    y_train_log = np.log1p(y_train_raw)

    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )

    feature_cols = list(X_train.columns)

    # 4. K-Fold Cross Validation Setup
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_mlp = np.zeros(len(X_train))

    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_mlp = np.zeros(len(X_test))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr_df = X_train.iloc[train_idx]
        y_tr = y_train_log[train_idx]
        X_va_df = X_train.iloc[val_idx]
        y_va = y_train_log[val_idx]
        X_te_df = X_test.copy()

        # Model 1: LightGBM
        lgb_model = get_lightgbm_regressor({"random_state": 42 + fold})
        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=[(X_va_df, y_va)],
            eval_metric="rmse",
            callbacks=lgb_callbacks,
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va_df)
        test_lgb += lgb_model.predict(X_te_df) / n_splits

        # Model 2: XGBoost
        xgb_model = get_xgboost_regressor(
            {"early_stopping_rounds": 50, "random_state": 42 + fold}
        )
        xgb_model.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_va_df)
        test_xgb += xgb_model.predict(X_te_df) / n_splits

        # Model 3: Tabular MLP
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr_scaled = scaler.fit_transform(imputer.fit_transform(X_tr_df))
        X_va_scaled = scaler.transform(imputer.transform(X_va_df))
        X_te_scaled = scaler.transform(imputer.transform(X_te_df))

        tr_tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
        tr_tensor_y = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        va_tensor_x = torch.tensor(X_va_scaled, dtype=torch.float32)
        va_tensor_y = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)
        te_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32)

        tr_loader = DataLoader(
            TensorDataset(tr_tensor_x, tr_tensor_y), batch_size=64, shuffle=True
        )

        mlp_model, mlp_criterion, mlp_optimizer = get_neural_components(
            input_dim=len(feature_cols), lr=1e-3, weight_decay=1e-4
        )
        mlp_model.to(device)

        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(35):
            mlp_model.train()
            for batch_x, batch_y in tr_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                mlp_optimizer.zero_grad()
                pred = mlp_model(batch_x)
                loss = mlp_criterion(pred, batch_y)
                loss.backward()
                mlp_optimizer.step()

            mlp_model.eval()
            with torch.no_grad():
                va_x_dev, va_y_dev = va_tensor_x.to(device), va_tensor_y.to(device)
                va_pred = mlp_model(va_x_dev)
                val_loss = mlp_criterion(va_pred, va_y_dev).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {
                        k: v.cpu().clone() for k, v in mlp_model.state_dict().items()
                    }

        mlp_model.load_state_dict(best_weights)
        mlp_model.eval()
        with torch.no_grad():
            oof_mlp[val_idx] = mlp_model(va_tensor_x.to(device)).cpu().squeeze().numpy()
            test_mlp += (
                mlp_model(te_tensor_x.to(device)).cpu().squeeze().numpy() / n_splits
            )

        fold_lgb_rmsle = compute_rmsle(
            y_train_raw[val_idx], np.expm1(np.clip(oof_lgb[val_idx], 0, None))
        )
        fold_xgb_rmsle = compute_rmsle(
            y_train_raw[val_idx], np.expm1(np.clip(oof_xgb[val_idx], 0, None))
        )
        fold_mlp_rmsle = compute_rmsle(
            y_train_raw[val_idx], np.expm1(np.clip(oof_mlp[val_idx], 0, None))
        )
        print(
            f"Fold {fold + 1}/{n_splits} - LGB: {fold_lgb_rmsle:.4f} | XGB: {fold_xgb_rmsle:.4f} | MLP: {fold_mlp_rmsle:.4f}"
        )

    # 5. Ensemble Blend & Evaluation
    oof_ensemble_log = 0.45 * oof_lgb + 0.45 * oof_xgb + 0.10 * oof_mlp
    test_ensemble_log = 0.45 * test_lgb + 0.45 * test_xgb + 0.10 * test_mlp

    oof_pred_revenue = np.expm1(np.clip(oof_ensemble_log, 0, None))
    test_pred_revenue = np.expm1(np.clip(test_ensemble_log, 0, None))

    final_score = compute_rmsle(y_train_raw, oof_pred_revenue)

    # 6. Save Submission File
    sub_df = pd.DataFrame({"id": test_ids.astype(int), "revenue": test_pred_revenue})
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
