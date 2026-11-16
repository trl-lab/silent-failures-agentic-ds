import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# ---------------------------------------------------------
# 1. Metric Definition & Ordinal Threshold Optimizer
# ---------------------------------------------------------


def compute_qwk(y_true, y_pred):
    """Computes Quadratic Weighted Kappa (QWK) between ground truth and predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class OptimizedRounder:
    """Optimizes decision thresholds to map continuous ensemble predictions

    to discrete ordinal classes (0, 1, 2, 3, 4) maximizing Quadratic Weighted Kappa.
    Uses multi-start Powell and Nelder-Mead optimization with empirical quantile initialization.
    """

    def __init__(self, initial_coef=None):
        if initial_coef is None:
            self.coef_ = [0.5, 1.5, 2.5, 3.5]
        else:
            self.coef_ = list(initial_coef)

    def _loss(self, coef, X, y):
        c = np.sort(coef)
        X_p = np.zeros(len(X), dtype=int)
        X_p[X < c[0]] = 0
        X_p[(X >= c[0]) & (X < c[1])] = 1
        X_p[(X >= c[1]) & (X < c[2])] = 2
        X_p[(X >= c[2]) & (X < c[3])] = 3
        X_p[X >= c[3]] = 4
        return -compute_qwk(y, X_p)

    def fit(self, X, y):
        """Fit optimal thresholds using multi-start Powell and Nelder-Mead optimization."""
        counts = pd.Series(y).value_counts().sort_index()
        cum_dist = (counts / len(y)).cumsum().values
        base_thresholds = [float(np.percentile(X, 100 * cum_dist[i])) for i in range(4)]

        best_loss = float("inf")
        best_coef = base_thresholds

        # Multiple starting points with small perturbations around empirical quantiles
        candidate_starts = [base_thresholds]
        for delta in [-0.1, -0.05, 0.05, 0.1]:
            candidate_starts.append([t + delta * (i + 1) for i, t in enumerate(base_thresholds)])

        for init_t in candidate_starts:
            res_powell = minimize(self._loss, init_t, args=(X, y), method="Powell", tol=1e-5)
            if res_powell.fun < best_loss:
                best_loss = res_powell.fun
                best_coef = sorted(res_powell.x)

            res_nm = minimize(
                self._loss,
                sorted(res_powell.x),
                args=(X, y),
                method="Nelder-Mead",
                tol=1e-5,
            )
            if res_nm.fun < best_loss:
                best_loss = res_nm.fun
                best_coef = sorted(res_nm.x)

        self.coef_ = sorted(best_coef)
        return self

    def predict(self, X):
        """Discretize continuous predictions using learned thresholds."""
        X_p = np.copy(X)
        res = np.zeros(len(X_p), dtype=int)
        res[X_p < self.coef_[0]] = 0
        res[(X_p >= self.coef_[0]) & (X_p < self.coef_[1])] = 1
        res[(X_p >= self.coef_[1]) & (X_p < self.coef_[2])] = 2
        res[(X_p >= self.coef_[2]) & (X_p < self.coef_[3])] = 3
        res[X_p >= self.coef_[3]] = 4
        return res


def optimize_blend_weights(oof_matrix, y_true):
    """Optimizes simplex-constrained ensemble weights minimizing MSE on OOF predictions."""
    n_models = oof_matrix.shape[1]

    def loss_func(weights):
        pred = np.dot(oof_matrix, weights)
        return np.mean((pred - y_true) ** 2)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    weights = res.x / np.sum(res.x)
    return weights


# ---------------------------------------------------------
# 2. Model Parameter Configurations
# ---------------------------------------------------------


def get_lgb_model_params(random_seed=42):
    """Returns LightGBM regression parameters configured with robust regularization."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(random_seed=42):
    """Returns XGBoost regression parameters configured for tabular-text ordinal representations."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
    }


def get_cb_model_params(random_seed=42):
    """Returns CatBoost regression parameters configured with symmetric tree structures."""
    return {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_seed,
        "verbose": False,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
    }


def get_lgb_clf_params(random_seed=42):
    """Returns LightGBM multiclass classification parameters."""
    return {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_clf_params(random_seed=42):
    """Returns XGBoost multiclass classification parameters."""
    return {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
    }


# ---------------------------------------------------------
# 3. Data Loading & Feature Engineering Helpers
# ---------------------------------------------------------


def extract_text_meta_features(df, text_col="Description"):
    text_series = df[text_col].fillna("").astype(str)

    char_len = text_series.str.len()
    word_count = text_series.apply(lambda x: len(x.split()))
    avg_word_len = char_len / (word_count + 1)
    num_uppercase = text_series.apply(lambda x: sum(1 for c in x if c.isupper()))
    uppercase_ratio = num_uppercase / (char_len + 1)
    num_exclamations = text_series.apply(lambda x: x.count("!"))
    num_question = text_series.apply(lambda x: x.count("?"))
    num_digits = text_series.apply(lambda x: sum(1 for c in x if c.isdigit()))

    urgent_words = [
        "urgent", "emergency", "asap", "help", "save", "please", "foster",
        "deadline", "abandoned", "dying", "critical", "immediately", "shelter"
    ]
    positive_words = [
        "cute", "playful", "loving", "sweet", "friendly", "trained", "healthy",
        "gentle", "calm", "active", "quiet", "protective", "loyal", "affectionate"
    ]
    health_words = [
        "vaccinated", "dewormed", "neutered", "spayed", "sterilized", "vet",
        "treatment", "sick", "injured", "parvo", "mange", "flea", "tick",
        "wound", "special need", "medical"
    ]
    fee_words = ["free", "waived", "fee", "rm", "cost", "deposit", "donation"]

    def count_keywords(text, kw_list):
        t_lower = text.lower()
        return sum(1 for w in kw_list if w in t_lower)

    urgent_cnt = text_series.apply(lambda x: count_keywords(x, urgent_words))
    pos_cnt = text_series.apply(lambda x: count_keywords(x, positive_words))
    health_cnt = text_series.apply(lambda x: count_keywords(x, health_words))
    fee_kw_cnt = text_series.apply(lambda x: count_keywords(x, fee_words))

    meta_df = pd.DataFrame(
        {
            f"{text_col}_len": char_len,
            f"{text_col}_word_count": word_count,
            f"{text_col}_avg_word_len": avg_word_len,
            f"{text_col}_num_upper": num_uppercase,
            f"{text_col}_upper_ratio": uppercase_ratio,
            f"{text_col}_num_exclamations": num_exclamations,
            f"{text_col}_num_question": num_question,
            f"{text_col}_num_digits": num_digits,
            f"{text_col}_urgent_cnt": urgent_cnt,
            f"{text_col}_pos_cnt": pos_cnt,
            f"{text_col}_health_cnt": health_cnt,
            f"{text_col}_fee_kw_cnt": fee_kw_cnt,
        },
        index=df.index,
    )
    return meta_df


def extract_name_features(df, name_col="Name"):
    name_series = df[name_col].fillna("").astype(str).str.strip().str.lower()
    unnamed_keywords = {
        "",
        "no name",
        "unnamed",
        "unknown",
        "none",
        "no",
        "baby",
        "puppy",
        "kitten",
    }
    has_valid_name = (
        ~name_series.isin(unnamed_keywords) & (name_series.str.len() > 1)
    ).astype(int)
    name_len = name_series.str.len()
    name_words = name_series.apply(lambda x: len(x.split()))

    return pd.DataFrame(
        {
            "has_valid_name": has_valid_name,
            "name_len": name_len,
            "name_words": name_words,
        },
        index=df.index,
    )


def engineer_tabular_features(df):
    df = df.copy()

    df["Age"] = df["Age"].fillna(0).astype(float)
    df["Fee"] = df["Fee"].fillna(0).astype(float)
    df["Quantity"] = df["Quantity"].fillna(1).clip(lower=1).astype(float)
    df["PhotoAmt"] = df["PhotoAmt"].fillna(0).astype(float)
    df["VideoAmt"] = df["VideoAmt"].fillna(0).astype(float)

    # Ratios and Per-Pet features
    df["fee_per_pet"] = df["Fee"] / df["Quantity"]
    df["photo_per_pet"] = df["PhotoAmt"] / df["Quantity"]
    df["video_per_pet"] = df["VideoAmt"] / df["Quantity"]
    df["total_media"] = df["PhotoAmt"] + df["VideoAmt"]
    df["has_photo"] = (df["PhotoAmt"] > 0).astype(int)
    df["has_video"] = (df["VideoAmt"] > 0).astype(int)
    df["is_free"] = (df["Fee"] == 0).astype(int)

    # Age features
    df["is_baby"] = (df["Age"] <= 3).astype(int)
    df["is_young"] = ((df["Age"] > 3) & (df["Age"] <= 12)).astype(int)
    df["is_senior"] = (df["Age"] >= 72).astype(int)
    df["age_years"] = df["Age"] / 12.0
    df["log_age"] = np.log1p(df["Age"])
    df["log_fee"] = np.log1p(df["Fee"])

    # Breed features
    df["is_purebred"] = (
        (df["Breed2"] == 0) & (df["Breed1"] != 307) & (df["Breed1"] != 0)
    ).astype(int)
    df["has_secondary_breed"] = (df["Breed2"] > 0).astype(int)

    # Color features
    df["num_colors"] = (
        (df["Color1"] > 0).astype(int)
        + (df["Color2"] > 0).astype(int)
        + (df["Color3"] > 0).astype(int)
    )

    # Health and Veterinary Care Index
    df["is_vaccinated"] = (df["Vaccinated"] == 1).astype(int)
    df["is_dewormed"] = (df["Dewormed"] == 1).astype(int)
    df["is_sterilized"] = (df["Sterilized"] == 1).astype(int)
    df["fully_vetted"] = (
        df["is_vaccinated"] & df["is_dewormed"] & df["is_sterilized"]
    ).astype(int)
    df["vet_care_score"] = df["is_vaccinated"] + df["is_dewormed"] + df["is_sterilized"]

    # Health status
    df["is_healthy"] = (df["Health"] == 1).astype(int)
    df["has_health_issue"] = (df["Health"].isin([2, 3])).astype(int)

    # Categorical interactions
    df["type_gender"] = df["Type"].astype(str) + "_" + df["Gender"].astype(str)
    df["maturity_fur"] = (
        df["MaturitySize"].astype(str) + "_" + df["FurLength"].astype(str)
    )
    df["breed_pair"] = df["Breed1"].astype(str) + "_" + df["Breed2"].astype(str)
    df["state_type"] = df["State"].astype(str) + "_" + df["Type"].astype(str)

    return df


def compute_group_aggregations(train_df, test_df):
    """Computes group-level statistical aggregations and relative differences/ratios."""
    group_cols = ["Breed1", "State", "Type", "breed_pair"]
    target_num_cols = ["Age", "Fee", "PhotoAmt"]

    train_feat = pd.DataFrame(index=train_df.index)
    test_feat = pd.DataFrame(index=test_df.index)

    for g in group_cols:
        for val_col in target_num_cols:
            grouped = train_df.groupby(g)[val_col].agg(["mean", "std", "min", "max"])
            grouped.columns = [
                f"{g}_{val_col}_{stat}" for stat in ["mean", "std", "min", "max"]
            ]
            grouped[f"{g}_{val_col}_std"] = grouped[f"{g}_{val_col}_std"].fillna(0)

            global_mean = float(train_df[val_col].mean())
            global_min = float(train_df[val_col].min())
            global_max = float(train_df[val_col].max())

            train_merged = train_df[[g, val_col]].merge(grouped, on=g, how="left")
            test_merged = test_df[[g, val_col]].merge(grouped, on=g, how="left")

            train_mean_s = train_merged[f"{g}_{val_col}_mean"].fillna(global_mean)
            test_mean_s = test_merged[f"{g}_{val_col}_mean"].fillna(global_mean)

            train_feat[f"{g}_{val_col}_mean"] = train_mean_s.values
            train_feat[f"{g}_{val_col}_std"] = train_merged[f"{g}_{val_col}_std"].fillna(0).values
            train_feat[f"{g}_{val_col}_min"] = train_merged[f"{g}_{val_col}_min"].fillna(global_min).values
            train_feat[f"{g}_{val_col}_max"] = train_merged[f"{g}_{val_col}_max"].fillna(global_max).values

            test_feat[f"{g}_{val_col}_mean"] = test_mean_s.values
            test_feat[f"{g}_{val_col}_std"] = test_merged[f"{g}_{val_col}_std"].fillna(0).values
            test_feat[f"{g}_{val_col}_min"] = test_merged[f"{g}_{val_col}_min"].fillna(global_min).values
            test_feat[f"{g}_{val_col}_max"] = test_merged[f"{g}_{val_col}_max"].fillna(global_max).values

            train_feat[f"{g}_{val_col}_diff"] = train_df[val_col].values - train_mean_s.values
            test_feat[f"{g}_{val_col}_diff"] = test_df[val_col].values - test_mean_s.values

            train_feat[f"{g}_{val_col}_ratio"] = (train_df[val_col].values + 1e-3) / (train_mean_s.values + 1e-3)
            test_feat[f"{g}_{val_col}_ratio"] = (test_df[val_col].values + 1e-3) / (test_mean_s.values + 1e-3)

    return train_feat, test_feat


def compute_oof_target_encoding(
    train_df, test_df, y_train, fold_assignments, n_splits, te_cols, smoothing=10.0
):
    """Computes leak-free smoothed out-of-fold target encodings for high-cardinality categorical features."""
    oof_te = pd.DataFrame(index=train_df.index)
    test_te = pd.DataFrame(index=test_df.index)
    global_mean = float(np.mean(y_train))

    for c in te_cols:
        oof_te[f"{c}_te"] = global_mean
        test_te[f"{c}_te"] = 0.0

    for fold in range(n_splits):
        tr_idx = np.where(fold_assignments != fold)[0]
        va_idx = np.where(fold_assignments == fold)[0]

        y_tr = y_train[tr_idx]
        prior = float(np.mean(y_tr))

        for c in te_cols:
            fold_df = pd.DataFrame({c: train_df[c].iloc[tr_idx], "target": y_tr})
            stats = fold_df.groupby(c)["target"].agg(["count", "mean"])
            te_map = (
                (stats["count"] * stats["mean"] + smoothing * prior)
                / (stats["count"] + smoothing)
            ).to_dict()

            oof_te.iloc[va_idx, oof_te.columns.get_loc(f"{c}_te")] = (
                train_df[c]
                .iloc[va_idx]
                .map(te_map)
                .fillna(prior)
                .values
            )
            test_te[f"{c}_te"] += (
                test_df[c].map(te_map).fillna(prior).values / n_splits
            )

    return oof_te, test_te


# ---------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Load datasets
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "AdoptionSpeed"
    y_train = train_df[target_col].values

    # Feature extraction
    train_meta_desc = extract_text_meta_features(train_df, "Description")
    test_meta_desc = extract_text_meta_features(test_df, "Description")

    train_name_feat = extract_name_features(train_df, "Name")
    test_name_feat = extract_name_features(test_df, "Name")

    train_tabular = engineer_tabular_features(train_df)
    test_tabular = engineer_tabular_features(test_df)

    # Frequency encodings strictly calculated on training split
    freq_cols = [
        "RescuerID",
        "State",
        "Breed1",
        "Breed2",
        "type_gender",
        "maturity_fur",
        "breed_pair",
        "state_type",
    ]
    for col in freq_cols:
        freq_map = train_tabular[col].value_counts(normalize=True).to_dict()
        train_tabular[f"{col}_freq"] = train_tabular[col].map(freq_map).fillna(0)
        test_tabular[f"{col}_freq"] = test_tabular[col].map(freq_map).fillna(0)

    # Rescuer aggregated statistics from train
    rescuer_counts = train_tabular["RescuerID"].value_counts().to_dict()
    train_tabular["rescuer_total_listings"] = (
        train_tabular["RescuerID"].map(rescuer_counts).fillna(1)
    )
    test_tabular["rescuer_total_listings"] = (
        test_tabular["RescuerID"].map(rescuer_counts).fillna(1)
    )

    rescuer_stats = (
        train_tabular.groupby("RescuerID")
        .agg({"Fee": "mean", "Age": "mean", "is_free": "mean"})
        .rename(
            columns={
                "Fee": "rescuer_mean_fee",
                "Age": "rescuer_mean_age",
                "is_free": "rescuer_pct_free",
            }
        )
    )

    train_tabular = train_tabular.merge(rescuer_stats, on="RescuerID", how="left")
    test_tabular = test_tabular.merge(rescuer_stats, on="RescuerID", how="left")

    for c in ["rescuer_mean_fee", "rescuer_mean_age", "rescuer_pct_free"]:
        mean_val = train_tabular[c].mean()
        train_tabular[c] = train_tabular[c].fillna(mean_val)
        test_tabular[c] = test_tabular[c].fillna(mean_val)

    # Hierarchical group aggregations (Breed1, State, Type, breed_pair)
    train_group_feat, test_group_feat = compute_group_aggregations(
        train_tabular, test_tabular
    )

    # Multi-granularity Text Feature Extraction (Word + Char TF-IDF with 32 SVD components)
    train_desc_raw = train_df["Description"].fillna("").astype(str).tolist()
    test_desc_raw = test_df["Description"].fillna("").astype(str).tolist()

    word_tfidf = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 3), min_df=2
    )
    char_tfidf = TfidfVectorizer(
        max_features=5000, analyzer="char_wb", ngram_range=(3, 5), min_df=5
    )

    train_word = word_tfidf.fit_transform(train_desc_raw)
    test_word = word_tfidf.transform(test_desc_raw)

    train_char = char_tfidf.fit_transform(train_desc_raw)
    test_char = char_tfidf.transform(test_desc_raw)

    train_tfidf_all = hstack([train_word, train_char])
    test_tfidf_all = hstack([test_word, test_char])

    n_svd_components = 32
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
    train_svd = svd.fit_transform(train_tfidf_all)
    test_svd = svd.transform(test_tfidf_all)

    svd_cols = [f"desc_svd_{i}" for i in range(n_svd_components)]
    train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
    test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)

    # Stratified 5-Fold Setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_tabular, y_train)):
        fold_assignments[val_idx] = fold

    # Leakage-free Out-Of-Fold Target Encoding
    te_target_cols = ["Breed1", "State", "RescuerID", "breed_pair"]
    train_te, test_te = compute_oof_target_encoding(
        train_tabular,
        test_tabular,
        y_train,
        fold_assignments,
        n_splits,
        te_target_cols,
        smoothing=15.0,
    )

    # Drop non-numeric identifier / string columns
    cols_to_drop = [
        "PetID",
        "Name",
        "Description",
        "RescuerID",
        "type_gender",
        "maturity_fur",
        "breed_pair",
        "state_type",
        "AdoptionSpeed",
    ]

    X_train_numeric = train_tabular.drop(
        columns=[c for c in cols_to_drop if c in train_tabular.columns]
    )
    X_test_numeric = test_tabular.drop(
        columns=[c for c in cols_to_drop if c in test_tabular.columns]
    )

    X_train_full = pd.concat(
        [
            X_train_numeric,
            train_group_feat,
            train_te,
            train_meta_desc,
            train_name_feat,
            train_svd_df,
        ],
        axis=1,
    )
    X_test_full = pd.concat(
        [
            X_test_numeric,
            test_group_feat,
            test_te,
            test_meta_desc,
            test_name_feat,
            test_svd_df,
        ],
        axis=1,
    )

    feature_names = list(X_train_full.columns)
    X_test_full = X_test_full[feature_names]

    oof_continuous = np.zeros(len(train_df), dtype=float)
    test_continuous = np.zeros(len(test_df), dtype=float)

    # 5-Fold Training Loop with Diverse 3-Way Ensemble (LGBM + XGBoost + CatBoost)
    for fold in range(n_splits):
        train_idx = np.where(fold_assignments != fold)[0]
        val_idx = np.where(fold_assignments == fold)[0]

        X_tr, y_tr = X_train_full.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train_full.iloc[val_idx], y_train[val_idx]

        # 1. LightGBM Regressor
        lgb_params = get_lgb_model_params(random_seed=42 + fold)
        train_lgb_ds = lgb.Dataset(X_tr, label=y_tr)
        val_lgb_ds = lgb.Dataset(X_va, label=y_va, reference=train_lgb_ds)

        lgb_callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model_lgb = lgb.train(
            lgb_params,
            train_lgb_ds,
            num_boost_round=2000,
            valid_sets=[val_lgb_ds],
            callbacks=lgb_callbacks,
        )

        val_preds_lgb = model_lgb.predict(
            X_va, num_iteration=model_lgb.best_iteration
        ).astype(float)
        test_preds_lgb = model_lgb.predict(
            X_test_full, num_iteration=model_lgb.best_iteration
        ).astype(float)

        # 2. XGBoost Regressor
        xgb_params = get_xgb_model_params(random_seed=42 + fold)
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test_full)

        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        val_preds_xgb = model_xgb.predict(dval).astype(float)
        test_preds_xgb = model_xgb.predict(dtest).astype(float)

        # 3. CatBoost Regressor
        cb_params = get_cb_model_params(random_seed=42 + fold)
        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            verbose=False,
        )

        val_preds_cb = model_cb.predict(X_va).astype(float)
        test_preds_cb = model_cb.predict(X_test_full).astype(float)

        # Diverse 3-way continuous regression blend (40% LGBM, 35% XGBoost, 25% CatBoost)
        val_preds_blend = (
            0.40 * val_preds_lgb + 0.35 * val_preds_xgb + 0.25 * val_preds_cb
        )
        oof_continuous[val_idx] = val_preds_blend

        test_preds_blend = (
            0.40 * test_preds_lgb + 0.35 * test_preds_xgb + 0.25 * test_preds_cb
        )
        test_continuous += test_preds_blend / n_splits

    # Global Threshold Optimization
    global_rounder = OptimizedRounder()
    global_rounder.fit(oof_continuous, y_train)

    oof_discrete = global_rounder.predict(oof_continuous)
    final_validation_score = compute_qwk(y_train, oof_discrete)

    # Discretize Test Predictions using identical optimized thresholds
    test_discrete = global_rounder.predict(test_continuous)

    # Format and save submission
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_df)), "AdoptionSpeed": test_discrete.astype(int)}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()
