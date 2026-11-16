import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------------------------------------
# 1. Setup Directories and Load Data
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Standardize string formatting and handle missing values
cat_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

for col in cat_cols:
    train_df[col] = (
        train_df[col]
        .astype(str)
        .str.strip()
        .replace({"?": "Unknown", "nan": "Unknown"})
    )
    test_df[col] = (
        test_df[col].astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
    )

# Normalize target variable and capture output format
if "income" in train_df.columns:
    if train_df["income"].dtype == object or isinstance(
        train_df["income"].iloc[0], str
    ):
        target_clean = train_df["income"].astype(str).str.strip().str.rstrip(".")
        train_df["target"] = target_clean.apply(lambda x: 1 if ">50K" in x else 0)
        pos_label = ">50K"
        neg_label = "<=50K"
    else:
        train_df["target"] = train_df["income"].astype(int)
        pos_label = 1
        neg_label = 0
else:
    raise ValueError("Target column 'income' not found in training data.")


# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------
def create_domain_features(df):
    data = df.copy()

    # Financial indicators & non-linear transformations
    cap_gain = data["capital-gain"].astype(float)
    cap_loss = data["capital-loss"].astype(float)
    data["capital_net"] = cap_gain - cap_loss
    data["log_capital_gain"] = np.log1p(cap_gain)
    data["log_capital_loss"] = np.log1p(cap_loss)
    data["log_capital_net"] = np.sign(data["capital_net"]) * np.log1p(
        np.abs(data["capital_net"])
    )
    data["has_capital_gain"] = (cap_gain > 0).astype(int)
    data["has_capital_loss"] = (cap_loss > 0).astype(int)
    data["has_capital_activity"] = ((cap_gain > 0) | (cap_loss > 0)).astype(int)
    # 99,999 capital gain ceiling flag & high loss indicator
    data["is_cap_gain_max"] = (cap_gain >= 99999).astype(int)
    data["is_cap_loss_high"] = (cap_loss >= 2000).astype(int)

    # Age and work intensity interactions
    age = data["age"].astype(float)
    hours = data["hours-per-week"].astype(float)
    edu_num = data["education-num"].astype(float)

    # Mincer's potential work experience & labor economics indicators
    mincer_exp = np.clip(age - edu_num - 6.0, 0.0, None)
    data["mincer_experience"] = mincer_exp
    data["mincer_experience_sq"] = (mincer_exp / 10.0) ** 2
    data["exp_age_ratio"] = mincer_exp / (age + 1.0)
    data["exp_to_edu_ratio"] = mincer_exp / (edu_num + 1.0)
    data["career_peak_proxy"] = np.clip(age - 45.0, -25.0, 25.0)

    data["work_intensity"] = age * hours
    data["edu_age_ratio"] = edu_num / (age + 1.0)
    data["hours_per_edu"] = hours / (edu_num + 1.0)
    data["edu_hours_prod"] = edu_num * hours
    data["exp_hours_prod"] = mincer_exp * hours

    data["capital_gain_rate"] = cap_gain / ((hours * 52.0) + 1.0)
    data["capital_gain_per_exp"] = cap_gain / (mincer_exp + 1.0)
    data["capital_net_per_age"] = data["capital_net"] / (age + 1.0)

    data["is_full_time"] = (hours == 40).astype(int)
    data["is_over_time"] = (hours > 40).astype(int)
    data["is_part_time"] = (hours < 40).astype(int)

    # Demographics & Family interactions
    data["is_married"] = (
        data["marital-status"]
        .str.contains("Married-civ-spouse|Married-AF-spouse", regex=True)
        .astype(int)
    )
    data["is_us_native"] = (data["native-country"] == "United-States").astype(int)
    data["is_male"] = (data["sex"] == "Male").astype(int)
    data["is_prime_male_married"] = (
        (age >= 25) & (age <= 55) & (data["is_male"] == 1) & (data["is_married"] == 1)
    ).astype(int)

    # Career life-stages
    data["career_stage"] = pd.cut(
        age,
        bins=[-1, 25, 35, 50, 65, 120],
        labels=["entry", "early", "prime", "mature", "retirement"],
    ).astype(str)

    # Occupation skill tiers and education tiers
    skill_map = {
        "Exec-managerial": 3,
        "Prof-specialty": 3,
        "Tech-support": 3,
        "Sales": 2,
        "Craft-repair": 2,
        "Adm-clerical": 2,
        "Protective-serv": 2,
        "Other-service": 1,
        "Handlers-cleaners": 1,
        "Machine-op-inspct": 1,
        "Transport-moving": 1,
        "Farming-fishing": 1,
        "Priv-house-serv": 1,
        "Armed-Forces": 1,
    }
    data["occ_skill_tier"] = (
        data["occupation"].map(skill_map).fillna(0).astype(int)
    )

    edu_tier_map = {
        "Doctorate": 4,
        "Prof-school": 4,
        "Masters": 3,
        "Bachelors": 3,
        "Assoc-acdm": 2,
        "Assoc-voc": 2,
        "Some-college": 2,
    }
    data["edu_tier"] = data["education"].map(edu_tier_map).fillna(1).astype(int)
    data["is_high_edu_high_occ"] = (
        (data["occ_skill_tier"] == 3) & (data["edu_tier"] >= 3)
    ).astype(int)

    # Log-transformed fnlwgt interactions
    if "fnlwgt" in data.columns:
        fnlwgt = data["fnlwgt"].astype(float)
        data["log_fnlwgt"] = np.log1p(fnlwgt)
        data["fnlwgt_age_ratio"] = fnlwgt / (age + 1.0)
        data["fnlwgt_hours_prod"] = data["log_fnlwgt"] * hours

    # Cross-categorical interactions
    data["occ_edu"] = data["occupation"] + "_" + data["education"]
    data["marital_rel"] = data["marital-status"] + "_" + data["relationship"]
    data["workclass_occ"] = data["workclass"] + "_" + data["occupation"]
    data["race_sex"] = data["race"] + "_" + data["sex"]
    data["occupation_sex"] = data["occupation"] + "_" + data["sex"]
    data["education_sex"] = data["education"] + "_" + data["sex"]
    data["marital_workclass"] = data["marital-status"] + "_" + data["workclass"]
    data["occ_race"] = data["occupation"] + "_" + data["race"]
    data["occ_edu_tier_combo"] = (
        data["occ_skill_tier"].astype(str) + "_" + data["edu_tier"].astype(str)
    )

    return data


train_feat = create_domain_features(train_df)
test_feat = create_domain_features(test_df)

# Stratified K-Fold Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_feat["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_feat, train_feat["target"])
):
    train_feat.loc[val_idx, "fold"] = fold

# Leakage-Free Statistical Aggregations & Target / Frequency Encodings
categorical_features = cat_cols + [
    "career_stage",
    "occ_edu",
    "marital_rel",
    "workclass_occ",
    "race_sex",
    "occupation_sex",
    "education_sex",
    "marital_workclass",
    "occ_race",
    "occ_edu_tier_combo",
]
num_agg_cols = [
    "hours-per-week",
    "age",
    "education-num",
    "capital_net",
    "mincer_experience",
]

# Frequency encoding
for col in categorical_features:
    freq = train_feat[col].value_counts(normalize=True)
    train_feat[f"{col}_freq"] = train_feat[col].map(freq).fillna(0).astype(float)
    test_feat[f"{col}_freq"] = test_feat[col].map(freq).fillna(0).astype(float)

# Out-of-fold Smoothed Target Encoding
te_smoothing = 10.0
global_target_mean = train_feat["target"].mean()

for col in categorical_features:
    train_feat[f"{col}_te"] = global_target_mean
    test_feat[f"{col}_te"] = global_target_mean

for fold in range(5):
    tr_mask = train_feat["fold"] != fold
    val_mask = train_feat["fold"] == fold
    fold_prior = train_feat.loc[tr_mask, "target"].mean()

    for col in categorical_features:
        stats = (
            train_feat[tr_mask]
            .groupby(col, observed=False)["target"]
            .agg(["count", "mean"])
        )
        smoothed = (stats["count"] * stats["mean"] + te_smoothing * fold_prior) / (
            stats["count"] + te_smoothing
        )
        train_feat.loc[val_mask, f"{col}_te"] = (
            train_feat.loc[val_mask, col].map(smoothed).fillna(fold_prior).astype(float)
        )

# Test target encoding using full train stats
for col in categorical_features:
    full_stats = (
        train_feat.groupby(col, observed=False)["target"].agg(["count", "mean"])
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"]
        + te_smoothing * global_target_mean
    ) / (full_stats["count"] + te_smoothing)
    test_feat[f"{col}_te"] = (
        test_feat[col].map(full_smoothed).fillna(global_target_mean).astype(float)
    )

# Out-of-fold group aggregations
agg_groupings = ["occupation", "education", "workclass", "occupation_sex"]
for group_col in agg_groupings:
    for val_col in num_agg_cols:
        train_feat[f"{group_col}_{val_col}_mean"] = 0.0
        train_feat[f"{group_col}_{val_col}_std"] = 0.0

        for fold in range(5):
            tr_mask = train_feat["fold"] != fold
            val_mask = train_feat["fold"] == fold

            group_stats = (
                train_feat[tr_mask].groupby(group_col)[val_col].agg(["mean", "std"])
            )
            train_feat.loc[val_mask, f"{group_col}_{val_col}_mean"] = (
                train_feat.loc[val_mask, group_col]
                .map(group_stats["mean"])
                .fillna(train_feat[tr_mask][val_col].mean())
            )
            train_feat.loc[val_mask, f"{group_col}_{val_col}_std"] = (
                train_feat.loc[val_mask, group_col].map(group_stats["std"]).fillna(0.0)
            )

        full_group_stats = train_feat.groupby(group_col)[val_col].agg(["mean", "std"])
        test_feat[f"{group_col}_{val_col}_mean"] = (
            test_feat[group_col]
            .map(full_group_stats["mean"])
            .fillna(train_feat[val_col].mean())
        )
        test_feat[f"{group_col}_{val_col}_std"] = (
            test_feat[group_col].map(full_group_stats["std"]).fillna(0.0)
        )

        train_feat[f"{val_col}_diff_{group_col}_mean"] = (
            train_feat[val_col] - train_feat[f"{group_col}_{val_col}_mean"]
        )
        test_feat[f"{val_col}_diff_{group_col}_mean"] = (
            test_feat[val_col] - test_feat[f"{group_col}_{val_col}_mean"]
        )
        train_feat[f"{val_col}_zscore_{group_col}"] = (
            train_feat[val_col] - train_feat[f"{group_col}_{val_col}_mean"]
        ) / (train_feat[f"{group_col}_{val_col}_std"] + 1e-5)
        test_feat[f"{val_col}_zscore_{group_col}"] = (
            test_feat[val_col] - test_feat[f"{group_col}_{val_col}_mean"]
        ) / (test_feat[f"{group_col}_{val_col}_std"] + 1e-5)

# Convert categorical columns to category dtype for tree models
for col in categorical_features:
    train_feat[col] = train_feat[col].astype("category")
    test_feat[col] = test_feat[col].astype("category")


# ---------------------------------------------------------
# 3. Model Architecture Configurations
# ---------------------------------------------------------
def get_lgbm_model(random_state=42):
    return LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state=42):
    return XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_model(random_state=42):
    if CATBOOST_AVAILABLE:
        return CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            early_stopping_rounds=50,
            verbose=False,
            cat_features=categorical_features,
        )
    else:
        return HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )


def compute_competition_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# ---------------------------------------------------------
# 4. Training, Validation & Ensembling
# ---------------------------------------------------------
feature_cols = [
    col for col in train_feat.columns if col not in ["target", "fold", "income", "id"]
]

X = train_feat[feature_cols].copy()
y = train_feat["target"].values
folds = train_feat["fold"].values
X_test = test_feat[feature_cols].copy()

oof_lgb_probs = np.zeros(len(train_feat), dtype=np.float64)
oof_xgb_probs = np.zeros(len(train_feat), dtype=np.float64)
oof_cat_probs = np.zeros(len(train_feat), dtype=np.float64)
test_lgb_probs = np.zeros(len(test_feat), dtype=np.float64)
test_xgb_probs = np.zeros(len(test_feat), dtype=np.float64)
test_cat_probs = np.zeros(len(test_feat), dtype=np.float64)

n_splits = 5

for fold in range(n_splits):
    tr_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_train, y_train = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = get_lgbm_model(random_state=42 + fold)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    oof_lgb_probs[val_idx] = val_lgb_pred
    test_lgb_probs += lgb_model.predict_proba(X_test)[:, 1] / n_splits

    # XGBoost
    xgb_model = get_xgb_model(random_state=42 + fold)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    oof_xgb_probs[val_idx] = val_xgb_pred
    test_xgb_probs += xgb_model.predict_proba(X_test)[:, 1] / n_splits

    # CatBoost
    cat_model = get_catboost_model(random_state=42 + fold)
    if CATBOOST_AVAILABLE:
        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
    else:
        cat_model.fit(X_train, y_train)

    val_cat_pred = cat_model.predict_proba(X_val)[:, 1]
    oof_cat_probs[val_idx] = val_cat_pred
    test_cat_probs += cat_model.predict_proba(X_test)[:, 1] / n_splits

    fold_lgb_acc = accuracy_score(y_val, (val_lgb_pred >= 0.5).astype(int))
    fold_xgb_acc = accuracy_score(y_val, (val_xgb_pred >= 0.5).astype(int))
    fold_cat_acc = accuracy_score(y_val, (val_cat_pred >= 0.5).astype(int))
    print(
        f"Fold {fold + 1}/{n_splits} - LightGBM: {fold_lgb_acc:.5f}, XGBoost: {fold_xgb_acc:.5f}, CatBoost: {fold_cat_acc:.5f}"
    )

# Continuous Ensemble Weight & Decision Threshold Optimization
def find_best_threshold(y_true, probs):
    best_th = 0.5
    best_acc = 0.0
    thresholds = np.linspace(0.35, 0.65, 301)
    for th in thresholds:
        acc = accuracy_score(y_true, (probs >= th).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_th, best_acc


best_acc = 0.0
best_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
best_threshold = 0.5

oof_matrix = np.column_stack([oof_lgb_probs, oof_xgb_probs, oof_cat_probs])
test_matrix = np.column_stack([test_lgb_probs, test_xgb_probs, test_cat_probs])

# Simplex grid initialization
step = 0.02
for w1 in np.arange(0.0, 1.0 + step / 2, step):
    for w2 in np.arange(0.0, 1.0 - w1 + step / 2, step):
        w3 = max(0.0, 1.0 - w1 - w2)
        w = np.array([w1, w2, w3])
        blended = oof_matrix @ w
        th, acc = find_best_threshold(y, blended)
        if acc > best_acc:
            best_acc = acc
            best_weights = w
            best_threshold = th


# Nelder-Mead optimization refinement
def objective(params):
    w = np.maximum(params[:3], 0.0)
    if np.sum(w) == 0:
        return 0.0
    w = w / np.sum(w)
    th = params[3]
    blended = oof_matrix @ w
    acc = accuracy_score(y, (blended >= th).astype(int))
    return -acc


init_params = np.array([best_weights[0], best_weights[1], best_weights[2], best_threshold])
opt_res = minimize(
    objective,
    init_params,
    method="Nelder-Mead",
    options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-5},
)

if -opt_res.fun > best_acc:
    opt_w = np.maximum(opt_res.x[:3], 0.0)
    opt_w = opt_w / np.sum(opt_w)
    best_weights = opt_w
    best_threshold = float(opt_res.x[3])
    best_acc = -opt_res.fun

print(f"Optimal Ensemble Weights (LGB, XGB, CAT): {best_weights.round(4)}")
print(f"Optimal Decision Threshold: {best_threshold:.4f}")

final_oof_probs = oof_matrix @ best_weights
final_oof_preds = (final_oof_probs >= best_threshold).astype(int)
final_score = compute_competition_metric(y, final_oof_preds)

final_test_probs = test_matrix @ best_weights
final_test_preds = (final_test_probs >= best_threshold).astype(int)

# Map predictions back to original target domain representations
test_income_labels = [
    pos_label if pred == 1 else neg_label for pred in final_test_preds
]

submission = pd.DataFrame(
    {"id": np.arange(len(test_df), dtype=int), "income": test_income_labels}
)

submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
